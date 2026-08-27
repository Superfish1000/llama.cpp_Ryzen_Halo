// Persistent block engine: model-agnostic fused region kernels for decode.
// First block: moe_block - replaces the generic MoE region emitted by
// llama.cpp's build_moe_ffn (softmax gating + norm_topk_prob variant):
//   RMS_NORM -> MUL -> MUL_MAT(router) -> SOFT_MAX -> RESHAPE -> ARGSORT ->
//   VIEW(topk) -> GET_ROWS -> RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
//   -> RESHAPE -> MUL_MAT_ID(gate) -> MUL_MAT_ID(up) -> GLU(swiglu) ->
//   MUL_MAT_ID(down) -> MUL -> VIEW*topk -> ADD*(topk-1) -> ADD(residual)
// matched by op/shape/type only - no model names. Shapes are runtime args;
// quant block-counts are a small template set. Opt-in: HALO_BLOCK=1.
#include "common.cuh"
#include <vector>
#include <unordered_map>

#define HB_NTH 256

struct hb_moe_params {
    const float * x;            // residual input (rms_norm src)
    const float * nw;           // ffn norm weight
    float         eps;
    const float * wr;           // router weight [nexp, ne] f32
    const void  * wg;           // gate experts q4_K
    const void  * wu;           // up experts
    const void  * wd;           // down experts
    int64_t sg_row, sg_exp;     // gate/up strides in blocks (row, expert)
    int64_t sd_row, sd_exp;     // down strides in blocks
    float * out;                // final residual output buffer
    int ne, nf, nexp, topk;
    float clamp_min, clamp_max;
};

struct hb_scratch {
    int   bar_count, bar_sense;
    // dynamic region follows: yq[ne] i8; yd/ys[ne/32]; logits[nexp];
    // ids[topk] i32; pw[topk]; h[topk*nf]; hq[topk*nf] i8; hd/hs[topk*nf/32];
    // part[topk*ne]
};

static __device__ void hb_bar(int * count, volatile int * sense, int * lsense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int my = 1 - *lsense;
        *lsense = my;
        if (atomicAdd(count, 1) == (int) gridDim.x - 1) {
            *count = 0;
            __threadfence();
            *sense = my;
        } else {
            while (__hip_atomic_load((const int *) sense, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != my) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
}

static __device__ __forceinline__ void hb_scale_min(int j, const uint8_t * s, float & sc, float & mn) {
    if (j < 4) { sc = (float)(s[j] & 63);        mn = (float)(s[j+4] & 63); }
    else       { sc = (float)((s[j+4] & 0xF) | ((s[j-4] >> 6) << 4));
                 mn = (float)((s[j+4] >>  4) | ((s[j]   >> 6) << 4)); }
}

// one q4_K row dot against LDS-staged q8 activations (NSB blocks)
template <int NSB>
static __device__ __noinline__ float hb_q4row(const block_q4_K * B, const int8_t * syq,
                                 const float * syd, const float * sys) {
    float a = 0.f;
    for (int sb = 0; sb < NSB; ++sb) {
        uint4 wv[9];
        const uint4 * bx = (const uint4 *)(B + sb);
#pragma unroll
        for (int i = 0; i < 9; ++i) { wv[i] = bx[i]; }
        const uint8_t * raw = (const uint8_t *) wv;
        const float d = __half2float(((const __half *) raw)[0]);
        const float dmin = __half2float(((const __half *) raw)[1]);
        const uint8_t * sc8 = raw + 4;
        const uint32_t * qs32 = (const uint32_t *)(raw + 16);
        const uint32_t * yq32 = (const uint32_t *) &syq[sb*256];
        int is[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { is[j] = 0; }
#pragma unroll
        for (int g = 0; g < 4; ++g) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint32_t q = qs32[g*8 + t];
                is[2*g]   = __builtin_amdgcn_sudot4(false, (int)( q       & 0x0F0F0F0Fu), true, (int) yq32[(2*g)*8 + t], is[2*g], false);
                is[2*g+1] = __builtin_amdgcn_sudot4(false, (int)((q >> 4) & 0x0F0F0F0Fu), true, (int) yq32[(2*g+1)*8 + t], is[2*g+1], false);
            }
        }
        float sMul = 0.f, sMin = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float sc, mn;
            hb_scale_min(j, sc8, sc, mn);
            sMul = fmaf(sc*syd[sb*8 + j], (float) is[j], sMul);
            sMin = fmaf(mn, sys[sb*8 + j], sMin);
        }
        a += d*sMul - dmin*sMin;
    }
    return a;
}

template <int NSB_GU, int NSB_DN, int NE_, int NF_, int NEXP_, int TOPK_>
static __global__ void __launch_bounds__(HB_NTH) hb_moe_block(hb_moe_params P, void * scr) {
    constexpr int ne = NE_, nf = NF_, nexp = NEXP_, topk = TOPK_;
    int * bar = (int *) scr;
    int8_t * yq = (int8_t *)((char *) scr + 16);
    float * yd = (float *)(yq + ne);
    float * ys = yd + ne/32;
    float * logits = ys + ne/32;
    int   * ids = (int *)(logits + nexp);
    float * pw  = (float *)(ids + topk);
    float * h   = pw + topk;
    int8_t * hq = (int8_t *)(h + (size_t) topk*nf);
    float * hd  = (float *)(hq + (size_t) topk*nf);
    float * hs  = hd + (size_t) topk*nf/32;
    float * part = hs + (size_t) topk*nf/32;

    __shared__ int lsense;
    __shared__ int8_t syq[NSB_GU*256];
    __shared__ float  syd[NSB_GU*8], sys[NSB_GU*8];
    __shared__ float s_red[HB_NTH/32];
    if (threadIdx.x == 0) { lsense = ((volatile int *) bar)[1]; }
    __syncthreads();

    // ---- front: redundant per-WG sumsq; warp roles: router rows | y-quant ----
    {
        float ss = 0.f;
        for (int i = threadIdx.x; i < ne; i += blockDim.x) {
            ss = fmaf(P.x[i], P.x[i], ss);
        }
        ss += __shfl_down(ss, 16, 32); ss += __shfl_down(ss, 8, 32);
        ss += __shfl_down(ss, 4, 32);  ss += __shfl_down(ss, 2, 32); ss += __shfl_down(ss, 1, 32);
        if ((threadIdx.x & 31) == 0) { s_red[threadIdx.x >> 5] = ss; }
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < HB_NTH/32; ++i) { tot += s_red[i]; }
        const float rrms = rsqrtf(tot/ne + P.eps);
        const int warp = threadIdx.x/32, lane = threadIdx.x & 31;
        const int RW = 3;
        if (warp < RW) {
            for (int r = warp*gridDim.x + blockIdx.x; r < nexp; r += RW*gridDim.x) {
                const float4 * a4 = (const float4 *)(P.wr + (size_t) r*ne);
                const float4 * x4 = (const float4 *) P.x;
                const float4 * w4 = (const float4 *) P.nw;
                float acc = 0.f;
                for (int i = lane; i < ne/4; i += 32) {
                    const float4 a = a4[i], xb = x4[i], wb = w4[i];
                    acc = fmaf(a.x, xb.x*rrms*wb.x, acc); acc = fmaf(a.y, xb.y*rrms*wb.y, acc);
                    acc = fmaf(a.z, xb.z*rrms*wb.z, acc); acc = fmaf(a.w, xb.w*rrms*wb.w, acc);
                }
                acc += __shfl_down(acc, 16, 32); acc += __shfl_down(acc, 8, 32);
                acc += __shfl_down(acc, 4, 32);  acc += __shfl_down(acc, 2, 32); acc += __shfl_down(acc, 1, 32);
                if (lane == 0) { logits[r] = acc; }
            }
        } else {
            const int qwarps = (blockDim.x/32 - RW)*gridDim.x;
            const int qid = (warp - RW)*gridDim.x + blockIdx.x;
            for (int s = qid; s < ne/32; s += qwarps) {
                const float v = P.x[s*32 + lane]*rrms*P.nw[s*32 + lane];
                float amax = fabsf(v), sum = v;
                amax = fmaxf(amax, __shfl_xor(amax, 16, 32)); sum += __shfl_xor(sum, 16, 32);
                amax = fmaxf(amax, __shfl_xor(amax, 8, 32));  sum += __shfl_xor(sum, 8, 32);
                amax = fmaxf(amax, __shfl_xor(amax, 4, 32));  sum += __shfl_xor(sum, 4, 32);
                amax = fmaxf(amax, __shfl_xor(amax, 2, 32));  sum += __shfl_xor(sum, 2, 32);
                amax = fmaxf(amax, __shfl_xor(amax, 1, 32));  sum += __shfl_xor(sum, 1, 32);
                const float dq = amax/127.f;
                if (lane == 0) { yd[s] = dq; ys[s] = sum; }
                const float inv = dq > 0.f ? 1.f/dq : 0.f;
                yq[s*32 + lane] = (int8_t) lrintf(v*inv);
            }
        }
    }
    hb_bar(bar, bar + 1, &lsense);

    // ---- topk on WG0 (softmax + argmax rounds + clamp-normalized weights);
    //      all WGs stage y into LDS meanwhile ----
    if (blockIdx.x == 0) {
        __shared__ float p[NEXP_];
        __shared__ float red2[2];
        __shared__ int   s_bi[HB_NTH/32];
        __shared__ float s_bv[HB_NTH/32];
        const int tid = threadIdx.x;
        if (tid == 0) {
            float m2 = -1e30f;
            for (int i = 0; i < nexp; ++i) { m2 = fmaxf(m2, logits[i]); }
            red2[0] = m2;
        }
        __syncthreads();
        const float gmax = red2[0];
        float den = 0.f;
        for (int i = tid; i < nexp; i += blockDim.x) {
            const float e = __expf(logits[i] - gmax);
            p[i] = e;
            den += e;
        }
        den += __shfl_xor(den, 16, 32); den += __shfl_xor(den, 8, 32);
        den += __shfl_xor(den, 4, 32);  den += __shfl_xor(den, 2, 32); den += __shfl_xor(den, 1, 32);
        if (tid == 0) { red2[1] = 0.f; }
        __syncthreads();
        if ((tid & 31) == 0) { atomicAdd(&red2[1], den); }
        __syncthreads();
        for (int t = 0; t < topk; ++t) {
            float bv = -1.f; int bi = -1;
            if (tid < nexp) { bv = p[tid]; bi = tid; }
            for (int i = tid + blockDim.x; i < nexp; i += blockDim.x) {
                if (p[i] > bv) { bv = p[i]; bi = i; }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor(bv, off, 32);
                const int   oi = __shfl_xor(bi, off, 32);
                if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
            }
            if ((tid & 31) == 0) { s_bv[tid >> 5] = bv; s_bi[tid >> 5] = bi; }
            __syncthreads();
            if (tid == 0) {
                float fv = s_bv[0]; int fi = s_bi[0];
                for (int i = 1; i < HB_NTH/32; ++i) {
                    if (s_bv[i] > fv || (s_bv[i] == fv && s_bi[i] >= 0 && (fi < 0 || s_bi[i] < fi))) { fv = s_bv[i]; fi = s_bi[i]; }
                }
                ids[t] = fi;
                pw[t] = fv;
                p[fi] = -2.f;
            }
            __syncthreads();
        }
        if (tid == 0) {
            const float dsum = red2[1];
            float wsum = 0.f;
            for (int t = 0; t < topk; ++t) { pw[t] /= dsum; wsum += pw[t]; }
            wsum = fminf(fmaxf(wsum, P.clamp_min), P.clamp_max);
            for (int t = 0; t < topk; ++t) { pw[t] /= wsum; }
        }
    }
    for (int i = threadIdx.x; i < ne; i += blockDim.x) { syq[i] = yq[i]; }
    for (int i = threadIdx.x; i < ne/32; i += blockDim.x) { syd[i] = yd[i]; sys[i] = ys[i]; }
    hb_bar(bar, bar + 1, &lsense);

    // ---- gate+up + GLU (swiglu) ----
    {
        const int nth = gridDim.x*blockDim.x;
        for (int u = blockIdx.x*blockDim.x + threadIdx.x; u < topk*nf; u += nth) {
            const int slot = u/nf, row = u%nf;
            const int e = ids[slot];
            float acc[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const block_q4_K * Wm = m == 0
                    ? (const block_q4_K *) P.wg + (size_t) e*P.sg_exp + (size_t) row*P.sg_row
                    : (const block_q4_K *) P.wu + (size_t) e*P.sg_exp + (size_t) row*P.sg_row;
                acc[m] = hb_q4row<NSB_GU>(Wm, syq, syd, sys);
            }
            const float g = acc[0];
            h[(size_t) slot*nf + row] = (g/(1.f + __expf(-g)))*acc[1];
        }
    }
    hb_bar(bar, bar + 1, &lsense);

    // ---- h quantize (per 32-block) ----
    {
        const int nth = gridDim.x*blockDim.x;
        const int nsub = topk*(nf/32);
        for (int u = blockIdx.x*blockDim.x + threadIdx.x; u < nsub; u += nth) {
            const int slot = u/(nf/32), s = u%(nf/32);
            float amax = 0.f, sum = 0.f;
            for (int l = 0; l < 32; ++l) {
                const float v = h[(size_t) slot*nf + s*32 + l];
                amax = fmaxf(amax, fabsf(v)); sum += v;
            }
            const float dq = amax/127.f;
            hd[(size_t) slot*(nf/32) + s] = dq;
            hs[(size_t) slot*(nf/32) + s] = sum;
            const float inv = dq > 0.f ? 1.f/dq : 0.f;
            for (int l = 0; l < 32; ++l) {
                hq[(size_t) slot*nf + s*32 + l] = (int8_t) lrintf(h[(size_t) slot*nf + s*32 + l]*inv);
            }
        }
    }
    hb_bar(bar, bar + 1, &lsense);

    // ---- down (per-slot partials) ----
    {
        const int nth = gridDim.x*blockDim.x;
        for (int u = blockIdx.x*blockDim.x + threadIdx.x; u < topk*ne; u += nth) {
            const int slot = u/ne, row = u%ne;
            const int e = ids[slot];
            const block_q4_K * B = (const block_q4_K *) P.wd + (size_t) e*P.sd_exp + (size_t) row*P.sd_row;
            part[(size_t) slot*ne + row] = hb_q4row<NSB_DN>(B,
                hq + (size_t) slot*nf, hd + (size_t) slot*(nf/32), hs + (size_t) slot*(nf/32));
        }
    }
    hb_bar(bar, bar + 1, &lsense);

    // ---- weighted sum + residual ----
    {
        const int nth = gridDim.x*blockDim.x;
        for (int r = blockIdx.x*blockDim.x + threadIdx.x; r < ne; r += nth) {
            float a = P.x[r];
            for (int s = 0; s < topk; ++s) { a = fmaf(part[(size_t) s*ne + r], pw[s], a); }
            P.out[r] = a;
        }
    }
}

// ---- host: scanner + launcher ----

struct hb_match {
    int start;                 // RMS_NORM node index
    hb_moe_params P;           // pointers refreshed each eval
};

static bool hb_validate_moe(const ggml_cgraph * g, int i, hb_moe_params & P) {
    // expected op sequence from build_moe_ffn (softmax + norm_topk variant)
    const ggml_tensor * rn = g->nodes[i];
    if (rn->op != GGML_OP_RMS_NORM || rn->ne[1] != 1 || rn->ne[2] != 1) return false;
    const int topk_guess = 8;   // refined below from the VIEW node
    const int need = 13 + 2 + 4 + 1 + topk_guess + (topk_guess - 1) + 1;
    if (i + need >= g->n_nodes) return false;
    const ggml_tensor * ml   = g->nodes[i+1];
    const ggml_tensor * rout = g->nodes[i+2];
    const ggml_tensor * smax = g->nodes[i+3];
    const ggml_tensor * rsh1 = g->nodes[i+4];
    const ggml_tensor * asrt = g->nodes[i+5];
    const ggml_tensor * topv = g->nodes[i+6];
    const ggml_tensor * grow = g->nodes[i+7];
    const ggml_tensor * rsh2 = g->nodes[i+8];
    const ggml_tensor * srow = g->nodes[i+9];
    const ggml_tensor * clmp = g->nodes[i+10];
    const ggml_tensor * divi = g->nodes[i+11];
    const ggml_tensor * rsh3 = g->nodes[i+12];
    const ggml_tensor * rsh4 = g->nodes[i+13];
    const ggml_tensor * gate = g->nodes[i+14];
    const ggml_tensor * up   = g->nodes[i+15];
    const ggml_tensor * glu  = g->nodes[i+16];
    const ggml_tensor * down = g->nodes[i+17];
    const ggml_tensor * wmul = g->nodes[i+18];
    if (ml->op != GGML_OP_MUL || ml->src[0] != rn) return false;
    if (rout->op != GGML_OP_MUL_MAT || rout->src[1] != ml || rout->src[0]->type != GGML_TYPE_F32) return false;
    if (smax->op != GGML_OP_SOFT_MAX || smax->src[0] != rout) return false;
    {   // require plain softmax (scale 1, no bias, no mask)
        float sp[2];
        memcpy(sp, smax->op_params, sizeof(sp));
        if (sp[0] != 1.0f || sp[1] != 0.0f || smax->src[1] != nullptr) return false;
    }
    if (rsh1->op != GGML_OP_RESHAPE || rsh1->src[0] != smax) return false;
    if (asrt->op != GGML_OP_ARGSORT || asrt->src[0] != smax) return false;
    if (topv->op != GGML_OP_VIEW || topv->src[0] != asrt) return false;
    const int topk = (int) topv->ne[0];
    if (topk < 1 || topk > 16) return false;
    if (grow->op != GGML_OP_GET_ROWS || grow->src[0] != rsh1 || grow->src[1] != topv) return false;
    if (rsh2->op != GGML_OP_RESHAPE || rsh2->src[0] != grow) return false;
    if (srow->op != GGML_OP_SUM_ROWS || srow->src[0] != rsh2) return false;
    if (clmp->op != GGML_OP_CLAMP || clmp->src[0] != srow) return false;
    if (divi->op != GGML_OP_DIV || divi->src[0] != rsh2 || divi->src[1] != clmp) return false;
    if (rsh3->op != GGML_OP_RESHAPE || rsh3->src[0] != divi) return false;
    if (rsh4->op != GGML_OP_RESHAPE || rsh4->src[0] != ml) return false;
    if (gate->op != GGML_OP_MUL_MAT_ID || gate->src[1] != rsh4 || gate->src[2] != topv) return false;
    if (up->op   != GGML_OP_MUL_MAT_ID || up->src[1]   != rsh4 || up->src[2]   != topv) return false;
    if (glu->op != GGML_OP_GLU || glu->src[0] != gate || glu->src[1] != up) return false;
    if (ggml_get_glu_op((ggml_tensor *) glu) != GGML_GLU_OP_SWIGLU) return false;
    if (down->op != GGML_OP_MUL_MAT_ID || down->src[1] != glu || down->src[2] != topv) return false;
    if (wmul->op != GGML_OP_MUL || wmul->src[0] != down || wmul->src[1] != rsh3) return false;
    // tail: topk VIEWs of wmul, then topk-1 ADDs, then residual ADD
    int j = i + 19;
    for (int v = 0; v < topk; ++v, ++j) {
        if (g->nodes[j]->op != GGML_OP_VIEW || g->nodes[j]->src[0] != wmul) return false;
    }
    const ggml_tensor * acc = g->nodes[j - topk];   // first view
    for (int a = 0; a < topk - 1; ++a, ++j) {
        const ggml_tensor * ad = g->nodes[j];
        if (ad->op != GGML_OP_ADD) return false;
        if (!((ad->src[0] == acc && ad->src[1]->op == GGML_OP_VIEW && ad->src[1]->src[0] == wmul) ||
              (ad->src[1] == acc && ad->src[0]->op == GGML_OP_VIEW && ad->src[0]->src[0] == wmul))) return false;
        acc = ad;
    }
    const ggml_tensor * res = g->nodes[j];
    if (res->op != GGML_OP_ADD) return false;
    const ggml_tensor * x = rn->src[0];
    if (!((res->src[0] == acc && res->src[1] == x) || (res->src[1] == acc && res->src[0] == x))) return false;
    // shape/type gates
    const ggml_tensor * wg = gate->src[0], * wu = up->src[0], * wd = down->src[0], * wnorm = ml->src[1];
    if (wg->type != GGML_TYPE_Q4_K || wu->type != GGML_TYPE_Q4_K || wd->type != GGML_TYPE_Q4_K) return false;
    const int64_t ne = rn->ne[0], nf = gate->ne[0], nexp = rout->ne[0];
    if (ne % 256 || nf % 256) return false;
    if (!(ne/256 == 8 && nf/256 == 3)) return false;   // instantiated set; grows with the phase library
    if (nexp > 512) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(wnorm) || ggml_nelements(wnorm) != ne) return false;
    if ((wg->nb[1] % sizeof(block_q4_K)) || (wd->nb[1] % sizeof(block_q4_K))) return false;

    P.eps = 0.f;
    memcpy(&P.eps, rn->op_params, sizeof(float));
    float cl[2];
    memcpy(cl, clmp->op_params, sizeof(cl));
    P.clamp_min = cl[0]; P.clamp_max = cl[1];
    P.ne = (int) ne; P.nf = (int) nf; P.nexp = (int) nexp; P.topk = topk;
    P.x = (const float *) x->data;
    P.nw = (const float *) wnorm->data;
    P.wr = (const float *) rout->src[0]->data;
    P.wg = wg->data; P.wu = wu->data; P.wd = wd->data;
    P.sg_row = wg->nb[1]/sizeof(block_q4_K);
    P.sg_exp = wg->nb[2]/sizeof(block_q4_K);
    P.sd_row = wd->nb[1]/sizeof(block_q4_K);
    P.sd_exp = wd->nb[2]/sizeof(block_q4_K);
    P.out = (float *) res->data;
    return true;
}

static int hb_region_len(int topk) { return 19 + topk + (topk - 1) + 1; }

bool ggml_cuda_halo_try_moeblock(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
                                 std::vector<const ggml_tensor *> & skip_list) {
    static const bool en = getenv("HALO_BLOCK") != nullptr;
    if (!en) {
        return false;
    }
    // topology cache: validation is O(region), decisions cached per (uid, i)
    static std::unordered_map<long long, std::vector<int>> uid_hits;   // uid -> matched starts
    static long long cur_uid = -1;
    static std::vector<int> * hits = nullptr;
    if ((long long) cgraph->uid != cur_uid || cgraph->uid == 0) {
        if (cgraph->uid == 0) { uid_hits.clear(); }
        cur_uid = (long long) cgraph->uid;
        auto & v = uid_hits[cur_uid];
        if (v.empty()) {
            hb_moe_params tmp;
            for (int k = 0; k < cgraph->n_nodes; ++k) {
                if (cgraph->nodes[k]->op == GGML_OP_RMS_NORM && hb_validate_moe(cgraph, k, tmp)) {
                    v.push_back(k);
                }
            }
            if (v.empty()) v.push_back(-1);   // sentinel: scanned, none found
        }
        hits = &v;
    }
    if (hits == nullptr || hits->empty() || (*hits)[0] == -1) {
        return false;
    }
    bool ismatch = false;
    for (int s : *hits) { if (s == i) { ismatch = true; break; } }
    if (!ismatch) {
        return false;
    }
    hb_moe_params P;
    if (!hb_validate_moe(cgraph, i, P)) {
        return false;
    }
    // persistent barrier + scratch
    static void * scr = nullptr;
    static size_t scr_sz = 0;
    const size_t need = 16 + P.ne + 2*(P.ne/32)*4 + P.nexp*4 + P.topk*8
                      + (size_t) P.topk*P.nf*4 + (size_t) P.topk*P.nf
                      + 2*((size_t) P.topk*P.nf/32)*4 + (size_t) P.topk*P.ne*4 + 256;
    if (scr_sz < need) {
        if (scr) { (void) cudaFree(scr); }
        CUDA_CHECK(cudaMalloc(&scr, need));
        CUDA_CHECK(cudaMemset(scr, 0, need));
        scr_sz = need;
    }
    static int nwg = 0;
    if (nwg == 0) {
        nwg = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        if (nwg <= 40) nwg *= 2;   // RDNA reports WGPs; barrier geometry wants CUs
        if (nwg > 80) nwg = 80;
    }
    if (P.ne == 2048 && P.nf == 768 && P.nexp == 128 && P.topk == 8) {
        hb_moe_block<8, 3, 2048, 768, 128, 8><<<nwg, HB_NTH, 0, ctx.stream()>>>(P, scr);
    } else {
        return false;   // shape not in the instantiation set; stock path handles it
    }
    if (getenv("HALO_DEBUG") != nullptr) {
        static int nreg = 0;
        if (nreg++ < 3) {
            fprintf(stderr, "halo-moeblock: i=%d ne=%d nf=%d nexp=%d topk=%d nwg=%d\n", i, P.ne, P.nf, P.nexp, P.topk, nwg);
        }
    }
    const int len = hb_region_len(P.topk);
    for (int k = i + 1; k < i + len && k < cgraph->n_nodes; ++k) {
        skip_list.push_back(cgraph->nodes[k]);
    }
    return true;
}
