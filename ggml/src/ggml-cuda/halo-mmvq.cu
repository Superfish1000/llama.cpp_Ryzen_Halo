// gfx1151 (RDNA3.5) custom q4_K decode matvec — "halo" path.
// Consumes raw f32 activations (quantizes to q8 in LDS: the separate
// quantize_q8_1 kernel is never launched for these ops), streams weights with
// per-thread row walks + 2-stage prefetch, dots via v_dot4 (sudot4).
// Measured on Strix Halo: 207.6 GB/s cold vs incumbent's ~183 average.
// Hooked from ggml_cuda_mul_mat_vec_q for: Q4_K, ids path (MUL_MAT_ID),
// ncols_dst==1, k in {768, 2048}; optional gate fusion (SWIGLU/GEGLU/MUL).
#include "common.cuh"
#include "unary.cuh"

#define HALO_WG 128

static __device__ __forceinline__ void halo_scale_min(int j, const uint8_t * s, float & sc, float & mn) {
    if (j < 4) { sc = (float)(s[j] & 63);        mn = (float)(s[j+4] & 63); }
    else       { sc = (float)((s[j+4] & 0xF) | ((s[j-4] >> 6) << 4));
                 mn = (float)((s[j+4] >>  4) | ((s[j]   >> 6) << 4)); }
}

static __device__ __forceinline__ int halo_dot4(uint32_t w_nib, int y4, int acc) {
#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA3_5) || defined(RDNA4))
    return __builtin_amdgcn_sudot4(false, (int) w_nib, true, y4, acc, false);
#else
    const int8_t * yb = (const int8_t *) &y4;
    acc += (int)( w_nib        & 0xFF) * yb[0]
         + (int)((w_nib >>  8) & 0xFF) * yb[1]
         + (int)((w_nib >> 16) & 0xFF) * yb[2]
         + (int)((w_nib >> 24) & 0xFF) * yb[3];
    return acc;
#endif
}

// One thread owns one (expert-slot, row); walks NSB superblocks of up (+gate).
template <int NSB, bool HAS_GATE>
static __global__ void halo_mmv_q4k(
        const void * __restrict__ vx, const void * __restrict__ vgate,
        const float * __restrict__ y, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const int nrows, const int nch_dst,
        const int64_t stride_row_blk, const int64_t stride_ch_blk,
        const int64_t stride_ch_dst, const int glu_op,
        const int nch_y, const int64_t stride_ch_y) {
    __shared__ int8_t syq[6144];           // up to nch_y*NSB*256 (max 8*3*256)
    __shared__ float  syd[192];
    __shared__ float  ysm[192];
    const int nsub = nch_y*NSB*8;
    for (int s = threadIdx.x; s < nsub; s += HALO_WG) {
        const int cy = s / (NSB*8);
        const int ls = s % (NSB*8);
        const float * yc = y + (int64_t) cy*stride_ch_y;
        float amax = 0.f, sum = 0.f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const float v = yc[ls*32 + l];
            amax = fmaxf(amax, fabsf(v)); sum += v;
        }
        const float dq = amax / 127.f;
        syd[s] = dq; ysm[s] = sum;
        const float inv = dq > 0.f ? 1.f/dq : 0.f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            syq[s*32 + l] = (int8_t) lrintf(yc[ls*32 + l] * inv);
        }
    }
    __syncthreads();

    const int64_t gr = (int64_t) blockIdx.x*HALO_WG + threadIdx.x;
    if (gr >= (int64_t) nch_dst * nrows) {
        return;
    }
    const int c   = (int)(gr / nrows);
    const int row = (int)(gr % nrows);
    const int expert = ids ? ids[c] : c;
    const int cy = c % nch_y;                       // y channel for this slot
    const int ybase = cy*NSB*8;                     // sub-block offset in LDS

    const uint4 * bx = (const uint4 *)((const block_q4_K *) vx + (int64_t) expert*stride_ch_blk + (int64_t) row*stride_row_blk);
    const uint4 * bg = nullptr;
    if (HAS_GATE) {
        bg = (const uint4 *)((const block_q4_K *) vgate + (int64_t) expert*stride_ch_blk + (int64_t) row*stride_row_blk);
    }

    uint4 wA[9], wB[9], gA[9], gB[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { wA[i] = bx[i]; }
    if (HAS_GATE) {
#pragma unroll
        for (int i = 0; i < 9; ++i) { gA[i] = bg[i]; }
    }

    float acc_x = 0.f, acc_g = 0.f;
    for (int sb = 0; sb < NSB; ++sb) {
        if (sb + 1 < NSB) {
#pragma unroll
            for (int i = 0; i < 9; ++i) { wB[i] = bx[(sb+1)*9 + i]; }
            if (HAS_GATE) {
#pragma unroll
                for (int i = 0; i < 9; ++i) { gB[i] = bg[(sb+1)*9 + i]; }
            }
        }
        const uint32_t * yq = (const uint32_t *) &syq[(ybase/8 + sb)*256];
        {
            const uint8_t  * raw    = (const uint8_t  *) wA;
            const float      d      = __half2float(((const __half *) raw)[0]);
            const float      dmin   = __half2float(((const __half *) raw)[1]);
            const uint8_t  * scales = raw + 4;
            const uint32_t * qs32   = (const uint32_t *)(raw + 16);
            int isum[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { isum[j] = 0; }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const uint32_t q = qs32[g*8 + t];
                    isum[2*g]   = halo_dot4( q       & 0x0F0F0F0Fu, (int) yq[(2*g)*8   + t], isum[2*g]);
                    isum[2*g+1] = halo_dot4((q >> 4) & 0x0F0F0F0Fu, (int) yq[(2*g+1)*8 + t], isum[2*g+1]);
                }
            }
            float sMul = 0.f, sMin = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sc, mn; halo_scale_min(j, scales, sc, mn);
                sMul = fmaf(sc * syd[ybase + sb*8 + j], (float) isum[j], sMul);
                sMin = fmaf(mn, ysm[ybase + sb*8 + j], sMin);
            }
            acc_x += d*sMul - dmin*sMin;
        }
        if (HAS_GATE) {
            const uint8_t  * raw    = (const uint8_t  *) gA;
            const float      d      = __half2float(((const __half *) raw)[0]);
            const float      dmin   = __half2float(((const __half *) raw)[1]);
            const uint8_t  * scales = raw + 4;
            const uint32_t * qs32   = (const uint32_t *)(raw + 16);
            int isum[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { isum[j] = 0; }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const uint32_t q = qs32[g*8 + t];
                    isum[2*g]   = halo_dot4( q       & 0x0F0F0F0Fu, (int) yq[(2*g)*8   + t], isum[2*g]);
                    isum[2*g+1] = halo_dot4((q >> 4) & 0x0F0F0F0Fu, (int) yq[(2*g+1)*8 + t], isum[2*g+1]);
                }
            }
            float sMul = 0.f, sMin = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sc, mn; halo_scale_min(j, scales, sc, mn);
                sMul = fmaf(sc * syd[ybase + sb*8 + j], (float) isum[j], sMul);
                sMin = fmaf(mn, ysm[ybase + sb*8 + j], sMin);
            }
            acc_g += d*sMul - dmin*sMin;
        }
#pragma unroll
        for (int i = 0; i < 9; ++i) { wA[i] = wB[i]; }
        if (HAS_GATE) {
#pragma unroll
            for (int i = 0; i < 9; ++i) { gA[i] = gB[i]; }
        }
    }

    float result = acc_x;
    if (HAS_GATE) {
        switch ((ggml_glu_op) glu_op) {
            case GGML_GLU_OP_SWIGLU:
                result *= ggml_cuda_op_silu_single(acc_g);
                break;
            case GGML_GLU_OP_GEGLU:
                result *= ggml_cuda_op_gelu_single(acc_g);
                break;
            default:
                result *= acc_g;
                break;
        }
    }
    dst[(int64_t) c*stride_ch_dst + row] = result;
}

// ---- host side ----

bool ggml_cuda_halo_mmvq_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (getenv("HALO_MMVQ_DISABLE") != nullptr) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = src0->ne[0];
    if (k != 2048 && k != 768 && k != 4096) {
        return false;
    }
    if (ids) {
        if (dst->ne[2] != 1) {              // single token (ncols_dst == 1)
            return false;
        }
        if (ids->type != GGML_TYPE_I32) {
            return false;
        }
        const int64_t nch_y = src1->ne[1];
        if (nch_y != 1 && nch_y != dst->ne[1]) {
            return false;
        }
        if (nch_y * (k/256) * 256 > 6144) { // LDS budget for staged y
            return false;
        }
    } else {
        if (getenv("HALO_DENSE") == nullptr) {
            return false;                   // dense path opt-in until proven
        }
        if (dst->ne[1] != 1 || dst->ne[2] != 1 || src0->ne[2] != 1) {
            return false;
        }
        if (src0->ne[1] < 1024) {           // too few rows to fill the GPU
            return false;
        }
        if ((k/256) * 256 > 6144) {
            return false;
        }
    }
    if (src1->nb[0] != sizeof(float)) {
        return false;
    }
    if (fusion) {
        if (fusion->x_bias || fusion->gate_bias || fusion->x_scale || fusion->gate_scale) {
            return false;
        }
        if (fusion->gate) {
            if (fusion->gate->type != GGML_TYPE_Q4_K || !ggml_are_same_stride(fusion->gate, src0)) {
                return false;
            }
        }
    }
    return true;
}

void ggml_cuda_halo_mmvq(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    const int64_t k       = src0->ne[0];
    const int64_t nrows   = src0->ne[1];
    const int64_t nch_dst = ids ? dst->ne[1] : 1;
    const size_t  ts0     = ggml_type_size(src0->type);

    const int64_t stride_row_blk = src0->nb[1] / ts0;
    const int64_t stride_ch_blk  = src0->nb[2] / ts0;
    const int64_t stride_ch_dst  = dst->nb[1] / sizeof(float);
    const int     nch_y          = ids ? (int) src1->ne[1] : 1;
    const int64_t stride_ch_y    = src1->nb[1] / sizeof(float);

    const void * gate = (fusion && fusion->gate) ? fusion->gate->data : nullptr;
    const int glu_op = (fusion && fusion->gate) ? (int) fusion->glu_op : 0;

    const int64_t total  = nch_dst * nrows;
    const int     blocks = (int)((total + HALO_WG - 1) / HALO_WG);
    cudaStream_t  stream = ctx.stream();

    const int32_t * ids_d = ids ? (const int32_t *) ids->data : nullptr;
#define HALO_LAUNCH(NSBV, GATEV, GATEPTR, GLUV)     halo_mmv_q4k<NSBV, GATEV><<<blocks, HALO_WG, 0, stream>>>(         src0->data, GATEPTR, (const float *) src1->data, ids_d,         (float *) dst->data, (int) nrows, (int) nch_dst, stride_row_blk, stride_ch_blk,         stride_ch_dst, GLUV, nch_y, stride_ch_y)
    switch (k) {
        case 2048: if (gate) { HALO_LAUNCH(8,  true, gate, glu_op); } else { HALO_LAUNCH(8,  false, nullptr, 0); } break;
        case  768: if (gate) { HALO_LAUNCH(3,  true, gate, glu_op); } else { HALO_LAUNCH(3,  false, nullptr, 0); } break;
        case 4096: if (gate) { HALO_LAUNCH(16, true, gate, glu_op); } else { HALO_LAUNCH(16, false, nullptr, 0); } break;
        default: GGML_ABORT("halo: unsupported k");
    }
#undef HALO_LAUNCH
}
