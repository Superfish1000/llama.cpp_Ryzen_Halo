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
        const int nch_y, const int64_t stride_ch_y, const float * __restrict__ x_bias) {
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
    if (x_bias != nullptr) {
        result += x_bias[row];
    }
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


// ---------------- Q6_K variant (v3: Vulkan-style 16-lane cooperative) ----------------
// 16 lanes share each superblock via 2B typed loads (alignment-immune, coalesced),
// scales cached in LDS cooperatively, dp4a with ones-vector -32 correction.
// Lane map (from ggml-vulkan mul_mat_vec_q6_k.comp): itid=tid%16, v_im=itid/8,
// v_in=itid%8, l0=4*v_in; ql_off=64*v_im+l0, qh_off=32*v_im+l0, s_off=8*v_im+v_in/4.
template <int NSB>
static __global__ void halo_mmv_q6k(
        const void * __restrict__ vx, const float * __restrict__ y,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int nrows, const int nch_dst,
        const int64_t stride_row_bytes, const int64_t stride_ch_bytes,
        const int64_t stride_ch_dst,
        const int nch_y, const int64_t stride_ch_y, const float * __restrict__ x_bias) {
    __shared__ int8_t syq[6144];
    __shared__ float  syd[192];
    const int nsub32 = nch_y*NSB*8;
    for (int s = threadIdx.x; s < nsub32; s += HALO_WG) {
        const int cy = s / (NSB*8);
        const int ls = s % (NSB*8);
        const float * yc = y + (int64_t) cy*stride_ch_y;
        float amax = 0.f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            amax = fmaxf(amax, fabsf(yc[ls*32 + l]));
        }
        const float dq = amax / 127.f;
        syd[s] = dq;
        const float inv = dq > 0.f ? 1.f/dq : 0.f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            syq[s*32 + l] = (int8_t) lrintf(yc[ls*32 + l] * inv);
        }
    }
    __syncthreads();

    // one row per 16-lane group; HALO_WG/16 groups per workgroup
    const int tid  = threadIdx.x;
    const int itid = tid & 15;
    const int grp  = tid >> 4;
    const int64_t grow = (int64_t) blockIdx.x*(HALO_WG/16) + grp;
    if (grow >= (int64_t) nch_dst * nrows) {
        return;
    }
    const int c   = (int)(grow / nrows);
    const int row = (int)(grow % nrows);
    const int expert = ids ? ids[c] : c;
    const int cy = c % nch_y;
    const uint8_t * base = (const uint8_t *) vx + (int64_t) expert*stride_ch_bytes + (int64_t) row*stride_row_bytes;

    const int v_im = itid >> 3;          // 0/1: which 128-half
    const int v_in = itid & 7;           // 0..7
    const int l0   = 4*v_in;             // 0..28
    const int is   = v_in >> 2;          // 0/1
    const int ql_off = 64*v_im + l0;     // bytes into ql[128]
    const int qh_off = 32*v_im + l0;     // bytes into qh[64]
    const int s_off  = 8*v_im + is;      // scale index base
    const int yq_off = (32*v_im + v_in); // y word index base within sb (4-quant words)

    float acc = 0.f;
    for (int sb = 0; sb < NSB; ++sb) {
        const uint8_t  * blk  = base + (size_t) sb*210;
        const uint16_t * ql16 = (const uint16_t *)(blk);
        const uint16_t * qh16 = (const uint16_t *)(blk + 128);
        const int8_t   * scp  = (const int8_t   *)(blk + 192);
        const float      d    = __half2float(*(const __half *)(blk + 208));

        const uint32_t ql0  = (uint32_t) ql16[ql_off/2]      | ((uint32_t) ql16[ql_off/2 + 1]  << 16);
        const uint32_t ql32 = (uint32_t) ql16[ql_off/2 + 16] | ((uint32_t) ql16[ql_off/2 + 17] << 16);
        const uint32_t qh   = (uint32_t) qh16[qh_off/2]      | ((uint32_t) qh16[qh_off/2 + 1]  << 16);

        const uint32_t q0 = ( ql0        & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
        const uint32_t q1 = ( ql32       & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2);
        const uint32_t q2 = ((ql0  >> 4) & 0x0F0F0F0Fu) | ( qh & 0x30303030u);
        const uint32_t q3 = ((ql32 >> 4) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2);

        // scales: two per lane-region set, read directly (2B loads, cheap via L1)
        const float sc0 = (float) scp[s_off    ];
        const float sc1 = (float) scp[s_off + 2];
        const float sc2 = (float) scp[s_off + 4];
        const float sc3 = (float) scp[s_off + 6];

        const uint32_t * yq = (const uint32_t *) &syq[(cy*NSB + sb)*256];
        const int y0 = (int) yq[yq_off     ];
        const int y1 = (int) yq[yq_off +  8];
        const int y2 = (int) yq[yq_off + 16];
        const int y3 = (int) yq[yq_off + 24];

        const int iq0 = halo_dot4(q0, y0, 0), iy0 = halo_dot4(0x01010101u, y0, 0);
        const int iq1 = halo_dot4(q1, y1, 0), iy1 = halo_dot4(0x01010101u, y1, 0);
        const int iq2 = halo_dot4(q2, y2, 0), iy2 = halo_dot4(0x01010101u, y2, 0);
        const int iq3 = halo_dot4(q3, y3, 0), iy3 = halo_dot4(0x01010101u, y3, 0);

        const float * yd = &syd[(cy*NSB + sb)*8];
        const float dq0 = yd[4*v_im + 0];
        const float dq1 = yd[4*v_im + 1];
        const float dq2 = yd[4*v_im + 2];
        const float dq3 = yd[4*v_im + 3];

        acc = fmaf(d, sc0*dq0*(float)(iq0 - 32*iy0)
                    + sc1*dq1*(float)(iq1 - 32*iy1)
                    + sc2*dq2*(float)(iq2 - 32*iy2)
                    + sc3*dq3*(float)(iq3 - 32*iy3), acc);
    }
    // reduce 16 lanes
    acc += __shfl_down(acc, 8, 16);
    acc += __shfl_down(acc, 4, 16);
    acc += __shfl_down(acc, 2, 16);
    acc += __shfl_down(acc, 1, 16);
    if (itid == 0) {
        if (x_bias != nullptr) {
            acc += x_bias[row];
        }
        dst[(int64_t) c*stride_ch_dst + row] = acc;
    }
}

// ---------------- fused QKV kernel (q4,q4,q6 / all-q4) ----------------
// Three matvecs sharing one activation vector, one launch. Row ranges map to
// tensors: [0,r0) -> W0/dst0, [r0,r1) -> W1/dst1, [r1,r2) -> W2/dst2.
template <int NSB, bool V_IS_Q6>
static __global__ void halo_mmv_qkv(
        const void * __restrict__ w0, const void * __restrict__ w1, const void * __restrict__ w2,
        const float * __restrict__ y,
        float * __restrict__ d0, float * __restrict__ d1, float * __restrict__ d2,
        const int r0, const int r1, const int r2,
        const int64_t srb0, const int64_t srb1, const int64_t srb2) {
    __shared__ int8_t syq[NSB*256];
    __shared__ float  syd[NSB*8];
    __shared__ float  ysm[NSB*8];
    __shared__ float  ysm16[NSB*16];
    for (int s = threadIdx.x; s < NSB*8; s += HALO_WG) {
        float amax = 0.f, s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            const float v = y[s*32 + l];
            amax = fmaxf(amax, fabsf(v)); s0 += v;
        }
#pragma unroll
        for (int l = 16; l < 32; ++l) {
            const float v = y[s*32 + l];
            amax = fmaxf(amax, fabsf(v)); s1 += v;
        }
        const float dq = amax / 127.f;
        syd[s] = dq; ysm[s] = s0 + s1; ysm16[2*s] = s0; ysm16[2*s + 1] = s1;
        const float inv = dq > 0.f ? 1.f/dq : 0.f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            syq[s*32 + l] = (int8_t) lrintf(y[s*32 + l] * inv);
        }
    }
    __syncthreads();

    const int row = blockIdx.x*HALO_WG + threadIdx.x;
    if (row >= r2) {
        return;
    }
    const int t = row < r0 ? 0 : (row < r1 ? 1 : 2);
    const int lrow = t == 0 ? row : (t == 1 ? row - r0 : row - r1);
    const uint8_t * base = (const uint8_t *)(t == 0 ? w0 : (t == 1 ? w1 : w2))
                         + (int64_t) lrow * (t == 0 ? srb0 : (t == 1 ? srb1 : srb2));   // byte strides
    float * dst = t == 0 ? d0 : (t == 1 ? d1 : d2);

    float acc = 0.f;
    if (!V_IS_Q6 || t < 2) {
        const uint4 * bx = (const uint4 *) base;
        uint4 wA[9], wB[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) { wA[i] = bx[i]; }
        for (int sb = 0; sb < NSB; ++sb) {
            if (sb + 1 < NSB) {
#pragma unroll
                for (int i = 0; i < 9; ++i) { wB[i] = bx[(sb+1)*9 + i]; }
            }
            const uint8_t  * raw    = (const uint8_t  *) wA;
            const float      d      = __half2float(((const __half *) raw)[0]);
            const float      dmin   = __half2float(((const __half *) raw)[1]);
            const uint8_t  * scales = raw + 4;
            const uint32_t * qs32   = (const uint32_t *)(raw + 16);
            const uint32_t * yq     = (const uint32_t *) &syq[sb*256];
            int isum[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { isum[j] = 0; }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
#pragma unroll
                for (int tt = 0; tt < 8; ++tt) {
                    const uint32_t q = qs32[g*8 + tt];
                    isum[2*g]   = halo_dot4( q       & 0x0F0F0F0Fu, (int) yq[(2*g)*8   + tt], isum[2*g]);
                    isum[2*g+1] = halo_dot4((q >> 4) & 0x0F0F0F0Fu, (int) yq[(2*g+1)*8 + tt], isum[2*g+1]);
                }
            }
            float sMul = 0.f, sMin = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float sc, mn; halo_scale_min(j, scales, sc, mn);
                sMul = fmaf(sc * syd[sb*8 + j], (float) isum[j], sMul);
                sMin = fmaf(mn, ysm[sb*8 + j], sMin);
            }
            acc += d*sMul - dmin*sMin;
#pragma unroll
            for (int i = 0; i < 9; ++i) { wA[i] = wB[i]; }
        }
    } else {
        for (int sb = 0; sb < NSB; ++sb) {
            const int off = sb*210;
            const uint4 * rp4 = (const uint4 *) base;
            uint4 wbuf[14];
#pragma unroll
            for (int i = 0; i < 14; ++i) { wbuf[i] = rp4[(off >> 4) + i]; }
            uint32_t img[53];
            {
                const uint32_t * in = (const uint32_t *) wbuf;
                const int skip = (off & 15) >> 2;
                const int shb  = (off & 3) * 8;
                if (shb == 0) {
#pragma unroll
                    for (int i = 0; i < 53; ++i) { img[i] = in[i + skip]; }
                } else {
#pragma unroll
                    for (int i = 0; i < 53; ++i) { img[i] = (in[i + skip] >> shb) | (in[i + skip + 1] << (32 - shb)); }
                }
            }
            const uint32_t * ql32 = img;
            const uint32_t * qh32 = img + 32;
            const int8_t   * sc   = (const int8_t *)(img + 48);
            const float      d    = __half2float(*(const __half *)((const uint8_t *) img + 208));
            const uint32_t * yq = (const uint32_t *) &syq[sb*256];
            const float    * yd = &syd[sb*8];
            const float    * ys = &ysm16[sb*16];
            float bacc = 0.f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                int isum[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { isum[j] = 0; }
#pragma unroll
                for (int tt = 0; tt < 8; ++tt) {
                    const uint32_t ql0   = ql32[n*16 + tt];
                    const uint32_t ql32b = ql32[n*16 + 8 + tt];
                    const uint32_t qh0   = qh32[n*8 + tt];
                    const uint32_t q1 = ( ql0         & 0x0F0F0F0Fu) | ((qh0        & 0x03030303u) << 4);
                    const uint32_t q2 = ( ql32b       & 0x0F0F0F0Fu) | (((qh0 >> 2) & 0x03030303u) << 4);
                    const uint32_t q3 = ((ql0   >> 4) & 0x0F0F0F0Fu) | (((qh0 >> 4) & 0x03030303u) << 4);
                    const uint32_t q4 = ((ql32b >> 4) & 0x0F0F0F0Fu) | (((qh0 >> 6) & 0x03030303u) << 4);
                    const int is = tt >> 2;
                    const int yb = n*32;
                    isum[is    ] = halo_dot4(q1, (int) yq[yb + tt     ], isum[is    ]);
                    isum[is + 2] = halo_dot4(q2, (int) yq[yb + 8 + tt ], isum[is + 2]);
                    isum[is + 4] = halo_dot4(q3, (int) yq[yb + 16 + tt], isum[is + 4]);
                    isum[is + 6] = halo_dot4(q4, (int) yq[yb + 24 + tt], isum[is + 6]);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int jj = n*8 + j;
                    bacc = fmaf((float) sc[jj], yd[jj >> 1]*(float) isum[j] - 32.f*ys[jj], bacc);
                }
            }
            acc += d * bacc;
        }
    }
    dst[lrow] = acc;
}

void ggml_cuda_halo_qkv_launch(
        cudaStream_t stream,
        const void * wq, const void * wk, const void * wv, const float * y,
        float * dq, float * dk, float * dv,
        int nrq, int nrk, int nrv,
        int64_t srbq, int64_t srbk, int64_t srbv,
        int64_t k, bool v_is_q6) {
    const int r0 = nrq, r1 = nrq + nrk, r2 = nrq + nrk + nrv;
    const int blocks = (r2 + HALO_WG - 1) / HALO_WG;
    if (k == 2048) {
        if (v_is_q6) {
            halo_mmv_qkv<8, true ><<<blocks, HALO_WG, 0, stream>>>(wq, wk, wv, y, dq, dk, dv, r0, r1, r2, srbq, srbk, srbv);
        } else {
            halo_mmv_qkv<8, false><<<blocks, HALO_WG, 0, stream>>>(wq, wk, wv, y, dq, dk, dv, r0, r1, r2, srbq, srbk, srbv);
        }
    } else {
        GGML_ABORT("halo qkv: unsupported k");
    }
}

// Scan for a decode-time QKV triple starting at node i: MUL_MAT(wq 2048x4096 q4_K)
// followed within a short window by MUL_MAT(wk 2048x512 q4_K) and
// MUL_MAT(wv 2048x512 q4_K|q6_K), all sharing src1 (the attention input).
// On match: launch the fused kernel, append the k/v nodes to skip_list, return true.
bool ggml_cuda_halo_try_qkv(
        ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
        std::vector<const ggml_tensor *> & skip_list) {
    if (getenv("HALO_QKV_ENABLE") == nullptr) {   // fused QKV loses to stream concurrency; opt-in for research
        return false;
    }
    const ggml_tensor * qn = cgraph->nodes[i];
    if (qn->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * wq = qn->src[0];
    const ggml_tensor * y  = qn->src[1];
    if (wq->type != GGML_TYPE_Q4_K || y->type != GGML_TYPE_F32 ||
        qn->type != GGML_TYPE_F32 || qn->ne[1] != 1 || qn->ne[2] != 1) {
        return false;
    }
    if (wq->ne[0] != 2048 || wq->ne[1] < 1024 || !ggml_is_contiguous(y)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    const ggml_tensor * kn = nullptr;
    const ggml_tensor * vn = nullptr;
    const int lim = i + 7 < cgraph->n_nodes ? i + 7 : cgraph->n_nodes;
    for (int j = i + 1; j < lim; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op != GGML_OP_MUL_MAT || n->src[1] != y) {
            continue;
        }
        if (n->ne[1] != 1 || n->ne[2] != 1 || n->type != GGML_TYPE_F32) {
            return false;
        }
        const ggml_tensor * w = n->src[0];
        if (w->ne[0] != wq->ne[0] || w->ne[1] > wq->ne[1]) {
            return false;
        }
        if (!kn) {
            if (w->type != GGML_TYPE_Q4_K) {
                return false;
            }
            kn = n;
        } else {
            if (w->type == GGML_TYPE_Q4_K) {
                vn = n;
            } else if (w->type == GGML_TYPE_Q6_K &&
                       (w->nb[1] % 16) == 0 && (((uintptr_t) w->data) % 16) == 0) {
                vn = n;
            } else {
                return false;
            }
            break;
        }
    }
    if (!kn || !vn) {
        return false;
    }
    const ggml_tensor * wk = kn->src[0];
    const ggml_tensor * wv = vn->src[0];
    ggml_cuda_halo_qkv_launch(
        ctx.stream(),
        wq->data, wk->data, wv->data, (const float *) y->data,
        (float *) qn->data, (float *) kn->data, (float *) vn->data,
        (int) wq->ne[1], (int) wk->ne[1], (int) wv->ne[1],
        wq->nb[1], wk->nb[1], wv->nb[1],              // byte strides throughout
        wq->ne[0], wv->type == GGML_TYPE_Q6_K);
    skip_list.push_back(kn);
    skip_list.push_back(vn);
    return true;
}

// ---------------- F16 kernels: row-walk (mid-m) + 16-lane (giant/small m) ----------------
template <int WGS>
static __global__ void halo_mmv_f16_deep(
        const __half * __restrict__ A, const float * __restrict__ y,
        float * __restrict__ dst, const int nrows, const int k,
        const int64_t stride_row_halves, const float * __restrict__ x_bias) {
    extern __shared__ float hsy[];
    for (int i = threadIdx.x; i < k; i += WGS) hsy[i] = y[i];
    __syncthreads();
    const int row = blockIdx.x*WGS + threadIdx.x;
    if (row >= nrows) return;
    const uint4 * base = (const uint4 *)(A + (size_t) row*stride_row_halves);
    float acc = 0.f;
    const int nv = k/8;
    for (int v = 0; v < nv; v += 8) {
        uint4 w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = base[v + i];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __half2 * h2 = (const __half2 *) &w[i];
            const float * yb = &hsy[(v + i)*8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 hf = __half22float2(h2[j]);
                acc = fmaf(hf.x, yb[2*j],   acc);
                acc = fmaf(hf.y, yb[2*j+1], acc);
            }
        }
    }
    if (x_bias != nullptr) acc += x_bias[row];
    dst[row] = acc;
}

template <int WGS>
static __global__ void halo_mmv_f16_l16(
        const __half * __restrict__ A, const float * __restrict__ y,
        float * __restrict__ dst, const int nrows, const int k,
        const int64_t stride_row_halves, const float * __restrict__ x_bias) {
    extern __shared__ float hsy[];
    for (int i = threadIdx.x; i < k; i += WGS) hsy[i] = y[i];
    __syncthreads();
    const int itid = threadIdx.x & 15;
    const int grp  = threadIdx.x >> 4;
    const int64_t grow = (int64_t) blockIdx.x*(WGS/16) + grp;
    if (grow >= nrows) return;
    const int row = (int) grow;
    const uint4 * base = (const uint4 *)(A + (size_t) row*stride_row_halves);
    float acc = 0.f;
    const int nv = k/8;
    for (int v = itid; v < nv; v += 16) {
        const uint4 w = base[v];
        const __half2 * h2 = (const __half2 *) &w;
        const float * yb = &hsy[v*8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 hf = __half22float2(h2[j]);
            acc = fmaf(hf.x, yb[2*j],   acc);
            acc = fmaf(hf.y, yb[2*j+1], acc);
        }
    }
    acc += __shfl_down(acc, 8, 16);
    acc += __shfl_down(acc, 4, 16);
    acc += __shfl_down(acc, 2, 16);
    acc += __shfl_down(acc, 1, 16);
    if (itid == 0) {
        float r = acc;
        if (x_bias != nullptr) r += x_bias[row];
        dst[row] = r;
    }
}

// host: support + launch for dense decode f16 matvec
bool ggml_cuda_halo_f16_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (getenv("HALO_F16_DISABLE") != nullptr) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) return false;
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ids) return false;
    if (dst->ne[1] != 1 || dst->ne[2] != 1 || src0->ne[2] != 1) return false;
    const int64_t k = src0->ne[0];
    if (k % 8 != 0 || k > 16384) return false;          // LDS: k*4 bytes
    if (src0->ne[1] < 32768) return false;              // head-class only: stock mmvf wins below (256-lane co-op, ~198 GB/s in-graph)
    if ((src0->nb[1] % 16) != 0 || (((uintptr_t) src0->data) % 16) != 0) return false;
    if (!ggml_is_contiguous(src1) || src1->ne[1] != 1) return false;
    if (fusion) {
        if (fusion->gate || fusion->gate_bias || fusion->x_scale || fusion->gate_scale) return false;
        if (fusion->x_bias) {
            if (fusion->x_bias->type != GGML_TYPE_F32 || fusion->x_bias->ne[0] != dst->ne[0]) return false;
        }
    }
    return true;
}

void ggml_cuda_halo_f16(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    const int nrows = (int) src0->ne[1];
    const int k     = (int) src0->ne[0];
    const int64_t srh = src0->nb[1] / sizeof(__half);
    const float * x_bias = (fusion && fusion->x_bias) ? (const float *) fusion->x_bias->data : nullptr;
    cudaStream_t stream = ctx.stream();
    const size_t lds = (size_t) k * sizeof(float);
    if (true) {   // 16-lane everywhere: row-walk collapses under in-graph TLB pressure
        const int blocks = (int)(((int64_t) nrows*16 + HALO_WG - 1) / HALO_WG);
        halo_mmv_f16_l16<HALO_WG><<<blocks, HALO_WG, lds, stream>>>(
            (const __half *) src0->data, (const float *) src1->data, (float *) dst->data,
            nrows, k, srh, x_bias);
    } else {
        const int blocks = (nrows + HALO_WG - 1) / HALO_WG;
        halo_mmv_f16_deep<HALO_WG><<<blocks, HALO_WG, lds, stream>>>(
            (const __half *) src0->data, (const float *) src1->data, (float *) dst->data,
            nrows, k, srh, x_bias);
    }
}

// ---- host side ----

bool ggml_cuda_halo_mmvq_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (getenv("HALO_MMVQ_DISABLE") != nullptr) {
        return false;
    }
    const bool dbg = getenv("HALO_DEBUG") != nullptr;
#define HALO_REJ(why) do { if (dbg) fprintf(stderr, "halo-rej %s k=%lld r=%lld ids=%d ne1=%lld sne1=%lld sne2=%lld fus=%d gate=%d\n", why, (long long)src0->ne[0], (long long)src0->ne[1], ids?1:0, (long long)dst->ne[1], (long long)src1->ne[1], (long long)src1->ne[2], fusion?1:0, (fusion&&fusion->gate)?1:0); return false; } while(0)

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    if ((src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        HALO_REJ("types");
    }
    if (src0->type == GGML_TYPE_Q6_K) {
        if (getenv("HALO_Q6_DISABLE") != nullptr) {
            return false;
        }
        if (ids && (src0->nb[1] % 16) != 0) {
            HALO_REJ("q6idsalign");   // stock beats v3 on misaligned expert rows
        }
        if (fusion && fusion->gate) {
            HALO_REJ("q6gate");
        }

    }
    const int64_t k = src0->ne[0];
    if (k != 2048 && k != 768 && k != 4096) {
        HALO_REJ("ksize");
    }
    if (ids) {
        if (dst->ne[2] != 1) {
            HALO_REJ("ncols");
        }
        if (ids->type != GGML_TYPE_I32) {
            HALO_REJ("idstype");
        }
        const int64_t nch_y = src1->ne[1];
        if (nch_y != 1 && nch_y != dst->ne[1]) {
            HALO_REJ("nchy");
        }
        if (nch_y * (k/256) * 256 > 6144) {
            HALO_REJ("lds");
        }
    } else {
        if (getenv("HALO_DENSE_DISABLE") != nullptr) {
            return false;
        }
        if (dst->ne[1] != 1 || dst->ne[2] != 1 || src0->ne[2] != 1) {
            HALO_REJ("denseshape");
        }
        if (src0->ne[1] < (src0->type == GGML_TYPE_Q6_K ? 128 : 1024)) {
            HALO_REJ("denserows");
        }
        if ((k/256) * 256 > 6144) {
            HALO_REJ("denselds");
        }
    }
    if (src1->nb[0] != sizeof(float)) {
        HALO_REJ("stride");
    }
    if (fusion) {
        if (fusion->gate_bias || fusion->x_scale || fusion->gate_scale) {
            HALO_REJ("bias");
        }
        if (fusion->x_bias) {
            if (ids || fusion->gate || fusion->x_bias->type != GGML_TYPE_F32 || fusion->x_bias->ne[0] != dst->ne[0]) {
                HALO_REJ("bias");
            }
        }
        if (fusion->gate) {
            if (fusion->gate->type != GGML_TYPE_Q4_K || !ggml_are_same_stride(fusion->gate, src0)) {
                HALO_REJ("gatestride");
            }
        }
    }
#undef HALO_REJ
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
    const float * x_bias = (fusion && fusion->x_bias) ? (const float *) fusion->x_bias->data : nullptr;

    const int64_t total  = nch_dst * nrows;
    const int     blocks = (int)((total + HALO_WG - 1) / HALO_WG);
    cudaStream_t  stream = ctx.stream();

    const int32_t * ids_d = ids ? (const int32_t *) ids->data : nullptr;
    if (src0->type == GGML_TYPE_Q6_K) {
        const int64_t srb = src0->nb[1];
        const int64_t scb = src0->nb[2];
        if (getenv("HALO_DEBUG") != nullptr) {
            static int dbg_count = 0;
            if (dbg_count++ < 6) {
                fprintf(stderr, "halo-q6-accept k=%lld nrows=%lld nch_dst=%lld srb=%lld scb=%lld schd=%lld nch_y=%d schy=%lld xptr=%p yptr=%p ids=%p sne=[%lld,%lld,%lld] snb=[%zu,%zu,%zu] dne=[%lld,%lld,%lld] dnb=[%zu,%zu]\n",
                    (long long) k, (long long) nrows, (long long) nch_dst, (long long) srb, (long long) scb,
                    (long long) stride_ch_dst, nch_y, (long long) stride_ch_y,
                    src0->data, src1->data, ids ? ids->data : nullptr,
                    (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2],
                    src1->nb[0], src1->nb[1], src1->nb[2],
                    (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2],
                    dst->nb[0], dst->nb[1]);
            }
        }
        const int q6blocks = (int)((total*16 + HALO_WG - 1) / HALO_WG);
#define HALO_LAUNCH6(NSBV)         halo_mmv_q6k<NSBV><<<q6blocks, HALO_WG, 0, stream>>>(             src0->data, (const float *) src1->data, ids_d, (float *) dst->data,             (int) nrows, (int) nch_dst, srb, scb, stride_ch_dst, nch_y, stride_ch_y, x_bias)
        switch (k) {
            case 2048: HALO_LAUNCH6(8);  break;
            case  768: HALO_LAUNCH6(3);  break;
            case 4096: HALO_LAUNCH6(16); break;
            default: GGML_ABORT("halo q6: unsupported k");
        }
#undef HALO_LAUNCH6
        return;
    }
#define HALO_LAUNCH(NSBV, GATEV, GATEPTR, GLUV)     halo_mmv_q4k<NSBV, GATEV><<<blocks, HALO_WG, 0, stream>>>(         src0->data, GATEPTR, (const float *) src1->data, ids_d,         (float *) dst->data, (int) nrows, (int) nch_dst, stride_row_blk, stride_ch_blk,         stride_ch_dst, GLUV, nch_y, stride_ch_y, x_bias)
    switch (k) {
        case 2048: if (gate) { HALO_LAUNCH(8,  true, gate, glu_op); } else { HALO_LAUNCH(8,  false, nullptr, 0); } break;
        case  768: if (gate) { HALO_LAUNCH(3,  true, gate, glu_op); } else { HALO_LAUNCH(3,  false, nullptr, 0); } break;
        case 4096: if (gate) { HALO_LAUNCH(16, true, gate, glu_op); } else { HALO_LAUNCH(16, false, nullptr, 0); } break;
        default: GGML_ABORT("halo: unsupported k");
    }
#undef HALO_LAUNCH
}
