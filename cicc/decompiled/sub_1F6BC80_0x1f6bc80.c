// Function: sub_1F6BC80
// Address: 0x1f6bc80
//
__int64 *__fastcall sub_1F6BC80(
        __int64 a1,
        unsigned __int64 a2,
        __int16 *a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        double a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int128 a11,
        __int128 a12)
{
  __int64 *v14; // r13
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int16 *v17; // rdx
  __int64 *v18; // rax
  unsigned int *v19; // rsi
  __int64 v20; // r14
  unsigned int *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  const void **v26; // r8
  __int128 v28; // [rsp+0h] [rbp-70h]
  __m128 v29; // [rsp+10h] [rbp-60h]
  __int128 v32; // [rsp+30h] [rbp-40h]

  *(_QWORD *)&v28 = a4;
  v14 = **(__int64 ***)a1;
  *((_QWORD *)&v28 + 1) = a5;
  v29 = (__m128)_mm_loadu_si128((const __m128i *)&a12);
  *(_QWORD *)&v32 = sub_1D309E0(
                      v14,
                      157,
                      *(_QWORD *)(a1 + 16),
                      **(unsigned int **)(a1 + 24),
                      *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
                      0,
                      *(double *)v29.m128_u64,
                      a8,
                      *(double *)a9.m128i_i64,
                      a11);
  *((_QWORD *)&v32 + 1) = v15;
  v16 = sub_1D309E0(
          **(__int64 ***)a1,
          157,
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          0,
          *(double *)v29.m128_u64,
          a8,
          *(double *)a9.m128i_i64,
          a10);
  v18 = sub_1D3A900(
          v14,
          **(_DWORD **)(a1 + 8),
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          a6,
          v29,
          a8,
          a9,
          v16,
          v17,
          v32,
          v29.m128_i64[0],
          v29.m128_i64[1]);
  v19 = *(unsigned int **)(a1 + 8);
  v20 = (__int64)v18;
  v21 = *(unsigned int **)(a1 + 24);
  v23 = v22;
  v24 = *(_QWORD *)(a1 + 16);
  v25 = *v21;
  v26 = (const void **)*((_QWORD *)v21 + 1);
  *(_QWORD *)&a12 = v20;
  *((_QWORD *)&a12 + 1) = v23;
  return sub_1D3A900(v14, *v19, v24, v25, v26, a6, v29, a8, a9, a2, a3, v28, v20, v23);
}
