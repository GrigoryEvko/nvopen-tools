// Function: sub_1F6BDB0
// Address: 0x1f6bdb0
//
__int64 *__fastcall sub_1F6BDB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int16 a6,
        double a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int128 a11,
        __int128 a12)
{
  __m128 v13; // xmm0
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  __int16 *v18; // rdx
  __int64 *v19; // r14
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r13
  unsigned __int64 v25; // rax
  __int16 *v26; // rdx
  __int128 v28; // [rsp-20h] [rbp-90h]
  __int128 v29; // [rsp-20h] [rbp-90h]
  __int64 *v30; // [rsp+10h] [rbp-60h]
  __int128 v32; // [rsp+20h] [rbp-50h]
  __int128 v34; // [rsp+90h] [rbp+20h]

  v13 = (__m128)_mm_loadu_si128((const __m128i *)&a12);
  *(_QWORD *)&v32 = a4;
  *((_QWORD *)&v32 + 1) = a5;
  v30 = **(__int64 ***)a1;
  v14 = sub_1D309E0(
          v30,
          157,
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          0,
          *(double *)v13.m128_u64,
          a8,
          *(double *)a9.m128i_i64,
          a11);
  v16 = v15;
  v17 = sub_1D309E0(
          **(__int64 ***)a1,
          157,
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          0,
          *(double *)v13.m128_u64,
          a8,
          *(double *)a9.m128i_i64,
          a10);
  *((_QWORD *)&v28 + 1) = v16;
  *(_QWORD *)&v28 = v14;
  v19 = sub_1D3A900(
          v30,
          **(_DWORD **)(a1 + 8),
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          a6,
          v13,
          a8,
          a9,
          v17,
          v18,
          v28,
          v13.m128_i64[0],
          v13.m128_i64[1]);
  v21 = v20;
  v22 = sub_1D309E0(
          **(__int64 ***)a1,
          157,
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          0,
          *(double *)v13.m128_u64,
          a8,
          *(double *)a9.m128i_i64,
          v32);
  v24 = v23;
  *((_QWORD *)&v29 + 1) = a3;
  *(_QWORD *)&v29 = a2;
  v25 = sub_1D309E0(
          **(__int64 ***)a1,
          157,
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
          0,
          *(double *)v13.m128_u64,
          a8,
          *(double *)a9.m128i_i64,
          v29);
  *(_QWORD *)&v34 = v22;
  *((_QWORD *)&v34 + 1) = v24;
  return sub_1D3A900(
           v30,
           **(_DWORD **)(a1 + 8),
           *(_QWORD *)(a1 + 16),
           **(unsigned int **)(a1 + 24),
           *(const void ***)(*(_QWORD *)(a1 + 24) + 8LL),
           a6,
           v13,
           a8,
           a9,
           v25,
           v26,
           v34,
           (__int64)v19,
           v21);
}
