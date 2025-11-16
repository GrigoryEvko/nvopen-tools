// Function: sub_22735E0
// Address: 0x22735e0
//
__m128i *__fastcall sub_22735E0(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11,
        __int64 a12,
        __int128 a13,
        __int128 a14,
        __int64 a15)
{
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm7
  __m128i v19; // xmm5
  __int64 v20; // rax
  __m128i v21; // xmm4
  __m128i v22; // xmm0
  __m128i v23; // xmm6
  _BYTE v25[24]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v26; // [rsp+18h] [rbp-48h] BYREF
  __int64 v27; // [rsp+28h] [rbp-38h]

  v15 = _mm_loadu_si128((const __m128i *)&a11);
  *(__m128i *)&v25[8] = _mm_loadu_si128((const __m128i *)&a10);
  v16 = _mm_loadu_si128((const __m128i *)&a7);
  v17 = _mm_loadu_si128((const __m128i *)&a8);
  v26 = v15;
  v18 = _mm_loadu_si128((const __m128i *)&a13);
  v19 = _mm_loadu_si128((const __m128i *)&v25[16]);
  *(_QWORD *)v25 = a9;
  v20 = a12;
  v21 = _mm_loadu_si128((const __m128i *)v25);
  v22 = _mm_loadu_si128((const __m128i *)&a14);
  a1[8].m128i_i64[0] = a15;
  v27 = v20;
  v23 = _mm_loadu_si128((const __m128i *)&v26.m128i_u64[1]);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  a1->m128i_i64[1] = 0x400000003LL;
  a1[1] = v16;
  a1[2] = v17;
  a1[3] = v21;
  a1[4] = v19;
  a1[5] = v23;
  a1[6] = v18;
  a1[7] = v22;
  return a1;
}
