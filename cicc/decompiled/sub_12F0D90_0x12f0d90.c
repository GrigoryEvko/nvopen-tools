// Function: sub_12F0D90
// Address: 0x12f0d90
//
__m128i *__fastcall sub_12F0D90(
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
        __int64 a12)
{
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  __m128i v19; // xmm6
  _BYTE v20[24]; // [rsp+0h] [rbp-30h] BYREF
  __m128i v21; // [rsp+18h] [rbp-18h] BYREF
  __int64 v22; // [rsp+28h] [rbp-8h]

  a1->m128i_i64[1] = 0x400000002LL;
  v13 = _mm_loadu_si128((const __m128i *)&a10);
  v14 = _mm_loadu_si128((const __m128i *)&a11);
  v15 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)v20 = a9;
  *(__m128i *)&v20[8] = v13;
  v16 = _mm_loadu_si128((const __m128i *)&a8);
  v17 = _mm_loadu_si128((const __m128i *)v20);
  v22 = a12;
  v21 = v14;
  v18 = _mm_loadu_si128((const __m128i *)&v20[16]);
  v19 = _mm_loadu_si128((const __m128i *)&v21.m128i_u64[1]);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  a1[1] = v15;
  a1[2] = v16;
  a1[3] = v17;
  a1[4] = v18;
  a1[5] = v19;
  return a1;
}
