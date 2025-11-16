// Function: sub_12F1570
// Address: 0x12f1570
//
__m128i *__fastcall sub_12F1570(
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
        __int64 a15,
        __int128 a16,
        __int128 a17,
        __int64 a18)
{
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  __m128i v22; // xmm6
  __m128i v23; // xmm7
  __m128i v24; // xmm0
  __m128i v25; // xmm3
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  _BYTE v32[24]; // [rsp+0h] [rbp-80h] BYREF
  __m128i v33; // [rsp+18h] [rbp-68h] BYREF
  __int64 v34; // [rsp+28h] [rbp-58h]
  _BYTE v35[24]; // [rsp+50h] [rbp-30h] BYREF
  __m128i v36; // [rsp+68h] [rbp-18h] BYREF
  __int64 v37; // [rsp+78h] [rbp-8h]

  v18 = _mm_loadu_si128((const __m128i *)&a11);
  v19 = _mm_loadu_si128((const __m128i *)&a16);
  *(__m128i *)&v32[8] = _mm_loadu_si128((const __m128i *)&a10);
  v20 = _mm_loadu_si128((const __m128i *)&a7);
  v21 = _mm_loadu_si128((const __m128i *)&a8);
  *(_QWORD *)v32 = a9;
  v33 = v18;
  v22 = _mm_loadu_si128((const __m128i *)v32);
  v23 = _mm_loadu_si128((const __m128i *)&v32[16]);
  v34 = a12;
  v24 = _mm_loadu_si128((const __m128i *)&v33.m128i_u64[1]);
  v25 = _mm_loadu_si128((const __m128i *)&a17);
  *(__m128i *)&v35[8] = v19;
  *(_QWORD *)v35 = a15;
  v26 = _mm_loadu_si128((const __m128i *)&a13);
  v36 = v25;
  v37 = a18;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  a1[1] = v20;
  a1[2] = v21;
  a1[3] = v22;
  a1[4] = v23;
  a1[5] = v24;
  a1[6] = v26;
  v27 = _mm_loadu_si128((const __m128i *)&a14);
  v28 = _mm_loadu_si128((const __m128i *)v35);
  v29 = _mm_loadu_si128((const __m128i *)&v35[16]);
  v30 = _mm_loadu_si128((const __m128i *)&v36.m128i_u64[1]);
  a1->m128i_i64[1] = 0x400000004LL;
  a1[7] = v27;
  a1[8] = v28;
  a1[9] = v29;
  a1[10] = v30;
  return a1;
}
