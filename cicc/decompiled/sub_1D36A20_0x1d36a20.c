// Function: sub_1D36A20
// Address: 0x1d36a20
//
__int64 *__fastcall sub_1D36A20(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11)
{
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  __int128 v16; // [rsp-10h] [rbp-60h]
  _OWORD v17[5]; // [rsp+0h] [rbp-50h] BYREF

  v11 = _mm_loadu_si128((const __m128i *)&a8);
  *((_QWORD *)&v16 + 1) = 5;
  v12 = _mm_loadu_si128((const __m128i *)&a9);
  v13 = _mm_loadu_si128((const __m128i *)&a10);
  *(_QWORD *)&v16 = v17;
  v14 = _mm_loadu_si128((const __m128i *)&a11);
  v17[0] = _mm_loadu_si128((const __m128i *)&a7);
  v17[1] = v11;
  v17[2] = v12;
  v17[3] = v13;
  v17[4] = v14;
  return sub_1D359D0(a1, a2, a3, a4, a5, 0, *(double *)v17, *(double *)v11.m128i_i64, v12, v16);
}
