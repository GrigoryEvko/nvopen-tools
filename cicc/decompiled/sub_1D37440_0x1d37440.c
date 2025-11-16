// Function: sub_1D37440
// Address: 0x1d37440
//
__int64 *__fastcall sub_1D37440(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        const void ***a4,
        int a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int128 a11)
{
  __m128i v11; // xmm1
  __int128 v13; // [rsp-10h] [rbp-30h]
  _OWORD v14[2]; // [rsp+0h] [rbp-20h] BYREF

  v11 = _mm_loadu_si128((const __m128i *)&a11);
  *((_QWORD *)&v13 + 1) = 2;
  *(_QWORD *)&v13 = v14;
  v14[0] = _mm_loadu_si128((const __m128i *)&a10);
  v14[1] = v11;
  return sub_1D36D80(a1, a2, a3, a4, a5, *(double *)v14, *(double *)v11.m128i_i64, a9, a6, v13);
}
