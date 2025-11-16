// Function: sub_1D24690
// Address: 0x1d24690
//
__int64 *__fastcall sub_1D24690(
        _QWORD *a1,
        unsigned __int16 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11,
        __int128 a12)
{
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  _OWORD v16[4]; // [rsp+0h] [rbp-40h] BYREF

  v12 = _mm_loadu_si128((const __m128i *)&a10);
  v13 = _mm_loadu_si128((const __m128i *)&a11);
  v14 = _mm_loadu_si128((const __m128i *)&a12);
  v16[0] = _mm_loadu_si128((const __m128i *)&a9);
  v16[1] = v12;
  v16[2] = v13;
  v16[3] = v14;
  return sub_1D24450(a1, a2, a3, a4, a5, a6, a7, a8, (__int64 *)v16, 4);
}
