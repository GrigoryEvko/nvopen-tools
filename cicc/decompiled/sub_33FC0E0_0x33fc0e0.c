// Function: sub_33FC0E0
// Address: 0x33fc0e0
//
unsigned __int8 *__fastcall sub_33FC0E0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10)
{
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  _OWORD v14[4]; // [rsp+0h] [rbp-40h] BYREF

  v10 = _mm_loadu_si128((const __m128i *)&a8);
  v11 = _mm_loadu_si128((const __m128i *)&a9);
  v12 = _mm_loadu_si128((const __m128i *)&a10);
  v14[0] = _mm_loadu_si128((const __m128i *)&a7);
  v14[1] = v10;
  v14[2] = v11;
  v14[3] = v12;
  return sub_33FBA10(a1, a2, a3, a4, a5, a6, (__int64)v14, 4);
}
