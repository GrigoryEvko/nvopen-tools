// Function: sub_33FC180
// Address: 0x33fc180
//
unsigned __int8 *__fastcall sub_33FC180(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
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
  _OWORD v16[5]; // [rsp+0h] [rbp-50h] BYREF

  v11 = _mm_loadu_si128((const __m128i *)&a8);
  v12 = _mm_loadu_si128((const __m128i *)&a9);
  v13 = _mm_loadu_si128((const __m128i *)&a10);
  v14 = _mm_loadu_si128((const __m128i *)&a11);
  v16[0] = _mm_loadu_si128((const __m128i *)&a7);
  v16[1] = v11;
  v16[2] = v12;
  v16[3] = v13;
  v16[4] = v14;
  return sub_33FBA10(a1, a2, a3, a4, a5, a6, (__int64)v16, 5);
}
