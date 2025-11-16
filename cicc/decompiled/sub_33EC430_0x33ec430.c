// Function: sub_33EC430
// Address: 0x33ec430
//
__int64 *__fastcall sub_33EC430(
        _QWORD *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  _QWORD v13[4]; // [rsp+0h] [rbp-50h] BYREF
  __m128i v14; // [rsp+20h] [rbp-30h]
  __m128i v15; // [rsp+30h] [rbp-20h]
  __m128i v16; // [rsp+40h] [rbp-10h]

  v9 = _mm_loadu_si128((const __m128i *)&a7);
  v10 = _mm_loadu_si128((const __m128i *)&a8);
  v13[0] = a3;
  v11 = _mm_loadu_si128((const __m128i *)&a9);
  v13[1] = a4;
  v13[2] = a5;
  v13[3] = a6;
  v14 = v9;
  v15 = v10;
  v16 = v11;
  return sub_33EC210(a1, a2, (__int64)v13, 5);
}
