// Function: sub_33EC3F0
// Address: 0x33ec3f0
//
__int64 *__fastcall sub_33EC3F0(
        _QWORD *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  _QWORD v11[4]; // [rsp+0h] [rbp-40h] BYREF
  __m128i v12; // [rsp+20h] [rbp-20h]
  __m128i v13; // [rsp+30h] [rbp-10h]

  v8 = _mm_loadu_si128((const __m128i *)&a7);
  v9 = _mm_loadu_si128((const __m128i *)&a8);
  v11[0] = a3;
  v11[1] = a4;
  v11[2] = a5;
  v11[3] = a6;
  v12 = v8;
  v13 = v9;
  return sub_33EC210(a1, a2, (__int64)v11, 4);
}
