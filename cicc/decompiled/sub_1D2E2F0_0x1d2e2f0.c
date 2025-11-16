// Function: sub_1D2E2F0
// Address: 0x1d2e2f0
//
__int64 *__fastcall sub_1D2E2F0(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __m128i v7; // xmm0
  _QWORD v9[4]; // [rsp+0h] [rbp-30h] BYREF
  __m128i v10; // [rsp+20h] [rbp-10h]

  v7 = _mm_loadu_si128((const __m128i *)&a7);
  v9[0] = a3;
  v9[1] = a4;
  v9[2] = a5;
  v9[3] = a6;
  v10 = v7;
  return sub_1D2E160(a1, a2, (__int64)v9, 3);
}
