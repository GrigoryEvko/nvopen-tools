// Function: sub_33E6A90
// Address: 0x33e6a90
//
__int64 __fastcall sub_33E6A90(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11)
{
  unsigned __int64 v12; // rax
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // rdx
  __int64 v16; // r9
  _OWORD v18[5]; // [rsp+0h] [rbp-50h] BYREF

  v12 = sub_33E5110(a1, a4, a5, a7, a8);
  v13 = _mm_loadu_si128((const __m128i *)&a10);
  v14 = _mm_loadu_si128((const __m128i *)&a11);
  v18[0] = _mm_loadu_si128((const __m128i *)&a9);
  v18[1] = v13;
  v18[2] = v14;
  return sub_33E66D0(a1, a2, a3, v12, v15, v16, (unsigned __int64 *)v18, 3);
}
