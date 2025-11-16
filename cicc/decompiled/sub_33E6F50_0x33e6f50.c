// Function: sub_33E6F50
// Address: 0x33e6f50
//
__int64 *__fastcall sub_33E6F50(
        __int64 *a1,
        int a2,
        __int64 a3,
        unsigned __int16 a4,
        __int64 a5,
        const __m128i *a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10)
{
  unsigned __int64 v13; // rax
  __m128i v14; // xmm1
  __int64 v15; // rdx
  _OWORD v18[5]; // [rsp+10h] [rbp-50h] BYREF

  v13 = sub_33E5110(a1, a7, a8, 1, 0);
  v14 = _mm_loadu_si128((const __m128i *)&a10);
  v18[0] = _mm_loadu_si128((const __m128i *)&a9);
  v18[1] = v14;
  return sub_33E6BC0(a1, a2, a3, a4, a5, a6, v13, v15, (unsigned __int64 *)v18, 2);
}
