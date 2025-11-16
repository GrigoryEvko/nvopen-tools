// Function: sub_2178FB0
// Address: 0x2178fb0
//
__int64 *__fastcall sub_2178FB0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int16 v9; // ax

  v9 = *(_WORD *)(a1 + 24);
  if ( v9 == 199 )
    return sub_2177C80(a1, *(double *)a7.m128i_i64, a8, a9, a2, a3, a4);
  if ( v9 > 199 )
  {
    if ( v9 == 200 )
      return sub_2177B20(a1, *(double *)a7.m128i_i64, a8, a9, a2, a3, a4);
    return 0;
  }
  if ( v9 != 44 )
  {
    if ( v9 == 80 )
      return sub_21789C0(a1, a2, a3, a7);
    return 0;
  }
  return (__int64 *)sub_21781F0(a1, a2, a3, a7, a8, a9, a4, a5, a6);
}
