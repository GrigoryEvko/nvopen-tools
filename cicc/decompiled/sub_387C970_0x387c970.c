// Function: sub_387C970
// Address: 0x387c970
//
__int64 __fastcall sub_387C970(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  if ( *((_BYTE *)a1 + 256) )
    return sub_387C030((__int64)a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  else
    return sub_3879DA0(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
