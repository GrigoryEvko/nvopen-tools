// Function: sub_38A2440
// Address: 0x38a2440
//
__int64 __fastcall sub_38A2440(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  if ( *(_DWORD *)(a1 + 64) == 8 )
    return sub_38A2390((__int64 **)a1, a2, 0, *(double *)a3.m128_u64, a4, a5);
  else
    return sub_3897BF0(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
