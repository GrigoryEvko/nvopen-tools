// Function: sub_1677880
// Address: 0x1677880
//
__int64 __fastcall sub_1677880(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  if ( *(_BYTE *)(a2 + 16) > 3u )
    return 0;
  else
    return sub_1675980(*(_QWORD *)(a1 + 8), a2, 1, a3, a4, a5, a6, a7, a8, a9, a10);
}
