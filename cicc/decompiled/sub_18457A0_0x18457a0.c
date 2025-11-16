// Function: sub_18457A0
// Address: 0x18457a0
//
__int64 __fastcall sub_18457A0(
        __m128 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        __m128 a8,
        __int64 a9,
        __int64 a10)
{
  double v10; // xmm4_8
  double v11; // xmm5_8

  if ( sub_15E4F60(a10) || (*(_BYTE *)(a10 + 32) & 0xFu) - 7 > 1 || (unsigned __int8)sub_15E3650(a10, 0) )
    return 0;
  else
    return sub_18449B0(a10, a1, a2, a3, a4, v10, v11, a7, a8);
}
