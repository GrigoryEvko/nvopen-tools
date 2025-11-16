// Function: sub_164BE60
// Address: 0x164be60
//
__int64 __fastcall sub_164BE60(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  if ( (*(_BYTE *)(a1 + 17) & 1) != 0 )
  {
    sub_164BAF0(a1);
    if ( (*(_BYTE *)(a1 + 23) & 0x10) == 0 )
      return sub_164B400(a1);
  }
  else if ( (*(_BYTE *)(a1 + 23) & 0x10) == 0 )
  {
    return sub_164B400(a1);
  }
  sub_16302F0((__int64 ***)a1, a2, a3, a4, a5, a6, a7, a8, a9);
  return sub_164B400(a1);
}
