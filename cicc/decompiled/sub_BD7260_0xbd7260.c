// Function: sub_BD7260
// Address: 0xbd7260
//
__int64 __fastcall sub_BD7260(__int64 a1, __int64 a2)
{
  char v2; // al

  if ( (*(_BYTE *)(a1 + 1) & 1) == 0 )
  {
    v2 = *(_BYTE *)(a1 + 7);
    if ( (v2 & 8) == 0 )
      goto LABEL_3;
LABEL_6:
    sub_BA6130(a1);
    if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
      return sub_BD6840(a1);
    goto LABEL_7;
  }
  sub_BD6EE0(a1);
  v2 = *(_BYTE *)(a1 + 7);
  if ( (v2 & 8) != 0 )
    goto LABEL_6;
LABEL_3:
  if ( (v2 & 0x20) == 0 )
    return sub_BD6840(a1);
LABEL_7:
  sub_B91E30(a1, a2);
  return sub_BD6840(a1);
}
