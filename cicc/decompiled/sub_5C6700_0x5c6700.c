// Function: sub_5C6700
// Address: 0x5c6700
//
__int64 __fastcall sub_5C6700(__int64 a1, __int64 a2, char a3)
{
  char v3; // al
  char v4; // dl
  char v6; // al
  char v7; // dl

  if ( a3 == 21 )
  {
    v6 = *(_BYTE *)(a2 + 41);
    v7 = *(_BYTE *)(a1 + 8);
    if ( (v6 & 0x10) == 0 || v7 != 17 )
    {
      if ( (v6 & 0x20) == 0 )
      {
        if ( v7 == 16 )
        {
          *(_BYTE *)(a2 + 41) |= 0x10u;
          return a2;
        }
        goto LABEL_15;
      }
      if ( v7 != 16 )
      {
LABEL_15:
        *(_BYTE *)(a2 + 41) |= 0x20u;
        return a2;
      }
    }
LABEL_10:
    sub_6851C0(2907, a1 + 56);
    return a2;
  }
  if ( a3 != 12 )
    sub_721090(a1);
  v3 = *(_BYTE *)(a2 + 120);
  v4 = *(_BYTE *)(a1 + 8);
  if ( (v3 & 0x40) != 0 && v4 == 17 )
    goto LABEL_10;
  if ( v3 < 0 )
  {
    if ( v4 != 16 )
      goto LABEL_7;
    goto LABEL_10;
  }
  if ( v4 != 16 )
  {
LABEL_7:
    *(_BYTE *)(a2 + 120) |= 0x80u;
    return a2;
  }
  *(_BYTE *)(a2 + 120) |= 0x40u;
  return a2;
}
