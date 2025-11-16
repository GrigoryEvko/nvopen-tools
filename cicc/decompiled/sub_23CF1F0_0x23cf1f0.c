// Function: sub_23CF1F0
// Address: 0x23cf1f0
//
__int64 __fastcall sub_23CF1F0(_DWORD *a1, _BYTE *a2)
{
  int v2; // eax
  char v3; // dl
  unsigned int v4; // r8d
  int v5; // ecx
  int v7; // r8d
  int v8; // eax
  int v9; // eax

  v2 = sub_23CF1A0((__int64)a1);
  if ( !a2 )
    return 0;
  v3 = a2[33];
  v4 = 1;
  if ( (v3 & 0x40) != 0 )
    return v4;
  v5 = a1[141];
  if ( v5 != 1 )
  {
    if ( v5 != 4 )
    {
      if ( v5 == 5 )
      {
        if ( !v2 )
          return v4;
        if ( (a2[32] & 0xF) != 1 && !sub_B2FC80((__int64)a2) )
        {
          v8 = a2[32] & 0xF;
          if ( (unsigned int)(v8 - 4) > 1 && v8 != 2 )
          {
            LOBYTE(v7) = (((a2[32] & 0xF) + 7) & 0xFu) <= 1;
            LOBYTE(v8) = v8 == 3;
            return (v8 | v7) ^ 1u;
          }
        }
      }
      return 0;
    }
    return v4;
  }
  if ( (v3 & 3) == 1 )
    return 0;
  if ( a1[139] == 14 )
  {
    v9 = a1[140];
    if ( (v9 == 29 || v9 == 1) && ((a2[32] & 0xF) == 1 || sub_B2FC80((__int64)a2)) && *a2 == 3 )
      return 0;
  }
  LOBYTE(v4) = (a2[32] & 0xF) != 9;
  return v4;
}
