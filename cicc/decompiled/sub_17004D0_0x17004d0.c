// Function: sub_17004D0
// Address: 0x17004d0
//
__int64 __fastcall sub_17004D0(__int64 a1, __int64 a2, _BYTE *a3)
{
  _BOOL4 v4; // ecx
  int v5; // r15d
  int v6; // edx
  _BOOL4 v7; // r14d
  bool v8; // zf
  char v9; // al
  bool v10; // dl
  unsigned int v11; // eax
  int v13; // edx

  if ( a3 )
  {
    if ( (a3[33] & 0x40) != 0 )
      return 1;
    sub_1633DF0(a2);
    v5 = sub_1700490(a1);
    if ( (a3[33] & 3) == 1 )
      return 0;
  }
  else
  {
    if ( sub_1633DF0(a2) )
      return 0;
    v5 = sub_1700490(a1);
  }
  v6 = *(_DWORD *)(a1 + 524);
  if ( v6 == 1 )
    return 1;
  LOBYTE(v4) = v6 == 3 && *(_DWORD *)(a1 + 516) == 15;
  v7 = v4;
  if ( v4 )
    return 1;
  if ( a3 )
  {
    v8 = !sub_17004A0(a1);
    v9 = a3[32];
    if ( !v8 && (v9 & 0xF) == 9 )
      return 0;
    if ( (v9 & 0x30) != 0 )
      return 1;
    v6 = *(_DWORD *)(a1 + 524);
  }
  if ( v6 != 3 )
  {
    if ( v5 && !(unsigned int)sub_1633D40(a2) )
      return 0;
    v10 = 1;
    if ( !a3 )
    {
LABEL_20:
      v11 = *(_DWORD *)(a1 + 504) - 16;
      if ( v11 > 2 && v10 )
      {
        LOBYTE(v11) = v5 == 0;
        return v7 | v11;
      }
      return 0;
    }
    if ( (a3[32] & 0xF) == 1 || sub_15E4F60((__int64)a3) )
    {
      if ( a3[16] || !(unsigned __int8)sub_1560180((__int64)(a3 + 112), 31) )
      {
        v10 = (a3[33] & 0x1C) == 0;
        v7 = (*(_BYTE *)(a1 + 841) & 2) != 0;
        if ( (*(_BYTE *)(a1 + 841) & 2) != 0 )
          LOBYTE(v7) = a3[16] == 3;
        goto LABEL_20;
      }
      return 0;
    }
    return 1;
  }
  if ( !v5 )
    return 1;
  if ( !a3 )
    return 0;
  if ( (a3[32] & 0xF) == 1 )
    return 0;
  if ( sub_15E4F60((__int64)a3) )
    return 0;
  v13 = a3[32] & 0xF;
  if ( (unsigned int)(v13 - 4) <= 1 || v13 == 2 )
    return 0;
  LOBYTE(v13) = v13 == 3;
  return (v13 | ((((a3[32] & 0xF) + 7) & 0xFu) <= 1)) ^ 1u;
}
