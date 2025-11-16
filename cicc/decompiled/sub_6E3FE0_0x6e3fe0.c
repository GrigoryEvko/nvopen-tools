// Function: sub_6E3FE0
// Address: 0x6e3fe0
//
__int64 __fastcall sub_6E3FE0(__int64 a1, int a2, __int64 a3)
{
  char v4; // al
  __int64 v5; // r12
  __int64 v7; // r15
  char v8; // al
  unsigned __int8 v9; // cl
  _BYTE *v10; // rdi
  __int64 v11; // rax

  v4 = *(_BYTE *)(a3 + 16);
  if ( v4 == 1 )
  {
    v5 = *(_QWORD *)(a3 + 144);
  }
  else
  {
    if ( v4 != 2 )
      return 0;
    v5 = *(_QWORD *)(a3 + 288);
    if ( !v5 )
    {
      if ( *(_BYTE *)(a3 + 317) != 12 || *(_BYTE *)(a3 + 320) != 1 )
        return 0;
      v5 = sub_72E9A0(a3 + 144);
    }
  }
  if ( v5 == a1 )
  {
LABEL_37:
    if ( !v5 || a1 == v5 )
      return 0;
    if ( *(_BYTE *)(v5 + 24) != 5 )
      goto LABEL_31;
    return v5;
  }
  if ( !v5 )
    return 0;
  while ( 1 )
  {
    v8 = *(_BYTE *)(v5 + 24);
    if ( v8 == 1 )
    {
      v9 = *(_BYTE *)(v5 + 56);
      if ( v9 > 0x15u )
      {
        if ( v9 != 116 )
        {
LABEL_19:
          if ( (*(_BYTE *)(v5 + 27) & 2) == 0 )
            goto LABEL_30;
          if ( !(unsigned int)sub_730740(v5) && (*(_BYTE *)(v5 + 59) & 1) == 0 )
            goto LABEL_37;
        }
      }
      else if ( v9 <= 3u || ((1LL << v9) & 0x202310) == 0 )
      {
        goto LABEL_19;
      }
      v5 = *(_QWORD *)(v5 + 72);
      goto LABEL_13;
    }
    if ( v8 == 2 )
      break;
    if ( v8 != 5 )
      goto LABEL_30;
    v7 = *(_QWORD *)(v5 + 56);
    if ( !(unsigned int)sub_7307F0(v7) )
      goto LABEL_37;
    v5 = sub_6E3F50(v7);
LABEL_13:
    if ( !v5 )
      return 0;
    if ( v5 == a1 )
      goto LABEL_37;
  }
  v10 = *(_BYTE **)(v5 + 56);
  if ( v10[173] == 12 && (v10[177] & 0x20) == 0 && v10[176] == 1 )
  {
    v11 = sub_72E9A0(v10);
    if ( !v11 )
      goto LABEL_37;
    v5 = v11;
    goto LABEL_13;
  }
LABEL_30:
  if ( a1 == v5 )
    return 0;
LABEL_31:
  if ( !(unsigned int)sub_730740(v5)
    && ((unsigned int)(a2 - 1) > 2 || *(_BYTE *)(v5 + 24) != 1 || (*(_BYTE *)(v5 + 59) & 1) == 0) )
  {
    return 0;
  }
  return v5;
}
