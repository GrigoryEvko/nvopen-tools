// Function: sub_8D5070
// Address: 0x8d5070
//
_BOOL8 __fastcall sub_8D5070(__int64 a1)
{
  __int64 v1; // rbx
  char i; // al
  __int64 v4; // rax
  __int64 v5; // r12
  char v6; // al
  int v7; // r14d
  __int64 v8; // r13
  _QWORD *v9; // rax
  char v10; // al

  if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 && (sub_8D4C10(a1, dword_4F077C4 != 2) & 2) != 0 )
  {
    if ( (_DWORD)qword_4F077B4 )
    {
      if ( (unsigned __int64)(qword_4F077A0 - 30400LL) <= 0x2580 )
        return 0;
    }
    else if ( dword_4F077BC && !sub_8D3A70(a1) )
    {
      return 0;
    }
  }
  v1 = sub_8D4130(a1);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (unsigned __int8)(i - 2) <= 3u )
    return 1;
  if ( i == 6 )
    return (*(_BYTE *)(v1 + 168) & 1) == 0;
  if ( (unsigned __int8)(i - 19) <= 1u || i == 13 )
    return 1;
  if ( (unsigned __int8)(i - 9) > 2u )
    return 0;
  v4 = *(_QWORD *)(*(_QWORD *)v1 + 96LL);
  if ( *(_QWORD *)(v4 + 24) )
  {
    if ( (*(_BYTE *)(v4 + 177) & 2) == 0 )
      return 0;
  }
  if ( (*(_WORD *)(v4 + 176) & 0x8A0) != 0 )
    return 0;
  v5 = *(_QWORD *)(v4 + 8);
  if ( !v5 )
    return 1;
  v6 = *(_BYTE *)(v5 + 80);
  v7 = 0;
  if ( v6 != 17 )
    goto LABEL_16;
  v5 = *(_QWORD *)(v5 + 88);
  if ( !v5 )
    return 1;
  v6 = *(_BYTE *)(v5 + 80);
  v7 = 1;
LABEL_16:
  if ( v6 == 20 )
    goto LABEL_21;
LABEL_17:
  v8 = *(_QWORD *)(v5 + 88);
  v9 = **(_QWORD ***)(*(_QWORD *)(v8 + 152) + 168LL);
  if ( !v9 || *v9 )
  {
    v10 = *(_BYTE *)(v8 + 206);
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(v8 + 193) & 0x10) == 0 )
  {
    v10 = *(_BYTE *)(v8 + 206);
    if ( (v10 & 8) == 0 )
    {
LABEL_20:
      if ( (v10 & 0x10) == 0 || !(unsigned int)sub_72F500(*(_QWORD *)(v5 + 88), v1, 0, 1, 1) )
      {
LABEL_21:
        while ( v7 )
        {
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            break;
          if ( *(_BYTE *)(v5 + 80) != 20 )
            goto LABEL_17;
        }
        return 1;
      }
    }
  }
  if ( (*(_BYTE *)(v8 + 194) & 4) != 0 )
    goto LABEL_21;
  return 0;
}
