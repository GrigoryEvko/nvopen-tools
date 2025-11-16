// Function: sub_5C8600
// Address: 0x5c8600
//
_BYTE *__fastcall sub_5C8600(__int64 a1, __int64 a2)
{
  __int64 i; // rax
  char v3; // al
  __int64 j; // rax
  _BYTE *v5; // r13
  __int64 k; // rax
  __int64 v8; // rax

  if ( *(char *)(a2 + 192) >= 0 || (*(_BYTE *)(a2 + 89) & 4) != 0 )
  {
    if ( *(_BYTE *)(a2 + 174) != 5 )
      goto LABEL_3;
  }
  else
  {
    sub_684B10(3507, a1 + 56, "__global__");
    if ( *(_BYTE *)(a2 + 174) != 5 )
      goto LABEL_3;
  }
  sub_684AA0(7, 3644, a1 + 56);
LABEL_3:
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 56LL) )
    sub_684AA0(7, 3647, a1 + 56);
  v3 = *(_BYTE *)(a2 + 198);
  if ( !unk_4D045EC && (v3 & 0x30) == 0x10 || (v3 & 8) != 0 )
    sub_6851C0(3481, a1 + 56);
  if ( (*(_BYTE *)(a2 + 195) & 8) == 0 )
  {
    if ( (*(_BYTE *)(a2 + 207) & 0x10) != 0 )
    {
      sub_684AA0(7, 3506, a1 + 56);
    }
    else
    {
      v8 = sub_8D21C0(*(_QWORD *)(a2 + 152));
      if ( !(unsigned int)sub_8D2600(*(_QWORD *)(v8 + 160)) )
        sub_684AA0(7, 3505, a1 + 56);
    }
  }
  for ( j = *(_QWORD *)(a2 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  if ( (*(_BYTE *)(*(_QWORD *)(j + 168) + 16LL) & 1) != 0 )
    sub_6851C0(3503, a1 + 56);
  *(_BYTE *)(a2 + 198) |= 0x30u;
  v5 = sub_5C6B80(a1, (_BYTE *)a2, 11);
  if ( (*(_BYTE *)(a1 + 11) & 1) != 0 )
  {
    for ( k = *(_QWORD *)(a2 + 152); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    sub_5C6D70(**(_QWORD ***)(k + 168), *(_QWORD *)(a1 + 56));
  }
  return v5;
}
