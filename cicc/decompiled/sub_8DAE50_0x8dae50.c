// Function: sub_8DAE50
// Address: 0x8dae50
//
__int64 __fastcall sub_8DAE50(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  unsigned int i; // r8d
  char v9; // dl
  char j; // al
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // r8

  for ( i = 0; ; i = 1 )
  {
    while ( 1 )
    {
      v9 = *(_BYTE *)(a1 + 140);
      if ( v9 != 12 )
        break;
      a1 = *(_QWORD *)(a1 + 160);
    }
    for ( j = *(_BYTE *)(a2 + 140); j == 12; j = *(_BYTE *)(a2 + 140) )
      a2 = *(_QWORD *)(a2 + 160);
    if ( j != v9 )
      break;
    if ( j == 6 )
    {
      if ( (*(_BYTE *)(a1 + 168) & 1) != 0 || (*(_BYTE *)(a2 + 168) & 1) != 0 )
        break;
LABEL_17:
      a1 = *(_QWORD *)(a1 + 160);
      a2 = *(_QWORD *)(a2 + 160);
      continue;
    }
    if ( j != 13 )
    {
      if ( j != 8 )
        break;
      if ( !(unsigned int)sub_8D1590(a1, a2)
        && ((*(_BYTE *)(a1 + 169) & 0x20) != 0 || (*(_WORD *)(a1 + 168) & 0x180) != 0 || *(_QWORD *)(a1 + 176))
        && ((*(_BYTE *)(a2 + 169) & 0x20) != 0 || (*(_WORD *)(a2 + 168) & 0x180) != 0 || *(_QWORD *)(a2 + 176)) )
      {
LABEL_19:
        i = 0;
        break;
      }
      goto LABEL_17;
    }
    v12 = sub_8D4890(a1);
    v14 = sub_8D4890(a2);
    if ( v12 != v14 && !(unsigned int)sub_8D97D0(v12, v14, 0, v13, v15) )
      goto LABEL_19;
    a1 = sub_8D4870(a1);
    a2 = sub_8D4870(a2);
  }
  *a3 = a1;
  *a4 = a2;
  return i;
}
