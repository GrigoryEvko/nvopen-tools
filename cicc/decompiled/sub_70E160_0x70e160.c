// Function: sub_70E160
// Address: 0x70e160
//
__int64 __fastcall sub_70E160(__int64 a1, int a2)
{
  __int64 v2; // rbx
  char i; // al
  unsigned int v5; // r12d
  __int64 v6; // r12
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rbx
  int v10; // eax

  v2 = a1;
  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  if ( i != 2 )
  {
    if ( i != 8 )
      goto LABEL_5;
    return 0;
  }
  if ( (*(_BYTE *)(v2 + 161) & 8) != 0 )
    return 0;
LABEL_5:
  if ( a2 && !(unsigned int)sub_70DD40(a1, 0, 0) )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)(v2 + 140) - 9) > 2u )
    return 1;
  if ( !(unsigned int)sub_5F0640(v2)
    || (unsigned int)sub_8D3E60(v2)
    || (*(_BYTE *)(v2 + 176) & 0x10) != 0
    || !unk_4D041C0 )
  {
    return 0;
  }
  v6 = sub_72FD90(*(_QWORD *)(v2 + 160), 7);
  if ( v6 )
  {
    v7 = 1;
    do
    {
      v8 = sub_8D4130(*(_QWORD *)(v6 + 120));
      if ( !(unsigned int)sub_70E160(v8, 0) )
        v7 = 0;
      v6 = sub_72FD90(*(_QWORD *)(v6 + 112), 7);
    }
    while ( v6 );
    if ( !v7 )
      return 0;
  }
  v5 = 1;
  v9 = *(_QWORD *)(*(_QWORD *)(v2 + 168) + 8LL);
  if ( !v9 )
    return 1;
  do
  {
    v10 = sub_70E160(*(_QWORD *)(v9 + 40), 0);
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v10 )
      v5 = 0;
  }
  while ( v9 );
  return v5;
}
