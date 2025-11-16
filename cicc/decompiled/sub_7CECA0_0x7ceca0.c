// Function: sub_7CECA0
// Address: 0x7ceca0
//
_QWORD *__fastcall sub_7CECA0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rdi

  v2 = *(_BYTE *)(a1 + 89);
  if ( (v2 & 1) != 0 )
  {
    v6 = sub_72B7F0(a1);
    v7 = (*(_BYTE *)(v6 + 89) & 4) == 0;
    v8 = *(_QWORD *)(v6 + 40);
    if ( !v7 )
    {
      v9 = *(_QWORD *)(v8 + 32);
      if ( !v9 )
      {
LABEL_11:
        v4 = 0;
        return sub_7CEBB0(v4, a2);
      }
      for ( ; (*(_BYTE *)(v9 + 89) & 4) != 0; v9 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 32LL) )
        ;
      v8 = *(_QWORD *)(v9 + 40);
    }
    if ( v8 && *(_BYTE *)(v8 + 28) == 3 )
    {
      v4 = *(_QWORD *)(v8 + 32);
      return sub_7CEBB0(v4, a2);
    }
    goto LABEL_11;
  }
  if ( (v2 & 4) != 0 )
  {
    do
      a1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    while ( (*(_BYTE *)(a1 + 89) & 4) != 0 );
  }
  v3 = *(_QWORD *)(a1 + 40);
  if ( v3 && *(_BYTE *)(v3 + 28) == 3 )
    v4 = *(_QWORD *)(v3 + 32);
  else
    v4 = 0;
  return sub_7CEBB0(v4, a2);
}
