// Function: sub_2851E20
// Address: 0x2851e20
//
char __fastcall sub_2851E20(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // eax

  v4 = sub_C52410();
  v5 = v4 + 1;
  v6 = sub_C959E0();
  v7 = (_QWORD *)v4[2];
  if ( v7 )
  {
    v8 = v4 + 1;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= v8[4] )
      v5 = v8;
  }
  if ( v5 == (_QWORD *)((char *)sub_C52410() + 8) )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  v11 = v5[7];
  if ( !v11 )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  v12 = v5 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= dword_5001788 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_15;
    }
    v12 = (_QWORD *)v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_15:
  if ( v5 + 6 == v12 )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  if ( dword_5001788 < *((_DWORD *)v12 + 8) )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  if ( *((int *)v12 + 9) <= 0 )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  if ( !byte_5001808 )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  v15 = *(_DWORD *)(a2 + 24);
  if ( *(_DWORD *)(a1 + 24) == v15 )
    return sub_DFA190(*(_QWORD *)(a1 + 16), (_DWORD *)(a1 + 24), (_DWORD *)(a2 + 24));
  else
    return *(_DWORD *)(a1 + 24) < v15;
}
