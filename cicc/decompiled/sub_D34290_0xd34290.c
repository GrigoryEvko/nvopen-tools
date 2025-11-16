// Function: sub_D34290
// Address: 0xd34290
//
bool sub_D34290()
{
  _QWORD *v0; // r12
  _QWORD *v1; // rbx
  unsigned __int64 v2; // rsi
  _QWORD *v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  bool result; // al

  v0 = sub_C52410();
  v1 = v0 + 1;
  v2 = sub_C959E0();
  v3 = (_QWORD *)v0[2];
  if ( v3 )
  {
    v4 = v0 + 1;
    do
    {
      while ( 1 )
      {
        v5 = v3[2];
        v6 = v3[3];
        if ( v2 <= v3[4] )
          break;
        v3 = (_QWORD *)v3[3];
        if ( !v6 )
          goto LABEL_6;
      }
      v4 = v3;
      v3 = (_QWORD *)v3[2];
    }
    while ( v5 );
LABEL_6:
    if ( v1 != v4 && v2 >= v4[4] )
      v1 = v4;
  }
  if ( v1 == (_QWORD *)((char *)sub_C52410() + 8) )
    return 0;
  v7 = v1[7];
  if ( !v7 )
    return 0;
  v8 = v1 + 6;
  do
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v7 + 16);
      v10 = *(_QWORD *)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) >= dword_4F87448 )
        break;
      v7 = *(_QWORD *)(v7 + 24);
      if ( !v10 )
        goto LABEL_15;
    }
    v8 = (_QWORD *)v7;
    v7 = *(_QWORD *)(v7 + 16);
  }
  while ( v9 );
LABEL_15:
  result = 0;
  if ( v1 + 6 != v8 && dword_4F87448 >= *((_DWORD *)v8 + 8) )
    return *((_DWORD *)v8 + 9) > 0;
  return result;
}
