// Function: sub_2D91110
// Address: 0x2d91110
//
__int64 sub_2D91110()
{
  __int64 v0; // rbx
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  unsigned __int64 v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // esi
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v14; // [rsp+8h] [rbp-28h]

  v0 = qword_501CE48;
  v1 = sub_C52410();
  v2 = v1 + 1;
  v3 = sub_C959E0();
  v4 = (_QWORD *)v1[2];
  if ( v4 )
  {
    v5 = v1 + 1;
    do
    {
      while ( 1 )
      {
        v6 = v4[2];
        v7 = v4[3];
        if ( v3 <= v4[4] )
          break;
        v4 = (_QWORD *)v4[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v4;
      v4 = (_QWORD *)v4[2];
    }
    while ( v6 );
LABEL_6:
    if ( v2 != v5 && v3 >= v5[4] )
      v2 = v5;
  }
  if ( v2 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_19;
  v8 = v2[7];
  if ( !v8 )
    goto LABEL_19;
  v9 = *(_DWORD *)(v0 + 8);
  v10 = v2 + 6;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v8 + 16);
      v12 = *(_QWORD *)(v8 + 24);
      if ( *(_DWORD *)(v8 + 32) >= v9 )
        break;
      v8 = *(_QWORD *)(v8 + 24);
      if ( !v12 )
        goto LABEL_15;
    }
    v10 = (_QWORD *)v8;
    v8 = *(_QWORD *)(v8 + 16);
  }
  while ( v11 );
LABEL_15:
  if ( v2 + 6 == v10 || v9 < *((_DWORD *)v10 + 8) || !*((_DWORD *)v10 + 9) )
  {
LABEL_19:
    BYTE4(v14) = 0;
    return v14;
  }
  else
  {
    BYTE4(v14) = 1;
    LODWORD(v14) = *(_DWORD *)(qword_501CE48 + 136);
    return v14;
  }
}
