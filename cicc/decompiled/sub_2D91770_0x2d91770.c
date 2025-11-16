// Function: sub_2D91770
// Address: 0x2d91770
//
__int16 sub_2D91770()
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
  __int16 result; // ax

  v0 = qword_501CD30;
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
    return 0;
  v8 = v2[7];
  if ( !v8 )
    return 0;
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
    return 0;
  LOBYTE(result) = *(_BYTE *)(qword_501CD30 + 136);
  HIBYTE(result) = 1;
  return result;
}
