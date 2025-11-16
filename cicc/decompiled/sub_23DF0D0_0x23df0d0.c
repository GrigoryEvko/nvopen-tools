// Function: sub_23DF0D0
// Address: 0x23df0d0
//
__int64 __fastcall sub_23DF0D0(int *a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  int v10; // esi
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 result; // rax

  v2 = sub_C52410();
  v3 = v2 + 1;
  v4 = sub_C959E0();
  v5 = (_QWORD *)v2[2];
  if ( v5 )
  {
    v6 = v2 + 1;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v4 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v6 && v4 >= v6[4] )
      v3 = v6;
  }
  if ( v3 == (_QWORD *)((char *)sub_C52410() + 8) )
    return 0;
  v9 = v3[7];
  if ( !v9 )
    return 0;
  v10 = *a1;
  v11 = v3 + 6;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v9 + 16);
      v13 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= v10 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v13 )
        goto LABEL_15;
    }
    v11 = (_QWORD *)v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v12 );
LABEL_15:
  result = 0;
  if ( v3 + 6 != v11 && v10 >= *((_DWORD *)v11 + 8) )
    return *((unsigned int *)v11 + 9);
  return result;
}
