// Function: sub_A16110
// Address: 0xa16110
//
__int64 __fastcall sub_A16110(__int64 **a1, _QWORD *a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // r9
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx

  if ( (*a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v3 = *(_QWORD *)(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  v4 = **a1;
  v5 = *(_QWORD **)(v4 + 56);
  v6 = (_QWORD *)(v4 + 48);
  if ( !v5 )
    return 0;
  v7 = (_QWORD *)(v4 + 48);
  do
  {
    while ( 1 )
    {
      v8 = v5[2];
      v9 = v5[3];
      if ( v3 <= v5[4] )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v9 )
        goto LABEL_8;
    }
    v7 = v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v8 );
LABEL_8:
  if ( v6 != v7 && v3 >= v7[4] )
    return *((unsigned int *)v7 + 10);
  else
    return 0;
}
