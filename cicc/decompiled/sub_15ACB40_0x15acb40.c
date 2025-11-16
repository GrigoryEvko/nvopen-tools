// Function: sub_15ACB40
// Address: 0x15acb40
//
__int64 __fastcall sub_15ACB40(_QWORD *a1)
{
  _QWORD *v2; // r13
  unsigned int v3; // r12d
  _QWORD *v4; // rbx
  _QWORD *v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  _QWORD *i; // r13
  __int64 v12; // rdi
  int v13; // eax
  _QWORD *j; // rbx
  _QWORD *v15; // rdi
  int v16; // eax
  __int64 v17; // rdi

  v2 = a1 + 9;
  v3 = 0;
  v4 = (_QWORD *)a1[10];
  if ( a1 + 9 != v4 )
  {
    do
    {
      while ( 1 )
      {
        v5 = v4;
        v4 = (_QWORD *)v4[1];
        v6 = sub_161F640(v5);
        if ( v7 > 8 && *(_QWORD *)v6 == 0x6762642E6D766C6CLL && *(_BYTE *)(v6 + 8) == 46 )
          break;
        v8 = sub_161F640(v5);
        if ( v9 == 9 && *(_QWORD *)v8 == 0x6F63672E6D766C6CLL && *(_BYTE *)(v8 + 8) == 118 )
          break;
        if ( v2 == v4 )
          goto LABEL_10;
      }
      v3 = 1;
      sub_161F540(v5);
    }
    while ( v2 != v4 );
  }
LABEL_10:
  v10 = (_QWORD *)a1[4];
  for ( i = a1 + 3; i != v10; v3 |= v13 )
  {
    v12 = (__int64)(v10 - 7);
    if ( !v10 )
      v12 = 0;
    v13 = sub_15AC2E0(v12);
    v10 = (_QWORD *)v10[1];
  }
  for ( j = (_QWORD *)a1[2]; a1 + 1 != j; v3 |= v16 )
  {
    v15 = j - 7;
    if ( !j )
      v15 = 0;
    v16 = sub_1626EF0(v15, 0);
    j = (_QWORD *)j[1];
  }
  v17 = a1[21];
  if ( v17 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL))(v17);
  return v3;
}
