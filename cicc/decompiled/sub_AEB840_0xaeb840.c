// Function: sub_AEB840
// Address: 0xaeb840
//
__int64 __fastcall sub_AEB840(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  unsigned int v5; // r12d
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *i; // r13
  __int64 v13; // rdi
  int v14; // eax
  _QWORD *j; // rbx
  _QWORD *v16; // rdi
  int v17; // eax
  __int64 v18; // rdi

  v4 = a1 + 9;
  v5 = 0;
  v6 = (_QWORD *)a1[10];
  if ( a1 + 9 != v6 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v6;
        v6 = (_QWORD *)v6[1];
        v8 = sub_B91B20(v7);
        if ( v9 > 8 && *(_QWORD *)v8 == 0x6762642E6D766C6CLL && *(_BYTE *)(v8 + 8) == 46 )
          break;
        v10 = sub_B91B20(v7);
        if ( a3 == 9 && *(_QWORD *)v10 == 0x6F63672E6D766C6CLL && *(_BYTE *)(v10 + 8) == 118 )
          break;
        if ( v4 == v6 )
          goto LABEL_10;
      }
      v5 = 1;
      sub_B91A20(v7);
    }
    while ( v4 != v6 );
  }
LABEL_10:
  v11 = (_QWORD *)a1[4];
  for ( i = a1 + 3; i != v11; v5 |= v14 )
  {
    v13 = (__int64)(v11 - 7);
    if ( !v11 )
      v13 = 0;
    v14 = sub_AEAD90(v13, a2, a3);
    v11 = (_QWORD *)v11[1];
  }
  for ( j = (_QWORD *)a1[2]; a1 + 1 != j; v5 |= v17 )
  {
    v16 = j - 7;
    if ( !j )
      v16 = 0;
    v17 = sub_B98000(v16, 0);
    j = (_QWORD *)j[1];
  }
  v18 = a1[20];
  if ( v18 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 40LL))(v18);
  return v5;
}
