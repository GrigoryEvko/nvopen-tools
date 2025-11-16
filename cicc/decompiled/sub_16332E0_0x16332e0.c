// Function: sub_16332E0
// Address: 0x16332e0
//
void __fastcall sub_16332E0(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  __int64 v4; // rdi
  _QWORD *i; // r12
  __int64 v6; // rdi
  _QWORD *j; // r8
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  _QWORD *k; // r8
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx

  v1 = a1 + 3;
  v2 = (_QWORD *)a1[4];
  if ( v2 != a1 + 3 )
  {
    do
    {
      v4 = (__int64)(v2 - 7);
      if ( !v2 )
        v4 = 0;
      sub_15E0C30(v4);
      v2 = (_QWORD *)v2[1];
    }
    while ( v1 != v2 );
  }
  for ( i = (_QWORD *)a1[2]; a1 + 1 != i; i = (_QWORD *)i[1] )
  {
    v6 = (__int64)(i - 7);
    if ( !i )
      v6 = 0;
    sub_15E5530(v6);
  }
  for ( j = (_QWORD *)a1[6]; a1 + 5 != j; j = (_QWORD *)j[1] )
  {
    if ( !j )
      BUG();
    v8 = 24LL * (*((_DWORD *)j - 7) & 0xFFFFFFF);
    if ( (*((_BYTE *)j - 25) & 0x40) != 0 )
    {
      v9 = (_QWORD *)*(j - 7);
      v10 = &v9[(unsigned __int64)v8 / 8];
    }
    else
    {
      v10 = j - 6;
      v9 = &j[v8 / 0xFFFFFFFFFFFFFFF8LL - 6];
    }
    for ( ; v10 != v9; v9 += 3 )
    {
      if ( *v9 )
      {
        v11 = v9[1];
        v12 = v9[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v12 = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
      }
      *v9 = 0;
    }
  }
  for ( k = (_QWORD *)a1[8]; a1 + 7 != k; k = (_QWORD *)k[1] )
  {
    if ( !k )
      BUG();
    v14 = 24LL * (*((_DWORD *)k - 7) & 0xFFFFFFF);
    if ( (*((_BYTE *)k - 25) & 0x40) != 0 )
    {
      v15 = (_QWORD *)*(k - 7);
      v16 = &v15[(unsigned __int64)v14 / 8];
    }
    else
    {
      v16 = k - 6;
      v15 = &k[v14 / 0xFFFFFFFFFFFFFFF8LL - 6];
    }
    for ( ; v16 != v15; v15 += 3 )
    {
      if ( *v15 )
      {
        v17 = v15[1];
        v18 = v15[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v18 = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
      }
      *v15 = 0;
    }
  }
}
