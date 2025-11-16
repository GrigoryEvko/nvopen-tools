// Function: sub_1249720
// Address: 0x1249720
//
void __fastcall sub_1249720(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r15
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  bool v15; // cf
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r15
  char v19; // al
  __int64 v20; // rdi
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // [rsp-50h] [rbp-50h]
  __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v6 = a2;
  v7 = a1[1];
  v8 = *a1;
  v9 = v7 - *a1;
  v10 = 0x8E38E38E38E38E39LL * (v9 >> 3);
  v11 = 0x1C71C71C71C71C7LL - v10;
  if ( a2 <= 0x8E38E38E38E38E39LL * ((a1[2] - v7) >> 3) )
  {
    v12 = a2;
    v13 = a1[1];
    do
    {
      if ( v13 )
      {
        *(_BYTE *)v13 = 0;
        *(_QWORD *)(v13 + 8) = v13 + 32;
        *(_QWORD *)(v13 + 16) = 0;
        *(_QWORD *)(v13 + 24) = 40;
      }
      v13 += 72;
      --v12;
    }
    while ( v12 );
    a1[1] = v7 + 72 * a2;
    return;
  }
  if ( v11 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v14 = 0x8E38E38E38E38E39LL * ((v7 - v8) >> 3);
  if ( a2 >= v10 )
    v14 = a2;
  v15 = __CFADD__(v10, v14);
  v16 = v10 + v14;
  if ( v15 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v16 )
    {
      v25 = 0;
      v28 = 0;
      goto LABEL_15;
    }
    if ( v16 > 0x1C71C71C71C71C7LL )
      v16 = 0x1C71C71C71C71C7LL;
    v23 = 72 * v16;
  }
  v27 = v23;
  v24 = sub_22077B0(v23);
  v7 = a1[1];
  v28 = v24;
  v8 = *a1;
  a4 = v24 + v27;
  v25 = v24 + v27;
LABEL_15:
  v17 = v9 + v28;
  do
  {
    if ( v17 )
    {
      *(_BYTE *)v17 = 0;
      *(_QWORD *)(v17 + 8) = v17 + 32;
      *(_QWORD *)(v17 + 16) = 0;
      *(_QWORD *)(v17 + 24) = 40;
    }
    v17 += 72;
    --a2;
  }
  while ( a2 );
  if ( v8 != v7 )
  {
    v18 = v28;
    do
    {
      while ( 1 )
      {
        if ( v18 )
        {
          v19 = *(_BYTE *)v8;
          *(_QWORD *)(v18 + 16) = 0;
          *(_QWORD *)(v18 + 24) = 40;
          *(_BYTE *)v18 = v19;
          *(_QWORD *)(v18 + 8) = v18 + 32;
          if ( *(_QWORD *)(v8 + 16) )
            break;
        }
        v8 += 72;
        v18 += 72;
        if ( v8 == v7 )
          goto LABEL_25;
      }
      a2 = v8 + 8;
      v20 = v18 + 8;
      v26 = v8;
      v18 += 72;
      sub_12495E0(v20, v8 + 8, v8, a4, a5, v11);
      v8 = v26 + 72;
    }
    while ( v26 + 72 != v7 );
LABEL_25:
    v21 = a1[1];
    v7 = *a1;
    if ( v21 != *a1 )
    {
      do
      {
        v22 = *(_QWORD *)(v7 + 8);
        if ( v22 != v7 + 32 )
          _libc_free(v22, a2);
        v7 += 72;
      }
      while ( v21 != v7 );
      v7 = *a1;
    }
  }
  if ( v7 )
    j_j___libc_free_0(v7, a1[2] - v7);
  *a1 = v28;
  a1[1] = v28 + 72 * (v6 + v10);
  a1[2] = v25;
}
