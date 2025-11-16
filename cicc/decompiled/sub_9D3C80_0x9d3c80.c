// Function: sub_9D3C80
// Address: 0x9d3c80
//
__int64 __fastcall sub_9D3C80(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  char *v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // rdi
  int v15; // ecx
  __int64 v16; // rbx
  char *v17; // rax
  char v18; // si
  char v19; // al
  __int64 v20; // rdi
  char *i; // r14
  char *v22; // rdi
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  char *v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = (char *)*a1;
  v6 = 0x8E38E38E38E38E39LL * ((v4 - *a1) >> 3);
  if ( v6 == 0x1C71C71C71C71C7LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v6 )
    v7 = 0x8E38E38E38E38E39LL * ((v4 - (__int64)v5) >> 3);
  v9 = a2;
  v10 = __CFADD__(v7, v6);
  v11 = v7 - 0x71C71C71C71C71C7LL * ((v4 - (__int64)v5) >> 3);
  v12 = a2 - (_QWORD)v5;
  if ( v10 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v26 = a3;
    v25 = sub_22077B0(v24);
    v12 = a2 - (_QWORD)v5;
    v8 = a2;
    v31 = v25;
    a3 = v26;
    v27 = v25 + v24;
    v13 = v25 + 72;
    goto LABEL_7;
  }
  if ( v11 )
  {
    if ( v11 > 0x1C71C71C71C71C7LL )
      v11 = 0x1C71C71C71C71C7LL;
    v24 = 72 * v11;
    goto LABEL_31;
  }
  v27 = 0;
  v13 = 72;
  v31 = 0;
LABEL_7:
  v14 = v31 + v12;
  if ( v31 + v12 )
  {
    v15 = *(_DWORD *)(a3 + 16);
    *(_BYTE *)v14 = *(_BYTE *)a3;
    *(_QWORD *)(v14 + 8) = v14 + 24;
    *(_QWORD *)(v14 + 16) = 0xC00000000LL;
    if ( v15 )
    {
      a2 = a3 + 8;
      v30 = v8;
      sub_9C31C0(v14 + 8, (char **)(a3 + 8));
      v8 = v30;
    }
  }
  if ( (char *)v8 != v5 )
  {
    v16 = v31;
    v17 = v5;
    while ( 1 )
    {
      if ( v16
        && (v18 = *v17,
            *(_DWORD *)(v16 + 16) = 0,
            *(_DWORD *)(v16 + 20) = 12,
            *(_BYTE *)v16 = v18,
            *(_QWORD *)(v16 + 8) = v16 + 24,
            *((_DWORD *)v17 + 4)) )
      {
        v28 = v8;
        v29 = v17;
        sub_9C2E20(v16 + 8, (__int64)(v17 + 8));
        v8 = v28;
        a2 = v16 + 72;
        v17 = v29 + 72;
        if ( (char *)v28 == v29 + 72 )
        {
LABEL_17:
          v13 = v16 + 144;
          break;
        }
      }
      else
      {
        v17 += 72;
        a2 = v16 + 72;
        if ( (char *)v8 == v17 )
          goto LABEL_17;
      }
      v16 = a2;
    }
  }
  if ( v8 != v4 )
  {
    do
    {
      while ( 1 )
      {
        v19 = *(_BYTE *)v9;
        *(_DWORD *)(v13 + 16) = 0;
        *(_DWORD *)(v13 + 20) = 12;
        *(_BYTE *)v13 = v19;
        *(_QWORD *)(v13 + 8) = v13 + 24;
        if ( *(_DWORD *)(v9 + 16) )
          break;
        v9 += 72;
        v13 += 72;
        if ( v4 == v9 )
          goto LABEL_23;
      }
      a2 = v9 + 8;
      v20 = v13 + 8;
      v9 += 72;
      v13 += 72;
      sub_9C2E20(v20, a2);
    }
    while ( v4 != v9 );
  }
LABEL_23:
  for ( i = v5; (char *)v4 != i; i += 72 )
  {
    v22 = (char *)*((_QWORD *)i + 1);
    if ( v22 != i + 24 )
      _libc_free(v22, a2);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - (_QWORD)v5);
  a1[1] = v13;
  *a1 = v31;
  a1[2] = v27;
  return v27;
}
