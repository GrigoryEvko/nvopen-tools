// Function: sub_22B1EE0
// Address: 0x22b1ee0
//
unsigned __int64 __fastcall sub_22B1EE0(unsigned __int64 *a1, int *a2, __int64 a3, __int64 a4)
{
  int *v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // r8
  int *v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rbx
  _DWORD *v17; // rax
  unsigned __int64 v18; // rsi
  int v19; // esi
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdi
  unsigned __int64 i; // r14
  unsigned __int64 v24; // rdi
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  _DWORD *v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  unsigned __int64 v33; // [rsp+28h] [rbp-38h]

  v5 = (int *)a1[1];
  v6 = *a1;
  v7 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v5 - *a1) >> 3);
  if ( v7 == 0x1C71C71C71C71C7LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (__int64)a2;
  if ( v7 )
    v8 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v5 - v6) >> 3);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x71C71C71C71C71C7LL * ((__int64)((__int64)v5 - v6) >> 3);
  v13 = (__int64)a2 - v6;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v28 = a3;
    v27 = sub_22077B0(v26);
    v13 = (__int64)a2 - v6;
    v9 = (__int64)a2;
    v33 = v27;
    a3 = v28;
    v29 = v27 + v26;
    v14 = v27 + 72;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x1C71C71C71C71C7LL )
      v12 = 0x1C71C71C71C71C7LL;
    v26 = 72 * v12;
    goto LABEL_31;
  }
  v29 = 0;
  v14 = 72;
  v33 = 0;
LABEL_7:
  v15 = v33 + v13;
  if ( v33 + v13 )
  {
    a4 = *(unsigned int *)(a3 + 16);
    *(_DWORD *)v15 = *(_DWORD *)a3;
    *(_QWORD *)(v15 + 8) = v15 + 24;
    *(_QWORD *)(v15 + 16) = 0xC00000000LL;
    if ( (_DWORD)a4 )
    {
      v32 = v9;
      sub_22AD3C0(v15 + 8, a3 + 8, a3, a4, v9, v13);
      v9 = v32;
    }
  }
  if ( v9 != v6 )
  {
    v16 = v33;
    v17 = (_DWORD *)v6;
    while ( 1 )
    {
      if ( v16
        && (v19 = *v17,
            *(_DWORD *)(v16 + 16) = 0,
            *(_DWORD *)(v16 + 20) = 12,
            *(_DWORD *)v16 = v19,
            *(_QWORD *)(v16 + 8) = v16 + 24,
            a3 = (unsigned int)v17[4],
            (_DWORD)a3) )
      {
        v30 = v9;
        v31 = v17;
        sub_22AD3C0(v16 + 8, (__int64)(v17 + 2), a3, a4, v9, v13);
        v9 = v30;
        v18 = v16 + 72;
        v17 = v31 + 18;
        if ( (_DWORD *)v30 == v31 + 18 )
        {
LABEL_17:
          v14 = v16 + 144;
          break;
        }
      }
      else
      {
        v17 += 18;
        v18 = v16 + 72;
        if ( (_DWORD *)v9 == v17 )
          goto LABEL_17;
      }
      v16 = v18;
    }
  }
  if ( (int *)v9 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v20 = *v10;
        *(_DWORD *)(v14 + 16) = 0;
        *(_DWORD *)(v14 + 20) = 12;
        *(_DWORD *)v14 = v20;
        *(_QWORD *)(v14 + 8) = v14 + 24;
        if ( v10[4] )
          break;
        v10 += 18;
        v14 += 72;
        if ( v5 == v10 )
          goto LABEL_23;
      }
      v21 = (__int64)(v10 + 2);
      v22 = v14 + 8;
      v10 += 18;
      v14 += 72;
      sub_22AD3C0(v22, v21, a3, a4, v9, v13);
    }
    while ( v5 != v10 );
  }
LABEL_23:
  for ( i = v6; v5 != (int *)i; i += 72LL )
  {
    v24 = *(_QWORD *)(i + 8);
    if ( v24 != i + 24 )
      _libc_free(v24);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  a1[1] = v14;
  *a1 = v33;
  a1[2] = v29;
  return v29;
}
