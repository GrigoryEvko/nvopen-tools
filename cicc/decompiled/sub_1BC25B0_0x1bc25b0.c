// Function: sub_1BC25B0
// Address: 0x1bc25b0
//
char **__fastcall sub_1BC25B0(char **a1, char *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  bool v6; // zf
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rdx
  __int64 v12; // rbx
  char *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // r12
  char *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // r14
  unsigned __int64 v21; // rdi
  void *v22; // rdi
  __int64 v24; // rbx
  __int64 v25; // rax
  char *v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  char *v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v26 = a1[1];
  v4 = (v26 - *a1) >> 3;
  v29 = *a1;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = (v26 - *a1) >> 3;
  v6 = v4 == 0;
  v8 = 1;
  if ( !v6 )
    v8 = (v26 - *a1) >> 3;
  v9 = __CFADD__(v5, v8);
  v10 = v5 + v8;
  v11 = (char *)(a2 - v29);
  if ( v9 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v27 = 0;
      v12 = 8;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0xFFFFFFFFFFFFFFFLL )
      v10 = 0xFFFFFFFFFFFFFFFLL;
    v24 = 8 * v10;
  }
  v25 = sub_22077B0(v24);
  v11 = (char *)(a2 - v29);
  v30 = v25;
  v27 = v25 + v24;
  v12 = v25 + 8;
LABEL_7:
  v13 = &v11[v30];
  if ( v13 )
  {
    v14 = *a3;
    *a3 = 0;
    *(_QWORD *)v13 = v14;
  }
  if ( a2 != v29 )
  {
    v15 = (_QWORD *)v30;
    v16 = v29;
    while ( 1 )
    {
      v18 = *(_QWORD *)v16;
      if ( v15 )
        break;
      if ( !v18 )
        goto LABEL_12;
      v19 = 112LL * *(_QWORD *)(v18 - 8);
      v20 = v18 + v19;
      while ( v20 != v18 )
      {
        v20 -= 112;
        v21 = *(_QWORD *)(v20 + 32);
        if ( v21 != v20 + 48 )
          _libc_free(v21);
      }
      v16 += 8;
      j_j_j___libc_free_0_0(v18 - 8);
      v17 = 8;
      if ( a2 == v16 )
      {
LABEL_21:
        v12 = (__int64)(v15 + 2);
        goto LABEL_22;
      }
LABEL_13:
      v15 = (_QWORD *)v17;
    }
    *v15 = v18;
    *(_QWORD *)v16 = 0;
LABEL_12:
    v16 += 8;
    v17 = (__int64)(v15 + 1);
    if ( a2 == v16 )
      goto LABEL_21;
    goto LABEL_13;
  }
LABEL_22:
  if ( a2 != v26 )
  {
    v22 = (void *)v12;
    v12 += v26 - a2;
    memcpy(v22, a2, v26 - a2);
  }
  if ( v29 )
    j_j___libc_free_0(v29, a1[2] - v29);
  *a1 = (char *)v30;
  a1[1] = (char *)v12;
  a1[2] = (char *)v27;
  return a1;
}
