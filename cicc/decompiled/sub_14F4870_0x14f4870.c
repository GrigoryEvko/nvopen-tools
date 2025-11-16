// Function: sub_14F4870
// Address: 0x14f4870
//
__int64 __fastcall sub_14F4870(char **a1, char *a2, _QWORD *a3)
{
  char *v3; // r15
  char *v4; // r14
  __int64 v5; // rax
  bool v7; // zf
  __int64 v9; // rsi
  __int64 v10; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  char *v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r8
  char *v17; // rdx
  _QWORD *v18; // rdx
  char *v19; // rax
  void *v20; // rdi
  __int64 v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-48h]
  _QWORD *v25; // [rsp+10h] [rbp-40h]
  _QWORD *v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (v3 - *a1) >> 3;
  if ( v5 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = v5 == 0;
  v9 = (a1[1] - *a1) >> 3;
  v10 = 1;
  if ( !v7 )
    v10 = (a1[1] - *a1) >> 3;
  v12 = __CFADD__(v9, v10);
  v13 = v9 + v10;
  v14 = (char *)(a2 - v4);
  if ( v12 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v13 )
    {
      v27 = 0;
      v15 = 8;
      v16 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0xFFFFFFFFFFFFFFFLL )
      v13 = 0xFFFFFFFFFFFFFFFLL;
    v22 = 8 * v13;
  }
  v24 = a3;
  v23 = sub_22077B0(v22);
  v14 = (char *)(a2 - v4);
  a3 = v24;
  v16 = v23;
  v27 = v22 + v23;
  v15 = v23 + 8;
LABEL_7:
  v17 = &v14[v16];
  if ( v17 )
    *(_QWORD *)v17 = *a3;
  if ( a2 != v4 )
  {
    v18 = (_QWORD *)v16;
    v19 = v4;
    do
    {
      if ( v18 )
        *v18 = *(_QWORD *)v19;
      v19 += 8;
      ++v18;
    }
    while ( v19 != a2 );
    v15 = v16 + a2 - v4 + 8;
  }
  if ( a2 != v3 )
  {
    v20 = (void *)v15;
    v25 = (_QWORD *)v16;
    v15 += v3 - a2;
    memcpy(v20, a2, v3 - a2);
    v16 = (__int64)v25;
  }
  if ( v4 )
  {
    v26 = (_QWORD *)v16;
    j_j___libc_free_0(v4, a1[2] - v4);
    v16 = (__int64)v26;
  }
  *a1 = (char *)v16;
  a1[1] = (char *)v15;
  a1[2] = (char *)v27;
  return v27;
}
