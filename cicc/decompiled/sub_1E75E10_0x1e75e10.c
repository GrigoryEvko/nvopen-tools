// Function: sub_1E75E10
// Address: 0x1e75e10
//
void __fastcall sub_1E75E10(char **a1, unsigned __int64 a2)
{
  char *v4; // rdi
  char *v5; // rsi
  char *v6; // rdx
  __int64 v7; // rbx
  unsigned __int64 v8; // r14
  char *v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // r8
  _DWORD *v16; // rax
  unsigned __int64 v17; // rcx
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rax
  _QWORD *v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = *a1;
  v6 = *a1;
  v7 = v4 - *a1;
  v8 = v7 >> 3;
  if ( (a1[2] - v4) >> 3 >= a2 )
  {
    v9 = v4;
    v10 = a2;
    do
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = 0;
        *((_DWORD *)v9 + 1) = -1;
      }
      v9 += 8;
      --v10;
    }
    while ( v10 );
    a1[1] = &v4[8 * a2];
    return;
  }
  if ( 0xFFFFFFFFFFFFFFFLL - v8 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v11 = a2;
  if ( v8 >= a2 )
    v11 = (v4 - *a1) >> 3;
  v12 = __CFADD__(v8, v11);
  v13 = v8 + v11;
  if ( v12 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v13 )
    {
      v14 = 0;
      v15 = 0;
      goto LABEL_15;
    }
    if ( v13 > 0xFFFFFFFFFFFFFFFLL )
      v13 = 0xFFFFFFFFFFFFFFFLL;
    v20 = 8 * v13;
  }
  v21 = sub_22077B0(v20);
  v5 = *a1;
  v4 = a1[1];
  v15 = (_QWORD *)v21;
  v14 = v21 + v20;
  v6 = *a1;
LABEL_15:
  v16 = (_DWORD *)((char *)v15 + v7);
  v17 = a2;
  do
  {
    if ( v16 )
    {
      *v16 = 0;
      v16[1] = -1;
    }
    v16 += 2;
    --v17;
  }
  while ( v17 );
  if ( v5 != v4 )
  {
    v18 = v15;
    v19 = (_QWORD *)((char *)v15 + v4 - v5);
    do
    {
      if ( v18 )
        *v18 = *(_QWORD *)v6;
      ++v18;
      v6 += 8;
    }
    while ( v18 != v19 );
    v4 = v5;
  }
  if ( v4 )
  {
    v22 = v15;
    j_j___libc_free_0(v4, a1[2] - v4);
    v15 = v22;
  }
  *a1 = (char *)v15;
  a1[2] = (char *)v14;
  a1[1] = (char *)&v15[v8 + a2];
}
