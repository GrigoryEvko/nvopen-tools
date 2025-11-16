// Function: sub_2C83610
// Address: 0x2c83610
//
unsigned __int64 __fastcall sub_2C83610(unsigned __int64 *a1, unsigned __int64 *a2, __int64 *a3)
{
  unsigned __int64 v3; // r14
  __int64 v4; // rax
  __int64 v7; // rcx
  bool v8; // zf
  __int64 v9; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // rdx
  __int64 v14; // rbx
  char *v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rbx
  unsigned __int64 *v18; // r15
  __int64 v19; // rcx
  unsigned __int64 v20; // rdi
  void *v21; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 *v25; // [rsp+18h] [rbp-48h]
  unsigned __int64 v26; // [rsp+20h] [rbp-40h]
  unsigned __int64 v27; // [rsp+28h] [rbp-38h]

  v3 = *a1;
  v25 = (unsigned __int64 *)a1[1];
  v4 = (__int64)((__int64)v25 - *a1) >> 3;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = (__int64)((__int64)v25 - v3) >> 3;
  v8 = v4 == 0;
  v9 = 1;
  if ( !v8 )
    v9 = (__int64)((__int64)v25 - v3) >> 3;
  v11 = __CFADD__(v7, v9);
  v12 = v7 + v9;
  v13 = (char *)a2 - v3;
  if ( v11 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v26 = 0;
      v14 = 8;
      v27 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0xFFFFFFFFFFFFFFFLL )
      v12 = 0xFFFFFFFFFFFFFFFLL;
    v23 = 8 * v12;
  }
  v24 = sub_22077B0(v23);
  v13 = (char *)a2 - v3;
  v27 = v24;
  v26 = v24 + v23;
  v14 = v24 + 8;
LABEL_7:
  v15 = &v13[v27];
  if ( v15 )
  {
    v16 = *a3;
    *a3 = 0;
    *(_QWORD *)v15 = v16;
  }
  if ( a2 != (unsigned __int64 *)v3 )
  {
    v17 = (_QWORD *)v27;
    v18 = (unsigned __int64 *)v3;
    while ( 1 )
    {
      v20 = *v18;
      if ( v17 )
        break;
      if ( !v20 )
        goto LABEL_12;
      ++v18;
      j_j___libc_free_0(v20);
      v19 = 8;
      if ( a2 == v18 )
      {
LABEL_17:
        v14 = (__int64)(v17 + 2);
        goto LABEL_18;
      }
LABEL_13:
      v17 = (_QWORD *)v19;
    }
    *v17 = v20;
    *v18 = 0;
LABEL_12:
    ++v18;
    v19 = (__int64)(v17 + 1);
    if ( a2 == v18 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v25 )
  {
    v21 = (void *)v14;
    v14 += (char *)v25 - (char *)a2;
    memcpy(v21, a2, (char *)v25 - (char *)a2);
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  a1[1] = v14;
  *a1 = v27;
  a1[2] = v26;
  return v26;
}
