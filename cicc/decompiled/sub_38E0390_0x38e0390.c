// Function: sub_38E0390
// Address: 0x38e0390
//
unsigned __int64 __fastcall sub_38E0390(unsigned __int64 *a1, unsigned __int64 *a2, __int64 *a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  bool v8; // zf
  __int64 v9; // rcx
  __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // r8
  __int64 v14; // rcx
  char *v15; // r8
  __int64 v16; // rax
  _QWORD *v17; // r15
  unsigned __int64 *v18; // r13
  __int64 v19; // rcx
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // rdi
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 *v25; // [rsp+10h] [rbp-50h]
  unsigned __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27; // [rsp+20h] [rbp-40h]
  unsigned __int64 v28; // [rsp+28h] [rbp-38h]
  __int64 v29; // [rsp+28h] [rbp-38h]
  unsigned __int64 v30; // [rsp+28h] [rbp-38h]

  v4 = *a1;
  v25 = (unsigned __int64 *)a1[1];
  v5 = (__int64)((__int64)v25 - *a1) >> 3;
  if ( v5 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = v5 == 0;
  v9 = (__int64)((__int64)v25 - *a1) >> 3;
  v10 = 1;
  if ( !v8 )
    v10 = (__int64)((__int64)v25 - *a1) >> 3;
  v11 = __CFADD__(v9, v10);
  v12 = v9 + v10;
  v13 = (char *)a2 - v4;
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
  v30 = v23;
  v24 = sub_22077B0(v23);
  v13 = (char *)a2 - v4;
  v27 = v24;
  v14 = v24 + 8;
  v26 = v24 + v30;
LABEL_7:
  v15 = &v13[v27];
  if ( v15 )
  {
    v16 = *a3;
    *a3 = 0;
    *(_QWORD *)v15 = v16;
  }
  if ( a2 != (unsigned __int64 *)v4 )
  {
    v17 = (_QWORD *)v27;
    v18 = (unsigned __int64 *)v4;
    while ( 1 )
    {
      v20 = *v18;
      if ( v17 )
        break;
      if ( !v20 )
        goto LABEL_12;
      v21 = *(_QWORD *)(v20 + 72);
      if ( v21 )
      {
        v28 = *v18;
        j_j___libc_free_0(v21);
        v20 = v28;
      }
      ++v18;
      j_j___libc_free_0(v20);
      v19 = 8;
      if ( v18 == a2 )
      {
LABEL_19:
        v14 = (__int64)(v17 + 2);
        goto LABEL_20;
      }
LABEL_13:
      v17 = (_QWORD *)v19;
    }
    *v17 = v20;
    *v18 = 0;
LABEL_12:
    ++v18;
    v19 = (__int64)(v17 + 1);
    if ( v18 == a2 )
      goto LABEL_19;
    goto LABEL_13;
  }
LABEL_20:
  if ( a2 != v25 )
    v14 = (__int64)memcpy((void *)v14, a2, (char *)v25 - (char *)a2) + (char *)v25 - (char *)a2;
  if ( v4 )
  {
    v29 = v14;
    j_j___libc_free_0(v4);
    v14 = v29;
  }
  a1[1] = v14;
  *a1 = v27;
  a1[2] = v26;
  return v26;
}
