// Function: sub_2647A60
// Address: 0x2647a60
//
unsigned __int64 *__fastcall sub_2647A60(unsigned __int64 *a1, char *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  bool v6; // zf
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rdx
  __int64 v11; // r12
  char *v12; // rdx
  __int64 v13; // rax
  char *v14; // rbx
  _QWORD *i; // r12
  __int64 v16; // rax
  _QWORD *v17; // r15
  unsigned __int64 v18; // rdi
  __int64 v19; // r13
  unsigned __int64 v20; // r14
  volatile signed __int32 *v21; // rdi
  __int64 v22; // r13
  unsigned __int64 v23; // r14
  volatile signed __int32 *v24; // rdi
  unsigned __int64 v25; // rdi
  void *v26; // rdi
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  char *v30; // [rsp+10h] [rbp-60h]
  unsigned __int64 v31; // [rsp+18h] [rbp-58h]
  unsigned __int64 v33; // [rsp+28h] [rbp-48h]
  unsigned __int64 v34; // [rsp+30h] [rbp-40h]

  v30 = (char *)a1[1];
  v4 = (__int64)&v30[-*a1] >> 3;
  v33 = *a1;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = (__int64)&v30[-*a1] >> 3;
  v6 = v4 == 0;
  v7 = 1;
  if ( !v6 )
    v7 = (__int64)&v30[-*a1] >> 3;
  v8 = __CFADD__(v5, v7);
  v9 = v5 + v7;
  v10 = &a2[-v33];
  if ( v8 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v31 = 0;
      v11 = 8;
      v34 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      v9 = 0xFFFFFFFFFFFFFFFLL;
    v28 = 8 * v9;
  }
  v29 = sub_22077B0(v28);
  v10 = &a2[-v33];
  v34 = v29;
  v11 = v29 + 8;
  v31 = v29 + v28;
LABEL_7:
  v12 = &v10[v34];
  if ( v12 )
  {
    v13 = *a3;
    *a3 = 0;
    *(_QWORD *)v12 = v13;
  }
  v14 = (char *)v33;
  if ( a2 != (char *)v33 )
  {
    for ( i = (_QWORD *)v34; ; i = (_QWORD *)v16 )
    {
      v17 = *(_QWORD **)v14;
      if ( i )
        break;
      if ( !v17 )
        goto LABEL_12;
      v18 = v17[12];
      if ( v18 )
        j_j___libc_free_0(v18);
      v19 = v17[10];
      v20 = v17[9];
      if ( v19 != v20 )
      {
        do
        {
          v21 = *(volatile signed __int32 **)(v20 + 8);
          if ( v21 )
            sub_A191D0(v21);
          v20 += 16LL;
        }
        while ( v19 != v20 );
        v20 = v17[9];
      }
      if ( v20 )
        j_j___libc_free_0(v20);
      v22 = v17[7];
      v23 = v17[6];
      if ( v22 != v23 )
      {
        do
        {
          v24 = *(volatile signed __int32 **)(v23 + 8);
          if ( v24 )
            sub_A191D0(v24);
          v23 += 16LL;
        }
        while ( v22 != v23 );
        v23 = v17[6];
      }
      if ( v23 )
        j_j___libc_free_0(v23);
      v25 = v17[3];
      if ( (_QWORD *)v25 != v17 + 5 )
        _libc_free(v25);
      v14 += 8;
      j_j___libc_free_0((unsigned __int64)v17);
      v16 = 8;
      if ( a2 == v14 )
      {
LABEL_35:
        v11 = (__int64)(i + 2);
        goto LABEL_36;
      }
LABEL_13:
      ;
    }
    *i = v17;
    *(_QWORD *)v14 = 0;
LABEL_12:
    v14 += 8;
    v16 = (__int64)(i + 1);
    if ( a2 == v14 )
      goto LABEL_35;
    goto LABEL_13;
  }
LABEL_36:
  if ( a2 != v30 )
  {
    v26 = (void *)v11;
    v11 += v30 - a2;
    memcpy(v26, a2, v30 - a2);
  }
  if ( v33 )
    j_j___libc_free_0(v33);
  *a1 = v34;
  a1[1] = v11;
  a1[2] = v31;
  return a1;
}
