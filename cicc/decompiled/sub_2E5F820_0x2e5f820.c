// Function: sub_2E5F820
// Address: 0x2e5f820
//
unsigned __int64 *__fastcall sub_2E5F820(unsigned __int64 *a1, unsigned __int64 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  bool v6; // zf
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rdx
  __int64 v11; // r13
  char *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 *v14; // r12
  _QWORD *i; // r13
  __int64 v16; // rax
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // rdi
  void *v23; // rdi
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 *v27; // [rsp+10h] [rbp-60h]
  unsigned __int64 v28; // [rsp+18h] [rbp-58h]
  unsigned __int64 v30; // [rsp+28h] [rbp-48h]
  unsigned __int64 v31; // [rsp+30h] [rbp-40h]

  v27 = (unsigned __int64 *)a1[1];
  v4 = (__int64)((__int64)v27 - *a1) >> 3;
  v30 = *a1;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = (__int64)((__int64)v27 - *a1) >> 3;
  v6 = v4 == 0;
  v7 = 1;
  if ( !v6 )
    v7 = (__int64)((__int64)v27 - *a1) >> 3;
  v8 = __CFADD__(v5, v7);
  v9 = v5 + v7;
  v10 = (char *)a2 - v30;
  if ( v8 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v28 = 0;
      v11 = 8;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      v9 = 0xFFFFFFFFFFFFFFFLL;
    v25 = 8 * v9;
  }
  v26 = sub_22077B0(v25);
  v10 = (char *)a2 - v30;
  v31 = v26;
  v11 = v26 + 8;
  v28 = v26 + v25;
LABEL_7:
  v12 = &v10[v31];
  if ( v12 )
  {
    v13 = *a3;
    *a3 = 0;
    *(_QWORD *)v12 = v13;
  }
  v14 = (unsigned __int64 *)v30;
  if ( a2 != (unsigned __int64 *)v30 )
  {
    for ( i = (_QWORD *)v31; ; i = (_QWORD *)v16 )
    {
      v17 = *v14;
      if ( i )
        break;
      if ( !v17 )
        goto LABEL_12;
      v18 = *(_QWORD *)(v17 + 176);
      if ( v18 != v17 + 192 )
        _libc_free(v18);
      v19 = *(_QWORD *)(v17 + 88);
      if ( v19 != v17 + 104 )
        _libc_free(v19);
      sub_C7D6A0(*(_QWORD *)(v17 + 64), 8LL * *(unsigned int *)(v17 + 80), 8);
      v20 = *(unsigned __int64 **)(v17 + 40);
      v21 = *(unsigned __int64 **)(v17 + 32);
      if ( v20 != v21 )
      {
        do
        {
          if ( *v21 )
            sub_2E5DCD0(*v21);
          ++v21;
        }
        while ( v20 != v21 );
        v21 = *(unsigned __int64 **)(v17 + 32);
      }
      if ( v21 )
        j_j___libc_free_0((unsigned __int64)v21);
      v22 = *(_QWORD *)(v17 + 8);
      if ( v22 != v17 + 24 )
        _libc_free(v22);
      ++v14;
      j_j___libc_free_0(v17);
      v16 = 8;
      if ( v14 == a2 )
      {
LABEL_30:
        v11 = (__int64)(i + 2);
        goto LABEL_31;
      }
LABEL_13:
      ;
    }
    *i = v17;
    *v14 = 0;
LABEL_12:
    ++v14;
    v16 = (__int64)(i + 1);
    if ( v14 == a2 )
      goto LABEL_30;
    goto LABEL_13;
  }
LABEL_31:
  if ( a2 != v27 )
  {
    v23 = (void *)v11;
    v11 += (char *)v27 - (char *)a2;
    memcpy(v23, a2, (char *)v27 - (char *)a2);
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  *a1 = v31;
  a1[1] = v11;
  a1[2] = v28;
  return a1;
}
