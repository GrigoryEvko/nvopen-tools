// Function: sub_CD1880
// Address: 0xcd1880
//
void __fastcall sub_CD1880(__int64 *a1, char *a2, size_t a3, unsigned __int8 *a4)
{
  _BYTE *v7; // rdi
  unsigned __int8 v8; // r14
  size_t v9; // r15
  size_t v10; // rdx
  char *v11; // r12
  char *v12; // r15
  char *v13; // rax
  size_t v14; // r15
  int v15; // esi
  size_t v16; // rdx
  char *v17; // rdi
  size_t v18; // rdx
  bool v19; // cf
  __int64 v20; // rdi
  __int64 v21; // r14
  char *v22; // rdx
  __int64 v23; // r14
  __int64 v24; // r15
  char *v25; // r8
  size_t v26; // rdx
  char *v27; // r12
  size_t v28; // rdx
  size_t v29; // rax
  __int64 v30; // rax
  unsigned __int8 *v31; // [rsp-48h] [rbp-48h]
  char *v32; // [rsp-48h] [rbp-48h]
  char *v33; // [rsp-40h] [rbp-40h]
  char *v34; // [rsp-40h] [rbp-40h]
  size_t v35; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    return;
  v7 = (_BYTE *)a1[1];
  if ( a1[2] - (__int64)v7 < a3 )
  {
    v17 = &v7[-*a1];
    if ( a3 > 0x7FFFFFFFFFFFFFFFLL - (__int64)v17 )
      sub_4262D8((__int64)"vector::_M_fill_insert");
    v18 = (size_t)v17;
    if ( a3 >= (unsigned __int64)v17 )
      v18 = a3;
    v19 = __CFADD__(v18, v17);
    v20 = (__int64)&v17[v18];
    v21 = v20;
    v22 = &a2[-*a1];
    if ( v19 || v20 < 0 )
    {
      v21 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else if ( !v20 )
    {
      v23 = 0;
      v24 = 0;
      goto LABEL_19;
    }
    v31 = a4;
    v33 = &a2[-*a1];
    v30 = sub_22077B0(v21);
    v22 = v33;
    a4 = v31;
    v24 = v30;
    v23 = v30 + v21;
LABEL_19:
    memset(&v22[v24], *a4, a3);
    v25 = (char *)*a1;
    v26 = (size_t)&a2[-*a1];
    v27 = (char *)(v24 + v26 + a3);
    if ( a2 == (char *)*a1 )
    {
      v28 = 0;
      v29 = a1[1] - (_QWORD)a2;
      if ( !v29 )
        goto LABEL_21;
    }
    else
    {
      v34 = (char *)*a1;
      memmove((void *)v24, (const void *)*a1, v26);
      v25 = v34;
      v29 = a1[1] - (_QWORD)a2;
      if ( !v29 )
        goto LABEL_26;
    }
    v32 = v25;
    v35 = v29;
    memcpy(v27, a2, v29);
    v25 = v32;
    v28 = v35;
LABEL_21:
    v27 += v28;
    if ( !v25 )
    {
LABEL_22:
      *a1 = v24;
      a1[1] = (__int64)v27;
      a1[2] = v23;
      return;
    }
LABEL_26:
    j_j___libc_free_0(v25, a1[2] - (_QWORD)v25);
    goto LABEL_22;
  }
  v8 = *a4;
  v9 = v7 - a2;
  if ( a3 < v7 - a2 )
  {
    v12 = &v7[-a3];
    v13 = (char *)memmove(v7, &v7[-a3], a3);
    a1[1] += a3;
    v14 = v12 - a2;
    if ( v14 )
      memmove(&v13[-v14], a2, v14);
    v15 = v8;
    v16 = a3;
    goto LABEL_11;
  }
  v10 = a3 - v9;
  if ( v10 )
  {
    v11 = &v7[v10];
    memset(v7, v8, v10);
    v7 = v11;
  }
  a1[1] = (__int64)v7;
  if ( v9 )
  {
    memmove(v7, a2, v9);
    a1[1] += v9;
    v15 = v8;
    v16 = v9;
LABEL_11:
    memset(a2, v15, v16);
  }
}
