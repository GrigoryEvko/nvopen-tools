// Function: sub_12DD860
// Address: 0x12dd860
//
__int64 __fastcall sub_12DD860(__int64 *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // r14
  __int64 v7; // rax
  __int64 v10; // rcx
  bool v11; // zf
  __int64 v12; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  char *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rbx
  char *v19; // rdx
  _QWORD *v20; // rbx
  char *v21; // r15
  __int64 v22; // rdi
  void *v23; // rdi
  __int64 v25; // rbx
  __int64 v26; // rax
  char *v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v6 = (char *)*a1;
  v27 = (char *)a1[1];
  v7 = (__int64)&v27[-*a1] >> 3;
  if ( v7 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = (v27 - v6) >> 3;
  v11 = v7 == 0;
  v12 = 1;
  if ( !v11 )
    v12 = (v27 - v6) >> 3;
  v14 = __CFADD__(v10, v12);
  v15 = v10 + v12;
  v16 = (char *)(a2 - v6);
  v17 = v14;
  if ( v14 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v15 )
    {
      v28 = 0;
      v18 = 8;
      v29 = 0;
      goto LABEL_7;
    }
    if ( v15 > 0xFFFFFFFFFFFFFFFLL )
      v15 = 0xFFFFFFFFFFFFFFFLL;
    v25 = 8 * v15;
  }
  v26 = sub_22077B0(v25);
  v16 = (char *)(a2 - v6);
  v29 = v26;
  v28 = v26 + v25;
  v18 = v26 + 8;
LABEL_7:
  v19 = &v16[v29];
  if ( v19 )
    *(_QWORD *)v19 = *(_QWORD *)a3;
  if ( a2 != v6 )
  {
    v20 = (_QWORD *)v29;
    v21 = v6;
    while ( 1 )
    {
      v22 = *(_QWORD *)v21;
      if ( v20 )
        break;
      if ( !v22 )
        goto LABEL_12;
      v30 = *(_QWORD *)v21;
      v21 += 8;
      sub_16025D0(v22, a3, v19, v17, a5, a6);
      a3 = 8;
      j_j___libc_free_0(v30, 8);
      v17 = 8;
      if ( v21 == a2 )
      {
LABEL_17:
        v18 = (__int64)(v20 + 2);
        goto LABEL_18;
      }
LABEL_13:
      v20 = (_QWORD *)v17;
    }
    *v20 = v22;
    *(_QWORD *)v21 = 0;
LABEL_12:
    v21 += 8;
    v17 = (__int64)(v20 + 1);
    if ( v21 == a2 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v27 )
  {
    v23 = (void *)v18;
    v18 += v27 - a2;
    memcpy(v23, a2, v27 - a2);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - (_QWORD)v6);
  a1[1] = v18;
  *a1 = v29;
  a1[2] = v28;
  return v28;
}
