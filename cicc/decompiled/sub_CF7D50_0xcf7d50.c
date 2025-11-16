// Function: sub_CF7D50
// Address: 0xcf7d50
//
__int64 __fastcall sub_CF7D50(__int64 *a1, char *a2, _QWORD *a3)
{
  char *v3; // r14
  __int64 v4; // rax
  _QWORD *v5; // r8
  __int64 v6; // rcx
  bool v7; // zf
  __int64 v8; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rdx
  __int64 v13; // rbx
  char *v14; // rdx
  _QWORD *v15; // rbx
  char *v16; // r15
  __int64 v17; // rcx
  _QWORD *v18; // rdi
  void *v19; // rdi
  __int64 v21; // rbx
  __int64 v22; // rax
  _QWORD *v23; // [rsp+8h] [rbp-58h]
  char *v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = (char *)*a1;
  v24 = (char *)a1[1];
  v4 = (__int64)&v24[-*a1] >> 3;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = a3;
  v6 = (v24 - v3) >> 3;
  v7 = v4 == 0;
  v8 = 1;
  if ( !v7 )
    v8 = (v24 - v3) >> 3;
  v10 = __CFADD__(v6, v8);
  v11 = v6 + v8;
  v12 = (char *)(a2 - v3);
  if ( v10 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v25 = 0;
      v13 = 8;
      v26 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0xFFFFFFFFFFFFFFFLL )
      v11 = 0xFFFFFFFFFFFFFFFLL;
    v21 = 8 * v11;
  }
  v23 = v5;
  v22 = sub_22077B0(v21);
  v12 = (char *)(a2 - v3);
  v5 = v23;
  v26 = v22;
  v25 = v22 + v21;
  v13 = v22 + 8;
LABEL_7:
  v14 = &v12[v26];
  if ( v14 )
    *(_QWORD *)v14 = *v5;
  if ( a2 != v3 )
  {
    v15 = (_QWORD *)v26;
    v16 = v3;
    while ( 1 )
    {
      v18 = *(_QWORD **)v16;
      if ( v15 )
        break;
      if ( !v18 )
        goto LABEL_12;
      v16 += 8;
      (*(void (__fastcall **)(_QWORD *, char *, char *, _QWORD, _QWORD *))(*v18 + 8LL))(v18, a2, v14, *v18, v5);
      v17 = 8;
      if ( a2 == v16 )
      {
LABEL_17:
        v13 = (__int64)(v15 + 2);
        goto LABEL_18;
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
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v24 )
  {
    v19 = (void *)v13;
    v13 += v24 - a2;
    memcpy(v19, a2, v24 - a2);
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[2] - (_QWORD)v3);
  a1[1] = v13;
  *a1 = v26;
  a1[2] = v25;
  return v25;
}
