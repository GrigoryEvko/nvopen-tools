// Function: sub_1E764A0
// Address: 0x1e764a0
//
__int64 __fastcall sub_1E764A0(__int64 *a1, char *a2, _QWORD *a3)
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
  __int64 v15; // rax
  _QWORD *v16; // rbx
  char *v17; // r15
  __int64 v18; // rcx
  _QWORD *v19; // rdi
  void *v20; // rdi
  __int64 v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-58h]
  char *v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h]
  __int64 v27; // [rsp+28h] [rbp-38h]

  v3 = (char *)*a1;
  v25 = (char *)a1[1];
  v4 = (__int64)&v25[-*a1] >> 3;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = a3;
  v6 = (v25 - v3) >> 3;
  v7 = v4 == 0;
  v8 = 1;
  if ( !v7 )
    v8 = (v25 - v3) >> 3;
  v10 = __CFADD__(v6, v8);
  v11 = v6 + v8;
  v12 = (char *)(a2 - v3);
  if ( v10 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v26 = 0;
      v13 = 8;
      v27 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0xFFFFFFFFFFFFFFFLL )
      v11 = 0xFFFFFFFFFFFFFFFLL;
    v22 = 8 * v11;
  }
  v24 = v5;
  v23 = sub_22077B0(v22);
  v12 = (char *)(a2 - v3);
  v5 = v24;
  v27 = v23;
  v26 = v23 + v22;
  v13 = v23 + 8;
LABEL_7:
  v14 = &v12[v27];
  if ( v14 )
  {
    v15 = *v5;
    *v5 = 0;
    *(_QWORD *)v14 = v15;
  }
  if ( a2 != v3 )
  {
    v16 = (_QWORD *)v27;
    v17 = v3;
    while ( 1 )
    {
      v19 = *(_QWORD **)v17;
      if ( v16 )
        break;
      if ( !v19 )
        goto LABEL_12;
      v17 += 8;
      (*(void (__fastcall **)(_QWORD *, char *, char *, _QWORD, _QWORD *))(*v19 + 16LL))(v19, a2, v14, *v19, v5);
      v18 = 8;
      if ( a2 == v17 )
      {
LABEL_17:
        v13 = (__int64)(v16 + 2);
        goto LABEL_18;
      }
LABEL_13:
      v16 = (_QWORD *)v18;
    }
    *v16 = v19;
    *(_QWORD *)v17 = 0;
LABEL_12:
    v17 += 8;
    v18 = (__int64)(v16 + 1);
    if ( a2 == v17 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v25 )
  {
    v20 = (void *)v13;
    v13 += v25 - a2;
    memcpy(v20, a2, v25 - a2);
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[2] - (_QWORD)v3);
  a1[1] = v13;
  *a1 = v27;
  a1[2] = v26;
  return v26;
}
