// Function: sub_127D720
// Address: 0x127d720
//
char *__fastcall sub_127D720(__int64 a1, _BYTE *a2, _QWORD *a3)
{
  __int64 v3; // r14
  _BYTE *v4; // r8
  __int64 v5; // rax
  __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rcx
  signed __int64 v12; // rdx
  __int64 v13; // rbx
  char *result; // rax
  char *v15; // r9
  __int64 v16; // r15
  signed __int64 v17; // r10
  char *v18; // r14
  __int64 v19; // rsi
  char *v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  _BYTE *v25; // [rsp+10h] [rbp-40h]
  signed __int64 v26; // [rsp+10h] [rbp-40h]
  char *v27; // [rsp+18h] [rbp-38h]
  _BYTE *v28; // [rsp+18h] [rbp-38h]
  char *v29; // [rsp+18h] [rbp-38h]
  _BYTE *v30; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_BYTE **)a1;
  v5 = (v3 - *(_QWORD *)a1) >> 3;
  if ( v5 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v5 )
    v7 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3;
  v9 = __CFADD__(v7, v5);
  v10 = v7 + v5;
  v11 = (char *)v9;
  v12 = a2 - v4;
  if ( v9 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0xFFFFFFFFFFFFFFFLL )
      v10 = 0xFFFFFFFFFFFFFFFLL;
    v21 = 8 * v10;
  }
  v26 = a2 - v4;
  v30 = *(_BYTE **)a1;
  v22 = sub_22077B0(v21);
  v4 = v30;
  v12 = v26;
  v11 = (char *)v22;
  v13 = v22 + v21;
LABEL_7:
  result = &v11[v12];
  if ( &v11[v12] )
    *(_QWORD *)result = *a3;
  v15 = &v11[v12 + 8];
  v16 = *(_QWORD *)(a1 + 16);
  v17 = v3 - (_QWORD)a2;
  v18 = &v15[v3 - (_QWORD)a2];
  if ( v12 > 0 )
  {
    v23 = v17;
    v24 = (__int64)&v11[v12 + 8];
    v28 = v4;
    v20 = (char *)memmove(v11, v4, v12);
    v4 = v28;
    v17 = v23;
    v15 = (char *)v24;
    v11 = v20;
    v19 = v16 - (_QWORD)v28;
    if ( v23 <= 0 )
      goto LABEL_14;
    goto LABEL_16;
  }
  if ( v17 > 0 )
  {
LABEL_16:
    v25 = v4;
    v29 = v11;
    result = (char *)memcpy(v15, a2, v17);
    v4 = v25;
    v11 = v29;
    if ( !v25 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v4 )
  {
LABEL_13:
    v19 = v16 - (_QWORD)v4;
LABEL_14:
    v27 = v11;
    result = (char *)j_j___libc_free_0(v4, v19);
    v11 = v27;
  }
LABEL_12:
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = v13;
  return result;
}
