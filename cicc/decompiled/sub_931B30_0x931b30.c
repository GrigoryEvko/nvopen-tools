// Function: sub_931B30
// Address: 0x931b30
//
char *__fastcall sub_931B30(__int64 a1, const void *a2, char *a3)
{
  __int64 v3; // r14
  _BYTE *v4; // r9
  __int64 v5; // rax
  __int64 v7; // rdx
  const void *v8; // r8
  bool v9; // cf
  __int64 v10; // rax
  __int64 v11; // r13
  signed __int64 v12; // rdx
  __int64 v13; // r13
  char *v14; // r12
  __int64 v15; // rax
  char *result; // rax
  char *v17; // r11
  __int64 v18; // r15
  signed __int64 v19; // r10
  char *v20; // r14
  __int64 v21; // rsi
  const void *v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  size_t n; // [rsp+10h] [rbp-40h]
  signed __int64 na; // [rsp+10h] [rbp-40h]
  _BYTE *v26; // [rsp+18h] [rbp-38h]
  _BYTE *v27; // [rsp+18h] [rbp-38h]
  _BYTE *v28; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_BYTE **)a1;
  v5 = v3 - *(_QWORD *)a1;
  if ( v5 == 0x7FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v5 )
    v7 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
  v9 = __CFADD__(v7, v5);
  v10 = v7 + v5;
  v11 = v10;
  v12 = (_BYTE *)a2 - v4;
  if ( v9 || v10 < 0 )
  {
    v11 = 0x7FFFFFFFFFFFFFFFLL;
  }
  else if ( !v10 )
  {
    v13 = 0;
    v14 = 0;
    goto LABEL_10;
  }
  n = (_BYTE *)a2 - v4;
  v26 = *(_BYTE **)a1;
  v15 = sub_22077B0(v11);
  v4 = v26;
  v12 = n;
  v8 = a2;
  v14 = (char *)v15;
  v13 = v15 + v11;
LABEL_10:
  result = &v14[v12];
  if ( &v14[v12] )
    *result = *a3;
  v17 = &v14[v12 + 1];
  v18 = *(_QWORD *)(a1 + 16);
  v19 = v3 - (_QWORD)v8;
  v20 = &v17[v3 - (_QWORD)v8];
  if ( v12 > 0 )
  {
    v22 = v8;
    v23 = (__int64)&v14[v12 + 1];
    na = v19;
    v27 = v4;
    memmove(v14, v4, v12);
    v4 = v27;
    v19 = na;
    v17 = (char *)v23;
    v8 = v22;
    v21 = v18 - (_QWORD)v27;
    if ( na <= 0 )
      goto LABEL_17;
    goto LABEL_19;
  }
  if ( v19 > 0 )
  {
LABEL_19:
    v28 = v4;
    result = (char *)memcpy(v17, v8, v19);
    v4 = v28;
    if ( !v28 )
      goto LABEL_15;
    goto LABEL_16;
  }
  if ( v4 )
  {
LABEL_16:
    v21 = v18 - (_QWORD)v4;
LABEL_17:
    result = (char *)j_j___libc_free_0(v4, v21);
  }
LABEL_15:
  *(_QWORD *)a1 = v14;
  *(_QWORD *)(a1 + 8) = v20;
  *(_QWORD *)(a1 + 16) = v13;
  return result;
}
