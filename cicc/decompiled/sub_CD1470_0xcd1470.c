// Function: sub_CD1470
// Address: 0xcd1470
//
char *__fastcall sub_CD1470(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r8
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  signed __int64 v11; // rdx
  __int64 v12; // r13
  char *v13; // rcx
  char *result; // rax
  char *v15; // r9
  __int64 v16; // r15
  signed __int64 v17; // r10
  char *v18; // r14
  __int64 v19; // rsi
  char *v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  _BYTE *v25; // [rsp+10h] [rbp-40h]
  signed __int64 v26; // [rsp+10h] [rbp-40h]
  char *v27; // [rsp+18h] [rbp-38h]
  _BYTE *v28; // [rsp+18h] [rbp-38h]
  char *v29; // [rsp+18h] [rbp-38h]
  _BYTE *v30; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *(_QWORD *)a1) >> 2);
  if ( v6 == 0xAAAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 2);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x5555555555555555LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 2);
  v11 = a2 - v5;
  if ( v9 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v12 = 0;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0xAAAAAAAAAAAAAAALL )
      v10 = 0xAAAAAAAAAAAAAAALL;
    v21 = 12 * v10;
  }
  v26 = a2 - v5;
  v30 = *(_BYTE **)a1;
  v22 = sub_22077B0(v21);
  v5 = v30;
  v11 = v26;
  v13 = (char *)v22;
  v12 = v22 + v21;
LABEL_7:
  result = &v13[v11];
  if ( &v13[v11] )
  {
    *(_QWORD *)result = *(_QWORD *)a3;
    *((_DWORD *)result + 2) = *(_DWORD *)(a3 + 8);
  }
  v15 = &v13[v11 + 12];
  v16 = *(_QWORD *)(a1 + 16);
  v17 = v4 - (_QWORD)a2;
  v18 = &v15[v4 - (_QWORD)a2];
  if ( v11 > 0 )
  {
    v23 = v17;
    v24 = (__int64)&v13[v11 + 12];
    v28 = v5;
    v20 = (char *)memmove(v13, v5, v11);
    v5 = v28;
    v17 = v23;
    v15 = (char *)v24;
    v13 = v20;
    v19 = v16 - (_QWORD)v28;
    if ( v23 <= 0 )
      goto LABEL_14;
    goto LABEL_16;
  }
  if ( v17 > 0 )
  {
LABEL_16:
    v25 = v5;
    v29 = v13;
    result = (char *)memcpy(v15, a2, v17);
    v5 = v25;
    v13 = v29;
    if ( !v25 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v5 )
  {
LABEL_13:
    v19 = v16 - (_QWORD)v5;
LABEL_14:
    v27 = v13;
    result = (char *)j_j___libc_free_0(v5, v19);
    v13 = v27;
  }
LABEL_12:
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = v12;
  return result;
}
