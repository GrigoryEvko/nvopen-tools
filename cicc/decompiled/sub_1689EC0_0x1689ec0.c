// Function: sub_1689EC0
// Address: 0x1689ec0
//
char *__fastcall sub_1689EC0(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r14
  _BYTE *v3; // r8
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // rdx
  __int64 v10; // r13
  char *v11; // r15
  char *result; // rax
  char *v13; // r9
  signed __int64 v14; // r10
  char *v15; // r14
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  _BYTE *v21; // [rsp+10h] [rbp-40h]
  _BYTE *v22; // [rsp+10h] [rbp-40h]
  signed __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]
  _BYTE *v25; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_BYTE **)a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * ((v2 - *(_QWORD *)a1) >> 3);
  if ( v4 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v7 = __CFADD__(v5, v4);
  v8 = v5 - 0x5555555555555555LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v9 = a2 - v3;
  if ( v7 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v8 )
    {
      v10 = 0;
      v11 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x555555555555555LL )
      v8 = 0x555555555555555LL;
    v17 = 24 * v8;
  }
  v23 = a2 - v3;
  v25 = *(_BYTE **)a1;
  v18 = sub_22077B0(v17);
  v3 = v25;
  v9 = v23;
  v11 = (char *)v18;
  v10 = v18 + v17;
LABEL_7:
  result = &v11[v9];
  if ( &v11[v9] )
  {
    *((_QWORD *)result + 2) = 0;
    *(_OWORD *)result = 0;
  }
  v13 = &v11[v9 + 24];
  v14 = v2 - (_QWORD)a2;
  v24 = *(_QWORD *)(a1 + 16);
  v15 = &v13[v2 - (_QWORD)a2];
  if ( v9 > 0 )
  {
    v19 = v14;
    v20 = (__int64)&v11[v9 + 24];
    v21 = v3;
    memmove(v11, v3, v9);
    v3 = v21;
    v14 = v19;
    v13 = (char *)v20;
    v16 = v24 - (_QWORD)v21;
    if ( v19 <= 0 )
      goto LABEL_14;
    goto LABEL_16;
  }
  if ( v14 > 0 )
  {
LABEL_16:
    v22 = v3;
    result = (char *)memcpy(v13, a2, v14);
    v3 = v22;
    if ( !v22 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v3 )
  {
LABEL_13:
    v16 = v24 - (_QWORD)v3;
LABEL_14:
    result = (char *)j_j___libc_free_0(v3, v16);
  }
LABEL_12:
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v15;
  *(_QWORD *)(a1 + 16) = v10;
  return result;
}
