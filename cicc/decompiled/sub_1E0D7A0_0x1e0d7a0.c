// Function: sub_1E0D7A0
// Address: 0x1e0d7a0
//
__int64 __fastcall sub_1E0D7A0(char **a1, char *a2, __int64 *a3)
{
  char *v6; // rbx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  signed __int64 v11; // r8
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  char *v17; // r14
  _QWORD *i; // r13
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rsi
  char *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdi
  char *v27; // rdi
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // [rsp+18h] [rbp-48h]
  char *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  _QWORD *v34; // [rsp+28h] [rbp-38h]

  v6 = a1[1];
  v32 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((v6 - *a1) >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((v6 - *a1) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((v6 - *a1) >> 3);
  v11 = a2 - v32;
  if ( v9 )
  {
    v29 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v31 = 0;
      v12 = 24;
      v34 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v29 = 24 * v10;
  }
  v30 = sub_22077B0(v29);
  v11 = a2 - v32;
  v34 = (_QWORD *)v30;
  v12 = v30 + 24;
  v31 = v30 + v29;
LABEL_7:
  v13 = (_QWORD *)((char *)v34 + v11);
  if ( (_QWORD *)((char *)v34 + v11) )
  {
    v14 = *a3;
    *a3 = 0;
    *v13 = v14;
    v15 = a3[1];
    a3[1] = 0;
    v13[1] = v15;
    v16 = a3[2];
    a3[2] = 0;
    v13[2] = v16;
  }
  v17 = v32;
  if ( a2 != v32 )
  {
    for ( i = v34; ; i = (_QWORD *)v20 )
    {
      v21 = *((_QWORD *)v17 + 2);
      v22 = *(_QWORD *)v17;
      if ( i )
        break;
      v23 = v21 - v22;
      if ( !v22 )
        goto LABEL_12;
      j_j___libc_free_0(v22, v23);
      v17 += 24;
      v20 = 24;
      if ( v17 == a2 )
      {
LABEL_17:
        v12 = (__int64)(i + 6);
        goto LABEL_18;
      }
LABEL_13:
      ;
    }
    *i = v22;
    v19 = *((_QWORD *)v17 + 1);
    i[2] = v21;
    i[1] = v19;
    *((_QWORD *)v17 + 2) = 0;
    *(_QWORD *)v17 = 0;
LABEL_12:
    v17 += 24;
    v20 = (__int64)(i + 3);
    if ( v17 == a2 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v6 )
  {
    v24 = a2;
    v25 = v12;
    do
    {
      v26 = *(_QWORD *)v24;
      v24 += 24;
      v25 += 24;
      *(_QWORD *)(v25 - 24) = v26;
      *(_QWORD *)(v25 - 16) = *((_QWORD *)v24 - 2);
      *(_QWORD *)(v25 - 8) = *((_QWORD *)v24 - 1);
    }
    while ( v24 != v6 );
    v12 += 8 * ((unsigned __int64)(v24 - a2 - 24) >> 3) + 24;
  }
  v27 = v32;
  if ( v32 )
  {
    v33 = v12;
    j_j___libc_free_0(v27, a1[2] - v27);
    v12 = v33;
  }
  a1[1] = (char *)v12;
  *a1 = (char *)v34;
  a1[2] = (char *)v31;
  return v31;
}
