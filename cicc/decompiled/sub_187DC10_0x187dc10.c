// Function: sub_187DC10
// Address: 0x187dc10
//
__int64 __fastcall sub_187DC10(__int64 *a1, char *a2)
{
  char *v3; // rbx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // rsi
  __int64 v10; // rdx
  _QWORD *v11; // rax
  char *v12; // r14
  _QWORD *i; // r13
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rsi
  char *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdi
  char *v22; // rdi
  __int64 v24; // r13
  __int64 v25; // [rsp+8h] [rbp-48h]
  char *v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  v3 = (char *)a1[1];
  v26 = (char *)*a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v3[-*a1] >> 3);
  if ( v4 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v3[-*a1] >> 3);
  v7 = __CFADD__(v5, v4);
  v8 = v5 - 0x5555555555555555LL * ((__int64)&v3[-*a1] >> 3);
  v9 = a2 - v26;
  if ( v7 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v8 )
    {
      v25 = 0;
      v10 = 24;
      v28 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x555555555555555LL )
      v8 = 0x555555555555555LL;
    v24 = 24 * v8;
  }
  v28 = sub_22077B0(v24);
  v10 = v28 + 24;
  v25 = v28 + v24;
LABEL_7:
  v11 = (_QWORD *)(v28 + v9);
  if ( v28 + v9 )
  {
    *v11 = 0;
    v11[1] = 0;
    v11[2] = 0;
  }
  v12 = v26;
  if ( a2 != v26 )
  {
    for ( i = (_QWORD *)v28; ; i = (_QWORD *)v15 )
    {
      v16 = *((_QWORD *)v12 + 2);
      v17 = *(_QWORD *)v12;
      if ( i )
        break;
      v18 = v16 - v17;
      if ( !v17 )
        goto LABEL_12;
      j_j___libc_free_0(v17, v18);
      v12 += 24;
      v15 = 24;
      if ( v12 == a2 )
      {
LABEL_17:
        v10 = (__int64)(i + 6);
        goto LABEL_18;
      }
LABEL_13:
      ;
    }
    *i = v17;
    v14 = *((_QWORD *)v12 + 1);
    i[2] = v16;
    i[1] = v14;
    *((_QWORD *)v12 + 2) = 0;
    *(_QWORD *)v12 = 0;
LABEL_12:
    v12 += 24;
    v15 = (__int64)(i + 3);
    if ( v12 == a2 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v3 )
  {
    v19 = a2;
    v20 = v10;
    do
    {
      v21 = *(_QWORD *)v19;
      v19 += 24;
      v20 += 24;
      *(_QWORD *)(v20 - 24) = v21;
      *(_QWORD *)(v20 - 16) = *((_QWORD *)v19 - 2);
      *(_QWORD *)(v20 - 8) = *((_QWORD *)v19 - 1);
    }
    while ( v19 != v3 );
    v10 += 8 * ((unsigned __int64)(v19 - a2 - 24) >> 3) + 24;
  }
  v22 = v26;
  if ( v26 )
  {
    v27 = v10;
    j_j___libc_free_0(v22, a1[2] - (_QWORD)v22);
    v10 = v27;
  }
  a1[1] = v10;
  *a1 = v28;
  a1[2] = v25;
  return v25;
}
