// Function: sub_1695EB0
// Address: 0x1695eb0
//
__int64 *__fastcall sub_1695EB0(__int64 *a1, char *a2)
{
  char *v2; // r14
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  char *v5; // r13
  bool v6; // cf
  unsigned __int64 v7; // rax
  char *v8; // rbx
  __int64 v9; // r12
  char *v10; // rax
  char *v11; // rbx
  _QWORD *i; // r12
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  char *v15; // r15
  char *v16; // rdi
  char *v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r12
  __int64 v22; // [rsp+0h] [rbp-50h]
  char *v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v2 = (char *)a1[1];
  v24 = (char *)*a1;
  v3 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v2[-*a1] >> 3);
  if ( v3 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v4 = 1;
  v5 = a2;
  if ( v3 )
    v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v2[-*a1] >> 3);
  v6 = __CFADD__(v4, v3);
  v7 = v4 - 0x5555555555555555LL * ((__int64)&v2[-*a1] >> 3);
  v8 = (char *)(a2 - v24);
  if ( v6 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_30:
    v25 = sub_22077B0(v21);
    v22 = v25 + v21;
    v9 = v25 + 24;
    goto LABEL_7;
  }
  if ( v7 )
  {
    if ( v7 > 0x555555555555555LL )
      v7 = 0x555555555555555LL;
    v21 = 24 * v7;
    goto LABEL_30;
  }
  v22 = 0;
  v9 = 24;
  v25 = 0;
LABEL_7:
  v10 = &v8[v25];
  if ( &v8[v25] )
  {
    *((_QWORD *)v10 + 1) = v10;
    *(_QWORD *)v10 = v10;
    *((_QWORD *)v10 + 2) = 0;
  }
  v11 = v24;
  if ( a2 == v24 )
    goto LABEL_25;
  for ( i = (_QWORD *)v25; ; i = v14 )
  {
    v15 = *(char **)v11;
    if ( i )
    {
      *i = v15;
      v13 = (_QWORD *)*((_QWORD *)v11 + 1);
      i[1] = v13;
      i[2] = *((_QWORD *)v11 + 2);
      if ( v15 != v11 )
      {
        *v13 = i;
        *(_QWORD *)(*i + 8LL) = i;
        *((_QWORD *)v11 + 1) = v11;
        *(_QWORD *)v11 = v11;
        *((_QWORD *)v11 + 2) = 0;
        goto LABEL_13;
      }
      i[1] = i;
      *i = i;
      v15 = *(char **)v11;
    }
    if ( v11 != v15 )
      break;
LABEL_13:
    v11 += 24;
    v14 = i + 3;
    if ( v11 == a2 )
      goto LABEL_19;
LABEL_14:
    ;
  }
  do
  {
    v16 = v15;
    v15 = *(char **)v15;
    j_j___libc_free_0(v16, 32);
  }
  while ( v15 != v11 );
  v11 += 24;
  v14 = i + 3;
  if ( v11 != a2 )
    goto LABEL_14;
LABEL_19:
  v9 = (__int64)(i + 6);
  if ( a2 != v2 )
  {
    do
    {
      v18 = *(char **)v5;
      v19 = (__int64 *)*((_QWORD *)v5 + 1);
      v20 = *((_QWORD *)v5 + 2);
      *(_QWORD *)v9 = *(_QWORD *)v5;
      *(_QWORD *)(v9 + 8) = v19;
      *(_QWORD *)(v9 + 16) = v20;
      if ( v5 == v18 )
      {
        *(_QWORD *)(v9 + 8) = v9;
        *(_QWORD *)v9 = v9;
      }
      else
      {
        *v19 = v9;
        *(_QWORD *)(*(_QWORD *)v9 + 8LL) = v9;
        *((_QWORD *)v5 + 1) = v5;
        *(_QWORD *)v5 = v5;
        *((_QWORD *)v5 + 2) = 0;
      }
      v5 += 24;
      v9 += 24;
LABEL_25:
      ;
    }
    while ( v5 != v2 );
  }
  if ( v24 )
    j_j___libc_free_0(v24, a1[2] - (_QWORD)v24);
  *a1 = v25;
  a1[1] = v9;
  a1[2] = v22;
  return a1;
}
