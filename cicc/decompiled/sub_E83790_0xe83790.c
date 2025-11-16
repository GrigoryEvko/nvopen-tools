// Function: sub_E83790
// Address: 0xe83790
//
char **__fastcall sub_E83790(char **a1, char *a2, _QWORD *a3)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // rdx
  __int64 v10; // rbx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  char *v15; // r15
  _QWORD *i; // rbx
  __int64 v17; // rax
  _QWORD *v18; // r13
  _QWORD *v19; // r14
  __int64 v20; // rsi
  char *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  char *v29; // [rsp+28h] [rbp-48h]
  _QWORD *v30; // [rsp+30h] [rbp-40h]
  char *v31; // [rsp+38h] [rbp-38h]

  v31 = a1[1];
  v29 = *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * ((v31 - *a1) >> 3);
  if ( v5 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * ((v31 - *a1) >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x5555555555555555LL * ((v31 - *a1) >> 3);
  v9 = a2 - v29;
  if ( v7 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v8 )
    {
      v28 = 0;
      v10 = 24;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x555555555555555LL )
      v8 = 0x555555555555555LL;
    v25 = 24 * v8;
  }
  v27 = a3;
  v26 = sub_22077B0(v25);
  v9 = a2 - v29;
  a3 = v27;
  v30 = (_QWORD *)v26;
  v28 = v26 + v25;
  v10 = v26 + 24;
LABEL_7:
  v11 = (_QWORD *)((char *)v30 + v9);
  if ( (_QWORD *)((char *)v30 + v9) )
  {
    v12 = *a3;
    *a3 = 0;
    *v11 = v12;
    v13 = a3[1];
    a3[1] = 0;
    v11[1] = v13;
    v14 = a3[2];
    a3[2] = 0;
    v11[2] = v14;
  }
  v15 = v29;
  if ( a2 != v29 )
  {
    for ( i = v30; ; i = (_QWORD *)v17 )
    {
      v18 = (_QWORD *)*((_QWORD *)v15 + 1);
      v19 = *(_QWORD **)v15;
      if ( i )
        break;
      if ( v19 == v18 )
      {
        v20 = *((_QWORD *)v15 + 2) - (_QWORD)v19;
      }
      else
      {
        do
        {
          if ( (_QWORD *)*v19 != v19 + 2 )
            j_j___libc_free_0(*v19, v19[2] + 1LL);
          v19 += 4;
        }
        while ( v19 != v18 );
        v19 = *(_QWORD **)v15;
        v20 = *((_QWORD *)v15 + 2) - *(_QWORD *)v15;
      }
      if ( !v19 )
        goto LABEL_12;
      v15 += 24;
      j_j___libc_free_0(v19, v20);
      v17 = 24;
      if ( v15 == a2 )
      {
LABEL_22:
        v10 = (__int64)(i + 6);
        goto LABEL_23;
      }
LABEL_13:
      ;
    }
    *i = v19;
    i[1] = v18;
    i[2] = *((_QWORD *)v15 + 2);
    *((_QWORD *)v15 + 2) = 0;
    *((_QWORD *)v15 + 1) = 0;
    *(_QWORD *)v15 = 0;
LABEL_12:
    v15 += 24;
    v17 = (__int64)(i + 3);
    if ( v15 == a2 )
      goto LABEL_22;
    goto LABEL_13;
  }
LABEL_23:
  if ( a2 != v31 )
  {
    v21 = a2;
    v22 = v10;
    do
    {
      v23 = *(_QWORD *)v21;
      v22 += 24;
      v21 += 24;
      *(_QWORD *)(v22 - 24) = v23;
      *(_QWORD *)(v22 - 16) = *((_QWORD *)v21 - 2);
      *(_QWORD *)(v22 - 8) = *((_QWORD *)v21 - 1);
    }
    while ( v21 != v31 );
    v10 += 8 * ((unsigned __int64)(v21 - a2 - 24) >> 3) + 24;
  }
  if ( v29 )
    j_j___libc_free_0(v29, a1[2] - v29);
  a1[1] = (char *)v10;
  *a1 = (char *)v30;
  a1[2] = (char *)v28;
  return a1;
}
