// Function: sub_1F08390
// Address: 0x1f08390
//
__int64 *__fastcall sub_1F08390(__int64 *a1, _QWORD *a2, _QWORD *a3)
{
  _QWORD *v4; // r13
  __int64 v5; // rax
  bool v6; // zf
  __int64 v8; // rsi
  __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rdx
  __int64 v13; // rbx
  char *v14; // rax
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  _QWORD *v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rdx
  _QWORD *v20; // rbx
  _QWORD *i; // r14
  _QWORD *v22; // rcx
  _QWORD *v23; // rax
  _QWORD *v24; // rsi
  _QWORD *v25; // r15
  _QWORD *v26; // rdi
  _QWORD *j; // r12
  _QWORD *v28; // rdx
  __int64 *v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  _QWORD *v35; // [rsp+0h] [rbp-60h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  _QWORD *v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v4 = (_QWORD *)a1[1];
  v38 = (_QWORD *)*a1;
  v5 = ((__int64)v4 - *a1) >> 5;
  if ( v5 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = v5 == 0;
  v8 = ((__int64)v4 - *a1) >> 5;
  v9 = 1;
  if ( !v6 )
    v9 = ((__int64)v4 - *a1) >> 5;
  v10 = __CFADD__(v8, v9);
  v11 = v8 + v9;
  v12 = (char *)((char *)a2 - (char *)v38);
  if ( v10 )
  {
    v33 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v36 = 0;
      v13 = 32;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x3FFFFFFFFFFFFFFLL )
      v11 = 0x3FFFFFFFFFFFFFFLL;
    v33 = 32 * v11;
  }
  v35 = a3;
  v34 = sub_22077B0(v33);
  v12 = (char *)((char *)a2 - (char *)v38);
  a3 = v35;
  v39 = v34;
  v36 = v34 + v33;
  v13 = v34 + 32;
LABEL_7:
  v14 = &v12[v39];
  if ( &v12[v39] )
  {
    v15 = (_QWORD *)a3[1];
    v16 = v14 + 8;
    v17 = (_QWORD *)a3[2];
    v18 = a3[3];
    *(_QWORD *)v14 = *a3;
    v19 = a3 + 1;
    *((_QWORD *)v14 + 1) = v15;
    *((_QWORD *)v14 + 2) = v17;
    *((_QWORD *)v14 + 3) = v18;
    if ( a3 + 1 == v15 )
    {
      *((_QWORD *)v14 + 2) = v16;
      *((_QWORD *)v14 + 1) = v16;
    }
    else
    {
      *v17 = v16;
      *(_QWORD *)(*((_QWORD *)v14 + 1) + 8LL) = v16;
      a3[2] = v19;
      a3[1] = v19;
      a3[3] = 0;
    }
  }
  if ( a2 != v38 )
  {
    v20 = (_QWORD *)v39;
    for ( i = v38 + 1; ; i += 4 )
    {
      if ( v20 )
      {
        *v20 = *(i - 1);
        v22 = (_QWORD *)*i;
        v23 = v20 + 1;
        v20[1] = *i;
        v24 = (_QWORD *)i[1];
        v20[2] = v24;
        v20[3] = i[2];
        if ( i != v22 )
        {
          *v24 = v23;
          *(_QWORD *)(v20[1] + 8LL) = v23;
          i[1] = i;
          *i = i;
          i[2] = 0;
          goto LABEL_14;
        }
        v20[2] = v23;
        v20[1] = v23;
      }
      v25 = (_QWORD *)*i;
      while ( i != v25 )
      {
        v26 = v25;
        v25 = (_QWORD *)*v25;
        j_j___libc_free_0(v26, 24);
      }
LABEL_14:
      if ( a2 == i + 3 )
      {
        v13 = (__int64)(v20 + 8);
        break;
      }
      v20 += 4;
    }
  }
  if ( a2 != v4 )
  {
    for ( j = a2 + 1; ; j += 4 )
    {
      v28 = (_QWORD *)*j;
      v29 = (__int64 *)j[1];
      v30 = j[2];
      *(_QWORD *)v13 = *(j - 1);
      v31 = v13 + 8;
      *(_QWORD *)(v13 + 8) = v28;
      *(_QWORD *)(v13 + 16) = v29;
      *(_QWORD *)(v13 + 24) = v30;
      if ( v28 == j )
      {
        *(_QWORD *)(v13 + 16) = v31;
        *(_QWORD *)(v13 + 8) = v31;
      }
      else
      {
        *v29 = v31;
        *(_QWORD *)(*(_QWORD *)(v13 + 8) + 8LL) = v31;
        j[1] = j;
        *j = j;
        j[2] = 0;
      }
      v13 += 32;
      if ( v4 == j + 3 )
        break;
    }
  }
  if ( v38 )
    j_j___libc_free_0(v38, a1[2] - (_QWORD)v38);
  *a1 = v39;
  a1[1] = v13;
  a1[2] = v36;
  return a1;
}
