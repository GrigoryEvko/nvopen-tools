// Function: sub_1696100
// Address: 0x1696100
//
char **__fastcall sub_1696100(char **a1, char *a2, _QWORD *a3, __int64 *a4)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  char *v8; // r13
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rbx
  __int64 v12; // r12
  char *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rbx
  __m128i *v16; // rax
  char *v17; // rbx
  _QWORD *i; // r12
  _QWORD *v19; // rdx
  _QWORD *v20; // rax
  char *v21; // r15
  char *v22; // rdi
  char *v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // [rsp+10h] [rbp-60h]
  char *v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]
  _QWORD *v34; // [rsp+30h] [rbp-40h]
  char *v35; // [rsp+38h] [rbp-38h]

  v35 = a1[1];
  v31 = *a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v35 - *a1) >> 3);
  if ( v6 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v8 = a2;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((v35 - *a1) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x5555555555555555LL * ((v35 - *a1) >> 3);
  v11 = (char *)(a2 - v31);
  if ( v9 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_31:
    v34 = a3;
    v28 = sub_22077B0(v27);
    a3 = v34;
    v32 = v28;
    v29 = v28 + v27;
    v12 = v28 + 24;
    goto LABEL_7;
  }
  if ( v10 )
  {
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v27 = 24 * v10;
    goto LABEL_31;
  }
  v29 = 0;
  v12 = 24;
  v32 = 0;
LABEL_7:
  v13 = &v11[v32];
  if ( &v11[v32] )
  {
    v14 = *a4;
    v15 = *a3;
    *((_QWORD *)v13 + 1) = v13;
    *(_QWORD *)v13 = v13;
    v33 = v14;
    for ( *((_QWORD *)v13 + 2) = 0; v33 != v15; ++*((_QWORD *)v13 + 2) )
    {
      v15 += 16;
      v16 = (__m128i *)sub_22077B0(32);
      v16[1] = _mm_loadu_si128((const __m128i *)(v15 - 16));
      sub_2208C80(v16, v13);
    }
  }
  v17 = v31;
  if ( a2 != v31 )
  {
    for ( i = (_QWORD *)v32; ; i = v20 )
    {
      v21 = *(char **)v17;
      if ( i )
      {
        *i = v21;
        v19 = (_QWORD *)*((_QWORD *)v17 + 1);
        i[1] = v19;
        i[2] = *((_QWORD *)v17 + 2);
        if ( v21 != v17 )
        {
          *v19 = i;
          *(_QWORD *)(*i + 8LL) = i;
          *((_QWORD *)v17 + 1) = v17;
          *(_QWORD *)v17 = v17;
          *((_QWORD *)v17 + 2) = 0;
LABEL_14:
          v17 += 24;
          v20 = i + 3;
          if ( v17 == a2 )
            goto LABEL_20;
          continue;
        }
        i[1] = i;
        *i = i;
        v21 = *(char **)v17;
      }
      if ( v17 == v21 )
        goto LABEL_14;
      do
      {
        v22 = v21;
        v21 = *(char **)v21;
        j_j___libc_free_0(v22, 32);
      }
      while ( v17 != v21 );
      v17 += 24;
      v20 = i + 3;
      if ( v17 == a2 )
      {
LABEL_20:
        v12 = (__int64)(i + 6);
        break;
      }
    }
  }
  if ( a2 != v35 )
  {
    do
    {
      while ( 1 )
      {
        v23 = *(char **)v8;
        v24 = (__int64 *)*((_QWORD *)v8 + 1);
        v25 = *((_QWORD *)v8 + 2);
        *(_QWORD *)v12 = *(_QWORD *)v8;
        *(_QWORD *)(v12 + 8) = v24;
        *(_QWORD *)(v12 + 16) = v25;
        if ( v23 == v8 )
          break;
        *v24 = v12;
        *(_QWORD *)(*(_QWORD *)v12 + 8LL) = v12;
        v12 += 24;
        *((_QWORD *)v8 + 1) = v8;
        *(_QWORD *)v8 = v8;
        v8 += 24;
        *((_QWORD *)v8 - 1) = 0;
        if ( v8 == v35 )
          goto LABEL_26;
      }
      *(_QWORD *)(v12 + 8) = v12;
      v8 += 24;
      *(_QWORD *)v12 = v12;
      v12 += 24;
    }
    while ( v8 != v35 );
  }
LABEL_26:
  if ( v31 )
    j_j___libc_free_0(v31, a1[2] - v31);
  *a1 = (char *)v32;
  a1[1] = (char *)v12;
  a1[2] = (char *)v29;
  return a1;
}
