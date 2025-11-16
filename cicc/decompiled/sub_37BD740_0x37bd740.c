// Function: sub_37BD740
// Address: 0x37bd740
//
void __fastcall sub_37BD740(unsigned __int64 *a1, char *a2, unsigned __int64 a3, int *a4)
{
  char *v7; // r12
  char *v8; // rbx
  unsigned __int64 v9; // r9
  char *v10; // rdi
  signed __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 v13; // r13
  char *v14; // r8
  char *v15; // rdx
  char *v16; // rax
  unsigned __int64 v17; // rax
  char *v18; // r13
  unsigned __int64 v19; // r13
  char *v20; // rax
  unsigned __int64 v21; // rsi
  char *v22; // rcx
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  bool v26; // cf
  unsigned __int64 v27; // rax
  char *v28; // r12
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  char *v31; // rax
  unsigned __int64 v32; // rsi
  _DWORD *v33; // rsi
  char *v34; // rax
  char *v35; // r10
  char *v36; // rax
  char *v37; // r12
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // [rsp-58h] [rbp-58h]
  _DWORD *v42; // [rsp-50h] [rbp-50h]
  unsigned __int64 v43; // [rsp-50h] [rbp-50h]
  int v44; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    return;
  v7 = a2;
  v8 = a2;
  v9 = a1[2];
  v10 = (char *)a1[1];
  if ( (__int64)(v9 - (_QWORD)v10) >> 2 >= a3 )
  {
    v11 = v10 - a2;
    v44 = *a4;
    v12 = (v10 - a2) >> 2;
    if ( a3 >= v12 )
    {
      v19 = a3 - v12;
      v20 = v10;
      if ( v19 )
      {
        v21 = v19;
        do
        {
          if ( v20 )
            *(_DWORD *)v20 = v44;
          v20 += 4;
          --v21;
        }
        while ( v21 );
        v20 = &v10[4 * v19];
      }
      a1[1] = (unsigned __int64)v20;
      if ( v10 == v8 )
      {
        a1[1] = (unsigned __int64)&v20[v11];
      }
      else
      {
        v22 = &v20[v10 - v8];
        do
        {
          if ( v20 )
            *(_DWORD *)v20 = *(_DWORD *)v8;
          v20 += 4;
          v8 += 4;
        }
        while ( v20 != v22 );
        a1[1] += v11;
        do
        {
          *(_DWORD *)v7 = v44;
          v7 += 4;
        }
        while ( v10 != v7 );
      }
    }
    else
    {
      v13 = 4 * a3;
      v14 = &v10[-4 * a3];
      if ( v10 == v14 )
      {
        v17 = (unsigned __int64)v10;
      }
      else
      {
        v15 = &v10[-v13];
        v16 = v10;
        do
        {
          if ( v16 )
            *(_DWORD *)v16 = *(_DWORD *)v15;
          v16 += 4;
          v15 += 4;
        }
        while ( &v10[v13] != v16 );
        v17 = a1[1];
      }
      a1[1] = v13 + v17;
      if ( v14 != a2 )
        memmove(&a2[v13], a2, v14 - a2);
      v18 = &a2[v13];
      if ( v18 != a2 )
      {
        do
        {
          *(_DWORD *)v7 = v44;
          v7 += 4;
        }
        while ( v18 != v7 );
      }
    }
    return;
  }
  v23 = *a1;
  v24 = (__int64)&v10[-*a1] >> 2;
  if ( a3 > 0x1FFFFFFFFFFFFFFFLL - v24 )
    sub_4262D8((__int64)"vector::_M_fill_insert");
  v25 = (__int64)&v10[-*a1] >> 2;
  if ( a3 >= v24 )
    v25 = a3;
  v26 = __CFADD__(v25, v24);
  v27 = v25 + v24;
  v28 = &a2[-v23];
  v29 = v26;
  if ( v26 )
  {
    v38 = 0x7FFFFFFFFFFFFFFCLL;
  }
  else
  {
    if ( !v27 )
    {
      v30 = 0;
      goto LABEL_36;
    }
    v40 = 0x1FFFFFFFFFFFFFFFLL;
    if ( v27 <= 0x1FFFFFFFFFFFFFFFLL )
      v40 = v27;
    v38 = 4 * v40;
  }
  v43 = v38;
  v39 = sub_22077B0(v38);
  v23 = *a1;
  v10 = (char *)a1[1];
  v30 = v39;
  v29 = v39 + v43;
LABEL_36:
  v31 = &v28[v30];
  v32 = a3;
  do
  {
    if ( v31 )
      *(_DWORD *)v31 = *a4;
    v31 += 4;
    --v32;
  }
  while ( v32 );
  if ( v8 == (char *)v23 )
  {
    v35 = (char *)v30;
  }
  else
  {
    v33 = (_DWORD *)v23;
    v34 = (char *)v30;
    v35 = &v8[v30 - v23];
    do
    {
      if ( v34 )
        *(_DWORD *)v34 = *v33;
      v34 += 4;
      ++v33;
    }
    while ( v34 != v35 );
  }
  v36 = &v35[4 * a3];
  if ( v8 == v10 )
  {
    v37 = &v35[4 * a3];
  }
  else
  {
    v37 = &v36[v10 - v8];
    do
    {
      if ( v36 )
        *(_DWORD *)v36 = *(_DWORD *)v8;
      v36 += 4;
      v8 += 4;
    }
    while ( v36 != v37 );
  }
  if ( v23 )
  {
    v41 = v29;
    v42 = (_DWORD *)v30;
    j_j___libc_free_0(v23);
    v29 = v41;
    v30 = (unsigned __int64)v42;
  }
  *a1 = v30;
  a1[1] = (unsigned __int64)v37;
  a1[2] = v29;
}
