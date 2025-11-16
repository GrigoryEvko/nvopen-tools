// Function: sub_C99050
// Address: 0xc99050
//
__int64 __fastcall sub_C99050(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 *v4; // rdi
  __int64 *v5; // rdi
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  _QWORD *v10; // rdi
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 *v13; // rdi
  __int64 *v14; // rdi
  __int64 *v15; // rdi
  __int64 result; // rax
  __int64 v17; // r13
  _QWORD *v18; // r12
  _QWORD *v19; // rbx
  _QWORD *v20; // r15
  _QWORD *v21; // rdi
  _QWORD *v22; // rdi
  _QWORD *v23; // rdi
  _QWORD *v24; // rdi
  _QWORD *v25; // rdi
  _QWORD *v26; // rdi
  __int64 v27; // [rsp+8h] [rbp-38h]

  v3 = a1 + 2081;
  v4 = (__int64 *)a1[2078];
  if ( v4 != v3 )
    _libc_free(v4, a2);
  v5 = (__int64 *)a1[2073];
  if ( v5 != a1 + 2075 )
  {
    a2 = a1[2075] + 1;
    j_j___libc_free_0(v5, a2);
  }
  v6 = a1[2068];
  if ( *((_DWORD *)a1 + 4139) )
  {
    v7 = *((unsigned int *)a1 + 4138);
    if ( (_DWORD)v7 )
    {
      v8 = 8 * v7;
      v9 = 0;
      do
      {
        v10 = *(_QWORD **)(v6 + v9);
        if ( v10 != (_QWORD *)-8LL && v10 )
        {
          a2 = *v10 + 25LL;
          sub_C7D6A0((__int64)v10, a2, 8);
          v6 = a1[2068];
        }
        v9 += 8;
      }
      while ( v9 != v8 );
    }
  }
  _libc_free(v6, a2);
  v11 = (__int64 *)a1[18];
  v12 = &v11[16 * (unsigned __int64)*((unsigned int *)a1 + 38)];
  if ( v11 != v12 )
  {
    do
    {
      v12 -= 16;
      v13 = (__int64 *)v12[10];
      if ( v13 != v12 + 12 )
      {
        a2 = v12[12] + 1;
        j_j___libc_free_0(v13, a2);
      }
      v14 = (__int64 *)v12[6];
      if ( v14 != v12 + 8 )
      {
        a2 = v12[8] + 1;
        j_j___libc_free_0(v14, a2);
      }
      v15 = (__int64 *)v12[2];
      if ( v15 != v12 + 4 )
      {
        a2 = v12[4] + 1;
        j_j___libc_free_0(v15, a2);
      }
    }
    while ( v11 != v12 );
    v12 = (__int64 *)a1[18];
  }
  if ( v12 != a1 + 20 )
    _libc_free(v12, a2);
  result = *((unsigned int *)a1 + 2);
  v17 = *a1 + 8 * result;
  v27 = *a1;
  if ( *a1 != v17 )
  {
    do
    {
      v18 = *(_QWORD **)(v17 - 8);
      v17 -= 8;
      if ( v18 )
      {
        v19 = (_QWORD *)v18[17];
        v20 = (_QWORD *)v18[16];
        if ( v19 != v20 )
        {
          do
          {
            v21 = (_QWORD *)v20[10];
            if ( v21 != v20 + 12 )
              j_j___libc_free_0(v21, v20[12] + 1LL);
            v22 = (_QWORD *)v20[6];
            if ( v22 != v20 + 8 )
              j_j___libc_free_0(v22, v20[8] + 1LL);
            v23 = (_QWORD *)v20[2];
            if ( v23 != v20 + 4 )
              j_j___libc_free_0(v23, v20[4] + 1LL);
            v20 += 16;
          }
          while ( v19 != v20 );
          v20 = (_QWORD *)v18[16];
        }
        if ( v20 )
          j_j___libc_free_0(v20, v18[18] - (_QWORD)v20);
        v24 = (_QWORD *)v18[10];
        if ( v24 != v18 + 12 )
          j_j___libc_free_0(v24, v18[12] + 1LL);
        v25 = (_QWORD *)v18[6];
        if ( v25 != v18 + 8 )
          j_j___libc_free_0(v25, v18[8] + 1LL);
        v26 = (_QWORD *)v18[2];
        if ( v26 != v18 + 4 )
          j_j___libc_free_0(v26, v18[4] + 1LL);
        a2 = 152;
        result = j_j___libc_free_0(v18, 152);
      }
    }
    while ( v27 != v17 );
    v17 = *a1;
  }
  if ( (__int64 *)v17 != a1 + 2 )
    return _libc_free(v17, a2);
  return result;
}
