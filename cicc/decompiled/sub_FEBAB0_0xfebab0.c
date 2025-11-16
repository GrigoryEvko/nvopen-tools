// Function: sub_FEBAB0
// Address: 0xfebab0
//
unsigned __int64 __fastcall sub_FEBAB0(__int64 *a1, char *a2, __int64 a3)
{
  char *v4; // r12
  char *v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r14
  unsigned __int64 result; // rax
  __int64 v15; // rdx
  char *v16; // rax
  __int64 v17; // rdx
  char *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v4 = (char *)a1[1];
  v5 = (char *)*a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v4[-*a1] >> 4);
  if ( v6 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((a1[1] - *a1) >> 4);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x5555555555555555LL * ((a1[1] - *a1) >> 4);
  v10 = (char *)(a2 - v5);
  if ( v8 )
  {
    v20 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v11 = 48;
      v12 = 0;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x2AAAAAAAAAAAAAALL )
      v9 = 0x2AAAAAAAAAAAAAALL;
    v20 = 48 * v9;
  }
  v22 = a3;
  v25 = v20;
  v21 = sub_22077B0(v20);
  v10 = (char *)(a2 - v5);
  a3 = v22;
  v13 = v21;
  v11 = v21 + 48;
  v12 = v21 + v25;
LABEL_7:
  result = (unsigned __int64)&v10[v13];
  if ( &v10[v13] )
  {
    *(_QWORD *)result = *(_QWORD *)a3;
    *(_QWORD *)(result + 8) = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(result + 16) = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(result + 24) = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(result + 32) = *(_QWORD *)(a3 + 32);
    *(_DWORD *)(result + 40) = *(_DWORD *)(a3 + 40);
  }
  if ( a2 != v5 )
  {
    v15 = v13;
    v16 = v5;
    do
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v16;
        *(_QWORD *)(v15 + 8) = *((_QWORD *)v16 + 1);
        *(_QWORD *)(v15 + 16) = *((_QWORD *)v16 + 2);
        *(_QWORD *)(v15 + 24) = *((_QWORD *)v16 + 3);
        *(_QWORD *)(v15 + 32) = *((_QWORD *)v16 + 4);
        *(_DWORD *)(v15 + 40) = *((_DWORD *)v16 + 10);
      }
      v16 += 48;
      v15 += 48;
    }
    while ( a2 != v16 );
    result = (0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(a2 - 48 - v5) >> 4)) & 0xFFFFFFFFFFFFFFFLL;
    v11 = v13 + 16 * (3 * result + 6);
  }
  if ( a2 != v4 )
  {
    v17 = v11;
    v18 = a2;
    do
    {
      v19 = *(_QWORD *)v18;
      v18 += 48;
      v17 += 48;
      *(_QWORD *)(v17 - 48) = v19;
      *(_QWORD *)(v17 - 40) = *((_QWORD *)v18 - 5);
      *(_QWORD *)(v17 - 32) = *((_QWORD *)v18 - 4);
      *(_QWORD *)(v17 - 24) = *((_QWORD *)v18 - 3);
      *(_QWORD *)(v17 - 16) = *((_QWORD *)v18 - 2);
      *(_DWORD *)(v17 - 8) = *((_DWORD *)v18 - 2);
    }
    while ( v4 != v18 );
    result = 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v4 - a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3);
    v11 += result;
  }
  if ( v5 )
  {
    v23 = v12;
    v24 = v11;
    result = j_j___libc_free_0(v5, a1[2] - (_QWORD)v5);
    v12 = v23;
    v11 = v24;
  }
  *a1 = v13;
  a1[1] = v11;
  a1[2] = v12;
  return result;
}
