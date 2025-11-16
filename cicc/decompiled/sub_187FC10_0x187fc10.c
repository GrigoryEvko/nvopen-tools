// Function: sub_187FC10
// Address: 0x187fc10
//
__int64 *__fastcall sub_187FC10(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  bool v7; // cf
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 i; // rbx
  __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // r8d
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rdi
  int v21; // edi
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v2 = a2;
  v3 = a1[1];
  v4 = *a1;
  v29 = *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * ((v3 - *a1) >> 4);
  if ( v5 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * ((v3 - v4) >> 4);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x5555555555555555LL * ((v3 - v4) >> 4);
  v9 = a2 - v29;
  if ( v7 )
  {
    v25 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_30:
    v26 = sub_22077B0(v25);
    v9 = a2 - v29;
    v30 = v26;
    v27 = v26 + v25;
    v10 = v26 + 48;
    goto LABEL_7;
  }
  if ( v8 )
  {
    if ( v8 > 0x2AAAAAAAAAAAAAALL )
      v8 = 0x2AAAAAAAAAAAAAALL;
    v25 = 48 * v8;
    goto LABEL_30;
  }
  v27 = 0;
  v10 = 48;
  v30 = 0;
LABEL_7:
  v11 = v30 + v9;
  if ( v30 + v9 )
  {
    *(_DWORD *)(v11 + 8) = 0;
    *(_QWORD *)(v11 + 16) = 0;
    *(_QWORD *)(v11 + 24) = v11 + 8;
    *(_QWORD *)(v11 + 32) = v11 + 8;
    *(_QWORD *)(v11 + 40) = 0;
  }
  v12 = v29;
  if ( a2 == v29 )
    goto LABEL_25;
  for ( i = v30; ; i = v17 )
  {
    if ( i )
    {
      v14 = *(_QWORD *)(v12 + 16);
      v15 = i + 8;
      if ( v14 )
      {
        v16 = *(_DWORD *)(v12 + 8);
        *(_QWORD *)(i + 16) = v14;
        *(_DWORD *)(i + 8) = v16;
        *(_QWORD *)(i + 24) = *(_QWORD *)(v12 + 24);
        *(_QWORD *)(i + 32) = *(_QWORD *)(v12 + 32);
        *(_QWORD *)(v14 + 8) = v15;
        *(_QWORD *)(i + 40) = *(_QWORD *)(v12 + 40);
        *(_QWORD *)(v12 + 16) = 0;
        *(_QWORD *)(v12 + 24) = v12 + 8;
        *(_QWORD *)(v12 + 32) = v12 + 8;
        *(_QWORD *)(v12 + 40) = 0;
        goto LABEL_13;
      }
      *(_DWORD *)(i + 8) = 0;
      *(_QWORD *)(i + 16) = 0;
      *(_QWORD *)(i + 24) = v15;
      *(_QWORD *)(i + 32) = v15;
      *(_QWORD *)(i + 40) = 0;
    }
    v18 = *(_QWORD *)(v12 + 16);
    if ( v18 )
      break;
LABEL_13:
    v12 += 48;
    v17 = i + 48;
    if ( v12 == v2 )
      goto LABEL_19;
LABEL_14:
    ;
  }
  do
  {
    sub_1876060(*(_QWORD *)(v18 + 24));
    v19 = v18;
    v18 = *(_QWORD *)(v18 + 16);
    j_j___libc_free_0(v19, 40);
  }
  while ( v18 );
  v12 += 48;
  v17 = i + 48;
  if ( v12 != v2 )
    goto LABEL_14;
LABEL_19:
  v10 = i + 96;
  while ( v2 != v3 )
  {
    v23 = *(_QWORD *)(v2 + 16);
    v24 = v10 + 8;
    if ( v23 )
    {
      v21 = *(_DWORD *)(v2 + 8);
      *(_QWORD *)(v10 + 16) = v23;
      *(_DWORD *)(v10 + 8) = v21;
      *(_QWORD *)(v10 + 24) = *(_QWORD *)(v2 + 24);
      *(_QWORD *)(v10 + 32) = *(_QWORD *)(v2 + 32);
      *(_QWORD *)(v23 + 8) = v24;
      v22 = *(_QWORD *)(v2 + 40);
      *(_QWORD *)(v2 + 16) = 0;
      *(_QWORD *)(v10 + 40) = v22;
      *(_QWORD *)(v2 + 24) = v2 + 8;
      *(_QWORD *)(v2 + 32) = v2 + 8;
      *(_QWORD *)(v2 + 40) = 0;
    }
    else
    {
      *(_DWORD *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = 0;
      *(_QWORD *)(v10 + 24) = v24;
      *(_QWORD *)(v10 + 32) = v24;
      *(_QWORD *)(v10 + 40) = 0;
    }
    v2 += 48;
    v10 += 48;
LABEL_25:
    ;
  }
  if ( v29 )
    j_j___libc_free_0(v29, a1[2] - v29);
  *a1 = v30;
  a1[1] = v10;
  a1[2] = v27;
  return a1;
}
