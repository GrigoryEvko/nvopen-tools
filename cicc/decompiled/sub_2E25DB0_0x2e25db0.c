// Function: sub_2E25DB0
// Address: 0x2e25db0
//
void __fastcall sub_2E25DB0(unsigned __int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // cf
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r10
  void *v19; // rdi
  unsigned int v20; // r11d
  size_t v21; // rdx
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  unsigned int v25; // [rsp-54h] [rbp-54h]
  unsigned int v26; // [rsp-54h] [rbp-54h]
  __int64 v27; // [rsp-50h] [rbp-50h]
  void **v28; // [rsp-50h] [rbp-50h]
  __int64 v29; // [rsp-50h] [rbp-50h]
  unsigned __int64 v30; // [rsp-48h] [rbp-48h]
  unsigned __int64 v31; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v7 = a1[1];
  v8 = *a1;
  v9 = v7 - *a1;
  v31 = v9 >> 5;
  if ( a2 <= (__int64)(a1[2] - v7) >> 5 )
  {
    v10 = a2;
    v11 = a1[1];
    do
    {
      if ( v11 )
      {
        *(_DWORD *)(v11 + 8) = 0;
        *(_QWORD *)v11 = v11 + 16;
        *(_DWORD *)(v11 + 12) = 4;
      }
      v11 += 32LL;
      --v10;
    }
    while ( v10 );
    a1[1] = 32 * a2 + v7;
    return;
  }
  if ( 0x3FFFFFFFFFFFFFFLL - v31 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v12 = (__int64)(a1[1] - *a1) >> 5;
  if ( a2 >= v31 )
    v12 = a2;
  v13 = __CFADD__(v31, v12);
  v14 = v31 + v12;
  if ( v13 )
  {
    v23 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v14 )
    {
      v30 = 0;
      v15 = 0;
      goto LABEL_15;
    }
    if ( v14 > 0x3FFFFFFFFFFFFFFLL )
      v14 = 0x3FFFFFFFFFFFFFFLL;
    v23 = 32 * v14;
  }
  v29 = a1[1] - *a1;
  v24 = sub_22077B0(v23);
  v7 = a1[1];
  v8 = *a1;
  v30 = v24;
  v9 = v29;
  v15 = v24 + v23;
LABEL_15:
  v16 = v9 + v30;
  v17 = a2;
  do
  {
    if ( v16 )
    {
      *(_DWORD *)(v16 + 8) = 0;
      *(_QWORD *)v16 = v16 + 16;
      *(_DWORD *)(v16 + 12) = 4;
    }
    v16 += 32LL;
    --v17;
  }
  while ( v17 );
  if ( v7 != v8 )
  {
    v18 = v30;
    do
    {
      while ( 1 )
      {
        if ( v18 )
        {
          v19 = (void *)(v18 + 16);
          *(_DWORD *)(v18 + 8) = 0;
          *(_QWORD *)v18 = v18 + 16;
          *(_DWORD *)(v18 + 12) = 4;
          v20 = *(_DWORD *)(v8 + 8);
          if ( v8 != v18 )
          {
            if ( v20 )
              break;
          }
        }
        v8 += 32LL;
        v18 += 32;
        if ( v7 == v8 )
          goto LABEL_28;
      }
      v21 = 4LL * v20;
      if ( v20 <= 4
        || (v26 = *(_DWORD *)(v8 + 8),
            v28 = (void **)v18,
            sub_C8D5F0(v18, (const void *)(v18 + 16), v20, 4u, v20, a6),
            v18 = (__int64)v28,
            v20 = v26,
            v21 = 4LL * *(unsigned int *)(v8 + 8),
            v19 = *v28,
            v21) )
      {
        v25 = v20;
        v27 = v18;
        memcpy(v19, *(const void **)v8, v21);
        v20 = v25;
        v18 = v27;
      }
      v8 += 32LL;
      *(_DWORD *)(v18 + 8) = v20;
      v18 += 32;
    }
    while ( v7 != v8 );
LABEL_28:
    v22 = a1[1];
    v8 = *a1;
    if ( v22 != *a1 )
    {
      do
      {
        if ( *(_QWORD *)v8 != v8 + 16 )
          _libc_free(*(_QWORD *)v8);
        v8 += 32LL;
      }
      while ( v22 != v8 );
      v8 = *a1;
    }
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  a1[2] = v15;
  *a1 = v30;
  a1[1] = 32 * (a2 + v31) + v30;
}
