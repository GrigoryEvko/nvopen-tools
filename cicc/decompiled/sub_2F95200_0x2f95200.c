// Function: sub_2F95200
// Address: 0x2f95200
//
void __fastcall sub_2F95200(unsigned __int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r10
  void *v19; // rdi
  unsigned int v20; // r11d
  size_t v21; // rdx
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int v25; // [rsp-54h] [rbp-54h]
  unsigned int v26; // [rsp-54h] [rbp-54h]
  __int64 v27; // [rsp-50h] [rbp-50h]
  void **v28; // [rsp-50h] [rbp-50h]
  __int64 v29; // [rsp-50h] [rbp-50h]
  unsigned __int64 v30; // [rsp-48h] [rbp-48h]
  unsigned __int64 v31; // [rsp-48h] [rbp-48h]
  unsigned __int64 v32; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v7 = a1[1];
  v8 = *a1;
  v9 = v7 - *a1;
  v10 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 4);
  if ( a2 <= 0xAAAAAAAAAAAAAAABLL * ((__int64)(a1[2] - v7) >> 4) )
  {
    v11 = a2;
    v12 = a1[1];
    do
    {
      if ( v12 )
      {
        *(_DWORD *)(v12 + 8) = 0;
        *(_QWORD *)v12 = v12 + 16;
        *(_DWORD *)(v12 + 12) = 4;
      }
      v12 += 48LL;
      --v11;
    }
    while ( v11 );
    a1[1] = 48 * a2 + v7;
    return;
  }
  if ( 0x2AAAAAAAAAAAAAALL - v10 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v13 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v7 - *a1) >> 4);
  if ( a2 >= v10 )
    v13 = a2;
  v14 = __CFADD__(v10, v13);
  v15 = v10 + v13;
  if ( v14 )
  {
    v23 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v15 )
    {
      v30 = 0;
      v32 = 0;
      goto LABEL_15;
    }
    if ( v15 > 0x2AAAAAAAAAAAAAALL )
      v15 = 0x2AAAAAAAAAAAAAALL;
    v23 = 48 * v15;
  }
  v29 = v7 - *a1;
  v31 = v23;
  v24 = sub_22077B0(v23);
  v7 = a1[1];
  v8 = *a1;
  v9 = v29;
  v32 = v24;
  v30 = v24 + v31;
LABEL_15:
  v16 = v9 + v32;
  v17 = a2;
  do
  {
    if ( v16 )
    {
      *(_DWORD *)(v16 + 8) = 0;
      *(_QWORD *)v16 = v16 + 16;
      *(_DWORD *)(v16 + 12) = 4;
    }
    v16 += 48LL;
    --v17;
  }
  while ( v17 );
  if ( v7 != v8 )
  {
    v18 = v32;
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
        v8 += 48LL;
        v18 += 48;
        if ( v7 == v8 )
          goto LABEL_28;
      }
      v21 = 8LL * v20;
      if ( v20 <= 4
        || (v26 = *(_DWORD *)(v8 + 8),
            v28 = (void **)v18,
            sub_C8D5F0(v18, (const void *)(v18 + 16), v20, 8u, v20, a6),
            v18 = (__int64)v28,
            v20 = v26,
            v21 = 8LL * *(unsigned int *)(v8 + 8),
            v19 = *v28,
            v21) )
      {
        v25 = v20;
        v27 = v18;
        memcpy(v19, *(const void **)v8, v21);
        v20 = v25;
        v18 = v27;
      }
      v8 += 48LL;
      *(_DWORD *)(v18 + 8) = v20;
      v18 += 48;
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
        v8 += 48LL;
      }
      while ( v22 != v8 );
      v8 = *a1;
    }
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  *a1 = v32;
  a1[1] = v32 + 48 * (a2 + v10);
  a1[2] = v30;
}
