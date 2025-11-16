// Function: sub_CB3120
// Address: 0xcb3120
//
__int64 __fastcall sub_CB3120(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  unsigned int v3; // ecx
  __int64 v4; // rax
  unsigned __int64 v5; // r13
  unsigned __int64 i; // r12
  _QWORD *v7; // r15
  unsigned __int64 v8; // rbx
  __int64 v9; // r8
  _QWORD *v10; // r14
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // r15
  _QWORD *v15; // rdi
  unsigned __int64 v16; // r13
  unsigned __int64 j; // r12
  _QWORD *v18; // r15
  unsigned __int64 v19; // rbx
  __int64 v20; // r8
  _QWORD *v21; // r14
  __int64 v22; // r8
  __int64 v23; // rcx
  __int64 v24; // r14
  __int64 v25; // r15
  _QWORD *v26; // rdi
  __int64 *v27; // rbx
  __int64 *v28; // r12
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 result; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // r13
  __int64 *v36; // rbx
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 v39; // rax
  _QWORD *v40; // [rsp+8h] [rbp-48h]
  _QWORD *v41; // [rsp+8h] [rbp-48h]
  _QWORD *v43; // [rsp+18h] [rbp-38h]
  _QWORD *v44; // [rsp+18h] [rbp-38h]

  v1 = *(unsigned int *)(a1 + 24);
  v43 = *(_QWORD **)(a1 + 16);
  v40 = &v43[v1];
  if ( v43 != v40 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v3 = (unsigned int)(((__int64)v43 - v2) >> 3) >> 7;
      v4 = 4096LL << v3;
      if ( v3 >= 0x1E )
        v4 = 0x40000000000LL;
      v5 = *v43 + v4;
      if ( *v43 == *(_QWORD *)(v2 + 8 * v1 - 8) )
        v5 = *(_QWORD *)a1;
      for ( i = ((*v43 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 240; v5 >= i; i += 240LL )
      {
        v7 = *(_QWORD **)(i - 208);
        v8 = i - 240;
        v9 = 4LL * *(unsigned int *)(i - 200);
        v10 = &v7[v9];
        if ( v7 != &v7[v9] )
        {
          do
          {
            v10 -= 4;
            if ( (_QWORD *)*v10 != v10 + 2 )
            {
              v1 = v10[2] + 1LL;
              j_j___libc_free_0(*v10, v1);
            }
          }
          while ( v7 != v10 );
          v10 = *(_QWORD **)(v8 + 32);
        }
        if ( v10 != (_QWORD *)(i - 192) )
          _libc_free(v10, v1);
        v11 = *(_QWORD *)(v8 + 8);
        if ( *(_DWORD *)(v8 + 20) )
        {
          v12 = *(unsigned int *)(v8 + 16);
          if ( (_DWORD)v12 )
          {
            v13 = 8 * v12;
            v14 = 0;
            do
            {
              v15 = *(_QWORD **)(v11 + v14);
              if ( v15 && v15 != (_QWORD *)-8LL )
              {
                v1 = *v15 + 33LL;
                sub_C7D6A0((__int64)v15, v1, 8);
                v11 = *(_QWORD *)(v8 + 8);
              }
              v14 += 8;
            }
            while ( v13 != v14 );
          }
        }
        _libc_free(v11, v1);
      }
      if ( v40 == ++v43 )
        break;
      v2 = *(_QWORD *)(a1 + 16);
      v1 = *(unsigned int *)(a1 + 24);
    }
  }
  v44 = *(_QWORD **)(a1 + 64);
  v41 = &v44[2 * *(unsigned int *)(a1 + 72)];
  if ( v41 != v44 )
  {
    do
    {
      v16 = *v44 + v44[1];
      for ( j = ((*v44 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 240; v16 >= j; j += 240LL )
      {
        v18 = *(_QWORD **)(j - 208);
        v19 = j - 240;
        v20 = 4LL * *(unsigned int *)(j - 200);
        v21 = &v18[v20];
        if ( v18 != &v18[v20] )
        {
          do
          {
            v21 -= 4;
            if ( (_QWORD *)*v21 != v21 + 2 )
            {
              v1 = v21[2] + 1LL;
              j_j___libc_free_0(*v21, v1);
            }
          }
          while ( v18 != v21 );
          v21 = *(_QWORD **)(v19 + 32);
        }
        if ( v21 != (_QWORD *)(j - 192) )
          _libc_free(v21, v1);
        v22 = *(_QWORD *)(v19 + 8);
        if ( *(_DWORD *)(v19 + 20) )
        {
          v23 = *(unsigned int *)(v19 + 16);
          if ( (_DWORD)v23 )
          {
            v24 = 8 * v23;
            v25 = 0;
            do
            {
              v26 = *(_QWORD **)(v22 + v25);
              if ( v26 != (_QWORD *)-8LL && v26 )
              {
                v1 = *v26 + 33LL;
                sub_C7D6A0((__int64)v26, v1, 8);
                v22 = *(_QWORD *)(v19 + 8);
              }
              v25 += 8;
            }
            while ( v24 != v25 );
          }
        }
        _libc_free(v22, v1);
      }
      v44 += 2;
    }
    while ( v41 != v44 );
    v27 = *(__int64 **)(a1 + 64);
    v28 = &v27[2 * *(unsigned int *)(a1 + 72)];
    while ( v28 != v27 )
    {
      v29 = v27[1];
      v30 = *v27;
      v27 += 2;
      sub_C7D6A0(v30, v29, 16);
    }
  }
  result = a1;
  v32 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v32 )
  {
    v33 = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v34 = *v33;
    v35 = v33 + 1;
    *(_QWORD *)a1 = *v33;
    *(_QWORD *)(a1 + 8) = v34 + 4096;
    v36 = &v33[v32];
    if ( v36 != v33 + 1 )
    {
      while ( 1 )
      {
        v37 = *v35;
        v38 = (unsigned int)(v35 - v33) >> 7;
        v39 = 4096LL << v38;
        if ( v38 >= 0x1E )
          v39 = 0x40000000000LL;
        ++v35;
        sub_C7D6A0(v37, v39, 16);
        if ( v36 == v35 )
          break;
        v33 = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  return result;
}
