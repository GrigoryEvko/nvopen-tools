// Function: sub_3178480
// Address: 0x3178480
//
void __fastcall sub_3178480(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 i; // r12
  void *v12; // rdi
  int v13; // eax
  unsigned int v14; // r14d
  int v15; // eax
  __int64 v16; // rax
  const void *v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi

  v6 = *((unsigned int *)a1 + 2);
  v7 = 5 * v6;
  v8 = *a1 + 176 * v6;
  if ( *a1 != v8 )
  {
    for ( i = *a1 + 128; ; i += 176 )
    {
      if ( a2 )
      {
        *(_QWORD *)a2 = *(_QWORD *)(i - 128);
        *(_QWORD *)(a2 + 8) = *(_QWORD *)(i - 120);
        v15 = *(_DWORD *)(i - 112);
        *(_DWORD *)(a2 + 32) = 0;
        *(_DWORD *)(a2 + 16) = v15;
        *(_QWORD *)(a2 + 24) = a2 + 40;
        *(_DWORD *)(a2 + 36) = 4;
        if ( *(_DWORD *)(i - 96) )
          sub_3174A00(a2 + 24, (char **)(i - 104), v7, a4, a5, a6);
        v12 = (void *)(a2 + 128);
        *(_DWORD *)(a2 + 104) = *(_DWORD *)(i - 24);
        v13 = *(_DWORD *)(i - 20);
        *(_QWORD *)(a2 + 112) = a2 + 128;
        *(_DWORD *)(a2 + 108) = v13;
        *(_DWORD *)(a2 + 120) = 0;
        *(_DWORD *)(a2 + 124) = 6;
        v14 = *(_DWORD *)(i - 8);
        if ( v14 )
        {
          a5 = a2 + 112;
          if ( a2 + 112 != i - 16 )
          {
            v16 = *(_QWORD *)(i - 16);
            if ( v16 == i )
            {
              v17 = (const void *)i;
              v7 = 8LL * v14;
              if ( v14 <= 6
                || (sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v14, 8u, a5, v14),
                    v12 = *(void **)(a2 + 112),
                    v17 = *(const void **)(i - 16),
                    (v7 = 8LL * *(unsigned int *)(i - 8)) != 0) )
              {
                memcpy(v12, v17, v7);
              }
              *(_DWORD *)(a2 + 120) = v14;
              *(_DWORD *)(i - 8) = 0;
            }
            else
            {
              *(_QWORD *)(a2 + 112) = v16;
              *(_DWORD *)(a2 + 120) = *(_DWORD *)(i - 8);
              *(_DWORD *)(a2 + 124) = *(_DWORD *)(i - 4);
              *(_QWORD *)(i - 16) = i;
              *(_DWORD *)(i - 4) = 0;
              *(_DWORD *)(i - 8) = 0;
            }
          }
        }
      }
      a2 += 176;
      if ( v8 == i + 48 )
        break;
    }
    v18 = *a1;
    v19 = *a1 + 176LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v19 )
    {
      do
      {
        v19 -= 176;
        v20 = *(_QWORD *)(v19 + 112);
        if ( v20 != v19 + 128 )
          _libc_free(v20);
        v21 = *(_QWORD *)(v19 + 24);
        if ( v21 != v19 + 40 )
          _libc_free(v21);
      }
      while ( v19 != v18 );
    }
  }
}
