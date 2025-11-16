// Function: sub_24FE0F0
// Address: 0x24fe0f0
//
void __fastcall sub_24FE0F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rax
  __int64 *v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  int v12; // r14d
  __int64 *v13; // rsi
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // rdx

  if ( a1 != a2 )
  {
    v8 = *(__int64 **)a2;
    v9 = (__int64 *)(a2 + 16);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_DWORD *)(a2 + 8);
      if ( v10 <= v11 )
      {
        if ( *(_DWORD *)(a2 + 8) )
        {
          v16 = *(_QWORD *)a1;
          v17 = *(_QWORD *)a1 + 32 * v10;
          do
          {
            v18 = *v9;
            v16 += 32LL;
            v9 += 4;
            *(_QWORD *)(v16 - 32) = v18;
            *(__m128i *)(v16 - 24) = _mm_loadu_si128((const __m128i *)(v9 - 3));
            *(_QWORD *)(v16 - 8) = *(v9 - 1);
          }
          while ( v16 != v17 );
        }
        goto LABEL_8;
      }
      if ( v10 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 0x20u, a5, a6);
        v11 = 0;
        v14 = 4LL * *(unsigned int *)(a2 + 8);
        v13 = *(__int64 **)a2;
        if ( *(_QWORD *)a2 == v14 * 8 + *(_QWORD *)a2 )
          goto LABEL_8;
      }
      else
      {
        v13 = (__int64 *)(a2 + 16);
        if ( *(_DWORD *)(a1 + 8) )
        {
          v19 = *(_QWORD *)a1;
          v11 *= 32LL;
          v20 = *(_QWORD *)a1 + v11;
          do
          {
            v21 = *v9;
            v19 += 32LL;
            v9 += 4;
            *(_QWORD *)(v19 - 32) = v21;
            *(__m128i *)(v19 - 24) = _mm_loadu_si128((const __m128i *)(v9 - 3));
            *(_QWORD *)(v19 - 8) = *(v9 - 1);
          }
          while ( v19 != v20 );
          v9 = *(__int64 **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v13 = (__int64 *)(*(_QWORD *)a2 + v11);
        }
        v14 = 4 * v10;
        if ( v13 == &v9[v14] )
          goto LABEL_8;
      }
      memcpy((void *)(v11 + *(_QWORD *)a1), v13, v14 * 8 - v11);
LABEL_8:
      *(_DWORD *)(a1 + 8) = v12;
      *(_DWORD *)(a2 + 8) = 0;
      return;
    }
    v15 = *(_QWORD *)a1;
    if ( v15 != a1 + 16 )
    {
      _libc_free(v15);
      v8 = *(__int64 **)a2;
    }
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v9;
    *(_QWORD *)(a2 + 8) = 0;
  }
}
