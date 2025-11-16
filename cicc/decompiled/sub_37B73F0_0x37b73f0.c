// Function: sub_37B73F0
// Address: 0x37b73f0
//
void __fastcall sub_37B73F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v8; // rax
  const __m128i *v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  int v12; // r14d
  const __m128i *v13; // rsi
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int32 v18; // edx
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  __int32 v21; // edx

  if ( a1 != a2 )
  {
    v8 = *(const __m128i **)a2;
    v9 = (const __m128i *)(a2 + 16);
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
          v17 = a2 + 72 * v10 + 16;
          do
          {
            v18 = v9->m128i_i32[0];
            v9 = (const __m128i *)((char *)v9 + 72);
            v16 += 72LL;
            *(_DWORD *)(v16 - 72) = v18;
            *(__m128i *)(v16 - 64) = _mm_loadu_si128(v9 - 4);
            *(__m128i *)(v16 - 48) = _mm_loadu_si128(v9 - 3);
            *(__m128i *)(v16 - 32) = _mm_loadu_si128(v9 - 2);
            *(_QWORD *)(v16 - 16) = v9[-1].m128i_i64[0];
            *(_DWORD *)(v16 - 8) = v9[-1].m128i_i32[2];
          }
          while ( v9 != (const __m128i *)v17 );
        }
        goto LABEL_8;
      }
      if ( v10 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 0x48u, a5, a6);
        v11 = 0;
        v13 = *(const __m128i **)a2;
        v14 = 72LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 == v14 + *(_QWORD *)a2 )
          goto LABEL_8;
      }
      else
      {
        v13 = (const __m128i *)(a2 + 16);
        if ( *(_DWORD *)(a1 + 8) )
        {
          v19 = *(_QWORD *)a1;
          v20 = 72 * v11;
          v11 *= 72LL;
          do
          {
            v21 = v9->m128i_i32[0];
            v9 = (const __m128i *)((char *)v9 + 72);
            v19 += 72LL;
            *(_DWORD *)(v19 - 72) = v21;
            *(__m128i *)(v19 - 64) = _mm_loadu_si128(v9 - 4);
            *(__m128i *)(v19 - 48) = _mm_loadu_si128(v9 - 3);
            *(__m128i *)(v19 - 32) = _mm_loadu_si128(v9 - 2);
            *(_QWORD *)(v19 - 16) = v9[-1].m128i_i64[0];
            *(_DWORD *)(v19 - 8) = v9[-1].m128i_i32[2];
          }
          while ( (const __m128i *)(a2 + v20 + 16) != v9 );
          v9 = *(const __m128i **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v13 = (const __m128i *)(*(_QWORD *)a2 + v20);
        }
        v14 = 72 * v10;
        if ( v13 == (const __m128i *)&v9->m128i_i8[v14] )
          goto LABEL_8;
      }
      memcpy((void *)(v11 + *(_QWORD *)a1), v13, v14 - v11);
LABEL_8:
      *(_DWORD *)(a1 + 8) = v12;
      *(_DWORD *)(a2 + 8) = 0;
      return;
    }
    v15 = *(_QWORD *)a1;
    if ( v15 != a1 + 16 )
    {
      _libc_free(v15);
      v8 = *(const __m128i **)a2;
    }
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v9;
    *(_QWORD *)(a2 + 8) = 0;
  }
}
