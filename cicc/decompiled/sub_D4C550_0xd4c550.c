// Function: sub_D4C550
// Address: 0xd4c550
//
void __fastcall sub_D4C550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v8; // rax
  const __m128i *v9; // rbx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  int v12; // r14d
  unsigned __int64 v13; // rdx
  const __m128i *v14; // rdx
  __m128i *v15; // rax
  const __m128i *i; // rsi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rsi

  if ( a1 != a2 )
  {
    v8 = *(const __m128i **)a2;
    v9 = (const __m128i *)(a2 + 16);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = v10;
      if ( v10 <= v11 )
      {
        if ( v10 )
        {
          v18 = *(_QWORD *)a1;
          do
          {
            v19 = v9[2].m128i_i64[0];
            v9 = (const __m128i *)((char *)v9 + 40);
            v18 += 40;
            *(_QWORD *)(v18 - 8) = v19;
            *(_QWORD *)(v18 - 24) = v9[-2].m128i_i64[1];
            *(_DWORD *)(v18 - 16) = v9[-1].m128i_i32[0];
            *(_QWORD *)(v18 - 40) = v9[-3].m128i_i64[1];
            *(_DWORD *)(v18 - 32) = v9[-2].m128i_i32[0];
          }
          while ( v9 != (const __m128i *)(a2 + 40 * v10 + 16) );
        }
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 12);
        if ( v10 > v13 )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_CE3550(a1, v10, v13, a4, a5, a6);
          v9 = *(const __m128i **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v11 = 0;
          v14 = *(const __m128i **)a2;
        }
        else
        {
          v14 = v9;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v20 = *(_QWORD *)a1;
            v21 = 40 * v11;
            v11 *= 40LL;
            do
            {
              v22 = v9[2].m128i_i64[0];
              v9 = (const __m128i *)((char *)v9 + 40);
              v20 += 40;
              *(_QWORD *)(v20 - 8) = v22;
              *(_QWORD *)(v20 - 24) = v9[-2].m128i_i64[1];
              *(_DWORD *)(v20 - 16) = v9[-1].m128i_i32[0];
              *(_QWORD *)(v20 - 40) = v9[-3].m128i_i64[1];
              *(_DWORD *)(v20 - 32) = v9[-2].m128i_i32[0];
            }
            while ( v9 != (const __m128i *)(a2 + v21 + 16) );
            v9 = *(const __m128i **)a2;
            v10 = *(unsigned int *)(a2 + 8);
            v14 = (const __m128i *)(*(_QWORD *)a2 + v21);
          }
        }
        v15 = (__m128i *)(*(_QWORD *)a1 + v11);
        for ( i = (const __m128i *)((char *)v9 + 40 * v10); i != v14; v15 = (__m128i *)((char *)v15 + 40) )
        {
          if ( v15 )
          {
            *v15 = _mm_loadu_si128(v14);
            v15[1] = _mm_loadu_si128(v14 + 1);
            v15[2].m128i_i64[0] = v14[2].m128i_i64[0];
          }
          v14 = (const __m128i *)((char *)v14 + 40);
        }
      }
      *(_DWORD *)(a1 + 8) = v12;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = *(_QWORD *)a1;
      if ( v17 != a1 + 16 )
      {
        _libc_free(v17, a2);
        v8 = *(const __m128i **)a2;
      }
      *(_QWORD *)a1 = v8;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v9;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
