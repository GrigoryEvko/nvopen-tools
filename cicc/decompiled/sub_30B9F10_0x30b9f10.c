// Function: sub_30B9F10
// Address: 0x30b9f10
//
void __fastcall sub_30B9F10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdi
  int v10; // r13d
  __int64 v11; // rsi
  const __m128i *v12; // rax
  __m128i *i; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int64 v20; // rsi
  __int64 v21; // rax

  if ( (__int64 *)a1 != a2 )
  {
    v8 = *((unsigned int *)a2 + 2);
    v9 = *(unsigned int *)(a1 + 8);
    v10 = v8;
    if ( v8 <= v9 )
    {
      if ( v8 )
      {
        v14 = *a2;
        v15 = *(_QWORD *)a1;
        v16 = *a2 + 40 * v8;
        do
        {
          v17 = *(_QWORD *)(v14 + 32);
          v14 += 40;
          v15 += 40;
          *(_QWORD *)(v15 - 8) = v17;
          *(__m128i *)(v15 - 24) = _mm_loadu_si128((const __m128i *)(v14 - 24));
          *(__m128i *)(v15 - 40) = _mm_loadu_si128((const __m128i *)(v14 - 40));
        }
        while ( v16 != v14 );
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_30B9D50(a1, v8, a3, a4, a5, a6);
        v8 = *((unsigned int *)a2 + 2);
        v9 = 0;
      }
      else if ( v9 )
      {
        v18 = *a2;
        v19 = *(_QWORD *)a1;
        v9 *= 40LL;
        v20 = *a2 + v9;
        do
        {
          v21 = *(_QWORD *)(v18 + 32);
          v18 += 40;
          v19 += 40;
          *(_QWORD *)(v19 - 8) = v21;
          *(__m128i *)(v19 - 24) = _mm_loadu_si128((const __m128i *)(v18 - 24));
          *(__m128i *)(v19 - 40) = _mm_loadu_si128((const __m128i *)(v18 - 40));
        }
        while ( v18 != v20 );
        v8 = *((unsigned int *)a2 + 2);
      }
      v11 = *a2 + 40 * v8;
      v12 = (const __m128i *)(v9 + *a2);
      for ( i = (__m128i *)(v9 + *(_QWORD *)a1); (const __m128i *)v11 != v12; i = (__m128i *)((char *)i + 40) )
      {
        if ( i )
        {
          *i = _mm_loadu_si128(v12);
          i[1] = _mm_loadu_si128(v12 + 1);
          i[2].m128i_i64[0] = v12[2].m128i_i64[0];
        }
        v12 = (const __m128i *)((char *)v12 + 40);
      }
    }
    *(_DWORD *)(a1 + 8) = v10;
  }
}
