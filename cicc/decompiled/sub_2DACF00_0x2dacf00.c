// Function: sub_2DACF00
// Address: 0x2dacf00
//
void __fastcall sub_2DACF00(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
        v16 = *a2 + 24 * v8;
        do
        {
          v17 = *(_QWORD *)(v14 + 16);
          v14 += 24;
          v15 += 24;
          *(_QWORD *)(v15 - 8) = v17;
          *(_QWORD *)(v15 - 16) = *(_QWORD *)(v14 - 16);
          *(_QWORD *)(v15 - 24) = *(_QWORD *)(v14 - 24);
        }
        while ( v16 != v14 );
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_2DACD40(a1, v8, a3, a4, a5, a6);
        v8 = *((unsigned int *)a2 + 2);
        v9 = 0;
      }
      else if ( v9 )
      {
        v18 = *a2;
        v19 = *(_QWORD *)a1;
        v9 *= 24LL;
        v20 = *a2 + v9;
        do
        {
          v21 = *(_QWORD *)(v18 + 16);
          v18 += 24;
          v19 += 24;
          *(_QWORD *)(v19 - 8) = v21;
          *(_QWORD *)(v19 - 16) = *(_QWORD *)(v18 - 16);
          *(_QWORD *)(v19 - 24) = *(_QWORD *)(v18 - 24);
        }
        while ( v18 != v20 );
        v8 = *((unsigned int *)a2 + 2);
      }
      v11 = *a2 + 24 * v8;
      v12 = (const __m128i *)(v9 + *a2);
      for ( i = (__m128i *)(v9 + *(_QWORD *)a1); (const __m128i *)v11 != v12; i = (__m128i *)((char *)i + 24) )
      {
        if ( i )
        {
          *i = _mm_loadu_si128(v12);
          i[1].m128i_i64[0] = v12[1].m128i_i64[0];
        }
        v12 = (const __m128i *)((char *)v12 + 24);
      }
    }
    *(_DWORD *)(a1 + 8) = v10;
  }
}
