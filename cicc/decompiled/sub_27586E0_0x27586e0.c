// Function: sub_27586E0
// Address: 0x27586e0
//
void __fastcall sub_27586E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
          *(_QWORD *)(v15 - 24) = *(_QWORD *)(v14 - 24);
          *(_DWORD *)(v15 - 16) = *(_DWORD *)(v14 - 16);
          *(_QWORD *)(v15 - 40) = *(_QWORD *)(v14 - 40);
          *(_DWORD *)(v15 - 32) = *(_DWORD *)(v14 - 32);
        }
        while ( v16 != v14 );
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_CE3550(a1, v8, a3, a4, a5, a6);
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
          *(_QWORD *)(v19 - 24) = *(_QWORD *)(v18 - 24);
          *(_DWORD *)(v19 - 16) = *(_DWORD *)(v18 - 16);
          *(_QWORD *)(v19 - 40) = *(_QWORD *)(v18 - 40);
          *(_DWORD *)(v19 - 32) = *(_DWORD *)(v18 - 32);
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
