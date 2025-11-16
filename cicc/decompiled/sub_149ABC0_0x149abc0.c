// Function: sub_149ABC0
// Address: 0x149abc0
//
void __fastcall sub_149ABC0(__int64 a1, const __m128i **a2, __int64 a3)
{
  const __m128i *v5; // r13
  const __m128i *v6; // r12
  __m128i *v7; // rdi
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rsi
  __m128i *v10; // rax
  size_t v11; // rdx
  __int8 *v12; // r15
  __int64 v13; // rax
  __m128i *v14; // r14
  __m128i *i; // rax
  const __m128i *v16; // r12

  if ( a2 != (const __m128i **)a1 )
  {
    v5 = a2[1];
    v6 = *a2;
    v7 = *(__m128i **)a1;
    v8 = (char *)v5 - (char *)v6;
    v9 = *(_QWORD *)(a1 + 16) - (_QWORD)v7;
    if ( (char *)v5 - (char *)v6 > v9 )
    {
      if ( v8 )
      {
        if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(v7, v9, a3);
        v13 = sub_22077B0((char *)v5 - (char *)v6);
        v7 = *(__m128i **)a1;
        v14 = (__m128i *)v13;
        v9 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
      }
      else
      {
        v14 = 0;
      }
      for ( i = v14; v5 != v6; i = (__m128i *)((char *)i + 40) )
      {
        if ( i )
        {
          *i = _mm_loadu_si128(v6);
          i[1] = _mm_loadu_si128(v6 + 1);
          i[2].m128i_i64[0] = v6[2].m128i_i64[0];
        }
        v6 = (const __m128i *)((char *)v6 + 40);
      }
      if ( v7 )
        j_j___libc_free_0(v7, v9);
      v12 = &v14->m128i_i8[v8];
      *(_QWORD *)a1 = v14;
      *(_QWORD *)(a1 + 16) = v12;
      goto LABEL_7;
    }
    v10 = *(__m128i **)(a1 + 8);
    v11 = (char *)v10 - (char *)v7;
    if ( v8 > (char *)v10 - (char *)v7 )
    {
      if ( v11 )
      {
        memmove(v7, v6, v11);
        v10 = *(__m128i **)(a1 + 8);
        v7 = *(__m128i **)a1;
        v5 = a2[1];
        v6 = *a2;
        v11 = (size_t)v10 - *(_QWORD *)a1;
      }
      v16 = (const __m128i *)((char *)v6 + v11);
      if ( v16 != v5 )
      {
        do
        {
          if ( v10 )
          {
            *v10 = _mm_loadu_si128(v16);
            v10[1] = _mm_loadu_si128(v16 + 1);
            v10[2].m128i_i64[0] = v16[2].m128i_i64[0];
          }
          v16 = (const __m128i *)((char *)v16 + 40);
          v10 = (__m128i *)((char *)v10 + 40);
        }
        while ( v16 != v5 );
        v12 = (__int8 *)(*(_QWORD *)a1 + v8);
        goto LABEL_7;
      }
    }
    else if ( v5 != v6 )
    {
      memmove(v7, v6, (char *)v5 - (char *)v6);
      v7 = *(__m128i **)a1;
    }
    v12 = &v7->m128i_i8[v8];
LABEL_7:
    *(_QWORD *)(a1 + 8) = v12;
  }
}
