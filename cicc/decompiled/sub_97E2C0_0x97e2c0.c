// Function: sub_97E2C0
// Address: 0x97e2c0
//
void __fastcall sub_97E2C0(__int64 a1, const __m128i **a2, __int64 a3, __int64 a4)
{
  const __m128i *v5; // r13
  const __m128i *v6; // r12
  __m128i *v7; // rdi
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // r8
  __m128i *v10; // rax
  size_t v11; // rdx
  __int8 *v12; // r15
  __int64 v13; // rax
  __m128i *v14; // r14
  const __m128i *v15; // rsi
  __m128i *v16; // rax
  __m128i *v17; // r13
  const __m128i *v18; // rdx
  __m128i *v19; // r13

  if ( a2 != (const __m128i **)a1 )
  {
    v5 = a2[1];
    v6 = *a2;
    v7 = *(__m128i **)a1;
    v8 = (char *)v5 - (char *)*a2;
    v9 = *(_QWORD *)(a1 + 16) - (_QWORD)v7;
    if ( v8 > v9 )
    {
      if ( v8 )
      {
        if ( v8 > 0x7FFFFFFFFFFFFFC0LL )
          sub_4261EA(v7, a2, a3, a4);
        v13 = sub_22077B0((char *)a2[1] - (char *)*a2);
        v7 = *(__m128i **)a1;
        v14 = (__m128i *)v13;
        v9 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
      }
      else
      {
        v14 = 0;
      }
      if ( v5 != v6 )
      {
        v15 = v6;
        v16 = v14;
        v17 = (__m128i *)((char *)v14 + (char *)v5 - (char *)v6);
        do
        {
          if ( v16 )
          {
            *v16 = _mm_loadu_si128(v15);
            v16[1] = _mm_loadu_si128(v15 + 1);
            v16[2] = _mm_loadu_si128(v15 + 2);
            v16[3] = _mm_loadu_si128(v15 + 3);
          }
          v16 += 4;
          v15 += 4;
        }
        while ( v16 != v17 );
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
        memmove(v7, *a2, v11);
        v10 = *(__m128i **)(a1 + 8);
        v7 = *(__m128i **)a1;
        v5 = a2[1];
        v6 = *a2;
        v11 = (size_t)v10 - *(_QWORD *)a1;
      }
      v18 = (const __m128i *)((char *)v6 + v11);
      if ( v18 != v5 )
      {
        v19 = (__m128i *)((char *)v10 + (char *)v5 - (char *)v18);
        do
        {
          if ( v10 )
          {
            *v10 = _mm_loadu_si128(v18);
            v10[1] = _mm_loadu_si128(v18 + 1);
            v10[2] = _mm_loadu_si128(v18 + 2);
            v10[3] = _mm_loadu_si128(v18 + 3);
          }
          v10 += 4;
          v18 += 4;
        }
        while ( v19 != v10 );
        v12 = (__int8 *)(*(_QWORD *)a1 + v8);
        goto LABEL_7;
      }
    }
    else if ( v5 != v6 )
    {
      memmove(v7, *a2, (char *)a2[1] - (char *)*a2);
      v7 = *(__m128i **)a1;
    }
    v12 = &v7->m128i_i8[v8];
LABEL_7:
    *(_QWORD *)(a1 + 8) = v12;
  }
}
