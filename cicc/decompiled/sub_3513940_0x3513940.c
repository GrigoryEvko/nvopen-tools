// Function: sub_3513940
// Address: 0x3513940
//
void __fastcall sub_3513940(__int64 a1)
{
  __m128i *v1; // r13
  __int64 v2; // rax
  __m128i *v3; // r15
  __int64 v4; // r14
  __m128i *v5; // rax
  __m128i *v6; // r12
  __m128i *v7; // rcx
  __m128i *v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rsi
  const __m128i *v11; // rax

  v1 = *(__m128i **)a1;
  v2 = 24LL * *(unsigned int *)(a1 + 8);
  v3 = (__m128i *)(*(_QWORD *)a1 + v2);
  v4 = 0xAAAAAAAAAAAAAAABLL * (v2 >> 3);
  if ( v2 )
  {
    while ( 1 )
    {
      v5 = (__m128i *)sub_2207800(24 * v4);
      v6 = v5;
      if ( v5 )
        break;
      v4 >>= 1;
      if ( !v4 )
        goto LABEL_9;
    }
    v7 = (__m128i *)((char *)v5 + 24 * v4);
    *v5 = _mm_loadu_si128(v1);
    v5[1].m128i_i64[0] = v1[1].m128i_i64[0];
    v8 = (__m128i *)((char *)v5 + 24);
    if ( v7 == (__m128i *)&v6[1].m128i_u64[1] )
    {
      v11 = v6;
    }
    else
    {
      do
      {
        v9 = _mm_loadu_si128((__m128i *)((char *)v8 - 24));
        v10 = v8[-1].m128i_i64[1];
        v8 = (__m128i *)((char *)v8 + 24);
        *(__m128i *)((char *)v8 - 24) = v9;
        v8[-1].m128i_i64[1] = v10;
      }
      while ( v7 != v8 );
      v11 = (__m128i *)((char *)v6 + 24 * v4 - 24);
    }
    *v1 = _mm_loadu_si128(v11);
    v1[1].m128i_i64[0] = v11[1].m128i_i64[0];
    sub_3513850(v1, v3, v6, (const __m128i *)v4);
  }
  else
  {
LABEL_9:
    v6 = 0;
    sub_3512A80((unsigned __int64 *)v1, (unsigned __int64 *)v3);
  }
  j_j___libc_free_0((unsigned __int64)v6);
}
