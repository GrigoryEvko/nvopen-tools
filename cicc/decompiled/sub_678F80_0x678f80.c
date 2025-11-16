// Function: sub_678F80
// Address: 0x678f80
//
__m128i *__fastcall sub_678F80(const __m128i *a1)
{
  const __m128i *v1; // r12
  _QWORD *m128i_i64; // r14
  __m128i *v3; // r13
  __m128i v4; // xmm0
  __m128i v5; // xmm3
  __m128i *v6; // rbx

  if ( !a1 )
    return 0;
  v1 = a1;
  m128i_i64 = 0;
  v3 = 0;
  while ( 1 )
  {
    v6 = (__m128i *)qword_4CFDE88;
    if ( qword_4CFDE88 )
      qword_4CFDE88 = *(_QWORD *)qword_4CFDE88;
    else
      v6 = (__m128i *)sub_823970(64);
    v6->m128i_i64[0] = 0;
    v6[3].m128i_i64[0] = 0;
    sub_879020(&v6->m128i_u64[1], 0);
    v6[3].m128i_i64[1] = 0;
    v4 = _mm_loadu_si128(v1);
    if ( !v3 )
      v3 = v6;
    *v6 = v4;
    v6[1] = _mm_loadu_si128(v1 + 1);
    v6[2] = _mm_loadu_si128(v1 + 2);
    v5 = _mm_loadu_si128(v1 + 3);
    v6->m128i_i64[0] = 0;
    v6[3] = v5;
    if ( m128i_i64 )
      *m128i_i64 = v6;
    v1 = (const __m128i *)v1->m128i_i64[0];
    if ( !v1 )
      break;
    m128i_i64 = v6->m128i_i64;
  }
  return v3;
}
