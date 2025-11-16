// Function: sub_740A90
// Address: 0x740a90
//
__m128i *__fastcall sub_740A90(const __m128i *a1, unsigned int a2, _QWORD *a3)
{
  __m128i *v4; // r12
  const __m128i *v5; // r15
  __m128i *v6; // rbx
  _QWORD *m128i_i64; // r13

  v4 = (__m128i *)sub_727590();
  *v4 = _mm_loadu_si128(a1);
  v4[1] = _mm_loadu_si128(a1 + 1);
  v4[2] = _mm_loadu_si128(a1 + 2);
  v4[3] = _mm_loadu_si128(a1 + 3);
  v4[4] = _mm_loadu_si128(a1 + 4);
  v5 = (const __m128i *)a1->m128i_i64[0];
  if ( a1->m128i_i64[0] )
  {
    v6 = 0;
    do
    {
      while ( 1 )
      {
        m128i_i64 = v6->m128i_i64;
        v6 = (__m128i *)sub_7275F0();
        *v6 = _mm_loadu_si128(v5);
        v6[1] = _mm_loadu_si128(v5 + 1);
        v6[2] = _mm_loadu_si128(v5 + 2);
        v6[3].m128i_i64[0] = v5[3].m128i_i64[0];
        if ( (v6[-1].m128i_i8[8] & 1) != 0 && (v5[-1].m128i_i8[8] & 1) == 0 )
          v6->m128i_i64[1] = (__int64)sub_73F780(v5->m128i_i64[1], a2, a3);
        v6->m128i_i64[0] = 0;
        if ( !m128i_i64 )
          break;
        *m128i_i64 = v6;
        v5 = (const __m128i *)v5->m128i_i64[0];
        if ( !v5 )
          return v4;
      }
      v4->m128i_i64[0] = (__int64)v6;
      v5 = (const __m128i *)v5->m128i_i64[0];
    }
    while ( v5 );
  }
  return v4;
}
