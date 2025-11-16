// Function: sub_72F240
// Address: 0x72f240
//
__m128i *__fastcall sub_72F240(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __m128i *v2; // r13
  _QWORD *m128i_i64; // r12
  __int64 v4; // rax
  __m128i *v5; // rax
  __int64 v6; // rdx

  if ( !a1 )
    return 0;
  v1 = a1;
  v2 = (__m128i *)sub_725090(a1->m128i_u8[8]);
  *v2 = _mm_loadu_si128(a1);
  m128i_i64 = v2->m128i_i64;
  v2[1] = _mm_loadu_si128(a1 + 1);
  v2[2] = _mm_loadu_si128(a1 + 2);
  v4 = a1[3].m128i_i64[0];
  v2->m128i_i64[0] = 0;
  v2[3].m128i_i64[0] = v4;
  while ( 1 )
  {
    v1 = (const __m128i *)v1->m128i_i64[0];
    if ( !v1 )
      break;
    v5 = (__m128i *)sub_725090(v1->m128i_u8[8]);
    *v5 = _mm_loadu_si128(v1);
    v5[1] = _mm_loadu_si128(v1 + 1);
    v5[2] = _mm_loadu_si128(v1 + 2);
    v6 = v1[3].m128i_i64[0];
    v5->m128i_i64[0] = 0;
    v5[3].m128i_i64[0] = v6;
    *m128i_i64 = v5;
    m128i_i64 = v5->m128i_i64;
  }
  return v2;
}
