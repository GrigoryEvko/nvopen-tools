// Function: sub_853ED0
// Address: 0x853ed0
//
__m128i *__fastcall sub_853ED0(const __m128i *a1)
{
  const __m128i *v1; // rbx
  _QWORD *m128i_i64; // r13
  __m128i *v3; // r12
  __int64 v4; // rdx
  __m128i *v5; // rax

  if ( !a1 )
    return 0;
  v1 = a1;
  m128i_i64 = 0;
  v3 = 0;
  while ( 1 )
  {
    v5 = (__m128i *)unk_4D03D28;
    if ( unk_4D03D28 )
      unk_4D03D28 = *unk_4D03D28;
    else
      v5 = (__m128i *)sub_823970(104);
    *v5 = _mm_loadu_si128(v1);
    v5[1] = _mm_loadu_si128(v1 + 1);
    v5[2] = _mm_loadu_si128(v1 + 2);
    v5[3] = _mm_loadu_si128(v1 + 3);
    v5[4] = _mm_loadu_si128(v1 + 4);
    v5[5] = _mm_loadu_si128(v1 + 5);
    v4 = v1[6].m128i_i64[0];
    v5[4].m128i_i8[8] &= 0xF6u;
    v5[6].m128i_i64[0] = v4;
    if ( !v3 )
      v3 = v5;
    v5->m128i_i64[0] = 0;
    if ( m128i_i64 )
      *m128i_i64 = v5;
    v1 = (const __m128i *)v1->m128i_i64[0];
    if ( !v1 )
      break;
    m128i_i64 = v5->m128i_i64;
  }
  return v3;
}
