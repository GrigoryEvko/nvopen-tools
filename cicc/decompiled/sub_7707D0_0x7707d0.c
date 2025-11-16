// Function: sub_7707D0
// Address: 0x7707d0
//
__m128i *__fastcall sub_7707D0(__int64 a1, const __m128i *a2)
{
  __m128i *v2; // r12
  const __m128i *v4; // rdi

  v2 = (__m128i *)sub_724DC0();
  *v2 = _mm_loadu_si128(a2);
  v2[1] = _mm_loadu_si128(a2 + 1);
  v2[2] = _mm_loadu_si128(a2 + 2);
  v2[3] = _mm_loadu_si128(a2 + 3);
  v2[4] = _mm_loadu_si128(a2 + 4);
  v2[5] = _mm_loadu_si128(a2 + 5);
  v2[6] = _mm_loadu_si128(a2 + 6);
  v2[7] = _mm_loadu_si128(a2 + 7);
  v2[8] = _mm_loadu_si128(a2 + 8);
  v2[9] = _mm_loadu_si128(a2 + 9);
  v2[10] = _mm_loadu_si128(a2 + 10);
  v2[11] = _mm_loadu_si128(a2 + 11);
  v2[12] = _mm_loadu_si128(a2 + 12);
  v2[7].m128i_i64[1] = *(_QWORD *)(a1 + 88);
  *(_QWORD *)(a1 + 88) = v2;
  if ( v2[10].m128i_i8[13] != 6 )
    return v2;
  v4 = (const __m128i *)v2[12].m128i_i64[1];
  if ( !v4 )
    return v2;
  v2[12].m128i_i64[1] = sub_72A820(v4);
  return v2;
}
