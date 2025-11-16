// Function: sub_886210
// Address: 0x886210
//
__m128i *__fastcall sub_886210(const __m128i *a1, int a2, int a3)
{
  __m128i *v4; // r12
  __m128i v5; // xmm6

  v4 = (__m128i *)sub_87EBB0(a1[5].m128i_u8[0], a1->m128i_i64[0], (const __m128i *)a1[3].m128i_i64);
  *v4 = _mm_loadu_si128(a1);
  v4[1] = _mm_loadu_si128(a1 + 1);
  v4[2] = _mm_loadu_si128(a1 + 2);
  v4[3] = _mm_loadu_si128(a1 + 3);
  v4[4] = _mm_loadu_si128(a1 + 4);
  v4[5] = _mm_loadu_si128(a1 + 5);
  v5 = _mm_loadu_si128(a1 + 6);
  v4->m128i_i64[1] = 0;
  v4[1].m128i_i64[0] = 0;
  v4[1].m128i_i64[1] = 0;
  v4[6] = v5;
  sub_885A00((__int64)v4, a2, a3);
  return v4;
}
