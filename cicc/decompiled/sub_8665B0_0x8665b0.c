// Function: sub_8665B0
// Address: 0x8665b0
//
__m128i *__fastcall sub_8665B0(const __m128i *a1)
{
  __m128i *result; // rax
  __int64 v2; // rdx

  result = (__m128i *)sub_866270(a1[2].m128i_i32[0]);
  *result = _mm_loadu_si128(a1);
  result[1] = _mm_loadu_si128(a1 + 1);
  result[2] = _mm_loadu_si128(a1 + 2);
  result[3] = _mm_loadu_si128(a1 + 3);
  result[4] = _mm_loadu_si128(a1 + 4);
  result[5] = _mm_loadu_si128(a1 + 5);
  v2 = a1[6].m128i_i64[0];
  result->m128i_i64[0] = 0;
  result[6].m128i_i64[0] = v2;
  return result;
}
