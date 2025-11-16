// Function: sub_880B70
// Address: 0x880b70
//
__m128i *__fastcall sub_880B70(const __m128i *a1, __int64 a2)
{
  __m128i *result; // rax
  __int64 v3; // rdx

  result = (__m128i *)sub_880AD0(a2);
  *result = _mm_loadu_si128(a1);
  result[1] = _mm_loadu_si128(a1 + 1);
  result[2] = _mm_loadu_si128(a1 + 2);
  result[3] = _mm_loadu_si128(a1 + 3);
  result[4] = _mm_loadu_si128(a1 + 4);
  result[5] = _mm_loadu_si128(a1 + 5);
  result[6] = _mm_loadu_si128(a1 + 6);
  result[7] = _mm_loadu_si128(a1 + 7);
  v3 = a1[8].m128i_i64[0];
  result[3].m128i_i8[9] &= ~1u;
  result[8].m128i_i64[0] = v3;
  result->m128i_i64[0] = 0;
  result[5].m128i_i64[1] = 0;
  result->m128i_i64[1] = a2;
  result[4].m128i_i64[0] = *(_QWORD *)(a2 + 88);
  return result;
}
