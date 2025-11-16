// Function: sub_8921F0
// Address: 0x8921f0
//
__m128i *__fastcall sub_8921F0(__int64 a1)
{
  const __m128i *v1; // r12
  __m128i *result; // rax

  v1 = *(const __m128i **)(a1 + 96);
  if ( v1[1].m128i_i8[0] == 53 )
    return (__m128i *)v1[1].m128i_i64[1];
  result = (__m128i *)sub_727110();
  result[1] = _mm_loadu_si128(v1 + 1);
  result->m128i_i64[0] = *(_QWORD *)(a1 + 64);
  v1[1].m128i_i64[1] = (__int64)result;
  v1[1].m128i_i8[0] = 53;
  return result;
}
