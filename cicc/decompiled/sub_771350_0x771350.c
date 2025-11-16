// Function: sub_771350
// Address: 0x771350
//
__m128i *__fastcall sub_771350(__int64 a1, int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __m128i *v5; // rax
  __m128i v6; // xmm0
  __m128i *result; // rax

  v5 = (__m128i *)(a1 + 16LL * a5);
  v6 = _mm_loadu_si128(v5);
  v5->m128i_i64[0] = a3;
  v5->m128i_i64[1] = a4;
  do
  {
    a5 = a2 & (a5 + 1);
    result = (__m128i *)(a1 + 16LL * a5);
  }
  while ( result->m128i_i64[0] );
  *result = v6;
  return result;
}
