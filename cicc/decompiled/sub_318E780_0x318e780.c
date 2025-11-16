// Function: sub_318E780
// Address: 0x318e780
//
__m128i *__fastcall sub_318E780(__m128i *a1, const __m128i *a2)
{
  __m128i v2; // xmm0

  v2 = _mm_loadu_si128(a2);
  a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
  *a1 = v2;
  return a1;
}
