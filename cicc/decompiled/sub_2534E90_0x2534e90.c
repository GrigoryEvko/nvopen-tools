// Function: sub_2534E90
// Address: 0x2534e90
//
__m128i *__fastcall sub_2534E90(__m128i *a1, __int64 a2)
{
  __m128i v2; // xmm0

  v2 = _mm_loadu_si128((const __m128i *)(a2 + 104));
  a1[1].m128i_i64[0] = *(_QWORD *)(a2 + 120);
  *a1 = v2;
  return a1;
}
