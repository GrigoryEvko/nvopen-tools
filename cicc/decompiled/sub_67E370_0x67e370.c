// Function: sub_67E370
// Address: 0x67e370
//
void __fastcall sub_67E370(__int64 a1, const __m128i *a2)
{
  *(__m128i *)(a1 + 72) = _mm_loadu_si128(a2);
  a2->m128i_i64[0] = 0;
  a2->m128i_i64[1] = 0;
}
