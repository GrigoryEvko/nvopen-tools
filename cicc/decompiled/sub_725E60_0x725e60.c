// Function: sub_725E60
// Address: 0x725e60
//
__m128i *sub_725E60()
{
  __m128i *result; // rax

  result = (__m128i *)sub_7247C0(32);
  result->m128i_i8[0] &= 0x80u;
  result->m128i_i64[1] = 0;
  result[1] = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  return result;
}
