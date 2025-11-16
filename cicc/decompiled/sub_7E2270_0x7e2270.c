// Function: sub_7E2270
// Address: 0x7e2270
//
__m128i *__fastcall sub_7E2270(__int64 a1)
{
  __m128i *result; // rax

  result = sub_735FB0(a1, 3, -1);
  result[5].m128i_i8[8] &= 0x8Fu;
  result[10].m128i_i64[1] |= 0x4000000008000uLL;
  return result;
}
