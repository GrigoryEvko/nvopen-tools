// Function: sub_7E20D0
// Address: 0x7e20d0
//
__m128i *__fastcall sub_7E20D0(__int64 a1, int a2)
{
  __m128i *result; // rax

  result = sub_735FB0(a1, (a2 == 0) + 2, -1);
  result[10].m128i_i8[14] |= 4u;
  result[5].m128i_i8[8] &= 0x8Fu;
  return result;
}
