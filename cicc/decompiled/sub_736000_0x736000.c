// Function: sub_736000
// Address: 0x736000
//
__m128i *__fastcall sub_736000(__int64 a1)
{
  __m128i *result; // rax

  result = sub_735FB0(a1, 3, dword_4F04C5C);
  result[10].m128i_i8[11] |= 0x80u;
  return result;
}
