// Function: sub_73D660
// Address: 0x73d660
//
__m128i *__fastcall sub_73D660(const __m128i *a1, int a2)
{
  unsigned int v3; // r13d
  __m128i *v4; // rax

  if ( (a1[8].m128i_i8[12] & 0xFB) != 8 )
    return (__m128i *)a1;
  v3 = sub_8D4C10(a1, dword_4F077C4 != 2);
  if ( (a2 & v3) == 0 )
    return (__m128i *)a1;
  v4 = sub_73D4C0(a1, dword_4F077C4 == 2);
  return sub_73C570(v4, ~a2 & v3);
}
