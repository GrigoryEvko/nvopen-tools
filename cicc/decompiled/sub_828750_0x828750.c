// Function: sub_828750
// Address: 0x828750
//
__m128i *__fastcall sub_828750(__m128i **a1, const __m128i **a2)
{
  const __m128i *v3; // rsi
  __m128i *result; // rax

  sub_73E8D0(a1, a2);
  v3 = *a2;
  result = (__m128i *)((*a2)[8].m128i_i8[12] & 0xFB);
  if ( ((*a2)[8].m128i_i8[12] & 0xFB) == 8 )
  {
    result = (__m128i *)sub_8D5780(*a1, v3);
    if ( (_DWORD)result )
    {
      result = sub_73D4C0(*a2, dword_4F077C4 == 2);
      *a2 = result;
    }
  }
  return result;
}
