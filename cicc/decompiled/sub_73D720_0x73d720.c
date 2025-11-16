// Function: sub_73D720
// Address: 0x73d720
//
__m128i *__fastcall sub_73D720(const __m128i *a1)
{
  int v1; // eax

  v1 = dword_4F077C4;
  if ( dword_4F077C4 != 2 || dword_4D03F94 )
    return sub_73D4C0(a1, v1 == 2);
  if ( !(unsigned int)sub_8D3A70(a1) )
  {
    v1 = dword_4F077C4;
    return sub_73D4C0(a1, v1 == 2);
  }
  return (__m128i *)a1;
}
