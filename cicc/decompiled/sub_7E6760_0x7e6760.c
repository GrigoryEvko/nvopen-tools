// Function: sub_7E6760
// Address: 0x7e6760
//
__m128i *__fastcall sub_7E6760(const __m128i *a1, __int64 a2)
{
  __int64 v3; // rax

  if ( !(unsigned int)sub_8D23E0(a2) || (unsigned int)sub_8D23E0(a1) )
    return (__m128i *)a2;
  v3 = sub_8D40F0(a2);
  return sub_73C420(a1, v3);
}
