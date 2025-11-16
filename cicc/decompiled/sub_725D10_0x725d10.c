// Function: sub_725D10
// Address: 0x725d10
//
__m128i *__fastcall sub_725D10(__int8 a1)
{
  __m128i *v1; // r12

  if ( (unsigned __int8)a1 > 2u )
    v1 = (__m128i *)sub_7246D0(272);
  else
    v1 = (__m128i *)sub_7247C0(272);
  sub_725B90(v1);
  v1[8].m128i_i8[8] = a1;
  return v1;
}
