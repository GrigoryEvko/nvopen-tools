// Function: sub_8E32F0
// Address: 0x8e32f0
//
__m128i *__fastcall sub_8E32F0(__m128i *a1)
{
  if ( a1[8].m128i_i8[12] == 7 )
    a1 = sub_73F380(a1);
  return sub_8DC200((__int64)a1, (unsigned int (__fastcall *)(__m128i *, _QWORD, __m128i **))sub_8E3320, 0x10u);
}
