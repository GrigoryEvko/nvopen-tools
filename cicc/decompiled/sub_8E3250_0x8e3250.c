// Function: sub_8E3250
// Address: 0x8e3250
//
__m128i *__fastcall sub_8E3250(const __m128i *a1)
{
  if ( a1[8].m128i_i8[12] == 7 )
    return sub_73F2D0(a1);
  else
    return sub_8DC200((__int64)a1, (unsigned int (__fastcall *)(__m128i *, _QWORD, __m128i **))sub_8E3280, 0x10u);
}
