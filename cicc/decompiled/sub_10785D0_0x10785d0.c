// Function: sub_10785D0
// Address: 0x10785d0
//
void __fastcall sub_10785D0(__m128i *src, char *a2)
{
  __int64 v2; // rbx

  if ( a2 - (char *)src <= 560 )
  {
    sub_10776B0(src->m128i_i8, a2);
  }
  else
  {
    v2 = 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * ((a2 - (char *)src) >> 3)) >> 1);
    sub_10785D0(src);
    sub_10785D0(&src->m128i_i8[v2]);
    sub_1078420(
      src,
      (__m128i *)((char *)src + v2),
      (__int64)a2,
      0xCCCCCCCCCCCCCCCDLL * (v2 >> 3),
      0xCCCCCCCCCCCCCCCDLL * ((a2 - &src->m128i_i8[v2]) >> 3));
  }
}
