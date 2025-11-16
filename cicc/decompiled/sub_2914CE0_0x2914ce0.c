// Function: sub_2914CE0
// Address: 0x2914ce0
//
void __fastcall sub_2914CE0(__m128i *src, const __m128i *a2)
{
  signed __int64 v2; // rbx

  if ( (char *)a2 - (char *)src <= 336 )
  {
    sub_2913170(src, a2);
  }
  else
  {
    v2 = 8
       * (((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)src) >> 3)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)src) >> 3)) >> 1));
    sub_2914CE0(src);
    sub_2914CE0(&src->m128i_i8[v2]);
    sub_2914B10(
      src,
      (__m128i *)((char *)src + v2),
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 3),
      0xAAAAAAAAAAAAAAABLL * (((char *)a2 - &src->m128i_i8[v2]) >> 3));
  }
}
