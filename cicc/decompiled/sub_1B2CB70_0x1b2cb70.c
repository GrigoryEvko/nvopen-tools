// Function: sub_1B2CB70
// Address: 0x1b2cb70
//
void __fastcall sub_1B2CB70(__m128i *src, const __m128i *a2, __int64 a3)
{
  signed __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 672 )
  {
    sub_1B2C670(src, a2, a3);
  }
  else
  {
    v4 = ((0xAAAAAAAAAAAAAAABLL * (a2 - src)) & 0xFFFFFFFFFFFFFFFELL)
       + ((__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - src)) >> 1);
    sub_1B2CB70(src);
    sub_1B2CB70(&src[v4]);
    sub_1B2C980(
      (__int64)src,
      &src[v4],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * ((v4 * 16) >> 4),
      0xAAAAAAAAAAAAAAABLL * (a2 - &src[v4]),
      a3);
  }
}
