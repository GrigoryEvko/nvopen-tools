// Function: sub_2A47830
// Address: 0x2a47830
//
void __fastcall sub_2A47830(__m128i *src, const __m128i *a2, __int64 a3)
{
  signed __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 672 )
  {
    sub_2A45D80(src, a2, a3);
  }
  else
  {
    v4 = ((0xAAAAAAAAAAAAAAABLL * (a2 - src)) & 0xFFFFFFFFFFFFFFFELL)
       + ((__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - src)) >> 1);
    sub_2A47830(src);
    sub_2A47830(&src[v4]);
    sub_2A47640(
      (__int64)src,
      &src[v4],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * ((v4 * 16) >> 4),
      0xAAAAAAAAAAAAAAABLL * (a2 - &src[v4]),
      a3);
  }
}
