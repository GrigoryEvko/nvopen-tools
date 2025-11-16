// Function: sub_31D6340
// Address: 0x31d6340
//
void __fastcall sub_31D6340(char *src, char *a2)
{
  signed __int64 v2; // rbx

  if ( a2 - src <= 336 )
  {
    sub_31D59B0(src, a2);
  }
  else
  {
    v2 = 8
       * (((0xAAAAAAAAAAAAAAABLL * ((a2 - src) >> 3)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - src) >> 3)) >> 1));
    sub_31D6340(src);
    sub_31D6340(&src[v2]);
    sub_31D61C0(
      src,
      (__m128i *)&src[v2],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 3),
      0xAAAAAAAAAAAAAAABLL * ((a2 - &src[v2]) >> 3));
  }
}
