// Function: sub_3512A80
// Address: 0x3512a80
//
void __fastcall sub_3512A80(unsigned __int64 *src, unsigned __int64 *a2)
{
  signed __int64 v2; // rbx

  if ( (char *)a2 - (char *)src <= 336 )
  {
    sub_3510CA0(src, a2);
  }
  else
  {
    v2 = ((0xAAAAAAAAAAAAAAABLL * (a2 - src)) & 0xFFFFFFFFFFFFFFFELL)
       + ((__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - src)) >> 1);
    sub_3512A80(src);
    sub_3512A80(&src[v2]);
    sub_3512900(
      (char *)src,
      (__m128i *)&src[v2],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * ((v2 * 8) >> 3),
      0xAAAAAAAAAAAAAAABLL * (a2 - &src[v2]));
  }
}
