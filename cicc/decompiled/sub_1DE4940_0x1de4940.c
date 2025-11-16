// Function: sub_1DE4940
// Address: 0x1de4940
//
void __fastcall sub_1DE4940(unsigned __int64 *src, unsigned __int64 *a2)
{
  signed __int64 v2; // rbx

  if ( (char *)a2 - (char *)src <= 336 )
  {
    sub_1DE3E80(src, a2);
  }
  else
  {
    v2 = ((0xAAAAAAAAAAAAAAABLL * (a2 - src)) & 0xFFFFFFFFFFFFFFFELL)
       + ((__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - src)) >> 1);
    sub_1DE4940(src);
    sub_1DE4940(&src[v2]);
    sub_1DE47C0(
      (char *)src,
      (__m128i *)&src[v2],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * ((v2 * 8) >> 3),
      0xAAAAAAAAAAAAAAABLL * (a2 - &src[v2]));
  }
}
