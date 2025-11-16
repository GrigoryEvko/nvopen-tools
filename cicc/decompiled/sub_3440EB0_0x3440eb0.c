// Function: sub_3440EB0
// Address: 0x3440eb0
//
void __fastcall sub_3440EB0(__int64 a1, __int64 a2)
{
  signed __int64 v2; // rbx

  if ( a2 - a1 <= 336 )
  {
    sub_3440410(a1, a2);
  }
  else
  {
    v2 = 8
       * (((0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3)) >> 1));
    sub_3440EB0(a1, a1 + v2);
    sub_3440EB0(a1 + v2, a2);
    sub_3440CF0(
      a1,
      (__m128i *)(a1 + v2),
      a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 3),
      0xAAAAAAAAAAAAAAABLL * ((a2 - (a1 + v2)) >> 3));
  }
}
