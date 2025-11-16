// Function: sub_35E5FE0
// Address: 0x35e5fe0
//
void __fastcall sub_35E5FE0(char *a1, char *a2)
{
  signed __int64 v2; // rbx

  if ( a2 - a1 <= 336 )
  {
    sub_35E5200((__int64)a1, a2);
  }
  else
  {
    v2 = 8
       * (((0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3)) >> 1));
    sub_35E5FE0(a1, &a1[v2]);
    sub_35E5FE0(&a1[v2], a2);
    sub_35E5E40(
      a1,
      &a1[v2],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 3),
      0xAAAAAAAAAAAAAAABLL * ((a2 - &a1[v2]) >> 3));
  }
}
