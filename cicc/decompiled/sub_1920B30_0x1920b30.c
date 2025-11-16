// Function: sub_1920B30
// Address: 0x1920b30
//
void __fastcall sub_1920B30(char *a1, __int64 a2)
{
  signed __int64 v2; // rbx

  if ( a2 - (__int64)a1 <= 336 )
  {
    sub_1920210((__int64)a1, a2);
  }
  else
  {
    v2 = 8
       * (((0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3)) >> 1));
    sub_1920B30(a1, &a1[v2]);
    sub_1920B30(&a1[v2], a2);
    sub_1920970(
      a1,
      &a1[v2],
      a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 3),
      0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)&a1[v2]) >> 3));
  }
}
