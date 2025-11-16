// Function: sub_1E45130
// Address: 0x1e45130
//
void __fastcall sub_1E45130(char *a1, __int64 a2)
{
  signed __int64 v2; // rbx

  if ( a2 - (__int64)a1 <= 1344 )
  {
    sub_1E44500((__int64)a1, a2);
  }
  else
  {
    v2 = 32
       * (((0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 5)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 5)) >> 1));
    sub_1E45130(a1, &a1[v2]);
    sub_1E45130(&a1[v2], a2);
    sub_1E44D60(
      a1,
      &a1[v2],
      a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 5),
      0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)&a1[v2]) >> 5));
  }
}
