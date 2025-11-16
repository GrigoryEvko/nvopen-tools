// Function: sub_E7C160
// Address: 0xe7c160
//
void __fastcall sub_E7C160(__int64 *a1, __int64 a2)
{
  signed __int64 v2; // rbx

  if ( a2 - (__int64)a1 <= 1344 )
  {
    sub_E733B0((__int64)a1, a2);
  }
  else
  {
    v2 = 4
       * (((0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 5)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 5)) >> 1));
    sub_E7C160(a1, &a1[v2]);
    sub_E7C160(&a1[v2], a2);
    sub_E7BF80(
      a1,
      &a1[v2],
      a2,
      0xAAAAAAAAAAAAAAABLL * ((v2 * 8) >> 5),
      0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)&a1[v2]) >> 5));
  }
}
