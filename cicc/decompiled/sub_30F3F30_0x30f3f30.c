// Function: sub_30F3F30
// Address: 0x30f3f30
//
void __fastcall sub_30F3F30(__int64 a1, __int64 *a2)
{
  signed __int64 v2; // rbx

  if ( (__int64)a2 - a1 <= 336 )
  {
    sub_30F3AA0(a1, a2);
  }
  else
  {
    v2 = 8
       * (((0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 3)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 3)) >> 1));
    sub_30F3F30(a1, a1 + v2);
    sub_30F3F30(a1 + v2, a2);
    sub_30F3D90(
      a1,
      a1 + v2,
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 3),
      0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1 - v2) >> 3));
  }
}
