// Function: sub_2635310
// Address: 0x2635310
//
void __fastcall sub_2635310(__int64 a1, _QWORD *a2)
{
  signed __int64 v2; // rbx

  if ( (__int64)a2 - a1 <= 672 )
  {
    sub_261EEF0(a1, a2);
  }
  else
  {
    v2 = 16
       * (((0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 4)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 4)) >> 1));
    sub_2635310(a1, a1 + v2);
    sub_2635310(a1 + v2, a2);
    sub_2635180(
      a1,
      a1 + v2,
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 4),
      0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1 - v2) >> 4));
  }
}
