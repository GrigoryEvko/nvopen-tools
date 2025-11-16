// Function: sub_1888AF0
// Address: 0x1888af0
//
void __fastcall sub_1888AF0(__int64 a1, _QWORD *a2)
{
  signed __int64 v2; // rbx

  if ( (__int64)a2 - a1 <= 672 )
  {
    sub_18772A0(a1, a2);
  }
  else
  {
    v2 = 16
       * (((0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 4)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 4)) >> 1));
    sub_1888AF0(a1, a1 + v2);
    sub_1888AF0(a1 + v2, a2);
    sub_1888960(
      a1,
      a1 + v2,
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * (v2 >> 4),
      0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1 - v2) >> 4));
  }
}
