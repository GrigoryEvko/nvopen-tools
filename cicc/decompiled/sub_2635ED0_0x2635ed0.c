// Function: sub_2635ED0
// Address: 0x2635ed0
//
__int64 __fastcall sub_2635ED0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  signed __int64 v7; // rbx
  __int64 v9; // [rsp-10h] [rbp-50h]

  v6 = (__int64)(0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-a1] >> 4) + 1) / 2;
  v7 = 16
     * (v6
      + ((0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-a1] >> 4)
        + 1
        + ((0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-a1] >> 4) + 1) >> 63))
       & 0xFFFFFFFFFFFFFFFELL));
  if ( v6 <= a4 )
  {
    sub_261F2A0(a1, (char *)(a1 + v7), a3);
    sub_261F2A0(a1 + v7, a2, a3);
  }
  else
  {
    sub_2635ED0(a1, a1 + v7);
    sub_2635ED0(a1 + v7, a2);
  }
  sub_26353C0(
    a1,
    a1 + v7,
    (__int64)a2,
    0xAAAAAAAAAAAAAAABLL * (v7 >> 4),
    0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-a1 - v7] >> 4),
    a3,
    a4);
  return v9;
}
