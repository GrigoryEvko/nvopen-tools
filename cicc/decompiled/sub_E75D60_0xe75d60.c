// Function: sub_E75D60
// Address: 0xe75d60
//
__int64 __fastcall sub_E75D60(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // rax
  signed __int64 v7; // rbx
  __int64 v9; // [rsp-10h] [rbp-50h]

  v6 = (__int64)(0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5) + 1) / 2;
  v7 = 32
     * (v6
      + ((0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5) + 1 + ((0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5) + 1) >> 63))
       & 0xFFFFFFFFFFFFFFFELL));
  if ( v6 <= a4 )
  {
    sub_E73BA0(a1, a1 + v7, (__int64)a3);
    sub_E73BA0(a1 + v7, a2, (__int64)a3);
  }
  else
  {
    sub_E75D60(a1, a1 + v7);
    sub_E75D60(a1 + v7, a2);
  }
  sub_E74590(
    a1,
    (__int64 *)(a1 + v7),
    a2,
    0xAAAAAAAAAAAAAAABLL * (v7 >> 5),
    0xAAAAAAAAAAAAAAABLL * ((a2 - (a1 + v7)) >> 5),
    a3,
    a4);
  return v9;
}
