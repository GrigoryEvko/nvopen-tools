// Function: sub_2620680
// Address: 0x2620680
//
void __fastcall sub_2620680(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx

  if ( (__int64)a2 - a1 <= 1120 )
  {
    sub_261F4F0(a1, a2);
  }
  else
  {
    v2 = 80 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - a1) >> 4)) >> 1);
    sub_2620680(a1, a1 + v2);
    sub_2620680(a1 + v2, a2);
    sub_2620360(
      a1,
      a1 + v2,
      (__int64)a2,
      0xCCCCCCCCCCCCCCCDLL * (v2 >> 4),
      0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - a1 - v2) >> 4));
  }
}
