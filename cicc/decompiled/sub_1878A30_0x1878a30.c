// Function: sub_1878A30
// Address: 0x1878a30
//
void __fastcall sub_1878A30(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx

  if ( (__int64)a2 - a1 <= 1120 )
  {
    sub_18778A0(a1, a2);
  }
  else
  {
    v2 = 80 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - a1) >> 4)) >> 1);
    sub_1878A30(a1, a1 + v2);
    sub_1878A30(a1 + v2, a2);
    sub_1878710(
      a1,
      a1 + v2,
      (__int64)a2,
      0xCCCCCCCCCCCCCCCDLL * (v2 >> 4),
      0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - a1 - v2) >> 4));
  }
}
