// Function: sub_22A86F0
// Address: 0x22a86f0
//
void __fastcall sub_22A86F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx

  if ( a2 - a1 <= 784 )
  {
    sub_22A7520(a1, a2);
  }
  else
  {
    v2 = 56 * ((0x6DB6DB6DB6DB6DB7LL * ((a2 - a1) >> 3)) >> 1);
    sub_22A86F0(a1, a1 + v2);
    sub_22A86F0(a1 + v2, a2);
    sub_22A84E0(
      a1,
      (char *)(a1 + v2),
      a2,
      0x6DB6DB6DB6DB6DB7LL * (v2 >> 3),
      0x6DB6DB6DB6DB6DB7LL * ((a2 - (a1 + v2)) >> 3));
  }
}
