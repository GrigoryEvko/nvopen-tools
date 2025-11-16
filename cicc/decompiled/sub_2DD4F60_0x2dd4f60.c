// Function: sub_2DD4F60
// Address: 0x2dd4f60
//
void __fastcall sub_2DD4F60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx

  if ( a2 - a1 <= 1120 )
  {
    sub_2DD38C0(a1, a2, a3, a4, a5, a6);
  }
  else
  {
    v6 = 80 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 4)) >> 1);
    sub_2DD4F60(a1, a1 + v6);
    sub_2DD4F60(a1 + v6, a2);
    sub_2DD4CA0(a1, a1 + v6, a2, 0xCCCCCCCCCCCCCCCDLL * (v6 >> 4), 0xCCCCCCCCCCCCCCCDLL * ((a2 - (a1 + v6)) >> 4));
  }
}
