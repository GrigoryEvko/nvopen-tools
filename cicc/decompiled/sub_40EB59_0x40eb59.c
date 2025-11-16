// Function: sub_40EB59
// Address: 0x40eb59
//
__int64 __fastcall sub_40EB59(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  const char *v7; // r8

  v7 = (const char *)(a3 + 1);
  if ( a4 == 2 )
  {
    sub_40E1DF(a1, 0xAu, "%%%s", (const char *)(a3 + 1));
  }
  else if ( a4 )
  {
    sub_40E1DF(a1, 0xAu, "%%%d%s", a5, v7);
  }
  else
  {
    sub_40E1DF(a1, 0xAu, "%%-%d%s", a5, v7);
  }
  return a1;
}
