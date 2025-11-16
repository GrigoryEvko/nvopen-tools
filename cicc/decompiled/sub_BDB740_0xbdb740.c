// Function: sub_BDB740
// Address: 0xbdb740
//
unsigned __int64 __fastcall sub_BDB740(__int64 a1, __int64 a2)
{
  char v2; // bl

  v2 = sub_AE5020(a1, a2);
  return (((unsigned __int64)(sub_9208B0(a1, a2) + 7) >> 3) + (1LL << v2) - 1) >> v2 << v2;
}
