// Function: sub_12F97F0
// Address: 0x12f97f0
//
bool __fastcall sub_12F97F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool result; // al
  __int64 v6; // rdx

  if ( (~a1 & 0x7FF0000000000000LL) == 0 && (a1 & 0xFFFFFFFFFFFFFLL) != 0
    || (~a2 & 0x7FF0000000000000LL) == 0 && (a2 & 0xFFFFFFFFFFFFFLL) != 0 )
  {
    v6 = 0x7FF0000000000000LL;
    if ( (a1 & 0x7FF8000000000000LL) == 0x7FF0000000000000LL && (a1 & 0x7FFFFFFFFFFFFLL) != 0
      || (result = 0, a4 = 0x7FF0000000000000LL, (a2 & 0x7FF8000000000000LL) == 0x7FF0000000000000LL)
      && (v6 = 0x7FFFFFFFFFFFFLL, (a2 & 0x7FFFFFFFFFFFFLL) != 0) )
    {
      sub_12F9B70(16, a2, v6, a4, a5);
      return 0;
    }
  }
  else
  {
    result = 1;
    if ( a1 != a2 )
      return ((a2 | a1) & 0x7FFFFFFFFFFFFFFFLL) == 0;
  }
  return result;
}
