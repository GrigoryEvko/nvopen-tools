// Function: sub_D788C0
// Address: 0xd788c0
//
__int64 __fastcall sub_D788C0(unsigned __int64 a1, __int16 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rax

  result = 0x80000000LL;
  if ( a1 )
  {
    _BitScanReverse64(&v3, a1);
    return a2 + 63 - ((unsigned int)v3 ^ 0x3F);
  }
  return result;
}
