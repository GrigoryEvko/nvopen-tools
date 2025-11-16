// Function: sub_12FBB40
// Address: 0x12fbb40
//
__int64 __fastcall sub_12FBB40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  result = a4;
  if ( (a1 & 0x7FFF800000000000LL) == 0x7FFF000000000000LL && a2 | a1 & 0x7FFFFFFFFFFFLL )
  {
    sub_12F9B70(16);
    return a2;
  }
  if ( (a3 & 0x7FFF800000000000LL) == 0x7FFF000000000000LL && a4 | a3 & 0x7FFFFFFFFFFFLL )
  {
    sub_12F9B70(16);
    result = a4;
  }
  if ( (~a1 & 0x7FFF000000000000LL) == 0 )
  {
    if ( a2 )
      return a2;
    if ( (a1 & 0xFFFFFFFFFFFFLL) != 0 )
      return 0;
  }
  return result;
}
