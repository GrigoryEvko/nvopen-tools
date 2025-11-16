// Function: sub_12FCA60
// Address: 0x12fca60
//
__int64 __fastcall sub_12FCA60(unsigned __int64 a1)
{
  __int64 result; // rax
  char v2; // cl
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rcx
  __int64 v6; // rdx

  if ( a1 > 0x7000000000000000LL )
    return 232;
  result = 0;
  if ( a1 )
  {
    if ( a1 <= 8 )
    {
      if ( a1 != 1 )
      {
        _BitScanReverse64(&v5, a1 - 1);
        _BitScanReverse64((unsigned __int64 *)&v6, 1LL << ((unsigned __int8)v5 + 1));
        if ( (int)v6 > 2 )
          return (unsigned int)(v6 - 3);
      }
    }
    else
    {
      v2 = 7;
      _BitScanReverse64((unsigned __int64 *)&v3, 2 * a1 - 1);
      if ( (unsigned int)v3 >= 7 )
        v2 = v3;
      v4 = (((-1LL << (v2 - 3)) & (a1 - 1)) >> (v2 - 3)) & 3;
      if ( (unsigned int)v3 < 6 )
        LODWORD(v3) = 6;
      return (unsigned int)(v4 + 4 * v3 - 23);
    }
  }
  return result;
}
