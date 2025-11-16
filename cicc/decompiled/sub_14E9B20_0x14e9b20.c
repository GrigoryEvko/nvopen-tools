// Function: sub_14E9B20
// Address: 0x14e9b20
//
__int64 __fastcall sub_14E9B20(unsigned __int64 a1, unsigned __int64 a2)
{
  char v2; // al
  __int64 v3; // rcx

  if ( (a1 & 0x10) != 0 )
  {
    if ( ((a1 >> 4) & 2) == 0 )
    {
      v2 = 1;
      LOBYTE(v3) = a2 <= 2;
      return (unsigned __int8)((((a1 & 0x40) != 0) << 6) | (32 * v3) | (16 * v2) | a1 & 0xF);
    }
  }
  else if ( a2 > 2 )
  {
    v2 = 0;
    v3 = (a1 >> 5) & 1;
    return (unsigned __int8)((((a1 & 0x40) != 0) << 6) | (32 * v3) | (16 * v2) | a1 & 0xF);
  }
  v2 = 1;
  LOBYTE(v3) = 1;
  return (unsigned __int8)((((a1 & 0x40) != 0) << 6) | (32 * v3) | (16 * v2) | a1 & 0xF);
}
