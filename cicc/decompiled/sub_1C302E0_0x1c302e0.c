// Function: sub_1C302E0
// Address: 0x1c302e0
//
char __fastcall sub_1C302E0(unsigned int a1)
{
  if ( a1 == 4251 )
    return 1;
  if ( a1 <= 0x109B )
  {
    if ( a1 != 3851 )
    {
      if ( a1 <= 0xF0B )
      {
        if ( a1 > 0xE39 )
          return a1 == 3846;
        else
          return a1 > 0xE37;
      }
      else if ( a1 > 0xF58 )
      {
        return a1 - 4188 <= 1;
      }
      else
      {
        return a1 > 0xF56;
      }
    }
    return 1;
  }
  if ( a1 == 4258 )
    return 1;
  return a1 - 4473 <= 0x28 && ((1LL << ((unsigned __int8)a1 - 121)) & 0x18000000041LL) != 0;
}
