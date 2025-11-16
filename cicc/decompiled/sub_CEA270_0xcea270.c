// Function: sub_CEA270
// Address: 0xcea270
//
char __fastcall sub_CEA270(unsigned int a1)
{
  char result; // al

  if ( a1 == 9156 )
    return 1;
  if ( a1 <= 0x23C4 )
  {
    if ( a1 != 8496 )
    {
      result = 1;
      if ( a1 <= 0x2130 )
      {
        if ( a1 != 8490 )
          return ((a1 - 8131) & 0xFFFFFFFD) == 0;
      }
      else if ( a1 != 9154 )
      {
        return ((a1 - 8647) & 0xFFFFFFFD) == 0;
      }
      return result;
    }
    return 1;
  }
  result = 0;
  if ( a1 <= 0x256A )
  {
    if ( a1 > 0x2538 )
      return ((1LL << ((unsigned __int8)a1 - 57)) & 0x2800000000081LL) != 0;
    else
      return ((a1 - 9258) & 0xFFFFFFF7) == 0;
  }
  return result;
}
