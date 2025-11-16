// Function: sub_17044F0
// Address: 0x17044f0
//
bool __fastcall sub_17044F0(int a1, unsigned int a2)
{
  bool result; // al

  if ( a2 <= 0x1C && ((1LL << a2) & 0x1C019800) != 0 )
  {
    if ( a2 == 26 )
    {
      return (unsigned int)(a1 - 27) <= 1;
    }
    else if ( a2 == 27 )
    {
      return a1 == 26;
    }
    else
    {
      result = 0;
      if ( a2 == 15 )
        return ((a1 - 11) & 0xFFFFFFFD) == 0;
    }
  }
  else
  {
    result = 0;
    if ( (unsigned int)(a1 - 26) <= 2 )
      return a2 - 23 <= 2;
  }
  return result;
}
