// Function: sub_F075A0
// Address: 0xf075a0
//
bool __fastcall sub_F075A0(int a1, unsigned int a2)
{
  bool result; // al

  if ( a2 <= 0x1E && ((1LL << a2) & 0x70066000) != 0 )
  {
    if ( a2 == 28 )
    {
      return (unsigned int)(a1 - 29) <= 1;
    }
    else if ( a2 == 29 )
    {
      return a1 == 28;
    }
    else
    {
      result = 0;
      if ( a2 == 17 )
        return (a1 & 0xFFFFFFFD) == 13;
    }
  }
  else
  {
    result = 0;
    if ( (unsigned int)(a1 - 28) <= 2 )
      return a2 - 25 <= 2;
  }
  return result;
}
