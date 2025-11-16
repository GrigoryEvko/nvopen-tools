// Function: sub_370BBB0
// Address: 0x370bbb0
//
__int64 __fastcall sub_370BBB0(__int16 a1, unsigned __int16 a2)
{
  __int64 result; // rax

  if ( a2 <= 7u )
  {
    result = 2;
    if ( a1 != 22 )
    {
      result = 1;
      if ( a1 != 30006 )
      {
        LOBYTE(result) = a1 == 20;
        return (unsigned int)(3 * result);
      }
    }
  }
  else if ( a2 == 208 )
  {
    result = 1;
    if ( a1 != 335 )
    {
      result = 3;
      if ( a1 != 341 )
      {
        LOBYTE(result) = a1 == 334;
        return (unsigned int)(2 * result);
      }
    }
  }
  else
  {
    return 0;
  }
  return result;
}
