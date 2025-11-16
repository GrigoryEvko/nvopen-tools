// Function: sub_2FE5730
// Address: 0x2fe5730
//
__int64 __fastcall sub_2FE5730(
        __int16 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int a7)
{
  __int64 result; // rax

  result = a3;
  if ( a1 != 12 )
  {
    result = a4;
    if ( a1 != 13 )
    {
      result = a5;
      if ( a1 != 14 )
      {
        result = a6;
        if ( a1 != 15 )
        {
          result = 729;
          if ( a1 == 16 )
            return a7;
        }
      }
    }
  }
  return result;
}
