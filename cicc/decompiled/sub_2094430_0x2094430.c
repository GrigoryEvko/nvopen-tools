// Function: sub_2094430
// Address: 0x2094430
//
const char *__fastcall sub_2094430(unsigned int a1)
{
  const char *result; // rax

  result = "<post-inc>";
  if ( a1 != 3 )
  {
    if ( a1 > 3 )
    {
      result = byte_3F871B3;
      if ( a1 == 4 )
        return "<post-dec>";
    }
    else
    {
      result = "<pre-inc>";
      if ( a1 != 1 )
      {
        result = byte_3F871B3;
        if ( a1 == 2 )
          return "<pre-dec>";
      }
    }
  }
  return result;
}
