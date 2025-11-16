// Function: sub_3418530
// Address: 0x3418530
//
const char *__fastcall sub_3418530(unsigned int a1)
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
