// Function: sub_A6DE70
// Address: 0xa6de70
//
char *__fastcall sub_A6DE70(char a1)
{
  char *result; // rax

  result = "write";
  if ( a1 != 2 )
  {
    result = "readwrite";
    if ( a1 != 3 )
    {
      result = "none";
      if ( a1 == 1 )
        return "read";
    }
  }
  return result;
}
