// Function: sub_AF33F0
// Address: 0xaf33f0
//
char *__fastcall sub_AF33F0(int a1)
{
  char *result; // rax

  result = "None";
  if ( a1 != 2 )
  {
    result = "Apple";
    if ( a1 != 3 )
    {
      result = "GNU";
      if ( a1 != 1 )
        return 0;
    }
  }
  return result;
}
