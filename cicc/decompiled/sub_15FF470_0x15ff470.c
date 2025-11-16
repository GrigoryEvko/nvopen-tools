// Function: sub_15FF470
// Address: 0x15ff470
//
__int64 __fastcall sub_15FF470(unsigned int a1)
{
  __int64 result; // rax

  result = a1;
  switch ( a1 )
  {
    case ' ':
    case '!':
    case '"':
    case '#':
    case '$':
    case '%':
      return result;
    case '&':
      result = 34;
      break;
    case '\'':
      result = 35;
      break;
    case '(':
      result = 36;
      break;
    case ')':
      result = 37;
      break;
  }
  return result;
}
