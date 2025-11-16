// Function: sub_B52E90
// Address: 0xb52e90
//
__int64 __fastcall sub_B52E90(unsigned int a1)
{
  __int64 result; // rax

  result = a1;
  switch ( a1 )
  {
    case ' ':
    case '!':
    case '&':
    case '\'':
    case '(':
    case ')':
      return result;
    case '"':
      result = 38;
      break;
    case '#':
      result = 39;
      break;
    case '$':
      result = 40;
      break;
    case '%':
      result = 41;
      break;
    default:
      BUG();
  }
  return result;
}
