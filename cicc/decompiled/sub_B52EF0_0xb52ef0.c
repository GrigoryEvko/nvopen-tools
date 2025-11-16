// Function: sub_B52EF0
// Address: 0xb52ef0
//
__int64 __fastcall sub_B52EF0(unsigned int a1)
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
    default:
      BUG();
  }
  return result;
}
