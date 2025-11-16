// Function: sub_15FF420
// Address: 0x15ff420
//
__int64 __fastcall sub_15FF420(unsigned int a1)
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
  }
  return result;
}
