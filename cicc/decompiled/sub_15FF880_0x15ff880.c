// Function: sub_15FF880
// Address: 0x15ff880
//
bool __fastcall sub_15FF880(int a1, int a2)
{
  bool result; // al

  result = 1;
  if ( a1 != a2 )
  {
    switch ( a1 )
    {
      case ' ':
        result = 1;
        if ( ((a2 - 35) & 0xFFFFFFFD) != 0 )
          result = ((a2 - 39) & 0xFFFFFFFD) == 0;
        break;
      case '"':
        result = (a2 & 0xFFFFFFFD) == 33;
        break;
      case '$':
        result = (a2 & 0xFFFFFFFB) == 33;
        break;
      case '&':
        result = a2 == 39 || a2 == 33;
        break;
      case '(':
        result = (a2 & 0xFFFFFFF7) == 33;
        break;
      default:
        result = 0;
        break;
    }
  }
  return result;
}
