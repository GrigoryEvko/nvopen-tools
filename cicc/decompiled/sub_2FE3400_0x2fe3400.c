// Function: sub_2FE3400
// Address: 0x2fe3400
//
char __fastcall sub_2FE3400(__int64 a1, unsigned int a2)
{
  char result; // al

  if ( a2 > 0x62 )
  {
    if ( a2 > 0xBC )
      return a2 - 279 < 8;
    else
      return a2 > 0xB9 || a2 - 172 < 0xC;
  }
  else
  {
    result = 0;
    if ( a2 > 0x37 )
    {
      switch ( a2 )
      {
        case '8':
        case ':':
        case '?':
        case '@':
        case 'D':
        case 'F':
        case 'L':
        case 'M':
        case 'R':
        case 'S':
        case '`':
        case 'b':
          return 1;
        default:
          result = 0;
          break;
      }
    }
  }
  return result;
}
