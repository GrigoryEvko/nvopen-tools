// Function: sub_1D12DA0
// Address: 0x1d12da0
//
bool __fastcall sub_1D12DA0(__int64 a1, unsigned int a2)
{
  bool result; // al

  if ( a2 > 0x78 )
    return a2 - 180 < 4;
  result = 0;
  if ( a2 > 0x33 )
  {
    switch ( a2 )
    {
      case '4':
      case '6':
      case ';':
      case '<':
      case '@':
      case 'B':
      case 'F':
      case 'G':
      case 'L':
      case 'N':
      case 'p':
      case 'q':
      case 'r':
      case 's':
      case 't':
      case 'u':
      case 'v':
      case 'w':
      case 'x':
        result = 1;
        break;
      default:
        result = 0;
        break;
    }
  }
  return result;
}
