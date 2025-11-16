// Function: sub_2C1A9C0
// Address: 0x2c1a9c0
//
bool __fastcall sub_2C1A9C0(__int64 a1)
{
  bool result; // al

  if ( (unsigned int)*(unsigned __int8 *)(a1 + 160) - 13 <= 0x11 )
    return 1;
  if ( sub_2C1A9B0(a1) )
    return 1;
  result = sub_2C1A990(a1);
  if ( result )
    return 1;
  switch ( *(_BYTE *)(a1 + 160) )
  {
    case '5':
    case '9':
    case 'J':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'T':
    case 'U':
      return 1;
    default:
      return result;
  }
  return result;
}
