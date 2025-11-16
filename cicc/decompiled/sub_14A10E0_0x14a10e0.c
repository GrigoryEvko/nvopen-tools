// Function: sub_14A10E0
// Address: 0x14a10e0
//
__int64 __fastcall sub_14A10E0(__int64 a1, unsigned int a2)
{
  if ( a2 > 0x1189 )
    return 1;
  if ( a2 > 0x1182 )
    return ((1LL << ((unsigned __int8)a2 + 125)) & 0x49) == 0 ? 1 : 4;
  if ( a2 <= 0x95 )
  {
    if ( a2 > 2 )
    {
      switch ( a2 )
      {
        case 3u:
        case 4u:
        case 0xEu:
        case 0xFu:
        case 0x12u:
        case 0x13u:
        case 0x14u:
        case 0x17u:
        case 0x1Bu:
        case 0x1Cu:
        case 0x1Du:
        case 0x24u:
        case 0x25u:
        case 0x26u:
        case 0x4Cu:
        case 0x4Du:
        case 0x71u:
        case 0x72u:
        case 0x74u:
        case 0x75u:
        case 0x90u:
        case 0x95u:
          return 0;
        default:
          return 1;
      }
    }
    return 1;
  }
  return a2 != 191 && a2 != 215;
}
