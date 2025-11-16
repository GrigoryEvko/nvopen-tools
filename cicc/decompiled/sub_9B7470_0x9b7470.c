// Function: sub_9B7470
// Address: 0x9b7470
//
__int64 __fastcall sub_9B7470(unsigned int a1)
{
  __int64 result; // rax

  if ( a1 > 0x5A )
  {
    switch ( a1 )
    {
      case 0xAAu:
      case 0xACu:
      case 0xADu:
      case 0xAEu:
      case 0xAFu:
      case 0xB0u:
      case 0xB4u:
      case 0xB5u:
      case 0xCFu:
      case 0xD4u:
      case 0xDAu:
      case 0xDBu:
      case 0xDCu:
      case 0xDFu:
      case 0xEBu:
      case 0xEDu:
      case 0xF6u:
      case 0xF8u:
      case 0xF9u:
      case 0xFAu:
      case 0x11Cu:
      case 0x11Du:
      case 0x134u:
      case 0x135u:
      case 0x136u:
      case 0x137u:
      case 0x139u:
      case 0x145u:
      case 0x146u:
      case 0x147u:
      case 0x148u:
      case 0x149u:
      case 0x14Au:
      case 0x14Bu:
      case 0x14Cu:
      case 0x14Fu:
      case 0x152u:
      case 0x15Cu:
      case 0x15Du:
      case 0x163u:
      case 0x167u:
      case 0x16Au:
      case 0x16Du:
      case 0x16Eu:
      case 0x16Fu:
      case 0x170u:
      case 0x173u:
        goto LABEL_6;
      default:
        return 0;
    }
  }
  if ( !a1 )
    return 0;
  switch ( a1 )
  {
    case 1u:
    case 2u:
    case 0xAu:
    case 0xCu:
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x14u:
    case 0x15u:
    case 0x1Au:
    case 0x3Fu:
    case 0x40u:
    case 0x41u:
    case 0x42u:
    case 0x43u:
    case 0x58u:
    case 0x59u:
    case 0x5Au:
LABEL_6:
      result = 1;
      break;
    default:
      return 0;
  }
  return result;
}
