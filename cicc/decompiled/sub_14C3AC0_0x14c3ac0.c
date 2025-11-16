// Function: sub_14C3AC0
// Address: 0x14c3ac0
//
__int64 __fastcall sub_14C3AC0(unsigned int a1)
{
  __int64 result; // rax

  if ( a1 > 0x37 )
  {
    switch ( a1 )
    {
      case 0x60u:
      case 0x61u:
      case 0x63u:
      case 0x64u:
      case 0x7Au:
      case 0x7Bu:
      case 0x7Cu:
      case 0x84u:
      case 0x8Bu:
      case 0x8Cu:
      case 0x92u:
      case 0x93u:
      case 0xBBu:
      case 0xBCu:
      case 0xC2u:
      case 0xC4u:
      case 0xCEu:
        goto LABEL_6;
      default:
        return 0;
    }
  }
  if ( a1 <= 4 )
    return 0;
  switch ( a1 )
  {
    case 5u:
    case 6u:
    case 8u:
    case 0xDu:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x21u:
    case 0x36u:
    case 0x37u:
LABEL_6:
      result = 1;
      break;
    default:
      return 0;
  }
  return result;
}
