// Function: sub_B52F50
// Address: 0xb52f50
//
__int64 __fastcall sub_B52F50(unsigned int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0u:
    case 1u:
    case 6u:
    case 7u:
    case 8u:
    case 9u:
    case 0xEu:
    case 0xFu:
    case 0x20u:
    case 0x21u:
      result = a1;
      break;
    case 2u:
      result = 4;
      break;
    case 3u:
      result = 5;
      break;
    case 4u:
      result = 2;
      break;
    case 5u:
      result = 3;
      break;
    case 0xAu:
      result = 12;
      break;
    case 0xBu:
      result = 13;
      break;
    case 0xCu:
      result = 10;
      break;
    case 0xDu:
      result = 11;
      break;
    case 0x22u:
      result = 36;
      break;
    case 0x23u:
      result = 37;
      break;
    case 0x24u:
      result = 34;
      break;
    case 0x25u:
      result = 35;
      break;
    case 0x26u:
      result = 40;
      break;
    case 0x27u:
      result = 41;
      break;
    case 0x28u:
      result = 38;
      break;
    case 0x29u:
      result = 39;
      break;
    default:
      BUG();
  }
  return result;
}
