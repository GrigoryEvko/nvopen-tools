// Function: sub_B53110
// Address: 0xb53110
//
__int64 __fastcall sub_B53110(unsigned int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 3u:
      result = 2;
      break;
    case 5u:
      result = 4;
      break;
    case 0xBu:
      result = 10;
      break;
    case 0xDu:
      result = 12;
      break;
    case 0x23u:
      result = 34;
      break;
    case 0x25u:
      result = 36;
      break;
    case 0x27u:
      result = 38;
      break;
    case 0x29u:
      result = 40;
      break;
    default:
      result = a1;
      break;
  }
  return result;
}
