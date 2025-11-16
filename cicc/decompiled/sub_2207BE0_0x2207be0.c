// Function: sub_2207BE0
// Address: 0x2207be0
//
char *__fastcall sub_2207BE0(char a1)
{
  char *result; // rax

  switch ( a1 & 0x3D )
  {
    case 1:
    case 0x11:
      result = "a";
      break;
    case 5:
    case 0x15:
      result = "ab";
      break;
    case 8:
      result = "r";
      break;
    case 9:
    case 0x19:
      result = "a+";
      break;
    case 0xC:
      result = "rb";
      break;
    case 0xD:
    case 0x1D:
      result = "a+b";
      break;
    case 0x10:
    case 0x30:
      result = "w";
      break;
    case 0x14:
    case 0x34:
      result = "wb";
      break;
    case 0x18:
      result = "r+";
      break;
    case 0x1C:
      result = "r+b";
      break;
    case 0x38:
      result = "w+";
      break;
    case 0x3C:
      result = "w+b";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
