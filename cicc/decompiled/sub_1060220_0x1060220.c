// Function: sub_1060220
// Address: 0x1060220
//
char *__fastcall sub_1060220(char a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = (char *)&unk_3F8E8D0;
      break;
    case 1:
      result = "RO";
      break;
    case 2:
      result = "DB";
      break;
    case 3:
      result = "TC";
      break;
    case 4:
      result = "UA";
      break;
    case 5:
      result = "RW";
      break;
    case 6:
      result = "GL";
      break;
    case 7:
      result = (char *)&unk_3F8E8D3;
      break;
    case 8:
      result = "SV";
      break;
    case 9:
      result = "BS";
      break;
    case 10:
      result = "DS";
      break;
    case 11:
      result = "UC";
      break;
    case 12:
      result = (char *)&unk_42EACF9;
      break;
    case 13:
      result = "TB";
      break;
    case 15:
      result = "TC0";
      break;
    case 16:
      result = "TD";
      break;
    case 17:
      result = "SV64";
      break;
    case 18:
      result = "SV3264";
      break;
    case 20:
      result = "TL";
      break;
    case 21:
      result = (char *)&unk_3F6AD79;
      break;
    case 22:
      result = "TE";
      break;
    default:
      result = "Unknown";
      break;
  }
  return result;
}
