// Function: sub_15FF290
// Address: 0x15ff290
//
char *__fastcall sub_15FF290(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "false";
      break;
    case 1:
      result = "oeq";
      break;
    case 2:
      result = "ogt";
      break;
    case 3:
      result = "oge";
      break;
    case 4:
      result = "olt";
      break;
    case 5:
      result = "ole";
      break;
    case 6:
      result = "one";
      break;
    case 7:
      result = "ord";
      break;
    case 8:
      result = "uno";
      break;
    case 9:
      result = "ueq";
      break;
    case 10:
    case 34:
      result = "ugt";
      break;
    case 11:
    case 35:
      result = "uge";
      break;
    case 12:
    case 36:
      result = "ult";
      break;
    case 13:
    case 37:
      result = "ule";
      break;
    case 14:
      result = "une";
      break;
    case 15:
      result = "true";
      break;
    case 32:
      result = "eq";
      break;
    case 33:
      result = (char *)&unk_432C6B1;
      break;
    case 38:
      result = (char *)&unk_3F2AD80;
      break;
    case 39:
      result = (char *)&unk_3F2AD84;
      break;
    case 40:
      result = (char *)&unk_3F2AD88;
      break;
    case 41:
      result = (char *)&unk_3F2AD8C;
      break;
    default:
      result = "unknown";
      break;
  }
  return result;
}
