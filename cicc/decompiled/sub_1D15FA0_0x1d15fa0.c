// Function: sub_1D15FA0
// Address: 0x1d15fa0
//
void *__fastcall sub_1D15FA0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  void *result; // rax
  _QWORD v4[3]; // [rsp+0h] [rbp-20h] BYREF

  v4[0] = a1;
  v4[1] = a2;
  if ( !(_BYTE)a1 )
  {
    v2 = v4;
    sub_1F58D20(v4);
LABEL_5:
    LOBYTE(a1) = sub_1F596B0(v2);
    goto LABEL_3;
  }
  if ( (unsigned __int8)(a1 - 14) > 0x5Fu )
  {
LABEL_3:
    switch ( (char)a1 )
    {
      case 8:
        goto LABEL_11;
      case 9:
        goto LABEL_12;
      case 10:
        goto LABEL_10;
      case 11:
        result = sub_16982A0();
        break;
      case 12:
        result = sub_1698290();
        break;
      case 13:
        result = sub_16982C0();
        break;
      default:
        goto LABEL_5;
    }
    return result;
  }
  switch ( (char)a1 )
  {
    case 'V':
    case 'W':
    case 'X':
    case 'b':
    case 'c':
    case 'd':
LABEL_11:
      result = sub_1698260();
      break;
    case 'Y':
    case 'Z':
    case '[':
    case '\\':
    case ']':
    case 'e':
    case 'f':
    case 'g':
    case 'h':
    case 'i':
LABEL_12:
      result = sub_1698270();
      break;
    case '^':
    case '_':
    case '`':
    case 'a':
    case 'j':
    case 'k':
    case 'l':
    case 'm':
LABEL_10:
      result = sub_1698280();
      break;
    default:
      goto LABEL_5;
  }
  return result;
}
