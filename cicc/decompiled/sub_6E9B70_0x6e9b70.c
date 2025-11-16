// Function: sub_6E9B70
// Address: 0x6e9b70
//
__int64 __fastcall sub_6E9B70(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 )
  {
    switch ( (char)a1 )
    {
      case 5:
        return 27;
      case 6:
        return 26;
      case 7:
        return 3;
      case 11:
        return 0;
      case 13:
        return 28;
      case 14:
        return 29;
      case 37:
      case 38:
        return (unsigned int)a1;
      case 47:
        return 117;
      default:
        goto LABEL_10;
    }
  }
  LODWORD(a1) = a1 - 5;
  switch ( (char)a1 )
  {
    case 0:
      result = 39;
      break;
    case 1:
      result = 40;
      break;
    case 2:
      result = 41;
      break;
    case 3:
      result = 42;
      break;
    case 4:
      result = 43;
      break;
    case 5:
      result = 57;
      break;
    case 6:
      result = 55;
      break;
    case 7:
      result = 56;
      break;
    case 10:
      result = 73;
      break;
    case 11:
      result = 61;
      break;
    case 12:
      result = 60;
      break;
    case 13:
      result = 74;
      break;
    case 14:
      result = 75;
      break;
    case 15:
      result = 76;
      break;
    case 16:
      result = 77;
      break;
    case 17:
      result = 78;
      break;
    case 18:
      result = 83;
      break;
    case 19:
      result = 81;
      break;
    case 20:
      result = 82;
      break;
    case 21:
      result = 53;
      break;
    case 22:
      result = 54;
      break;
    case 23:
      result = 80;
      break;
    case 24:
      result = 79;
      break;
    case 25:
      result = 58;
      break;
    case 26:
      result = 59;
      break;
    case 27:
      result = 63;
      break;
    case 28:
      result = 62;
      break;
    case 29:
      result = 64;
      break;
    case 30:
      result = 87;
      break;
    case 31:
      result = 88;
      break;
    case 32:
      result = 35;
      break;
    case 33:
      result = 36;
      break;
    case 34:
      result = 91;
      break;
    case 35:
      result = 97;
      break;
    case 38:
      result = 92;
      break;
    case 40:
      result = 71;
      break;
    case 41:
      result = 72;
      break;
    default:
      a1 = (unsigned int)a1;
LABEL_10:
      sub_721090(a1);
  }
  return result;
}
