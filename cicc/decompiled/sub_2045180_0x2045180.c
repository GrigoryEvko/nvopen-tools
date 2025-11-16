// Function: sub_2045180
// Address: 0x2045180
//
__int64 __fastcall sub_2045180(char a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 1:
    case 111:
    case 112:
    case 113:
    case 114:
      result = 0;
      break;
    case 2:
    case 14:
    case 56:
      result = 1;
      break;
    case 3:
    case 17:
    case 24:
    case 59:
    case 62:
      result = 8;
      break;
    case 4:
    case 8:
    case 18:
    case 25:
    case 33:
    case 60:
    case 63:
    case 68:
      result = 16;
      break;
    case 5:
    case 9:
    case 19:
    case 26:
    case 34:
    case 41:
    case 61:
    case 64:
    case 69:
    case 74:
    case 86:
    case 89:
    case 98:
    case 101:
      result = 32;
      break;
    case 6:
    case 10:
    case 20:
    case 27:
    case 35:
    case 42:
    case 49:
    case 65:
    case 70:
    case 75:
    case 80:
    case 87:
    case 90:
    case 94:
    case 99:
    case 102:
    case 106:
    case 110:
      result = 64;
      break;
    case 7:
    case 12:
    case 13:
    case 21:
    case 28:
    case 36:
    case 43:
    case 50:
    case 55:
    case 66:
    case 71:
    case 76:
    case 81:
    case 88:
    case 91:
    case 95:
    case 100:
    case 103:
    case 107:
      result = 128;
      break;
    case 11:
      result = 80;
      break;
    case 15:
    case 57:
      result = 2;
      break;
    case 16:
    case 58:
      result = 4;
      break;
    case 22:
    case 30:
    case 38:
    case 45:
    case 52:
    case 73:
    case 78:
    case 83:
    case 93:
    case 97:
    case 105:
    case 109:
      result = 512;
      break;
    case 23:
    case 31:
    case 39:
    case 46:
    case 53:
    case 79:
    case 84:
      result = 1024;
      break;
    case 29:
    case 37:
    case 44:
    case 51:
    case 67:
    case 72:
    case 77:
    case 82:
    case 92:
    case 96:
    case 104:
    case 108:
      result = 256;
      break;
    case 32:
    case 40:
    case 47:
    case 54:
    case 85:
      result = 2048;
      break;
    case 48:
      result = 4096;
      break;
  }
  return result;
}
