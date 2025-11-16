// Function: sub_38E2C30
// Address: 0x38e2c30
//
__int64 __fastcall sub_38E2C30(int a1, _DWORD *a2, char a3)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 12:
      *a2 = 0;
      result = 4;
      break;
    case 13:
      *a2 = 17;
      result = 4;
      break;
    case 15:
      *a2 = 2;
      result = 6;
      break;
    case 23:
      *a2 = 11;
      result = 6;
      break;
    case 28:
      *a2 = 3;
      result = 3;
      break;
    case 29:
      *a2 = 13;
      result = 5;
      break;
    case 30:
      *a2 = 7;
      result = 1;
      break;
    case 31:
      *a2 = 18;
      result = 5;
      break;
    case 32:
      *a2 = 1;
      result = 5;
      break;
    case 33:
      *a2 = 6;
      result = 2;
      break;
    case 35:
    case 41:
      *a2 = 12;
      result = 3;
      break;
    case 36:
      *a2 = 10;
      result = 6;
      break;
    case 38:
      *a2 = 8;
      result = 3;
      break;
    case 39:
      *a2 = 9;
      result = 3;
      break;
    case 40:
      *a2 = 14;
      result = 6;
      break;
    case 42:
      *a2 = 4;
      result = 3;
      break;
    case 43:
      *a2 = 5;
      result = 3;
      break;
    case 44:
      *a2 = 15 - ((a3 == 0) - 1);
      result = 6;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
