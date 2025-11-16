// Function: sub_1205440
// Address: 0x1205440
//
__int64 __fastcall sub_1205440(int a1, _BYTE *a2)
{
  __int64 result; // rax

  *a2 = 1;
  switch ( a1 )
  {
    case 28:
      result = 8;
      break;
    case 29:
      result = 7;
      break;
    case 30:
      result = 2;
      break;
    case 31:
      result = 3;
      break;
    case 32:
      result = 4;
      break;
    case 33:
      result = 5;
      break;
    case 34:
      result = 6;
      break;
    case 37:
      result = 10;
      break;
    case 38:
      result = 1;
      break;
    case 45:
      result = 9;
      break;
    case 46:
      result = 0;
      break;
    default:
      *a2 = 0;
      result = 0;
      break;
  }
  return result;
}
