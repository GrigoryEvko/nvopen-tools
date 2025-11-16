// Function: sub_3887320
// Address: 0x3887320
//
__int64 __fastcall sub_3887320(int a1, _BYTE *a2)
{
  __int64 result; // rax

  *a2 = 1;
  switch ( a1 )
  {
    case 26:
      result = 8;
      break;
    case 27:
      result = 7;
      break;
    case 28:
      result = 2;
      break;
    case 29:
      result = 3;
      break;
    case 30:
      result = 4;
      break;
    case 31:
      result = 5;
      break;
    case 32:
      result = 6;
      break;
    case 35:
      result = 10;
      break;
    case 36:
      result = 1;
      break;
    case 43:
      result = 9;
      break;
    case 44:
      result = 0;
      break;
    default:
      *a2 = 0;
      result = 0;
      break;
  }
  return result;
}
