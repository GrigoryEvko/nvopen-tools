// Function: sub_AC47B0
// Address: 0xac47b0
//
__int64 __fastcall sub_AC47B0(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 13:
    case 15:
    case 30:
      result = 1;
      break;
    case 14:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
      result = 0;
      break;
    default:
      BUG();
  }
  return result;
}
