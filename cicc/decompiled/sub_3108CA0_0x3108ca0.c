// Function: sub_3108CA0
// Address: 0x3108ca0
//
__int64 __fastcall sub_3108CA0(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 1:
    case 2:
    case 5:
    case 6:
    case 9:
      result = 1;
      break;
    case 3:
    case 4:
    case 7:
    case 8:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
      result = 0;
      break;
    default:
      BUG();
  }
  return result;
}
