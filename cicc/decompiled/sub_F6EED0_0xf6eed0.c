// Function: sub_F6EED0
// Address: 0xf6eed0
//
__int64 __fastcall sub_F6EED0(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 387:
      result = 13;
      break;
    case 388:
      result = 28;
      break;
    case 389:
      result = 14;
      break;
    case 390:
    case 392:
      result = 54;
      break;
    case 394:
      result = 18;
      break;
    case 395:
      result = 17;
      break;
    case 396:
      result = 29;
      break;
    case 397:
    case 398:
    case 399:
    case 400:
      result = 53;
      break;
    case 401:
      result = 30;
      break;
    default:
      BUG();
  }
  return result;
}
