// Function: sub_36D7030
// Address: 0x36d7030
//
__int64 __fastcall sub_36D7030(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 8324:
    case 8420:
      result = 1;
      break;
    case 8325:
    case 8421:
      result = 2;
      break;
    case 8326:
    case 8329:
    case 8422:
      result = 3;
      break;
    case 8327:
    case 8330:
    case 8423:
      result = 4;
      break;
    case 8328:
    case 8331:
    case 8424:
      result = 5;
      break;
    default:
      BUG();
  }
  return result;
}
