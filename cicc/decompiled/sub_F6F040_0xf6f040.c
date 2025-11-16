// Function: sub_F6F040
// Address: 0xf6f040
//
__int64 __fastcall sub_F6F040(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 6:
      result = 330;
      break;
    case 7:
      result = 329;
      break;
    case 8:
      result = 366;
      break;
    case 9:
      result = 365;
      break;
    case 12:
      result = 248;
      break;
    case 13:
      result = 237;
      break;
    case 14:
      result = 246;
      break;
    case 15:
      result = 235;
      break;
    default:
      BUG();
  }
  return result;
}
