// Function: sub_F6F100
// Address: 0xf6f100
//
__int64 __fastcall sub_F6F100(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 6:
      result = 40;
      break;
    case 7:
      result = 38;
      break;
    case 8:
      result = 36;
      break;
    case 9:
      result = 34;
      break;
    case 12:
      result = 4;
      break;
    case 13:
      result = 2;
      break;
    default:
      BUG();
  }
  return result;
}
