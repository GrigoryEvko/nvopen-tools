// Function: sub_15FF820
// Address: 0x15ff820
//
__int64 __fastcall sub_15FF820(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 9:
    case 11:
    case 13:
    case 15:
    case 32:
    case 35:
    case 37:
    case 39:
    case 41:
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
