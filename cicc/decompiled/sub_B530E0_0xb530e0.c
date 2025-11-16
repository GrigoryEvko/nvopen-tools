// Function: sub_B530E0
// Address: 0xb530e0
//
__int64 __fastcall sub_B530E0(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 2:
    case 4:
    case 10:
    case 12:
    case 34:
    case 36:
    case 38:
    case 40:
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
