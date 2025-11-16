// Function: sub_B530B0
// Address: 0xb530b0
//
__int64 __fastcall sub_B530B0(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 3:
    case 5:
    case 11:
    case 13:
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
