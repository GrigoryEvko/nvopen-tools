// Function: sub_15FF850
// Address: 0x15ff850
//
__int64 __fastcall sub_15FF850(int a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 2:
    case 4:
    case 6:
    case 33:
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
