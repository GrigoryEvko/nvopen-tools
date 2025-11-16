// Function: sub_8770E0
// Address: 0x8770e0
//
__int64 __fastcall sub_8770E0(char a1)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 3:
    case 4:
    case 6:
    case 7:
      result = 1;
      break;
    case 1:
    case 2:
    case 8:
    case 9:
    case 11:
    case 13:
    case 15:
    case 16:
    case 17:
      result = 0;
      break;
    default:
      sub_721090();
  }
  return result;
}
