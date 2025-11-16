// Function: sub_36D7800
// Address: 0x36d7800
//
__int64 __fastcall sub_36D7800(__int64 a1)
{
  __int64 result; // rax

  result = sub_2EAC1E0(a1);
  switch ( (int)result )
  {
    case 0:
    case 1:
    case 3:
    case 4:
    case 5:
      return result;
    case 2:
      result = 0;
      break;
    default:
      if ( (_DWORD)result != 101 )
        result = 0;
      break;
  }
  return result;
}
