// Function: sub_725ED0
// Address: 0x725ed0
//
__int64 __fastcall sub_725ED0(__int64 a1, char a2)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 174) = a2;
  switch ( a2 )
  {
    case 0:
      *(_WORD *)(a1 + 176) = 0;
      result = 0;
      break;
    case 1:
    case 2:
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      break;
    case 3:
    case 4:
      return result;
    case 5:
      *(_BYTE *)(a1 + 176) = 0;
      break;
    case 6:
    case 7:
      *(_QWORD *)(a1 + 176) = 0;
      break;
    default:
      sub_721090();
  }
  return result;
}
