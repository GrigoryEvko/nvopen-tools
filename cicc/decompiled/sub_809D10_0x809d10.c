// Function: sub_809D10
// Address: 0x809d10
//
__int64 __fastcall sub_809D10(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 result; // rax

  *a2 = 0;
  *a3 = 0;
  result = *(unsigned __int8 *)(a1 + 48);
  switch ( *(_BYTE *)(a1 + 48) )
  {
    case 1:
      return result;
    case 2:
    case 6:
      result = *(_QWORD *)(a1 + 56);
      *a3 = result;
      break;
    case 3:
    case 5:
      result = sub_6E3F50(a1);
      *a2 = result;
      break;
    default:
      sub_721090();
  }
  return result;
}
