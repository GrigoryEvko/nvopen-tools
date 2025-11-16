// Function: sub_6E2DD0
// Address: 0x6e2dd0
//
__int64 __fastcall sub_6E2DD0(__int64 a1, char a2)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 16) = a2;
  switch ( a2 )
  {
    case 0:
      return result;
    case 1:
      *(_QWORD *)(a1 + 144) = 0;
      *(_QWORD *)(a1 + 136) = 0;
      break;
    case 2:
      result = sub_724C70(a1 + 144, 0);
      break;
    case 3:
    case 4:
    case 6:
      *(_QWORD *)(a1 + 136) = 0;
      break;
    case 5:
      *(_QWORD *)(a1 + 144) = 0;
      break;
    default:
      sub_721090(a1);
  }
  return result;
}
