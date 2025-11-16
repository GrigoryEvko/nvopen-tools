// Function: sub_D960E0
// Address: 0xd960e0
//
__int64 __fastcall sub_D960E0(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_WORD *)(a1 + 24) )
  {
    case 0:
    case 1:
    case 0xF:
      result = 0;
      break;
    case 2:
    case 3:
    case 4:
    case 0xE:
      result = a1 + 32;
      break;
    case 5:
    case 6:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
      result = *(_QWORD *)(a1 + 32);
      break;
    case 7:
      result = a1 + 32;
      break;
    default:
      BUG();
  }
  return result;
}
