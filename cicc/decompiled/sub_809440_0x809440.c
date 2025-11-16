// Function: sub_809440
// Address: 0x809440
//
_BOOL8 __fastcall sub_809440(__int64 a1)
{
  _BOOL8 result; // rax

  switch ( *(_BYTE *)(a1 + 140) )
  {
    case 0:
    case 1:
    case 3:
    case 0x13:
    case 0x14:
      result = 0;
      break;
    case 2:
      result = (*(_BYTE *)(a1 + 161) & 8) != 0;
      break;
    case 5:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x12:
      result = 1;
      break;
    case 6:
      result = (unsigned int)sub_7E1E50(a1) == 0;
      break;
    case 0xE:
      result = (unsigned int)sub_8D3EA0(a1) == 0;
      break;
    default:
      sub_721090();
  }
  return result;
}
