// Function: sub_B46E30
// Address: 0xb46e30
//
__int64 __fastcall sub_B46E30(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_BYTE *)a1 )
  {
    case 0x1E:
    case 0x23:
    case 0x24:
      result = 0;
      break;
    case 0x1F:
      result = (unsigned int)((*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 3) + 1;
      break;
    case 0x20:
      result = (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1;
      break;
    case 0x21:
    case 0x27:
      result = (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) - 1;
      break;
    case 0x22:
      result = 2;
      break;
    case 0x25:
      result = *(_WORD *)(a1 + 2) & 1;
      break;
    case 0x26:
      result = 1;
      break;
    case 0x28:
      result = (unsigned int)(*(_DWORD *)(a1 + 88) + 1);
      break;
    default:
      BUG();
  }
  return result;
}
