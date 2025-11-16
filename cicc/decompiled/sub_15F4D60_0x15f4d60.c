// Function: sub_15F4D60
// Address: 0x15f4d60
//
__int64 __fastcall sub_15F4D60(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x18:
    case 0x1C:
    case 0x22:
      result = (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) - 1;
      break;
    case 0x19:
    case 0x1E:
    case 0x1F:
      result = 0;
      break;
    case 0x1A:
      result = (unsigned int)((*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 3) + 1;
      break;
    case 0x1B:
      result = (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1;
      break;
    case 0x1D:
      result = 2;
      break;
    case 0x20:
      result = *(_WORD *)(a1 + 18) & 1;
      break;
    case 0x21:
      result = 1;
      break;
  }
  return result;
}
