// Function: sub_E97A00
// Address: 0xe97a00
//
__int64 __fastcall sub_E97A00(__int64 a1)
{
  __int64 result; // rax
  int v2; // eax

  switch ( *(_DWORD *)(a1 + 44) )
  {
    case 1:
    case 9:
      result = 1;
      break;
    case 5:
      v2 = *(_DWORD *)(a1 + 48);
      if ( v2 == 32 )
        result = 6;
      else
        result = 5 * (unsigned int)(v2 == 31) + 2;
      break;
    case 0x1B:
      result = 5 * (unsigned int)(*(_DWORD *)(a1 + 48) == 31) + 3;
      break;
    case 0x1C:
      result = 5 * (unsigned int)(*(_DWORD *)(a1 + 48) == 31) + 4;
      break;
    case 0x1E:
      result = 10;
      break;
    case 0x1F:
      result = (unsigned int)(*(_DWORD *)(a1 + 48) == 31) + 11;
      break;
    default:
      BUG();
  }
  return result;
}
