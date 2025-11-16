// Function: sub_CC7C70
// Address: 0xcc7c70
//
unsigned __int64 __fastcall sub_CC7C70(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rax

  switch ( *(_DWORD *)(a1 + 44) )
  {
    case 1:
    case 9:
      result = 5;
      break;
    case 5:
    case 0x1B:
      result = sub_CC78E0(a1);
      if ( !(_DWORD)result )
        result = 2LL * (*(_DWORD *)(a1 + 32) == 3) + 5;
      break;
    case 0x1F:
      v2 = sub_CC78E0(a1);
      result = ((HIDWORD(v2) & 0x7FFFFFFF | 0x8000000080000000LL) << 32) | (unsigned int)(v2 + 16);
      break;
    default:
      BUG();
  }
  return result;
}
