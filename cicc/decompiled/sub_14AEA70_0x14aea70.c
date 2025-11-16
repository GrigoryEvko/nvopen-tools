// Function: sub_14AEA70
// Address: 0x14aea70
//
__int64 __fastcall sub_14AEA70(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case ')':
    case '*':
    case ',':
    case '-':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v2 = *(_QWORD *)(a1 - 8);
      else
        v2 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      result = *(_QWORD *)(v2 + 24);
      break;
    case '6':
    case '7':
      result = *(_QWORD *)(a1 - 24);
      break;
    case ':':
      result = *(_QWORD *)(a1 - 72);
      break;
    case ';':
      result = *(_QWORD *)(a1 - 48);
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
