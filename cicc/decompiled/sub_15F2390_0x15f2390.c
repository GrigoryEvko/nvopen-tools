// Function: sub_15F2390
// Address: 0x15f2390
//
__int64 __fastcall sub_15F2390(__int64 a1)
{
  __int64 result; // rax

  result = (unsigned int)*(unsigned __int8 *)(a1 + 16) - 35;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '#':
    case '%':
    case '\'':
    case '/':
      *(_BYTE *)(a1 + 17) &= 0xF9u;
      break;
    case ')':
    case '*':
    case '0':
    case '1':
      *(_BYTE *)(a1 + 17) &= ~2u;
      break;
    case '8':
      result = sub_15FA2E0(a1, 0);
      break;
    default:
      return result;
  }
  return result;
}
