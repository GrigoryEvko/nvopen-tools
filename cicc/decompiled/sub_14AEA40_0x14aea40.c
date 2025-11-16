// Function: sub_14AEA40
// Address: 0x14aea40
//
__int64 __fastcall sub_14AEA40(__int64 a1)
{
  __int64 result; // rax

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '#':
    case '%':
    case '\'':
    case '/':
    case '1':
    case '4':
    case '8':
    case '<':
    case '>':
    case 'G':
    case 'H':
    case 'K':
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
