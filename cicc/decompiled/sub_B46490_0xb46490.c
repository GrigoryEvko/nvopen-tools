// Function: sub_B46490
// Address: 0xb46490
//
__int64 __fastcall sub_B46490(__int64 a1)
{
  __int64 result; // rax
  unsigned __int16 v2; // ax

  switch ( *(_BYTE *)a1 )
  {
    case '"':
    case '(':
    case 'U':
      result = (unsigned int)sub_B49E20(a1) ^ 1;
      break;
    case '&':
    case '>':
    case '@':
    case 'A':
    case 'B':
    case 'Q':
    case 'Y':
      result = 1;
      break;
    case '=':
      v2 = *(_WORD *)(a1 + 2);
      if ( ((v2 >> 7) & 6) != 0 )
        LOBYTE(v2) = 1;
      result = v2 & 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
