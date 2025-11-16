// Function: sub_216FF90
// Address: 0x216ff90
//
__int64 __fastcall sub_216FF90(__int64 a1)
{
  int v1; // eax
  unsigned int v2; // r8d
  __int64 result; // rax

  v1 = *(__int16 *)(a1 + 24);
  v2 = 1;
  if ( *(_WORD *)(a1 + 24) == 145 )
    return v2;
  v2 = 0;
  if ( (v1 & 0x8000u) == 0 )
    return v2;
  switch ( -v1 )
  {
    case 245:
    case 249:
    case 257:
    case 261:
    case 289:
    case 293:
    case 301:
    case 305:
      return 1;
    default:
      result = 0;
      break;
  }
  return result;
}
