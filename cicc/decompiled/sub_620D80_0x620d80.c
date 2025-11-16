// Function: sub_620D80
// Address: 0x620d80
//
_WORD *__fastcall sub_620D80(_WORD *a1, __int64 a2)
{
  _WORD *i; // rdx
  unsigned __int64 v3; // rcx
  _WORD *result; // rax

  for ( i = a1 + 7; ; --i )
  {
    *i = a2;
    v3 = a2 >> 16;
    if ( a2 < 0 )
      v3 = (a2 >> 16) | 0xFFFF000000000000LL;
    result = i - 1;
    a2 = v3;
    if ( i == a1 )
      break;
  }
  return result;
}
