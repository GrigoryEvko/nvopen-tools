// Function: sub_2EF2BA0
// Address: 0x2ef2ba0
//
unsigned __int64 __fastcall sub_2EF2BA0(unsigned __int64 *a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int8 v5; // si

  result = *a1;
  if ( (*a1 & 0xFFFFFFFFFFFFFFF9LL) != 0 && (*(_BYTE *)a1 & 4) != 0 )
  {
    v2 = result >> 3;
    if ( (*(_BYTE *)a1 & 2) != 0 )
    {
      v3 = v2 & 0xFFFFFFFFFFE00000LL;
      v4 = 1;
      v5 = 0;
    }
    else
    {
      v3 = v2 & 0xFFFFFFFFE0000000LL;
      v4 = 0;
      v5 = 1;
    }
    return (8 * v3) | v5 | (unsigned __int64)(2 * v4);
  }
  return result;
}
