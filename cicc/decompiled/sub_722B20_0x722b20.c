// Function: sub_722B20
// Address: 0x722b20
//
__int64 __fastcall sub_722B20(unsigned __int64 a1, _WORD *a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rax

  if ( a1 > 0xFFFF )
  {
    result = 0;
    if ( a1 <= 0x10FFFF )
    {
      v3 = (a1 - 0x10000) >> 10;
      BYTE1(v3) |= 0xD8u;
      a2[1] = a1 & 0x3FF | 0xDC00;
      *a2 = v3;
      return 2;
    }
  }
  else
  {
    *a2 = a1;
    return 1;
  }
  return result;
}
