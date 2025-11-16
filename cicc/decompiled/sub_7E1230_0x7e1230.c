// Function: sub_7E1230
// Address: 0x7e1230
//
__int64 __fastcall sub_7E1230(_BYTE *a1, int a2, int a3, int a4)
{
  __int64 result; // rax

  result = (unsigned __int8)a1[156];
  a1[156] |= 1u;
  if ( a2 )
  {
    result = (unsigned int)result | 5;
    a1[156] = result;
  }
  a1[156] |= 0x20u;
  if ( a3 )
    a1[136] = 2;
  if ( a4 )
    a1[168] &= 0xF8u;
  return result;
}
