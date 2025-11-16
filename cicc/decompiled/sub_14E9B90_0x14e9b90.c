// Function: sub_14E9B90
// Address: 0x14e9b90
//
__int64 __fastcall sub_14E9B90(char a1)
{
  __int64 result; // rax

  result = -(a1 & 1);
  if ( a1 < 0 )
    result = -(a1 & 1) | 1u;
  if ( (a1 & 2) != 0 )
    result = (unsigned int)result | 2;
  if ( (a1 & 4) != 0 )
    result = (unsigned int)result | 4;
  if ( (a1 & 8) != 0 )
    result = (unsigned int)result | 8;
  if ( (a1 & 0x10) != 0 )
    result = (unsigned int)result | 0x10;
  if ( (a1 & 0x20) != 0 )
    result = (unsigned int)result | 0x20;
  if ( (a1 & 0x40) != 0 )
    return (unsigned int)result | 0x40;
  return result;
}
