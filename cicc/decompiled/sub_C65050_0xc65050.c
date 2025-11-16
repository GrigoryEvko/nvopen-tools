// Function: sub_C65050
// Address: 0xc65050
//
__int64 __fastcall sub_C65050(__int16 a1)
{
  __int64 result; // rax
  unsigned int v2; // edx
  unsigned int v3; // edx
  unsigned int v4; // edx

  result = a1 & 3;
  v2 = a1 & 3;
  BYTE1(v2) = 2;
  if ( (a1 & 4) != 0 )
    result = v2;
  v3 = result;
  if ( (a1 & 8) != 0 )
  {
    BYTE1(v3) = BYTE1(result) | 1;
    result = v3;
  }
  v4 = result;
  if ( (a1 & 0x10) != 0 )
  {
    LOBYTE(v4) = result | 0x80;
    result = v4;
  }
  if ( (a1 & 0x20) != 0 )
    result = (unsigned int)result | 0x40;
  if ( (a1 & 0x40) != 0 )
    result = (unsigned int)result | 0x20;
  if ( (a1 & 0x80u) != 0 )
    result = (unsigned int)result | 0x10;
  if ( (a1 & 0x100) != 0 )
    result = (unsigned int)result | 8;
  if ( (a1 & 0x200) != 0 )
    return (unsigned int)result | 4;
  return result;
}
