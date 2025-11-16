// Function: sub_C65100
// Address: 0xc65100
//
__int64 __fastcall sub_C65100(__int16 a1)
{
  __int64 result; // rax
  unsigned int v2; // edx

  result = a1 & 3;
  if ( (a1 & 0x60) != 0 )
    result = a1 & 3 | 0x60u;
  v2 = result;
  if ( (a1 & 0x90) != 0 )
  {
    LOBYTE(v2) = result | 0x90;
    result = v2;
  }
  if ( (a1 & 0x108) != 0 )
    result = (unsigned int)result | 0x108;
  if ( (a1 & 0x204) != 0 )
    return (unsigned int)result | 0x204;
  return result;
}
