// Function: sub_C650C0
// Address: 0xc650c0
//
__int64 __fastcall sub_C650C0(__int16 a1)
{
  __int64 result; // rax
  unsigned int v2; // edx

  result = a1 & 3;
  if ( (a1 & 0x40) != 0 )
    result = a1 & 3 | 0x60u;
  v2 = result;
  if ( (a1 & 0x80u) != 0 )
  {
    LOBYTE(v2) = result | 0x90;
    result = v2;
  }
  if ( (a1 & 0x100) != 0 )
    result = (unsigned int)result | 0x108;
  if ( (a1 & 0x200) != 0 )
    return (unsigned int)result | 0x204;
  return result;
}
