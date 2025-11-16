// Function: sub_723DE0
// Address: 0x723de0
//
__int64 __fastcall sub_723DE0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int8 *v3; // r8
  unsigned __int64 v4; // rax
  int v5; // ecx
  char v6; // si

  v2 = *a1;
  v3 = a1;
  v4 = a2 ^ 0xFFFFFFFFLL;
  if ( !(_BYTE)v2 )
    return a2;
  do
  {
    ++v3;
    v5 = 8;
    do
    {
      v6 = v4 ^ v2;
      v4 >>= 1;
      if ( (v6 & 1) != 0 )
        v4 ^= 0xEDB88320;
      v2 >>= 1;
      --v5;
    }
    while ( v5 );
    v2 = *v3;
  }
  while ( (_BYTE)v2 );
  return v4 ^ 0xFFFFFFFF;
}
