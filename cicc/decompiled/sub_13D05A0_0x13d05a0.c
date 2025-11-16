// Function: sub_13D05A0
// Address: 0x13d05a0
//
__int64 __fastcall sub_13D05A0(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r8d
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r8

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 > 0x40 )
    return sub_16A5810();
  v2 = 64;
  v3 = ~(*(_QWORD *)a1 << (64 - (unsigned __int8)v1));
  if ( v3 )
  {
    _BitScanReverse64(&v4, v3);
    return (unsigned int)v4 ^ 0x3F;
  }
  return v2;
}
