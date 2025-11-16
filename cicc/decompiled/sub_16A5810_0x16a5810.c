// Function: sub_16A5810
// Address: 0x16a5810
//
__int64 __fastcall sub_16A5810(__int64 a1)
{
  int v1; // r9d
  int v2; // ecx
  unsigned int v3; // r8d
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r8
  __int64 v7; // rdx

  v1 = 64;
  v2 = *(_DWORD *)(a1 + 8) & 0x3F;
  if ( v2 )
  {
    v1 = *(_DWORD *)(a1 + 8) & 0x3F;
    LOBYTE(v2) = 64 - v2;
  }
  v3 = 64;
  v4 = ((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6;
  if ( *(_QWORD *)(*(_QWORD *)a1 + 8LL * ((int)v4 - 1)) << v2 != -1 )
  {
    _BitScanReverse64(&v5, ~(*(_QWORD *)(*(_QWORD *)a1 + 8LL * ((int)v4 - 1)) << v2));
    v3 = v5 ^ 0x3F;
  }
  if ( v1 != v3 )
    return v3;
  LODWORD(v4) = v4 - 2;
  if ( (v4 & 0x80000000) != 0LL )
    return v3;
  v4 = (int)v4;
  while ( 1 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v4);
    if ( v7 != -1 )
      break;
    --v4;
    v3 += 64;
    if ( (v4 & 0x80000000) != 0LL )
      return v3;
  }
  _BitScanReverse64((unsigned __int64 *)&v7, ~v7);
  return ((unsigned int)v7 ^ 0x3F) + v3;
}
