// Function: sub_1D12E90
// Address: 0x1d12e90
//
__int64 __fastcall sub_1D12E90(__int64 a1)
{
  _BYTE *v1; // rax
  unsigned int v2; // r8d
  __int16 v3; // dx
  int v4; // esi
  unsigned int v5; // ecx

  v1 = *(_BYTE **)(a1 + 40);
  v2 = 1;
  if ( *v1 == 111 )
    return v2;
  v3 = *(_WORD *)(a1 + 24);
  if ( v3 == 194 || v3 == 212 )
    return v2;
  v4 = *(_DWORD *)(a1 + 60);
  if ( v4 == 1 )
    return 0;
  v5 = 1;
  while ( v1[16 * v5] != 111 )
  {
    if ( ++v5 == v4 )
      return 0;
  }
  return 1;
}
