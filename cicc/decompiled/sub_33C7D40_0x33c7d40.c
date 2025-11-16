// Function: sub_33C7D40
// Address: 0x33c7d40
//
__int64 __fastcall sub_33C7D40(__int64 a1)
{
  _WORD *v1; // rax
  unsigned int v2; // r8d
  int v3; // edx
  int v4; // esi
  unsigned int v5; // ecx

  v1 = *(_WORD **)(a1 + 48);
  v2 = 1;
  if ( *v1 == 262 )
    return v2;
  v3 = *(_DWORD *)(a1 + 24);
  if ( v3 == 309 || v3 == 328 )
    return v2;
  v4 = *(_DWORD *)(a1 + 68);
  if ( v4 == 1 )
    return 0;
  v5 = 1;
  while ( v1[8 * v5] != 262 )
  {
    if ( ++v5 == v4 )
      return 0;
  }
  return 1;
}
