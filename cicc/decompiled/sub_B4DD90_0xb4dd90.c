// Function: sub_B4DD90
// Address: 0xb4dd90
//
__int64 __fastcall sub_B4DD90(__int64 a1)
{
  unsigned int v1; // ecx
  unsigned int v2; // edx

  v1 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v1 == 1 )
    return 1;
  v2 = 1;
  while ( **(_BYTE **)(a1 + 32 * (v2 - (unsigned __int64)v1)) == 17 )
  {
    if ( v1 == ++v2 )
      return 1;
  }
  return 0;
}
