// Function: sub_15FA290
// Address: 0x15fa290
//
__int64 __fastcall sub_15FA290(__int64 a1)
{
  unsigned int v1; // ecx
  unsigned int v2; // edx

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v1 == 1 )
    return 1;
  v2 = 1;
  while ( *(_BYTE *)(*(_QWORD *)(a1 + 24 * (v2 - (unsigned __int64)v1)) + 16LL) == 13 )
  {
    if ( v1 == ++v2 )
      return 1;
  }
  return 0;
}
