// Function: sub_2E211B0
// Address: 0x2e211b0
//
__int64 __fastcall sub_2E211B0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r9
  unsigned int v3; // ecx
  __int16 *i; // rdx
  int v5; // esi

  v2 = *(_QWORD *)(*a1 + 8LL);
  v3 = *(_DWORD *)(v2 + 24LL * a2 + 16) & 0xFFF;
  for ( i = (__int16 *)(*(_QWORD *)(*a1 + 56LL) + 2LL * (*(_DWORD *)(v2 + 24LL * a2 + 16) >> 12)); ; ++i )
  {
    if ( !i )
      return 0;
    if ( *(_DWORD *)(a1[6] + 216LL * v3 + 204) )
      break;
    v5 = *i;
    v3 += v5;
    if ( !(_WORD)v5 )
      return 0;
  }
  return 1;
}
