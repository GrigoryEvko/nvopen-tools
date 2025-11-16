// Function: sub_2F422A0
// Address: 0x2f422a0
//
__int64 __fastcall sub_2F422A0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rax
  unsigned int v5; // ecx
  __int16 *i; // rdx
  int v7; // esi

  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(_QWORD *)(v2 + 8);
  v4 = *(_DWORD *)(v3 + 24LL * a2 + 16) >> 12;
  v5 = *(_DWORD *)(v3 + 24LL * a2 + 16) & 0xFFF;
  for ( i = (__int16 *)(*(_QWORD *)(v2 + 56) + 2 * v4); ; ++i )
  {
    if ( !i )
      return 1;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 808) + 4LL * v5) )
      break;
    v7 = *i;
    v5 += v7;
    if ( !(_WORD)v7 )
      return 1;
  }
  return 0;
}
