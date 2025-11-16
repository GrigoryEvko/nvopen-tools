// Function: sub_2FF8950
// Address: 0x2ff8950
//
__int64 __fastcall sub_2FF8950(__int64 a1, int a2, _DWORD *a3)
{
  __int64 v4; // rax
  unsigned int v5; // esi

  if ( (*(_DWORD *)(a1 + 40) & 0xFFFFFF) == 0 )
    return 0;
  v4 = *(_QWORD *)(a1 + 32);
  v5 = 0;
  while ( *(_BYTE *)v4
       || (*(_BYTE *)(v4 + 3) & 0x10) != 0
       || a2 != *(_DWORD *)(v4 + 8)
       || (*(_WORD *)(v4 + 2) & 0xFF0) == 0 )
  {
    ++v5;
    v4 += 40;
    if ( (*(_DWORD *)(a1 + 40) & 0xFFFFFF) == v5 )
      return 0;
  }
  *a3 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 40LL * (unsigned int)sub_2E89F40(a1, v5) + 8);
  return 1;
}
