// Function: sub_2F4D680
// Address: 0x2f4d680
//
__int64 __fastcall sub_2F4D680(__int64 a1, int a2)
{
  __int64 v2; // rax

  if ( a2 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  if ( !v2 )
    return 0;
  if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
  {
    v2 = *(_QWORD *)(v2 + 32);
    if ( !v2 || (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
      return 0;
  }
  while ( (*(_WORD *)(v2 + 2) & 0xFF0) == 0 )
  {
    v2 = *(_QWORD *)(v2 + 32);
    if ( !v2 || (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
      return 0;
  }
  return 1;
}
