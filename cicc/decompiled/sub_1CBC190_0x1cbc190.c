// Function: sub_1CBC190
// Address: 0x1cbc190
//
__int64 __fastcall sub_1CBC190(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx

  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 0 )
    return 0;
  v5 = 0;
  v6 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  while ( 1 )
  {
    v7 = a2 - v6;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v7 = *(_QWORD *)(a2 - 8);
    if ( a3 == *(_QWORD *)(v7 + v5) )
      break;
    v5 += 24;
    if ( v5 == v6 )
      return 0;
  }
  return 1;
}
