// Function: sub_2D106C0
// Address: 0x2d106c0
//
__int64 __fastcall sub_2D106C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx

  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 0 )
    return 0;
  v5 = 0;
  v6 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  while ( 1 )
  {
    v7 = a2 - v6;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v7 = *(_QWORD *)(a2 - 8);
    if ( a3 == *(_QWORD *)(v7 + v5) )
      break;
    v5 += 32;
    if ( v5 == v6 )
      return 0;
  }
  return 1;
}
