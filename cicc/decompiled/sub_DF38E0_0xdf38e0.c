// Function: sub_DF38E0
// Address: 0xdf38e0
//
__int64 __fastcall sub_DF38E0(__int64 a1, __int64 a2)
{
  __int16 i; // ax

  while ( 1 )
  {
    for ( i = *(_WORD *)(a2 + 24); i == 8; i = *(_WORD *)(a2 + 24) )
      a2 = **(_QWORD **)(a2 + 32);
    if ( i != 5 )
      break;
    a2 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a2 + 40) - 1));
    if ( *(_BYTE *)(sub_D95540(a2) + 8) != 14 )
      return 0;
  }
  if ( i != 15 )
    return 0;
  return *(_QWORD *)(a2 - 8);
}
