// Function: sub_1498160
// Address: 0x1498160
//
__int64 __fastcall sub_1498160(__int64 a1, __int64 a2)
{
  __int16 i; // ax

  while ( 1 )
  {
    for ( i = *(_WORD *)(a2 + 24); i == 7; i = *(_WORD *)(a2 + 24) )
      a2 = **(_QWORD **)(a2 + 32);
    if ( i != 4 )
      break;
    a2 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a2 + 40) - 1));
    if ( *(_BYTE *)(sub_1456040(a2) + 8) != 15 )
      return 0;
  }
  if ( i != 10 )
    return 0;
  return *(_QWORD *)(a2 - 8);
}
