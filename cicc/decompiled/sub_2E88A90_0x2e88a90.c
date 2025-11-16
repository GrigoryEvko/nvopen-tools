// Function: sub_2E88A90
// Address: 0x2e88a90
//
bool __fastcall sub_2E88A90(__int64 a1, __int64 a2, int a3)
{
  if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & a2) == 0 )
    goto LABEL_5;
LABEL_2:
  if ( a3 == 1 )
    return 1;
  while ( (*(_BYTE *)(a1 + 44) & 8) != 0 )
  {
    a1 = *(_QWORD *)(a1 + 8);
    if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & a2) != 0 )
      goto LABEL_2;
LABEL_5:
    if ( a3 == 2 && *(_WORD *)(a1 + 68) != 21 )
      return 0;
  }
  return a3 == 2;
}
