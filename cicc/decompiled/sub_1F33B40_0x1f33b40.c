// Function: sub_1F33B40
// Address: 0x1f33b40
//
__int64 __fastcall sub_1F33B40(int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx

  if ( a1 < 0 )
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 272) + 8LL * (unsigned int)a1);
  if ( !v3 )
    return 0;
  if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
  {
LABEL_5:
    v4 = *(_QWORD *)(v3 + 16);
LABEL_6:
    if ( **(_WORD **)(v4 + 16) != 12 && a2 != *(_QWORD *)(v4 + 24) )
      return 1;
    while ( 1 )
    {
      v3 = *(_QWORD *)(v3 + 32);
      if ( !v3 )
        return 0;
      if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 && *(_QWORD *)(v3 + 16) != v4 )
      {
        v4 = *(_QWORD *)(v3 + 16);
        goto LABEL_6;
      }
    }
  }
  while ( 1 )
  {
    v3 = *(_QWORD *)(v3 + 32);
    if ( !v3 )
      return 0;
    if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
      goto LABEL_5;
  }
}
