// Function: sub_1ED88E0
// Address: 0x1ed88e0
//
__int64 __fastcall sub_1ED88E0(int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int16 v5; // dx

  if ( a1 < 0 )
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 272) + 8LL * (unsigned int)a1);
  if ( !v3 )
    return 1;
  if ( (*(_BYTE *)(v3 + 4) & 8) == 0 )
  {
LABEL_5:
    v4 = *(_QWORD *)(v3 + 16);
LABEL_6:
    if ( a2 != v4 )
    {
      v5 = **(_WORD **)(v4 + 16);
      if ( v5 == 15 || v5 == 10 )
        return 0;
    }
    while ( 1 )
    {
      v3 = *(_QWORD *)(v3 + 32);
      if ( !v3 )
        return 1;
      if ( (*(_BYTE *)(v3 + 4) & 8) == 0 && *(_QWORD *)(v3 + 16) != v4 )
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
      return 1;
    if ( (*(_BYTE *)(v3 + 4) & 8) == 0 )
      goto LABEL_5;
  }
}
