// Function: sub_5E4F90
// Address: 0x5e4f90
//
__int64 __fastcall sub_5E4F90(__int64 a1)
{
  char v1; // al
  __int64 v2; // rsi
  __int64 v4; // rsi
  __int64 v5; // rdx

  v1 = *(_BYTE *)(a1 + 89);
  if ( (v1 & 1) != 0 )
  {
    if ( *(_BYTE *)(a1 + 140) != 9
      || (v5 = *(_QWORD *)(a1 + 168), (*(_BYTE *)(v5 + 109) & 0x20) == 0)
      || !*(_QWORD *)(v5 + 96) )
    {
      v2 = 3603;
      if ( dword_4CF7FB8 )
        v2 = 4 * (unsigned int)(dword_4CF7FB8 != 1) + 3606;
      sub_685260(7, v2, dword_4F07508, a1);
      v1 = *(_BYTE *)(a1 + 89);
    }
  }
  if ( (v1 & 4) == 0 || (unsigned __int8)((*(_BYTE *)(a1 + 88) & 3) - 1) > 1u )
    return 0;
  v4 = 3604;
  if ( dword_4CF7FB8 )
    v4 = 4 * (unsigned int)(dword_4CF7FB8 != 1) + 3607;
  sub_685260(7, v4, dword_4F07508, a1);
  return 0;
}
