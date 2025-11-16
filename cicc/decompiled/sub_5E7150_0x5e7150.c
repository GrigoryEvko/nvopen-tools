// Function: sub_5E7150
// Address: 0x5e7150
//
__int64 __fastcall sub_5E7150(__int64 a1)
{
  unsigned int v1; // r8d

  v1 = 0;
  if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x10 )
  {
    if ( *(_BYTE *)(a1 + 140) != 2 || (v1 = 1, (*(_BYTE *)(a1 + 161) & 8) == 0) )
    {
      v1 = 1;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL) && unk_4D04734 == 3 )
        return *(_BYTE *)(a1 + 178) & 1;
    }
  }
  return v1;
}
