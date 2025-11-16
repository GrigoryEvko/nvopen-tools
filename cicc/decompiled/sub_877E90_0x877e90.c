// Function: sub_877E90
// Address: 0x877e90
//
void __fastcall sub_877E90(__int64 a1, __int64 a2, __int64 a3)
{
  if ( a3 )
  {
    while ( (*(_BYTE *)(a3 + 124) & 1) != 0 )
      a3 = *(_QWORD *)(a3 + 128);
  }
  else
  {
    if ( (int)dword_4F04C5C <= 0 )
      return;
    if ( dword_4F04C5C > dword_4F04C34 )
      return;
    a3 = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 184) + 32LL);
    if ( !a3 )
      return;
  }
  if ( a1 )
  {
    *(_BYTE *)(a1 + 81) &= ~0x10u;
    *(_QWORD *)(a1 + 64) = a3;
  }
  if ( a2 )
  {
    *(_BYTE *)(a2 + 89) &= ~4u;
    *(_QWORD *)(a2 + 40) = *(_QWORD *)(a3 + 128);
  }
}
