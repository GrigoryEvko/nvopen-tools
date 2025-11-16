// Function: sub_69A7F0
// Address: 0x69a7f0
//
__int64 __fastcall sub_69A7F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 i; // rax

  if ( !unk_4F04C50 )
    return 0;
  v1 = *(_QWORD *)(unk_4F04C50 + 32LL);
  if ( !v1 )
    return 0;
  if ( (*(_BYTE *)(v1 + 198) & 0x18) == 0x10 )
    return 0;
  if ( !a1 )
    return 0;
  v2 = sub_724D80(6);
  if ( (*(_BYTE *)(a1 + 195) & 8) != 0 )
    return 0;
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
    sub_73F170(a1, v2);
  else
    sub_72D3B0(a1, v2, 1);
  return v2;
}
