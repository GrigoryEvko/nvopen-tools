// Function: sub_6EE750
// Address: 0x6ee750
//
__int64 __fastcall sub_6EE750(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char v4; // dl

  if ( !a2 || *(_BYTE *)(a2 + 16) != 4 )
    return sub_72D2E0(a1, 0);
  v3 = *(_QWORD *)(a2 + 136);
  v4 = *(_BYTE *)(v3 + 80);
  if ( v4 == 16 )
  {
    v3 = **(_QWORD **)(v3 + 88);
    v4 = *(_BYTE *)(v3 + 80);
  }
  if ( v4 == 24 )
    v3 = *(_QWORD *)(v3 + 88);
  return sub_73F0A0(
           *(_QWORD *)(*(_QWORD *)(v3 + 88) + 152LL),
           *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 88) + 40LL) + 32LL),
           0);
}
