// Function: sub_5E65C0
// Address: 0x5e65c0
//
__int64 __fastcall sub_5E65C0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  char v3; // al
  __int64 i; // r12
  char v5; // al

  v2 = a1;
  v3 = *(_BYTE *)(a1 + 80);
  if ( v3 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v3 = *(_BYTE *)(a1 + 80);
  }
  if ( v3 == 24 )
    a1 = *(_QWORD *)(a1 + 88);
  for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *a2 = i;
  v5 = sub_877F80();
  return sub_82EAF0(i, v2, v5 == 3);
}
