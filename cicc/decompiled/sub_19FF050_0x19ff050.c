// Function: sub_19FF050
// Address: 0x19ff050
//
__int64 __fastcall sub_19FF050(__int64 a1, int a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  char v4; // dl

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 <= 0x17u )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v3 || *(_QWORD *)(v3 + 8) || a2 != v2 - 24 )
    return 0;
  v4 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v4 == 16 )
    v4 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
  if ( (unsigned __int8)(v4 - 1) > 5u && v2 != 76 )
    return a1;
  if ( sub_15F2480(a1) )
    return a1;
  return 0;
}
