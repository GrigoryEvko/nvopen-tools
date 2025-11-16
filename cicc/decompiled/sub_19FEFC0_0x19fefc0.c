// Function: sub_19FEFC0
// Address: 0x19fefc0
//
__int64 __fastcall sub_19FEFC0(__int64 a1, int a2, int a3)
{
  unsigned __int8 v3; // al
  __int64 v4; // rcx
  int v5; // ecx
  char v6; // dl

  v3 = *(_BYTE *)(a1 + 16);
  if ( v3 <= 0x17u )
    return 0;
  v4 = *(_QWORD *)(a1 + 8);
  if ( !v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 8) )
    return 0;
  v5 = v3 - 24;
  if ( a2 != v5 && a3 != v5 )
    return 0;
  v6 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v6 == 16 )
    v6 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
  if ( (unsigned __int8)(v6 - 1) > 5u && v3 != 76 )
    return a1;
  if ( sub_15F2480(a1) )
    return a1;
  return 0;
}
