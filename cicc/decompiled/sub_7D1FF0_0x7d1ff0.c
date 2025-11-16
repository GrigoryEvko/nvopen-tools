// Function: sub_7D1FF0
// Address: 0x7d1ff0
//
_BOOL8 __fastcall sub_7D1FF0(__int64 a1, __int16 a2)
{
  __int64 v2; // rbx
  char v3; // al
  char v4; // al
  char v6; // al
  char v7; // al

  v2 = a1;
  v3 = *(_BYTE *)(a1 + 80);
  if ( v3 == 16 )
  {
    v2 = **(_QWORD **)(a1 + 88);
    v3 = *(_BYTE *)(v2 + 80);
  }
  if ( v3 == 24 )
    v2 = *(_QWORD *)(v2 + 88);
  if ( (a2 & 1) == 0 )
    goto LABEL_15;
  v4 = *(_BYTE *)(v2 + 80);
  if ( v4 == 19 )
    goto LABEL_32;
  if ( (unsigned __int8)(v4 - 4) <= 1u )
    goto LABEL_15;
  if ( v4 != 3 )
  {
    if ( v4 != 23 )
      goto LABEL_10;
LABEL_32:
    if ( (a2 & 2) != 0 )
      goto LABEL_22;
LABEL_16:
    if ( (a2 & 0x200) != 0 && *(_BYTE *)(v2 + 80) != 23 )
      return 0;
    goto LABEL_17;
  }
  if ( (unsigned int)sub_8D3A70(*(_QWORD *)(v2 + 88)) )
    goto LABEL_15;
  v4 = *(_BYTE *)(v2 + 80);
  if ( v4 == 23 )
    goto LABEL_32;
  if ( v4 == 3 )
  {
    if ( (unsigned int)sub_8D3D40(*(_QWORD *)(v2 + 88)) || !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
      goto LABEL_15;
    goto LABEL_45;
  }
LABEL_10:
  if ( dword_4F077BC && qword_4F077A8 > 0x76BFu )
  {
LABEL_45:
    if ( !dword_4D044A0 )
      return 0;
    v4 = *(_BYTE *)(v2 + 80);
  }
  if ( v4 == 6 )
    goto LABEL_42;
  if ( v4 != 3 || !(unsigned int)sub_8D2870(*(_QWORD *)(v2 + 88)) )
    return 0;
LABEL_15:
  if ( (a2 & 2) == 0 )
    goto LABEL_16;
  v4 = *(_BYTE *)(v2 + 80);
  if ( (unsigned __int8)(v4 - 4) <= 2u )
    goto LABEL_16;
LABEL_22:
  if ( !dword_4F077BC )
    goto LABEL_23;
  if ( qword_4F077A8 <= 0x9E33u || v4 != 3 )
  {
    if ( (a2 & 0x4000) != 0 && qword_4F077A8 > 0x9E33u )
    {
LABEL_25:
      if ( v4 != 19 )
        goto LABEL_26;
      goto LABEL_42;
    }
LABEL_23:
    if ( unk_4D04234 && v4 == 3 )
      goto LABEL_42;
    goto LABEL_25;
  }
  if ( !*(_BYTE *)(v2 + 104) && ((a2 & 0x4000) != 0 || !unk_4D04234) )
  {
LABEL_26:
    if ( (*(_WORD *)(v2 + 80) & 0x40FF) != 0x4003 )
      return 0;
    goto LABEL_16;
  }
LABEL_42:
  if ( (a2 & 0x200) != 0 )
    return 0;
LABEL_17:
  if ( (a2 & 0x800) == 0 )
    return 1;
  v6 = *(_BYTE *)(v2 + 80);
  if ( (unsigned __int8)(v6 - 4) <= 1u )
    return 1;
  if ( v6 != 3 )
    return v6 == 19;
  if ( (unsigned int)sub_8D3A70(*(_QWORD *)(v2 + 88)) )
    return 1;
  v7 = *(_BYTE *)(v2 + 80);
  if ( v7 == 19 )
    return 1;
  if ( v7 != 3 )
    return 0;
  if ( (unsigned int)sub_8D3D40(*(_QWORD *)(v2 + 88)) )
    return 1;
  return (*(_BYTE *)(v2 + 81) & 0x40) != 0;
}
