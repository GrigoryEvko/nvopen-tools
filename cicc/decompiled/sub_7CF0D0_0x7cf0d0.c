// Function: sub_7CF0D0
// Address: 0x7cf0d0
//
_BOOL8 __fastcall sub_7CF0D0(__int64 a1, __int64 a2, _DWORD *a3)
{
  char v4; // al
  _BOOL8 result; // rax
  char v6; // al
  unsigned int v7; // edx
  char v8; // al

  if ( !*a3 )
    goto LABEL_12;
  v4 = *(_BYTE *)(a2 + 80);
  if ( v4 == 19 )
    goto LABEL_26;
  if ( (unsigned __int8)(v4 - 4) <= 1u )
    goto LABEL_12;
  if ( v4 != 3 )
  {
    if ( v4 != 23 )
    {
LABEL_6:
      if ( !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
        goto LABEL_7;
      goto LABEL_55;
    }
LABEL_26:
    if ( !a3[1] )
      goto LABEL_14;
LABEL_27:
    if ( !dword_4F077BC )
      goto LABEL_46;
    if ( qword_4F077A8 > 0x9E33u && v4 == 3 )
    {
      if ( *(_BYTE *)(a2 + 104) || !a3[8] && unk_4D04234 )
        goto LABEL_33;
LABEL_48:
      if ( (*(_WORD *)(a2 + 80) & 0x40FF) != 0x4003 )
        return 0;
      if ( !a3[3] )
        goto LABEL_35;
      goto LABEL_16;
    }
    if ( !a3[8] || qword_4F077A8 <= 0x9E33u )
    {
LABEL_46:
      if ( unk_4D04234 && v4 == 3 )
      {
LABEL_33:
        if ( !a3[3] )
          goto LABEL_35;
        goto LABEL_34;
      }
    }
    if ( v4 == 19 )
      goto LABEL_14;
    goto LABEL_48;
  }
  if ( (unsigned int)sub_8D3A70(*(_QWORD *)(a2 + 88)) )
    goto LABEL_12;
  v4 = *(_BYTE *)(a2 + 80);
  if ( v4 == 23 )
    goto LABEL_26;
  if ( v4 != 3 )
    goto LABEL_6;
  if ( (unsigned int)sub_8D3D40(*(_QWORD *)(a2 + 88)) || !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
    goto LABEL_12;
LABEL_55:
  if ( !dword_4D044A0 )
    return 0;
  v4 = *(_BYTE *)(a2 + 80);
LABEL_7:
  if ( v4 != 6 )
  {
    if ( v4 != 3 || !(unsigned int)sub_8D2870(*(_QWORD *)(a2 + 88)) )
      return 0;
LABEL_12:
    if ( !a3[1] || (v4 = *(_BYTE *)(a2 + 80), (unsigned __int8)(v4 - 4) <= 2u) )
    {
LABEL_14:
      if ( !a3[3] )
        goto LABEL_35;
      v4 = *(_BYTE *)(a2 + 80);
      if ( (unsigned __int8)(v4 - 4) <= 1u )
        goto LABEL_35;
LABEL_16:
      if ( v4 != 3 )
        goto LABEL_17;
LABEL_34:
      if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(a2 + 88)) )
      {
        v8 = *(_BYTE *)(a2 + 80);
        if ( v8 == 19 )
        {
LABEL_18:
          if ( a3[2] )
            return 0;
          goto LABEL_19;
        }
        if ( v8 != 3 || !(unsigned int)sub_8D3D40(*(_QWORD *)(a2 + 88)) && (*(_BYTE *)(a2 + 81) & 0x40) == 0 )
          return 0;
      }
      goto LABEL_35;
    }
    goto LABEL_27;
  }
  if ( a3[3] )
  {
LABEL_17:
    if ( *(_BYTE *)(a2 + 80) != 19 )
      return 0;
    goto LABEL_18;
  }
LABEL_35:
  if ( a3[2] )
  {
    if ( *(_BYTE *)(a2 + 80) != 23 )
      return 0;
    if ( a3[8] )
      goto LABEL_20;
    goto LABEL_38;
  }
LABEL_19:
  if ( a3[8] )
  {
LABEL_20:
    if ( !a3[7] )
      goto LABEL_23;
    v6 = *(_BYTE *)(a2 + 80);
    if ( (v6 & 0xEF) != 3 && (dword_4F077C4 != 2 || (unsigned __int8)(v6 - 4) > 2u) )
      return 0;
    if ( !a3[6] )
      goto LABEL_23;
LABEL_41:
    if ( v6 == 19 )
      goto LABEL_23;
    return 0;
  }
LABEL_38:
  if ( a3[6] && a3[7] )
  {
    v6 = *(_BYTE *)(a2 + 80);
    goto LABEL_41;
  }
LABEL_23:
  result = 1;
  if ( a3[34] )
  {
    v7 = a3[32];
    if ( v7 )
      return v7 >= *(_DWORD *)(a1 + 44);
  }
  return result;
}
