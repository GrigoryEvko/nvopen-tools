// Function: sub_7CF4E0
// Address: 0x7cf4e0
//
_BOOL8 __fastcall sub_7CF4E0(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BOOL8 result; // rax
  char v6; // al
  int v7; // eax
  char v8; // al
  char v9; // al
  char v10; // dl
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  int v15; // eax
  char v16; // al
  int v17; // eax
  int v18; // eax
  int v19; // eax
  __int64 v20; // [rsp-28h] [rbp-28h]
  __int64 v21; // [rsp-28h] [rbp-28h]
  __int64 v22; // [rsp-28h] [rbp-28h]
  __int64 v23; // [rsp-20h] [rbp-20h]
  __int64 v24; // [rsp-20h] [rbp-20h]
  __int64 v25; // [rsp-20h] [rbp-20h]

  if ( (*(_BYTE *)(a3 + 81) & 0x10) == 0 )
    return 0;
  if ( *(_BYTE *)(a3 + 80) != 3
    || !*(_BYTE *)(a3 + 104)
    || (v11 = a1[11]) != 0 && dword_4D0489C
    || dword_4F077BC
    && (a1[15]
     || !(*a1 | v11)
     && (a1[14]
      || qword_4F077A8 <= 0x76BFu
      || qword_4F077A8 <= 0x9E33u && (a1[16] || a1[17])
      || a1[18]
      && *(_BYTE *)(a4 + 80) == 3
      && *(_BYTE *)(a4 + 104)
      && (v12 = *(_QWORD *)(a4 + 88), (*(_BYTE *)(v12 + 177) & 0x10) != 0)
      && *(_QWORD *)(*(_QWORD *)(v12 + 168) + 168LL)))
    || a1[10] )
  {
    if ( *(_QWORD *)(a3 + 64) != a2 )
      return 0;
    if ( !a1[1] )
      goto LABEL_18;
  }
  else
  {
    if ( !a1[1] )
    {
      if ( !a1[3] )
      {
        if ( !a1[2] )
        {
          v13 = *(_QWORD *)(a4 + 88);
          if ( a2 == v13 )
            return 0;
          if ( a2 )
          {
            if ( v13 )
            {
              if ( dword_4F07588 )
              {
                v14 = *(_QWORD *)(a2 + 32);
                if ( *(_QWORD *)(v13 + 32) == v14 )
                {
                  if ( v14 )
                    return 0;
                }
              }
            }
          }
        }
        if ( a2 != *(_QWORD *)(a3 + 64) )
          return 0;
        goto LABEL_26;
      }
      if ( a2 != *(_QWORD *)(a3 + 64) )
        return 0;
      goto LABEL_20;
    }
    if ( a2 != *(_QWORD *)(a3 + 64) )
      return 0;
  }
  v6 = *(_BYTE *)(a4 + 80);
  if ( v6 != 19 )
  {
    if ( (unsigned __int8)(v6 - 4) <= 1u )
      goto LABEL_26;
    if ( v6 == 3 )
    {
      v22 = a3;
      v25 = a4;
      v18 = sub_8D3A70(*(_QWORD *)(a4 + 88));
      a4 = v25;
      a3 = v22;
      if ( !v18 )
      {
        v6 = *(_BYTE *)(v25 + 80);
        if ( v6 != 23 )
        {
          if ( v6 != 3 )
            goto LABEL_12;
          v19 = sub_8D3D40(*(_QWORD *)(v25 + 88));
          a4 = v25;
          a3 = v22;
          if ( !v19 && dword_4F077BC && qword_4F077A8 > 0x76BFu )
            goto LABEL_89;
        }
      }
    }
    else if ( v6 != 23 )
    {
LABEL_12:
      if ( !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
      {
LABEL_14:
        if ( v6 == 6 )
        {
          if ( a1[3] )
            return 0;
          goto LABEL_26;
        }
        if ( v6 != 3 )
          return 0;
        v20 = a3;
        v23 = a4;
        if ( !(unsigned int)sub_8D2870(*(_QWORD *)(a4 + 88)) )
          return 0;
        v7 = a1[3];
        a3 = v20;
        a4 = v23;
        goto LABEL_19;
      }
LABEL_89:
      if ( !dword_4D044A0 )
        return 0;
      v6 = *(_BYTE *)(a4 + 80);
      goto LABEL_14;
    }
  }
LABEL_18:
  v7 = a1[3];
LABEL_19:
  if ( !v7 )
    goto LABEL_26;
LABEL_20:
  v8 = *(_BYTE *)(a4 + 80);
  if ( (unsigned __int8)(v8 - 4) <= 1u )
    goto LABEL_26;
  if ( v8 != 3 )
  {
    if ( v8 != 19 )
      return 0;
    goto LABEL_23;
  }
  v21 = a3;
  v24 = a4;
  v15 = sub_8D3A70(*(_QWORD *)(a4 + 88));
  a4 = v24;
  a3 = v21;
  if ( v15 )
    goto LABEL_26;
  v16 = *(_BYTE *)(v24 + 80);
  if ( v16 == 19 )
  {
LABEL_23:
    if ( !a1[2] )
      goto LABEL_37;
    v9 = 19;
    goto LABEL_28;
  }
  if ( v16 != 3 )
    return 0;
  v17 = sub_8D3D40(*(_QWORD *)(v24 + 88));
  a4 = v24;
  a3 = v21;
  if ( !v17 && (*(_BYTE *)(v24 + 81) & 0x40) == 0 )
    return 0;
LABEL_26:
  if ( !a1[2] )
    goto LABEL_37;
  v9 = *(_BYTE *)(a4 + 80);
  if ( (unsigned __int8)(v9 - 4) <= 2u )
    goto LABEL_37;
LABEL_28:
  if ( !dword_4F077BC )
    goto LABEL_33;
  if ( qword_4F077A8 <= 0x9E33u || v9 != 3 )
  {
    if ( a1[9] && qword_4F077A8 > 0x9E33u )
    {
LABEL_35:
      if ( v9 != 19 )
        goto LABEL_36;
      goto LABEL_37;
    }
LABEL_33:
    if ( unk_4D04234 && v9 == 3 )
      goto LABEL_37;
    goto LABEL_35;
  }
  if ( !*(_BYTE *)(a4 + 104) && (a1[9] || !unk_4D04234) )
  {
LABEL_36:
    if ( (*(_WORD *)(a4 + 80) & 0x40FF) == 0x4003 )
      goto LABEL_37;
    return 0;
  }
LABEL_37:
  if ( (*(_BYTE *)(a3 + 83) & 0x40) != 0 && *(_BYTE *)(a3 + 80) != 16 && (*(_BYTE *)(a3 + 82) & 0x10) == 0 )
    return 0;
  result = 1;
  if ( a1[15] )
  {
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 )
        {
          v10 = *(_BYTE *)(a4 + 80);
          if ( (unsigned __int8)(v10 - 3) > 3u && v10 != 19 )
            return (*(_WORD *)(a4 + 80) & 0x40FF) == 16387;
        }
      }
    }
  }
  return result;
}
