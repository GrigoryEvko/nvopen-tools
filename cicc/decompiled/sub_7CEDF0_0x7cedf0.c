// Function: sub_7CEDF0
// Address: 0x7cedf0
//
_BOOL8 __fastcall sub_7CEDF0(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BOOL8 result; // rax
  char v5; // al
  char v6; // al
  char v7; // al
  unsigned int v8; // ecx
  int v9; // eax
  char v10; // al
  int v11; // eax
  int v12; // eax
  int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-20h]
  __int64 v15; // [rsp+0h] [rbp-20h]
  __int64 v16; // [rsp+0h] [rbp-20h]
  __int64 v17; // [rsp+8h] [rbp-18h]
  __int64 v18; // [rsp+8h] [rbp-18h]
  __int64 v19; // [rsp+8h] [rbp-18h]

  if ( (*(_BYTE *)(a4 + 83) & 0x40) != 0 && !a1[4] && !a1[3]
    || *(_DWORD *)(a3 + 40) != *(_DWORD *)(a2 + 24)
    || dword_4F04BA0[*(unsigned __int8 *)(a3 + 80)] != 2 )
  {
    return 0;
  }
  if ( !*a1 )
    goto LABEL_9;
  v7 = *(_BYTE *)(a4 + 80);
  if ( v7 == 19 )
  {
    if ( !a1[2] )
      goto LABEL_17;
    goto LABEL_13;
  }
  if ( (unsigned __int8)(v7 - 4) <= 1u )
    goto LABEL_17;
  if ( v7 != 3 )
  {
    if ( v7 == 23 )
      goto LABEL_38;
    goto LABEL_32;
  }
  v16 = a3;
  v19 = a4;
  v12 = sub_8D3A70(*(_QWORD *)(a4 + 88));
  a4 = v19;
  a3 = v16;
  if ( !v12 )
  {
    v7 = *(_BYTE *)(v19 + 80);
    if ( v7 == 23 )
      goto LABEL_38;
    if ( v7 != 3 )
    {
LABEL_32:
      if ( !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
        goto LABEL_34;
      goto LABEL_52;
    }
    v13 = sub_8D3D40(*(_QWORD *)(v19 + 88));
    a4 = v19;
    a3 = v16;
    if ( !v13 && dword_4F077BC && qword_4F077A8 > 0x76BFu )
    {
LABEL_52:
      if ( !dword_4D044A0 )
        return 0;
      v7 = *(_BYTE *)(a4 + 80);
LABEL_34:
      if ( v7 != 6 )
      {
        if ( v7 != 3 )
          return 0;
        v14 = a3;
        v17 = a4;
        if ( !(unsigned int)sub_8D2870(*(_QWORD *)(a4 + 88)) )
          return 0;
        a4 = v17;
        a3 = v14;
        goto LABEL_9;
      }
LABEL_38:
      if ( a1[2] )
        return 0;
      goto LABEL_17;
    }
  }
LABEL_9:
  if ( a1[2] )
  {
    v5 = *(_BYTE *)(a4 + 80);
    if ( (unsigned __int8)(v5 - 4) > 1u )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 19 )
          return 0;
        goto LABEL_13;
      }
      v15 = a3;
      v18 = a4;
      v9 = sub_8D3A70(*(_QWORD *)(a4 + 88));
      a4 = v18;
      a3 = v15;
      if ( !v9 )
      {
        v10 = *(_BYTE *)(v18 + 80);
        if ( v10 != 19 )
        {
          if ( v10 != 3 )
            return 0;
          v11 = sub_8D3D40(*(_QWORD *)(v18 + 88));
          a4 = v18;
          a3 = v15;
          if ( !v11 && (*(_BYTE *)(v18 + 81) & 0x40) == 0 )
            return 0;
          goto LABEL_17;
        }
LABEL_13:
        if ( !a1[1] )
          goto LABEL_26;
        v6 = 19;
LABEL_19:
        if ( dword_4F077BC )
        {
          if ( qword_4F077A8 > 0x9E33u && v6 == 3 )
          {
            if ( *(_BYTE *)(a4 + 104) )
              goto LABEL_26;
            if ( a1[3] )
              return 0;
          }
          else if ( a1[3] && qword_4F077A8 > 0x9E33u )
          {
            return 0;
          }
        }
        if ( unk_4D04234 && v6 == 3 )
          goto LABEL_26;
        return 0;
      }
    }
  }
LABEL_17:
  if ( a1[1] )
  {
    v6 = *(_BYTE *)(a4 + 80);
    if ( (unsigned __int8)(v6 - 4) > 2u )
      goto LABEL_19;
  }
LABEL_26:
  result = 1;
  if ( a1[6] )
  {
    v8 = a1[8];
    result = 1;
    if ( v8 )
      return v8 >= *(_DWORD *)(a3 + 44);
  }
  return result;
}
