// Function: sub_71CA50
// Address: 0x71ca50
//
_BOOL8 __fastcall sub_71CA50(__int64 a1, _DWORD *a2, int a3, __int64 a4, int a5, __int64 a6)
{
  __int64 i; // rbx
  __int64 v9; // r15
  __int64 j; // r14
  int v11; // r9d
  _BOOL4 v12; // r9d
  char v14; // dl
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rcx
  int v20; // eax
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdx
  char v26; // al
  __int64 v27; // rax
  _BOOL4 v28; // [rsp+Ch] [rbp-44h]
  _BOOL4 v29; // [rsp+Ch] [rbp-44h]
  _BOOL4 v30; // [rsp+Ch] [rbp-44h]
  _BOOL4 v32; // [rsp+18h] [rbp-38h]

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v9 = *(_QWORD *)(i + 160);
  for ( j = v9; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v11 = sub_8D2600(j);
  if ( !v11 )
  {
    v14 = *(_BYTE *)(j + 140);
    if ( v14 == 12 )
    {
      v15 = j;
      do
      {
        v15 = *(_QWORD *)(v15 + 160);
        v14 = *(_BYTE *)(v15 + 140);
      }
      while ( v14 == 12 );
    }
    if ( a5 || !v14 )
      return 1;
    if ( dword_4F077C4 == 2 )
    {
      v24 = sub_8D23B0(j);
      v11 = 0;
      if ( v24 )
      {
        sub_8AE000(j);
        v11 = 0;
      }
    }
    v28 = v11;
    if ( a3 )
    {
      v16 = sub_8D23B0(j);
      v12 = v28;
      if ( !v16 )
      {
        if ( !dword_4D041AC && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0xC34Fu)
          || (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u
          || (*(_BYTE *)(j + 176) & 0x20) == 0 )
        {
          return 1;
        }
        v19 = (__int64)a2;
        if ( !a2 )
          return v12;
        goto LABEL_38;
      }
      if ( !dword_4D04964 && a6 && (*(_BYTE *)(a6 + 195) & 8) != 0 )
      {
        if ( a2 )
          sub_685330(0x542u, a2, v9);
        return 1;
      }
      if ( !a2 )
        return v12;
      v25 = *(_QWORD *)(i + 168);
      v26 = *(_BYTE *)(v25 + 17);
      if ( (v26 & 8) != 0 )
      {
        *(_BYTE *)(v25 + 17) = v26 | 8;
        return v12;
      }
      *(_BYTE *)(v25 + 17) = v26 | 8;
    }
    else
    {
      v20 = sub_8D25A0(j);
      v12 = v28;
      if ( v20 && (v21 = sub_8D3410(j), v12 = v28, !v21) || (v29 = v12, v22 = sub_8D32E0(j), v12 = v29, v22) )
      {
        if ( !dword_4D041AC || (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u || (*(_BYTE *)(j + 176) & 0x20) == 0 )
          return 1;
        if ( !a2 )
          return v12;
        v19 = (__int64)a2;
LABEL_38:
        sub_5EB950(8u, 323, v9, v19);
        return 0;
      }
      if ( !a2 )
        return v12;
      if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u || (v23 = sub_8D23B0(j), v12 = v29, !v23) )
      {
        sub_6851C0(0x61u, a2);
        v27 = sub_72C930(97);
        v12 = 0;
        *(_QWORD *)(i + 160) = v27;
        return v12;
      }
    }
    v30 = v12;
    sub_625A80(v9, (__int64)a2, (_QWORD *)a6, v17, v18);
    return v30;
  }
  if ( (*(_BYTE *)(v9 + 140) & 0xFB) != 8
    || !(unsigned int)sub_8D4C10(v9, dword_4F077C4 != 2)
    || a3
    || dword_4F077C4 == 2
    || !dword_4D04964 )
  {
    return 1;
  }
  v12 = byte_4F07472[0] != 8;
  if ( a2 )
  {
    v32 = byte_4F07472[0] != 8;
    sub_684AC0(byte_4F07472[0], 0x330u);
    return v32;
  }
  return v12;
}
