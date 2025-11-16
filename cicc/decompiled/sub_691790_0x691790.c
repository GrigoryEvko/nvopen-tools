// Function: sub_691790
// Address: 0x691790
//
char __fastcall sub_691790(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // r12
  unsigned int v5; // ebx
  bool v6; // r15
  bool v7; // r14
  __int64 v8; // rax
  __int64 v9; // r8
  char v10; // al
  _BOOL4 v11; // edx
  bool v12; // si
  __int64 v13; // rcx
  _BOOL4 v14; // r10d
  __int16 v15; // di
  __int64 v16; // r9
  __int64 v17; // rdx
  char *v18; // rdx
  char *v19; // rcx
  unsigned int v20; // r14d
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  _BOOL4 v26; // [rsp+Ch] [rbp-44h]
  _BOOL4 v27; // [rsp+14h] [rbp-3Ch]
  int v28; // [rsp+18h] [rbp-38h]
  int v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  _BOOL4 v31; // [rsp+18h] [rbp-38h]

  v4 = a1;
  v5 = a2;
  if ( a1 )
  {
    v6 = (*(_BYTE *)(a1 + 198) & 0x18) != 16;
    v7 = (*(_BYTE *)(a1 + 198) & 0x30) == 16;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  LOBYTE(v8) = dword_4F04C64;
  if ( unk_4F04C48 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0
    && unk_4D03B90 != -1
    && (*(_BYTE *)(unk_4D03B98 + 176LL * unk_4D03B90 + 5) & 8) != 0 )
  {
    return v8;
  }
  if ( dword_4F04C64 != -1 )
  {
    v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v8 + 14) & 2) != 0 )
      return v8;
  }
  LOBYTE(v8) = qword_4D03C50;
  if ( qword_4D03C50 )
  {
    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0 )
      return v8;
  }
  if ( dword_4F04C58 == -1 )
  {
    v11 = 0;
    v13 = 0;
    v14 = 1;
    v9 = 0;
    LOBYTE(v8) = 0;
    if ( !a1 )
    {
LABEL_45:
      v16 = v5;
LABEL_46:
      if ( !v5 )
        return v8;
LABEL_47:
      if ( (unsigned int)v13 | v11 )
      {
LABEL_48:
        v8 = (__int64)&qword_4D045BC;
        if ( qword_4D045BC )
          goto LABEL_49;
        v28 = v16;
        LODWORD(v8) = sub_6E5430(a1, a2, 0, v13, v9, v16);
        LODWORD(v16) = v28;
        if ( !(_DWORD)v8 )
          goto LABEL_49;
        goto LABEL_60;
      }
LABEL_49:
      if ( (_DWORD)v16 )
        return v8;
      v18 = "__device__";
      if ( !v7 )
        v18 = "host";
      goto LABEL_52;
    }
  }
  else
  {
    v9 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
    v10 = *(_BYTE *)(v9 + 198);
    v11 = (v10 & 0x20) != 0;
    v12 = (v10 & 0x30) == 16;
    v13 = v12;
    v14 = (v10 & 0x18) != 16;
    LOBYTE(v8) = 0;
    if ( (*(_BYTE *)(v9 - 8) & 0x10) == 0 )
    {
      LOBYTE(v8) = 1;
      if ( a1 )
      {
        if ( !*(_BYTE *)(a1 + 174) )
        {
          v15 = *(_WORD *)(a1 + 176);
          LOBYTE(v8) = v12 && v15 != 0;
          if ( (_BYTE)v8 )
          {
            if ( v15 == 25724 || v15 == 3421 )
            {
              LOBYTE(v8) = v15 == 25724 || v15 == 3421;
              if ( unk_4D045E8 <= 0x33u )
              {
                v26 = v14;
                v27 = v11;
                v30 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
                sub_6851C0(0xE64u, a3);
                v9 = v30;
                v11 = v27;
                v13 = v12;
                v14 = v26;
                LOBYTE(v8) = v15 == 25724 || v15 == 3421;
              }
            }
          }
          else
          {
            LOBYTE(v8) = 1;
          }
        }
      }
    }
    if ( (*(_BYTE *)(v9 + 193) & 0x10) != 0 )
      return v8;
    a1 = 0x8000000000000LL;
    a2 = *(_QWORD *)(v9 + 200) & 0x8000001000000LL;
    if ( a2 == 0x8000000000000LL && (*(_BYTE *)(v9 + 192) & 2) == 0 )
      return v8;
    if ( !v4 )
      goto LABEL_45;
  }
  a2 = *(unsigned __int8 *)(v4 + 193);
  if ( (a2 & 0x10) != 0 )
    return v8;
  a1 = *(_QWORD *)(v4 + 200) & 0x8000001000000LL;
  if ( a1 == 0x8000000000000LL && (*(_BYTE *)(v4 + 192) & 2) == 0 )
    return v8;
  if ( !(_BYTE)v8 )
    goto LABEL_45;
  LOBYTE(v8) = *(_BYTE *)(v9 + 197);
  v16 = (*(_BYTE *)(v4 + 198) & 0x20) != 0;
  if ( (v8 & 0x10) != 0 )
  {
    if ( v5 )
      goto LABEL_47;
    goto LABEL_41;
  }
  if ( !(v11 | (unsigned int)v13) )
  {
    if ( v5 )
      goto LABEL_49;
    goto LABEL_41;
  }
  if ( v7 || !v6 )
  {
    if ( v5 )
      goto LABEL_48;
LABEL_41:
    if ( (*(_BYTE *)(v4 + 198) & 0x20) == 0 )
      return v8;
    goto LABEL_54;
  }
  if ( (*(_BYTE *)(v4 + 198) & 0x20) != 0 )
  {
    if ( v5 )
    {
      v8 = (__int64)&qword_4D045BC + 4;
      if ( qword_4D045BC )
        return v8;
      v31 = (*(_BYTE *)(v4 + 198) & 0x20) != 0;
      LODWORD(v8) = sub_6E5430(a1, a2, 0, v13, v9, v16);
      LODWORD(v16) = v31;
      if ( !(_DWORD)v8 )
        return v8;
LABEL_60:
      v29 = v16;
      LOBYTE(v8) = sub_6851C0(0xD98u, a3);
      LODWORD(v16) = v29;
      goto LABEL_49;
    }
LABEL_54:
    v19 = "must";
    v18 = "__global__";
    goto LABEL_53;
  }
  if ( a1 == 0x8000000000000LL && (*(_BYTE *)(v4 + 192) & 2) == 0 )
  {
LABEL_73:
    if ( !v5 )
      return v8;
LABEL_74:
    v17 = (unsigned int)qword_4D045BC | HIDWORD(qword_4D045BC);
    if ( !qword_4D045BC )
    {
LABEL_75:
      if ( (unsigned int)sub_6E5430(a1, a2, v17, v13, v9, v16) )
        sub_6851C0(0xD98u, a3);
    }
LABEL_39:
    v18 = "host";
LABEL_52:
    v19 = "cannot";
LABEL_53:
    LOBYTE(v8) = sub_686610(0xDB4u, a3, (__int64)v18, (__int64)v19);
    return v8;
  }
  if ( (*(_BYTE *)(v9 + 193) & 0x10) != 0 )
    goto LABEL_46;
  a1 = 0x8000000000000LL;
  v13 = *(_QWORD *)(v9 + 200) & 0x8000001000000LL;
  if ( v13 == 0x8000000000000LL && (*(_BYTE *)(v9 + 192) & 2) == 0 )
    goto LABEL_73;
  if ( (a2 & 4) != 0 )
    goto LABEL_37;
  if ( (*(_BYTE *)(v4 + 192) & 2) == 0 )
  {
    if ( (a2 & 2) == 0 )
      goto LABEL_73;
    v13 = (__int64)&dword_4D04530;
    if ( dword_4D04530 )
      goto LABEL_73;
  }
  if ( v14 )
  {
    if ( (v8 & 8) == 0 )
    {
      v23 = sub_8258E0(v9, 0);
      v24 = sub_8258E0(v4, 1);
      a2 = (__int64)a3;
      a1 = (*(_BYTE *)(v4 + 193) & 2) == 0 ? 3468 : 3470;
      LOBYTE(v8) = sub_6865F0(a1, a3, v24, v23);
      if ( !v5 )
        return v8;
      goto LABEL_38;
    }
LABEL_37:
    if ( !v5 )
      return v8;
LABEL_38:
    v17 = HIDWORD(qword_4D045BC) | (unsigned int)qword_4D045BC;
    if ( !qword_4D045BC )
      goto LABEL_75;
    goto LABEL_39;
  }
  if ( v11 )
    v20 = 3476 - ((a2 & 2) == 0);
  else
    v20 = 3474 - ((a2 & 2) == 0);
  v21 = sub_8258E0(v9, 0);
  v22 = sub_8258E0(v4, 1);
  a2 = (__int64)a3;
  a1 = v20;
  LOBYTE(v8) = sub_686610(v20, a3, v22, v21);
  if ( v5 )
    goto LABEL_74;
  return v8;
}
