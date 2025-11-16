// Function: sub_861D10
// Address: 0x861d10
//
void __fastcall sub_861D10(__int64 a1, unsigned __int8 a2, __int64 *a3, int a4, int a5, __int16 a6)
{
  unsigned __int8 v7; // bl
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // rbx
  unsigned __int64 v11; // r14
  __int64 v12; // rax
  char v13; // dl
  _BOOL4 v14; // ecx
  int v15; // esi
  bool v16; // al
  bool v17; // si
  __int64 v18; // rbx
  _BOOL4 v19; // r13d
  bool v20; // r14
  bool v21; // al
  char v22; // al
  char v23; // dl
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // [rsp+8h] [rbp-68h]
  unsigned __int8 v28; // [rsp+14h] [rbp-5Ch]
  bool v30; // [rsp+2Fh] [rbp-41h]
  bool v32; // [rsp+3Ch] [rbp-34h]

  v7 = a2;
  v8 = *a3;
  if ( a1 && (((a2 - 15) & 0xFD) == 0 || a2 == 2) && !a5 )
    *(_QWORD *)(a1 + 248) = v8;
  v9 = a3[1];
  if ( !unk_4D04950 || dword_4F04C34 )
  {
LABEL_23:
    if ( !a4 )
      goto LABEL_27;
    if ( !v7 )
    {
      sub_5D0FF0();
      v27 = *qword_4D03FD0;
      if ( !*qword_4D03FD0 )
      {
LABEL_69:
        if ( v8 )
        {
          v14 = 0;
          goto LABEL_35;
        }
        v32 = v7 == 0;
        v20 = v7 == 0 || v7 == 3;
LABEL_81:
        v15 = dword_4F077C4;
        if ( v20 )
          goto LABEL_82;
        goto LABEL_95;
      }
      goto LABEL_26;
    }
  }
  else
  {
    if ( dword_4F04C64 > 0 )
    {
      v10 = 776LL * dword_4F04C64;
      v11 = 776 * (dword_4F04C64 - (unsigned __int64)(unsigned int)(dword_4F04C64 - 1));
      while ( *(_BYTE *)(v10 + qword_4F04C68[0] + 4) == 6 )
      {
        if ( (unsigned int)sub_8D3E20(*(_QWORD *)(v10 + qword_4F04C68[0] + 208)) )
        {
          v7 = a2;
          goto LABEL_23;
        }
        if ( v10 == v11 )
          break;
        v10 -= 776;
      }
      v7 = a2;
    }
    if ( v7 != 6 )
      goto LABEL_23;
    if ( v8 )
    {
      v12 = v8;
      do
      {
        v13 = *(_BYTE *)(v12 + 80);
        if ( (unsigned __int8)(v13 - 4) <= 2u || v13 == 3 && !*(_BYTE *)(v12 + 104) )
          *(_BYTE *)(*(_QWORD *)v12 + 73LL) |= 4u;
        v12 = *(_QWORD *)(v12 + 16);
      }
      while ( v12 );
    }
    if ( !a4 )
      goto LABEL_31;
  }
  if ( *qword_4D03FD0 )
LABEL_26:
    sub_8CF730(a1);
LABEL_27:
  if ( (unsigned __int8)(v7 - 4) <= 1u || v7 == 11 )
  {
    v32 = v7 == 0;
    goto LABEL_65;
  }
  if ( v7 == 17 )
  {
    v27 = *(_QWORD *)(a1 + 32);
    goto LABEL_69;
  }
  if ( v7 != 6 )
  {
    v27 = 0;
    goto LABEL_69;
  }
LABEL_31:
  v14 = 0;
  v15 = dword_4F077C4;
  if ( dword_4F077C4 == 2 )
    v14 = (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 177LL) & 0x20) != 0;
  if ( v8 )
  {
    v27 = 0;
LABEL_35:
    v30 = v7 == 3;
    v16 = v7 == 3;
    v32 = v7 == 0;
    v17 = v7 == 0;
    v28 = v7;
    v18 = v8;
    v19 = v14;
    v20 = v17 || v16;
    while ( 1 )
    {
      while ( a2 == 1 )
      {
        v23 = *(_BYTE *)(v18 + 80);
        v22 = v23;
        if ( (unsigned __int8)(v23 - 4) <= 2u )
        {
          if ( v19 )
            goto LABEL_42;
          if ( !a4 )
            goto LABEL_59;
          goto LABEL_73;
        }
        if ( v23 == 3 )
        {
          if ( !*(_BYTE *)(v18 + 104) || v19 )
            goto LABEL_44;
          v21 = a4 == 0;
          goto LABEL_57;
        }
LABEL_41:
        if ( (unsigned __int8)(v22 - 14) > 1u )
          goto LABEL_42;
LABEL_50:
        v18 = *(_QWORD *)(v18 + 16);
        if ( !v18 )
          goto LABEL_80;
      }
      if ( a2 != 9 && !v19 )
      {
        v21 = a4 == 0;
        if ( !a4 )
        {
          if ( v30 )
            goto LABEL_40;
LABEL_57:
          if ( v21 )
          {
            v23 = *(_BYTE *)(v18 + 80);
LABEL_59:
            if ( (unsigned __int8)(v23 - 4) <= 1u && (unsigned int)sub_736990(*(_QWORD *)(v18 + 88)) || !a2 )
            {
LABEL_40:
              v22 = *(_BYTE *)(v18 + 80);
              goto LABEL_41;
            }
          }
        }
        if ( a2 == 2 && a6 < 0 )
          goto LABEL_40;
LABEL_73:
        sub_860B80(v18, a2, v27);
      }
      v22 = *(_BYTE *)(v18 + 80);
      if ( (unsigned __int8)(v22 - 14) <= 1u )
        goto LABEL_50;
      if ( !a4 || !v20 )
      {
LABEL_42:
        if ( v22 != 8 || (*(_BYTE *)(*(_QWORD *)(v18 + 88) + 145LL) & 1) == 0 )
LABEL_44:
          sub_8790E0(v18);
        if ( a2 == 6 || a2 == 16 || !a4 && v20 )
          goto LABEL_79;
        goto LABEL_49;
      }
      if ( a2 == 6 || a2 == 16 )
        goto LABEL_79;
LABEL_49:
      if ( a2 != 8 )
        goto LABEL_50;
LABEL_79:
      sub_879210(v18);
      v18 = *(_QWORD *)(v18 + 16);
      if ( !v18 )
      {
LABEL_80:
        v7 = v28;
        goto LABEL_81;
      }
    }
  }
  v32 = 0;
LABEL_95:
  while ( v9 )
  {
    if ( (*(_BYTE *)(v9 + 83) & 1) == 0 )
    {
      v24 = *(_QWORD *)(v9 + 8);
      v25 = *(_QWORD *)(*(_QWORD *)v9 + 40LL);
      if ( v25 && v25 != v9 )
      {
        do
        {
          v26 = v25;
          v25 = *(_QWORD *)(v25 + 8);
        }
        while ( v25 != v9 && v25 );
        *(_QWORD *)(v26 + 8) = v24;
      }
      else
      {
        *(_QWORD *)(*(_QWORD *)v9 + 40LL) = v24;
      }
    }
    v9 = *(_QWORD *)(v9 + 16);
  }
  v20 = 0;
LABEL_82:
  if ( v15 == 2 && a1 && (v7 == 2 || v7 == 17 || a4 && v20) )
    sub_860500(a1, v7 == 2 || v7 == 17, 0);
LABEL_65:
  if ( (unsigned __int8)(v7 - 3) <= 1u || v32 )
    sub_7302F0();
}
