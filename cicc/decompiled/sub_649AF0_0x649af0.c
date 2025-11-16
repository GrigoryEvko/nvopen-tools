// Function: sub_649AF0
// Address: 0x649af0
//
void __fastcall sub_649AF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v9; // al
  char v10; // dl
  char v11; // si
  char v12; // r15
  __int64 v13; // r9
  __int64 v14; // r8
  char v15; // al
  unsigned __int8 v16; // r8
  unsigned int v17; // r9d
  __int64 v18; // rsi
  __int64 v19; // rdi
  char v20; // al
  __int64 v21; // rdi
  char v22; // al
  __int64 v23; // rax
  char v24; // al
  char v25; // al
  bool v26; // zf
  __int64 v27; // [rsp-58h] [rbp-58h]
  unsigned __int8 v28; // [rsp-50h] [rbp-50h]
  __int64 v29; // [rsp-50h] [rbp-50h]
  unsigned int v30; // [rsp-44h] [rbp-44h]
  int v31; // [rsp-44h] [rbp-44h]
  __int64 v32; // [rsp-40h] [rbp-40h]

  if ( !*(_DWORD *)(a1 + 80) )
    return;
  v9 = *(_BYTE *)(a3 + 88);
  v10 = *(_BYTE *)(a1 + 84);
  v11 = v9 & 0x70;
  if ( (v9 & 0x70) == 0 )
  {
    *(_BYTE *)(a3 + 88) = (16 * (v10 & 7)) | v9 & 0x8F;
    *(_BYTE *)(a2 + 81) = (4 * (*(_BYTE *)(a1 + 85) & 1)) | *(_BYTE *)(a2 + 81) & 0xFB;
    if ( a4 )
      *(_BYTE *)(a4 + 81) = (4 * (*(_BYTE *)(a1 + 85) & 1)) | *(_BYTE *)(a4 + 81) & 0xFB;
    goto LABEL_5;
  }
  v12 = *(_BYTE *)(a2 + 80);
  if ( ((v9 >> 4) & 7) == v10 )
  {
    if ( !*(_BYTE *)(a1 + 85) )
      goto LABEL_5;
    if ( v12 != 11 || v11 == 32 || (v24 = *(_BYTE *)(a2 + 81), (v24 & 4) != 0) )
    {
      *(_BYTE *)(a2 + 81) |= 4u;
      if ( !a4 )
        goto LABEL_5;
    }
    else
    {
      v25 = v24 | 4;
      if ( !a4 )
      {
        *(_BYTE *)(a2 + 81) = v25;
        v21 = 8;
LABEL_43:
        a4 = *(_QWORD *)(a1 + 8);
        goto LABEL_25;
      }
      v26 = (*(_BYTE *)(a4 + 81) & 4) == 0;
      *(_BYTE *)(a2 + 81) = v25;
      if ( v26 )
      {
        *(_BYTE *)(a4 + 81) |= 4u;
        v21 = 8;
LABEL_25:
        sub_6853B0(v21, 337, a5, a4);
        if ( dword_4F077C4 != 2 )
          return;
        goto LABEL_26;
      }
    }
    *(_BYTE *)(a4 + 81) |= 4u;
    goto LABEL_5;
  }
  v13 = qword_4F04C68[0];
  v14 = dword_4F04C64;
  if ( v12 == 11 && v11 == 32 && (_DWORD)qword_4F077B4 != 0 && v10 == 3 )
  {
    v29 = a4;
    v31 = dword_4F04C64;
    v32 = qword_4F04C68[0];
    v23 = sub_736C60(20, *(_QWORD *)(a3 + 104));
    v13 = v32;
    v14 = v31;
    a4 = v29;
    if ( v23 )
      goto LABEL_5;
  }
  if ( *(_BYTE *)(a1 + 44) == 2 || (*(_BYTE *)(v13 + 776 * v14 + 9) & 0x10) == 0 )
  {
    v16 = 3;
  }
  else
  {
    if ( dword_4F077BC && (*(_BYTE *)(a1 + 64) & 1) != 0 )
    {
      v15 = *(_BYTE *)(a3 + 88) & 0x70;
      if ( v15 != 16 )
      {
        if ( *(_BYTE *)(a1 + 84) != 1 )
          goto LABEL_5;
        v16 = 3;
LABEL_16:
        if ( v15 != 48 )
        {
          v20 = 1;
          goto LABEL_20;
        }
        goto LABEL_17;
      }
      v16 = 3;
LABEL_51:
      v20 = *(_BYTE *)(a1 + 84);
      if ( v20 != 3 )
        goto LABEL_20;
      v18 = *(unsigned int *)(a1 + 40);
      v17 = 0;
      goto LABEL_18;
    }
    if ( v12 == 11 )
    {
      v21 = 8;
LABEL_42:
      if ( a4 )
        goto LABEL_25;
      goto LABEL_43;
    }
    if ( dword_4D04964 )
    {
      v16 = byte_4F07472[0];
      if ( byte_4F07472[0] > 7u )
      {
LABEL_41:
        v21 = v16;
        goto LABEL_42;
      }
    }
    else
    {
      v16 = 5;
    }
  }
  if ( !*(_BYTE *)(a1 + 85) || dword_4F077BC && (*(_BYTE *)(a1 + 64) & 1) != 0 )
  {
    v22 = *(_BYTE *)(a3 + 88);
    goto LABEL_38;
  }
  v22 = *(_BYTE *)(a3 + 88);
  if ( (*(_BYTE *)(a2 + 81) & 4) != 0 )
  {
LABEL_38:
    v15 = v22 & 0x70;
    if ( v15 == 16 )
      goto LABEL_51;
    if ( *(_BYTE *)(a1 + 84) == 1 )
      goto LABEL_16;
    if ( v16 == 3 )
      goto LABEL_5;
    goto LABEL_41;
  }
  if ( (v22 & 0x70) != 0x30 )
    goto LABEL_51;
LABEL_17:
  v17 = *(_DWORD *)(a1 + 40);
  v18 = 0;
LABEL_18:
  v27 = a4;
  v19 = *(_QWORD *)(a2 + 88);
  v28 = v16;
  v30 = v17;
  if ( v12 == 11 )
  {
    sub_736270(v19, v18);
    *(_QWORD *)(v19 + 40) = 0;
    sub_7362F0(v19, v30);
    v20 = *(_BYTE *)(a1 + 84);
    v16 = v28;
    a4 = v27;
  }
  else
  {
    sub_735DA0(v19, v18, 0);
    *(_QWORD *)(v19 + 40) = 0;
    sub_735E40(v19, v30);
    v20 = *(_BYTE *)(a1 + 84);
    a4 = v27;
    v16 = v28;
  }
LABEL_20:
  *(_BYTE *)(a3 + 88) = *(_BYTE *)(a3 + 88) & 0x8F | (16 * (v20 & 7));
  *(_BYTE *)(a2 + 81) = (4 * (*(_BYTE *)(a1 + 85) & 1)) | *(_BYTE *)(a2 + 81) & 0xFB;
  if ( !a4 )
  {
    if ( v16 == 3 )
      goto LABEL_5;
    v21 = v16;
    goto LABEL_43;
  }
  if ( *(_BYTE *)(a1 + 85) )
    *(_BYTE *)(a4 + 81) |= 4u;
  if ( v16 != 3 )
  {
    v21 = v16;
    goto LABEL_25;
  }
LABEL_5:
  if ( dword_4F077C4 != 2 )
    return;
LABEL_26:
  if ( (*(_BYTE *)(a3 + 88) & 0x50) != 0x10 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
    sub_649830(a2, a5, 1);
}
