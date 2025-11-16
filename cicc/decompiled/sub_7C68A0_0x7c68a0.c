// Function: sub_7C68A0
// Address: 0x7c68a0
//
__int64 __fastcall sub_7C68A0(__int64 *a1, FILE *a2, __int64 a3, __int64 a4)
{
  int v4; // r14d
  __int64 v7; // r9
  __int64 v8; // r8
  unsigned int v9; // r13d
  unsigned __int64 v10; // rdi
  FILE *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // r9
  __int64 v16; // r13
  int v17; // eax
  int v18; // r15d
  _BOOL8 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  _QWORD *v23; // rax
  int v24; // eax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  int v35; // [rsp+8h] [rbp-48h]
  int v36; // [rsp+Ch] [rbp-44h]
  unsigned int v37[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = a4;
  v36 = a3;
  v37[0] = 0;
  v7 = dword_4F077BC;
  if ( dword_4F077BC )
  {
    v8 = qword_4F077A8 >= 0x76C0u ? 0xF : 0;
    v9 = qword_4F077A8 < 0x76C0u ? 1 : 4097;
  }
  else
  {
    v8 = 0;
    v9 = 1;
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] == 1 )
      goto LABEL_5;
LABEL_37:
    sub_6851D0(0x28u);
    goto LABEL_38;
  }
  if ( word_4F06418[0] != 1 || (v7 = (__int64)&qword_4D04A00, (word_4D04A10 & 0x200) == 0) )
  {
    v35 = v8;
    v24 = sub_7C0F00(v9, 0, a3, a4, v8, v7);
    LODWORD(v8) = v35;
    if ( !v24 )
      goto LABEL_37;
  }
LABEL_5:
  v10 = v9;
  v11 = (FILE *)(unsigned int)v8;
  v16 = sub_7BF130(v9, (unsigned int)v8, v37);
  v17 = dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
    v13 = (__int64)&dword_4F077BC;
    v14 = dword_4F077BC;
    if ( !dword_4F077BC )
      goto LABEL_14;
    if ( v16 )
    {
      v12 = dword_4D047DC;
      if ( !dword_4D047DC )
        goto LABEL_14;
      goto LABEL_9;
    }
    v11 = (FILE *)v37[0];
    if ( !v37[0] )
      goto LABEL_63;
LABEL_69:
    sub_7B8B50(v10, (unsigned int *)v11, v12, v13, v14, (__int64)v15);
    goto LABEL_38;
  }
  v12 = (__int64)&unk_4F07778;
  if ( unk_4F07778 > 201102 || (v12 = (__int64)&dword_4F07774, (v14 = dword_4F07774) != 0) )
  {
    if ( !v16 )
    {
LABEL_55:
      v15 = &qword_4D04A00;
      v23 = qword_4D04A18;
      if ( qword_4D04A18 )
      {
LABEL_33:
        if ( (v23[10] & 0x41000) != 0 )
        {
          v11 = 0;
          v10 = (unsigned __int64)&qword_4D04A00;
          sub_8841F0(&qword_4D04A00, 0, 0, 0);
        }
        goto LABEL_14;
      }
      if ( !v37[0] )
        goto LABEL_63;
      goto LABEL_69;
    }
  }
  else
  {
    v13 = (__int64)&dword_4F077BC;
    v10 = dword_4F077BC;
    if ( !dword_4F077BC )
      goto LABEL_32;
    if ( !v16 )
      goto LABEL_55;
  }
  v12 = (__int64)&dword_4D047DC;
  v13 = dword_4D047DC;
  if ( !dword_4D047DC )
    goto LABEL_32;
LABEL_9:
  v12 = *(unsigned __int8 *)(v16 + 80);
  if ( (_BYTE)v12 == 3 )
  {
    if ( *(_BYTE *)(v16 + 104) )
    {
      v12 = *(_QWORD *)(v16 + 88);
      if ( (*(_BYTE *)(v12 + 177) & 0x10) != 0 )
      {
        v12 = *(_QWORD *)(v12 + 168);
        if ( *(_QWORD *)(v12 + 168) )
        {
          v10 = v16;
          v16 = sub_880FE0(v16);
          v17 = dword_4F077C4;
        }
      }
    }
  }
  else
  {
    v12 = (unsigned int)(v12 - 4);
    if ( (unsigned __int8)v12 <= 1u )
    {
      v12 = *(_QWORD *)(v16 + 88);
      if ( *(char *)(v12 + 177) < 0 )
      {
        v12 = *(_QWORD *)(v16 + 96);
        v16 = *(_QWORD *)(v12 + 72);
      }
    }
  }
  if ( v17 == 2 )
  {
LABEL_32:
    v15 = &qword_4D04A00;
    v23 = qword_4D04A18;
    if ( !qword_4D04A18 )
      goto LABEL_14;
    goto LABEL_33;
  }
LABEL_14:
  v18 = v37[0];
  if ( v37[0] )
  {
    sub_7B8B50(v10, (unsigned int *)v11, v12, v13, v14, (__int64)v15);
    if ( !v16 )
      goto LABEL_38;
    if ( *(_BYTE *)(v16 + 80) != 19 )
      goto LABEL_38;
    v26 = *(_QWORD *)(v16 + 88);
    if ( (*(_BYTE *)(v26 + 266) & 1) == 0 )
      goto LABEL_38;
LABEL_48:
    v18 = 1;
    v19 = 0;
LABEL_49:
    v16 = *(_QWORD *)(v26 + 200);
    goto LABEL_50;
  }
  if ( !v16 )
  {
LABEL_63:
    sub_6851A0(0x14u, a2, *(_QWORD *)(qword_4D04A00 + 8));
    sub_7B8B50(0x14u, (unsigned int *)a2, v31, v32, v33, v34);
    goto LABEL_38;
  }
  if ( (*(_BYTE *)(v16 + 82) & 4) != 0 )
  {
    sub_7B8B50(v10, (unsigned int *)v11, v12, v13, v14, (__int64)v15);
    if ( *(_BYTE *)(v16 + 80) != 19 )
      goto LABEL_38;
    goto LABEL_47;
  }
  if ( *(_BYTE *)(v16 + 80) != 19 )
  {
    v11 = a2;
    sub_6854C0(0x2DAu, a2, v16);
    sub_7B8B50(0x2DAu, (unsigned int *)a2, v27, v28, v29, v30);
    if ( *(_BYTE *)(v16 + 80) != 19 )
      goto LABEL_38;
LABEL_47:
    v26 = *(_QWORD *)(v16 + 88);
    if ( (*(_BYTE *)(v26 + 266) & 1) == 0 )
      goto LABEL_38;
    goto LABEL_48;
  }
  LOBYTE(v12) = a1 != 0;
  sub_7B8B50(v10, (unsigned int *)v11, v12, v13, v14, (__int64)v15);
  v19 = a1 != 0;
  if ( *(_BYTE *)(v16 + 80) != 19 )
  {
    if ( !a1 )
      goto LABEL_60;
    goto LABEL_20;
  }
  v26 = *(_QWORD *)(v16 + 88);
  if ( (*(_BYTE *)(v26 + 266) & 1) != 0 )
    goto LABEL_49;
LABEL_50:
  if ( !v19 )
    goto LABEL_59;
LABEL_20:
  v20 = sub_8794A0(a1, v11, v19);
  if ( (*(_BYTE *)(v20 + 160) & 2) != 0 )
    goto LABEL_59;
  v21 = *(_QWORD *)(v16 + 88);
  if ( (*(_BYTE *)(v21 + 160) & 2) != 0 )
    goto LABEL_59;
  if ( v4 || dword_4F077BC && v36 )
  {
    *(_BYTE *)(v20 + 266) |= 2u;
    goto LABEL_59;
  }
  v22 = *(_QWORD *)(v21 + 104);
  if ( unk_4D04854 )
    goto LABEL_26;
  if ( (unsigned int)sub_89B3C0(**(_QWORD **)(v20 + 32), **(_QWORD **)(v21 + 32), 0, 4, 0, 8) )
  {
    if ( unk_4D04854 )
      goto LABEL_26;
LABEL_59:
    if ( v18 )
      goto LABEL_38;
LABEL_60:
    sub_8767A0(4, v16, &qword_4D04A08, 1);
    return *(_QWORD *)(*(_QWORD *)(v16 + 88) + 104LL);
  }
  if ( !unk_4D04854 )
    goto LABEL_27;
LABEL_26:
  if ( (unsigned int)sub_8B2D80(v22, a1) )
    goto LABEL_59;
LABEL_27:
  sub_686C60(0x3E7u, a2, v16, *a1);
LABEL_38:
  v16 = sub_87F550();
  return *(_QWORD *)(*(_QWORD *)(v16 + 88) + 104LL);
}
