// Function: sub_7D3C60
// Address: 0x7d3c60
//
__int64 __fastcall sub_7D3C60(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, int a5, int a6, _DWORD *a7)
{
  int v7; // eax
  unsigned int v8; // r14d
  __int64 v9; // rdx
  int v11; // ebx
  unsigned int v12; // r8d
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rsi
  char v17; // al
  __int64 v18; // r12
  unsigned int v19; // r10d
  unsigned int v20; // r8d
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rsi
  int v24; // edx
  char v25; // al
  __int64 v26; // rbx
  char v27; // al
  char v28; // al
  int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // rax
  int v33; // eax
  int v34; // eax
  int v35; // eax
  int v36; // eax
  unsigned int v37; // [rsp+Ch] [rbp-84h]
  unsigned int v38; // [rsp+Ch] [rbp-84h]
  unsigned int v39; // [rsp+Ch] [rbp-84h]
  unsigned int v40; // [rsp+10h] [rbp-80h]
  unsigned int v41; // [rsp+10h] [rbp-80h]
  unsigned int v42; // [rsp+10h] [rbp-80h]
  int v43; // [rsp+14h] [rbp-7Ch]
  int v44; // [rsp+14h] [rbp-7Ch]
  int v45; // [rsp+14h] [rbp-7Ch]
  __int64 v46; // [rsp+18h] [rbp-78h]
  __int64 v47; // [rsp+18h] [rbp-78h]
  __int64 v48; // [rsp+18h] [rbp-78h]
  unsigned int v52; // [rsp+38h] [rbp-58h]
  unsigned int v53; // [rsp+38h] [rbp-58h]
  __int64 v54; // [rsp+38h] [rbp-58h]
  unsigned int v55; // [rsp+38h] [rbp-58h]
  __int64 v56; // [rsp+38h] [rbp-58h]
  __int64 v57; // [rsp+38h] [rbp-58h]
  int v58; // [rsp+40h] [rbp-50h]
  int v59; // [rsp+40h] [rbp-50h]
  int v60; // [rsp+40h] [rbp-50h]
  int v61; // [rsp+40h] [rbp-50h]
  int v62; // [rsp+48h] [rbp-48h]
  int v63; // [rsp+4Ch] [rbp-44h]

  v7 = a3 & 1;
  v8 = a3;
  v9 = a3 & 2;
  v11 = v7;
  v63 = v8 & 0x20;
  v62 = v8 & 0x80000;
  v12 = 0;
  if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 && unk_4D047C8 )
  {
    v60 = v9;
    v31 = sub_7D3BE0(a1, a2, v9, qword_4F04C68, 0);
    LODWORD(v9) = v60;
    v12 = v31;
    v13 = a2;
    if ( (*(_BYTE *)(a2 + 124) & 1) == 0 )
      goto LABEL_4;
  }
  else
  {
    v13 = a2;
    if ( (*(_BYTE *)(a2 + 124) & 1) == 0 )
      goto LABEL_4;
  }
  v55 = v12;
  v61 = v9;
  v13 = sub_735B70(a2);
  v12 = v55;
  LODWORD(v9) = v61;
LABEL_4:
  v14 = *(_QWORD *)(a2 + 128);
  v15 = *(_QWORD *)(*(_QWORD *)v13 + 96LL);
  v16 = *a1;
  if ( unk_4D03F98 )
  {
    if ( v14 )
    {
      if ( *(_QWORD *)(v16 + 64) )
      {
        v17 = *(_BYTE *)(v14 + 28);
        if ( v17 == 3 || !v17 )
        {
          v52 = v12;
          v58 = v9;
          sub_824D70(v14);
          v12 = v52;
          LODWORD(v9) = v58;
          v16 = *a1;
        }
      }
    }
  }
  v53 = v12;
  v59 = v9;
  v18 = sub_883800(v15, v16);
  if ( !v18 )
  {
LABEL_88:
    if ( v63 && (v8 & 0x2000) == 0 )
      goto LABEL_77;
    if ( v62 )
      goto LABEL_77;
    v18 = sub_7D4400((_DWORD)a1, a2, *(_QWORD *)(*(_QWORD *)(a2 + 128) + 184LL), v8, a4, a5, (__int64)a7, 1);
    if ( !v18 )
      goto LABEL_77;
    return v18;
  }
  v19 = v8;
  v20 = v53;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = v11;
  while ( 1 )
  {
    while ( 1 )
    {
      v25 = *(_BYTE *)(v18 + 80);
      v26 = v18;
      if ( v25 == 16 )
      {
        v26 = **(_QWORD **)(v18 + 88);
        v25 = *(_BYTE *)(v26 + 80);
      }
      if ( v25 == 24 )
        v26 = *(_QWORD *)(v26 + 88);
      if ( (*(_BYTE *)(v26 + 83) & 0x40) != 0 && (v8 & 0x8004020) == 0
        || (*(_BYTE *)(v18 + 81) & 0x10) != 0
        || a2 != *(_QWORD *)(v18 + 64) )
      {
        goto LABEL_12;
      }
      if ( !v24 )
        goto LABEL_22;
      v27 = *(_BYTE *)(v26 + 80);
      if ( v27 == 19 )
      {
        if ( (v8 & 0x800) != 0 )
          goto LABEL_26;
        goto LABEL_40;
      }
      if ( (unsigned __int8)(v27 - 4) <= 1u )
        goto LABEL_40;
      if ( v27 == 3 )
      {
        v38 = v19;
        v41 = v20;
        v44 = v24;
        v47 = v21;
        v56 = v22;
        v33 = sub_8D3A70(*(_QWORD *)(v26 + 88));
        v22 = v56;
        v21 = v47;
        v24 = v44;
        v20 = v41;
        v19 = v38;
        if ( v33 )
          goto LABEL_22;
        v27 = *(_BYTE *)(v26 + 80);
        if ( v27 == 23 )
        {
LABEL_82:
          if ( (v8 & 0x800) != 0 )
            goto LABEL_12;
          goto LABEL_40;
        }
        if ( v27 == 3 )
        {
          v34 = sub_8D3D40(*(_QWORD *)(v26 + 88));
          v22 = v56;
          v21 = v47;
          v24 = v44;
          v20 = v41;
          v19 = v38;
          if ( v34 || !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
            goto LABEL_22;
LABEL_94:
          if ( !dword_4D044A0 )
            goto LABEL_12;
          v27 = *(_BYTE *)(v26 + 80);
          goto LABEL_51;
        }
      }
      else if ( v27 == 23 )
      {
        goto LABEL_82;
      }
      if ( dword_4F077BC && qword_4F077A8 > 0x76BFu )
        goto LABEL_94;
LABEL_51:
      if ( v27 == 6 )
        goto LABEL_82;
      if ( v27 != 3 )
        goto LABEL_12;
      v37 = v19;
      v40 = v20;
      v43 = v24;
      v46 = v21;
      v54 = v22;
      v30 = sub_8D2870(*(_QWORD *)(v26 + 88));
      v22 = v54;
      v21 = v46;
      v24 = v43;
      v20 = v40;
      v19 = v37;
      if ( !v30 )
        goto LABEL_12;
LABEL_22:
      if ( (v8 & 0x800) == 0 )
        goto LABEL_40;
      v27 = *(_BYTE *)(v26 + 80);
      if ( (unsigned __int8)(v27 - 4) <= 1u )
        goto LABEL_40;
      if ( v27 != 3 )
        break;
      v39 = v19;
      v42 = v20;
      v45 = v24;
      v48 = v21;
      v57 = v22;
      v35 = sub_8D3A70(*(_QWORD *)(v26 + 88));
      v22 = v57;
      v21 = v48;
      v24 = v45;
      v20 = v42;
      v19 = v39;
      if ( !v35 )
      {
        v27 = *(_BYTE *)(v26 + 80);
        if ( v27 == 19 )
          goto LABEL_26;
        if ( v27 != 3 )
          goto LABEL_12;
        v36 = sub_8D3D40(*(_QWORD *)(v26 + 88));
        v22 = v57;
        v21 = v48;
        v24 = v45;
        v20 = v42;
        v19 = v39;
        if ( !v36 && (*(_BYTE *)(v26 + 81) & 0x40) == 0 )
          goto LABEL_12;
      }
LABEL_40:
      if ( !v59 )
        goto LABEL_27;
      v27 = *(_BYTE *)(v26 + 80);
      if ( (unsigned __int8)(v27 - 4) <= 2u )
        goto LABEL_42;
LABEL_57:
      if ( !dword_4F077BC )
        goto LABEL_58;
      if ( qword_4F077A8 <= 0x9E33u || v27 != 3 )
      {
        if ( (v8 & 0x4000) == 0 || qword_4F077A8 <= 0x9E33u )
        {
LABEL_58:
          if ( unk_4D04234 && v27 == 3 )
            goto LABEL_42;
        }
        if ( v27 != 19 )
          goto LABEL_61;
        goto LABEL_42;
      }
      if ( !*(_BYTE *)(v26 + 104) && ((v8 & 0x4000) != 0 || !unk_4D04234) )
      {
LABEL_61:
        if ( (*(_WORD *)(v26 + 80) & 0x40FF) != 0x4003 )
          goto LABEL_12;
      }
LABEL_42:
      if ( !v20 || (v8 & 0x204020) != 0 || v20 >= *(_DWORD *)(v18 + 44) )
      {
        if ( *(_BYTE *)(v18 + 80) != 3 )
          goto LABEL_84;
        v22 = v18;
      }
LABEL_12:
      v18 = *(_QWORD *)(v18 + 32);
      if ( !v18 )
        goto LABEL_35;
    }
    if ( v27 != 19 )
      goto LABEL_12;
LABEL_26:
    if ( v59 )
      goto LABEL_57;
LABEL_27:
    if ( v20 && (v8 & 0x204020) == 0 && v20 < *(_DWORD *)(v18 + 44) )
      goto LABEL_12;
    v28 = *(_BYTE *)(v26 + 80);
    if ( (unsigned __int8)(v28 - 4) <= 2u || v28 == 3 && *(_BYTE *)(v26 + 104) )
    {
      v23 = v18;
      goto LABEL_12;
    }
    if ( *(_BYTE *)(v18 + 80) != 23 || !dword_4D044B8 )
      break;
    v21 = v18;
    v18 = *(_QWORD *)(v18 + 32);
    if ( !v18 )
    {
LABEL_35:
      v18 = v22;
      v8 = v19;
      if ( !v22 )
      {
        v18 = v23;
        if ( !v23 )
        {
          v18 = v21;
          if ( !v21 )
            goto LABEL_88;
        }
      }
      if ( v63 )
        goto LABEL_63;
LABEL_37:
      if ( v62 )
        return v18;
      goto LABEL_75;
    }
  }
LABEL_84:
  v8 = v19;
  if ( !v63 )
    goto LABEL_37;
LABEL_63:
  if ( (v8 & 0x2000) == 0 || v62 )
    return v18;
LABEL_75:
  v32 = sub_7D4400((_DWORD)a1, a2, *(_QWORD *)(*(_QWORD *)(a2 + 128) + 184LL), v8, a4, a5, (__int64)a7, 1);
  if ( !v32 )
    return v18;
  v18 = sub_7D09E0(v32, v18, (__int64)a1, 1u, a4, v8, a7);
  if ( v18 )
    return v18;
LABEL_77:
  if ( v8 & 0x84020 | a6 )
    return 0;
  else
    return sub_7D4400((_DWORD)a1, a2, *(_QWORD *)(*(_QWORD *)(a2 + 128) + 184LL), v8, a4, a5, (__int64)a7, 0);
}
