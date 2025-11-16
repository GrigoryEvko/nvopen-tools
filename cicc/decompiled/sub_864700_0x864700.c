// Function: sub_864700
// Address: 0x864700
//
_BOOL8 __fastcall sub_864700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8)
{
  int v10; // r12d
  char v11; // al
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r11
  __int64 v16; // rbx
  char v17; // dl
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // esi
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // r9
  char v31; // cl
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  bool v35; // zf
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rbx
  char v39; // al
  __int64 v40; // r11
  __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  int v50; // eax
  __int64 v51; // rsi
  __int64 v52; // r8
  _DWORD *v53; // rdx
  _DWORD *v54; // rbx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r14
  char v60; // al
  __int64 v61; // r8
  __int64 v62; // rbx
  __int64 v63; // rax
  __int64 v64; // r11
  int v65; // r10d
  __int16 v66; // cx
  __int64 v67; // rdi
  int v68; // r13d
  int v69; // eax
  __int64 v70; // rdx
  __int16 v71; // si
  __int16 v72; // cx
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  unsigned int v77; // edx
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // [rsp+0h] [rbp-D0h]
  int v81; // [rsp+8h] [rbp-C8h]
  int v82; // [rsp+8h] [rbp-C8h]
  __int64 v83; // [rsp+10h] [rbp-C0h]
  __int64 v84; // [rsp+18h] [rbp-B8h]
  int v85; // [rsp+18h] [rbp-B8h]
  __int64 v87; // [rsp+28h] [rbp-A8h]
  __int64 v88; // [rsp+30h] [rbp-A0h]
  __int64 v89; // [rsp+38h] [rbp-98h]
  int v90; // [rsp+38h] [rbp-98h]
  int v91; // [rsp+38h] [rbp-98h]
  int v92; // [rsp+44h] [rbp-8Ch]
  __int128 v93; // [rsp+48h] [rbp-88h]
  __int64 v94; // [rsp+58h] [rbp-78h]
  __int128 v95; // [rsp+60h] [rbp-70h]
  __int64 v96; // [rsp+70h] [rbp-60h]
  __int64 v97; // [rsp+78h] [rbp-58h]
  __int64 v98; // [rsp+78h] [rbp-58h]
  unsigned int v99; // [rsp+80h] [rbp-50h]
  char v100; // [rsp+84h] [rbp-4Ch]
  _BOOL4 v101; // [rsp+88h] [rbp-48h]
  __int64 v102; // [rsp+88h] [rbp-48h]
  char v103; // [rsp+90h] [rbp-40h]
  __int64 v104; // [rsp+90h] [rbp-40h]
  int v105; // [rsp+90h] [rbp-40h]
  __int64 v106; // [rsp+90h] [rbp-40h]
  int v108; // [rsp+98h] [rbp-38h]

  *((_QWORD *)&v93 + 1) = a2;
  *(_QWORD *)&v93 = a3;
  v10 = dword_4F04C64;
  v92 = unk_4F04C2C;
  if ( !a5 )
  {
    v94 = 0;
    v12 = a4 == 0;
    v35 = *(_QWORD *)(a1 + 96) == 0;
    v88 = 0;
    dword_4F04C38 = 0;
    v95 = 0u;
    dword_4F04C58 = -1;
    qword_4F04C50 = 0;
    v96 = 0;
    v87 = *(_QWORD *)(a1 + 24);
    v101 = 0;
    v97 = *(_QWORD *)(a1 + 16);
    v100 = 1;
    v99 = 0;
    if ( !v35 )
      goto LABEL_12;
LABEL_61:
    v36 = v87;
    v87 = 0;
    v83 = v36;
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(a5 + 81) & 0x10) != 0 && (*(_BYTE *)(*(_QWORD *)(a5 + 64) + 89LL) & 1) != 0 )
  {
    dword_4F04C38 = 1;
    dword_4F04C58 = -1;
    qword_4F04C50 = 0;
    v11 = *(_BYTE *)(a5 + 80);
    if ( (unsigned __int8)(v11 - 19) > 3u )
      goto LABEL_31;
  }
  else
  {
    dword_4F04C38 = 0;
    dword_4F04C58 = -1;
    qword_4F04C50 = 0;
    v11 = *(_BYTE *)(a5 + 80);
    if ( (unsigned __int8)(v11 - 19) > 3u )
    {
LABEL_31:
      v87 = a1;
      v100 = 0;
      v95 = v93;
      v96 = a4;
      goto LABEL_6;
    }
  }
  v87 = *(_QWORD *)(a1 + 24);
  if ( v11 != 20 )
  {
LABEL_5:
    v100 = 1;
    v95 = 0u;
    v96 = 0;
LABEL_6:
    v101 = 0;
    v94 = 0;
    v97 = *(_QWORD *)(a1 + 16);
    v88 = 0;
    goto LABEL_7;
  }
  v21 = *(_QWORD *)(a5 + 88);
  v22 = *(_QWORD *)(v21 + 176);
  v101 = (*(_BYTE *)(v22 + 206) & 2) != 0;
  if ( (*(_BYTE *)(v22 + 194) & 0x40) == 0 )
  {
    if ( (*(_BYTE *)(v22 + 206) & 2) != 0 )
      goto LABEL_34;
    v100 = *(_BYTE *)(v22 + 202) >> 7;
    if ( *(char *)(v22 + 202) >= 0 )
      goto LABEL_5;
    v97 = *(_QWORD *)(*(_QWORD *)(sub_892400(v21) + 32) + 16LL);
LABEL_139:
    v94 = 0;
    v88 = 0;
    v95 = 0u;
    v96 = 0;
LABEL_7:
    if ( (a8 & 0x2002) != 0 )
      goto LABEL_37;
    goto LABEL_8;
  }
  do
    v22 = *(_QWORD *)(v22 + 232);
  while ( (*(_BYTE *)(v22 + 194) & 0x40) != 0 );
  if ( !v101 )
  {
    v34 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v22 + 96LL) + 32LL);
    switch ( *(_BYTE *)(v34 + 80) )
    {
      case 4:
      case 5:
        v74 = *(_QWORD *)(*(_QWORD *)(v34 + 96) + 80LL);
        break;
      case 6:
        v74 = *(_QWORD *)(*(_QWORD *)(v34 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v74 = *(_QWORD *)(*(_QWORD *)(v34 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v74 = *(_QWORD *)(v34 + 88);
        break;
      default:
        BUG();
    }
    v100 = 1;
    v97 = *(_QWORD *)(*(_QWORD *)(v74 + 328) + 16LL);
    goto LABEL_139;
  }
LABEL_34:
  v23 = *(_QWORD *)(a5 + 64);
  v94 = v23;
  if ( (*(_BYTE *)(v23 + 89) & 2) != 0 )
    v24 = sub_72F070(v23);
  else
    v24 = *(_QWORD *)(v23 + 40);
  v88 = v24;
  v97 = v24;
  v101 = 1;
  v100 = 1;
  v95 = 0u;
  v96 = 0;
  if ( (a8 & 0x2002) != 0 )
  {
LABEL_37:
    v25 = dword_4F04C64;
    v26 = 776LL * dword_4F04C64;
    v27 = qword_4F04C68[0] + v26;
    if ( (*(_BYTE *)(qword_4F04C68[0] + v26 + 6) & 2) != 0 )
    {
      v28 = 0;
      v29 = qword_4F04C68[0] + v26;
      v30 = qword_4F04C68[0] + 776LL * unk_4F04C48;
      if ( v27 == v30 )
        goto LABEL_107;
      do
      {
        v31 = *(_BYTE *)(v29 + 4);
        if ( v31 == 17 )
          break;
        if ( !v28 && (unsigned __int8)(v31 - 6) <= 1u )
          v28 = *(_QWORD *)(v29 + 208);
        v32 = *(int *)(v29 + 552);
        if ( (_DWORD)v32 == -1 )
          break;
        v29 = qword_4F04C68[0] + 776 * v32;
        if ( !v29 )
          break;
      }
      while ( v30 != v29 );
      if ( !v28 || (*(_BYTE *)(a5 + 81) & 0x10) == 0 )
        goto LABEL_107;
      v33 = *(_QWORD *)(a5 + 64);
      if ( v33 != v28 )
      {
        if ( !v33 )
          goto LABEL_107;
        v99 = dword_4F07588;
        if ( !dword_4F07588 )
        {
          v12 = v101 || a4 == 0;
          goto LABEL_11;
        }
        v57 = *(_QWORD *)(v33 + 32);
        if ( *(_QWORD *)(v28 + 32) != v57 || !v57 )
        {
LABEL_107:
          v99 = 0;
          v12 = v101 || a4 == 0;
          goto LABEL_11;
        }
      }
      if ( !v100 )
        goto LABEL_89;
      sub_85C120(9u, *(_DWORD *)(a1 + 8), *((__int64 *)&v93 + 1), v93, 0, a4, a5, a6, a1, 0, 0, 0, a8);
      if ( !v101 )
      {
        v25 = dword_4F04C64;
        v26 = 776LL * dword_4F04C64;
        v27 = v26 + qword_4F04C68[0];
        goto LABEL_89;
      }
    }
    else
    {
      if ( !v101 )
        goto LABEL_203;
      if ( !v100 )
        goto LABEL_89;
      sub_85C120(9u, *(_DWORD *)(a1 + 8), *((__int64 *)&v93 + 1), v93, 0, a4, a5, a6, a1, 0, 0, 0, a8);
    }
    v105 = -1;
    v82 = -1;
    v99 = v101;
    v85 = -1;
    v91 = -1;
    goto LABEL_97;
  }
LABEL_8:
  if ( (a8 & 0x1004) == 4 )
    goto LABEL_37;
  if ( !v101 )
  {
LABEL_203:
    v99 = 0;
    v12 = a4 == 0;
    goto LABEL_11;
  }
  a8 |= 0x200000u;
  v12 = 1;
  v99 = v101;
  if ( *(_DWORD *)(v88 + 240) == v10 && v10 )
  {
    v25 = dword_4F04C64;
    v26 = 776LL * dword_4F04C64;
    v27 = v26 + qword_4F04C68[0];
    if ( !v100 )
      goto LABEL_89;
    v82 = -1;
    v85 = -1;
    v105 = -1;
    v91 = -1;
LABEL_165:
    if ( *(_BYTE *)(v27 + 4) != 7 || *(_QWORD *)(v27 + 208) != v94 )
      sub_8646E0(v94, 0);
    sub_85C120(9u, *(_DWORD *)(a1 + 8), *((__int64 *)&v93 + 1), v93, 0, a4, a5, a6, a1, 0, 0, 0, a8);
    if ( !v101 )
    {
      v25 = dword_4F04C64;
LABEL_135:
      if ( !v99 )
        goto LABEL_115;
      v26 = 776LL * v25;
      v27 = v26 + qword_4F04C68[0];
LABEL_89:
      *(_BYTE *)(v27 + 7) |= 0x20u;
      goto LABEL_90;
    }
LABEL_97:
    switch ( *(_BYTE *)(a5 + 80) )
    {
      case 4:
      case 5:
        v73 = *(_QWORD *)(*(_QWORD *)(a5 + 96) + 80LL);
        break;
      case 6:
        v73 = *(_QWORD *)(*(_QWORD *)(a5 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v73 = *(_QWORD *)(*(_QWORD *)(a5 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v73 = *(_QWORD *)(a5 + 88);
        break;
      default:
        BUG();
    }
    v25 = dword_4F04C64;
    *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 424) = *(_DWORD *)(*(_QWORD *)(v73 + 32) + 44LL);
    goto LABEL_135;
  }
LABEL_11:
  if ( !*(_QWORD *)(a1 + 96) )
    goto LABEL_61;
LABEL_12:
  v83 = 0;
LABEL_13:
  v13 = v97;
  v14 = 0;
  v15 = 0;
  if ( !v97 )
    goto LABEL_25;
  v103 = v12;
  v16 = 0;
  while ( 1 )
  {
    v17 = *(_BYTE *)(v13 + 28);
    if ( !v17 )
      break;
    if ( v17 == 6 )
    {
      v18 = *(_QWORD *)(v13 + 32);
      if ( !v14 )
        v14 = *(_QWORD *)(v13 + 32);
      if ( (*(_BYTE *)(v18 + 89) & 2) != 0 )
        goto LABEL_57;
LABEL_23:
      v13 = *(_QWORD *)(v18 + 40);
      if ( !v13 )
        break;
    }
    else
    {
      if ( v17 == 3 )
      {
        v18 = *(_QWORD *)(v13 + 32);
        if ( !v16 )
          v16 = *(_QWORD *)(v13 + 32);
        if ( (*(_BYTE *)(v18 + 89) & 2) == 0 )
          goto LABEL_23;
LABEL_57:
        v13 = sub_72F070(v18);
        goto LABEL_17;
      }
      v13 = *(_QWORD *)(v13 + 16);
LABEL_17:
      if ( !v13 )
        break;
    }
  }
  v15 = v14;
  v14 = v16;
  v12 = v103;
LABEL_25:
  if ( !v12 )
  {
    if ( (*(_WORD *)(a5 + 80) & 0x10FF) != 0x14 )
    {
      v37 = sub_8807C0(a4);
      v15 = 0;
      v14 = v37;
      if ( (*(_BYTE *)(a4 + 81) & 0x10) != 0 )
        goto LABEL_187;
      v38 = a4;
      v39 = *(_BYTE *)(a4 + 80);
      if ( (unsigned __int8)(v39 - 4) <= 1u )
        goto LABEL_104;
      goto LABEL_65;
    }
LABEL_101:
    v106 = v15;
    v56 = sub_8807C0(a4);
    v15 = v106;
    v14 = v56;
    goto LABEL_102;
  }
  if ( !a5 || (unsigned __int8)(*(_BYTE *)(a5 + 80) - 20) <= 1u )
  {
LABEL_28:
    v104 = v15;
    v19 = sub_893360();
    v15 = v104;
    v20 = v19;
    goto LABEL_68;
  }
  if ( (*(_WORD *)(a5 + 80) & 0x10FF) == 0x14 )
    goto LABEL_101;
  if ( !a4 )
  {
    v14 = sub_8807C0(a5);
    if ( (*(_BYTE *)(a5 + 81) & 0x10) != 0 )
    {
      v79 = a5;
      goto LABEL_182;
    }
LABEL_200:
    v15 = 0;
    goto LABEL_28;
  }
  v14 = sub_8807C0(a4);
  if ( (*(_BYTE *)(a4 + 81) & 0x10) == 0 )
    goto LABEL_200;
LABEL_187:
  v79 = a4;
LABEL_182:
  v15 = *(_QWORD *)(v79 + 64);
LABEL_102:
  if ( v12 )
    goto LABEL_28;
  v38 = a4;
  v39 = *(_BYTE *)(a4 + 80);
  if ( (unsigned __int8)(v39 - 4) <= 1u )
  {
LABEL_104:
    v20 = *(_QWORD *)(*(_QWORD *)(v38 + 96) + 120LL);
    goto LABEL_68;
  }
LABEL_65:
  if ( v39 == 3 || v39 == 6 )
    v20 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  else
    v20 = *(_QWORD *)(*(_QWORD *)(a4 + 96) + 48LL);
LABEL_68:
  v84 = v15;
  v89 = v20;
  sub_85C120(0xAu, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a8);
  v40 = v84;
  v105 = a8 & 0x4000;
  if ( (a8 & 0x4000) == 0 && (*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224) != v89 || unk_4F04C48 != -1) )
  {
    if ( v89 )
    {
      v41 = v89;
      v90 = dword_4F04C64 + 1;
      sub_864230(v41, 1);
      v42 = v90;
      v40 = v84;
      v85 = dword_4F04C34;
      if ( v14 )
        goto LABEL_72;
      v91 = 0;
      v50 = dword_4F04C64;
      goto LABEL_83;
    }
    goto LABEL_172;
  }
  if ( !v89 )
  {
LABEL_172:
    if ( v14 )
    {
      v49 = *(_QWORD *)(qword_4F04C68[0] + 224LL);
      if ( v49 != v14 )
      {
        v85 = 0;
        LODWORD(v42) = -1;
        v105 = 0;
        goto LABEL_82;
      }
    }
    v85 = 0;
    v105 = 0;
    dword_4F04C5C = 0;
    v91 = 0;
    v50 = dword_4F04C64;
    goto LABEL_85;
  }
  v85 = dword_4F04C34;
  if ( v14 )
  {
    v42 = -1;
LABEL_72:
    v43 = qword_4F04C68[0];
    v44 = v14;
    if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * v85 + 224) == v14 )
    {
      v91 = v85;
      v50 = dword_4F04C64;
      v105 = v85;
    }
    else
    {
      while ( 1 )
      {
        v45 = *(_QWORD *)(v44 + 128);
        if ( *(_DWORD *)(v45 + 240) != -1 )
          break;
LABEL_78:
        v48 = *(_QWORD *)(v44 + 40);
        if ( v48 )
        {
          if ( *(_BYTE *)(v48 + 28) == 3 )
          {
            v44 = *(_QWORD *)(v48 + 32);
            if ( v44 )
              continue;
          }
        }
        v105 = 0;
        goto LABEL_81;
      }
      v46 = qword_4F04C68[0] + 776LL * v85;
      while ( v45 != *(_QWORD *)(v46 + 184) )
      {
        v47 = *(int *)(v46 + 552);
        if ( (_DWORD)v47 != -1 )
        {
          v46 = qword_4F04C68[0] + 776 * v47;
          if ( v46 )
            continue;
        }
        goto LABEL_78;
      }
      v105 = 1594008481 * ((v46 - qword_4F04C68[0]) >> 3);
      v43 = 776LL * v105 + qword_4F04C68[0];
LABEL_81:
      v49 = *(_QWORD *)(v43 + 224);
      if ( v49 == v14 )
      {
        v50 = dword_4F04C64;
        v91 = v105;
      }
      else
      {
LABEL_82:
        v80 = v40;
        v81 = v42;
        sub_85F170(v14, v49, v105);
        v40 = v80;
        v42 = v81;
        v50 = dword_4F04C64;
        v91 = dword_4F04C64;
      }
    }
LABEL_83:
    dword_4F04C5C = v91;
    if ( (_DWORD)v42 != -1 )
      *(_DWORD *)(qword_4F04C68[0] + 776 * v42 + 552) = 0;
  }
  else
  {
    v105 = 0;
    v91 = 0;
    dword_4F04C5C = 0;
    v50 = dword_4F04C64;
  }
LABEL_85:
  v82 = v50 + 1;
  v51 = v97;
  sub_85F1C0(v83, v97, v40, v96, *((__int64 *)&v95 + 1), v95, a8);
  if ( *(_QWORD *)(a1 + 96) )
  {
    switch ( *(_BYTE *)(a5 + 80) )
    {
      case 4:
      case 5:
        v58 = *(_QWORD *)(*(_QWORD *)(a5 + 96) + 80LL);
        break;
      case 6:
        v58 = *(_QWORD *)(*(_QWORD *)(a5 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v58 = *(_QWORD *)(*(_QWORD *)(a5 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v58 = *(_QWORD *)(a5 + 88);
        break;
      default:
        BUG();
    }
    v59 = *(_QWORD *)(*(_QWORD *)(v58 + 32) + 96LL);
    v60 = *(_BYTE *)(v59 + 80);
    if ( v60 == 9 || v60 == 7 )
    {
      v61 = *(_QWORD *)(v59 + 88);
    }
    else
    {
      v61 = 0;
      if ( v60 == 21 )
        v61 = *(_QWORD *)(*(_QWORD *)(v59 + 88) + 192LL);
    }
    v98 = v61;
    v62 = sub_892240(v59, v51);
    v63 = sub_892350(v98);
    sub_85E1C0(v87, 0, 0, v59, *(_QWORD *)(v62 + 32), v63, a8);
  }
  if ( v100 )
  {
    v27 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( !v99 )
    {
      sub_85C120(9u, *(_DWORD *)(a1 + 8), *((__int64 *)&v93 + 1), v93, 0, a4, a5, a6, a1, 0, 0, 0, a8);
      if ( !v101 )
        goto LABEL_115;
      goto LABEL_97;
    }
    v99 = 0;
    goto LABEL_165;
  }
LABEL_115:
  v64 = dword_4F04C64;
  if ( v10 >= dword_4F04C64 )
  {
    v102 = dword_4F04C64;
    v75 = qword_4F04C68[0];
    goto LABEL_150;
  }
  v102 = dword_4F04C64;
  v65 = -1;
  v66 = 0;
  v67 = 776LL * dword_4F04C64;
  v68 = 0;
  v69 = dword_4F04C64;
  v108 = a8 & 0x80000;
  while ( 2 )
  {
    if ( v69 == -1 )
      BUG();
    v70 = v67 + qword_4F04C68[0];
    v71 = *(unsigned __int8 *)(v67 + qword_4F04C68[0] + 4);
    if ( (_BYTE)v71 == 9 )
    {
      v65 = v69;
      LOBYTE(v52) = *(_QWORD *)(v70 + 368) == 0;
      v72 = v52 | v66;
      LOWORD(v52) = *(_WORD *)(v70 + 7) & 0xDFDF;
      *(_WORD *)(v70 + 7) = v52 | (v72 << 13) | 0x20;
      v66 = 1;
      if ( (a8 & 0x1000) != 0 )
        goto LABEL_123;
LABEL_121:
      if ( v10 >= *(_DWORD *)(v70 + 448) )
        *(_DWORD *)(v70 + 448) = -1;
    }
    else
    {
      LOWORD(v52) = v71 - 6;
      if ( (unsigned __int8)(v71 - 6) <= 1u )
        goto LABEL_121;
      if ( (_BYTE)v71 == 10 )
      {
        BYTE1(v52) = BYTE1(v108);
        if ( !v108 )
          goto LABEL_123;
        goto LABEL_121;
      }
      if ( (_BYTE)v71 == 17 || (_BYTE)v71 == 14 )
        goto LABEL_121;
    }
LABEL_123:
    if ( (a8 & 0x100) != 0 && !v68 )
    {
      v52 = *(_QWORD *)(a1 + 16);
      if ( *(_DWORD *)(v52 + 24) == *(_DWORD *)v70 )
      {
        v68 = 1;
      }
      else if ( (_BYTE)v71 != 9 )
      {
        *(_BYTE *)(v70 + 11) |= 0x10u;
      }
    }
    --v69;
    v67 -= 776;
    if ( v10 != v69 )
      continue;
    break;
  }
  v64 = (int)v64;
  v75 = qword_4F04C68[0];
  if ( v65 != -1 && (!v88 || *(_DWORD *)(v88 + 240) == -1) )
  {
    v78 = qword_4F04C68[0] + 776LL * v65;
    *(_BYTE *)(v78 + 7) &= ~0x20u;
    v75 = qword_4F04C68[0];
    *(_DWORD *)(v78 + 556) = v85;
    *(_DWORD *)(v78 + 560) = v105;
  }
LABEL_150:
  v76 = v82;
  if ( (int)v64 <= v82 )
    v76 = v64;
  if ( (_DWORD)v76 != v91 )
    *(_DWORD *)(v75 + 776 * v76 + 552) = v91;
  *(_DWORD *)(v75 + 776 * v102 + 520) = v91;
  dword_4F04C34 = v91;
  sub_85FE80(v64, 0, 0);
  v77 = *(_DWORD *)(a1 + 44);
  if ( !(unk_4D047BC | dword_4D047C8) && v93 != 0 )
    v77 = 0;
  sub_85FE80(dword_4F04C64, 1, v77);
  v25 = dword_4F04C64;
  v26 = 776LL * dword_4F04C64;
LABEL_90:
  v53 = (_DWORD *)(qword_4F04C68[0] + v26);
  v54 = v53;
  if ( v10 == v25 )
  {
    ++v53[142];
  }
  else
  {
    v53[143] = v10;
    v53[144] = v92;
    if ( a7 )
    {
      sub_7B8190();
      *((_BYTE *)v54 + 9) |= 1u;
      v25 = dword_4F04C64;
    }
  }
  return v10 != v25;
}
