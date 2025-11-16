// Function: sub_5F4F20
// Address: 0x5f4f20
//
__int64 __fastcall sub_5F4F20(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r12
  char v8; // al
  __int64 i; // rax
  _BOOL4 v10; // r14d
  __int64 v11; // r14
  __int64 v12; // rdi
  bool v13; // sf
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // r8
  __int64 v17; // rdi
  char v18; // al
  int v19; // eax
  __int64 v20; // r8
  unsigned __int8 v21; // si
  __int64 v22; // r8
  int v23; // edx
  char m; // al
  char v25; // al
  __int64 v26; // rdi
  __int64 n; // rax
  __int64 v29; // r14
  char v30; // r9
  __int64 v31; // rax
  int v32; // eax
  __int64 k; // rax
  __int64 v34; // rdi
  int v35; // eax
  __int64 v36; // rax
  int v37; // eax
  int v38; // eax
  int v39; // eax
  char v40; // r9
  unsigned int v41; // ecx
  int v42; // eax
  char v43; // dl
  int v44; // eax
  __int64 v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  char v50; // al
  char v51; // si
  char v52; // al
  char v53; // al
  int v54; // eax
  __int64 v55; // rdi
  int v56; // eax
  _QWORD *v57; // rbx
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rdi
  char j; // al
  unsigned int v62; // eax
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 v65; // rax
  char v66; // [rsp+0h] [rbp-80h]
  char v67; // [rsp+0h] [rbp-80h]
  int v68; // [rsp+0h] [rbp-80h]
  char v69; // [rsp+0h] [rbp-80h]
  char v70; // [rsp+0h] [rbp-80h]
  __int64 v71; // [rsp+8h] [rbp-78h]
  _BOOL4 v72; // [rsp+8h] [rbp-78h]
  char v73; // [rsp+13h] [rbp-6Dh]
  __int64 v75; // [rsp+18h] [rbp-68h]
  __int64 v76; // [rsp+20h] [rbp-60h]
  __int64 v77; // [rsp+20h] [rbp-60h]
  __int64 v78; // [rsp+28h] [rbp-58h]
  __int64 v79; // [rsp+28h] [rbp-58h]
  char v80; // [rsp+28h] [rbp-58h]
  char v81; // [rsp+28h] [rbp-58h]
  char v82; // [rsp+28h] [rbp-58h]
  char v83; // [rsp+28h] [rbp-58h]
  char v84; // [rsp+28h] [rbp-58h]
  int v85; // [rsp+28h] [rbp-58h]
  __int64 v87; // [rsp+30h] [rbp-50h]
  __int64 v88; // [rsp+30h] [rbp-50h]
  __int64 v89; // [rsp+30h] [rbp-50h]
  __int64 v90; // [rsp+30h] [rbp-50h]
  __int64 v91; // [rsp+30h] [rbp-50h]
  __int64 v92; // [rsp+38h] [rbp-48h]
  char v93; // [rsp+43h] [rbp-3Dh] BYREF
  _BOOL4 v94; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 v95[7]; // [rsp+48h] [rbp-38h] BYREF

  v5 = a2;
  v6 = *(_QWORD *)a2;
  v94 = (*(_BYTE *)(a3 + 560) & 0x20) != 0;
  v7 = sub_725D60();
  v73 = (*(_BYTE *)(a3 + 561) & 4) != 0;
  *(_BYTE *)(v7 + 144) = (4 * v73) | *(_BYTE *)(v7 + 144) & 0xFB;
  v8 = ((*(_BYTE *)(a3 + 561) & 8) != 0) | *(_BYTE *)(v7 + 145) & 0xFE;
  *(_BYTE *)(v7 + 145) = v8;
  *(_BYTE *)(v7 + 145) = (*(_BYTE *)(a3 + 561) >> 3) & 2 | v8 & 0xFD;
  for ( i = v6; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v75 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  if ( (*(_BYTE *)(a3 + 561) & 2) != 0 )
  {
    *(_BYTE *)(a1 + 17) |= 0x20u;
    *(_QWORD *)(a1 + 24) = 0;
    v95[0] = sub_72C930();
    goto LABEL_5;
  }
  v29 = *(_QWORD *)(a3 + 288);
  v30 = *(_BYTE *)(v29 + 140);
  v76 = v29;
  if ( v30 == 12 )
  {
    do
    {
      v29 = *(_QWORD *)(v29 + 160);
      v30 = *(_BYTE *)(v29 + 140);
    }
    while ( v30 == 12 );
  }
  else
  {
    v29 = *(_QWORD *)(a3 + 288);
  }
  v71 = *(_QWORD *)a2;
  if ( (*(_BYTE *)(a2 + 9) & 8) != 0 )
  {
    v67 = v30;
    v92 = *(_QWORD *)(a2 + 24);
    v79 = *(_QWORD *)(v92 + 120);
    v34 = (unsigned int)sub_67F240(v79);
    sub_685A50(v34, v92 + 64, v79, 8);
    *(_QWORD *)(v92 + 120) = sub_72C930();
    *(_BYTE *)(a2 + 9) &= ~8u;
    v30 = v67;
LABEL_88:
    if ( dword_4F077C4 != 2 )
      goto LABEL_89;
    goto LABEL_94;
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( unk_4F07778 <= 199900 || dword_4F077C0 || *(_BYTE *)(v71 + 140) == 11 || (*(_BYTE *)(v71 + 179) & 8) == 0 )
      goto LABEL_89;
    v66 = v30;
    v78 = *(_QWORD *)(a2 + 24);
    sub_6851C0(1029, v78 + 64);
    *(_QWORD *)(v78 + 120) = sub_72C930();
    *(_BYTE *)(v71 + 179) &= ~8u;
    v30 = v66;
    goto LABEL_88;
  }
LABEL_94:
  v81 = v30;
  v37 = sub_8D23B0(v29);
  v30 = v81;
  if ( v37 )
  {
    sub_8AE000(v29);
    v30 = v81;
  }
  if ( dword_4F077C4 == 2 )
  {
LABEL_97:
    if ( unk_4D047EC )
    {
      v82 = v30;
      v38 = sub_8DD010(v76);
      v30 = v82;
      if ( v38 )
      {
        sub_6851C0(1065, a1 + 8);
        goto LABEL_92;
      }
    }
    v83 = v30;
    v39 = sub_8D23B0(v29);
    v40 = v83;
    if ( !v39 )
    {
      if ( unk_4D042A4 && (unsigned __int8)(*(_BYTE *)(v29 + 140) - 9) <= 2u && (*(_BYTE *)(v29 + 179) & 8) != 0 )
      {
        if ( *(_BYTE *)(v71 + 140) == 11 )
        {
          *(_BYTE *)(v71 + 179) |= 8u;
        }
        else
        {
          if ( !dword_4F077B8 && (dword_4F077C4 == 2 || unk_4F07778 <= 199900 || unk_4D04964) )
          {
            sub_6851C0(1029, a1 + 8);
            goto LABEL_92;
          }
          *(_BYTE *)(v71 + 179) |= 8u;
        }
LABEL_151:
        if ( !v40 )
          goto LABEL_92;
        v68 = sub_8D32E0(v76);
        if ( (unsigned int)sub_8D5830(v76) )
        {
          sub_5EB950(8u, 322, v76, a1 + 8);
        }
        else if ( (unsigned int)sub_8D3B10(v71) )
        {
          if ( v68 )
          {
            sub_684AA0(qword_4D0495C == 0 ? 7 : 5, 934, a3 + 24);
            v72 = byte_4F07472[0] > 5u;
            v73 &= byte_4F07472[0] <= 5u;
LABEL_156:
            if ( (*(_BYTE *)(a3 + 9) & 0x10) != 0 )
            {
              v45 = 5;
              if ( unk_4D04964 )
                v45 = unk_4F07471;
              sub_684AA0(v45, 1572, a3 + 24);
            }
            if ( !v73 )
              goto LABEL_170;
            goto LABEL_161;
          }
LABEL_233:
          if ( !v73 )
            goto LABEL_93;
          v72 = 0;
LABEL_161:
          if ( (unsigned int)sub_8D2930(v29) )
          {
            if ( dword_4F077C4 == 2 )
            {
              if ( (*(_BYTE *)(a3 + 560) & 0x20) != 0 )
              {
                if ( (*(_BYTE *)(v76 + 140) & 0xFB) != 8 )
                  goto LABEL_170;
                if ( (unsigned int)sub_8D4C10(v76, 0) )
                {
                  v64 = 7;
                  if ( !unk_4D04964 )
                  {
                    v64 = 5;
                    if ( dword_4F077B4 )
                      v64 = unk_4F077A0 < 0x11170u ? 5 : 7;
                  }
                  sub_684AA0(v64, 3259, a3 + 24);
                }
              }
            }
            else if ( unk_4D04964
                   && (unk_4F07778 <= 199900 || !(unsigned int)sub_8D29A0(v29))
                   && ((*(_BYTE *)(v29 + 161) & 8) != 0 || (unsigned __int8)(*(_BYTE *)(v29 + 160) - 5) > 1u) )
            {
              sub_684AA0(byte_4F07472[0], 230, a3 + 24);
            }
            if ( unk_4D044D0 && (*(_BYTE *)(v76 + 140) & 0xFB) == 8 && (sub_8D4C10(v76, dword_4F077C4 != 2) & 8) != 0 )
              sub_6851C0(2780, a3 + 24);
          }
          else if ( !(unsigned int)sub_8D3D40(v29) )
          {
            sub_6851C0(106, a3 + 24);
            goto LABEL_92;
          }
LABEL_170:
          if ( !v72 )
            goto LABEL_93;
          goto LABEL_92;
        }
        if ( v68 )
        {
          v72 = 0;
          goto LABEL_156;
        }
        goto LABEL_233;
      }
      v58 = sub_8D2BE0(v29);
      v40 = v83;
      if ( !v58 )
        goto LABEL_151;
      v69 = v83;
      sub_685360(3414, a1 + 8);
LABEL_237:
      v40 = v69;
      goto LABEL_151;
    }
    if ( dword_4F077C4 != 2 || dword_4F077B4 | dword_4F077BC && (*(_BYTE *)(v71 + 176) & 0x10) == 0 )
    {
      v54 = sub_8D3410(v29);
      v40 = v83;
      if ( v54 )
      {
        v55 = sub_8D40F0(v29);
        v56 = sub_8D23B0(v55);
        v40 = v83;
        if ( !v56 )
        {
          if ( *(_BYTE *)(v71 + 140) != 11 )
          {
            *(_BYTE *)(a2 + 9) |= 8u;
            goto LABEL_223;
          }
          if ( dword_4F077B4 )
          {
            *(_BYTE *)(v71 + 179) |= 8u;
LABEL_223:
            if ( (*(_BYTE *)(a3 + 127) & 4) != 0 )
            {
              sub_6851C0(1049, &dword_4F063F8);
              v65 = sub_72C930();
              v40 = v83;
              v76 = v65;
            }
            else if ( (*(_BYTE *)(a2 + 8) & 9) != 8 && (!dword_4F077B4 || dword_4F077C4 != 2) )
            {
              if ( dword_4F077BC && **(_QWORD **)(v71 + 168) )
              {
                v57 = **(_QWORD ***)(v71 + 168);
                do
                {
                  if ( sub_72FD90(*(_QWORD *)(v57[5] + 160LL), 7) )
                  {
                    v40 = v83;
                    v5 = a2;
                    goto LABEL_151;
                  }
                  v57 = (_QWORD *)*v57;
                }
                while ( v57 );
                v40 = v83;
                v5 = a2;
              }
              v70 = v40;
              sub_6851C0(3251, &dword_4F063F8);
              v40 = v70;
            }
            goto LABEL_151;
          }
        }
      }
    }
    v84 = v40;
    v44 = sub_8D3D40(v76);
    v40 = v84;
    if ( v44 )
      goto LABEL_151;
    v59 = sub_8D4130(v76);
    v40 = v84;
    v60 = v59;
    for ( j = *(_BYTE *)(v59 + 140); j == 12; j = *(_BYTE *)(v60 + 140) )
      v60 = *(_QWORD *)(v60 + 160);
    if ( *(char *)(a2 + 8) >= 0 || (unsigned __int8)(j - 9) > 2u || !dword_4F077BC )
      goto LABEL_243;
    if ( qword_4F077A8 <= 0x76BFu )
      goto LABEL_151;
    v69 = v84;
    if ( !(unsigned int)sub_8DD3B0(v60) )
    {
LABEL_243:
      if ( dword_4F077C4 != 2 || (*(_BYTE *)(a1 + 17) & 0x20) == 0 || (*(_BYTE *)(a3 + 560) & 0x20) != 0 )
      {
        v62 = sub_67F240(v76);
        sub_685A50(v62, a1 + 8, v76, 8);
      }
      goto LABEL_92;
    }
    goto LABEL_237;
  }
LABEL_89:
  v80 = v30;
  v35 = sub_8D2310(v29);
  v30 = v80;
  if ( !v35 || *(_BYTE *)(a3 + 269) == 4 )
    goto LABEL_97;
  sub_6851C0(168, a1 + 8);
LABEL_92:
  v36 = sub_72C930();
  *(_BYTE *)(a3 + 560) &= 0x3Fu;
  v76 = v36;
LABEL_93:
  *(_QWORD *)(a3 + 288) = v76;
  v95[0] = v76;
LABEL_5:
  if ( (*(_WORD *)(a3 + 560) & 0x420) != 0x420 )
    *(_BYTE *)(v5 + 8) |= 8u;
  if ( dword_4F077C4 == 2 && *(_BYTE *)(v6 + 140) == 11 )
  {
    v43 = *(_BYTE *)(a3 + 560);
    if ( ((v43 & 0x40) == 0 || dword_4D04434)
      && !(unsigned int)sub_5E5070(v95[0], v6, (v43 & 0x40) != 0, 0, (*(_BYTE *)(a3 + 127) & 4) != 0, a1 + 8) )
    {
      v95[0] = sub_72C930();
    }
  }
  if ( (*(_BYTE *)(v7 + 144) & 4) != 0 )
  {
    sub_5E53A0(v7, *(_QWORD *)(a3 + 576), &v94, v95, a1 + 8);
    sub_724E30(a3 + 576);
  }
  v10 = v94;
  *(_QWORD *)(v7 + 120) = v95[0];
  if ( !v10 )
  {
    if ( (*(_BYTE *)(a3 + 560) & 0x40) == 0 )
    {
      v11 = sub_885AD0(8, a1, a4, (*(_BYTE *)(v7 + 145) & 2) != 0);
      sub_877D80(v7, v11);
      goto LABEL_14;
    }
LABEL_79:
    v31 = sub_87FA00(8, a3 + 24, *(unsigned int *)(qword_4F04C68[0] + 776LL * (int)a4));
    *(_BYTE *)(v7 + 144) |= 0x10u;
    *(_QWORD *)v7 = v31;
    v11 = v31;
    *(_QWORD *)(v7 + 64) = *(_QWORD *)(a3 + 24);
    goto LABEL_14;
  }
  if ( (*(_BYTE *)(a3 + 561) & 4) == 0 )
  {
    if ( (*(_BYTE *)(a3 + 560) & 0x40) == 0 )
    {
      v11 = sub_87F7E0(8, a1 + 8);
      *(_DWORD *)(v11 + 40) = a4;
      *(_QWORD *)v7 = v11;
      *(_QWORD *)(v7 + 64) = *(_QWORD *)(a3 + 24);
LABEL_14:
      *(_QWORD *)(v11 + 88) = v7;
      *(_QWORD *)a3 = v11;
      goto LABEL_15;
    }
    goto LABEL_79;
  }
  *(_QWORD *)v7 = sub_87EA80();
  *(_QWORD *)(v7 + 64) = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v7 + 72) = sub_7274B0(1);
  *(_QWORD *)(a3 + 520) = *(_QWORD *)(a3 + 564);
  if ( !dword_4F04C3C )
    sub_8699D0(v7, 8, *(_QWORD *)(a3 + 352));
  v11 = 0;
  if ( (*(_BYTE *)(a3 + 561) & 2) != 0 )
  {
    sub_6851C0(787, a3 + 24);
    *(_BYTE *)(a3 + 561) |= 0x20u;
  }
LABEL_15:
  sub_877E20(v11, v7, v6);
  if ( (*(_BYTE *)(a3 + 561) & 2) != 0 && *(_QWORD *)a1 )
  {
    sub_6854C0(786, a1 + 8, v11);
    *(_BYTE *)(a3 + 561) |= 0x20u;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v6 + 168) + 109LL) & 0x20) == 0 || (*(_BYTE *)(a3 + 131) & 4) != 0 )
  {
    if ( v11 && (*(_BYTE *)(a3 + 560) & 0x40) == 0 )
    {
      sub_8756F0(3, v11, a1 + 8, *(_QWORD *)(a3 + 352));
      if ( (*(_BYTE *)(a3 + 131) & 4) == 0 )
        sub_854980(v11, 0);
    }
    else
    {
      sub_854AB0();
    }
  }
  sub_729470(v7, a3 + 472);
  if ( *(_QWORD *)(a3 + 200) )
  {
    if ( *(char *)(a3 + 121) >= 0 )
      goto LABEL_25;
    goto LABEL_139;
  }
  if ( !*(_QWORD *)(a3 + 184) )
    goto LABEL_27;
  if ( *(char *)(a3 + 121) < 0 )
LABEL_139:
    *(_QWORD *)(a3 + 184) = sub_5CF190(*(const __m128i **)(a3 + 184));
LABEL_25:
  sub_6447A0(a3);
  sub_5CF700(*(__int64 **)(a3 + 200));
  sub_5CEC90(*(_QWORD **)(a3 + 200), v7, 8);
  sub_5CF700(*(__int64 **)(a3 + 184));
  sub_5CEC90(*(_QWORD **)(a3 + 184), v7, 8);
  sub_6447E0(a3);
  if ( *(_QWORD *)(v7 + 104) )
    sub_656C00(a3, 8, v7, 0, 1);
LABEL_27:
  v12 = *(_QWORD *)(v7 + 120);
  v13 = *(char *)(v7 + 90) < 0;
  v95[0] = v12;
  if ( v13 && !dword_4F077B8 )
  {
    *(_BYTE *)(v7 + 90) = (16 * *(_BYTE *)(a3 + 126)) & 0x40 | *(_BYTE *)(v7 + 90) & 0xBF;
    goto LABEL_30;
  }
  sub_8D9350(v12, a1 + 8);
  v41 = dword_4F077B8;
  *(_BYTE *)(v7 + 90) = (16 * *(_BYTE *)(a3 + 126)) & 0x40 | *(_BYTE *)(v7 + 90) & 0xBF;
  if ( !v41 )
  {
LABEL_30:
    v14 = *(_QWORD *)(v5 + 24);
    if ( v14 )
      goto LABEL_31;
LABEL_113:
    *(_QWORD *)(v6 + 160) = v7;
    *(_QWORD *)(v5 + 24) = v7;
    if ( dword_4F077C4 != 2 )
      goto LABEL_32;
    goto LABEL_114;
  }
  if ( *(_QWORD *)(a3 + 240) )
    sub_6851C0(1216, a3 + 248);
  *(_BYTE *)(v7 + 145) = (*(_BYTE *)(a3 + 130) >> 2) & 4 | *(_BYTE *)(v7 + 145) & 0xFB;
  v14 = *(_QWORD *)(v5 + 24);
  if ( !v14 )
    goto LABEL_113;
LABEL_31:
  *(_QWORD *)(v14 + 112) = v7;
  *(_QWORD *)(v5 + 24) = v7;
  if ( dword_4F077C4 != 2 )
    goto LABEL_32;
LABEL_114:
  *(_BYTE *)(v7 + 88) = *(_BYTE *)(v5 + 12) & 3 | *(_BYTE *)(v7 + 88) & 0xFC;
  if ( (*(_BYTE *)(a3 + 9) & 0x10) != 0 )
  {
    *(_BYTE *)(v7 + 144) |= 0x20u;
    *(_BYTE *)(v6 + 176) |= 8u;
  }
  if ( (unsigned int)sub_8D32E0(v95[0]) && (*(_BYTE *)(a3 + 128) & 1) == 0 )
    *(_DWORD *)(v75 + 176) = *(_DWORD *)(v75 + 176) & 0xFFD9DFFF | 0x260000;
  *(_BYTE *)(v75 + 179) |= 0x80u;
  if ( (unsigned int)sub_8DBE70(v95[0]) )
  {
    *(_BYTE *)(v75 + 180) |= 4u;
    if ( (*(_BYTE *)(a3 + 224) & 1) == 0 )
      goto LABEL_33;
    goto LABEL_121;
  }
LABEL_32:
  if ( (*(_BYTE *)(a3 + 224) & 1) == 0 )
    goto LABEL_33;
LABEL_121:
  sub_6851C0(1378, a1 + 8);
LABEL_33:
  v15 = sub_8D3410(v95[0]);
  v16 = v95[0];
  if ( v15 )
    v16 = sub_8D40F0(v95[0]);
  if ( (*(_BYTE *)(a3 + 128) & 1) == 0 )
  {
    v17 = v16;
    if ( dword_4F077C4 != 2 && (unk_4F07778 <= 199900 || !unk_4D04964) )
      v17 = v95[0];
    if ( (*(_BYTE *)(v17 + 140) & 0xFB) == 8 )
    {
      v87 = v16;
      v18 = sub_8D4C10(v17, dword_4F077C4 != 2);
      v16 = v87;
      if ( (v18 & 1) != 0 )
        goto LABEL_41;
    }
    v90 = v16;
    v32 = sub_8D3A70(v16);
    v16 = v90;
    if ( v32 )
    {
      for ( k = v90; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      if ( (*(_BYTE *)(k + 176) & 2) != 0 )
      {
LABEL_41:
        *(_BYTE *)(v6 + 176) |= 2u;
        if ( dword_4F077C4 == 2 )
          *(_DWORD *)(v75 + 176) = *(_DWORD *)(v75 + 176) & 0xFFF9DFFF | 0x60000;
      }
    }
  }
  v88 = v16;
  v19 = sub_8D4E00(v95[0]);
  v20 = v88;
  if ( v19 )
    *(_BYTE *)(v6 + 176) |= 4u;
  v21 = *(_BYTE *)(a3 + 560);
  if ( (v21 & 0x40) != 0 )
  {
    sub_5F40E0(v11, v21 >> 7);
    v20 = v88;
  }
  v89 = v20;
  if ( (unsigned int)sub_8D3B80(v95[0]) )
  {
    v22 = v89;
    v23 = 0;
    m = *(_BYTE *)(v89 + 140);
    if ( (m & 0xFB) == 8 )
    {
      v42 = sub_8D4C10(v89, dword_4F077C4 != 2);
      v22 = v89;
      v23 = v42;
      for ( m = *(_BYTE *)(v89 + 140); m == 12; m = *(_BYTE *)(v22 + 140) )
        v22 = *(_QWORD *)(v22 + 160);
    }
    if ( (unsigned __int8)(m - 9) > 2u )
    {
      *(_BYTE *)(v6 + 179) |= 4u;
    }
    else
    {
      if ( (*(_BYTE *)(v22 + 179) & 4) != 0 )
        *(_BYTE *)(v6 + 179) |= 4u;
      if ( dword_4F077C4 == 2 )
      {
        v46 = *(_QWORD *)(*(_QWORD *)v22 + 96LL);
        if ( *(char *)(v46 + 181) >= 0 )
          *(_BYTE *)(v75 + 181) &= ~0x80u;
        if ( (*(_BYTE *)(v46 + 178) & 0x20) != 0 )
          *(_BYTE *)(v75 + 178) |= 0x20u;
        if ( (*(_BYTE *)(v46 + 176) & 1) != 0 || !*(_QWORD *)(v46 + 16) && *(_QWORD *)(v46 + 8) )
        {
          v77 = v22;
          v85 = v23;
          v91 = v46;
          v63 = sub_5EB340(v46);
          v46 = v91;
          v23 = v85;
          v22 = v77;
          if ( !v63 || (*(_BYTE *)(*(_QWORD *)(v63 + 88) + 194LL) & 2) == 0 )
            *(_BYTE *)(v5 + 9) |= 0x10u;
        }
        if ( unk_4D04464 )
        {
          if ( (*(_BYTE *)(v46 + 176) & 8) != 0 )
            *(_BYTE *)(v5 + 10) |= 0x20u;
          if ( *(_QWORD *)(v46 + 32) )
            *(_BYTE *)(v5 + 10) |= 0x40u;
          v47 = *(_QWORD *)(v46 + 16);
          if ( v47 )
          {
            v48 = *(_QWORD *)(v47 + 88);
            if ( (*(_BYTE *)(v48 + 206) & 0x10) != 0 || (*(_BYTE *)(v48 + 88) & 3) != 0 )
              *(_BYTE *)(v5 + 10) |= 0x20u;
          }
        }
        v49 = *(_QWORD *)(v46 + 24);
        if ( v49 )
        {
          if ( (*(_BYTE *)(v46 + 177) & 2) != 0 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v49 + 88) + 206LL) & 0x10) != 0 )
              *(_BYTE *)(v5 + 9) |= 0x80u;
          }
          else
          {
            *(_BYTE *)(v5 + 9) |= 0x20u;
          }
        }
        if ( (*(_BYTE *)(v22 + 177) & 0x20) == 0 )
        {
          v50 = *(_BYTE *)(v46 + 177);
          v51 = *(_BYTE *)(v75 + 177);
          if ( (v51 & 0x40) != 0 )
          {
            if ( (v50 & 0x40) != 0 )
            {
              if ( (v23 & 0xFFFFFFFE) != 0 )
              {
                *(_BYTE *)(v5 + 10) |= 2u;
                v50 = *(_BYTE *)(v46 + 177);
              }
            }
            else
            {
              *(_BYTE *)(v75 + 177) = v51 & 0xBF;
              v50 = *(_BYTE *)(v46 + 177);
            }
          }
          if ( v50 < 0 )
            *(_BYTE *)(v75 + 177) |= 0x80u;
          if ( (*(_BYTE *)(v46 + 178) & 1) != 0 )
            *(_BYTE *)(v75 + 178) |= 1u;
          v52 = *(_BYTE *)(v75 + 177);
          if ( (v52 & 0x20) != 0 )
          {
            if ( (*(_BYTE *)(v46 + 177) & 0x20) != 0 )
            {
              if ( (v23 & 0xFFFFFFFE) != 0 )
                *(_BYTE *)(v5 + 10) |= 8u;
            }
            else
            {
              *(_BYTE *)(v75 + 177) = v52 & 0xDF;
            }
          }
          v53 = *(_BYTE *)(v46 + 178);
          if ( (v53 & 2) != 0 )
          {
            *(_BYTE *)(v75 + 178) |= 2u;
            v53 = *(_BYTE *)(v46 + 178);
          }
          if ( (v53 & 4) != 0 )
            *(_BYTE *)(v75 + 178) |= 4u;
        }
        if ( (*(_BYTE *)(v22 + 176) & 8) != 0 )
          *(_BYTE *)(v6 + 176) |= 8u;
        if ( *(char *)(v46 + 178) >= 0 )
          *(_BYTE *)(v5 + 8) |= 4u;
      }
      if ( (*(_BYTE *)(v22 + 179) & 1) != 0 )
        goto LABEL_54;
    }
    *(_BYTE *)(v5 + 11) |= 4u;
    goto LABEL_54;
  }
  if ( !v94 || (*(_BYTE *)(v7 + 144) & 4) == 0 )
  {
    *(_BYTE *)(v6 + 179) |= 4u;
    *(_BYTE *)(v5 + 11) |= 4u;
  }
  if ( dword_4F077C4 == 2 && (*(_BYTE *)(v5 + 8) & 4) == 0 && (unsigned int)sub_8D32E0(v95[0]) )
    *(_BYTE *)(v5 + 8) |= 4u;
LABEL_54:
  if ( qword_4CF8008 && *(_QWORD *)(qword_4CF8008 + 128) )
    *(_QWORD *)(qword_4CF8008 + 16) = v11;
  v25 = *(_BYTE *)(v5 + 8);
  if ( (v25 & 2) == 0 && *(_BYTE *)(v5 + 12) && (*(_WORD *)(a3 + 560) & 0x420) != 0x420 )
  {
    v25 |= 6u;
    *(_BYTE *)(v5 + 8) = v25;
  }
  v26 = v95[0];
  if ( (v25 & 0x20) == 0 )
  {
    if ( (*(_BYTE *)(a3 + 560) & 0x40) != 0 )
    {
      for ( n = v95[0]; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
        ;
      if ( (*(_BYTE *)(n + 176) & 2) != 0 )
        *(_BYTE *)(v5 + 8) |= 0x20u;
    }
    else if ( (*(_WORD *)(a3 + 560) & 0x420) != 0x420 )
    {
      if ( (unsigned int)sub_8D32E0(v95[0]) )
        goto LABEL_144;
      v26 = v95[0];
      if ( (*(_BYTE *)(v95[0] + 140) & 0xFB) != 8 )
        goto LABEL_67;
      if ( (sub_8D4C10(v95[0], dword_4F077C4 != 2) & 1) != 0 )
LABEL_144:
        *(_BYTE *)(v5 + 8) |= 0x20u;
      v26 = v95[0];
    }
  }
LABEL_67:
  *(_BYTE *)(v5 + 8) &= ~1u;
  sub_8D9610(v26, &v93);
  *(_WORD *)(v6 + 180) = *(_WORD *)(v6 + 180) & 0xFC3F | ((v93 & 0xF | (*(_WORD *)(v6 + 180) >> 6) & 0xF) << 6);
  return v7;
}
