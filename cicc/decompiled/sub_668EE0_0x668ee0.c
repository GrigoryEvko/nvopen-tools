// Function: sub_668EE0
// Address: 0x668ee0
//
_QWORD *__fastcall sub_668EE0(
        unsigned __int8 a1,
        __m128i *a2,
        _DWORD *a3,
        int a4,
        _DWORD *a5,
        int a6,
        int a7,
        int *a8,
        _DWORD *a9,
        _DWORD *a10,
        __int64 a11)
{
  char v13; // bl
  int v14; // r14d
  unsigned int v15; // r14d
  unsigned __int16 v16; // ax
  __int64 v17; // rdi
  __int64 k; // rdx
  int v19; // r9d
  __int64 v20; // rsi
  _BOOL4 v21; // r8d
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r11
  _QWORD *v25; // r14
  char v26; // al
  __int64 v27; // rax
  unsigned __int8 v29; // al
  bool i; // cc
  __m128i v31; // xmm7
  __int64 v32; // rax
  __int64 v33; // rax
  int v34; // eax
  unsigned __int16 v35; // ax
  int v36; // eax
  __int64 v37; // rax
  int v38; // r9d
  __int64 v39; // rdx
  __int64 v40; // rax
  char v41; // al
  __int64 v42; // rbx
  __int64 v43; // rcx
  __int64 *v44; // rdx
  int v45; // eax
  _QWORD *v46; // rcx
  __int64 v47; // rax
  char v48; // al
  char v49; // al
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // r9d
  char v56; // al
  __int64 j; // rax
  __int64 v58; // rdi
  int v59; // eax
  __int64 v60; // rdx
  int v61; // r10d
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  int v65; // r10d
  int v66; // eax
  const char *v67; // rdx
  int v68; // r10d
  _BOOL4 v69; // r8d
  int v70; // r11d
  int v71; // r9d
  _BOOL4 v72; // [rsp+0h] [rbp-C0h]
  int v73; // [rsp+0h] [rbp-C0h]
  int v74; // [rsp+8h] [rbp-B8h]
  __int64 v75; // [rsp+8h] [rbp-B8h]
  char v76; // [rsp+8h] [rbp-B8h]
  char v77; // [rsp+8h] [rbp-B8h]
  int v78; // [rsp+8h] [rbp-B8h]
  int v79; // [rsp+8h] [rbp-B8h]
  __int64 v80; // [rsp+8h] [rbp-B8h]
  int v81; // [rsp+8h] [rbp-B8h]
  int v82; // [rsp+10h] [rbp-B0h]
  _BOOL4 v83; // [rsp+10h] [rbp-B0h]
  _BOOL4 v84; // [rsp+10h] [rbp-B0h]
  _BOOL4 v85; // [rsp+10h] [rbp-B0h]
  _BOOL4 v86; // [rsp+10h] [rbp-B0h]
  __int64 v87; // [rsp+10h] [rbp-B0h]
  _BOOL4 v88; // [rsp+10h] [rbp-B0h]
  _BOOL4 v89; // [rsp+10h] [rbp-B0h]
  _BOOL4 v90; // [rsp+10h] [rbp-B0h]
  int v91; // [rsp+10h] [rbp-B0h]
  _BOOL4 v92; // [rsp+10h] [rbp-B0h]
  _BOOL4 v93; // [rsp+10h] [rbp-B0h]
  _BOOL4 v94; // [rsp+10h] [rbp-B0h]
  _BOOL4 v95; // [rsp+18h] [rbp-A8h]
  int v96; // [rsp+18h] [rbp-A8h]
  int v97; // [rsp+18h] [rbp-A8h]
  int v98; // [rsp+18h] [rbp-A8h]
  int v99; // [rsp+18h] [rbp-A8h]
  int v100; // [rsp+18h] [rbp-A8h]
  int v101; // [rsp+18h] [rbp-A8h]
  int v102; // [rsp+18h] [rbp-A8h]
  int v103; // [rsp+18h] [rbp-A8h]
  int v104; // [rsp+18h] [rbp-A8h]
  int v105; // [rsp+18h] [rbp-A8h]
  int v106; // [rsp+18h] [rbp-A8h]
  int v107; // [rsp+18h] [rbp-A8h]
  int v108; // [rsp+18h] [rbp-A8h]
  int v109; // [rsp+18h] [rbp-A8h]
  int v110; // [rsp+18h] [rbp-A8h]
  int v112; // [rsp+20h] [rbp-A0h]
  int v113; // [rsp+20h] [rbp-A0h]
  int v114; // [rsp+20h] [rbp-A0h]
  int v115; // [rsp+20h] [rbp-A0h]
  _BOOL4 v116; // [rsp+20h] [rbp-A0h]
  int v117; // [rsp+20h] [rbp-A0h]
  int v118; // [rsp+20h] [rbp-A0h]
  int v119; // [rsp+20h] [rbp-A0h]
  int v120; // [rsp+20h] [rbp-A0h]
  int v121; // [rsp+20h] [rbp-A0h]
  __int64 v122; // [rsp+20h] [rbp-A0h]
  int v123; // [rsp+20h] [rbp-A0h]
  int v124; // [rsp+20h] [rbp-A0h]
  int v125; // [rsp+20h] [rbp-A0h]
  unsigned int v127; // [rsp+34h] [rbp-8Ch]
  int v129; // [rsp+38h] [rbp-88h]
  int v130; // [rsp+38h] [rbp-88h]
  int v131; // [rsp+38h] [rbp-88h]
  int v132; // [rsp+38h] [rbp-88h]
  unsigned __int16 v133; // [rsp+4Ah] [rbp-76h] BYREF
  int v134; // [rsp+4Ch] [rbp-74h] BYREF
  char s[112]; // [rsp+50h] [rbp-70h] BYREF

  v13 = a6;
  v14 = -(a6 == 0);
  *a9 = 0;
  LOBYTE(v14) = 0;
  v15 = v14 + 2359553;
  v134 = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 || (unsigned int)sub_7C0F00(v15, 0) )
    {
LABEL_3:
      v16 = sub_7BE840(0, 0);
      v133 = v16;
      if ( dword_4F077C4 == 2
        && (unk_4F07778 > 201102 || dword_4F07774 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9EFBu)
        && v16 == 1 )
      {
        if ( (v13 & 1) != 0 || a1 == 6 )
        {
          v127 = 0;
          v17 = 1;
          goto LABEL_6;
        }
        sub_668C50(&v133, 73, 1);
        v16 = v133;
      }
      if ( v16 == 17 )
      {
        sub_7BEB10(17, s);
        if ( *(_WORD *)s == 75 )
        {
          v17 = 75;
          v133 = 75;
        }
        else
        {
          v17 = v133;
        }
        v127 = 1;
      }
      else
      {
        v127 = 0;
        v17 = v16;
      }
LABEL_6:
      v19 = sub_667FD0(v17, a1, v13, a7);
      if ( !dword_4F077BC || a1 == 6 )
      {
        v20 = 0;
        v21 = 0;
        goto LABEL_21;
      }
      v20 = 0;
      v21 = 0;
      if ( (unk_4D04A10 & 0x2001) != 1 )
        goto LABEL_21;
      if ( qword_4F077A8 > 0x76BFu )
      {
        if ( unk_4F04C48 != -1 )
        {
          k = (__int64)qword_4F04C68;
          v21 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0;
        }
LABEL_21:
        v23 = (unsigned int)dword_4F077C4;
        if ( dword_4F077C4 != 2 )
          goto LABEL_22;
        goto LABEL_35;
      }
      v21 = 1;
      if ( (unk_4D04A12 & 2) != 0 )
        goto LABEL_21;
      v22 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      k = *(unsigned __int8 *)(v22 + 4);
      if ( (unsigned __int8)(k - 3) <= 1u )
      {
        if ( xmmword_4D04A20.m128i_i64[0] != *(_QWORD *)(*(_QWORD *)(v22 + 184) + 32LL) )
          goto LABEL_21;
        if ( !v19 && v133 != 75 )
        {
          if ( xmmword_4D04A20.m128i_i64[0] )
          {
            v17 = (__int64)&qword_4D04A00;
            v54 = sub_7D4A40(&qword_4D04A00, xmmword_4D04A20.m128i_i64[0], 2);
            v55 = 0;
            goto LABEL_192;
          }
LABEL_243:
          v17 = unk_4F07288;
          v54 = sub_7D4600(unk_4F07288, &qword_4D04A00, 2);
          v55 = 0;
LABEL_192:
          if ( (unk_4D04A11 & 0x40) == 0 )
          {
            unk_4D04A10 &= ~0x80u;
            unk_4D04A18 = 0;
          }
          if ( v54 )
          {
            v20 = 0;
            v21 = 1;
            v19 = 0;
            goto LABEL_21;
          }
          goto LABEL_223;
        }
        v58 = 1285;
      }
      else
      {
        if ( xmmword_4D04A20.m128i_i64[0] || (_BYTE)k )
        {
          v20 = 0;
          v21 = 1;
          goto LABEL_21;
        }
        v58 = 1372;
        if ( !v19 )
        {
          if ( v133 != 75 )
            goto LABEL_243;
          v58 = 1372;
        }
      }
      v99 = v19;
      sub_684B30(v58, &dword_4F063F8);
      v55 = v99;
LABEL_223:
      v17 = (__int64)&qword_4D04A00;
      v100 = v55;
      sub_878790(&qword_4D04A00);
      v19 = v100;
      v20 = 0;
      v21 = 1;
      goto LABEL_21;
    }
  }
  else if ( word_4F06418[0] == 1 )
  {
    goto LABEL_3;
  }
  v17 = 40;
  sub_6851C0(40, dword_4F07508);
  v20 = 1;
  v21 = 0;
  v19 = 0;
  v127 = 0;
  v23 = (unsigned int)dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
LABEL_22:
    if ( dword_4D04360 )
    {
      LODWORD(v24) = -1;
      goto LABEL_24;
    }
  }
LABEL_35:
  v24 = *a8;
  k = qword_4F04C68[0] + 776 * v24;
  v29 = *(_BYTE *)(k + 4);
  for ( i = v29 <= 9u; v29 != 9; i = v29 <= 9u )
  {
    if ( i )
    {
      if ( !v29 || (unsigned __int8)(v29 - 2) <= 2u )
        goto LABEL_39;
    }
    else if ( v29 == 17 )
    {
      goto LABEL_39;
    }
    v29 = *(_BYTE *)(k - 772);
    k -= 776;
    LODWORD(v24) = v24 - 1;
  }
  v33 = *(_QWORD *)(k + 360);
  if ( (unk_4D04A10 & 0x10001) == 0
    || !v33
    || (v17 = *(unsigned __int8 *)(v33 + 80), k = (unsigned int)(v17 - 4), (unsigned __int8)(v17 - 4) > 1u)
    || (*(_BYTE *)(*(_QWORD *)(v33 + 88) + 177LL) & 0x30) != 0x30 )
  {
    LODWORD(v24) = dword_4F04C34;
  }
LABEL_39:
  if ( (_DWORD)v23 != 2 )
  {
LABEL_24:
    if ( !(_DWORD)v20 )
    {
      v25 = 0;
      goto LABEL_26;
    }
LABEL_41:
    *a2 = _mm_loadu_si128(xmmword_4F06660);
    a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v31 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v32 = *(_QWORD *)dword_4F07508;
    a2[1].m128i_i8[1] |= 0x20u;
    a2[3] = v31;
    a2->m128i_i64[1] = v32;
LABEL_42:
    v25 = 0;
    goto LABEL_31;
  }
  if ( (_DWORD)v20 )
    goto LABEL_41;
  v20 = 1;
  v17 = v15;
  v72 = v21;
  v74 = v24;
  v82 = v19;
  v34 = sub_7C8410(v15, 1, &v134);
  v19 = v82;
  LODWORD(v24) = v74;
  v21 = v72;
  if ( v34 )
    goto LABEL_93;
  if ( word_4F06418[0] != 1 )
    goto LABEL_57;
  if ( (unk_4D04A12 & 1) != 0 )
  {
LABEL_93:
    if ( v134 )
      goto LABEL_41;
    if ( !v82 && (!a4 || v133 != 75) )
    {
      if ( dword_4F077C4 == 2 )
      {
        if ( !unk_4D04A18 )
          goto LABEL_57;
        if ( (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x41000) != 0 )
        {
          v20 = 0;
          v17 = (__int64)&qword_4D04A00;
          sub_8841F0(&qword_4D04A00, 0, 0, 0);
          v19 = 0;
          LODWORD(v24) = v74;
          v21 = v72;
        }
      }
LABEL_98:
      v25 = (_QWORD *)unk_4D04A18;
      if ( !unk_4D04A18 )
        goto LABEL_57;
      v45 = *(unsigned __int8 *)(unk_4D04A18 + 80LL);
      if ( (_BYTE)v45 == 16 )
      {
        v25 = **(_QWORD ***)(unk_4D04A18 + 88LL);
        v45 = *((unsigned __int8 *)v25 + 80);
      }
      if ( (_BYTE)v45 == 24 )
      {
        v25 = (_QWORD *)v25[11];
        v45 = *((unsigned __int8 *)v25 + 80);
      }
      if ( (_BYTE)v45 != 3 )
      {
        if ( a1 != (_BYTE)v45 )
        {
LABEL_105:
          if ( (_BYTE)v45 == 19 )
          {
            if ( a1 == 6 || (unk_4D04A12 & 0x20) == 0 )
            {
              v23 = (__int64)v25;
              v25 = 0;
              goto LABEL_130;
            }
            if ( (unsigned int)qword_4F077B4 | dword_4D04964 )
            {
              v20 = 2613;
              v17 = 8;
              v94 = v21;
              v110 = v24;
              v125 = v19;
              sub_684AA0(8, 2613, &qword_4D04A08);
              v23 = (__int64)v25;
              v25 = 0;
              v19 = v125;
              LODWORD(v24) = v110;
              v21 = v94;
              goto LABEL_130;
            }
            v46 = v25;
            v25 = 0;
            goto LABEL_112;
          }
          k = (unsigned int)(v45 - 4);
          if ( (unsigned __int8)(v45 - 4) > 1u || !*(_QWORD *)(v25[12] + 72LL) || a1 == 6 )
          {
            v20 = (__int64)&qword_4D04A08;
            v17 = 469;
            sub_686A10(469, &qword_4D04A08, *(&off_4AF8080 + a1), v25);
            goto LABEL_41;
          }
LABEL_109:
          if ( (unk_4D04A12 & 0x20) == 0 )
            goto LABEL_26;
          if ( (unsigned int)qword_4F077B4 | dword_4D04964 )
          {
            v20 = 2613;
            v17 = 8;
            v93 = v21;
            v109 = v24;
            v124 = v19;
            sub_684AA0(8, 2613, &qword_4D04A08);
            v19 = v124;
            LODWORD(v24) = v109;
            v21 = v93;
            goto LABEL_113;
          }
          v46 = 0;
LABEL_112:
          v20 = 2613;
          v17 = 5;
          v75 = (__int64)v46;
          v83 = v21;
          v96 = v24;
          v113 = v19;
          sub_684AA0(5, 2613, &qword_4D04A08);
          v23 = v75;
          v19 = v113;
          LODWORD(v24) = v96;
          v21 = v83;
          if ( !v75 )
            goto LABEL_113;
LABEL_130:
          v48 = *(_BYTE *)(v23 + 80);
          if ( v48 == 19 )
          {
            v20 = 0x200000;
            v17 = v23;
            v101 = v24;
            v86 = v21;
            v117 = v19;
            v25 = (_QWORD *)sub_7BF840(v23, 0x200000, &v134);
            if ( v134 )
              goto LABEL_41;
            v19 = v117;
            LODWORD(v24) = v101;
            v21 = v86;
            goto LABEL_113;
          }
LABEL_131:
          if ( (unsigned __int8)(v48 - 4) <= 1u && *(_QWORD *)(*(_QWORD *)(v23 + 96) + 72LL) )
          {
            v25 = (_QWORD *)v23;
            goto LABEL_26;
          }
          if ( (unk_4D04A11 & 0x40) == 0 )
          {
            unk_4D04A10 &= ~0x80u;
            unk_4D04A18 = 0;
            if ( v25 )
              goto LABEL_26;
            goto LABEL_57;
          }
LABEL_113:
          if ( v25 )
            goto LABEL_26;
          goto LABEL_57;
        }
LABEL_206:
        if ( a1 == 6 )
          goto LABEL_26;
        goto LABEL_109;
      }
      v17 = v25[11];
      if ( *((_BYTE *)v25 + 104) )
      {
        k = *(_QWORD *)v17;
LABEL_198:
        v45 = *(unsigned __int8 *)(k + 80);
        v25 = (_QWORD *)k;
        if ( a1 == (_BYTE)v45 )
          goto LABEL_206;
        if ( (_BYTE)v45 != 3 )
          goto LABEL_105;
        v23 = *(unsigned __int8 *)(*(_QWORD *)(k + 88) + 140LL);
        goto LABEL_201;
      }
      v23 = *(unsigned __int8 *)(v17 + 140);
      if ( (_BYTE)v23 == 12 )
      {
        do
        {
          v17 = *(_QWORD *)(v17 + 160);
          v56 = *(_BYTE *)(v17 + 140);
        }
        while ( v56 == 12 );
      }
      else
      {
        v56 = *(_BYTE *)(v17 + 140);
      }
      k = *(_QWORD *)v17;
      if ( v21 )
      {
        if ( (unsigned __int8)(v56 - 9) <= 2u )
          goto LABEL_198;
      }
      else if ( dword_4F077BC && qword_4F077A8 <= 0x76BFu )
      {
        v87 = *(_QWORD *)v17;
        v102 = v24;
        v118 = v19;
        v59 = sub_8D2870(v17);
        v19 = v118;
        LODWORD(v24) = v102;
        v60 = v87;
        v21 = 0;
        if ( v59 )
        {
          v17 = v87;
          v91 = v102;
          v107 = v118;
          v122 = v60;
          v66 = sub_879510(v17);
          k = v122;
          v19 = v107;
          LODWORD(v24) = v91;
          v21 = 0;
          if ( !v66 )
            k = (__int64)v25;
        }
        else
        {
          k = (__int64)v25;
        }
        goto LABEL_198;
      }
LABEL_201:
      if ( (_BYTE)v23 != 14 )
      {
        if ( (*((_BYTE *)v25 + 81) & 0x20) == 0 )
        {
          v20 = (__int64)&qword_4D04A08;
          v17 = 1201;
          sub_6851A0(1201, &qword_4D04A08, *(_QWORD *)(*v25 + 8LL));
        }
        goto LABEL_41;
      }
      goto LABEL_206;
    }
    k = qword_4F04C68[0];
    v23 = (unsigned int)dword_4F077C4;
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 8) != 0 )
    {
      if ( dword_4F077C4 != 2 )
      {
        if ( dword_4F04C40 == -1 || !*(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
          goto LABEL_98;
        v49 = 0;
        goto LABEL_140;
      }
      v49 = 0;
    }
    else
    {
      if ( dword_4F077C4 != 2 )
      {
        if ( dword_4F04C40 == -1 )
          goto LABEL_98;
        if ( !*(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
        {
LABEL_142:
          if ( dword_4F077C4 == 2 )
          {
            v23 = (__int64)qword_4F04C68;
            v17 = (int)dword_4F04C40;
            *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) &= ~8u;
            k = qword_4F04C68[0];
            if ( *(_QWORD *)(qword_4F04C68[0] + 776 * v17 + 456) )
            {
              v85 = v21;
              v98 = v24;
              v115 = v19;
              sub_8845B0(v17);
              v21 = v85;
              LODWORD(v24) = v98;
              v19 = v115;
            }
          }
          goto LABEL_98;
        }
        v49 = 1;
        goto LABEL_140;
      }
      *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
      v49 = 1;
    }
    if ( unk_4D04A18 && (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x41000) != 0 )
    {
      v20 = 0;
      v17 = (__int64)&qword_4D04A00;
      v77 = v49;
      v103 = v24;
      sub_8841F0(&qword_4D04A00, 0, 0, 0);
      v19 = v82;
      LODWORD(v24) = v103;
      v21 = v72;
      v49 = v77;
    }
    k = (int)dword_4F04C40;
    if ( dword_4F04C40 == -1
      || (v23 = (__int64)qword_4F04C68, k = qword_4F04C68[0] + 776LL * (int)dword_4F04C40, !*(_QWORD *)(k + 456)) )
    {
LABEL_141:
      if ( !v49 )
        goto LABEL_98;
      goto LABEL_142;
    }
LABEL_140:
    v76 = v49;
    v84 = v21;
    v97 = v24;
    v114 = v19;
    sub_87DD80();
    v49 = v76;
    v21 = v84;
    LODWORD(v24) = v97;
    v19 = v114;
    goto LABEL_141;
  }
  k = (__int64)&dword_4F04C5C;
  if ( dword_4F04C5C != dword_4F04C34 || a1 == 6 )
    goto LABEL_57;
  v20 = 32;
  v17 = (__int64)&qword_4D04A00;
  v47 = sub_7D5DD0(&qword_4D04A00, 32);
  v19 = v82;
  LODWORD(v24) = v74;
  v23 = v47;
  v21 = v72;
  if ( !dword_4F077BC )
  {
    if ( !v47 )
      goto LABEL_57;
    v25 = 0;
    goto LABEL_130;
  }
  if ( qword_4F077A8 <= 0x765Bu
    || qword_4F077A8 > 0x9CA3u
    || (k = qword_4F04C68[0] + 776LL * dword_4F04C64, *(_BYTE *)(k + 4) != 9)
    || v47 != *(_QWORD *)(k + 368) )
  {
    v25 = 0;
    if ( !v47 )
      goto LABEL_57;
    goto LABEL_130;
  }
  v48 = *(_BYTE *)(v47 + 80);
  v25 = 0;
  if ( v48 != 19 )
    goto LABEL_131;
  v25 = *(_QWORD **)(k + 360);
  if ( v25 )
    goto LABEL_26;
LABEL_57:
  if ( (unk_4D04A10 & 1) != 0 )
  {
    *a5 = 0;
    if ( (unk_4D04A10 & 0x58) == 0 )
      goto LABEL_63;
    goto LABEL_59;
  }
  k = (__int64)&dword_4F04C34;
  v20 = (unsigned int)dword_4F04C34;
  if ( a1 == 6 || dword_4F04C34 != (_DWORD)v24 )
  {
    if ( (unk_4D04A10 & 0x58) == 0 )
      goto LABEL_63;
    goto LABEL_59;
  }
  v17 = qword_4D04A00;
  k = 0;
  while ( 1 )
  {
    v61 = k;
    v23 = qword_4F06C80[k];
    if ( v23 )
    {
      v25 = *(_QWORD **)v23;
      if ( **(_QWORD **)v23 == qword_4D04A00 )
        break;
    }
    if ( ++k == 11 )
    {
      if ( (unk_4D04A10 & 0x58) != 0 )
        goto LABEL_59;
      goto LABEL_63;
    }
  }
  v23 = (__int64)qword_4F04C68;
  v62 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 224);
  if ( (_DWORD)k )
  {
    k = (__int64)qword_4D04998;
    if ( v62 != qword_4D04998[11] )
      goto LABEL_255;
    goto LABEL_269;
  }
  k = (__int64)&dword_4F06900;
  if ( !dword_4F06900 || (k = (__int64)&dword_4D042AC, dword_4D042AC) )
  {
    if ( !dword_4F04C64 )
      goto LABEL_269;
LABEL_255:
    if ( (unk_4D04A10 & 0x58) == 0 )
      goto LABEL_63;
    goto LABEL_59;
  }
  k = (__int64)qword_4D049B8;
  if ( v62 != qword_4D049B8[11] )
    goto LABEL_255;
LABEL_269:
  v20 = (__int64)v25;
  v17 = 25;
  v78 = v61;
  v88 = v21;
  v104 = v24;
  v119 = v19;
  v64 = sub_854840(25, v25, 0, 1);
  v19 = v119;
  LODWORD(v24) = v104;
  v21 = v88;
  v65 = v78;
  if ( v64 )
  {
    if ( unk_4D04324 )
    {
      v20 = 878;
      v73 = v78;
      v80 = v64;
      sub_684AB0(&dword_4F063F8, 878);
      v65 = v73;
      v64 = v80;
      v21 = v88;
      LODWORD(v24) = v104;
      v19 = v119;
    }
    v17 = v64;
    v79 = v65;
    v89 = v21;
    v105 = v24;
    v120 = v19;
    sub_854000(v64);
    v19 = v120;
    LODWORD(v24) = v105;
    v21 = v89;
    v65 = v79;
  }
  else if ( unk_4F07584 )
  {
    if ( *((_DWORD *)v25 + 10) != -1 )
      goto LABEL_26;
    if ( v78 )
    {
      sprintf(s, "__cxxabiv1::%s", *(const char **)(*v25 + 8LL));
      v71 = v119;
      v70 = v104;
      v69 = v88;
      v68 = v78;
    }
    else
    {
      v67 = "std::";
      if ( !dword_4F06900 )
        v67 = byte_3F871B3;
      sprintf(s, "%s%s", v67, "type_info");
      v68 = 0;
      v69 = v88;
      v70 = v104;
      v71 = v119;
    }
    v20 = (__int64)&qword_4D04A08;
    v17 = 772;
    v81 = v68;
    v92 = v69;
    v108 = v70;
    v123 = v71;
    sub_6851A0(772, &qword_4D04A08, s);
    v65 = v81;
    v21 = v92;
    LODWORD(v24) = v108;
    v19 = v123;
  }
  if ( *((_DWORD *)v25 + 10) == -1 )
  {
    v23 = (__int64)qword_4F06C80;
    v17 = qword_4F06C80[v65];
    if ( v17 )
    {
      v20 = (unsigned int)v24;
      v90 = v21;
      v106 = v19;
      v121 = v24;
      sub_736B60(v17, (unsigned int)v24, &qword_4D04A08);
      LODWORD(v24) = v121;
      v19 = v106;
      v21 = v90;
      *a10 = 1;
    }
  }
LABEL_26:
  v26 = unk_4D04A10;
  if ( (unk_4D04A10 & 1) != 0 )
  {
    *a5 = 0;
    v26 = unk_4D04A10;
  }
  if ( (v26 & 0x58) != 0 )
  {
LABEL_59:
    v20 = (__int64)&qword_4D04A08;
    v17 = 502;
    sub_6851C0(502, &qword_4D04A08);
    goto LABEL_41;
  }
  if ( v25 )
  {
    *a2 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    a2[1] = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
    a2[2] = _mm_loadu_si128(&xmmword_4D04A20);
    a2[3] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    goto LABEL_31;
  }
LABEL_63:
  if ( (unk_4D04A11 & 0x20) != 0 )
    goto LABEL_41;
  v35 = v133;
  *a2 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  a2[1] = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
  a2[2] = _mm_loadu_si128(&xmmword_4D04A20);
  a2[3] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  if ( v35 == 75 )
  {
    v127 |= *a5;
    if ( v127 )
      goto LABEL_66;
LABEL_120:
    v36 = v19;
    goto LABEL_67;
  }
  if ( v127 )
  {
LABEL_66:
    v127 = (dword_4F077C4 != 1) & ((unsigned __int8)v13 ^ 1);
    v36 = v19 | v127;
    goto LABEL_67;
  }
  if ( v35 == 73 || v35 == 55 )
    goto LABEL_120;
  v17 = (unsigned int)*a3;
  if ( !(_DWORD)v17 )
    goto LABEL_120;
  *a3 = 0;
  v36 = v19;
LABEL_67:
  if ( !v36 )
    goto LABEL_147;
  v95 = v21;
  v112 = v24;
  v129 = v19;
  if ( v21 )
  {
    v20 = 2048;
    v17 = (__int64)a2;
    v37 = sub_7CFB70(a2, 2048);
    v38 = v129;
    LODWORD(v24) = v112;
    v21 = v95;
    v25 = (_QWORD *)v37;
    if ( !v37 )
      goto LABEL_146;
    if ( *(_BYTE *)(v37 + 80) != 3 )
      goto LABEL_71;
    for ( j = *(_QWORD *)(v37 + 88); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v25 = *(_QWORD **)j;
  }
  else
  {
    v20 = 2;
    v17 = (__int64)a2;
    v53 = sub_7CFB70(a2, 2);
    v38 = v129;
    LODWORD(v24) = v112;
    v21 = 0;
    v25 = (_QWORD *)v53;
  }
  if ( !v25 )
  {
LABEL_146:
    if ( v38 )
      goto LABEL_42;
    goto LABEL_147;
  }
  if ( *((_BYTE *)v25 + 80) != 3 || !*((_BYTE *)v25 + 104) )
  {
LABEL_71:
    if ( !v38 )
    {
      v41 = *((_BYTE *)v25 + 80);
      v42 = a1;
      goto LABEL_86;
    }
    goto LABEL_72;
  }
  k = (__int64)&qword_4D0495C;
  v23 = (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
  if ( !qword_4D0495C )
  {
    if ( v38 )
      goto LABEL_42;
    goto LABEL_154;
  }
  k = v25[11];
  v25 = *(_QWORD **)k;
  if ( v38 )
  {
    if ( !v25 )
      goto LABEL_42;
LABEL_72:
    if ( (*((_BYTE *)v25 + 81) & 2) == 0 )
    {
      if ( dword_4F077C4 == 2 || (v20 = v25[11], v23 = (unsigned int)dword_4F04C64, !dword_4F04C64) )
      {
LABEL_182:
        v42 = a1;
        *a9 = 1;
        v41 = *((_BYTE *)v25 + 80);
        goto LABEL_86;
      }
      v39 = 776LL * dword_4F04C64;
      v23 = 776LL * (unsigned int)(dword_4F04C64 - 1);
      v17 = dword_4F07588;
      v40 = qword_4F04C68[0] + v39 + 4;
      k = qword_4F04C68[0] + v39 - 772 - v23;
      while ( 1 )
      {
        if ( *(_BYTE *)v40 == 6 )
        {
          v23 = *(_QWORD *)(v40 + 204);
          if ( v20 == v23 )
            break;
          if ( v20 )
          {
            if ( v23 )
            {
              if ( dword_4F07588 )
              {
                v23 = *(_QWORD *)(v23 + 32);
                if ( *(_QWORD *)(v20 + 32) == v23 )
                {
                  if ( v23 )
                    break;
                }
              }
            }
          }
        }
        v40 -= 776;
        if ( k == v40 )
          goto LABEL_182;
      }
    }
    if ( a1 != 6 )
      goto LABEL_42;
    v41 = *((_BYTE *)v25 + 80);
    v42 = 6;
LABEL_86:
    if ( a1 == v41 )
      goto LABEL_31;
    v43 = (__int64)*(&off_4AF8080 + v42);
    v44 = &a2->m128i_i64[1];
    if ( !qword_4D0495C )
      goto LABEL_88;
    goto LABEL_179;
  }
  if ( v25 )
  {
    v41 = *((_BYTE *)v25 + 80);
    k = a1;
    if ( a1 == v41 )
      goto LABEL_31;
    v43 = (__int64)*(&off_4AF8080 + a1);
    v44 = &a2->m128i_i64[1];
LABEL_179:
    if ( v41 != 6 && a1 != 6 )
    {
      v20 = 469;
      v17 = 5;
      sub_6868B0(5, 469, v44, v43, v25);
      goto LABEL_31;
    }
LABEL_88:
    v20 = 469;
    v17 = 8;
    v130 = v24;
    sub_6868B0(8, 469, v44, v43, v25);
    if ( dword_4F077C4 == 2 )
      *a8 = v130;
    goto LABEL_41;
  }
LABEL_147:
  v23 = HIDWORD(qword_4D0495C);
  if ( HIDWORD(qword_4D0495C) || (v20 = (unsigned int)qword_4D0495C, (_DWORD)qword_4D0495C) )
  {
    if ( a1 != 6 )
    {
      v20 = 0;
      v17 = (__int64)a2;
      v116 = v21;
      v131 = v24;
      v50 = sub_7CFB70(a2, 0);
      LODWORD(v24) = v131;
      v21 = v116;
      v25 = (_QWORD *)v50;
      if ( v50 )
      {
        if ( *(_BYTE *)(v50 + 80) == 3 )
        {
          v63 = *(_QWORD *)(v50 + 88);
          for ( k = *(unsigned __int8 *)(v63 + 140); (_BYTE)k == 12; k = *(unsigned __int8 *)(v63 + 140) )
            v63 = *(_QWORD *)(v63 + 160);
          if ( (unsigned __int8)(k - 9) <= 2u )
          {
            LOBYTE(v23) = a1 == 5;
            if ( (a1 == 5) == ((_BYTE)k == 11) )
              goto LABEL_31;
          }
        }
        if ( (a2[1].m128i_i8[1] & 0x40) == 0 )
        {
          a2[1].m128i_i8[0] &= ~0x80u;
          a2[1].m128i_i64[1] = 0;
        }
      }
    }
  }
LABEL_154:
  k = v127;
  if ( v127 )
    goto LABEL_42;
  v42 = a1;
  v17 = (__int64)a2;
  v20 = a1;
  v132 = v24;
  v51 = sub_7D6E80(a2, a1, v21, (unsigned int)*a3);
  LODWORD(v24) = v132;
  v25 = (_QWORD *)v51;
  if ( !v51 )
  {
    if ( dword_4F077C4 == 2 || !dword_4D04360 )
    {
      *a8 = v132;
      goto LABEL_31;
    }
    goto LABEL_42;
  }
  v41 = *(_BYTE *)(v51 + 80);
  if ( v41 != 3 )
    goto LABEL_86;
  v52 = v25[11];
  if ( *((_BYTE *)v25 + 104) )
  {
LABEL_163:
    v25 = *(_QWORD **)v52;
    if ( !*(_QWORD *)v52 )
      goto LABEL_42;
    v41 = *((_BYTE *)v25 + 80);
    goto LABEL_86;
  }
  k = *(unsigned __int8 *)(v52 + 140);
  if ( (_BYTE)k != 12 )
  {
    if ( (_BYTE)k == 14 )
      goto LABEL_31;
    goto LABEL_163;
  }
  k = v25[11];
  do
  {
    k = *(_QWORD *)(k + 160);
    v23 = *(unsigned __int8 *)(k + 140);
  }
  while ( (_BYTE)v23 == 12 );
  if ( (_BYTE)v23 != 14 )
  {
    do
      v52 = *(_QWORD *)(v52 + 160);
    while ( *(_BYTE *)(v52 + 140) == 12 );
    goto LABEL_163;
  }
LABEL_31:
  v27 = qword_4F063F0;
  *(_QWORD *)(a11 + 24) = qword_4F063F0;
  *(_QWORD *)(a11 + 40) = v27;
  sub_7B8B50(v17, v20, k, v23);
  return v25;
}
