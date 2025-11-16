// Function: sub_7B40D0
// Address: 0x7b40d0
//
__int64 __fastcall sub_7B40D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rax
  char v7; // r12
  char *v8; // r13
  int v9; // r9d
  int v10; // eax
  int v11; // edx
  unsigned __int8 *v12; // r14
  unsigned int v13; // r14d
  char *v14; // r12
  int v15; // edi
  int v16; // ecx
  int v17; // esi
  int v18; // eax
  char v19; // dl
  unsigned int v20; // ecx
  char *v21; // rsi
  int v22; // edx
  __int64 *v23; // r15
  unsigned __int64 v24; // r10
  __int64 *v25; // rcx
  _DWORD *v26; // r8
  __int64 *v27; // rax
  char *v28; // r13
  char v29; // r15
  char v30; // r14
  char v31; // r15
  int v32; // eax
  int v33; // r12d
  int v34; // eax
  unsigned __int8 *v35; // r14
  char *v36; // r12
  int v37; // r14d
  const char *v38; // rax
  unsigned __int8 v39; // r14
  _BYTE *v40; // r14
  int v41; // eax
  int v42; // eax
  unsigned __int64 v43; // rdi
  int v44; // eax
  unsigned int v45; // r14d
  unsigned __int64 v46; // r10
  unsigned int v47; // r8d
  unsigned __int8 *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // r9
  unsigned __int64 v51; // r10
  __int64 v52; // rcx
  __int64 v53; // rax
  unsigned __int64 v54; // r10
  unsigned __int8 v56; // r12
  char *v57; // rax
  _BYTE *v58; // rax
  int v59; // eax
  int v60; // r15d
  unsigned __int8 *v61; // r12
  int v62; // esi
  int v63; // ecx
  int v64; // edi
  int v65; // eax
  int v66; // ecx
  char v67; // cl
  char v68; // cl
  int v69; // esi
  int v70; // esi
  char v71; // al
  unsigned __int8 *v72; // r14
  unsigned __int8 v73; // r12
  unsigned __int8 *v74; // r12
  unsigned __int64 v75; // rdi
  unsigned __int8 v76; // r12
  char v77; // al
  _BYTE *v78; // r12
  int v79; // r14d
  const char *v80; // rax
  char v81; // al
  _QWORD *v82; // rax
  __int64 v83; // r8
  __int64 v84; // rax
  unsigned __int8 *v85; // rcx
  int v86; // edx
  __int64 v87; // rax
  __int64 v88; // r8
  _BYTE *v89; // rax
  _QWORD *v90; // rax
  unsigned __int8 v91; // r12
  unsigned __int64 v92; // rdi
  char *v93; // rax
  char *v94; // rax
  unsigned __int64 v95; // rdx
  char v96; // si
  char *v97; // rax
  char v98; // cl
  int v99; // eax
  char v100; // r12
  int v101; // [rsp+10h] [rbp-D0h]
  int v102; // [rsp+14h] [rbp-CCh]
  unsigned __int8 *v103; // [rsp+18h] [rbp-C8h]
  char *v104; // [rsp+20h] [rbp-C0h]
  int v105; // [rsp+28h] [rbp-B8h]
  unsigned int v106; // [rsp+2Ch] [rbp-B4h]
  unsigned int v107; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v108; // [rsp+38h] [rbp-A8h]
  char *s; // [rsp+40h] [rbp-A0h]
  int v110; // [rsp+50h] [rbp-90h]
  unsigned __int64 v111; // [rsp+50h] [rbp-90h]
  char v112; // [rsp+5Bh] [rbp-85h]
  int v113; // [rsp+5Ch] [rbp-84h]
  _DWORD *v114; // [rsp+60h] [rbp-80h]
  bool v115; // [rsp+60h] [rbp-80h]
  __int64 *v116; // [rsp+60h] [rbp-80h]
  __int64 *v117; // [rsp+68h] [rbp-78h]
  unsigned __int64 v118; // [rsp+68h] [rbp-78h]
  unsigned __int64 v119; // [rsp+68h] [rbp-78h]
  unsigned __int64 v120; // [rsp+68h] [rbp-78h]
  _DWORD *v121; // [rsp+68h] [rbp-78h]
  unsigned __int64 v122; // [rsp+68h] [rbp-78h]
  unsigned __int64 v123; // [rsp+68h] [rbp-78h]
  unsigned __int64 v124; // [rsp+68h] [rbp-78h]
  unsigned __int64 v125; // [rsp+68h] [rbp-78h]
  char v126; // [rsp+68h] [rbp-78h]
  unsigned __int64 v127; // [rsp+68h] [rbp-78h]
  _BOOL4 i; // [rsp+70h] [rbp-70h]
  unsigned __int64 v129; // [rsp+70h] [rbp-70h]
  unsigned __int8 v130; // [rsp+70h] [rbp-70h]
  unsigned __int64 v131; // [rsp+70h] [rbp-70h]
  int src; // [rsp+78h] [rbp-68h]
  char *srca; // [rsp+78h] [rbp-68h]
  unsigned int srcb; // [rsp+78h] [rbp-68h]
  char srcc; // [rsp+78h] [rbp-68h]
  char v136; // [rsp+8Fh] [rbp-51h] BYREF
  unsigned int v137; // [rsp+90h] [rbp-50h] BYREF
  int v138; // [rsp+94h] [rbp-4Ch] BYREF
  unsigned __int64 v139; // [rsp+98h] [rbp-48h] BYREF
  _BYTE *v140; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v141; // [rsp+A8h] [rbp-38h] BYREF

  v6 = "0123456789";
  if ( unk_4D03D20 )
    v6 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
  s = v6;
  v110 = unk_4F07718;
  if ( unk_4F07718 )
  {
    v110 = qword_4F077B4;
    if ( (_DWORD)qword_4F077B4 )
    {
      v110 = 1;
      v7 = 1;
      goto LABEL_7;
    }
    if ( HIDWORD(qword_4F077B4) )
    {
      if ( qword_4F077A8 > 0x138E3u )
      {
        v140 = 0;
        v136 = 8;
        if ( unk_4D04190 )
          goto LABEL_346;
        if ( dword_4F077C0 )
        {
          if ( qword_4F077A8 <= 0x1116Fu )
          {
            v110 = 1;
            v7 = 1;
            goto LABEL_76;
          }
LABEL_346:
          v110 = 1;
          v7 = 1;
          v112 = 1;
          goto LABEL_78;
        }
        if ( dword_4F077BC )
        {
          v110 = 1;
          v7 = 1;
LABEL_113:
          if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x1FBCFu )
            goto LABEL_115;
LABEL_76:
          v112 = 0;
          if ( !HIDWORD(qword_4F077B4) )
            goto LABEL_9;
          goto LABEL_77;
        }
        v110 = 1;
        v7 = 1;
        v112 = 0;
LABEL_77:
        if ( (_DWORD)qword_4F077B4 )
          goto LABEL_9;
LABEL_78:
        v101 = 1;
        if ( qword_4F077A8 > 0x1FBCFu )
          goto LABEL_10;
        goto LABEL_9;
      }
      v140 = 0;
      v136 = 8;
      if ( unk_4D04190 )
      {
        v112 = 1;
        v7 = 0;
        goto LABEL_77;
      }
    }
    else
    {
      v140 = 0;
      v136 = 8;
      if ( unk_4D04190 )
      {
        v112 = 1;
        v7 = 0;
        goto LABEL_9;
      }
    }
    v7 = 0;
    goto LABEL_73;
  }
  v7 = 0;
LABEL_7:
  v140 = 0;
  v136 = 8;
  v112 = 1;
  if ( unk_4D04190 )
    goto LABEL_8;
LABEL_73:
  if ( !dword_4F077C0 )
  {
    v112 = 0;
    if ( !dword_4F077BC )
      goto LABEL_8;
    goto LABEL_113;
  }
  v112 = 0;
  if ( !(_DWORD)qword_4F077B4 )
  {
    if ( qword_4F077A8 <= 0x1116Fu )
      goto LABEL_76;
LABEL_115:
    v112 = 1;
    if ( !HIDWORD(qword_4F077B4) )
      goto LABEL_9;
    goto LABEL_78;
  }
LABEL_8:
  if ( HIDWORD(qword_4F077B4) )
    goto LABEL_77;
LABEL_9:
  v101 = 0;
LABEL_10:
  v8 = qword_4F06460;
  if ( !dword_4F17FA0
    && ((unsigned __int64)qword_4F06460 < qword_4F06498
     || (unsigned __int64)qword_4F06460 >= unk_4F06490
     || unk_4F06458
     || dword_4F17F78) )
  {
    if ( (_DWORD)qword_4F061D0 )
    {
      v141 = qword_4F061D0;
    }
    else
    {
      sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)&v141);
      v8 = qword_4F06460;
    }
  }
  else
  {
    v9 = (_DWORD)qword_4F06460 - qword_4F06498;
    v10 = *(_DWORD *)&word_4F06480;
    LODWORD(v141) = unk_4F0647C;
    if ( *(_DWORD *)&word_4F06480 && (unsigned __int64)qword_4F06460 < qword_4F06488[*(int *)&word_4F06480 - 1] )
      v10 = sub_7AB680((unsigned __int64)qword_4F06460);
    a6 = (unsigned int)(v9 + 1 - v10);
    WORD2(v141) = a6;
  }
  v102 = unk_4D04280;
  unk_4F061E0 = 0;
  if ( *v8 != 48 )
  {
    v103 = 0;
    v107 = 0;
    if ( *v8 == 46 )
      goto LABEL_90;
    while ( 1 )
    {
      v12 = (unsigned __int8 *)v8;
      if ( unk_4D03CE4 || v8[1] != 39 )
        goto LABEL_22;
      v12 = (unsigned __int8 *)(v8 + 1);
      if ( dword_4F0770C )
      {
        if ( !strchr(s, v8[2]) )
        {
          a6 = unk_4D03D20;
          if ( !unk_4D03D20 )
          {
            unk_4F061E0 = 1;
            if ( v103 != v12 )
            {
              sub_7B0EB0((unsigned __int64)v12, (__int64)dword_4F07508);
              sub_684AC0(8u, 0xA45u);
              v107 = 1;
              v103 = qword_4F06460 + 1;
              v12 = qword_4F06460 + 1;
            }
            goto LABEL_22;
          }
LABEL_89:
          v12 = (unsigned __int8 *)v8;
          goto LABEL_22;
        }
        unk_4F061E0 = 1;
      }
      else
      {
        if ( dword_4F077C4 != 2 && unk_4F07778 <= 202310 )
          goto LABEL_89;
        sub_7B0EB0((unsigned __int64)v12, (__int64)dword_4F07508);
        sub_684AC0(5u, 0xA44u);
        v12 = qword_4F06460;
      }
LABEL_22:
      v8 = (char *)(v12 + 1);
      qword_4F06460 = v12 + 1;
      v11 = v12[1];
      if ( (unsigned int)(v11 - 48) > 9 )
      {
        if ( (_BYTE)v11 != 46 )
        {
          if ( (v11 & 0xDF) != 0x45 )
          {
            src = unk_4D041A4;
            if ( !unk_4D041A4 )
              goto LABEL_36;
            if ( (unsigned __int8)((v11 & 0xDF) - 73) > 1u || (v13 = dword_4F05DC0[(char)v12[2] + 128]) != 0 )
            {
              src = 0;
              v113 = 0;
            }
            else
            {
              if ( !v7 )
              {
                v23 = &qword_4F06408;
LABEL_386:
                if ( !unk_4D03D20 )
                {
                  sub_7B0EB0((unsigned __int64)v8, (__int64)dword_4F07508);
                  sub_684AC0(7u, 0x5A1u);
                  v8 = qword_4F06460;
                }
                *v23 = (__int64)v8;
                v83 = (unsigned __int8)*v8;
                *v8 = 105;
                srcc = v83;
                sub_7CC8A0(v13, &v137, &qword_4F06410, &v136, v83, a6);
                *(_BYTE *)*v23 = srcc;
                if ( v140 )
                {
                  v84 = (__int64)&v140[*v23 - 1];
                  *v23 = v84;
                }
                else
                {
                  v84 = *v23;
                }
                v45 = 2;
                qword_4F06460 = (_BYTE *)(v84 + 1);
                return v45;
              }
              src = 1;
LABEL_36:
              v113 = 0;
            }
LABEL_37:
            v14 = v8;
            v15 = 0;
            v16 = 0;
            v17 = 0;
            a6 = 1;
            v18 = 0;
            while ( 1 )
            {
              v19 = v11 & 0xDF;
              if ( !v17 && v19 == 85 )
              {
                v17 = 1;
                if ( v18 == 1 )
                  v15 = 1;
              }
              else if ( v19 == 76 )
              {
                if ( v15 | v16 || v18 > 1 )
                  goto LABEL_43;
                ++v18;
                v15 = 0;
                v16 = 0;
              }
              else
              {
                if ( !unk_4D04184 || v19 != 90 || v18 | v16 )
                {
LABEL_43:
                  v20 = dword_4D04964;
                  if ( dword_4D04964 && !(unk_4D03D20 | unk_4D04298) && v18 == 2 )
                  {
                    v91 = unk_4F07471;
                    sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
                    sub_684AC0(v91, 0x1C2u);
                    v14 = qword_4F06460;
                    v20 = dword_4D04964;
                  }
                  v21 = v14 - 1;
                  v22 = unk_4F07718;
                  v105 = src;
                  v106 = 0;
                  v23 = &qword_4F06408;
                  if ( v8 == v14 )
                    v8 = 0;
                  v24 = 0;
                  i = 0;
                  goto LABEL_50;
                }
                v18 = 0;
                v16 = 1;
              }
              qword_4F06460 = ++v14;
              LOBYTE(v11) = *v14;
            }
          }
          goto LABEL_258;
        }
LABEL_90:
        v33 = 0;
        while ( 2 )
        {
          v35 = (unsigned __int8 *)v8;
          if ( !unk_4D03CE4 && v8[1] == 39 )
          {
            v35 = (unsigned __int8 *)(v8 + 1);
            if ( dword_4F0770C )
            {
              if ( strchr(s, v8[2]) )
              {
                unk_4F061E0 = 1;
                if ( v33 == 1 )
                  goto LABEL_91;
              }
              else
              {
                if ( unk_4D03D20 )
                  goto LABEL_97;
                unk_4F061E0 = 1;
              }
              if ( v103 != v35 )
              {
                sub_7B0EB0((unsigned __int64)v35, (__int64)dword_4F07508);
                sub_684AC0(8u, 0xA45u);
                v107 = 1;
                v103 = qword_4F06460 + 1;
                v35 = qword_4F06460 + 1;
              }
            }
            else if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
            {
              sub_7B0EB0((unsigned __int64)v35, (__int64)dword_4F07508);
              sub_684AC0(5u, 0xA44u);
              v35 = qword_4F06460;
            }
            else
            {
LABEL_97:
              v35 = (unsigned __int8 *)v8;
            }
          }
LABEL_91:
          v8 = (char *)(v35 + 1);
          v33 = 1;
          qword_4F06460 = v35 + 1;
          v34 = v35[1];
          if ( (unsigned int)(v34 - 48) > 9 )
          {
            if ( (_BYTE)v34 == 46 )
            {
              a6 = unk_4D03D20;
              if ( !unk_4D03D20 && (_DWORD)qword_4F077B4 )
              {
                do
                {
                  do
                  {
                    v85 = (unsigned __int8 *)v8++;
                    qword_4F06460 = v8;
                    v86 = (unsigned __int8)*v8;
                  }
                  while ( (unsigned int)(v86 - 48) <= 9 );
                }
                while ( (_BYTE)v86 == 46 );
                v45 = 283;
                qword_4F06408 = v85;
                sub_7CE2C0(
                  (_DWORD)qword_4F06410,
                  (_DWORD)v8,
                  17,
                  (_DWORD)v85 - (_DWORD)qword_4F06410 + 1,
                  (unsigned int)&v137,
                  (unsigned int)&v139,
                  0);
                return v45;
              }
            }
            else if ( (v34 & 0xDF) == 0x45 )
            {
              i = 1;
              v113 = 0;
              goto LABEL_191;
            }
            v40 = v35 + 1;
            v24 = 0;
            i = 0;
            v105 = 0;
            v8 = 0;
            v22 = unk_4F07718;
            v106 = 0;
            v23 = &qword_4F06408;
            goto LABEL_208;
          }
          continue;
        }
      }
    }
  }
  if ( (v8[1] & 0xDF) == 0x58 )
  {
    v36 = v8 + 1;
    qword_4F06460 = v8 + 1;
    v37 = unk_4D03D20;
    v38 = "0123456789ABCDEFabcdef";
    if ( unk_4D03D20 )
      v38 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
    s = (char *)v38;
    v107 = unk_4D03CE4;
    if ( unk_4D03CE4 )
    {
      v103 = 0;
      v107 = 0;
      goto LABEL_104;
    }
    if ( v8[2] == 39 )
    {
      if ( !dword_4F0770C )
      {
        if ( dword_4F077C4 == 2 || (v103 = 0, unk_4F07778 > 202310) )
        {
          sub_7B0EB0((unsigned __int64)(v8 + 2), (__int64)dword_4F07508);
          sub_684AC0(5u, 0xA44u);
          v36 = qword_4F06460;
          v107 = 0;
          v103 = 0;
        }
LABEL_104:
        for ( i = 0; ; i = 1 )
        {
          v8 = v36 + 1;
          qword_4F06460 = v36 + 1;
          v39 = v36[1];
          if ( !isxdigit(v39) )
            break;
          if ( !unk_4D03CE4 && v36[2] == 39 )
          {
            if ( dword_4F0770C )
            {
              if ( strchr(s, v36[3]) )
              {
                v8 = v36 + 2;
                unk_4F061E0 = 1;
              }
              else if ( !unk_4D03D20 )
              {
                unk_4F061E0 = 1;
                v8 = (char *)v103;
                if ( v36 + 2 != (char *)v103 )
                {
                  sub_7B0EB0((unsigned __int64)(v36 + 2), (__int64)dword_4F07508);
                  sub_684AC0(8u, 0xA45u);
                  v107 = 1;
                  v103 = qword_4F06460 + 1;
                  v8 = qword_4F06460 + 1;
                }
              }
            }
            else if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
            {
              sub_7B0EB0((unsigned __int64)(v36 + 2), (__int64)dword_4F07508);
              sub_684AC0(5u, 0xA44u);
              v8 = qword_4F06460;
            }
          }
          v36 = v8;
        }
        src = 0;
        LOBYTE(v11) = v39;
        if ( !(unk_4D04778 | v102) )
          goto LABEL_284;
        if ( v39 != 46 )
        {
          if ( (v39 & 0xDF) == 0x50 )
          {
            if ( i )
            {
              i = 0;
              v113 = 2;
            }
            else
            {
              v113 = 2;
              if ( !unk_4D03D20 )
              {
                sub_7B0EB0((unsigned __int64)(v36 + 1), (__int64)dword_4F07508);
                sub_684AC0(8u, 0xA6u);
                i = 0;
                v8 = qword_4F06460;
              }
            }
            goto LABEL_191;
          }
LABEL_284:
          v113 = 2;
          if ( v8 == qword_4F06410 + 2 && !unk_4D03D20 )
          {
            if ( dword_4F077C4 == 1 )
            {
              sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
              sub_684AC0(5u, 0x16u);
              v8 = qword_4F06460;
              src = 0;
            }
            else
            {
              sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
              sub_684AC0(8u, 0x16u);
              v8 = qword_4F06460;
              src = 0;
              v107 = 1;
            }
            LOBYTE(v11) = *qword_4F06460;
          }
          goto LABEL_37;
        }
        if ( !unk_4D03CE4 && v36[2] == 39 )
        {
          if ( dword_4F0770C )
          {
            if ( strchr(s, v36[3]) || !unk_4D03D20 )
            {
              v8 = (char *)v103;
              unk_4F061E0 = 1;
              if ( v36 + 2 != (char *)v103 )
              {
                sub_7B0EB0((unsigned __int64)(v36 + 2), (__int64)dword_4F07508);
                sub_684AC0(8u, 0xA45u);
                v107 = 1;
                v103 = qword_4F06460 + 1;
                v8 = qword_4F06460 + 1;
              }
            }
          }
          else if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
          {
            sub_7B0EB0((unsigned __int64)(v36 + 2), (__int64)dword_4F07508);
            sub_684AC0(5u, 0xA44u);
            v8 = qword_4F06460;
          }
        }
        while ( 1 )
        {
          v72 = (unsigned __int8 *)(v8 + 1);
          qword_4F06460 = v8 + 1;
          v73 = v8[1];
          if ( !isxdigit(v73) )
            break;
          if ( !unk_4D03CE4 && v8[2] == 39 )
          {
            if ( dword_4F0770C )
            {
              if ( strchr(s, v8[3]) )
              {
                v72 = (unsigned __int8 *)(v8 + 2);
                unk_4F061E0 = 1;
              }
              else if ( !unk_4D03D20 )
              {
                unk_4F061E0 = 1;
                v72 = v103;
                if ( v8 + 2 != (char *)v103 )
                {
                  sub_7B0EB0((unsigned __int64)(v8 + 2), (__int64)dword_4F07508);
                  sub_684AC0(8u, 0xA45u);
                  v107 = 1;
                  v103 = qword_4F06460 + 1;
                  v72 = qword_4F06460 + 1;
                }
              }
            }
            else if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
            {
              sub_7B0EB0((unsigned __int64)(v8 + 2), (__int64)dword_4F07508);
              sub_684AC0(5u, 0xA44u);
              v72 = qword_4F06460;
            }
          }
          i = 1;
          v8 = (char *)v72;
        }
        if ( !i )
        {
          if ( unk_4D03D20 )
          {
            if ( (v73 & 0xDF) == 0x50 )
            {
LABEL_311:
              i = 1;
              v8 = (char *)v72;
              v113 = 2;
              goto LABEL_191;
            }
LABEL_322:
            v8 = (char *)(v72 - 1);
            i = 1;
            qword_4F06460 = v72 - 1;
            v113 = 2;
            goto LABEL_191;
          }
          sub_7B0EB0((unsigned __int64)(v8 + 1), (__int64)dword_4F07508);
          sub_684AC0(8u, 0xA6u);
          v72 = qword_4F06460;
          v73 = *qword_4F06460;
        }
        if ( (v73 & 0xDF) == 0x50 )
          goto LABEL_311;
        v106 = unk_4D03D20;
        if ( !unk_4D03D20 )
        {
          v8 = 0;
          sub_7B0EB0((unsigned __int64)v72, (__int64)dword_4F07508);
          sub_684AC0(8u, 0xA6u);
          v40 = qword_4F06460;
          v24 = 0;
          v105 = 0;
          LOBYTE(v34) = *qword_4F06460;
          v22 = unk_4F07718;
          i = 1;
          v107 = 1;
          v23 = &qword_4F06408;
          goto LABEL_208;
        }
        goto LABEL_322;
      }
      if ( strchr(v38, v8[3]) || !v37 )
      {
        unk_4F061E0 = 1;
        sub_7B0EB0((unsigned __int64)(v8 + 2), (__int64)dword_4F07508);
        sub_684AC0(8u, 0xA45u);
        v107 = 1;
        v103 = qword_4F06460 + 1;
        v36 = qword_4F06460 + 1;
        goto LABEL_104;
      }
    }
    v103 = 0;
    goto LABEL_104;
  }
  if ( unk_4D0427C && (v8[1] & 0xDF) == 0x42 )
  {
    v78 = v8 + 1;
    qword_4F06460 = v8 + 1;
    v79 = unk_4D03D20;
    v80 = "01";
    if ( unk_4D03D20 )
      v80 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
    s = (char *)v80;
    v107 = unk_4D03CE4;
    if ( unk_4D03CE4 )
    {
      v103 = 0;
      v107 = 0;
    }
    else
    {
      if ( v8[2] != 39 )
      {
LABEL_355:
        v103 = 0;
        goto LABEL_356;
      }
      if ( dword_4F0770C )
      {
        v93 = strchr(v80, v8[3]);
        if ( v79 && !v93 )
          goto LABEL_355;
        unk_4F061E0 = 1;
        sub_7B0EB0((unsigned __int64)(v8 + 2), (__int64)dword_4F07508);
        sub_684AC0(8u, 0xA45u);
        v107 = 1;
        v78 = qword_4F06460 + 1;
        v103 = qword_4F06460 + 1;
      }
      else if ( dword_4F077C4 == 2 || (v103 = 0, unk_4F07778 > 202310) )
      {
        sub_7B0EB0((unsigned __int64)(v8 + 2), (__int64)dword_4F07508);
        sub_684AC0(5u, 0xA44u);
        v78 = qword_4F06460;
        v107 = 0;
        v103 = 0;
      }
    }
LABEL_356:
    v8 = v78 + 1;
    qword_4F06460 = v78 + 1;
    v11 = (unsigned __int8)v78[1];
    if ( (unsigned int)(v11 - 48) > 9 )
    {
      src = unk_4D03D20;
      if ( !unk_4D03D20 )
      {
        sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
        sub_684AC0(8u, 0x978u);
        v8 = qword_4F06460;
        v107 = 1;
        v113 = 3;
        LOBYTE(v11) = *qword_4F06460;
        goto LABEL_37;
      }
LABEL_407:
      src = 0;
      v113 = 3;
      goto LABEL_37;
    }
    while ( 1 )
    {
      if ( !unk_4D03CE4 && v78[2] == 39 )
      {
        if ( dword_4F0770C )
        {
          if ( strchr(s, (char)v78[3]) )
          {
            unk_4F061E0 = 1;
LABEL_365:
            v78 = qword_4F06460 + 1;
            goto LABEL_359;
          }
          if ( !unk_4D03D20 )
          {
            unk_4F061E0 = 1;
            if ( v78 + 2 != v103 )
            {
              sub_7B0EB0((unsigned __int64)(v78 + 2), (__int64)dword_4F07508);
              sub_684AC0(8u, 0xA45u);
              v107 = 1;
              v103 = qword_4F06460 + 1;
              v78 = qword_4F06460 + 1;
              goto LABEL_359;
            }
            goto LABEL_365;
          }
        }
        else if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
        {
          sub_7B0EB0((unsigned __int64)(v78 + 2), (__int64)dword_4F07508);
          sub_684AC0(5u, 0xA44u);
          v78 = qword_4F06460;
          goto LABEL_359;
        }
      }
      v78 = qword_4F06460;
LABEL_359:
      v8 = v78 + 1;
      qword_4F06460 = v78 + 1;
      v11 = (unsigned __int8)v78[1];
      if ( (unsigned int)(v11 - 48) > 9 )
        goto LABEL_407;
    }
  }
  v107 = unk_4D041A4;
  if ( unk_4D041A4 )
  {
    if ( (unsigned __int8)((v8[1] & 0xDF) - 73) <= 1u )
    {
      v40 = v8 + 1;
      qword_4F06460 = v8 + 1;
      if ( !v7 )
      {
        v106 = unk_4D03D20;
        if ( !unk_4D03D20 )
        {
          v92 = (unsigned __int64)(v8 + 1);
          v8 = 0;
          sub_7B0EB0(v92, (__int64)dword_4F07508);
          sub_684AC0(7u, 0x5A1u);
          v40 = qword_4F06460;
          i = 0;
          v24 = 0;
          v105 = 0;
          v22 = unk_4F07718;
          LOBYTE(v34) = *qword_4F06460;
          v110 = 0;
          v107 = 0;
          v23 = &qword_4F06408;
          v103 = 0;
          goto LABEL_208;
        }
        LOBYTE(v34) = v8[1];
        v22 = unk_4F07718;
        v62 = unk_4D041A4;
        if ( (unsigned __int8)((v34 & 0xDF) - 73) > 1u )
        {
          v24 = 0;
          i = 0;
          v8 = 0;
          v110 = 0;
          v23 = &qword_4F06408;
          v103 = 0;
          v107 = 0;
          v105 = 0;
          src = 0;
          v106 = 0;
          goto LABEL_216;
        }
        v64 = 0;
        v24 = 0;
        v8 = 0;
        i = 0;
        v23 = &qword_4F06408;
        v103 = 0;
        v107 = 0;
        v105 = 0;
        v110 = 0;
        v106 = 1;
        goto LABEL_215;
      }
      LOBYTE(v11) = v8[1];
      v103 = 0;
      ++v8;
      src = 1;
      v107 = 0;
      v113 = 0;
      goto LABEL_37;
    }
    v103 = 0;
    v107 = 0;
  }
  else
  {
    v103 = 0;
  }
  do
  {
    v74 = (unsigned __int8 *)v8;
    if ( !unk_4D03CE4 && v8[1] == 39 )
    {
      v74 = (unsigned __int8 *)(v8 + 1);
      if ( dword_4F0770C )
      {
        if ( strchr(s, v8[2]) )
        {
          unk_4F061E0 = 1;
          goto LABEL_327;
        }
        if ( unk_4D03D20 )
        {
LABEL_335:
          v74 = (unsigned __int8 *)v8;
          goto LABEL_327;
        }
        unk_4F061E0 = 1;
        if ( v103 != v74 )
        {
          sub_7B0EB0((unsigned __int64)v74, (__int64)dword_4F07508);
          sub_684AC0(8u, 0xA45u);
          v107 = 1;
          v103 = qword_4F06460 + 1;
          v74 = qword_4F06460 + 1;
        }
      }
      else
      {
        if ( dword_4F077C4 != 2 && unk_4F07778 <= 202310 )
          goto LABEL_335;
        sub_7B0EB0((unsigned __int64)v74, (__int64)dword_4F07508);
        sub_684AC0(5u, 0xA44u);
        v74 = qword_4F06460;
      }
    }
LABEL_327:
    v8 = (char *)(v74 + 1);
    qword_4F06460 = v74 + 1;
    v11 = v74[1];
  }
  while ( (unsigned int)(v11 - 48) <= 9 );
  if ( (_BYTE)v11 == 46 )
    goto LABEL_90;
  if ( (v11 & 0xDF) != 0x45 )
  {
    src = 0;
    v113 = 1;
    goto LABEL_37;
  }
LABEL_258:
  i = 0;
  v113 = 0;
LABEL_191:
  v59 = (unsigned __int8)v8[1];
  v40 = v8;
  if ( (((_BYTE)v59 - 43) & 0xFD) == 0 )
  {
    v40 = v8 + 1;
    qword_4F06460 = v8 + 1;
    v59 = (unsigned __int8)v8[2];
  }
  if ( unk_4D03CE4 || (_BYTE)v59 != 39 )
  {
LABEL_195:
    if ( (unsigned int)(v59 - 48) > 9 )
      goto LABEL_196;
    v8 = 0;
LABEL_198:
    v60 = 0;
    while ( 2 )
    {
      v61 = v40;
      if ( unk_4D03CE4 || v40[1] != 39 )
        goto LABEL_199;
      v61 = v40 + 1;
      if ( dword_4F0770C )
      {
        if ( strchr(s, (char)v40[2]) )
        {
          unk_4F061E0 = 1;
          if ( v60 != 1 )
          {
LABEL_205:
            if ( v103 != v61 )
            {
              sub_7B0EB0((unsigned __int64)v61, (__int64)dword_4F07508);
              sub_684AC0(8u, 0xA45u);
              v107 = 1;
              v103 = qword_4F06460 + 1;
              v61 = qword_4F06460 + 1;
            }
          }
LABEL_199:
          v40 = v61 + 1;
          v60 = 1;
          qword_4F06460 = v61 + 1;
          v34 = v61[1];
          if ( (unsigned int)(v34 - 48) > 9 )
          {
            v105 = 0;
            v24 = 0;
            v106 = 0;
            i = v113 == 2;
            v22 = unk_4F07718;
            v23 = &qword_4F06408;
            goto LABEL_208;
          }
          continue;
        }
        if ( !unk_4D03D20 )
        {
          unk_4F061E0 = 1;
          goto LABEL_205;
        }
      }
      else if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
      {
        sub_7B0EB0((unsigned __int64)v61, (__int64)dword_4F07508);
        sub_684AC0(5u, 0xA44u);
        v61 = qword_4F06460;
        goto LABEL_199;
      }
      break;
    }
    v61 = v40;
    goto LABEL_199;
  }
  if ( dword_4F0770C )
  {
    if ( strchr(s, (char)v40[2]) || !unk_4D03D20 )
    {
      unk_4F061E0 = 1;
      if ( v40 + 1 != v103 )
      {
        sub_7B0EB0((unsigned __int64)(v40 + 1), (__int64)dword_4F07508);
        sub_684AC0(8u, 0xA45u);
        v40 = qword_4F06460;
        v107 = 1;
        v103 = qword_4F06460 + 1;
      }
      qword_4F06460 = v103;
      v59 = (unsigned __int8)v40[2];
      v40 = v103;
      goto LABEL_195;
    }
LABEL_197:
    v40 = qword_4F06460;
    v8 = 0;
    goto LABEL_198;
  }
  if ( dword_4F077C4 == 2 || unk_4F07778 > 202310 )
  {
    sub_7B0EB0((unsigned __int64)(v40 + 1), (__int64)dword_4F07508);
    sub_684AC0(5u, 0xA44u);
    v40 = qword_4F06460;
    v59 = (unsigned __int8)qword_4F06460[1];
    goto LABEL_195;
  }
LABEL_196:
  v106 = unk_4D03D20;
  if ( unk_4D03D20 )
    goto LABEL_197;
  v22 = unk_4F07718;
  if ( !unk_4F07718 )
  {
    v75 = (unsigned __int64)(v40 + 1);
    if ( dword_4F077C4 == 1 )
    {
      sub_7B0EB0(v75, (__int64)dword_4F07508);
      sub_684AC0(5u, 0xA6u);
      v40 = qword_4F06460;
    }
    else
    {
      sub_7B0EB0(v75, (__int64)dword_4F07508);
      sub_684AC0(8u, 0xA6u);
      v40 = qword_4F06460;
      v107 = 1;
    }
    goto LABEL_198;
  }
  if ( i )
  {
    qword_4F06460 = v8;
    v40 = v8;
    v105 = 0;
    v24 = 0;
    v23 = &qword_4F06408;
    i = v113 == 2;
    LOBYTE(v34) = *v8;
    goto LABEL_208;
  }
  v14 = qword_4F06460;
  v24 = 0;
  v105 = 0;
  src = 0;
  v23 = &qword_4F06408;
  v20 = dword_4D04964;
  v21 = qword_4F06460 - 1;
LABEL_50:
  while ( 2 )
  {
    *v23 = (__int64)v21;
    v25 = (__int64 *)(HIDWORD(qword_4F077B4) | v22 | v20);
    if ( !(_DWORD)v25 )
    {
      v26 = (_DWORD *)unk_4D03D20;
      if ( unk_4D03D20 )
      {
        v25 = (__int64 *)&dword_4D0493C;
        if ( dword_4D0493C )
        {
          v25 = (__int64 *)dword_4D03D18;
          if ( !dword_4D03D18 )
          {
            v25 = &qword_4D03BD8;
            if ( !qword_4D03BD8 )
            {
              if ( dword_4F077C4 == 1 )
              {
LABEL_268:
                while ( dword_4F05DC0[*v14 + 128] )
                {
                  v21 = v14++;
                  qword_4F06460 = v14;
                }
              }
              else
              {
LABEL_184:
                v21 = v14 - 1;
              }
              *v23 = (__int64)v21;
              return 12;
            }
          }
        }
        goto LABEL_53;
      }
      if ( dword_4F077C4 != 1 )
        goto LABEL_146;
LABEL_152:
      if ( v22 )
        goto LABEL_177;
LABEL_153:
      v44 = v113;
      if ( v113 == 3 )
        goto LABEL_180;
      goto LABEL_154;
    }
LABEL_53:
    if ( dword_4F077C4 == 1 )
    {
      v26 = (_DWORD *)unk_4D03D20;
      if ( unk_4D03D20 )
        goto LABEL_268;
      goto LABEL_152;
    }
    v26 = &dword_4F0770C;
    v104 = v8;
    a6 = (__int64)dword_4F055C0;
    v27 = v23;
    v28 = (char *)v24;
    v29 = *(v14 - 1);
    v25 = v27;
LABEL_55:
    v30 = *v14;
    if ( *v26 )
      goto LABEL_56;
LABEL_58:
    if ( !dword_4F055C0[v30 + 128] )
      goto LABEL_138;
    while ( 1 )
    {
      if ( (unsigned int)(unsigned __int8)v30 - 48 > 9 && v30 != 46 )
      {
        if ( ((v30 - 43) & 0xFD) != 0 )
          break;
        v31 = v29 & 0xDF;
        if ( v31 != 69 && (!(unk_4D04778 | v102) || v31 != 80) )
          break;
      }
      v32 = *v26;
      v29 = v30;
      if ( !v28 )
        v28 = v14;
      qword_4F06460 = ++v14;
      v30 = *v14;
      if ( !v32 )
        goto LABEL_58;
LABEL_56:
      if ( unk_4D03CE4 || v30 != 39 )
        goto LABEL_58;
      v116 = v25;
      v121 = v26;
      v57 = strchr(s, v14[1]);
      a6 = (__int64)dword_4F055C0;
      v26 = v121;
      v25 = v116;
      if ( v57 )
      {
        unk_4F061E0 = 1;
        v58 = v14;
LABEL_174:
        v14 = v58 + 1;
        qword_4F06460 = v58 + 1;
        v30 = v58[1];
        goto LABEL_58;
      }
      if ( !unk_4D03D20 )
      {
        unk_4F061E0 = 1;
        v58 = v103;
        if ( v103 != (unsigned __int8 *)v14 )
        {
          sub_7B0EB0((unsigned __int64)v14, (__int64)dword_4F07508);
          sub_684AC0(8u, 0xA45u);
          v58 = qword_4F06460;
          v25 = v116;
          v107 = 1;
          v26 = v121;
          a6 = (__int64)dword_4F055C0;
          v103 = qword_4F06460;
        }
        goto LABEL_174;
      }
      a6 = (unsigned int)dword_4F055C0[167];
      if ( (_DWORD)a6 )
        break;
LABEL_138:
      v114 = v26;
      v117 = v25;
      v41 = sub_7B3CF0((unsigned __int8 *)v14, &v138, v14 == (char *)(*v25 + 1));
      v14 = qword_4F06460;
      v25 = v117;
      a6 = (__int64)dword_4F055C0;
      v26 = v114;
      if ( v41 )
      {
        v29 = v30;
        v42 = 1;
        if ( qword_4F06460 != (_BYTE *)(*v117 + 1) )
          v42 = src;
        src = v42;
        v14 = &qword_4F06460[v138];
        qword_4F06460 = v14;
        goto LABEL_55;
      }
    }
    v24 = (unsigned __int64)v28;
    v23 = v25;
    v8 = v104;
    if ( unk_4D03D20 )
      goto LABEL_184;
LABEL_146:
    v43 = *v23 + 1;
    v22 = unk_4F07718;
    if ( v14 == (char *)v43 )
      goto LABEL_152;
    v25 = (__int64 *)v107;
    if ( v107 )
      goto LABEL_152;
    if ( !unk_4F07718 || (src & 1) == 0 )
    {
      v123 = v24;
      sub_7B0EB0(v43, (__int64)dword_4F07508);
      sub_684AC0(8u, 0x13u);
      v24 = v123;
      v107 = 1;
      v22 = unk_4F07718;
      goto LABEL_152;
    }
    if ( v24 )
    {
      v118 = v24;
      sub_7B0EB0(v24, (__int64)dword_4F07508);
      sub_684AC0(8u, 0x9B2u);
      src = 1;
      v107 = 1;
      v24 = v118;
      v22 = unk_4F07718;
      goto LABEL_152;
    }
    src = 1;
LABEL_177:
    if ( !v8 || (src & 1) == 0 )
      goto LABEL_153;
    v44 = v113;
    *v23 = (__int64)(v8 - 1);
    if ( v113 == 3 )
    {
LABEL_180:
      v122 = v24;
      v45 = 4;
      sub_7CBA40(2, &v137, &v139, v25, v26, a6);
      v46 = v122;
      goto LABEL_159;
    }
LABEL_154:
    if ( v44 == 4 )
    {
      if ( !v102 && i )
      {
        v125 = v24;
        v76 = byte_4F07472[0];
        sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
        sub_684AC0(v76, 0x521u);
        v24 = v125;
      }
      v124 = v24;
      v45 = 2;
      sub_7CC8A0(i, &v137, &v139, &v136, v26, a6);
      v46 = v124;
    }
    else
    {
      v119 = v24;
      if ( v44 == 1 )
      {
        v45 = 4;
        sub_7CBA40(8, &v137, &v139, v25, v26, a6);
        v46 = v119;
      }
      else
      {
        v45 = 4;
        if ( v44 == 2 )
          sub_7CBA40(16, &v137, &v139, v25, v26, a6);
        else
          sub_7CBA40(10, &v137, &v139, v25, v26, a6);
        v46 = v119;
      }
    }
LABEL_159:
    v47 = v137;
    v120 = v46;
    if ( !unk_4F07718 || ((v107 ^ 1) & src) == 0 )
    {
      if ( v137 )
        goto LABEL_171;
      return v45;
    }
    if ( v137 > 0x1E )
      goto LABEL_171;
    v115 = ((0x40800001uLL >> v137) & 1) == 0;
    if ( ((0x40800001uLL >> v137) & 1) == 0 )
      goto LABEL_171;
    v48 = (unsigned __int8 *)(*v23 + 1);
    v140 = &qword_4F06460[~*v23];
    v49 = sub_7B3EE0(v48, &v140);
    v51 = v120;
    srca = (char *)v49;
    if ( v113 == 4 )
    {
      v87 = *v23;
      v108 = v120;
      ++*v23;
      v88 = *(unsigned __int8 *)(v87 + 1);
      *(_BYTE *)(v87 + 1) = 76;
      v126 = v88;
      sub_7CC8A0(i, &v137, &v139, &v136, v88, v50);
      v89 = (_BYTE *)*v23;
      v51 = v108;
      --*v23;
      *v89 = v126;
      if ( unk_4F063AD )
      {
LABEL_165:
        v52 = xmmword_4F06380[0].m128i_i64[0];
      }
      else
      {
        v90 = sub_72C610(6u);
        v51 = v108;
        v52 = (__int64)v90;
      }
    }
    else
    {
      if ( unk_4F063AD )
        goto LABEL_165;
      v82 = sub_72BA30(0xAu);
      v51 = v120;
      v52 = (__int64)v82;
    }
    v129 = v51;
    unk_4F06210 = v52;
    v53 = sub_881010(srca);
    v54 = v129;
    v8 = (char *)v53;
    qword_4F06218 = (_QWORD *)v53;
    if ( v53 )
    {
      v47 = v137;
      if ( !v137 )
        goto LABEL_168;
      if ( unk_4F04D80 )
        goto LABEL_168;
      v77 = *(_BYTE *)(v53 + 80);
      if ( v77 == 20 || v77 == 11 && (*(_BYTE *)(*((_QWORD *)v8 + 11) + 207LL) & 1) != 0 )
        goto LABEL_168;
LABEL_171:
      srcb = v47;
      v56 = v136;
      sub_7B0EB0(v139, (__int64)dword_4F07508);
      sub_684AC0(v56, srcb);
      return v45;
    }
    if ( (unsigned __int64)v140 > 4 )
      break;
    v130 = v110 & (unk_4D041A4 != 0);
    if ( !v130 )
      break;
    v127 = (unsigned __int64)v140;
    v111 = v54;
    v94 = strchr("IiJj", *srca);
    v95 = v127;
    v24 = v111;
    if ( v94 )
    {
      if ( v127 > 1 )
      {
        ++srca;
        goto LABEL_450;
      }
      if ( v113 != 4 )
        goto LABEL_467;
      goto LABEL_461;
    }
    if ( v127 <= 1 )
      break;
    v97 = strchr("IiJj", srca[v127 - 1]);
    v95 = v127;
    v24 = v111;
    if ( !v97 )
      break;
LABEL_450:
    if ( v113 != 4 )
    {
      v96 = *srca;
      switch ( *srca )
      {
        case 'L':
        case 'l':
          v98 = srca[1];
          if ( v95 == 3 )
          {
            v99 = v96 == v98 || (v98 & 0xDF) == 85;
          }
          else
          {
            if ( v96 == v98 )
              v115 = (srca[2] & 0xDF) == 85;
            v99 = v115;
          }
          goto LABEL_472;
        case 'U':
        case 'u':
          if ( v95 == 2 )
            break;
          if ( (srca[1] & 0xDF) == 0x5A )
          {
            if ( v95 != 3 )
              goto LABEL_398;
          }
          else
          {
            if ( (srca[1] & 0xDF) != 0x4C )
              goto LABEL_398;
            if ( v95 != 3 )
              v130 = srca[2] == srca[1];
            v99 = v130;
LABEL_472:
            if ( !v99 )
              goto LABEL_398;
          }
          break;
        case 'Z':
        case 'z':
          if ( v95 != 4 && (srca[1] & 0xDF) == 0x55 )
            break;
          goto LABEL_398;
        default:
          goto LABEL_398;
      }
LABEL_467:
      v13 = v113 == 2;
      v8 = (char *)(*v23 + 1);
      qword_4F06460 = v8;
      goto LABEL_386;
    }
    if ( v95 == 2 )
    {
      v131 = v24;
      v100 = *srca;
      if ( strchr("FfLlWwQq", *srca) )
      {
        v24 = v131;
        if ( (dword_4D04288 || (v100 & 0xDF) != 0x57) && (dword_4D04284 || (v100 & 0xDF) != 0x51) )
        {
LABEL_461:
          i = 0;
          v110 = 0;
          v107 = 0;
          v40 = (_BYTE *)(*v23 + 1);
          v22 = unk_4F07718;
          qword_4F06460 = v40;
          LOBYTE(v34) = *v40;
LABEL_208:
          v62 = unk_4D041A4;
          src = 0;
          if ( unk_4D041A4 && (unsigned __int8)((v34 & 0xDF) - 73) <= 1u )
          {
            v63 = v105;
            v64 = v110;
            if ( v110 )
              v63 = v110;
            v65 = 1;
            if ( v110 )
            {
              v65 = v106;
              v8 = v40;
            }
            v105 = v63;
            v106 = v65;
LABEL_215:
            qword_4F06460 = v40 + 1;
            LOBYTE(v34) = *++v40;
            v66 = v110;
            v110 = v64;
            src = v66;
          }
LABEL_216:
          v67 = v34 & 0xDF;
          if ( (v34 & 0xDF) == 0x4C
            || v67 == 70
            || (a6 = dword_4D04288) != 0 && v67 == 87
            || dword_4D04284 && v67 == 81
            || (v14 = v40, unk_4D04190 | v101) && v67 == 66 )
          {
            v14 = v40 + 1;
            if ( !v8 )
              v8 = v40;
            qword_4F06460 = v40 + 1;
            if ( v112 || unk_4D0428C )
            {
              if ( unk_4D04190 | v101 && v67 == 66 )
              {
                v68 = v40[1];
                if ( (_BYTE)v34 == 98 )
                {
                  if ( v68 == 102 )
                    goto LABEL_375;
LABEL_228:
                  qword_4F06460 = v40;
                  v14 = v40;
                  src = 1;
                  goto LABEL_229;
                }
                if ( v68 != 70 )
                  goto LABEL_228;
LABEL_375:
                if ( v40[2] != 49 || v40[3] != 54 )
                  goto LABEL_228;
LABEL_377:
                v14 = v40 + 4;
                qword_4F06460 = v40 + 4;
              }
              else
              {
                if ( v67 != 70 )
                  goto LABEL_229;
                v71 = v40[1];
                if ( v71 == 49 )
                {
                  v81 = v40[2];
                  if ( v81 == 54 )
                  {
                    if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x1FBCFu )
                      goto LABEL_228;
                    goto LABEL_253;
                  }
                  if ( v112 && v81 == 50 && v40[3] == 56 )
                    goto LABEL_377;
                }
                else
                {
                  if ( !v112 )
                    goto LABEL_229;
                  if ( v71 != 51 )
                  {
                    if ( v71 != 54 || v40[2] != 52 )
                      goto LABEL_229;
LABEL_252:
                    if ( v40[3] == 120 )
                      goto LABEL_377;
LABEL_253:
                    v14 = v40 + 3;
                    qword_4F06460 = v40 + 3;
                    goto LABEL_229;
                  }
                  if ( v40[2] == 50 )
                    goto LABEL_252;
                }
              }
            }
          }
LABEL_229:
          v20 = dword_4D04964;
          if ( !v62 || (a6 = v106) != 0 || (unsigned __int8)((*v14 & 0xDF) - 73) > 1u )
          {
            v113 = 4;
            v21 = v14 - 1;
          }
          else
          {
            v113 = 4;
            v106 = v110 == 0;
            v69 = v105;
            if ( v110 )
              v69 = v110;
            v105 = v69;
            v70 = src;
            if ( v110 )
              v70 = v110;
            qword_4F06460 = v14 + 1;
            src = v70;
            v21 = v14++;
          }
          continue;
        }
      }
    }
    break;
  }
LABEL_398:
  if ( v137 && !unk_4F04D80 )
  {
    v47 = v137;
    goto LABEL_171;
  }
LABEL_168:
  v45 = 8;
  *v23 = (__int64)(qword_4F06460 - 1);
  return v45;
}
