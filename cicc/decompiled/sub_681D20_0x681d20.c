// Function: sub_681D20
// Address: 0x681d20
//
int __fastcall sub_681D20(__int64 a1)
{
  int v1; // eax
  bool v2; // zf
  unsigned __int64 v3; // rbx
  int v4; // eax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned __int64 v9; // r12
  int v10; // esi
  unsigned __int64 v11; // r13
  _QWORD *v12; // rdi
  unsigned __int64 v13; // r15
  _BYTE *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rbx
  _BYTE *v17; // rbx
  __int64 v18; // rax
  _BYTE *v19; // r14
  size_t v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  FILE *v23; // r12
  __int64 v24; // r14
  char v25; // al
  __int64 v26; // r12
  int v27; // r15d
  const char *v28; // rax
  int result; // eax
  int v30; // ebx
  __int64 v31; // r12
  __int64 v32; // rbx
  __int64 k; // rbx
  __int64 v34; // rdi
  __int64 v35; // rax
  FILE *v36; // rsi
  __int64 v37; // rdi
  __int64 v38; // rax
  char v39; // al
  unsigned __int8 v40; // r12
  int v41; // r14d
  char v42; // al
  unsigned __int8 v43; // di
  _QWORD *v44; // rdi
  __int64 v45; // rax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rsi
  char v48; // cl
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // r14
  char v51; // al
  unsigned __int64 v52; // rax
  char i; // al
  unsigned __int8 v54; // di
  bool v55; // r12
  int v56; // edi
  int v57; // ebx
  unsigned int v58; // r14d
  _QWORD *v59; // r13
  __int64 v60; // rdx
  __int64 v61; // rsi
  __int64 v62; // rdx
  _BYTE *v63; // rdi
  const char *v64; // r13
  size_t v65; // rax
  _QWORD *v66; // r14
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rax
  unsigned __int8 v70; // al
  int v71; // edi
  char *v72; // r14
  char v73; // al
  _QWORD *v74; // r13
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // rdx
  size_t v78; // rax
  _QWORD *v79; // rbx
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 j; // rbx
  __int64 v84; // rax
  char *v85; // r14
  size_t v86; // rax
  int v87; // eax
  char *v88; // r12
  size_t v89; // rax
  size_t v90; // rax
  unsigned __int8 v91; // al
  char *v92; // r14
  size_t v93; // rax
  size_t v94; // rax
  int v95; // edi
  int v96; // edi
  size_t v97; // rax
  unsigned int v98; // edi
  int v99; // esi
  char *v100; // r12
  __int64 *v101; // r15
  int v102; // eax
  __int64 v103; // r12
  int v104; // r13d
  __int64 v105; // rax
  int v106; // r13d
  bool v107; // cc
  char *v108; // r15
  __int64 *v109; // r13
  int v110; // r14d
  char v111; // al
  __int64 v112; // rdx
  __int64 v113; // rdx
  unsigned int v114; // eax
  __int64 v115; // rax
  unsigned __int64 v116; // rdx
  unsigned __int64 v117; // rsi
  __int64 v118; // rdx
  char v119; // cl
  unsigned __int64 v120; // rdx
  __int64 v121; // rdx
  __int64 *v122; // rax
  int v123; // eax
  char *v124; // r12
  __int64 v125; // rax
  int v126; // r14d
  int v127; // r12d
  _QWORD *v128; // rdi
  int v129; // ebx
  __int64 v130; // rax
  int v131; // r15d
  char *v132; // rbx
  char v133; // r15
  unsigned __int16 v134; // r13
  __int64 v135; // rdi
  __int64 v136; // rdx
  char *v137; // r15
  __int64 v138; // rax
  __int64 v139; // rdi
  __int64 v140; // rdx
  char v141; // r9
  int v142; // eax
  __int64 v143; // rdi
  __int64 v144; // rdx
  __int64 v145; // rsi
  __int64 v146; // rax
  unsigned __int64 v147; // [rsp+0h] [rbp-B0h]
  char v148; // [rsp+0h] [rbp-B0h]
  char v149; // [rsp+0h] [rbp-B0h]
  char v150; // [rsp+0h] [rbp-B0h]
  __int64 v151; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v152; // [rsp+8h] [rbp-A8h]
  int v153; // [rsp+10h] [rbp-A0h]
  __int64 *v154; // [rsp+10h] [rbp-A0h]
  char v155; // [rsp+10h] [rbp-A0h]
  bool v156; // [rsp+18h] [rbp-98h]
  unsigned __int64 v157; // [rsp+18h] [rbp-98h]
  char *v158; // [rsp+18h] [rbp-98h]
  int v159; // [rsp+18h] [rbp-98h]
  char v160; // [rsp+23h] [rbp-8Dh]
  int v161; // [rsp+24h] [rbp-8Ch]
  char v162; // [rsp+24h] [rbp-8Ch]
  char v163; // [rsp+24h] [rbp-8Ch]
  int v164; // [rsp+24h] [rbp-8Ch]
  char v166[4]; // [rsp+3Ch] [rbp-74h] BYREF
  char s[112]; // [rsp+40h] [rbp-70h] BYREF

  sub_823800(qword_4D039E0);
  if ( *(_DWORD *)a1 )
    goto LABEL_2;
  v55 = 0;
  if ( unk_4D04728 && *(_BYTE *)(a1 + 180) <= 5u )
    v55 = *(_DWORD *)(a1 + 176) != 0;
  v56 = 2375;
  if ( !unk_4D042B0 )
  {
    v57 = 1;
    v21 = *(unsigned int *)(a1 + 136);
    if ( !(_DWORD)v21 )
      goto LABEL_148;
    v57 = *(_DWORD *)(a1 + 144);
    if ( !v57 )
    {
      if ( (unsigned int)v21 < unk_4F0647C && !HIDWORD(qword_4F07468) )
        sub_67BDC0(v21, (int *)(a1 + 172));
      v58 = *(_DWORD *)(a1 + 168);
      if ( dword_4F073CC[0] )
      {
        v59 = (_QWORD *)qword_4D039E0;
        v60 = *(_QWORD *)(qword_4D039E0 + 16);
        if ( (unsigned __int64)(v60 + 1) > *(_QWORD *)(qword_4D039E0 + 8) )
        {
          sub_823810(qword_4D039E0);
          v60 = v59[2];
        }
        *(_BYTE *)(v59[4] + v60) = 27;
        v61 = v59[2];
        v62 = v61 + 1;
        v59[2] = v61 + 1;
        if ( (unsigned __int64)(v61 + 2) > v59[1] )
        {
          sub_823810(v59);
          v62 = v59[2];
        }
        *(_BYTE *)(v59[4] + v62) = 5;
        ++v59[2];
      }
      v63 = *(_BYTE **)(a1 + 160);
      if ( *v63 != 45 || v63[1] )
      {
        if ( *(_QWORD *)(a1 + 152) )
        {
          v64 = (const char *)sub_723640(*(_QWORD *)(a1 + 152), 0, 0);
          v65 = strlen(v64);
          sub_8238B0(qword_4D039E0, v64, v65);
        }
        else
        {
          sub_722FC0(v63, qword_4D039E0, 0, 0);
        }
        if ( v58 )
        {
          sprintf(s, "(%lu)", v58);
          v97 = strlen(s);
          sub_8238B0(qword_4D039E0, s, v97);
        }
      }
      else
      {
        sprintf(s, "%lu", v58);
        v92 = sub_67C860(1491);
        v93 = strlen(v92);
        sub_8238B0(qword_4D039E0, v92, v93);
        sub_8238B0(qword_4D039E0, " ", 1);
        v94 = strlen(s);
        sub_8238B0(qword_4D039E0, s, v94);
      }
      v66 = (_QWORD *)qword_4D039E0;
      if ( dword_4F073CC[0] )
      {
        v67 = *(_QWORD *)(qword_4D039E0 + 16);
        if ( (unsigned __int64)(v67 + 1) > *(_QWORD *)(qword_4D039E0 + 8) )
        {
          sub_823810(qword_4D039E0);
          v67 = v66[2];
        }
        *(_BYTE *)(v66[4] + v67) = 27;
        v68 = v66[2];
        v69 = v68 + 1;
        v66[2] = v68 + 1;
        if ( (unsigned __int64)(v68 + 2) > v66[1] )
        {
          sub_823810(v66);
          v69 = v66[2];
        }
        *(_BYTE *)(v66[4] + v69) = 1;
        ++v66[2];
        v66 = (_QWORD *)qword_4D039E0;
      }
      v21 = (__int64)v66;
      sub_8238B0(v66, ": ", 2);
      goto LABEL_148;
    }
    v56 = 1490;
  }
  v57 = 0;
  v85 = sub_67C860(v56);
  v86 = strlen(v85);
  sub_8238B0(qword_4D039E0, v85, v86);
  v21 = qword_4D039E0;
  sub_8238B0(qword_4D039E0, ": ", 2);
LABEL_148:
  v70 = *(_BYTE *)(a1 + 180);
  if ( v70 > 7u || v70 < unk_4F07480 )
  {
    switch ( v70 )
    {
      case 2u:
        v96 = v57 + 2685;
        goto LABEL_188;
      case 4u:
        v96 = v57 + 1494;
LABEL_188:
        v72 = sub_67C860(v96);
        v73 = 4;
        goto LABEL_152;
      case 5u:
        v95 = v57 + 1496;
        goto LABEL_185;
      case 6u:
        v95 = v57 + 2657;
LABEL_185:
        v72 = sub_67C860(v95);
        v73 = 3;
        goto LABEL_152;
      case 7u:
      case 8u:
        break;
      case 9u:
        v71 = v57 + 1500;
        goto LABEL_151;
      case 0xAu:
        v71 = v57 + 1502;
        goto LABEL_151;
      case 0xBu:
        v71 = v57 + 1504;
        goto LABEL_151;
      default:
        goto LABEL_190;
    }
  }
  v71 = v57 + 1498;
LABEL_151:
  v72 = sub_67C860(v71);
  v73 = 2;
LABEL_152:
  v74 = (_QWORD *)qword_4D039E0;
  if ( dword_4F073CC[0] )
  {
    v75 = *(_QWORD *)(qword_4D039E0 + 16);
    if ( (unsigned __int64)(v75 + 1) > *(_QWORD *)(qword_4D039E0 + 8) )
    {
      v163 = v73;
      sub_823810(qword_4D039E0);
      v75 = v74[2];
      v73 = v163;
    }
    *(_BYTE *)(v74[4] + v75) = 27;
    v76 = v74[2];
    v77 = v76 + 1;
    v74[2] = v76 + 1;
    if ( (unsigned __int64)(v76 + 2) > v74[1] )
    {
      v162 = v73;
      sub_823810(v74);
      v77 = v74[2];
      v73 = v162;
    }
    *(_BYTE *)(v74[4] + v77) = v73;
    ++v74[2];
    v74 = (_QWORD *)qword_4D039E0;
  }
  v78 = strlen(v72);
  sub_8238B0(v74, v72, v78);
  if ( dword_4F073CC[0] )
  {
    v79 = (_QWORD *)qword_4D039E0;
    v80 = *(_QWORD *)(qword_4D039E0 + 16);
    if ( (unsigned __int64)(v80 + 1) > *(_QWORD *)(qword_4D039E0 + 8) )
    {
      sub_823810(qword_4D039E0);
      v80 = v79[2];
    }
    *(_BYTE *)(v79[4] + v80) = 27;
    v81 = v79[2];
    v82 = v81 + 1;
    v79[2] = v81 + 1;
    if ( (unsigned __int64)(v81 + 2) > v79[1] )
    {
      sub_823810(v79);
      v82 = v79[2];
    }
    *(_BYTE *)(v79[4] + v82) = 1;
    ++v79[2];
  }
  if ( v55 )
  {
    v87 = sub_67D2D0(*(_DWORD *)(a1 + 176));
    sprintf(s, "%d", v87);
    v88 = sub_67C860((unsigned int)(*(_BYTE *)(a1 + 180) > 7u) + 1506);
    sub_8238B0(qword_4D039E0, " #", 2);
    v89 = strlen(s);
    sub_8238B0(qword_4D039E0, s, v89);
    v90 = strlen(v88);
    sub_8238B0(qword_4D039E0, v88, v90);
  }
  sub_8238B0(qword_4D039E0, ": ", 2);
LABEL_2:
  sub_681B50(a1);
  v1 = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 )
  {
    if ( v1 == 2 )
    {
      v84 = *(_QWORD *)(a1 + 16);
      v3 = 2LL * (a1 != *(_QWORD *)(v84 + 40)) + 10;
      v161 = 2 * (a1 != *(_QWORD *)(v84 + 40)) + 20;
    }
    else
    {
      v2 = v1 == 3;
      v3 = 12;
      if ( v1 == 3 )
        v3 = 1;
      v4 = 11;
      if ( !v2 )
        v4 = 22;
      v161 = v4;
    }
  }
  else
  {
    v161 = 10;
    v3 = 0;
  }
  sub_8238B0(qword_4D039E0, *(_QWORD *)(qword_4D039E8 + 32), *(_QWORD *)(qword_4D039E8 + 16));
  v5 = qword_4D039E8;
  v6 = *(_QWORD *)(qword_4D039E8 + 16);
  if ( (unsigned __int64)(v6 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
  {
    sub_823810(qword_4D039E8);
    v5 = qword_4D039E8;
    v6 = *(_QWORD *)(qword_4D039E8 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v5 + 32) + v6) = 0;
  ++*(_QWORD *)(v5 + 16);
  v7 = qword_4D039E0;
  v8 = *(_QWORD *)(qword_4D039E0 + 16);
  if ( (unsigned __int64)(v8 + 1) > *(_QWORD *)(qword_4D039E0 + 8) )
  {
    sub_823810(qword_4D039E0);
    v7 = qword_4D039E0;
    v8 = *(_QWORD *)(qword_4D039E0 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v7 + 32) + v8) = 0;
  v9 = *(_QWORD *)(v7 + 16);
  v10 = dword_4F073CC[0];
  v11 = *(_QWORD *)(v7 + 32);
  *(_QWORD *)(v7 + 16) = v9 + 1;
  v153 = HIDWORD(qword_4F07468) | qword_4F07468;
  v156 = qword_4F07468 == 0;
  if ( !v10 )
  {
    v12 = (_QWORD *)qword_4D039D8;
    v13 = v3;
    while ( 1 )
    {
      v14 = (_BYTE *)(dword_4D039D0 - v13);
      if ( v13 )
      {
        v15 = v12[2];
        v16 = 0;
        do
        {
          if ( (unsigned __int64)(v15 + 1) > v12[1] )
          {
            sub_823810(v12);
            v12 = (_QWORD *)qword_4D039D8;
            v15 = *(_QWORD *)(qword_4D039D8 + 16);
          }
          ++v16;
          *(_BYTE *)(v12[4] + v15) = 32;
          v15 = v12[2] + 1LL;
          v12[2] = v15;
        }
        while ( v13 != v16 );
      }
      if ( (unsigned __int64)v14 >= v9 || !v156 )
      {
        v20 = strlen((const char *)v11);
        sub_8238B0(v12, v11, v20);
        goto LABEL_36;
      }
      v17 = &v14[v11 - 1];
      if ( v14[v11] == 32 )
        v17 = &v14[v11];
      if ( v11 < (unsigned __int64)v17 )
        break;
LABEL_33:
      if ( *v17 == 32 )
        goto LABEL_29;
LABEL_30:
      sub_8238B0(v12, v11, v14);
      v12 = (_QWORD *)qword_4D039D8;
      v18 = *(_QWORD *)(qword_4D039D8 + 16);
      if ( (unsigned __int64)(v18 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
      {
        sub_823810(qword_4D039D8);
        v12 = (_QWORD *)qword_4D039D8;
        v18 = *(_QWORD *)(qword_4D039D8 + 16);
      }
      v13 = v161;
      *(_BYTE *)(v12[4] + v18) = 10;
      ++v12[2];
      v19 = &v14[*v17 == 32];
      v9 -= (unsigned __int64)v19;
      v11 += (unsigned __int64)v19;
    }
    while ( *v17 != 32 )
    {
      if ( (_BYTE *)v11 == --v17 )
        goto LABEL_33;
    }
LABEL_29:
    v14 = &v17[-v11];
    goto LABEL_30;
  }
  v39 = *(_BYTE *)v11;
  if ( *(_BYTE *)v11 )
  {
    v160 = 1;
    v40 = 1;
    v41 = 1;
    v152 = 0;
    v157 = 0;
    v147 = 0;
    while ( 1 )
    {
      if ( v39 == 27 )
      {
        do
        {
          while ( !dword_4F073C8 )
          {
            v42 = *(_BYTE *)(v11 + 2);
            v11 += 2LL;
            if ( v42 != 27 )
              goto LABEL_79;
          }
          v43 = *(_BYTE *)(v11 + 1);
          v11 += 2LL;
          v40 = v43;
          sub_67BBF0(v43);
          v42 = *(_BYTE *)v11;
        }
        while ( *(_BYTE *)v11 == 27 );
LABEL_79:
        if ( !v42 )
          goto LABEL_36;
      }
      if ( !v41 )
      {
        v44 = (_QWORD *)qword_4D039D8;
        v45 = *(_QWORD *)(qword_4D039D8 + 16);
        goto LABEL_82;
      }
      if ( v40 == 1 )
        break;
      sub_67BBF0(1u);
      if ( v3 )
        goto LABEL_96;
LABEL_107:
      sub_67BBF0(v40);
      v44 = (_QWORD *)qword_4D039D8;
      v45 = *(_QWORD *)(qword_4D039D8 + 16);
LABEL_108:
      v157 = 0;
      v152 = dword_4D039D0 - v3;
LABEL_82:
      if ( (unsigned __int64)(v45 + 1) > v44[1] )
      {
        sub_823810(v44);
        v44 = (_QWORD *)qword_4D039D8;
        v45 = *(_QWORD *)(qword_4D039D8 + 16);
      }
      *(_BYTE *)(v44[4] + v45) = *(_BYTE *)v11;
      v46 = v44[2];
      v47 = v46 + 1;
      v44[2] = v46 + 1;
      if ( v153 )
      {
        v49 = v11;
        v41 = 0;
        goto LABEL_91;
      }
      v2 = v152-- == 1;
      v48 = *(_BYTE *)v11;
      if ( !v2 )
      {
        v51 = v160;
        v41 = 0;
        if ( v48 == 32 )
          v51 = v40;
        else
          v47 = v147;
        v147 = v47;
        v160 = v51;
        v52 = v157;
        if ( v48 == 32 )
          v52 = v11;
        v157 = v52;
        v49 = v11;
        goto LABEL_91;
      }
      if ( v48 == 32 )
      {
        v44[2] = v46;
        goto LABEL_88;
      }
      for ( i = *(_BYTE *)(v11 + 1); i == 27; i = *(_BYTE *)(v11 + 1) )
      {
        while ( !dword_4F073C8 )
        {
          i = *(_BYTE *)(v11 + 3);
          v11 += 2LL;
          if ( i != 27 )
            goto LABEL_116;
        }
        v54 = *(_BYTE *)(v11 + 2);
        v11 += 2LL;
        v40 = v54;
        sub_67BBF0(v54);
      }
LABEL_116:
      if ( !i )
      {
        v49 = v11;
        v41 = 0;
        goto LABEL_90;
      }
      v44 = (_QWORD *)qword_4D039D8;
      if ( i == 32 )
      {
        v46 = *(_QWORD *)(qword_4D039D8 + 16);
        ++v11;
        v47 = v46 + 1;
        goto LABEL_88;
      }
      if ( !v157 )
      {
        v46 = *(_QWORD *)(qword_4D039D8 + 16);
        v47 = v46 + 1;
LABEL_88:
        if ( v44[1] < v47 )
          goto LABEL_120;
        goto LABEL_89;
      }
      v40 = v160;
      v11 = v157;
      v46 = v147 - 1;
      *(_QWORD *)(qword_4D039D8 + 16) = v147 - 1;
      if ( v44[1] < v147 )
      {
LABEL_120:
        sub_823810(v44);
        v44 = (_QWORD *)qword_4D039D8;
        v46 = *(_QWORD *)(qword_4D039D8 + 16);
      }
LABEL_89:
      v41 = 1;
      *(_BYTE *)(v44[4] + v46) = 10;
      v49 = v11;
      ++v44[2];
LABEL_90:
      v3 = v161;
LABEL_91:
      v39 = *(_BYTE *)(v49 + 1);
      ++v11;
      if ( !v39 )
        goto LABEL_36;
    }
    if ( !v3 )
    {
      v44 = (_QWORD *)qword_4D039D8;
      v45 = *(_QWORD *)(qword_4D039D8 + 16);
      goto LABEL_108;
    }
LABEL_96:
    v44 = (_QWORD *)qword_4D039D8;
    v50 = 0;
    v45 = *(_QWORD *)(qword_4D039D8 + 16);
    do
    {
      if ( (unsigned __int64)(v45 + 1) > v44[1] )
      {
        sub_823810(v44);
        v44 = (_QWORD *)qword_4D039D8;
        v45 = *(_QWORD *)(qword_4D039D8 + 16);
      }
      ++v50;
      *(_BYTE *)(v44[4] + v45) = 32;
      v45 = v44[2] + 1LL;
      v44[2] = v45;
    }
    while ( v50 < v3 );
    if ( v40 == 1 )
      goto LABEL_108;
    goto LABEL_107;
  }
LABEL_36:
  v21 = qword_4D039D8;
  v22 = *(_QWORD *)(qword_4D039D8 + 16);
  if ( (unsigned __int64)(v22 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
  {
    sub_823810(qword_4D039D8);
    v21 = qword_4D039D8;
    v22 = *(_QWORD *)(qword_4D039D8 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v21 + 32) + v22) = 10;
  ++*(_QWORD *)(v21 + 16);
  v23 = qword_4D04908;
  if ( !qword_4D04908 || *(_DWORD *)a1 == 3 )
    goto LABEL_48;
  v24 = *(_QWORD *)(a1 + 16);
  if ( !v24 )
  {
    v91 = *(_BYTE *)(a1 + 180) - 4;
    if ( v91 <= 7u )
    {
      v24 = a1;
      v25 = aRwweeccccCli[v91];
      goto LABEL_43;
    }
LABEL_190:
    sub_721090(v21);
  }
  if ( (unsigned __int8)(*(_BYTE *)(v24 + 180) - 4) > 7u )
    goto LABEL_190;
  v25 = tolower(aRwweeccccCli[(unsigned __int8)(*(_BYTE *)(v24 + 180) - 4)]);
LABEL_43:
  putc(v25, v23);
  fputc(32, qword_4D04908);
  if ( *(_DWORD *)(a1 + 96) )
  {
    v26 = *(unsigned int *)(v24 + 128);
    v27 = *(unsigned __int16 *)(a1 + 100);
    v28 = (const char *)sub_723260(*(_QWORD *)(v24 + 120));
    fprintf(qword_4D04908, "\"%s\" %lu %d ", v28, v26, v27);
  }
  else
  {
    fwrite("\"\" 0 0 ", 1u, 7u, qword_4D04908);
  }
  if ( *(_BYTE *)(v24 + 180) == 11 )
    fwrite("(internal error) ", 1u, 0x11u, qword_4D04908);
  fputs(*(const char **)(qword_4D039E8 + 32), qword_4D04908);
  fputc(10, qword_4D04908);
LABEL_48:
  sub_823800(qword_4D039E8);
  result = a1;
  v30 = *(_DWORD *)a1;
  if ( !*(_DWORD *)a1 )
  {
    v31 = *(_QWORD *)(a1 + 24);
    if ( !v31 )
      goto LABEL_354;
    do
    {
      sub_681D20(v31);
      v31 = *(_QWORD *)(v31 + 8);
    }
    while ( v31 );
    result = a1;
    if ( !*(_DWORD *)a1 )
    {
LABEL_354:
      if ( HIDWORD(qword_4F07468) )
        goto LABEL_53;
      v37 = *(unsigned int *)(a1 + 136);
      if ( !(_DWORD)v37 || *(_DWORD *)(a1 + 144) )
        goto LABEL_53;
      v38 = sub_729B10(v37, v166, s, 0);
      if ( (!v38 || !*(_QWORD *)(v38 + 64)) && !HIDWORD(qword_4F07468) )
      {
        v98 = *(_DWORD *)(a1 + 136);
        v99 = unk_4F0647C;
        if ( v98 >= unk_4F0647C )
        {
          v100 = (char *)unk_4F06498;
          v101 = (__int64 *)unk_4F06458;
          if ( v98 != unk_4F0647C )
          {
            do
            {
              while ( 1 )
              {
                v102 = *((_DWORD *)v101 + 4);
                if ( (unsigned int)(v102 - 1) <= 1 )
                  break;
                v101 = (__int64 *)*v101;
              }
              v100 = (char *)v101[1];
              ++v99;
              v101 = (__int64 *)*v101;
              if ( v102 == 2 )
                v100 += 2;
            }
            while ( v98 != v99 );
          }
          v164 = 1;
          v154 = v101;
          v158 = v100;
          v103 = a1;
          while ( 1 )
          {
            v21 = qword_4D039E8;
            v104 = 2;
            v105 = *(_QWORD *)(qword_4D039E8 + 16);
            while ( 1 )
            {
              if ( (unsigned __int64)(v105 + 1) > *(_QWORD *)(v21 + 8) )
              {
                sub_823810(v21);
                v21 = qword_4D039E8;
                v105 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v21 + 32) + v105) = 32;
              v105 = *(_QWORD *)(v21 + 16) + 1LL;
              *(_QWORD *)(v21 + 16) = v105;
              if ( v104 == 1 )
                break;
              v104 = 1;
            }
            v106 = 0;
            if ( dword_4D03A04 > 0 )
            {
              do
              {
                if ( (unsigned __int64)(v105 + 1) > *(_QWORD *)(v21 + 8) )
                {
                  sub_823810(v21);
                  v21 = qword_4D039E8;
                  v105 = *(_QWORD *)(qword_4D039E8 + 16);
                }
                ++v106;
                *(_BYTE *)(*(_QWORD *)(v21 + 32) + v105) = 32;
                v105 = *(_QWORD *)(v21 + 16) + 1LL;
                v107 = dword_4D03A04 <= v106;
                *(_QWORD *)(v21 + 16) = v105;
              }
              while ( !v107 );
            }
            if ( !v30 || *(_WORD *)(v103 + 140) )
              break;
LABEL_214:
            sub_67BD40();
            if ( v164 == 2 )
              goto LABEL_70;
LABEL_215:
            ++v164;
            ++v30;
          }
          v108 = v158;
          v109 = v154;
          v110 = 1;
          while ( 1 )
          {
            while ( 1 )
            {
              if ( !v109 )
                goto LABEL_220;
LABEL_219:
              if ( (char *)v109[1] != v108 )
                goto LABEL_220;
              v114 = *((_DWORD *)v109 + 4);
              if ( v114 == 2 )
              {
LABEL_260:
                if ( !v30 )
                  goto LABEL_261;
LABEL_277:
                v21 = qword_4D039E8;
                v115 = *(_QWORD *)(qword_4D039E8 + 16);
                v117 = v115 + 1;
LABEL_248:
                if ( v117 > *(_QWORD *)(v21 + 8) )
                {
                  sub_823810(v21);
                  v21 = qword_4D039E8;
                  v115 = *(_QWORD *)(qword_4D039E8 + 16);
                }
                *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 94;
                ++*(_QWORD *)(v21 + 16);
                goto LABEL_214;
              }
              if ( v114 <= 2 )
                break;
              if ( v114 != 3 )
                goto LABEL_190;
              v21 = qword_4D039E8;
              v115 = *(_QWORD *)(qword_4D039E8 + 16);
              v117 = v115 + 1;
              if ( v30 && (unsigned __int16)v110 >= *(_WORD *)(v103 + 140) )
                goto LABEL_248;
              if ( v117 > *(_QWORD *)(qword_4D039E8 + 8) )
              {
                sub_823810(qword_4D039E8);
                v21 = qword_4D039E8;
                v115 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              ++v110;
              v108 += 2;
              *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 32;
              ++*(_QWORD *)(v21 + 16);
              v109 = (__int64 *)*v109;
            }
            if ( v114 )
            {
              v21 = qword_4D039E8;
              v115 = *(_QWORD *)(qword_4D039E8 + 16);
              v116 = *(_QWORD *)(qword_4D039E8 + 8);
              v117 = v115 + 1;
              if ( !v30 )
              {
                if ( v117 > v116 )
                {
                  sub_823810(qword_4D039E8);
                  v21 = qword_4D039E8;
                  v115 = *(_QWORD *)(qword_4D039E8 + 16);
                }
                *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 92;
                ++*(_QWORD *)(v21 + 16);
LABEL_261:
                sub_67BD40();
                goto LABEL_215;
              }
              if ( (unsigned __int16)v110 < *(_WORD *)(v103 + 140) )
              {
                if ( v117 > v116 )
                {
                  sub_823810(qword_4D039E8);
                  v21 = qword_4D039E8;
                  v115 = *(_QWORD *)(qword_4D039E8 + 16);
                }
                *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 32;
                v118 = *(_QWORD *)(v21 + 16);
                v115 = v118 + 1;
                *(_QWORD *)(v21 + 16) = v118 + 1;
LABEL_247:
                v117 = v118 + 2;
              }
              goto LABEL_248;
            }
            v21 = qword_4D039E8;
            v115 = *(_QWORD *)(qword_4D039E8 + 16);
            v120 = *(_QWORD *)(qword_4D039E8 + 8);
            v117 = v115 + 1;
            if ( v30 )
            {
              if ( (unsigned __int16)v110 >= *(_WORD *)(v103 + 140) )
                goto LABEL_248;
              if ( v117 > v120 )
              {
                sub_823810(qword_4D039E8);
                v21 = qword_4D039E8;
                v115 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 32;
              v121 = *(_QWORD *)(v21 + 16);
              v115 = v121 + 1;
              v117 = v121 + 2;
              *(_QWORD *)(v21 + 16) = v121 + 1;
              if ( *(_WORD *)(v103 + 140) <= (unsigned __int16)(v110 + 1) )
                goto LABEL_248;
              if ( v117 > *(_QWORD *)(v21 + 8) )
              {
                sub_823810(v21);
                v21 = qword_4D039E8;
                v115 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 32;
              v118 = *(_QWORD *)(v21 + 16);
              v115 = v118 + 1;
              *(_QWORD *)(v21 + 16) = v118 + 1;
              if ( *(_WORD *)(v103 + 140) <= (unsigned __int16)(v110 + 2) )
                goto LABEL_247;
              if ( *((_BYTE *)v109 + 24) != 9 )
              {
LABEL_271:
                if ( (unsigned __int64)(v115 + 1) > *(_QWORD *)(v21 + 8) )
                {
                  sub_823810(v21);
                  v21 = qword_4D039E8;
                  v115 = *(_QWORD *)(qword_4D039E8 + 16);
                }
                *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 32;
                ++*(_QWORD *)(v21 + 16);
                goto LABEL_274;
              }
            }
            else
            {
              if ( v117 > v120 )
              {
                sub_823810(qword_4D039E8);
                v21 = qword_4D039E8;
                v115 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = 63;
              v145 = *(_QWORD *)(v21 + 16);
              v146 = v145 + 1;
              *(_QWORD *)(v21 + 16) = v145 + 1;
              if ( (unsigned __int64)(v145 + 2) > *(_QWORD *)(v21 + 8) )
              {
                sub_823810(v21);
                v21 = qword_4D039E8;
                v146 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v21 + 32) + v146) = 63;
              v115 = *(_QWORD *)(v21 + 16) + 1LL;
              *(_QWORD *)(v21 + 16) = v115;
              if ( *((_BYTE *)v109 + 24) == 13 )
                goto LABEL_271;
            }
            if ( (unsigned __int64)(v115 + 1) > *(_QWORD *)(v21 + 8) )
            {
              sub_823810(v21);
              v21 = qword_4D039E8;
              v115 = *(_QWORD *)(qword_4D039E8 + 16);
            }
            *(_BYTE *)(*(_QWORD *)(v21 + 32) + v115) = *((_BYTE *)v109 + 24);
            ++*(_QWORD *)(v21 + 16);
LABEL_274:
            v122 = (__int64 *)*v109;
            v110 += 3;
            ++v108;
            if ( !*v109 )
            {
              v109 = 0;
              while ( 1 )
              {
LABEL_220:
                v111 = *v108;
                if ( !*v108 )
                  goto LABEL_260;
                if ( v111 == 10 )
                  v111 = *(_BYTE *)(sub_7AF1D0(v108) + 50);
                if ( v30 )
                {
                  if ( *(_WORD *)(v103 + 140) <= (unsigned __int16)v110 )
                    goto LABEL_277;
                  if ( v111 != 13 && ((((unsigned __int8)v30 ^ 1) & 1) != 0 || v111 == 9) )
                  {
LABEL_236:
                    v21 = qword_4D039E8;
                    v113 = *(_QWORD *)(qword_4D039E8 + 16);
                    if ( (unsigned __int64)(v113 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
                    {
                      v149 = v111;
                      sub_823810(qword_4D039E8);
                      v21 = qword_4D039E8;
                      v111 = v149;
                      v113 = *(_QWORD *)(qword_4D039E8 + 16);
                    }
                    *(_BYTE *)(*(_QWORD *)(v21 + 32) + v113) = v111;
                    ++*(_QWORD *)(v21 + 16);
                    goto LABEL_231;
                  }
                }
                else if ( v111 != 13 )
                {
                  goto LABEL_236;
                }
                v21 = qword_4D039E8;
                v112 = *(_QWORD *)(qword_4D039E8 + 16);
                if ( (unsigned __int64)(v112 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
                {
                  v148 = v111;
                  sub_823810(qword_4D039E8);
                  v21 = qword_4D039E8;
                  v111 = v148;
                  v112 = *(_QWORD *)(qword_4D039E8 + 16);
                }
                *(_BYTE *)(*(_QWORD *)(v21 + 32) + v112) = 32;
                ++*(_QWORD *)(v21 + 16);
LABEL_231:
                ++v110;
                if ( unk_4D0432C && unk_4F064A8 )
                {
                  v119 = *v108;
                  *v108 = v111;
                  if ( v111 < 0 )
                  {
                    v21 = (__int64)v108;
                    v150 = v119;
                    v123 = sub_721AB0(v108, 0, 0);
                    *v108 = v150;
                    if ( v123 > 1 )
                    {
                      v151 = v103;
                      v124 = &v108[v123 - 1];
                      do
                      {
                        ++v108;
                        if ( !v30 )
                        {
                          v21 = qword_4D039E8;
                          v125 = *(_QWORD *)(qword_4D039E8 + 16);
                          if ( (unsigned __int64)(v125 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
                          {
                            sub_823810(qword_4D039E8);
                            v21 = qword_4D039E8;
                            v125 = *(_QWORD *)(qword_4D039E8 + 16);
                          }
                          *(_BYTE *)(*(_QWORD *)(v21 + 32) + v125) = *v108;
                          ++*(_QWORD *)(v21 + 16);
                        }
                      }
                      while ( v108 != v124 );
                      v103 = v151;
                    }
                  }
                  else
                  {
                    *v108 = v119;
                  }
                }
                ++v108;
                if ( v109 )
                  goto LABEL_219;
              }
            }
            if ( *((_DWORD *)v122 + 4) == 1 && v122[1] == v109[1] )
              goto LABEL_260;
            v109 = (__int64 *)*v109;
          }
        }
        if ( !(unsigned int)sub_67BDC0(v98, (int *)(a1 + 172)) )
          goto LABEL_70;
        v126 = 0;
        v159 = 1;
        v127 = *(_DWORD *)(a1 + 172);
        while ( 2 )
        {
          v128 = (_QWORD *)qword_4D039E8;
          v129 = 2;
          v130 = *(_QWORD *)(qword_4D039E8 + 16);
          while ( 1 )
          {
            if ( (unsigned __int64)(v130 + 1) > v128[1] )
            {
              sub_823810(v128);
              v128 = (_QWORD *)qword_4D039E8;
              v130 = *(_QWORD *)(qword_4D039E8 + 16);
            }
            *(_BYTE *)(v128[4] + v130) = 32;
            v130 = v128[2] + 1LL;
            v128[2] = v130;
            if ( v129 == 1 )
              break;
            v129 = 1;
          }
          v131 = 0;
          if ( dword_4D03A04 > 0 )
          {
            do
            {
              if ( (unsigned __int64)(v130 + 1) > v128[1] )
              {
                sub_823810(v128);
                v128 = (_QWORD *)qword_4D039E8;
                v130 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              ++v131;
              *(_BYTE *)(v128[4] + v130) = 32;
              v130 = v128[2] + 1LL;
              v107 = dword_4D03A04 <= v131;
              v128[2] = v130;
            }
            while ( !v107 );
          }
          if ( !v126 )
          {
            v132 = (char *)qword_4CFFDA0;
            v133 = *(_BYTE *)qword_4CFFDA0;
            if ( *(_BYTE *)qword_4CFFDA0 == 10 )
              goto LABEL_330;
LABEL_304:
            v134 = 1;
            while ( v133 != 13 )
            {
              if ( v126 )
              {
                if ( *(_WORD *)(a1 + 140) <= v134 )
                  goto LABEL_320;
                if ( (((unsigned __int8)v126 ^ 1) & 1) == 0 && v133 != 9 )
                  goto LABEL_309;
              }
              v139 = qword_4D039E8;
              v140 = *(_QWORD *)(qword_4D039E8 + 16);
              if ( (unsigned __int64)(v140 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
              {
                sub_823810(qword_4D039E8);
                v139 = qword_4D039E8;
                v140 = *(_QWORD *)(qword_4D039E8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v139 + 32) + v140) = v133;
              ++*(_QWORD *)(v139 + 16);
LABEL_312:
              ++v134;
              if ( unk_4D0432C && v127 )
              {
                v141 = *v132;
                *v132 = v133;
                if ( v133 >= 0 )
                {
                  *v132 = v141;
                  v137 = v132;
                  goto LABEL_315;
                }
                v155 = v141;
                v142 = sub_721AB0(v132, 0, 0);
                *v132 = v155;
                if ( v142 > 1 )
                {
                  v137 = &v132[v142 - 1];
                  do
                  {
                    ++v132;
                    if ( !v126 )
                    {
                      v143 = qword_4D039E8;
                      v144 = *(_QWORD *)(qword_4D039E8 + 16);
                      if ( (unsigned __int64)(v144 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
                      {
                        sub_823810(qword_4D039E8);
                        v143 = qword_4D039E8;
                        v144 = *(_QWORD *)(qword_4D039E8 + 16);
                      }
                      *(_BYTE *)(*(_QWORD *)(v143 + 32) + v144) = *v132;
                      ++*(_QWORD *)(v143 + 16);
                    }
                  }
                  while ( v132 != v137 );
                  goto LABEL_315;
                }
              }
              v137 = v132;
LABEL_315:
              v132 = v137 + 1;
              v133 = v137[1];
              if ( v133 == 10 )
                goto LABEL_329;
            }
            if ( v132[1] == 10 )
            {
LABEL_329:
              if ( v126 )
              {
LABEL_320:
                v128 = (_QWORD *)qword_4D039E8;
                goto LABEL_321;
              }
LABEL_330:
              sub_67BD40();
LABEL_301:
              ++v159;
              ++v126;
              continue;
            }
            if ( v126 && *(_WORD *)(a1 + 140) <= v134 )
              goto LABEL_320;
LABEL_309:
            v135 = qword_4D039E8;
            v136 = *(_QWORD *)(qword_4D039E8 + 16);
            if ( (unsigned __int64)(v136 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
            {
              sub_823810(qword_4D039E8);
              v135 = qword_4D039E8;
              v136 = *(_QWORD *)(qword_4D039E8 + 16);
            }
            *(_BYTE *)(*(_QWORD *)(v135 + 32) + v136) = 32;
            ++*(_QWORD *)(v135 + 16);
            goto LABEL_312;
          }
          break;
        }
        if ( !*(_WORD *)(a1 + 140) )
          goto LABEL_300;
        v132 = (char *)qword_4CFFDA0;
        v133 = *(_BYTE *)qword_4CFFDA0;
        if ( *(_BYTE *)qword_4CFFDA0 != 10 )
          goto LABEL_304;
LABEL_321:
        v138 = v128[2];
        if ( (unsigned __int64)(v138 + 1) > v128[1] )
        {
          sub_823810(v128);
          v128 = (_QWORD *)qword_4D039E8;
          v138 = *(_QWORD *)(qword_4D039E8 + 16);
        }
        *(_BYTE *)(v128[4] + v138) = 94;
        ++v128[2];
LABEL_300:
        sub_67BD40();
        if ( v159 != 2 )
          goto LABEL_301;
      }
LABEL_70:
      result = a1;
      if ( !*(_DWORD *)a1 )
      {
LABEL_53:
        result = a1;
        if ( *(_BYTE *)(a1 + 180) != 2 )
        {
          v32 = *(_QWORD *)(a1 + 56);
          if ( !v32 )
            goto LABEL_353;
          do
          {
            sub_681D20(v32);
            v32 = *(_QWORD *)(v32 + 8);
          }
          while ( v32 );
          if ( !*(_DWORD *)a1 )
          {
LABEL_353:
            for ( j = *(_QWORD *)(a1 + 72); j; j = *(_QWORD *)(j + 8) )
            {
              *(_QWORD *)(j + 16) = a1;
              sub_681D20(j);
            }
          }
          for ( k = *(_QWORD *)(a1 + 40); k; k = *(_QWORD *)(k + 8) )
            sub_681D20(k);
          v34 = qword_4D039D8;
          v35 = *(_QWORD *)(qword_4D039D8 + 16);
          if ( !HIDWORD(qword_4F07468) )
          {
            if ( (unsigned __int64)(v35 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
            {
              sub_823810(qword_4D039D8);
              v34 = qword_4D039D8;
              v35 = *(_QWORD *)(qword_4D039D8 + 16);
            }
            *(_BYTE *)(*(_QWORD *)(v34 + 32) + v35) = 10;
            v35 = *(_QWORD *)(v34 + 16) + 1LL;
            *(_QWORD *)(v34 + 16) = v35;
          }
          if ( (unsigned __int64)(v35 + 1) > *(_QWORD *)(v34 + 8) )
          {
            sub_823810(v34);
            v34 = qword_4D039D8;
            v35 = *(_QWORD *)(qword_4D039D8 + 16);
          }
          *(_BYTE *)(*(_QWORD *)(v34 + 32) + v35) = 0;
          v36 = qword_4F07510;
          ++*(_QWORD *)(v34 + 16);
          fputs(*(const char **)(v34 + 32), v36);
          return fflush(qword_4F07510);
        }
      }
    }
  }
  return result;
}
