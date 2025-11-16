// Function: sub_6ED3D0
// Address: 0x6ed3d0
//
_QWORD *__fastcall sub_6ED3D0(__int64 a1, unsigned __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  _DWORD *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r11
  char v16; // al
  __int64 v17; // rax
  _QWORD *v18; // r15
  int v20; // eax
  _BYTE *v21; // rax
  unsigned __int8 v22; // r10
  __int64 v23; // r9
  _BYTE *v24; // r8
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // r10
  _DWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  _DWORD *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r9
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // r15
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rsi
  int v58; // eax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rax
  char v64; // r15
  char v65; // al
  int v66; // r15d
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rax
  int v75; // eax
  int v76; // eax
  __int64 i; // rdx
  int v78; // eax
  __int64 v79; // rdx
  __int64 v80; // r15
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // r15
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // r10
  bool v89; // zf
  __int64 v90; // rax
  __int64 v91; // rcx
  __int64 v92; // rax
  __int64 v93; // r15
  int v94; // eax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rdi
  int v99; // eax
  __int64 v100; // rax
  unsigned int v101; // eax
  __int64 v102; // rax
  _BOOL4 v103; // eax
  __int64 v104; // rdx
  int v105; // eax
  __int64 v106; // rax
  int v107; // eax
  __int64 v108; // rax
  __int64 v109; // rdi
  int v110; // eax
  __int64 v111; // r10
  __int64 v112; // rax
  __int64 v113; // rax
  unsigned __int8 v114; // [rsp+7h] [rbp-89h]
  _BYTE *v115; // [rsp+8h] [rbp-88h]
  __int64 v116; // [rsp+8h] [rbp-88h]
  __int64 v117; // [rsp+8h] [rbp-88h]
  __int64 v118; // [rsp+8h] [rbp-88h]
  __int64 v119; // [rsp+8h] [rbp-88h]
  __int64 *v120; // [rsp+10h] [rbp-80h]
  __int64 v121; // [rsp+10h] [rbp-80h]
  _BYTE *v122; // [rsp+10h] [rbp-80h]
  unsigned __int64 v123; // [rsp+10h] [rbp-80h]
  __int64 v124; // [rsp+10h] [rbp-80h]
  __int64 v125; // [rsp+10h] [rbp-80h]
  __int64 v126; // [rsp+10h] [rbp-80h]
  __int64 v127; // [rsp+10h] [rbp-80h]
  __int64 v128; // [rsp+10h] [rbp-80h]
  __int64 v129; // [rsp+10h] [rbp-80h]
  __int64 v130; // [rsp+10h] [rbp-80h]
  __int64 v131; // [rsp+18h] [rbp-78h]
  _BYTE *v132; // [rsp+18h] [rbp-78h]
  __int64 v133; // [rsp+18h] [rbp-78h]
  __int64 v134; // [rsp+18h] [rbp-78h]
  __int64 v135; // [rsp+18h] [rbp-78h]
  __int64 v136; // [rsp+18h] [rbp-78h]
  _BYTE *v137; // [rsp+18h] [rbp-78h]
  _BYTE *v138; // [rsp+18h] [rbp-78h]
  __int64 v139; // [rsp+18h] [rbp-78h]
  __int64 v140; // [rsp+18h] [rbp-78h]
  __int64 v141; // [rsp+18h] [rbp-78h]
  int v142; // [rsp+20h] [rbp-70h]
  __int16 v143; // [rsp+24h] [rbp-6Ch]
  __int16 v144; // [rsp+26h] [rbp-6Ah]
  int v145; // [rsp+28h] [rbp-68h]
  int v146; // [rsp+2Ch] [rbp-64h]
  __int16 v147; // [rsp+30h] [rbp-60h]
  __int64 v148; // [rsp+30h] [rbp-60h]
  __int64 v150; // [rsp+38h] [rbp-58h]
  __int64 v151; // [rsp+38h] [rbp-58h]
  __int64 v152; // [rsp+38h] [rbp-58h]
  __int64 v153; // [rsp+38h] [rbp-58h]
  __int64 v154; // [rsp+38h] [rbp-58h]
  __int64 v155; // [rsp+38h] [rbp-58h]
  _BYTE *v156; // [rsp+38h] [rbp-58h]
  __int64 v157; // [rsp+38h] [rbp-58h]
  int v158; // [rsp+38h] [rbp-58h]
  __int64 v159; // [rsp+38h] [rbp-58h]
  __int64 v160; // [rsp+38h] [rbp-58h]
  _BYTE *v161; // [rsp+38h] [rbp-58h]
  __int64 v162; // [rsp+38h] [rbp-58h]
  __int64 v163; // [rsp+38h] [rbp-58h]
  __int64 v164; // [rsp+38h] [rbp-58h]
  __int64 v165; // [rsp+38h] [rbp-58h]
  __int64 v166; // [rsp+38h] [rbp-58h]
  __int64 v167; // [rsp+38h] [rbp-58h]
  __int64 v168; // [rsp+38h] [rbp-58h]
  int v169; // [rsp+48h] [rbp-48h] BYREF
  _BYTE v170[4]; // [rsp+4Ch] [rbp-44h] BYREF
  __int64 v171; // [rsp+50h] [rbp-40h] BYREF
  __int64 v172[7]; // [rsp+58h] [rbp-38h] BYREF

  v7 = a1;
  v8 = (_DWORD *)a2;
  v9 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v169 = 0;
  v10 = *(_QWORD *)a1;
  v171 = v9;
  v142 = *(_DWORD *)(a1 + 28);
  v143 = *(_WORD *)(a1 + 32);
  v145 = *(_DWORD *)(a1 + 36);
  v144 = *(_WORD *)(a1 + 40);
  v146 = *(_DWORD *)(a1 + 44);
  v147 = *(_WORD *)(a1 + 48);
  if ( a2 )
    *(_DWORD *)a2 = 0;
  if ( a3 )
    *a3 = 0;
  if ( (unsigned int)sub_8D2310(v10) )
  {
    a2 = 0;
    v11 = v10;
    v15 = sub_72D2E0(v10, 0);
  }
  else
  {
    v11 = v10;
    v20 = sub_8D3410(v10);
    v15 = v10;
    if ( !v20 )
    {
      v11 = v10;
      if ( (unsigned int)sub_8D2690(v10) )
      {
        v15 = sub_72C570();
      }
      else
      {
        v11 = v10;
        v15 = sub_73D720(v10);
      }
    }
  }
  v16 = *(_BYTE *)(v7 + 24);
  if ( v16 == 3 )
  {
    v28 = *(_DWORD **)(v7 + 56);
    if ( (v28[42] & 0x401000) == 0 )
    {
      v29 = (_QWORD *)*((_QWORD *)v28 + 27);
      if ( v29 )
      {
        if ( *v29 )
        {
          if ( (*(_BYTE *)(*(_QWORD *)v28 + 81LL) & 2) == 0 )
          {
            v152 = v15;
            v45 = sub_892240(*(_QWORD *)v28, a2);
            v15 = v152;
            if ( (*(_BYTE *)(v45 + 80) & 1) == 0 && (*(_BYTE *)(*(_QWORD *)(v45 + 16) + 28LL) & 1) == 0 )
            {
              sub_8AA320(v45, 0, 1);
              v15 = v152;
            }
          }
        }
      }
    }
    if ( !v8
      || dword_4F077C4 != 2
      || *(_BYTE *)(v7 + 24) != 3
      || (a2 = 1, v150 = v15, v30 = *(_QWORD *)(v7 + 56), v31 = sub_6EA380(v30, 1, 0, 0), v15 = v150, (v27 = v31) == 0) )
    {
LABEL_12:
      if ( !v169 )
      {
LABEL_13:
        v17 = *(_QWORD *)v7;
        *(_BYTE *)(v7 + 25) &= 0xFCu;
        *(_QWORD *)v7 = v15;
        *(_QWORD *)(v7 + 8) = v17;
LABEL_14:
        v18 = (_QWORD *)v7;
        goto LABEL_15;
      }
      goto LABEL_27;
    }
    v35 = v30;
    if ( *(_BYTE *)(v31 + 173) != 12 )
    {
      if ( !*(_BYTE *)(qword_4D03C50 + 16LL)
        || (*(_BYTE *)(v30 - 8) & 1) == 0 && (v32 = dword_4F07270, dword_4F07270[0] == unk_4F073B8)
        || (*(_BYTE *)(v30 + 174) & 8) != 0
        || (v163 = v31, v95 = sub_73E830(v30), v27 = v163, !v95) )
      {
LABEL_76:
        if ( *(_QWORD *)(v27 + 144) )
        {
          v154 = v27;
          v172[0] = sub_724DC0(v30, a2, v32, v33, v35, v34);
          sub_72A510(v154, v172[0]);
          v52 = v172[0];
          *(_QWORD *)(v172[0] + 144) = 0;
          v155 = sub_73A460(v52);
          sub_724E30(v172);
          v27 = v155;
          if ( !v155 )
            goto LABEL_31;
        }
        goto LABEL_78;
      }
      if ( *(_BYTE *)(v95 + 24) == 3 )
      {
        v32 = *(_DWORD **)(v7 + 64);
        *(_QWORD *)(v95 + 64) = v32;
        *(_QWORD *)(v7 + 64) = 0;
      }
      v7 = v95;
LABEL_75:
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
      {
        v87 = sub_740630(v27);
        *(_QWORD *)(v87 + 144) = v7;
        v27 = v87;
        *(_DWORD *)(v7 + 28) = v142;
        *(_WORD *)(v7 + 32) = v143;
        *(_DWORD *)(v7 + 36) = v145;
        *(_WORD *)(v7 + 40) = v144;
        *(_DWORD *)(v7 + 44) = v146;
        *(_WORD *)(v7 + 48) = v147;
        goto LABEL_78;
      }
      goto LABEL_76;
    }
    v169 = 1;
    if ( (*(_BYTE *)(v7 - 8) & 1) != 0 && (*(_BYTE *)(v30 - 8) & 1) == 0 )
    {
      v101 = sub_867C20();
      sub_7296F0(v101, v172);
      v7 = sub_73E830(v30);
      sub_729730(LODWORD(v172[0]));
      v15 = v150;
      goto LABEL_12;
    }
    goto LABEL_43;
  }
  if ( v16 == 1 )
  {
    v21 = 0;
    v18 = *(_QWORD **)(v7 + 72);
    if ( v8 )
      v21 = v170;
    v22 = *(_BYTE *)(v7 + 56);
    v23 = v18[2];
    v24 = v21;
    if ( (*(_BYTE *)(v7 + 58) & 1) == 0 )
    {
      switch ( v22 )
      {
        case 3u:
          if ( !v21 )
            goto LABEL_12;
          v30 = v7;
          v159 = v15;
          v82 = sub_7196D0(v7, a2, v170, v12, v21, v23);
          v15 = v159;
          v27 = v82;
          if ( v82 )
            goto LABEL_131;
          if ( *((_BYTE *)v18 + 24) != 2 )
            goto LABEL_12;
          v98 = v18[7];
          if ( *(_BYTE *)(v98 + 173) == 12 )
          {
            v113 = *(_QWORD *)v7;
            *(_BYTE *)(v7 + 25) &= 0xFCu;
            v169 = 1;
            *(_QWORD *)(v7 + 8) = v113;
            *(_QWORD *)v7 = v159;
            goto LABEL_27;
          }
          if ( HIDWORD(qword_4F077B4) )
          {
            a2 = (unsigned __int64)v172;
            v99 = sub_72EA80(v98, v172, 1);
            v15 = v159;
            if ( v99 )
            {
              v30 = v172[0];
              v100 = sub_6EA7C0(v172[0]);
              v15 = v159;
              v27 = v100;
              if ( v100 )
              {
                if ( HIDWORD(qword_4F077B4) && (*(_BYTE *)(v172[0] + 174) & 8) != 0 )
                {
LABEL_131:
                  v83 = *(_QWORD *)v7;
                  *(_BYTE *)(v7 + 25) &= 0xFCu;
                  *(_QWORD *)v7 = v15;
                  *(_QWORD *)(v7 + 8) = v83;
                  goto LABEL_75;
                }
              }
            }
          }
          goto LABEL_12;
        case 4u:
          if ( !v21 )
            goto LABEL_12;
          v30 = v7;
          v160 = v15;
          v86 = sub_7196D0(v7, a2, v170, v12, v21, v23);
          v15 = v160;
          v27 = v86;
          if ( !v86 )
            goto LABEL_12;
          goto LABEL_131;
        case 6u:
          v127 = v15;
          v137 = v21;
          v84 = sub_6ED3D0(*(_QWORD *)(v7 + 72), v21, 0, a4, v21, v23);
          *(_QWORD *)(v7 + 72) = v84;
          v85 = v84;
          if ( !v137 || !a4 || *(_BYTE *)(v84 + 24) != 2 )
          {
            sub_73D8E0(v7, 5, v127, 0, v84);
            *(_QWORD *)(v7 + 8) = v10;
            goto LABEL_54;
          }
          sub_72A510(*(_QWORD *)(v84 + 56), v171);
          sub_6E5170(v171, v127, 0, 0, 0, 0, 1, (__int64)v172, a4);
          a2 = 5;
          v30 = v7;
          v164 = sub_73A460(v171);
          sub_73D8E0(v7, 5, v127, 0, v85);
          v27 = v164;
          *(_QWORD *)(v7 + 8) = v10;
          if ( !v164 )
            goto LABEL_54;
          goto LABEL_75;
        case 7u:
          if ( !word_4D04898 )
            goto LABEL_12;
          if ( *((_BYTE *)v18 + 24) != 5 || (*(_BYTE *)(v18[7] + 49LL) & 0x10) == 0 )
          {
            v122 = v21;
            v133 = v15;
            v56 = sub_8DF8D0(v10, *v18);
            v15 = v133;
            v24 = v122;
            if ( !v56 )
              goto LABEL_12;
          }
          goto LABEL_90;
        case 8u:
LABEL_90:
          if ( !v24 )
            goto LABEL_12;
          if ( (*((_BYTE *)v18 + 25) & 3) == 0 )
            goto LABEL_12;
          if ( (*(_BYTE *)(v7 + 58) & 2) != 0 )
            goto LABEL_12;
          v57 = *v18;
          v134 = v15;
          v58 = sub_8DF8D0(v10, *v18);
          v15 = v134;
          if ( !v58 )
            goto LABEL_12;
          v63 = sub_7196D0(v18, v57, v59, v60, v61, v62);
          v15 = v134;
          if ( !v63 )
            goto LABEL_12;
          v123 = v134;
          v135 = v63;
          sub_72A510(v63, v171);
          v30 = v171;
          a2 = v123;
          sub_6E5170(v171, v123, 0, 0, 0, 0, 1, (__int64)v172, a4);
          v15 = v123;
          if ( !LODWORD(v172[0]) )
            goto LABEL_85;
          v27 = v135;
          goto LABEL_75;
        case 0xCu:
        case 0xDu:
        case 0x6Eu:
          goto LABEL_69;
        case 0xEu:
          if ( !word_4D04898 || !v21 )
            goto LABEL_69;
          v30 = v7;
          v153 = v15;
          v50 = sub_7196D0(v7, a2, v170, v12, v21, v23);
          goto LABEL_74;
        case 0x19u:
          v67 = a4;
          a2 = (unsigned __int64)v21;
          v30 = *(_QWORD *)(v7 + 72);
          v156 = v21;
          v68 = sub_6ED3D0(v30, v21, 0, v67, v21, v23);
          v35 = (__int64)v156;
          *(_QWORD *)(v7 + 72) = v68;
          if ( !v156 || *(_BYTE *)(v68 + 24) != 2 )
          {
            *(_BYTE *)(v7 + 25) &= 0xFCu;
            *(_QWORD *)v7 = *(_QWORD *)v68;
            goto LABEL_54;
          }
          v27 = *(_QWORD *)(v68 + 56);
          *(_BYTE *)(v7 + 25) &= 0xFCu;
          *(_QWORD *)v7 = *(_QWORD *)v68;
          if ( !v27 )
            goto LABEL_54;
          goto LABEL_75;
        case 0x47u:
        case 0x48u:
          v18[2] = 0;
          v121 = v15;
          v114 = v22;
          v116 = v23;
          v132 = v21;
          v54 = sub_6ED3D0(v18, v21, 0, a4, v21, v23);
          v55 = sub_6ED3D0(v116, v132, 0, a4, v132, v116);
          *(_QWORD *)(v7 + 72) = v54;
          v15 = v121;
          *(_QWORD *)(v54 + 16) = v55;
          if ( !v132 || !a4 || *(_BYTE *)(v54 + 24) != 2 || *(_BYTE *)(v55 + 24) != 2 )
            goto LABEL_69;
          a2 = *(_QWORD *)(v54 + 56);
          sub_6E5200(v114, a2, *(_QWORD *)(v55 + 56), *(_QWORD *)v54, v171, (__int64)v172, (__int64)&v169, a4);
          v15 = v121;
          if ( v169 )
            goto LABEL_43;
LABEL_85:
          v153 = v15;
LABEL_73:
          v30 = v171;
          v50 = sub_73A460(v171);
LABEL_74:
          v27 = v50;
          v51 = *(_QWORD *)v7;
          *(_BYTE *)(v7 + 25) &= 0xFCu;
          *(_QWORD *)(v7 + 8) = v51;
          *(_QWORD *)v7 = v153;
          if ( v27 )
            goto LABEL_75;
          goto LABEL_54;
        case 0x5Cu:
          if ( !v21 || *((_BYTE *)v18 + 24) != 2 || *(_BYTE *)(v23 + 24) != 2 || dword_4D04964 )
            goto LABEL_12;
          v69 = v18[7];
          v124 = v15;
          v136 = v171;
          v157 = *(_QWORD *)(v23 + 56);
          if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(v69 + 128)) )
          {
            v74 = v69;
            v69 = v157;
            v157 = v74;
          }
          v75 = sub_717530(v69, 0, v70, v71, v72, v73);
          v15 = v124;
          if ( !v75 )
            goto LABEL_12;
          v76 = sub_8D2780(*(_QWORD *)(v157 + 128));
          v15 = v124;
          if ( !v76 )
            goto LABEL_12;
          if ( *(_BYTE *)(v157 + 173) != 1 )
            goto LABEL_12;
          for ( i = sub_8D46C0(*(_QWORD *)(v69 + 128)); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          v117 = v124;
          v125 = i;
          v78 = sub_8D29E0(i);
          v79 = v125;
          v15 = v117;
          if ( !v78 )
            goto LABEL_12;
          v126 = v117;
          v118 = v79;
          v80 = *(_QWORD *)(v69 + 184);
          v81 = sub_620FA0(v157, v172);
          v15 = v126;
          if ( LODWORD(v172[0]) || v81 < 0 || *(_QWORD *)(v80 + 176) <= (unsigned __int64)v81 )
            goto LABEL_12;
          v158 = *(unsigned __int8 *)(v118 + 160);
          a2 = *(char *)(*(_QWORD *)(v80 + 184) + v81) & (unsigned __int64)~(-1LL << dword_4F06BA0);
          sub_72BAF0(v136, a2, *(unsigned __int8 *)(v118 + 160));
          v15 = v126;
          if ( byte_4B6DF90[v158] )
          {
            a2 = dword_4F06BA0;
            sub_6215A0((__int16 *)(v136 + 176), dword_4F06BA0);
            v15 = v126;
          }
          goto LABEL_85;
        case 0x5Eu:
        case 0x5Fu:
          if ( !word_4D04898 )
            goto LABEL_12;
          if ( !v21 )
            goto LABEL_12;
          a2 = v171;
          v153 = v15;
          v49 = sub_71ABE0(v7, v171);
          v15 = v153;
          if ( !v49 )
            goto LABEL_12;
          goto LABEL_73;
        case 0x60u:
          v64 = *(_BYTE *)(v7 + 25);
          *(_BYTE *)(v7 + 25) = v64 & 0xFC;
          if ( !word_4D04898 || !v21 || (a2 = v171, v166 = v15, v107 = sub_719770(v7, v171, 0, 0), v15 = v166, !v107) )
          {
            v65 = v64 & 3 | *(_BYTE *)(v7 + 25) & 0xFC;
            v66 = v169;
            *(_BYTE *)(v7 + 25) = v65;
            if ( !v66 )
              goto LABEL_13;
            goto LABEL_27;
          }
          v30 = v171;
          v108 = sub_73A460(v171);
          *(_QWORD *)(v7 + 8) = v10;
          v27 = v108;
          *(_QWORD *)v7 = v166;
          if ( v108 )
            goto LABEL_75;
          goto LABEL_54;
        case 0x64u:
        case 0x65u:
          v115 = v21;
          v120 = (__int64 *)v18[2];
          v131 = v15;
          v46 = sub_731EE0(*(_QWORD *)(v7 + 72));
          v15 = v131;
          if ( v46 )
          {
            v47 = *v120;
            *((_BYTE *)v120 + 25) &= 0xFCu;
            *v120 = v131;
            v120[1] = v47;
          }
          else
          {
            a2 = (unsigned __int64)v115;
            v102 = sub_6ED3D0(v120, v115, 0, a4, v115, v120);
            v15 = v131;
            v18[2] = v102;
            if ( v115 && *(_BYTE *)(v102 + 24) == 2 )
            {
              v30 = (__int64)v18;
              v165 = v131;
              v141 = v102;
              v103 = sub_6ED2E0();
              v15 = v165;
              v104 = v141;
              if ( v103
                || dword_4F077BC
                && dword_4F077C4 == 2
                && (unk_4F07778 > 201102 || dword_4F07774)
                && !(_DWORD)qword_4F077B4
                && (a2 = 0, v30 = (__int64)v18, v105 = sub_731770(v18, 0), v15 = v165, v104 = v141, !v105) )
              {
                v27 = *(_QWORD *)(v104 + 56);
                v106 = *(_QWORD *)v7;
                *(_QWORD *)v7 = v15;
                *(_BYTE *)(v7 + 25) &= 0xFCu;
                *(_QWORD *)(v7 + 8) = v106;
                if ( !v27 )
                  goto LABEL_54;
                v32 = dword_4F07270;
                if ( dword_4F07270[0] == unk_4F073B8 )
                {
                  if ( qword_4F04C50 )
                  {
                    v7 = *(_QWORD *)(v27 + 144);
                    if ( !v7 )
                    {
LABEL_78:
                      *v8 = 1;
                      if ( a3 )
                        goto LABEL_30;
                      goto LABEL_79;
                    }
                  }
                }
                goto LABEL_75;
              }
            }
          }
LABEL_69:
          v48 = *(_QWORD *)v7;
          *(_BYTE *)(v7 + 25) &= 0xFCu;
          *(_QWORD *)v7 = v15;
          *(_QWORD *)(v7 + 8) = v48;
          goto LABEL_54;
        case 0x74u:
          if ( !v21 || *((_BYTE *)v18 + 24) != 2 )
            goto LABEL_102;
          if ( !a3 )
          {
            if ( !v169 )
              goto LABEL_15;
            sub_70FD90(*(_QWORD *)(v7 + 72), v171);
            goto LABEL_142;
          }
          v27 = v18[7];
          if ( !v27 )
          {
            if ( !v169 )
              goto LABEL_31;
            sub_70FD90(0, v171);
            goto LABEL_28;
          }
          if ( !*(_QWORD *)(v27 + 144) )
            goto LABEL_29;
          v167 = v18[7];
          v172[0] = sub_724DC0(v11, a2, v170, v12, v21, v23);
          sub_72A510(v167, v172[0]);
          v109 = v172[0];
          *(_QWORD *)(v172[0] + 144) = 0;
          v168 = sub_73A460(v109);
          sub_724E30(v172);
          v27 = v168;
          if ( v168 )
            goto LABEL_29;
          goto LABEL_31;
        default:
          goto LABEL_12;
      }
    }
    if ( v22 != 103 )
    {
      if ( v22 == 91 )
      {
        v148 = v15;
        v96 = sub_6ED3D0(v18[2], v21, 0, a4, v21, v23);
        v15 = v148;
        v18[2] = v96;
        v25 = v169;
        goto LABEL_25;
      }
LABEL_24:
      v25 = v169;
LABEL_25:
      v26 = *(_QWORD *)v7;
      *(_BYTE *)(v7 + 58) &= ~1u;
      *(_BYTE *)(v7 + 25) &= 0xFCu;
      *(_QWORD *)(v7 + 8) = v26;
      *(_QWORD *)v7 = v15;
      goto LABEL_26;
    }
    v88 = *(_QWORD *)(v23 + 16);
    v18[2] = 0;
    v89 = *(_BYTE *)(v23 + 24) == 8;
    *(_QWORD *)(v23 + 16) = 0;
    if ( !v89 )
    {
      a2 = (unsigned __int64)v21;
      v119 = v88;
      v128 = v15;
      v138 = v21;
      v90 = sub_6ED3D0(v23, v21, 0, a4, v21, v23);
      v88 = v119;
      v15 = v128;
      v24 = v138;
      v23 = v90;
    }
    if ( *(_BYTE *)(v88 + 24) != 8 )
    {
      v91 = a4;
      a2 = (unsigned __int64)v24;
      v129 = v23;
      v139 = v15;
      v161 = v24;
      v92 = sub_6ED3D0(v88, v24, 0, v91, v24, v23);
      v23 = v129;
      v15 = v139;
      v24 = v161;
      v88 = v92;
    }
    v18[2] = v23;
    *(_QWORD *)(v23 + 16) = v88;
    if ( !v24 || *((_BYTE *)v18 + 24) != 2 )
      goto LABEL_24;
    v93 = v18[7];
    if ( (*(_BYTE *)(v23 + 24) != 2 || *(_BYTE *)(v88 + 24) != 2) && *(_BYTE *)(v93 + 173) != 12 )
      goto LABEL_24;
    v130 = v88;
    v140 = v23;
    v162 = v15;
    v94 = sub_70FCE0(v93);
    v15 = v162;
    if ( v94 && *(_BYTE *)(v140 + 24) == 2 && *(_BYTE *)(v130 + 24) == 2 )
    {
      v30 = v93;
      v110 = sub_711520(v93, a2);
      v34 = v140;
      v111 = v130;
      v89 = v110 == 0;
      v112 = *(_QWORD *)v7;
      if ( v89 )
        v111 = v140;
      v27 = *(_QWORD *)(v111 + 56);
      *(_BYTE *)(v7 + 58) &= ~1u;
      *(_BYTE *)(v7 + 25) &= 0xFCu;
      *(_QWORD *)(v7 + 8) = v112;
      *(_QWORD *)v7 = v162;
      if ( v27 )
        goto LABEL_75;
      v25 = v169;
LABEL_26:
      if ( !v25 )
        goto LABEL_14;
      goto LABEL_27;
    }
    if ( *(_BYTE *)(v93 + 173) != 12
      && *(_BYTE *)(*(_QWORD *)(v140 + 56) + 173LL) != 12
      && *(_BYTE *)(*(_QWORD *)(v130 + 56) + 173LL) != 12 )
    {
      goto LABEL_24;
    }
    *(_BYTE *)(v7 + 58) &= ~1u;
    v169 = 1;
LABEL_43:
    v36 = *(_QWORD *)v7;
    *(_BYTE *)(v7 + 25) &= 0xFCu;
    *(_QWORD *)v7 = v15;
    *(_QWORD *)(v7 + 8) = v36;
LABEL_27:
    sub_70FD90(v7, v171);
    if ( a3 )
    {
LABEL_28:
      v27 = sub_73A460(v171);
      if ( !v27 )
      {
LABEL_31:
        v18 = 0;
        goto LABEL_15;
      }
LABEL_29:
      *v8 = 1;
LABEL_30:
      *a3 = v27;
      goto LABEL_31;
    }
LABEL_142:
    v27 = v171;
    if ( !v171 )
      goto LABEL_31;
    *v8 = 1;
LABEL_79:
    v53 = sub_73A720(v27);
    *(_QWORD *)(v53 + 8) = v10;
    v18 = (_QWORD *)v53;
    goto LABEL_15;
  }
  if ( v16 != 5 )
  {
    if ( v16 == 10 )
      sub_721090(v11);
    if ( v16 != 27 )
      goto LABEL_12;
    v30 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16LL);
    if ( (*(_BYTE *)(v7 + 64) & 1) == 0 )
      v30 = *(_QWORD *)(v30 + 16);
    v37 = 0;
    if ( v8 )
      v37 = v172;
    a2 = (unsigned __int64)v37;
    v38 = v37;
    v39 = sub_6ED3D0(v30, v37, 0, a4, v13, v14);
    if ( (*(_BYTE *)(v7 + 64) & 1) != 0 )
    {
      *(_QWORD *)(v39 + 16) = *(_QWORD *)(v30 + 16);
      v40 = *(_QWORD *)(v7 + 56);
    }
    else
    {
      v40 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16LL);
    }
    *(_QWORD *)(v40 + 16) = v39;
    v32 = *(_DWORD **)v39;
    v33 = *(_QWORD *)v7;
    if ( !v38 || *(_BYTE *)(v39 + 24) != 2 )
    {
      *(_BYTE *)(v7 + 25) &= 0xFCu;
      *(_QWORD *)(v7 + 8) = v33;
      *(_QWORD *)v7 = v32;
      goto LABEL_54;
    }
    v27 = *(_QWORD *)(v39 + 56);
    *(_BYTE *)(v7 + 25) &= 0xFCu;
    *(_QWORD *)(v7 + 8) = v33;
    *(_QWORD *)v7 = v32;
    if ( !v27 )
    {
LABEL_54:
      if ( !v169 )
        goto LABEL_14;
      goto LABEL_27;
    }
    goto LABEL_75;
  }
  v41 = *(_QWORD *)(v7 + 56);
  if ( (*(_DWORD *)(v41 + 48) & 0x50000FF) != 3 )
    goto LABEL_12;
  v18 = *(_QWORD **)(v41 + 56);
  if ( (*((_BYTE *)v18 + 25) & 3) != 0 )
    goto LABEL_12;
  v151 = v15;
  v42 = sub_7307F0(v41);
  v15 = v151;
  if ( !v42 )
    goto LABEL_12;
  if ( *v18 != v151 && !(unsigned int)sub_8D97D0(*v18, v151, 64, v43, v44) )
  {
    v15 = v151;
    goto LABEL_12;
  }
  *((_BYTE *)v18 + 27) |= 8u;
  if ( (*(_BYTE *)(v7 - 8) & 1) != 0 )
  {
    *(_BYTE *)(v7 + 24) = 38;
    v97 = qword_4F06BB0;
    qword_4F06BB0 = v7;
    *(_QWORD *)(v7 + 80) = v97;
  }
LABEL_102:
  if ( v169 )
  {
    v7 = (__int64)v18;
    goto LABEL_27;
  }
LABEL_15:
  sub_724E30(&v171);
  return v18;
}
