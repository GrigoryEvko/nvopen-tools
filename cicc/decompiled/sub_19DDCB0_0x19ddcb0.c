// Function: sub_19DDCB0
// Address: 0x19ddcb0
//
__int64 __fastcall sub_19DDCB0(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        double a9)
{
  __int64 *v9; // r13
  __int64 *v11; // rbx
  int v12; // edx
  __int64 *v13; // r15
  __int64 v14; // rbx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rax
  _BYTE *v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned __int8 *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  char v27; // al
  char v28; // al
  unsigned __int64 v29; // r14
  unsigned int v30; // eax
  __int64 v31; // rcx
  __int64 v32; // r10
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // r9
  __int64 **v37; // rbx
  __int64 **v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // rax
  bool v41; // al
  char v43; // al
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // r13
  __int64 v47; // rax
  unsigned __int64 v48; // r15
  unsigned int v49; // eax
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rcx
  int v53; // eax
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  __int64 v57; // rax
  int v58; // eax
  unsigned int v59; // eax
  __int64 v60; // rdi
  __int64 v61; // rsi
  __int64 v62; // r9
  unsigned __int64 v63; // r8
  _QWORD *v64; // rax
  _QWORD *v65; // rax
  unsigned int v66; // eax
  __int64 v67; // rsi
  __int64 v68; // r8
  unsigned __int64 v69; // rcx
  __int64 v70; // rax
  unsigned __int64 v71; // r13
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 *v74; // r15
  __int64 v75; // rsi
  int v76; // edi
  __int64 *v77; // r14
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  __int64 *v82; // rbx
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 v87; // rax
  int v88; // eax
  _QWORD *v89; // rax
  __int64 v90; // rax
  unsigned __int64 v91; // r14
  __int64 v92; // rax
  int v93; // eax
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 v96; // rdx
  int v97; // edi
  __int64 v98; // rax
  unsigned __int64 v99; // r9
  __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rdx
  unsigned __int8 *v104; // rsi
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned int v107; // esi
  int v108; // eax
  __int64 v109; // rax
  unsigned int v110; // esi
  int v111; // eax
  _QWORD *v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  _QWORD *v116; // rax
  __int64 v117; // rax
  __int64 v118; // [rsp+0h] [rbp-150h]
  unsigned __int64 v119; // [rsp+8h] [rbp-148h]
  __int64 v120; // [rsp+8h] [rbp-148h]
  __int64 v121; // [rsp+10h] [rbp-140h]
  __int64 v122; // [rsp+10h] [rbp-140h]
  unsigned __int64 v123; // [rsp+10h] [rbp-140h]
  __int64 v124; // [rsp+18h] [rbp-138h]
  __int64 v125; // [rsp+18h] [rbp-138h]
  __int64 v126; // [rsp+18h] [rbp-138h]
  __int64 v127; // [rsp+20h] [rbp-130h]
  __int64 v128; // [rsp+20h] [rbp-130h]
  unsigned __int64 v129; // [rsp+20h] [rbp-130h]
  __int64 v130; // [rsp+20h] [rbp-130h]
  __int64 v131; // [rsp+20h] [rbp-130h]
  __int64 v132; // [rsp+20h] [rbp-130h]
  __int64 v133; // [rsp+20h] [rbp-130h]
  unsigned __int64 v135; // [rsp+28h] [rbp-128h]
  __int64 v136; // [rsp+28h] [rbp-128h]
  __int64 v137; // [rsp+28h] [rbp-128h]
  __int64 v138; // [rsp+28h] [rbp-128h]
  __int64 v139; // [rsp+28h] [rbp-128h]
  unsigned __int64 v140; // [rsp+28h] [rbp-128h]
  unsigned __int64 v141; // [rsp+28h] [rbp-128h]
  unsigned __int64 v142; // [rsp+28h] [rbp-128h]
  unsigned __int64 v143; // [rsp+28h] [rbp-128h]
  unsigned __int64 v144; // [rsp+28h] [rbp-128h]
  unsigned __int64 v145; // [rsp+28h] [rbp-128h]
  unsigned __int64 v146; // [rsp+28h] [rbp-128h]
  __int64 v148; // [rsp+30h] [rbp-120h]
  __int64 v149; // [rsp+30h] [rbp-120h]
  __int64 v150; // [rsp+30h] [rbp-120h]
  __int64 v151; // [rsp+30h] [rbp-120h]
  __int64 v152; // [rsp+30h] [rbp-120h]
  __int64 v153; // [rsp+30h] [rbp-120h]
  __int64 v154; // [rsp+30h] [rbp-120h]
  __int64 v155; // [rsp+30h] [rbp-120h]
  __int64 v156; // [rsp+30h] [rbp-120h]
  __int64 v157; // [rsp+30h] [rbp-120h]
  unsigned __int64 v159; // [rsp+38h] [rbp-118h]
  unsigned int v160; // [rsp+38h] [rbp-118h]
  __int64 v161; // [rsp+38h] [rbp-118h]
  __int64 v162; // [rsp+38h] [rbp-118h]
  __int64 v163; // [rsp+38h] [rbp-118h]
  __int64 v164; // [rsp+38h] [rbp-118h]
  __int64 v165; // [rsp+38h] [rbp-118h]
  __int64 v166; // [rsp+38h] [rbp-118h]
  __int64 v167; // [rsp+38h] [rbp-118h]
  __int64 v168; // [rsp+38h] [rbp-118h]
  __int64 v169; // [rsp+38h] [rbp-118h]
  unsigned __int64 v170; // [rsp+38h] [rbp-118h]
  __int64 *v171; // [rsp+38h] [rbp-118h]
  unsigned __int64 v172; // [rsp+38h] [rbp-118h]
  unsigned __int64 v173; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v175; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v176[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v177; // [rsp+70h] [rbp-E0h]
  unsigned __int8 *v178[2]; // [rsp+80h] [rbp-D0h] BYREF
  __int16 v179; // [rsp+90h] [rbp-C0h]
  _BYTE *v180; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v181; // [rsp+A8h] [rbp-A8h]
  _BYTE v182[32]; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned __int8 *v183; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v184; // [rsp+D8h] [rbp-78h]
  __int64 *v185; // [rsp+E0h] [rbp-70h]
  __int64 v186; // [rsp+E8h] [rbp-68h]
  __int64 v187; // [rsp+F0h] [rbp-60h]
  int v188; // [rsp+F8h] [rbp-58h]
  __int64 v189; // [rsp+100h] [rbp-50h]
  __int64 v190; // [rsp+108h] [rbp-48h]

  v9 = a4;
  v11 = a1;
  v12 = *(_DWORD *)(a2 + 20);
  v180 = v182;
  v181 = 0x400000000LL;
  v13 = (__int64 *)(a2 + 24 * (1LL - (v12 & 0xFFFFFFF)));
  if ( (__int64 *)a2 == v13 )
  {
    v18 = v182;
  }
  else
  {
    do
    {
      v14 = sub_146F1B0(a1[3], *v13);
      v17 = (unsigned int)v181;
      if ( (unsigned int)v181 >= HIDWORD(v181) )
      {
        sub_16CD150((__int64)&v180, v182, 0, 8, v15, v16);
        v17 = (unsigned int)v181;
      }
      v13 += 3;
      *(_QWORD *)&v180[8 * v17] = v14;
      LODWORD(v181) = v181 + 1;
    }
    while ( (__int64 *)a2 != v13 );
    v11 = a1;
    v18 = v180;
    v9 = a4;
  }
  v19 = 8LL * a3;
  *(_QWORD *)&v18[v19] = sub_146F1B0(v11[3], (__int64)v9);
  if ( (unsigned __int8)sub_14C2730(v9, v11[1], 0, *v11, a2, v11[2]) )
  {
    v44 = v11[1];
    v45 = *v9;
    v46 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v45 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v90 = *(_QWORD *)(v45 + 32);
          v45 = *(_QWORD *)(v45 + 24);
          v46 *= v90;
          continue;
        case 1:
          v70 = 16;
          goto LABEL_76;
        case 2:
          v70 = 32;
          goto LABEL_76;
        case 3:
        case 9:
          v70 = 64;
          goto LABEL_76;
        case 4:
          v70 = 80;
          goto LABEL_76;
        case 5:
        case 6:
          v70 = 128;
          goto LABEL_76;
        case 7:
          v88 = sub_15A9520(v44, 0);
          v44 = v11[1];
          v70 = (unsigned int)(8 * v88);
          goto LABEL_76;
        case 0xB:
          v70 = *(_DWORD *)(v45 + 8) >> 8;
          goto LABEL_76;
        case 0xD:
          v89 = (_QWORD *)sub_15A9930(v44, v45);
          v44 = v11[1];
          v70 = 8LL * *v89;
          goto LABEL_76;
        case 0xE:
          v128 = v11[1];
          v124 = *(_QWORD *)(v45 + 24);
          v138 = *(_QWORD *)(v45 + 32);
          v91 = (unsigned int)sub_15A9FE0(v44, v124);
          v92 = sub_127FA20(v128, v124);
          v44 = v11[1];
          v70 = 8 * v91 * v138 * ((v91 + ((unsigned __int64)(v92 + 7) >> 3) - 1) / v91);
          goto LABEL_76;
        case 0xF:
          v93 = sub_15A9520(v44, *(_DWORD *)(v45 + 8) >> 8);
          v44 = v11[1];
          v70 = (unsigned int)(8 * v93);
LABEL_76:
          v71 = v70 * v46;
          v29 = 1;
          v72 = **(_QWORD **)(a2 + 24 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
          while ( 2 )
          {
            switch ( *(_BYTE *)(v72 + 8) )
            {
              case 1:
                v73 = 16;
                goto LABEL_79;
              case 2:
                v73 = 32;
                goto LABEL_79;
              case 3:
              case 9:
                v73 = 64;
                goto LABEL_79;
              case 4:
                v73 = 80;
                goto LABEL_79;
              case 5:
              case 6:
                v73 = 128;
                goto LABEL_79;
              case 7:
                v73 = 8 * (unsigned int)sub_15A9520(v44, 0);
                goto LABEL_79;
              case 0xB:
                v73 = *(_DWORD *)(v72 + 8) >> 8;
                goto LABEL_79;
              case 0xD:
                v73 = 8LL * *(_QWORD *)sub_15A9930(v44, v72);
                goto LABEL_79;
              case 0xE:
                v121 = *(_QWORD *)(v72 + 24);
                v139 = *(_QWORD *)(v72 + 32);
                v129 = (unsigned int)sub_15A9FE0(v44, v121);
                v73 = 8 * v139 * v129 * ((v129 + ((unsigned __int64)(sub_127FA20(v44, v121) + 7) >> 3) - 1) / v129);
                goto LABEL_79;
              case 0xF:
                v73 = 8 * (unsigned int)sub_15A9520(v44, *(_DWORD *)(v72 + 8) >> 8);
LABEL_79:
                if ( v71 < v73 * v29 )
                {
                  v74 = (__int64 *)&v180[v19];
                  *v74 = sub_14747F0(
                           v11[3],
                           *v74,
                           **(_QWORD **)(a2 + 24 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                           0);
                }
                goto LABEL_7;
              case 0x10:
                v94 = *(_QWORD *)(v72 + 32);
                v72 = *(_QWORD *)(v72 + 24);
                v29 *= v94;
                continue;
              default:
LABEL_159:
                ++*(_DWORD *)(v29 + 64);
                BUG();
            }
          }
      }
    }
  }
LABEL_7:
  v20 = sub_1487400((_QWORD *)v11[3], a2, (__int64)&v180, a7, a8);
  v21 = sub_19DD7C0((__int64)v11, v20, a2);
  if ( v21 )
  {
    v22 = sub_16498A0(a2);
    v23 = *(unsigned __int8 **)(a2 + 48);
    v183 = 0;
    v186 = v22;
    v24 = *(_QWORD *)(a2 + 40);
    v187 = 0;
    v184 = v24;
    v188 = 0;
    v189 = 0;
    v190 = 0;
    v185 = (__int64 *)(a2 + 24);
    v178[0] = v23;
    if ( v23 )
    {
      sub_1623A60((__int64)v178, (__int64)v23, 2);
      if ( v183 )
        sub_161E7C0((__int64)&v183, (__int64)v183);
      v183 = v178[0];
      if ( v178[0] )
        sub_1623210((__int64)v178, v178[0], (__int64)&v183);
    }
    v25 = *(_QWORD *)a2;
    v177 = 257;
    v26 = *(_QWORD *)v21;
    if ( v25 == *(_QWORD *)v21 )
    {
LABEL_22:
      v29 = v11[1];
      v30 = sub_15A9FE0(v29, a6);
      v31 = a6;
      v32 = 1;
      v33 = v30;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v31 + 8) )
        {
          case 1:
            v47 = 16;
            goto LABEL_49;
          case 2:
            v47 = 32;
            goto LABEL_49;
          case 3:
          case 9:
            v47 = 64;
            goto LABEL_49;
          case 4:
            v47 = 80;
            goto LABEL_49;
          case 5:
          case 6:
            v47 = 128;
            goto LABEL_49;
          case 7:
            v163 = v32;
            v55 = sub_15A9520(v29, 0);
            v32 = v163;
            v47 = (unsigned int)(8 * v55);
            goto LABEL_49;
          case 0xB:
            v47 = *(_DWORD *)(v31 + 8) >> 8;
            goto LABEL_49;
          case 0xD:
            v168 = v32;
            v65 = (_QWORD *)sub_15A9930(v29, v31);
            v32 = v168;
            v47 = 8LL * *v65;
            goto LABEL_49;
          case 0xE:
            v137 = v32;
            v169 = *(_QWORD *)(v31 + 32);
            v151 = *(_QWORD *)(v31 + 24);
            v66 = sub_15A9FE0(v29, v151);
            v67 = v151;
            v32 = v137;
            v68 = 1;
            v69 = v66;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v67 + 8) )
              {
                case 1:
                  v105 = 16;
                  goto LABEL_130;
                case 2:
                  v105 = 32;
                  goto LABEL_130;
                case 3:
                case 9:
                  v105 = 64;
                  goto LABEL_130;
                case 4:
                  v105 = 80;
                  goto LABEL_130;
                case 5:
                case 6:
                  v105 = 128;
                  goto LABEL_130;
                case 7:
                  v130 = v137;
                  v110 = 0;
                  v142 = v69;
                  v153 = v68;
                  goto LABEL_139;
                case 0xB:
                  v105 = *(_DWORD *)(v67 + 8) >> 8;
                  goto LABEL_130;
                case 0xD:
                  v133 = v137;
                  v146 = v69;
                  v157 = v68;
                  v116 = (_QWORD *)sub_15A9930(v29, v67);
                  v68 = v157;
                  v69 = v146;
                  v32 = v133;
                  v105 = 8LL * *v116;
                  goto LABEL_130;
                case 0xE:
                  v120 = v137;
                  v123 = v69;
                  v126 = v68;
                  v132 = *(_QWORD *)(v67 + 24);
                  v156 = *(_QWORD *)(v67 + 32);
                  v145 = (unsigned int)sub_15A9FE0(v29, v132);
                  v115 = sub_127FA20(v29, v132);
                  v68 = v126;
                  v69 = v123;
                  v32 = v120;
                  v105 = 8 * v156 * v145 * ((v145 + ((unsigned __int64)(v115 + 7) >> 3) - 1) / v145);
                  goto LABEL_130;
                case 0xF:
                  v130 = v137;
                  v142 = v69;
                  v153 = v68;
                  v110 = *(_DWORD *)(v67 + 8) >> 8;
LABEL_139:
                  v111 = sub_15A9520(v29, v110);
                  v68 = v153;
                  v69 = v142;
                  v32 = v130;
                  v105 = (unsigned int)(8 * v111);
LABEL_130:
                  v47 = 8 * v69 * v169 * ((v69 + ((unsigned __int64)(v105 * v68 + 7) >> 3) - 1) / v69);
                  goto LABEL_49;
                case 0x10:
                  v109 = *(_QWORD *)(v67 + 32);
                  v67 = *(_QWORD *)(v67 + 24);
                  v68 *= v109;
                  continue;
                default:
                  goto LABEL_159;
              }
            }
          case 0xF:
            v162 = v32;
            v53 = sub_15A9520(v29, *(_DWORD *)(v31 + 8) >> 8);
            v32 = v162;
            v47 = (unsigned int)(8 * v53);
LABEL_49:
            v149 = *(_QWORD *)(a2 + 64);
            v161 = v11[1];
            v48 = (v33 + ((unsigned __int64)(v32 * v47 + 7) >> 3) - 1) / v33 * v33;
            v49 = sub_15A9FE0(v161, v149);
            v50 = v161;
            v51 = v149;
            v52 = 1;
            v29 = v49;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v51 + 8) )
              {
                case 1:
                  v34 = 16;
                  goto LABEL_25;
                case 2:
                  v34 = 32;
                  goto LABEL_25;
                case 3:
                case 9:
                  v34 = 64;
                  goto LABEL_25;
                case 4:
                  v34 = 80;
                  goto LABEL_25;
                case 5:
                case 6:
                  v34 = 128;
                  goto LABEL_25;
                case 7:
                  v164 = v52;
                  v56 = sub_15A9520(v50, 0);
                  v52 = v164;
                  v34 = (unsigned int)(8 * v56);
                  goto LABEL_25;
                case 0xB:
                  v34 = *(_DWORD *)(v51 + 8) >> 8;
                  goto LABEL_25;
                case 0xD:
                  v167 = v52;
                  v64 = (_QWORD *)sub_15A9930(v50, v51);
                  v52 = v167;
                  v34 = 8LL * *v64;
                  goto LABEL_25;
                case 0xE:
                  v127 = v52;
                  v150 = v161;
                  v136 = *(_QWORD *)(v51 + 24);
                  v166 = *(_QWORD *)(v51 + 32);
                  v59 = sub_15A9FE0(v50, v136);
                  v60 = v150;
                  v61 = v136;
                  v62 = 1;
                  v52 = v127;
                  v63 = v59;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v61 + 8) )
                    {
                      case 1:
                        v106 = 16;
                        goto LABEL_132;
                      case 2:
                        v106 = 32;
                        goto LABEL_132;
                      case 3:
                      case 9:
                        v106 = 64;
                        goto LABEL_132;
                      case 4:
                        v106 = 80;
                        goto LABEL_132;
                      case 5:
                      case 6:
                        v106 = 128;
                        goto LABEL_132;
                      case 7:
                        v107 = 0;
                        v141 = v63;
                        v152 = v62;
                        goto LABEL_136;
                      case 0xB:
                        v106 = *(_DWORD *)(v61 + 8) >> 8;
                        goto LABEL_132;
                      case 0xD:
                        v143 = v63;
                        v154 = v62;
                        v112 = (_QWORD *)sub_15A9930(v60, v61);
                        v62 = v154;
                        v63 = v143;
                        v52 = v127;
                        v106 = 8LL * *v112;
                        goto LABEL_132;
                      case 0xE:
                        v118 = v127;
                        v119 = v63;
                        v122 = v62;
                        v125 = *(_QWORD *)(v61 + 24);
                        v131 = v150;
                        v155 = *(_QWORD *)(v61 + 32);
                        v144 = (unsigned int)sub_15A9FE0(v60, v125);
                        v114 = sub_127FA20(v131, v125);
                        v62 = v122;
                        v63 = v119;
                        v52 = v118;
                        v106 = 8 * v155 * v144 * ((v144 + ((unsigned __int64)(v114 + 7) >> 3) - 1) / v144);
                        goto LABEL_132;
                      case 0xF:
                        v141 = v63;
                        v152 = v62;
                        v107 = *(_DWORD *)(v61 + 8) >> 8;
LABEL_136:
                        v108 = sub_15A9520(v60, v107);
                        v62 = v152;
                        v63 = v141;
                        v52 = v127;
                        v106 = (unsigned int)(8 * v108);
LABEL_132:
                        v34 = 8 * v63 * v166 * ((v63 + ((unsigned __int64)(v106 * v62 + 7) >> 3) - 1) / v63);
                        goto LABEL_25;
                      case 0x10:
                        v113 = *(_QWORD *)(v61 + 32);
                        v61 = *(_QWORD *)(v61 + 24);
                        v62 *= v113;
                        continue;
                      default:
                        goto LABEL_159;
                    }
                  }
                case 0xF:
                  v165 = v52;
                  v58 = sub_15A9520(v50, *(_DWORD *)(v51 + 8) >> 8);
                  v52 = v165;
                  v34 = (unsigned int)(8 * v58);
LABEL_25:
                  v159 = v29 * ((v29 + ((unsigned __int64)(v52 * v34 + 7) >> 3) - 1) / v29);
                  v148 = v48 / v159;
                  if ( v48 % v159 )
                  {
                    v21 = 0;
                    goto LABEL_34;
                  }
                  v35 = sub_15A9650(v11[1], *(_QWORD *)a2);
                  v36 = v159;
                  v37 = (__int64 **)v35;
                  v38 = *(__int64 ***)a5;
                  if ( v35 == *(_QWORD *)a5 )
                    goto LABEL_29;
                  v135 = v159;
                  v177 = 257;
                  v160 = sub_16431D0((__int64)v38);
                  v39 = sub_16431D0((__int64)v37);
                  v36 = v135;
                  if ( v160 < v39 )
                  {
                    v170 = v135;
                    if ( *(_BYTE *)(a5 + 16) <= 0x10u )
                    {
                      v87 = sub_15A46C0(38, (__int64 ***)a5, v37, 0);
                      v36 = v135;
                      a5 = v87;
                      goto LABEL_29;
                    }
                    v95 = a5;
                    v96 = (__int64)v37;
                    v179 = 257;
                    v97 = 38;
                  }
                  else
                  {
                    if ( v160 <= v39 )
                      goto LABEL_29;
                    v170 = v135;
                    if ( *(_BYTE *)(a5 + 16) <= 0x10u )
                    {
                      v117 = sub_15A46C0(36, (__int64 ***)a5, v37, 0);
                      v36 = v135;
                      a5 = v117;
                      goto LABEL_29;
                    }
                    v96 = (__int64)v37;
                    v179 = 257;
                    v95 = a5;
                    v97 = 36;
                  }
                  v98 = sub_15FDBD0(v97, v95, v96, (__int64)v178, 0);
                  v99 = v170;
                  a5 = v98;
                  if ( v184 )
                  {
                    v140 = v170;
                    v171 = v185;
                    sub_157E9D0(v184 + 40, v98);
                    v100 = *v171;
                    v101 = *(_QWORD *)(a5 + 24);
                    *(_QWORD *)(a5 + 32) = v171;
                    v100 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(a5 + 24) = v100 | v101 & 7;
                    *(_QWORD *)(v100 + 8) = a5 + 24;
                    *v171 = *v171 & 7 | (a5 + 24);
                    v99 = v140;
                  }
                  v172 = v99;
                  sub_164B780(a5, v176);
                  v36 = v172;
                  if ( v183 )
                  {
                    v175 = v183;
                    sub_1623A60((__int64)&v175, (__int64)v183, 2);
                    v36 = v172;
                    v102 = *(_QWORD *)(a5 + 48);
                    v103 = a5 + 48;
                    if ( v102 )
                    {
                      sub_161E7C0(a5 + 48, v102);
                      v36 = v172;
                      v103 = a5 + 48;
                    }
                    v104 = v175;
                    *(_QWORD *)(a5 + 48) = v175;
                    if ( v104 )
                    {
                      v173 = v36;
                      sub_1623210((__int64)&v175, v104, v103);
                      v36 = v173;
                    }
                  }
LABEL_29:
                  if ( v36 != v48 )
                  {
                    v177 = 257;
                    v40 = sub_15A0680((__int64)v37, v148, 0);
                    if ( *(_BYTE *)(a5 + 16) > 0x10u || *(_BYTE *)(v40 + 16) > 0x10u )
                    {
                      v179 = 257;
                      a5 = sub_15FB440(15, (__int64 *)a5, v40, (__int64)v178, 0);
                      if ( v184 )
                      {
                        v82 = v185;
                        sub_157E9D0(v184 + 40, a5);
                        v83 = *(_QWORD *)(a5 + 24);
                        v84 = *v82;
                        *(_QWORD *)(a5 + 32) = v82;
                        v84 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(a5 + 24) = v84 | v83 & 7;
                        *(_QWORD *)(v84 + 8) = a5 + 24;
                        *v82 = *v82 & 7 | (a5 + 24);
                      }
                      sub_164B780(a5, v176);
                      if ( v183 )
                      {
                        v175 = v183;
                        sub_1623A60((__int64)&v175, (__int64)v183, 2);
                        v85 = *(_QWORD *)(a5 + 48);
                        if ( v85 )
                          sub_161E7C0(a5 + 48, v85);
                        v86 = v175;
                        *(_QWORD *)(a5 + 48) = v175;
                        if ( v86 )
                          sub_1623210((__int64)&v175, v86, a5 + 48);
                      }
                    }
                    else
                    {
                      a5 = sub_15A2C20((__int64 *)a5, v40, 0, 0, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
                    }
                  }
                  v179 = 257;
                  v21 = sub_12815B0((__int64 *)&v183, 0, (_BYTE *)v21, a5, (__int64)v178);
                  v41 = sub_15FA300(a2);
                  sub_15FA2E0(v21, v41);
                  sub_164B7C0(v21, a2);
LABEL_34:
                  if ( v183 )
                    sub_161E7C0((__int64)&v183, (__int64)v183);
                  break;
                case 0x10:
                  v57 = *(_QWORD *)(v51 + 32);
                  v51 = *(_QWORD *)(v51 + 24);
                  v52 *= v57;
                  continue;
                default:
                  goto LABEL_159;
              }
              goto LABEL_36;
            }
          case 0x10:
            v54 = *(_QWORD *)(v31 + 32);
            v31 = *(_QWORD *)(v31 + 24);
            v32 *= v54;
            continue;
          default:
            goto LABEL_159;
        }
      }
    }
    v27 = *(_BYTE *)(v26 + 8);
    if ( v27 == 16 )
    {
      v27 = *(_BYTE *)(**(_QWORD **)(v26 + 16) + 8LL);
      if ( v27 != 15 )
      {
LABEL_16:
        if ( v27 != 11 )
          goto LABEL_20;
        v28 = *(_BYTE *)(v25 + 8);
        if ( v28 == 16 )
          v28 = *(_BYTE *)(**(_QWORD **)(v25 + 16) + 8LL);
        if ( v28 != 15 )
        {
LABEL_20:
          if ( *(_BYTE *)(v21 + 16) <= 0x10u )
          {
            v21 = sub_15A46C0(47, (__int64 ***)v21, (__int64 **)v25, 0);
            goto LABEL_22;
          }
          v75 = v21;
          v179 = 257;
          v76 = 47;
          goto LABEL_83;
        }
        if ( *(_BYTE *)(v21 + 16) <= 0x10u )
        {
          v21 = sub_15A46C0(46, (__int64 ***)v21, (__int64 **)v25, 0);
          goto LABEL_22;
        }
        v75 = v21;
        v179 = 257;
        v76 = 46;
LABEL_83:
        v21 = sub_15FDBD0(v76, v75, v25, (__int64)v178, 0);
        if ( v184 )
        {
          v77 = v185;
          sub_157E9D0(v184 + 40, v21);
          v78 = *(_QWORD *)(v21 + 24);
          v79 = *v77;
          *(_QWORD *)(v21 + 32) = v77;
          v79 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v21 + 24) = v79 | v78 & 7;
          *(_QWORD *)(v79 + 8) = v21 + 24;
          *v77 = *v77 & 7 | (v21 + 24);
        }
        sub_164B780(v21, v176);
        if ( v183 )
        {
          v175 = v183;
          sub_1623A60((__int64)&v175, (__int64)v183, 2);
          v80 = *(_QWORD *)(v21 + 48);
          if ( v80 )
            sub_161E7C0(v21 + 48, v80);
          v81 = v175;
          *(_QWORD *)(v21 + 48) = v175;
          if ( v81 )
            sub_1623210((__int64)&v175, v81, v21 + 48);
        }
        goto LABEL_22;
      }
    }
    else if ( v27 != 15 )
    {
      goto LABEL_16;
    }
    v43 = *(_BYTE *)(v25 + 8);
    if ( v43 == 16 )
      v43 = *(_BYTE *)(**(_QWORD **)(v25 + 16) + 8LL);
    if ( v43 != 11 )
      goto LABEL_20;
    if ( *(_BYTE *)(v21 + 16) <= 0x10u )
    {
      v21 = sub_15A46C0(45, (__int64 ***)v21, (__int64 **)v25, 0);
      goto LABEL_22;
    }
    v75 = v21;
    v76 = 45;
    v179 = 257;
    goto LABEL_83;
  }
LABEL_36:
  if ( v180 != v182 )
    _libc_free((unsigned __int64)v180);
  return v21;
}
