// Function: sub_184D2E0
// Address: 0x184d2e0
//
__int64 __fastcall sub_184D2E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 *v8; // rsi
  __int64 *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned int v20; // r13d
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // rbx
  int v24; // r8d
  int v25; // r9d
  unsigned __int8 v26; // dl
  __int64 v27; // rbx
  char v28; // dl
  __int64 *v29; // rax
  __int64 *v30; // rsi
  __int64 *v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned __int64 v34; // r13
  int v35; // r8d
  int v36; // r9d
  __int64 *v37; // rax
  char v38; // dl
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  char v45; // al
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  unsigned __int64 v51; // rcx
  __int64 v52; // r9
  unsigned __int64 v53; // r15
  unsigned int v54; // eax
  unsigned __int64 v55; // r15
  int v56; // r8d
  int v57; // r9d
  char v58; // di
  __int64 *v59; // rax
  char v60; // dl
  __int64 v61; // rax
  __int64 *v62; // rsi
  __int64 *v63; // rcx
  _QWORD *v64; // r13
  int v65; // r8d
  int v66; // r9d
  __int64 v67; // rax
  __int64 *v68; // rax
  char v69; // dl
  __int64 *v70; // rsi
  __int64 *v71; // rcx
  __int64 v72; // rsi
  __int64 v73; // r9
  __int64 v74; // r8
  __int64 *v75; // rsi
  __int64 *v76; // rcx
  _QWORD *v77; // r13
  unsigned __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r15
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rax
  char v85; // al
  __int64 v86; // rcx
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r13
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // rax
  __int64 v101; // rax
  unsigned __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // r15
  __int64 v106; // rdx
  __int64 v107; // rax
  __int64 v108; // rax
  char v109; // al
  __int64 v110; // rcx
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // r13
  __int64 v114; // rax
  __int64 v115; // [rsp+0h] [rbp-2C0h]
  __int64 v116; // [rsp+8h] [rbp-2B8h]
  __int64 v117; // [rsp+8h] [rbp-2B8h]
  __int64 v118; // [rsp+10h] [rbp-2B0h]
  __int64 v119; // [rsp+10h] [rbp-2B0h]
  __int64 v120; // [rsp+10h] [rbp-2B0h]
  _QWORD *v121; // [rsp+10h] [rbp-2B0h]
  unsigned int v122; // [rsp+10h] [rbp-2B0h]
  __int64 v123; // [rsp+10h] [rbp-2B0h]
  bool v124; // [rsp+1Fh] [rbp-2A1h]
  char v125; // [rsp+1Fh] [rbp-2A1h]
  char v126; // [rsp+20h] [rbp-2A0h]
  __int64 v127; // [rsp+20h] [rbp-2A0h]
  unsigned int v128; // [rsp+20h] [rbp-2A0h]
  char v130; // [rsp+38h] [rbp-288h]
  unsigned __int64 v131; // [rsp+38h] [rbp-288h]
  unsigned __int64 v132; // [rsp+38h] [rbp-288h]
  unsigned __int64 v133; // [rsp+38h] [rbp-288h]
  __int64 v134; // [rsp+38h] [rbp-288h]
  unsigned __int64 v135; // [rsp+38h] [rbp-288h]
  __int64 v136; // [rsp+38h] [rbp-288h]
  __int64 v137; // [rsp+40h] [rbp-280h] BYREF
  __int64 v138; // [rsp+48h] [rbp-278h] BYREF
  _BYTE *v139; // [rsp+50h] [rbp-270h] BYREF
  __int64 v140; // [rsp+58h] [rbp-268h]
  _BYTE v141[256]; // [rsp+60h] [rbp-260h] BYREF
  __int64 v142; // [rsp+160h] [rbp-160h] BYREF
  __int64 *v143; // [rsp+168h] [rbp-158h]
  __int64 *v144; // [rsp+170h] [rbp-150h]
  __int64 v145; // [rsp+178h] [rbp-148h]
  int v146; // [rsp+180h] [rbp-140h]
  _BYTE v147[312]; // [rsp+188h] [rbp-138h] BYREF

  v2 = 0;
  v140 = 0x2000000000LL;
  v139 = v141;
  v142 = 0;
  v143 = (__int64 *)v147;
  v144 = (__int64 *)v147;
  v145 = 32;
  v146 = 0;
  v130 = sub_15E0470(a1);
  if ( !v130 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    if ( !v5 )
    {
      v20 = v140;
LABEL_38:
      if ( !v20 )
      {
LABEL_205:
        v2 = 36;
        goto LABEL_33;
      }
      while ( 1 )
      {
        v21 = v20--;
        v22 = *(_QWORD *)&v139[8 * v21 - 8];
        LODWORD(v140) = v20;
        v23 = (__int64)sub_1648700(v22);
        v26 = *(_BYTE *)(v23 + 16);
        switch ( v26 )
        {
          case 0x19u:
          case 0x4Bu:
            goto LABEL_41;
          case 0x1Du:
          case 0x4Eu:
            v126 = *(_BYTE *)(*(_QWORD *)v23 + 8LL);
            v124 = v126 != 0;
            if ( v26 <= 0x17u )
              goto LABEL_63;
            if ( v26 == 78 )
            {
              v45 = 1;
              v34 = v23 & 0xFFFFFFFFFFFFFFF8LL;
              v46 = v23 | 4;
            }
            else
            {
              if ( v26 != 29 )
              {
LABEL_63:
                v137 = 0;
                v33 = 56;
                v34 = 0;
                goto LABEL_64;
              }
              v45 = 0;
              v46 = v23 & 0xFFFFFFFFFFFFFFFBLL;
              v34 = v23 & 0xFFFFFFFFFFFFFFF8LL;
            }
            v137 = v46;
            v33 = v34 + 56;
            if ( v45 )
            {
              if ( (unsigned __int8)sub_1560260((_QWORD *)v33, -1, 36) )
                goto LABEL_65;
              if ( *(char *)(v34 + 23) < 0 )
              {
                v47 = sub_1648A40(v34);
                v49 = v48 + v47;
                v50 = 0;
                v120 = v49;
                if ( *(char *)(v34 + 23) < 0 )
                  v50 = sub_1648A40(v34);
                if ( (unsigned int)((v120 - v50) >> 4) )
                  goto LABEL_94;
              }
              v44 = *(_QWORD *)(v34 - 24);
              if ( *(_BYTE *)(v44 + 16) )
                goto LABEL_94;
LABEL_93:
              v138 = *(_QWORD *)(v44 + 112);
              if ( !(unsigned __int8)sub_1560260(&v138, -1, 36) )
                goto LABEL_94;
LABEL_65:
              if ( v126 )
              {
                while ( 1 )
                {
                  v23 = *(_QWORD *)(v23 + 8);
                  if ( !v23 )
                    goto LABEL_58;
                  v37 = v143;
                  if ( v144 != v143 )
                    goto LABEL_68;
                  v70 = &v143[HIDWORD(v145)];
                  if ( v143 == v70 )
                  {
LABEL_203:
                    if ( HIDWORD(v145) >= (unsigned int)v145 )
                    {
LABEL_68:
                      sub_16CCBA0((__int64)&v142, v23);
                      if ( v38 )
                        goto LABEL_69;
                    }
                    else
                    {
                      ++HIDWORD(v145);
                      *v70 = v23;
                      ++v142;
LABEL_69:
                      v39 = (unsigned int)v140;
                      if ( (unsigned int)v140 >= HIDWORD(v140) )
                      {
                        sub_16CD150((__int64)&v139, v141, 0, 8, v35, v36);
                        v39 = (unsigned int)v140;
                      }
                      *(_QWORD *)&v139[8 * v39] = v23;
                      LODWORD(v140) = v140 + 1;
                    }
                  }
                  else
                  {
                    v71 = 0;
                    while ( *v37 != v23 )
                    {
                      if ( *v37 == -2 )
                        v71 = v37;
                      if ( v70 == ++v37 )
                      {
                        if ( !v71 )
                          goto LABEL_203;
                        *v71 = v23;
                        --v146;
                        ++v142;
                        goto LABEL_69;
                      }
                    }
                  }
                }
              }
              goto LABEL_58;
            }
LABEL_64:
            if ( (unsigned __int8)sub_1560260((_QWORD *)v33, -1, 36) )
              goto LABEL_65;
            if ( *(char *)(v34 + 23) >= 0 )
              goto LABEL_218;
            v40 = sub_1648A40(v34);
            v42 = v41 + v40;
            v43 = 0;
            v119 = v42;
            if ( *(char *)(v34 + 23) < 0 )
              v43 = sub_1648A40(v34);
            if ( !(unsigned int)((v119 - v43) >> 4) )
            {
LABEL_218:
              v44 = *(_QWORD *)(v34 - 72);
              if ( !*(_BYTE *)(v44 + 16) )
                goto LABEL_93;
            }
LABEL_94:
            v51 = v137 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v137 & 4) != 0 )
            {
              v52 = *(_QWORD *)(v51 - 24);
              if ( !*(_BYTE *)(v52 + 16) )
                goto LABEL_96;
              v77 = (_QWORD *)(v51 + 56);
              v132 = v137 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (unsigned __int8)sub_1560260((_QWORD *)(v51 + 56), -1, 36) )
                goto LABEL_121;
              v78 = v132;
              if ( *(char *)(v132 + 23) >= 0 )
                goto LABEL_219;
              v79 = sub_1648A40(v132);
              v78 = v132;
              v81 = v79 + v80;
              v82 = 0;
              if ( *(char *)(v132 + 23) < 0 )
              {
                v83 = sub_1648A40(v132);
                v78 = v132;
                v82 = v83;
              }
              if ( !(unsigned int)((v81 - v82) >> 4) )
              {
LABEL_219:
                v84 = *(_QWORD *)(v78 - 24);
                if ( !*(_BYTE *)(v84 + 16) )
                {
                  v133 = v78;
                  v138 = *(_QWORD *)(v84 + 112);
                  v85 = sub_1560260(&v138, -1, 36);
                  v78 = v133;
                  if ( v85 )
                    goto LABEL_121;
                }
              }
              v134 = v78;
              if ( (unsigned __int8)sub_1560260(v77, -1, 37) )
                goto LABEL_121;
              v86 = v134;
              if ( *(char *)(v134 + 23) < 0 )
              {
                v87 = sub_1648A40(v134);
                v86 = v134;
                v89 = v87 + v88;
                if ( *(char *)(v134 + 23) >= 0 )
                {
                  v90 = 0;
                }
                else
                {
                  v90 = sub_1648A40(v134);
                  v86 = v134;
                }
                if ( v90 != v89 )
                {
                  while ( *(_DWORD *)(*(_QWORD *)v90 + 8LL) <= 1u )
                  {
                    v90 += 16;
                    if ( v89 == v90 )
                      goto LABEL_170;
                  }
                  goto LABEL_32;
                }
              }
LABEL_170:
              v91 = *(_QWORD *)(v86 - 24);
              if ( *(_BYTE *)(v91 + 16) )
                goto LABEL_32;
              goto LABEL_201;
            }
            v52 = *(_QWORD *)(v51 - 72);
            if ( *(_BYTE *)(v52 + 16) )
            {
              v64 = (_QWORD *)(v51 + 56);
              v131 = v137 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (unsigned __int8)sub_1560260((_QWORD *)(v51 + 56), -1, 36) )
                goto LABEL_121;
              v102 = v131;
              if ( *(char *)(v131 + 23) >= 0 )
                goto LABEL_220;
              v103 = sub_1648A40(v131);
              v102 = v131;
              v105 = v103 + v104;
              v106 = 0;
              if ( *(char *)(v131 + 23) < 0 )
              {
                v107 = sub_1648A40(v131);
                v102 = v131;
                v106 = v107;
              }
              if ( !(unsigned int)((v105 - v106) >> 4) )
              {
LABEL_220:
                v108 = *(_QWORD *)(v102 - 72);
                if ( !*(_BYTE *)(v108 + 16) )
                {
                  v135 = v102;
                  v138 = *(_QWORD *)(v108 + 112);
                  v109 = sub_1560260(&v138, -1, 36);
                  v102 = v135;
                  if ( v109 )
                    goto LABEL_121;
                }
              }
              v136 = v102;
              if ( (unsigned __int8)sub_1560260(v64, -1, 37) )
              {
LABEL_121:
                if ( !v126 )
                {
LABEL_122:
                  v130 = 1;
                  goto LABEL_58;
                }
                while ( 1 )
                {
                  v23 = *(_QWORD *)(v23 + 8);
                  if ( !v23 )
                    goto LABEL_122;
                  v68 = v143;
                  if ( v144 != v143 )
                    goto LABEL_130;
                  v75 = &v143[HIDWORD(v145)];
                  if ( v143 == v75 )
                  {
LABEL_210:
                    if ( HIDWORD(v145) < (unsigned int)v145 )
                    {
                      ++HIDWORD(v145);
                      *v75 = v23;
                      ++v142;
                      goto LABEL_125;
                    }
LABEL_130:
                    sub_16CCBA0((__int64)&v142, v23);
                    if ( v69 )
                    {
LABEL_125:
                      v67 = (unsigned int)v140;
                      if ( (unsigned int)v140 >= HIDWORD(v140) )
                      {
                        sub_16CD150((__int64)&v139, v141, 0, 8, v65, v66);
                        v67 = (unsigned int)v140;
                      }
                      *(_QWORD *)&v139[8 * v67] = v23;
                      LODWORD(v140) = v140 + 1;
                    }
                  }
                  else
                  {
                    v76 = 0;
                    while ( *v68 != v23 )
                    {
                      if ( *v68 == -2 )
                        v76 = v68;
                      if ( v75 == ++v68 )
                      {
                        if ( !v76 )
                          goto LABEL_210;
                        *v76 = v23;
                        --v146;
                        ++v142;
                        goto LABEL_125;
                      }
                    }
                  }
                }
              }
              v110 = v136;
              if ( *(char *)(v136 + 23) < 0 )
              {
                v111 = sub_1648A40(v136);
                v110 = v136;
                v113 = v111 + v112;
                if ( *(char *)(v136 + 23) >= 0 )
                {
                  v114 = 0;
                }
                else
                {
                  v114 = sub_1648A40(v136);
                  v110 = v136;
                }
                if ( v114 != v113 )
                {
                  while ( *(_DWORD *)(*(_QWORD *)v114 + 8LL) <= 1u )
                  {
                    v114 += 16;
                    if ( v113 == v114 )
                      goto LABEL_200;
                  }
                  goto LABEL_32;
                }
              }
LABEL_200:
              v91 = *(_QWORD *)(v110 - 72);
              if ( *(_BYTE *)(v91 + 16) )
                goto LABEL_32;
LABEL_201:
              v138 = *(_QWORD *)(v91 + 112);
              if ( !(unsigned __int8)sub_1560260(&v138, -1, 37) )
                goto LABEL_32;
              goto LABEL_121;
            }
LABEL_96:
            v127 = v52;
            v53 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v22 - (v51 - 24LL * (*(_DWORD *)(v51 + 20) & 0xFFFFFFF))) >> 3);
            v54 = sub_14DA610(&v137);
            if ( (unsigned __int64)(unsigned int)v53 >= *(_QWORD *)(v127 + 96) )
            {
              if ( v54 <= (unsigned int)v53 )
              {
                v128 = v53 + 1;
                v125 = v124 & (sub_1779030(&v137, (int)v53 + 1, 22) ^ 1);
                goto LABEL_99;
              }
LABEL_32:
              v2 = 0;
              goto LABEL_33;
            }
            v122 = v54;
            v72 = (unsigned int)(v53 + 1);
            v115 = v127;
            v128 = v53 + 1;
            v125 = v124 & (sub_1779030(&v137, v72, 22) ^ 1);
            if ( v122 <= (unsigned int)v53 )
              goto LABEL_99;
            v73 = v115;
            v74 = (unsigned int)v53;
            if ( (*(_BYTE *)(v115 + 18) & 1) != 0 )
            {
              sub_15E08E0(v115, v72);
              v73 = v115;
              v74 = (unsigned int)v53;
            }
            if ( !sub_1833FA0(a2, *(_QWORD *)(v73 + 88) + 40 * v74) )
            {
LABEL_99:
              v55 = v137 & 0xFFFFFFFFFFFFFFF8LL;
              v121 = (_QWORD *)((v137 & 0xFFFFFFFFFFFFFFF8LL) + 56);
              if ( (v137 & 4) != 0 )
              {
                if ( (unsigned __int8)sub_1560260((_QWORD *)((v137 & 0xFFFFFFFFFFFFFFF8LL) + 56), -1, 36) )
                  goto LABEL_101;
                if ( *(char *)(v55 + 23) >= 0 )
                  goto LABEL_221;
                v92 = sub_1648A40(v55);
                v94 = v93 + v92;
                v95 = 0;
                v117 = v94;
                if ( *(char *)(v55 + 23) < 0 )
                  v95 = sub_1648A40(v55);
                if ( !(unsigned int)((v117 - v95) >> 4) )
                {
LABEL_221:
                  v96 = *(_QWORD *)(v55 - 24);
                  if ( !*(_BYTE *)(v96 + 16) )
                  {
                    v138 = *(_QWORD *)(v96 + 112);
                    if ( (unsigned __int8)sub_1560260(&v138, -1, 36) )
                      goto LABEL_101;
                  }
                }
                if ( (unsigned __int8)sub_1560260(v121, -1, 37) )
                  goto LABEL_101;
                if ( *(char *)(v55 + 23) < 0 )
                {
                  v97 = sub_1648A40(v55);
                  v99 = v97 + v98;
                  if ( *(char *)(v55 + 23) >= 0 )
                  {
                    v100 = 0;
                  }
                  else
                  {
                    v123 = v97 + v98;
                    v100 = sub_1648A40(v55);
                    v99 = v123;
                  }
                  if ( v100 != v99 )
                  {
                    while ( *(_DWORD *)(*(_QWORD *)v100 + 8LL) <= 1u )
                    {
                      v100 += 16;
                      if ( v99 == v100 )
                        goto LABEL_185;
                    }
                    goto LABEL_30;
                  }
                }
LABEL_185:
                v101 = *(_QWORD *)(v55 - 24);
                if ( *(_BYTE *)(v101 + 16) )
                  goto LABEL_30;
              }
              else
              {
                if ( (unsigned __int8)sub_1560260(v121, -1, 36) )
                  goto LABEL_101;
                if ( *(char *)(v55 + 23) >= 0 )
                  goto LABEL_222;
                v10 = sub_1648A40(v55);
                v12 = v11 + v10;
                v13 = 0;
                v116 = v12;
                if ( *(char *)(v55 + 23) < 0 )
                  v13 = sub_1648A40(v55);
                if ( !(unsigned int)((v116 - v13) >> 4) )
                {
LABEL_222:
                  v14 = *(_QWORD *)(v55 - 72);
                  if ( !*(_BYTE *)(v14 + 16) )
                  {
                    v138 = *(_QWORD *)(v14 + 112);
                    if ( (unsigned __int8)sub_1560260(&v138, -1, 36) )
                      goto LABEL_101;
                  }
                }
                if ( (unsigned __int8)sub_1560260(v121, -1, 37) )
                  goto LABEL_101;
                if ( *(char *)(v55 + 23) < 0 )
                {
                  v15 = sub_1648A40(v55);
                  v17 = v15 + v16;
                  if ( *(char *)(v55 + 23) >= 0 )
                  {
                    v18 = 0;
                  }
                  else
                  {
                    v118 = v15 + v16;
                    v18 = sub_1648A40(v55);
                    v17 = v118;
                  }
                  if ( v18 != v17 )
                  {
                    while ( *(_DWORD *)(*(_QWORD *)v18 + 8LL) <= 1u )
                    {
                      v18 += 16;
                      if ( v17 == v18 )
                        goto LABEL_207;
                    }
LABEL_30:
                    if ( !(unsigned __int8)sub_1779030(&v137, v128, 37)
                      && !(unsigned __int8)sub_1779030(&v137, v128, 36) )
                    {
                      goto LABEL_32;
                    }
                    goto LABEL_101;
                  }
                }
LABEL_207:
                v101 = *(_QWORD *)(v55 - 72);
                if ( *(_BYTE *)(v101 + 16) )
                  goto LABEL_30;
              }
              v138 = *(_QWORD *)(v101 + 112);
              if ( !(unsigned __int8)sub_1560260(&v138, -1, 37) )
                goto LABEL_30;
LABEL_101:
              v58 = v130;
              if ( !(unsigned __int8)sub_1779030(&v137, v128, 36) )
                v58 = 1;
              v130 = v58;
            }
            if ( v125 )
            {
              while ( 1 )
              {
                v23 = *(_QWORD *)(v23 + 8);
                if ( !v23 )
                  break;
                v59 = v143;
                if ( v144 != v143 )
                  goto LABEL_107;
                v62 = &v143[HIDWORD(v145)];
                if ( v143 == v62 )
                {
LABEL_147:
                  if ( HIDWORD(v145) >= (unsigned int)v145 )
                  {
LABEL_107:
                    sub_16CCBA0((__int64)&v142, v23);
                    if ( v60 )
                      goto LABEL_108;
                  }
                  else
                  {
                    ++HIDWORD(v145);
                    *v62 = v23;
                    ++v142;
LABEL_108:
                    v61 = (unsigned int)v140;
                    if ( (unsigned int)v140 >= HIDWORD(v140) )
                    {
                      sub_16CD150((__int64)&v139, v141, 0, 8, v56, v57);
                      v61 = (unsigned int)v140;
                    }
                    *(_QWORD *)&v139[8 * v61] = v23;
                    LODWORD(v140) = v140 + 1;
                  }
                }
                else
                {
                  v63 = 0;
                  while ( v23 != *v59 )
                  {
                    if ( *v59 == -2 )
                      v63 = v59;
                    if ( v62 == ++v59 )
                    {
                      if ( !v63 )
                        goto LABEL_147;
                      *v63 = v23;
                      --v146;
                      ++v142;
                      goto LABEL_108;
                    }
                  }
                }
              }
            }
LABEL_58:
            v20 = v140;
            if ( !(_DWORD)v140 )
            {
LABEL_42:
              if ( v130 )
              {
                v2 = 37;
                goto LABEL_33;
              }
              goto LABEL_205;
            }
            continue;
          case 0x36u:
            if ( (*(_BYTE *)(v23 + 18) & 1) != 0 )
              goto LABEL_32;
            v130 = 1;
            if ( !v20 )
              goto LABEL_42;
            continue;
          case 0x38u:
          case 0x47u:
          case 0x48u:
          case 0x4Du:
          case 0x4Fu:
            v27 = *(_QWORD *)(v23 + 8);
            if ( v27 )
            {
              while ( 1 )
              {
                while ( 1 )
                {
                  v29 = v143;
                  if ( v144 != v143 )
                    break;
                  v30 = &v143[HIDWORD(v145)];
                  if ( v143 != v30 )
                  {
                    v31 = 0;
                    while ( *v29 != v27 )
                    {
                      if ( *v29 == -2 )
                        v31 = v29;
                      if ( v30 == ++v29 )
                      {
                        if ( !v31 )
                          goto LABEL_44;
                        *v31 = v27;
                        --v146;
                        ++v142;
                        goto LABEL_56;
                      }
                    }
                    goto LABEL_46;
                  }
LABEL_44:
                  if ( HIDWORD(v145) >= (unsigned int)v145 )
                    break;
                  ++HIDWORD(v145);
                  *v30 = v27;
                  v32 = (unsigned int)v140;
                  ++v142;
                  if ( (unsigned int)v140 >= HIDWORD(v140) )
                  {
LABEL_78:
                    sub_16CD150((__int64)&v139, v141, 0, 8, v24, v25);
                    v32 = (unsigned int)v140;
                  }
LABEL_57:
                  *(_QWORD *)&v139[8 * v32] = v27;
                  LODWORD(v140) = v140 + 1;
                  v27 = *(_QWORD *)(v27 + 8);
                  if ( !v27 )
                    goto LABEL_58;
                }
                sub_16CCBA0((__int64)&v142, v27);
                if ( v28 )
                {
LABEL_56:
                  v32 = (unsigned int)v140;
                  if ( (unsigned int)v140 >= HIDWORD(v140) )
                    goto LABEL_78;
                  goto LABEL_57;
                }
LABEL_46:
                v27 = *(_QWORD *)(v27 + 8);
                if ( !v27 )
                  goto LABEL_58;
              }
            }
LABEL_41:
            if ( !v20 )
              goto LABEL_42;
            break;
          default:
            goto LABEL_32;
        }
      }
    }
    while ( 1 )
    {
      v7 = v143;
      if ( v144 != v143 )
        goto LABEL_4;
      v8 = &v143[HIDWORD(v145)];
      if ( v143 != v8 )
      {
        v9 = 0;
        while ( *v7 != v5 )
        {
          if ( *v7 == -2 )
            v9 = v7;
          if ( v8 == ++v7 )
          {
            if ( !v9 )
              goto LABEL_75;
            *v9 = v5;
            v6 = (unsigned int)v140;
            --v146;
            ++v142;
            if ( (unsigned int)v140 < HIDWORD(v140) )
              goto LABEL_6;
            goto LABEL_16;
          }
        }
        goto LABEL_5;
      }
LABEL_75:
      if ( HIDWORD(v145) < (unsigned int)v145 )
      {
        ++HIDWORD(v145);
        *v8 = v5;
        ++v142;
      }
      else
      {
LABEL_4:
        sub_16CCBA0((__int64)&v142, v5);
      }
LABEL_5:
      v6 = (unsigned int)v140;
      if ( (unsigned int)v140 >= HIDWORD(v140) )
      {
LABEL_16:
        sub_16CD150((__int64)&v139, v141, 0, 8, v3, v4);
        v6 = (unsigned int)v140;
      }
LABEL_6:
      *(_QWORD *)&v139[8 * v6] = v5;
      v20 = v140 + 1;
      LODWORD(v140) = v140 + 1;
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        goto LABEL_38;
    }
  }
LABEL_33:
  if ( v144 != v143 )
    _libc_free((unsigned __int64)v144);
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
  return v2;
}
