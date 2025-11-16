// Function: sub_1779C20
// Address: 0x1779c20
//
__int64 __fastcall sub_1779C20(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned __int16 v13; // ax
  __int64 v14; // r15
  __int64 v15; // rbx
  unsigned int v16; // eax
  unsigned int v17; // ebx
  unsigned int v18; // eax
  int v19; // r8d
  __int64 v20; // r9
  __int64 v22; // rbx
  unsigned __int8 v23; // al
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // r9
  __int64 *v28; // r8
  __int64 v29; // rbx
  __int64 v30; // rsi
  __int64 v31; // rdx
  int v32; // eax
  unsigned __int16 v33; // ax
  __int64 v34; // rax
  __int64 v35; // rbx
  unsigned __int8 v36; // al
  __int64 v37; // rsi
  _QWORD *v38; // r15
  __int64 v39; // rax
  __int64 v40; // r12
  _QWORD *v41; // r13
  __int64 v42; // r15
  int v43; // r14d
  unsigned __int64 v44; // rbx
  char v45; // al
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // r13d
  __int64 v50; // rax
  __int64 *v51; // rdx
  char v52; // al
  char v53; // al
  int v54; // eax
  __int64 v55; // rbx
  __int64 v56; // rdx
  int v57; // eax
  int v58; // eax
  _BYTE **v59; // rdx
  _BYTE *v60; // r15
  char v61; // al
  _BYTE *v62; // rdx
  __int64 v63; // rcx
  unsigned __int8 v64; // al
  __int64 v65; // rdi
  unsigned __int8 v66; // si
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // r15
  _QWORD *v71; // rax
  __int64 v72; // rdi
  _BYTE *v73; // rax
  __int64 *v74; // r13
  __int64 v75; // rsi
  __int64 v76; // rsi
  unsigned __int8 *v77; // rsi
  __int64 **v78; // rax
  __int64 v79; // r13
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  _QWORD *v82; // rax
  __int64 *v83; // r15
  __int64 v84; // r12
  __int64 v85; // rsi
  unsigned int v86; // esi
  bool v87; // al
  __int64 *v88; // rdi
  __int64 v89; // rcx
  __int64 v90; // rsi
  __int64 v91; // rcx
  __int64 v92; // rbx
  unsigned __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rax
  unsigned int v96; // esi
  int v97; // eax
  unsigned int v98; // eax
  __int64 v99; // rcx
  __int64 v100; // rsi
  unsigned __int64 v101; // r11
  _QWORD *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned int v105; // esi
  int v106; // eax
  unsigned int v107; // eax
  __int64 v108; // r10
  __int64 v109; // rsi
  unsigned __int64 v110; // r11
  _QWORD *v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rax
  _QWORD *v116; // rax
  unsigned int v117; // esi
  int v118; // eax
  unsigned int v119; // esi
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rdx
  _QWORD *v124; // rax
  _QWORD *i; // rdx
  unsigned int v126; // eax
  __int64 v127; // r15
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 *v130; // rax
  __int64 v131; // rsi
  unsigned __int64 v132; // rcx
  __int64 v133; // rcx
  __int64 v134; // rdx
  unsigned __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // rax
  __int64 v139; // r13
  __int64 v140; // r15
  _QWORD *v141; // rax
  double v142; // xmm4_8
  double v143; // xmm5_8
  __int64 v144; // rax
  unsigned int v145; // r13d
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rcx
  unsigned __int64 v149; // rsi
  __int64 v150; // rcx
  __int64 v151; // rdx
  __int64 v152; // r12
  __int64 *v153; // [rsp+8h] [rbp-A8h]
  __int64 *v154; // [rsp+10h] [rbp-A0h]
  __int64 *v155; // [rsp+10h] [rbp-A0h]
  __int64 *v156; // [rsp+18h] [rbp-98h]
  unsigned __int64 v157; // [rsp+18h] [rbp-98h]
  unsigned __int64 v158; // [rsp+20h] [rbp-90h]
  __int64 *v159; // [rsp+20h] [rbp-90h]
  unsigned __int64 v160; // [rsp+20h] [rbp-90h]
  __int64 *v161; // [rsp+28h] [rbp-88h]
  __int64 v162; // [rsp+28h] [rbp-88h]
  __int64 *v163; // [rsp+28h] [rbp-88h]
  __int64 *v164; // [rsp+28h] [rbp-88h]
  __int64 *v165; // [rsp+28h] [rbp-88h]
  __int64 v166; // [rsp+28h] [rbp-88h]
  __int64 *v167; // [rsp+30h] [rbp-80h]
  __int64 *v168; // [rsp+30h] [rbp-80h]
  __int64 v169; // [rsp+30h] [rbp-80h]
  __int64 *v170; // [rsp+30h] [rbp-80h]
  __int64 *v171; // [rsp+30h] [rbp-80h]
  unsigned __int64 v172; // [rsp+30h] [rbp-80h]
  __int64 v173; // [rsp+30h] [rbp-80h]
  __int64 *v174; // [rsp+38h] [rbp-78h]
  __int64 *v175; // [rsp+38h] [rbp-78h]
  unsigned __int64 v176; // [rsp+38h] [rbp-78h]
  __int64 *v177; // [rsp+38h] [rbp-78h]
  unsigned __int64 v178; // [rsp+38h] [rbp-78h]
  unsigned __int64 v179; // [rsp+38h] [rbp-78h]
  unsigned __int64 v180; // [rsp+38h] [rbp-78h]
  unsigned __int64 v181; // [rsp+38h] [rbp-78h]
  unsigned __int64 v182; // [rsp+38h] [rbp-78h]
  __int64 v183; // [rsp+40h] [rbp-70h]
  __int64 v184; // [rsp+40h] [rbp-70h]
  __int64 *v185; // [rsp+40h] [rbp-70h]
  __int64 v186; // [rsp+40h] [rbp-70h]
  __int64 *v187; // [rsp+40h] [rbp-70h]
  __int64 *v188; // [rsp+40h] [rbp-70h]
  __int64 v189; // [rsp+40h] [rbp-70h]
  __int64 *v190; // [rsp+40h] [rbp-70h]
  __int64 v191; // [rsp+40h] [rbp-70h]
  __int64 v192; // [rsp+40h] [rbp-70h]
  __int64 v193; // [rsp+40h] [rbp-70h]
  __int64 v194; // [rsp+40h] [rbp-70h]
  __int64 v195; // [rsp+40h] [rbp-70h]
  __int64 v196; // [rsp+48h] [rbp-68h]
  __int64 v197; // [rsp+48h] [rbp-68h]
  __int64 *v198; // [rsp+48h] [rbp-68h]
  __int64 v199; // [rsp+48h] [rbp-68h]
  __int64 *v200; // [rsp+48h] [rbp-68h]
  unsigned __int64 v201; // [rsp+48h] [rbp-68h]
  __int64 v202; // [rsp+48h] [rbp-68h]
  unsigned __int64 v203; // [rsp+48h] [rbp-68h]
  __int64 v204; // [rsp+50h] [rbp-60h]
  __int64 **v205; // [rsp+50h] [rbp-60h]
  __int64 v206; // [rsp+50h] [rbp-60h]
  _BYTE *v207; // [rsp+50h] [rbp-60h]
  __int64 v208; // [rsp+58h] [rbp-58h]
  __int64 **v209; // [rsp+58h] [rbp-58h]
  __int64 ***v210; // [rsp+58h] [rbp-58h]
  __int64 v211[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v212; // [rsp+70h] [rbp-40h]

  v11 = a2;
  v12 = *(_QWORD *)(a2 - 24);
  v208 = *(_QWORD *)(a2 - 48);
  v13 = *(_WORD *)(a2 + 18);
  if ( ((v13 >> 7) & 6) == 0 && (v13 & 1) == 0 && !(unsigned __int8)sub_1649A90(v12) )
  {
    v22 = *(_QWORD *)(a2 - 48);
    v23 = *(_BYTE *)(v22 + 16);
    if ( v23 == 71 )
    {
      v22 = *(_QWORD *)(v22 - 24);
      v87 = sub_15F32D0(a2);
      v88 = *(__int64 **)v22;
      if ( !v87 || (unsigned __int8)sub_1776710((__int64)v88) )
      {
        if ( (*((_BYTE *)v88 + 8) != 16 || *(_BYTE *)(**(_QWORD **)(a2 - 48) + 8LL) == 16)
          && !sub_1642F90((__int64)v88, 128) )
        {
          sub_17767C0((__int64)a1, a2, (__int64 **)v22);
          return sub_170BC50((__int64)a1, v11);
        }
        goto LABEL_3;
      }
      v23 = *(_BYTE *)(v22 + 16);
    }
    if ( v23 > 0x17u )
    {
      v205 = 0;
      while ( v23 == 87 )
      {
        v24 = *(_QWORD *)(v22 - 24);
        if ( *(_BYTE *)(v24 + 16) != 83 )
          break;
        if ( v205 )
        {
          if ( *(__int64 ***)(v24 - 48) != v205 )
            break;
        }
        else
        {
          v205 = *(__int64 ***)(v24 - 48);
        }
        v25 = *(_QWORD *)(v24 - 24);
        if ( *(_BYTE *)(v25 + 16) != 13 || *(_DWORD *)(v22 + 64) != 1 )
          break;
        v26 = *(_DWORD *)(v25 + 32) <= 0x40u ? *(_QWORD *)(v25 + 24) : **(_QWORD **)(v25 + 24);
        if ( **(_DWORD **)(v22 + 56) != v26 )
          break;
        v22 = *(_QWORD *)(v22 - 48);
        v23 = *(_BYTE *)(v22 + 16);
        if ( v23 <= 0x17u )
        {
          v14 = a1[333];
          if ( v23 != 9 || !v205 )
            goto LABEL_4;
          v27 = *v205;
          v28 = *(__int64 **)v22;
          v29 = 1;
          v30 = (__int64)*v205;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v30 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v103 = *(_QWORD *)(v30 + 32);
                v30 = *(_QWORD *)(v30 + 24);
                v29 *= v103;
                continue;
              case 1:
                v89 = 16;
                goto LABEL_123;
              case 2:
                v89 = 32;
                goto LABEL_123;
              case 3:
              case 9:
                v89 = 64;
                goto LABEL_123;
              case 4:
                v89 = 80;
                goto LABEL_123;
              case 5:
              case 6:
                v89 = 128;
                goto LABEL_123;
              case 7:
                v185 = *v205;
                v96 = 0;
                v198 = v28;
                goto LABEL_135;
              case 0xB:
                v89 = *(_DWORD *)(v30 + 8) >> 8;
                goto LABEL_123;
              case 0xD:
                v187 = *v205;
                v200 = v28;
                v102 = (_QWORD *)sub_15A9930(a1[333], v30);
                v28 = v200;
                v27 = v187;
                v89 = 8LL * *v102;
                goto LABEL_123;
              case 0xE:
                v167 = *v205;
                v174 = v28;
                v186 = *(_QWORD *)(v30 + 24);
                v199 = *(_QWORD *)(v30 + 32);
                v98 = sub_15A9FE0(a1[333], v186);
                v27 = v167;
                v28 = v174;
                v99 = 1;
                v100 = v186;
                v101 = v98;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v100 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v114 = *(_QWORD *)(v100 + 32);
                      v100 = *(_QWORD *)(v100 + 24);
                      v99 *= v114;
                      continue;
                    case 1:
                      v112 = 16;
                      goto LABEL_157;
                    case 2:
                      v112 = 32;
                      goto LABEL_157;
                    case 3:
                    case 9:
                      v112 = 64;
                      goto LABEL_157;
                    case 4:
                      v112 = 80;
                      goto LABEL_157;
                    case 5:
                    case 6:
                      v112 = 128;
                      goto LABEL_157;
                    case 7:
                      v164 = v167;
                      v117 = 0;
                      v171 = v174;
                      v180 = v101;
                      v193 = v99;
                      goto LABEL_169;
                    case 0xB:
                      v112 = *(_DWORD *)(v100 + 8) >> 8;
                      goto LABEL_157;
                    case 0xD:
                      v163 = v167;
                      v170 = v174;
                      v179 = v101;
                      v192 = v99;
                      v116 = (_QWORD *)sub_15A9930(v14, v100);
                      v99 = v192;
                      v101 = v179;
                      v28 = v170;
                      v27 = v163;
                      v112 = 8LL * *v116;
                      goto LABEL_157;
                    case 0xE:
                      v154 = v167;
                      v156 = v174;
                      v158 = v101;
                      v162 = v99;
                      v169 = *(_QWORD *)(v100 + 24);
                      v191 = *(_QWORD *)(v100 + 32);
                      v178 = (unsigned int)sub_15A9FE0(v14, v169);
                      v115 = sub_127FA20(v14, v169);
                      v99 = v162;
                      v101 = v158;
                      v28 = v156;
                      v27 = v154;
                      v112 = 8 * v178 * v191 * ((v178 + ((unsigned __int64)(v115 + 7) >> 3) - 1) / v178);
                      goto LABEL_157;
                    case 0xF:
                      v164 = v167;
                      v171 = v174;
                      v180 = v101;
                      v117 = *(_DWORD *)(v100 + 8) >> 8;
                      v193 = v99;
LABEL_169:
                      v118 = sub_15A9520(v14, v117);
                      v99 = v193;
                      v101 = v180;
                      v28 = v171;
                      v27 = v164;
                      v112 = (unsigned int)(8 * v118);
LABEL_157:
                      v89 = 8 * v199 * v101 * ((v101 + ((unsigned __int64)(v112 * v99 + 7) >> 3) - 1) / v101);
                      break;
                  }
                  goto LABEL_123;
                }
              case 0xF:
                v185 = *v205;
                v198 = v28;
                v96 = *(_DWORD *)(v30 + 8) >> 8;
LABEL_135:
                v97 = sub_15A9520(v14, v96);
                v28 = v198;
                v27 = v185;
                v89 = (unsigned int)(8 * v97);
LABEL_123:
                v90 = (__int64)v28;
                v91 = v89 * v29 + 7;
                v92 = 1;
                v93 = v91 & 0xFFFFFFFFFFFFFFF8LL;
                break;
            }
            break;
          }
          while ( 2 )
          {
            switch ( *(_BYTE *)(v90 + 8) )
            {
              case 1:
                v94 = 16;
                goto LABEL_127;
              case 2:
                v94 = 32;
                goto LABEL_127;
              case 3:
              case 9:
                v94 = 64;
                goto LABEL_127;
              case 4:
                v94 = 80;
                goto LABEL_127;
              case 5:
              case 6:
                v94 = 128;
                goto LABEL_127;
              case 7:
                v175 = v27;
                v105 = 0;
                v188 = v28;
                v201 = v93;
                goto LABEL_147;
              case 0xB:
                v94 = *(_DWORD *)(v90 + 8) >> 8;
                goto LABEL_127;
              case 0xD:
                v177 = v27;
                v190 = v28;
                v203 = v93;
                v111 = (_QWORD *)sub_15A9930(v14, v90);
                v93 = v203;
                v28 = v190;
                v27 = v177;
                v94 = 8LL * *v111;
                goto LABEL_127;
              case 0xE:
                v168 = v28;
                v161 = v27;
                v176 = v93;
                v189 = *(_QWORD *)(v90 + 24);
                v202 = *(_QWORD *)(v90 + 32);
                v107 = sub_15A9FE0(v14, v189);
                v27 = v161;
                v28 = v168;
                v108 = 1;
                v93 = v176;
                v109 = v189;
                v110 = v107;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v109 + 8) )
                  {
                    case 1:
                      v113 = 16;
                      goto LABEL_160;
                    case 2:
                      v113 = 32;
                      goto LABEL_160;
                    case 3:
                    case 9:
                      v113 = 64;
                      goto LABEL_160;
                    case 4:
                      v113 = 80;
                      goto LABEL_160;
                    case 5:
                    case 6:
                      v113 = 128;
                      goto LABEL_160;
                    case 7:
                      v159 = v161;
                      v119 = 0;
                      v165 = v168;
                      v172 = v176;
                      v181 = v110;
                      v194 = v108;
                      goto LABEL_173;
                    case 0xB:
                      v113 = *(_DWORD *)(v109 + 8) >> 8;
                      goto LABEL_160;
                    case 0xD:
                      v159 = v161;
                      v165 = v168;
                      v172 = v176;
                      v181 = v110;
                      v194 = v108;
                      v113 = 8LL * *(_QWORD *)sub_15A9930(v14, v109);
                      goto LABEL_174;
                    case 0xE:
                      v153 = v161;
                      v155 = v168;
                      v160 = v110;
                      v166 = v108;
                      v195 = *(_QWORD *)(v109 + 32);
                      v157 = v176;
                      v173 = *(_QWORD *)(v109 + 24);
                      v182 = (unsigned int)sub_15A9FE0(v14, v173);
                      v120 = sub_127FA20(v14, v173);
                      v108 = v166;
                      v110 = v160;
                      v28 = v155;
                      v27 = v153;
                      v93 = v157;
                      v113 = 8 * v182 * v195 * ((v182 + ((unsigned __int64)(v120 + 7) >> 3) - 1) / v182);
                      goto LABEL_160;
                    case 0xF:
                      v159 = v161;
                      v165 = v168;
                      v172 = v176;
                      v119 = *(_DWORD *)(v109 + 8) >> 8;
                      v181 = v110;
                      v194 = v108;
LABEL_173:
                      v113 = 8 * (unsigned int)sub_15A9520(v14, v119);
LABEL_174:
                      v108 = v194;
                      v110 = v181;
                      v93 = v172;
                      v28 = v165;
                      v27 = v159;
LABEL_160:
                      v94 = 8 * v202 * v110 * ((v110 + ((unsigned __int64)(v108 * v113 + 7) >> 3) - 1) / v110);
                      goto LABEL_127;
                    case 0x10:
                      v121 = *(_QWORD *)(v109 + 32);
                      v109 = *(_QWORD *)(v109 + 24);
                      v108 *= v121;
                      continue;
                    default:
                      goto LABEL_275;
                  }
                }
              case 0xF:
                v175 = v27;
                v188 = v28;
                v201 = v93;
                v105 = *(_DWORD *)(v90 + 8) >> 8;
LABEL_147:
                v106 = sub_15A9520(v14, v105);
                v93 = v201;
                v28 = v188;
                v27 = v175;
                v94 = (unsigned int)(8 * v106);
LABEL_127:
                if ( v93 != ((v94 * v92 + 7) & 0xFFFFFFFFFFFFFFF8LL) )
                  goto LABEL_3;
                v95 = v27[4];
                if ( *((_BYTE *)v28 + 8) == 14 )
                {
                  if ( v28[4] != v95 )
                    goto LABEL_3;
                }
                else
                {
                  v123 = *((unsigned int *)v28 + 3);
                  if ( v123 != v95 )
                    goto LABEL_3;
                  v124 = (_QWORD *)v28[2];
                  for ( i = &v124[v123]; i != v124; ++v124 )
                  {
                    if ( *v124 != v27[3] )
                      goto LABEL_3;
                  }
                }
                if ( !sub_15F32D0(v11) || (unsigned __int8)sub_1776710((__int64)*v205) )
                {
                  sub_17767C0((__int64)a1, v11, v205);
                  return sub_170BC50((__int64)a1, v11);
                }
                break;
              case 0x10:
                v104 = *(_QWORD *)(v90 + 32);
                v90 = *(_QWORD *)(v90 + 24);
                v92 *= v104;
                continue;
              default:
                goto LABEL_275;
            }
            goto LABEL_3;
          }
        }
      }
    }
  }
LABEL_3:
  v14 = a1[333];
LABEL_4:
  v196 = v11;
  v15 = a1[330];
  v204 = a1[332];
  v16 = sub_15AAE50(v14, *(_QWORD *)v208);
  v17 = sub_1AE99B0(v12, v16, v14, v11, v15, v204);
  v18 = 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1;
  if ( v18 )
  {
    if ( v17 <= v18 )
    {
      if ( (unsigned __int8)sub_1778270(a1, v11) )
        return sub_170BC50((__int64)a1, v11);
      goto LABEL_26;
    }
    goto LABEL_114;
  }
  v86 = sub_15A9FE0(a1[333], *(_QWORD *)v208);
  if ( v17 > v86 )
  {
LABEL_114:
    sub_15F9450(v11, v17);
    if ( (unsigned __int8)sub_1778270(a1, v11) )
      return sub_170BC50((__int64)a1, v11);
    goto LABEL_26;
  }
  sub_15F9450(v11, v86);
  if ( (unsigned __int8)sub_1778270(a1, v11) )
    return sub_170BC50((__int64)a1, v11);
LABEL_26:
  v31 = *(_QWORD *)(v11 - 24);
  v32 = *(unsigned __int8 *)(v31 + 16);
  if ( (unsigned __int8)v32 > 0x17u )
  {
    v54 = v32 - 24;
  }
  else
  {
    if ( (_BYTE)v32 != 5 )
      goto LABEL_28;
    v54 = *(unsigned __int16 *)(v31 + 18);
  }
  if ( v54 != 47 )
    goto LABEL_28;
  v55 = *(_QWORD *)(v11 - 48);
  if ( *(_BYTE *)(v55 + 16) != 54 )
    goto LABEL_28;
  v56 = *(_QWORD *)(v55 - 24);
  v57 = *(unsigned __int8 *)(v56 + 16);
  if ( (unsigned __int8)v57 > 0x17u )
  {
    v58 = v57 - 24;
  }
  else
  {
    if ( (_BYTE)v57 != 5 )
      goto LABEL_28;
    v58 = *(unsigned __int16 *)(v56 + 18);
  }
  if ( v58 != 47 )
    goto LABEL_28;
  v59 = (*(_BYTE *)(v56 + 23) & 0x40) != 0
      ? *(_BYTE ***)(v56 - 8)
      : (_BYTE **)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
  v60 = *v59;
  if ( !*v59 || *(_BYTE *)(*(_QWORD *)v55 + 8LL) != 11 )
    goto LABEL_28;
  v61 = v60[16];
  v62 = *v59;
  if ( v61 == 71 )
  {
    v62 = (_BYTE *)*((_QWORD *)v60 - 3);
    v61 = v62[16];
  }
  if ( v61 != 79 )
    goto LABEL_28;
  v63 = *((_QWORD *)v62 - 9);
  if ( (unsigned __int8)(*(_BYTE *)(v63 + 16) - 75) > 1u )
    goto LABEL_28;
  v20 = *(_QWORD *)(v63 - 48);
  v64 = *(_BYTE *)(v20 + 16);
  if ( v64 <= 0x17u )
    goto LABEL_28;
  v65 = *(_QWORD *)(v63 - 24);
  v66 = *(_BYTE *)(v65 + 16);
  if ( v66 <= 0x17u )
    goto LABEL_28;
  v67 = *((_QWORD *)v62 - 6);
  if ( !v67 )
    goto LABEL_28;
  v68 = *((_QWORD *)v62 - 3);
  if ( !v68 || v64 != 54 )
    goto LABEL_28;
  v69 = *(_QWORD *)(v20 - 24);
  if ( v67 != v69 )
  {
    if ( v68 != v69 || v66 != 54 )
      goto LABEL_28;
LABEL_88:
    if ( v67 == *(_QWORD *)(v65 - 24) )
    {
LABEL_89:
      if ( *(_QWORD *)(v55 + 8) )
      {
        v207 = v60;
        v70 = *(_QWORD *)(v55 + 8);
        do
        {
          v71 = sub_1648700(v70);
          if ( *((_BYTE *)v71 + 16) != 55 )
            goto LABEL_28;
          v72 = *(v71 - 3);
          if ( !v72 )
            BUG();
          if ( v55 == v72 )
            goto LABEL_28;
          v73 = (_BYTE *)*(v71 - 3);
          if ( (*(_BYTE *)(v72 + 16) != 71 || (v73 = *(_BYTE **)(v72 - 24)) != 0) && v207 == v73 )
            goto LABEL_28;
          if ( (unsigned __int8)sub_1649A90(v72) )
            goto LABEL_28;
          v70 = *(_QWORD *)(v70 + 8);
        }
        while ( v70 );
        v60 = v207;
      }
      v74 = (__int64 *)a1[1];
      v74[1] = *(_QWORD *)(v55 + 40);
      v74[2] = v55 + 24;
      v75 = *(_QWORD *)(v55 + 48);
      v211[0] = v75;
      if ( v75 )
      {
        sub_1623A60((__int64)v211, v75, 2);
        v76 = *v74;
        if ( !*v74 )
          goto LABEL_102;
      }
      else
      {
        v76 = *v74;
        if ( !*v74 )
        {
LABEL_104:
          v212 = 257;
          v78 = (__int64 **)sub_17779C0((__int64)a1, v55, **(_QWORD **)(*(_QWORD *)v60 + 16LL), (__int64)v211);
          v79 = *(_QWORD *)(v55 + 8);
          v209 = v78;
          if ( !v79 )
            goto LABEL_242;
          v184 = v11;
          while ( 1 )
          {
            v82 = sub_1648700(v79);
            v83 = (__int64 *)a1[1];
            v84 = (__int64)v82;
            v83[1] = v82[5];
            v83[2] = (__int64)(v82 + 3);
            v85 = v82[6];
            v211[0] = v85;
            if ( v85 )
              break;
            v80 = *v83;
            if ( *v83 )
              goto LABEL_107;
LABEL_110:
            sub_17767C0((__int64)a1, v84, v209);
            v79 = *(_QWORD *)(v79 + 8);
            if ( !v79 )
            {
              v11 = v184;
LABEL_242:
              v138 = sub_1599EF0(*(__int64 ***)v55);
              v139 = *(_QWORD *)(v55 + 8);
              v210 = (__int64 ***)v138;
              if ( v139 )
              {
                v140 = *a1;
                do
                {
                  v141 = sub_1648700(v139);
                  sub_170B990(v140, (__int64)v141);
                  v139 = *(_QWORD *)(v139 + 8);
                }
                while ( v139 );
                if ( v210 == (__int64 ***)v55 )
                  v210 = (__int64 ***)sub_1599EF0(*v210);
                sub_164D160(v55, (__int64)v210, a3, a4, a5, a6, v142, v143, a9, a10);
              }
              sub_170BC50((__int64)a1, v55);
              return sub_170BC50((__int64)a1, v11);
            }
          }
          sub_1623A60((__int64)v211, v85, 2);
          v80 = *v83;
          if ( *v83 )
LABEL_107:
            sub_161E7C0((__int64)v83, v80);
          v81 = (unsigned __int8 *)v211[0];
          *v83 = v211[0];
          if ( v81 )
          {
            sub_1623210((__int64)v211, v81, (__int64)v83);
          }
          else if ( v211[0] )
          {
            sub_161E7C0((__int64)v211, v211[0]);
          }
          goto LABEL_110;
        }
      }
      sub_161E7C0((__int64)v74, v76);
LABEL_102:
      v77 = (unsigned __int8 *)v211[0];
      *v74 = v211[0];
      if ( v77 )
      {
        sub_1623210((__int64)v211, v77, (__int64)v74);
      }
      else if ( v211[0] )
      {
        sub_161E7C0((__int64)v211, v211[0]);
      }
      goto LABEL_104;
    }
    goto LABEL_28;
  }
  if ( v66 == 54 )
  {
    if ( v68 == *(_QWORD *)(v65 - 24) )
      goto LABEL_89;
    if ( v68 == v69 )
      goto LABEL_88;
  }
LABEL_28:
  if ( *(_BYTE *)(v12 + 16) == 56 && (unsigned __int8)sub_1776BE0(a1, v12, v11, (unsigned int *)v211, v19, v20) )
  {
    v127 = sub_15F4880(v12);
    v128 = sub_15A0680(
             **(_QWORD **)(v12 + 24 * (LODWORD(v211[0]) - (unsigned __int64)(*(_DWORD *)(v12 + 20) & 0xFFFFFFF))),
             0,
             0);
    if ( (*(_BYTE *)(v127 + 23) & 0x40) != 0 )
      v129 = *(_QWORD *)(v127 - 8);
    else
      v129 = v127 - 24LL * (*(_DWORD *)(v127 + 20) & 0xFFFFFFF);
    v130 = (__int64 *)(v129 + 24LL * LODWORD(v211[0]));
    if ( *v130 )
    {
      v131 = v130[1];
      v132 = v130[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v132 = v131;
      if ( v131 )
        *(_QWORD *)(v131 + 16) = *(_QWORD *)(v131 + 16) & 3LL | v132;
    }
    *v130 = v128;
    if ( v128 )
    {
      v133 = *(_QWORD *)(v128 + 8);
      v130[1] = v133;
      if ( v133 )
        *(_QWORD *)(v133 + 16) = (unsigned __int64)(v130 + 1) | *(_QWORD *)(v133 + 16) & 3LL;
      v130[2] = (v128 + 8) | v130[2] & 3;
      *(_QWORD *)(v128 + 8) = v130;
    }
    sub_15F2120(v127, v12);
    if ( *(_QWORD *)(v11 - 24) )
    {
      v134 = *(_QWORD *)(v11 - 16);
      v135 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v135 = v134;
      if ( v134 )
        *(_QWORD *)(v134 + 16) = *(_QWORD *)(v134 + 16) & 3LL | v135;
    }
    *(_QWORD *)(v11 - 24) = v127;
    v136 = *(_QWORD *)(v127 + 8);
    *(_QWORD *)(v11 - 16) = v136;
    if ( v136 )
      *(_QWORD *)(v136 + 16) = (v11 - 16) | *(_QWORD *)(v136 + 16) & 3LL;
    *(_QWORD *)(v11 - 8) = (v127 + 8) | *(_QWORD *)(v11 - 8) & 3LL;
    *(_QWORD *)(v127 + 8) = v11 - 24;
    sub_170B990(*a1, v127);
    return v196;
  }
  v33 = *(_WORD *)(v11 + 18);
  if ( ((v33 >> 7) & 6) != 0 || (v33 & 1) != 0 )
    return 0;
  v34 = *(_QWORD *)(v12 + 8);
  if ( v34 )
  {
    v35 = v12;
    do
    {
      if ( *(_QWORD *)(v34 + 8) )
        break;
      v36 = *(_BYTE *)(v35 + 16);
      if ( v36 <= 0x17u )
        break;
      if ( v36 == 53 )
        return sub_170BC50((__int64)a1, v11);
      if ( v36 == 56 )
      {
        v35 = *(_QWORD *)(v35 - 24LL * (*(_DWORD *)(v35 + 20) & 0xFFFFFFF));
      }
      else
      {
        if ( v36 != 72 || !(unsigned __int8)sub_1CCB220(v35) && !(unsigned __int8)sub_1CCB250(v35) )
          break;
        v35 = *(_QWORD *)(v35 - 24);
      }
      v34 = *(_QWORD *)(v35 + 8);
    }
    while ( v34 );
  }
  v37 = *(_QWORD *)(v11 + 40);
  v38 = (_QWORD *)(v11 + 24);
  if ( v11 + 24 == *(_QWORD *)(v37 + 48) )
    goto LABEL_51;
  v39 = v11;
  v197 = v12;
  v40 = v11 + 24;
  v41 = v38;
  v206 = (__int64)a1;
  v42 = v39;
  v43 = 6;
  while ( 1 )
  {
    v44 = *v41 & 0xFFFFFFFFFFFFFFF8LL;
    v41 = (_QWORD *)v44;
    if ( !v44 )
LABEL_275:
      BUG();
    v45 = *(_BYTE *)(v44 - 8);
    if ( v45 == 78 )
    {
      v46 = *(_QWORD *)(v44 - 48);
      if ( !*(_BYTE *)(v46 + 16)
        && (*(_BYTE *)(v46 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v46 + 36) - 35) <= 3 )
      {
        goto LABEL_48;
      }
      goto LABEL_44;
    }
    if ( v45 == 71 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v44 - 24) + 8LL) == 15 )
        goto LABEL_48;
      goto LABEL_44;
    }
    if ( v45 != 55 )
      break;
    v126 = *(unsigned __int16 *)(v44 - 6);
    if ( ((v126 >> 7) & 6) != 0 || (v126 & 1) != 0 || !sub_1776630(*(_QWORD *)(v44 - 48), *(_QWORD *)(v42 - 24)) )
    {
LABEL_50:
      v47 = v42;
      a1 = (__int64 *)v206;
      v38 = (_QWORD *)v40;
      v11 = v47;
      goto LABEL_51;
    }
    v41 = *(_QWORD **)(v44 + 8);
    --v43;
    sub_170BC50(v206, v44 - 24);
    v37 = *(_QWORD *)(v42 + 40);
LABEL_48:
    if ( *(_QWORD **)(v37 + 48) == v41 || !v43 )
      goto LABEL_50;
  }
  if ( v45 != 54 )
  {
LABEL_44:
    v183 = v44 - 24;
    if ( (unsigned __int8)sub_15F3040(v44 - 24) || (unsigned __int8)sub_15F2ED0(v183) || sub_15F3330(v183) )
      goto LABEL_50;
    v37 = *(_QWORD *)(v42 + 40);
    --v43;
    goto LABEL_48;
  }
  v137 = v42;
  v38 = (_QWORD *)v40;
  a1 = (__int64 *)v206;
  v11 = v137;
  if ( v208 == v44 - 24 && sub_1776630(*(_QWORD *)(v44 - 48), v197) )
    return sub_170BC50((__int64)a1, v11);
LABEL_51:
  v48 = **(_QWORD **)(v11 - 24);
  if ( *(_BYTE *)(v48 + 8) == 16 )
    v48 = **(_QWORD **)(v48 + 16);
  v49 = *(_DWORD *)(v48 + 8);
  v50 = sub_15F2060(v11);
  if ( !sub_15E4690(v50, v49 >> 8) )
  {
    v51 = *(__int64 **)(v11 - 24);
    v52 = *((_BYTE *)v51 + 16);
    if ( v52 == 56 )
      v52 = *(_BYTE *)(v51[-3 * (*((_DWORD *)v51 + 5) & 0xFFFFFFF)] + 16);
    if ( v52 == 15 )
    {
      v144 = *v51;
      if ( *(_BYTE *)(*v51 + 8) == 16 )
        v144 = **(_QWORD **)(v144 + 16);
      v145 = *(_DWORD *)(v144 + 8);
      v146 = sub_15F2060(v11);
      if ( !sub_15E4690(v146, v145 >> 8) )
      {
        if ( *(_BYTE *)(v208 + 16) != 9 )
        {
          v147 = sub_1599EF0(*(__int64 ***)v208);
          if ( *(_QWORD *)(v11 - 48) )
          {
            v148 = *(_QWORD *)(v11 - 40);
            v149 = *(_QWORD *)(v11 - 32) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v149 = v148;
            if ( v148 )
              *(_QWORD *)(v148 + 16) = v149 | *(_QWORD *)(v148 + 16) & 3LL;
          }
          *(_QWORD *)(v11 - 48) = v147;
          if ( v147 )
          {
            v150 = *(_QWORD *)(v147 + 8);
            *(_QWORD *)(v11 - 40) = v150;
            if ( v150 )
              *(_QWORD *)(v150 + 16) = (v11 - 40) | *(_QWORD *)(v150 + 16) & 3LL;
            v151 = *(_QWORD *)(v11 - 32);
            v152 = v11 - 48;
            *(_QWORD *)(v152 + 16) = (v147 + 8) | v151 & 3;
            *(_QWORD *)(v147 + 8) = v152;
          }
          if ( *(_BYTE *)(v208 + 16) > 0x17u )
            sub_170B990(*a1, v208);
        }
        return 0;
      }
    }
  }
  if ( *(_BYTE *)(v208 + 16) == 9 )
    return sub_170BC50((__int64)a1, v11);
  while ( 1 )
  {
    while ( 1 )
    {
      v38 = (_QWORD *)v38[1];
      if ( !v38 )
        goto LABEL_275;
      v53 = *((_BYTE *)v38 - 8);
      if ( v53 != 78 )
        break;
      v122 = *(v38 - 6);
      if ( *(_BYTE *)(v122 + 16)
        || (*(_BYTE *)(v122 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v122 + 36) - 35) > 3 )
      {
        return 0;
      }
    }
    if ( v53 != 71 )
      break;
    if ( *(_BYTE *)(*(v38 - 3) + 8LL) != 15 )
      return 0;
  }
  if ( v53 != 26 || (*((_DWORD *)v38 - 1) & 0xFFFFFFF) != 1 )
    return 0;
  sub_1779430(a1, v11);
  return 0;
}
