// Function: sub_111AB00
// Address: 0x111ab00
//
_QWORD *__fastcall sub_111AB00(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  __int64 v4; // rbx
  __int16 v6; // r11
  unsigned int v7; // r15d
  __int64 v9; // r10
  __int64 v11; // rdx
  int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // r15
  unsigned __int64 v15; // r8
  __int64 *v16; // rbx
  __int64 v17; // r12
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 v22; // rax
  int v24; // edx
  __int64 v25; // rdx
  __int64 v26; // rsi
  char v27; // al
  __int64 v28; // rax
  int v29; // eax
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  int v32; // ebx
  unsigned int v33; // eax
  __int64 v34; // r10
  int v35; // eax
  __int64 v36; // r10
  __int64 v37; // r10
  unsigned int v38; // eax
  bool v39; // zf
  char v40; // al
  _BYTE *v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // r10
  __int64 v44; // rdi
  unsigned int v45; // ebx
  __int64 *v46; // rax
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // rdx
  __int64 v50; // r13
  __int64 v51; // r12
  _QWORD *v52; // rax
  __int64 *v53; // rax
  __int64 *v54; // r13
  __int64 v55; // rax
  __int64 v56; // r9
  __int64 v57; // rax
  unsigned int v58; // edx
  int v59; // eax
  int v60; // eax
  int v61; // ebx
  __int64 v62; // rdx
  __int64 v63; // rax
  _QWORD *v64; // rax
  _QWORD **v65; // rdx
  int v66; // esi
  int v67; // eax
  __int64 *v68; // rax
  __int64 v69; // rax
  int v70; // eax
  __int64 v71; // rax
  __int64 v72; // r9
  int v73; // edx
  __int64 v74; // rdx
  __int64 v75; // r13
  __int64 v76; // r12
  _QWORD *v77; // rax
  unsigned int v78; // edx
  int v79; // eax
  int v80; // eax
  bool v81; // al
  __int64 *v82; // rax
  unsigned int v83; // edx
  int v84; // eax
  bool v85; // al
  int v86; // eax
  bool v87; // al
  unsigned int v88; // edx
  int v89; // eax
  bool v90; // al
  int v91; // eax
  bool v92; // al
  __int16 v93; // ax
  unsigned __int64 v94; // rbx
  __int64 *v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rax
  __int64 v98; // r9
  unsigned __int64 *v99; // rdi
  unsigned int v100; // eax
  __int64 v101; // rax
  __int64 *v102; // rdi
  __int64 v103; // r15
  __int64 v104; // rax
  __int64 v105; // r9
  int v106; // eax
  int v107; // eax
  unsigned __int64 v108; // rax
  unsigned int v109; // ebx
  unsigned int v110; // ebx
  __int64 *v111; // rdx
  __int64 v112; // rbx
  unsigned int v113; // r15d
  int v114; // eax
  __int64 v115; // rcx
  char v116; // r11
  __int64 v117; // rax
  unsigned __int64 v118; // rcx
  __int64 v119; // rax
  _BYTE *v120; // rax
  __int64 v121; // rax
  char v122; // r11
  __int64 v123; // r10
  __int64 v124; // rdx
  __int64 v125; // rbx
  _QWORD *v126; // rax
  unsigned int v127; // eax
  unsigned int v128; // eax
  int v129; // edx
  __int64 v130; // rdx
  __int64 v131; // r13
  __int64 v132; // r12
  _QWORD *v133; // rax
  char v134; // dl
  __int64 v135; // rsi
  int v136; // eax
  int v137; // eax
  int v138; // eax
  unsigned int v139; // ebx
  int v140; // eax
  bool v141; // al
  unsigned __int64 *v142; // rdi
  int v143; // eax
  __int64 v144; // rax
  __int64 *v145; // rdi
  __int64 v146; // r15
  __int64 v147; // rax
  __int64 v148; // r9
  __int64 v149; // [rsp+8h] [rbp-198h]
  __int64 v150; // [rsp+8h] [rbp-198h]
  __int64 v151; // [rsp+8h] [rbp-198h]
  __int64 v152; // [rsp+8h] [rbp-198h]
  __int64 v153; // [rsp+8h] [rbp-198h]
  __int64 v154; // [rsp+8h] [rbp-198h]
  _BYTE *v155; // [rsp+10h] [rbp-190h]
  __int64 v156; // [rsp+18h] [rbp-188h]
  __int64 v157; // [rsp+20h] [rbp-180h]
  char v158; // [rsp+28h] [rbp-178h]
  __int64 v159; // [rsp+28h] [rbp-178h]
  __int64 v160; // [rsp+28h] [rbp-178h]
  __int64 v161; // [rsp+28h] [rbp-178h]
  __int64 v162; // [rsp+30h] [rbp-170h]
  __int64 v163; // [rsp+30h] [rbp-170h]
  __int64 v164; // [rsp+30h] [rbp-170h]
  __int64 v165; // [rsp+30h] [rbp-170h]
  __int64 v166; // [rsp+30h] [rbp-170h]
  __int64 v168; // [rsp+38h] [rbp-168h]
  const void **v169; // [rsp+38h] [rbp-168h]
  __int64 v170; // [rsp+38h] [rbp-168h]
  __int64 v171; // [rsp+38h] [rbp-168h]
  bool v172; // [rsp+38h] [rbp-168h]
  unsigned int **v173; // [rsp+38h] [rbp-168h]
  __int64 v174; // [rsp+38h] [rbp-168h]
  __int64 v175; // [rsp+38h] [rbp-168h]
  char v176; // [rsp+38h] [rbp-168h]
  __int64 v177; // [rsp+40h] [rbp-160h]
  __int64 v178; // [rsp+40h] [rbp-160h]
  __int64 v179; // [rsp+40h] [rbp-160h]
  __int64 v180; // [rsp+40h] [rbp-160h]
  char v181; // [rsp+40h] [rbp-160h]
  unsigned int **v182; // [rsp+48h] [rbp-158h]
  __int64 v183; // [rsp+48h] [rbp-158h]
  __int64 v184; // [rsp+48h] [rbp-158h]
  __int64 v185; // [rsp+48h] [rbp-158h]
  __int64 v186; // [rsp+48h] [rbp-158h]
  __int64 v187; // [rsp+48h] [rbp-158h]
  __int64 v188; // [rsp+48h] [rbp-158h]
  int v189; // [rsp+48h] [rbp-158h]
  __int64 v190; // [rsp+48h] [rbp-158h]
  int v191; // [rsp+48h] [rbp-158h]
  unsigned int v192; // [rsp+48h] [rbp-158h]
  unsigned int v193; // [rsp+48h] [rbp-158h]
  unsigned int v194; // [rsp+48h] [rbp-158h]
  unsigned int v195; // [rsp+48h] [rbp-158h]
  unsigned int v196; // [rsp+48h] [rbp-158h]
  __int64 v197; // [rsp+48h] [rbp-158h]
  __int64 v198; // [rsp+48h] [rbp-158h]
  __int64 v199; // [rsp+48h] [rbp-158h]
  char v200; // [rsp+48h] [rbp-158h]
  char v201; // [rsp+48h] [rbp-158h]
  __int64 v202; // [rsp+48h] [rbp-158h]
  __int64 v203; // [rsp+48h] [rbp-158h]
  __int16 v204; // [rsp+48h] [rbp-158h]
  __int64 v205; // [rsp+48h] [rbp-158h]
  __int16 v206; // [rsp+48h] [rbp-158h]
  __int16 v207; // [rsp+56h] [rbp-14Ah]
  int v208; // [rsp+58h] [rbp-148h]
  unsigned int v209; // [rsp+58h] [rbp-148h]
  unsigned int v210; // [rsp+58h] [rbp-148h]
  unsigned int v211; // [rsp+58h] [rbp-148h]
  unsigned int v212; // [rsp+58h] [rbp-148h]
  unsigned int v213; // [rsp+58h] [rbp-148h]
  int v214; // [rsp+64h] [rbp-13Ch] BYREF
  const void **v215; // [rsp+68h] [rbp-138h] BYREF
  unsigned __int64 v216; // [rsp+70h] [rbp-130h] BYREF
  unsigned int v217; // [rsp+78h] [rbp-128h]
  __int64 v218; // [rsp+80h] [rbp-120h] BYREF
  int v219; // [rsp+88h] [rbp-118h]
  __int64 v220; // [rsp+90h] [rbp-110h] BYREF
  int v221; // [rsp+98h] [rbp-108h]
  __int64 v222; // [rsp+A0h] [rbp-100h] BYREF
  unsigned int v223; // [rsp+A8h] [rbp-F8h]
  __int64 v224; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned int v225; // [rsp+B8h] [rbp-E8h]
  __int64 v226; // [rsp+C0h] [rbp-E0h] BYREF
  unsigned int v227; // [rsp+C8h] [rbp-D8h]
  __int64 v228; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned int v229; // [rsp+D8h] [rbp-C8h]
  __int128 v230; // [rsp+E0h] [rbp-C0h] BYREF
  __int128 v231; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v232; // [rsp+100h] [rbp-A0h]
  __int64 v233; // [rsp+110h] [rbp-90h] BYREF
  unsigned int v234; // [rsp+118h] [rbp-88h]
  __int64 v235; // [rsp+120h] [rbp-80h] BYREF
  unsigned int v236; // [rsp+128h] [rbp-78h]
  __int16 v237; // [rsp+130h] [rbp-70h]
  const void ***v238; // [rsp+140h] [rbp-60h] BYREF
  unsigned int v239; // [rsp+148h] [rbp-58h]
  __int64 v240; // [rsp+150h] [rbp-50h] BYREF
  unsigned int v241; // [rsp+158h] [rbp-48h]
  __int16 v242; // [rsp+160h] [rbp-40h]

  v4 = *(_QWORD *)(a3 - 32);
  if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a3 + 80) )
    goto LABEL_253;
  v6 = *(_WORD *)(a2 + 2);
  v7 = *(_DWORD *)(v4 + 36);
  v9 = a2;
  v207 = v6 & 0x3F;
  v208 = v6 & 0x3F;
  if ( v7 == 359 )
  {
LABEL_25:
    v182 = (unsigned int **)a1[4];
    v22 = *(_QWORD *)(a3 + 16);
    if ( !v22 || (v21 = *(_QWORD *)(v22 + 8)) != 0 )
    {
LABEL_26:
      if ( (unsigned int)(v208 - 32) > 1 )
        goto LABEL_12;
      return sub_1118A30((__int64)a1, v9, a3, (__int64 **)a4);
    }
    v24 = *(_DWORD *)(a3 + 4);
    LOBYTE(v239) = 0;
    v25 = v24 & 0x7FFFFFF;
    v155 = *(_BYTE **)(a3 - 32 * v25);
    v26 = *(_QWORD *)(a3 + 32 * (1 - v25));
    v238 = &v215;
    v156 = v26;
    v27 = sub_991580((__int64)&v238, v26);
    v9 = a2;
    if ( !v27 )
      goto LABEL_86;
    v217 = 1;
    v28 = *(_QWORD *)(a3 - 32);
    v216 = 0;
    if ( v28 && !*(_BYTE *)v28 && *(_QWORD *)(v28 + 24) == *(_QWORD *)(a3 + 80) )
    {
      v29 = *(_DWORD *)(v28 + 36);
      if ( v29 == 359 )
      {
        v30 = *((_DWORD *)a4 + 2);
        v239 = v30;
        if ( v30 <= 0x40 )
        {
          v31 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
          if ( !v30 )
            v31 = 0;
          v238 = (const void ***)v31;
LABEL_37:
          v168 = v9;
          v216 = v31;
          v217 = v30;
          v239 = 0;
          sub_969240((__int64 *)&v238);
          v162 = v168;
          v158 = sub_B532C0((__int64)&v216, a4, v208);
          v32 = sub_B5B690(a3);
          v169 = v215;
          v33 = sub_B5B5E0(a3);
          sub_AB3450((__int64)&v222, v33, (__int64)v169, v32);
          v34 = v162;
          if ( v158 )
          {
            sub_ABB300((__int64)&v238, (__int64)&v222);
            sub_1110A30(&v222, (__int64 *)&v238);
            sub_1110A30(&v224, &v240);
            sub_969240(&v240);
            sub_969240((__int64 *)&v238);
            v34 = v162;
          }
          v170 = v34;
          sub_AB1A50((__int64)&v226, v208, (__int64)a4);
          v35 = sub_B5B5E0(a3);
          v36 = v170;
          if ( v35 == 13 )
          {
            sub_9865C0((__int64)&v230, (__int64)v215);
            sub_AADBC0((__int64)&v233, (__int64 *)&v230);
            sub_AB51C0((__int64)&v238, (__int64)&v226, (__int64)&v233);
            sub_1110A30(&v226, (__int64 *)&v238);
            sub_1110A30(&v228, &v240);
            sub_969240(&v240);
            sub_969240((__int64 *)&v238);
            sub_969240(&v235);
            sub_969240(&v233);
            sub_969240((__int64 *)&v230);
            v37 = v170;
          }
          else
          {
            DWORD2(v230) = *((_DWORD *)v215 + 2);
            if ( DWORD2(v230) > 0x40 )
            {
              sub_C43780((__int64)&v230, v215);
              v36 = v170;
            }
            else
            {
              *(_QWORD *)&v230 = *v215;
            }
            v149 = v36;
            sub_AADBC0((__int64)&v233, (__int64 *)&v230);
            sub_AB4F10((__int64)&v238, (__int64)&v226, (__int64)&v233);
            v37 = v149;
            if ( v227 > 0x40 && v226 )
            {
              j_j___libc_free_0_0(v226);
              v37 = v149;
            }
            v226 = (__int64)v238;
            v38 = v239;
            v239 = 0;
            v227 = v38;
            if ( v229 > 0x40 && v228 )
            {
              v150 = v37;
              j_j___libc_free_0_0(v228);
              v37 = v150;
              v228 = v240;
              v229 = v241;
              if ( v239 > 0x40 && v238 )
              {
                j_j___libc_free_0_0(v238);
                v37 = v150;
              }
            }
            else
            {
              v228 = v240;
              v229 = v241;
            }
            if ( v236 > 0x40 && v235 )
            {
              v151 = v37;
              j_j___libc_free_0_0(v235);
              v37 = v151;
            }
            if ( v234 > 0x40 && v233 )
            {
              v152 = v37;
              j_j___libc_free_0_0(v233);
              v37 = v152;
            }
            if ( DWORD2(v230) > 0x40 && (_QWORD)v230 )
            {
              v153 = v37;
              j_j___libc_free_0_0(v230);
              v37 = v153;
            }
          }
          v39 = v158 == 0;
          v159 = v37;
          v232 = 0;
          v230 = 0;
          v231 = 0;
          if ( v39 )
          {
            sub_ABB730((__int64)&v238, (__int64)&v222, (__int64)&v226);
            v9 = v159;
            if ( (_BYTE)v232 )
            {
LABEL_61:
              v160 = v9;
              if ( (_BYTE)v242 )
              {
                sub_1110A30((__int64 *)&v230, (__int64 *)&v238);
                sub_1110A30((__int64 *)&v231, &v240);
              }
              else
              {
                LOBYTE(v232) = 0;
                sub_969240((__int64 *)&v231);
                sub_969240((__int64 *)&v230);
              }
              v9 = v160;
              if ( !(_BYTE)v242 )
              {
                v40 = v232;
                goto LABEL_65;
              }
LABEL_210:
              v161 = v9;
              LOBYTE(v242) = 0;
              sub_969240(&v240);
              sub_969240((__int64 *)&v238);
              v40 = v232;
              v9 = v161;
LABEL_65:
              if ( v40 )
              {
                v154 = v9;
                v219 = 1;
                v218 = 0;
                v221 = 1;
                v220 = 0;
                sub_AAF830((__int64)&v230, &v214, (__int64)&v218, &v220);
                v242 = 257;
                v41 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v156 + 8), (__int64)&v220);
                v157 = sub_929C50(v182, v155, v41, (__int64)&v238, 0, 0);
                v177 = sub_AD8D80(*(_QWORD *)(v156 + 8), (__int64)&v218);
                v237 = 257;
                v42 = sub_BD2C40(72, unk_3F10FD0);
                v43 = v154;
                v21 = (__int64)v42;
                if ( v42 )
                {
                  sub_1113300((__int64)v42, v214, v157, v177, (__int64)&v233);
                  v43 = v154;
                }
                v183 = v43;
                sub_969240(&v220);
                sub_969240(&v218);
                v9 = v183;
                if ( (_BYTE)v232 )
                {
                  LOBYTE(v232) = 0;
                  sub_969240((__int64 *)&v231);
                  sub_969240((__int64 *)&v230);
                  v9 = v183;
                }
              }
LABEL_70:
              if ( v229 > 0x40 && v228 )
              {
                v184 = v9;
                j_j___libc_free_0_0(v228);
                v9 = v184;
              }
              if ( v227 > 0x40 && v226 )
              {
                v185 = v9;
                j_j___libc_free_0_0(v226);
                v9 = v185;
              }
              if ( v225 > 0x40 && v224 )
              {
                v186 = v9;
                j_j___libc_free_0_0(v224);
                v9 = v186;
              }
              if ( v223 > 0x40 && v222 )
              {
                v187 = v9;
                j_j___libc_free_0_0(v222);
                v9 = v187;
              }
              if ( v217 <= 0x40 || (v44 = v216) == 0 )
              {
LABEL_85:
                if ( v21 )
                  return (_QWORD *)v21;
                goto LABEL_86;
              }
LABEL_84:
              v188 = v9;
              j_j___libc_free_0_0(v44);
              v9 = v188;
              goto LABEL_85;
            }
          }
          else
          {
            sub_ABB970((__int64)&v238, (__int64)&v222, (__int64)&v226);
            v9 = v159;
            if ( (_BYTE)v232 )
              goto LABEL_61;
          }
          if ( !(_BYTE)v242 )
            goto LABEL_70;
          v127 = v239;
          LOBYTE(v232) = 1;
          v239 = 0;
          DWORD2(v230) = v127;
          *(_QWORD *)&v230 = v238;
          v128 = v241;
          v241 = 0;
          DWORD2(v231) = v128;
          *(_QWORD *)&v231 = v240;
          goto LABEL_210;
        }
        v134 = 1;
        v135 = -1;
      }
      else
      {
        if ( v29 != 371 )
          goto LABEL_253;
        v30 = *((_DWORD *)a4 + 2);
        v239 = v30;
        if ( v30 <= 0x40 )
        {
          v238 = 0;
          v31 = 0;
          goto LABEL_37;
        }
        v134 = 0;
        v135 = 0;
      }
      sub_C43690((__int64)&v238, v135, v134);
      v9 = a2;
      if ( v217 > 0x40 && v216 )
      {
        j_j___libc_free_0_0(v216);
        v31 = (unsigned __int64)v238;
        v30 = v239;
        v9 = a2;
      }
      else
      {
        v31 = (unsigned __int64)v238;
        v30 = v239;
      }
      goto LABEL_37;
    }
LABEL_253:
    BUG();
  }
  if ( v7 > 0x167 )
  {
    if ( v7 == 362 )
    {
LABEL_8:
      switch ( v6 & 0x3F )
      {
        case ' ':
        case '!':
          v58 = *((_DWORD *)a4 + 2);
          if ( v58 <= 0x40 )
          {
            if ( *a4 )
            {
              if ( *a4 == 1 )
                goto LABEL_114;
              if ( !v58 )
              {
LABEL_203:
                v61 = (v207 == 32) + 35;
LABEL_115:
                if ( v7 == 313 )
                {
                  v197 = v9;
                  v93 = sub_B52E90(v61);
                  v9 = v197;
                  LOWORD(v61) = v93;
                }
                v171 = v9;
                v62 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
                v178 = *(_QWORD *)(a3 - 32 * v62);
                v63 = *(_QWORD *)(a3 + 32 * (1 - v62));
                v242 = 257;
                v190 = v63;
                v64 = sub_BD2C40(72, unk_3F10FD0);
                v9 = v171;
                v21 = (__int64)v64;
                if ( v64 )
                {
                  v65 = *(_QWORD ***)(v178 + 8);
                  v66 = *((unsigned __int8 *)v65 + 8);
                  if ( (unsigned int)(v66 - 17) > 1 )
                  {
                    v69 = sub_BCB2A0(*v65);
                  }
                  else
                  {
                    v67 = *((_DWORD *)v65 + 8);
                    BYTE4(v233) = (_BYTE)v66 == 18;
                    LODWORD(v233) = v67;
                    v68 = (__int64 *)sub_BCB2A0(*v65);
                    v69 = sub_BCE1B0(v68, v233);
                  }
                  sub_B523C0(v21, v69, 53, v61, v178, v190, (__int64)&v238, 0, 0, 0);
                  return (_QWORD *)v21;
                }
                goto LABEL_86;
              }
              v141 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v58) == *a4;
LABEL_242:
              if ( !v141 )
              {
                if ( (unsigned int)(v208 - 32) > 1 )
                  goto LABEL_12;
                return sub_1118A30((__int64)a1, v9, a3, (__int64 **)a4);
              }
              goto LABEL_203;
            }
          }
          else
          {
            v189 = *((_DWORD *)a4 + 2);
            v59 = sub_C444A0((__int64)a4);
            v9 = a2;
            if ( v189 != v59 )
            {
              v60 = sub_C444A0((__int64)a4);
              v9 = a2;
              if ( v60 == v189 - 1 )
              {
LABEL_114:
                v61 = 3 * (v207 != 32) + 34;
                goto LABEL_115;
              }
              v140 = sub_C445E0((__int64)a4);
              v9 = a2;
              v141 = v189 == v140;
              goto LABEL_242;
            }
          }
          v61 = v208;
          goto LABEL_115;
        case '"':
          v78 = *((_DWORD *)a4 + 2);
          if ( v78 <= 0x40 )
          {
            if ( !*a4 || !v78 )
              goto LABEL_26;
            v81 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v78) == *a4;
          }
          else
          {
            v191 = *((_DWORD *)a4 + 2);
            v172 = v78 == 0;
            v79 = sub_C444A0((__int64)a4);
            v9 = a2;
            if ( v191 == v79 || v172 )
              goto LABEL_26;
            v80 = sub_C445E0((__int64)a4);
            v9 = a2;
            v81 = v191 == v80;
          }
          if ( !v81 )
          {
LABEL_132:
            v61 = 36;
            goto LABEL_115;
          }
          goto LABEL_26;
        case '$':
          v192 = *((_DWORD *)a4 + 2);
          if ( v192 > 0x40 )
          {
            v107 = sub_C444A0((__int64)a4);
            v9 = a2;
            if ( v192 - v107 > 0x40 )
              goto LABEL_136;
            v82 = *(__int64 **)*a4;
          }
          else
          {
            v82 = (__int64 *)*a4;
          }
          if ( (unsigned __int64)v82 <= 1 )
            goto LABEL_26;
LABEL_136:
          v61 = 35;
          goto LABEL_115;
        case '&':
          v83 = *((_DWORD *)a4 + 2);
          if ( !v83 )
            goto LABEL_136;
          if ( v83 <= 0x40 )
          {
            v85 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v83) == *a4;
          }
          else
          {
            v193 = *((_DWORD *)a4 + 2);
            v84 = sub_C445E0((__int64)a4);
            v83 = v193;
            v9 = a2;
            v85 = v193 == v84;
          }
          if ( v85 )
            goto LABEL_136;
          if ( v83 <= 0x40 )
          {
            v87 = *a4 == 0;
          }
          else
          {
            v179 = v9;
            v194 = v83;
            v86 = sub_C444A0((__int64)a4);
            v9 = v179;
            v87 = v194 == v86;
          }
          if ( !v87 )
            goto LABEL_26;
          v61 = 34;
          goto LABEL_115;
        case '(':
          v88 = *((_DWORD *)a4 + 2);
          if ( v88 <= 0x40 )
          {
            v90 = *a4 == 0;
          }
          else
          {
            v195 = *((_DWORD *)a4 + 2);
            v89 = sub_C444A0((__int64)a4);
            v88 = v195;
            v9 = a2;
            v90 = v195 == v89;
          }
          if ( v90 )
            goto LABEL_132;
          if ( v88 <= 0x40 )
          {
            v92 = *a4 == 1;
          }
          else
          {
            v180 = v9;
            v196 = v88;
            v91 = sub_C444A0((__int64)a4);
            v9 = v180;
            v92 = v196 - 1 == v91;
          }
          if ( !v92 )
            goto LABEL_26;
          v61 = 37;
          goto LABEL_115;
        default:
          goto LABEL_26;
      }
    }
    if ( v7 != 371 )
      goto LABEL_26;
    goto LABEL_25;
  }
  if ( v7 != 66 )
  {
    if ( v7 != 313 )
      goto LABEL_26;
    goto LABEL_8;
  }
  v11 = *(_QWORD *)(a3 + 16);
  v12 = v6 & 0x3F;
  if ( v11 && !*(_QWORD *)(v11 + 8) )
  {
    if ( (unsigned __int16)(v207 - 32) <= 2u )
    {
      v45 = *((_DWORD *)a4 + 2);
      if ( v45 > 0x40 )
      {
        v204 = *(_WORD *)(a2 + 2);
        v136 = sub_C444A0((__int64)a4);
        LOBYTE(v6) = v204;
        v9 = a2;
        if ( v45 - v136 > 0x40 )
          goto LABEL_92;
        v46 = *(__int64 **)*a4;
      }
      else
      {
        v46 = (__int64 *)*a4;
      }
      if ( v46 != (__int64 *)1 )
      {
LABEL_92:
        v12 = v208;
        goto LABEL_10;
      }
      goto LABEL_179;
    }
    if ( v207 != 36 )
      goto LABEL_92;
    v110 = *((_DWORD *)a4 + 2);
    if ( v110 > 0x40 )
    {
      v206 = *(_WORD *)(a2 + 2);
      v138 = sub_C444A0((__int64)a4);
      LOBYTE(v6) = v206;
      v9 = a2;
      v139 = v110 - v138;
      v12 = 36;
      if ( v139 > 0x40 )
        goto LABEL_10;
      v111 = *(__int64 **)*a4;
    }
    else
    {
      v111 = (__int64 *)*a4;
    }
    v12 = 36;
    if ( v111 == (__int64 *)2 )
    {
LABEL_179:
      v181 = v6;
      v173 = (unsigned int **)a1[4];
      v198 = v9;
      v112 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      sub_9AC3E0((__int64)&v233, v112, a1[12], 0, a1[16], v9, a1[15], 1);
      v113 = v236;
      v163 = v198;
      if ( v236 > 0x40 )
      {
        v137 = sub_C44630((__int64)&v235);
        v9 = v198;
        if ( v137 == 1 )
        {
          v227 = v113;
          v242 = 257;
          sub_C43780((__int64)&v226, (const void **)&v235);
          v113 = v227;
          v116 = v181;
          v9 = v198;
          if ( v227 > 0x40 )
          {
            sub_C43D10((__int64)&v226);
            v113 = v227;
            v119 = v226;
            v9 = v198;
            v116 = v181;
            goto LABEL_185;
          }
          v115 = v226;
LABEL_182:
          v117 = ~v115;
          v118 = 0;
          if ( v113 )
            v118 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v113;
          v119 = v118 & v117;
          v226 = v119;
LABEL_185:
          DWORD2(v230) = v113;
          *(_QWORD *)&v230 = v119;
          v227 = 0;
          v164 = v9;
          v200 = v116;
          v120 = (_BYTE *)sub_AD6220(*(_QWORD *)(v112 + 8), (__int64)&v230);
          v121 = sub_A82350(v173, (_BYTE *)v112, v120, (__int64)&v238);
          v122 = v200;
          v123 = v164;
          v124 = v121;
          if ( DWORD2(v230) > 0x40 && (_QWORD)v230 )
          {
            v174 = v121;
            j_j___libc_free_0_0(v230);
            v123 = v164;
            v124 = v174;
            v122 = v200;
          }
          if ( v227 > 0x40 && v226 )
          {
            v165 = v123;
            v175 = v124;
            v201 = v122;
            j_j___libc_free_0_0(v226);
            v123 = v165;
            v124 = v175;
            v122 = v201;
          }
          v202 = v123;
          v166 = v124;
          v176 = v122;
          v125 = sub_AD6530(*(_QWORD *)(v112 + 8), v112);
          v242 = 257;
          v126 = sub_BD2C40(72, unk_3F10FD0);
          v9 = v202;
          v21 = (__int64)v126;
          if ( v126 )
          {
            sub_1113300((__int64)v126, ((v176 & 0x3B) != 32) + 32, v166, v125, (__int64)&v238);
            v9 = v202;
          }
          if ( v236 > 0x40 && v235 )
          {
            v203 = v9;
            j_j___libc_free_0_0(v235);
            v9 = v203;
          }
          if ( v234 <= 0x40 )
            goto LABEL_85;
          v44 = v233;
          if ( !v233 )
            goto LABEL_85;
          goto LABEL_84;
        }
        if ( v235 )
        {
          j_j___libc_free_0_0(v235);
          v9 = v198;
        }
      }
      else
      {
        v199 = v235;
        v114 = sub_39FAC40(v235);
        v115 = v199;
        v116 = v181;
        v9 = v163;
        if ( v114 == 1 )
        {
          v242 = 257;
          goto LABEL_182;
        }
      }
      if ( v234 > 0x40 && v233 )
      {
        v205 = v9;
        j_j___libc_free_0_0(v233);
        v9 = v205;
      }
LABEL_86:
      v12 = *(_WORD *)(v9 + 2) & 0x3F;
    }
  }
LABEL_10:
  if ( (unsigned int)(v12 - 32) <= 1 )
    return sub_1118A30((__int64)a1, v9, a3, (__int64 **)a4);
  v4 = *(_QWORD *)(a3 - 32);
  if ( !v4 )
    goto LABEL_253;
LABEL_12:
  if ( *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a3 + 80) )
    goto LABEL_253;
  v13 = *(_DWORD *)(v4 + 36);
  v14 = *(_QWORD *)(a3 + 8);
  v15 = *((unsigned int *)a4 + 2);
  if ( v13 == 67 )
  {
    v47 = *(_QWORD *)(a3 + 16);
    if ( !v47 || *(_QWORD *)(v47 + 8) )
      return 0;
    if ( v207 == 34 )
    {
      v213 = *((_DWORD *)a4 + 2);
      if ( !sub_986EE0((__int64)a4, (unsigned int)v15) )
        return 0;
      v142 = a4;
      v54 = (__int64 *)&v230;
      v143 = sub_10E0080(v142, 0xFFFFFFFFFFFFFFFFLL);
      sub_F0A5D0((__int64)&v230, v213, v143 + 1);
      v242 = 257;
      v144 = sub_AD6530(v14, v213);
      v145 = (__int64 *)a1[4];
      v146 = v144;
      LODWORD(v144) = *(_DWORD *)(a3 + 4);
      v237 = 257;
      v147 = sub_10BC480(v145, *(_QWORD *)(a3 - 32 * (v144 & 0x7FFFFFF)), (__int64)&v230, (__int64)&v233);
      v57 = sub_B52500(53, 32, v147, v146, (__int64)&v238, v148, 0, 0);
    }
    else
    {
      if ( v207 != 36 )
        return 0;
      v210 = *((_DWORD *)a4 + 2);
      if ( sub_986EE0((__int64)a4, 1u) || sub_AAD8D0((__int64)a4, v210) )
        return 0;
      v99 = a4;
      v54 = (__int64 *)&v230;
      v100 = sub_10E0080(v99, 0xFFFFFFFFFFFFFFFFLL);
      sub_F0A5D0((__int64)&v230, v210, v100);
      v242 = 257;
      v101 = sub_AD6530(v14, v210);
      v102 = (__int64 *)a1[4];
      v103 = v101;
      LODWORD(v101) = *(_DWORD *)(a3 + 4);
      v237 = 257;
      v104 = sub_10BC480(v102, *(_QWORD *)(a3 - 32 * (v101 & 0x7FFFFFF)), (__int64)&v230, (__int64)&v233);
      v57 = sub_B52500(53, 33, v104, v103, (__int64)&v238, v105, 0, 0);
    }
    goto LABEL_110;
  }
  if ( v13 > 0x43 )
  {
    if ( v13 == 338 && sub_B532B0(v208) )
    {
      if ( sub_9867B0((__int64)a4) )
      {
        v73 = *(_DWORD *)(a3 + 4);
        v242 = 257;
        v74 = v73 & 0x7FFFFFF;
        v75 = *(_QWORD *)(a3 - 32 * v74);
        v76 = *(_QWORD *)(a3 + 32 * (1 - v74));
        v77 = sub_BD2C40(72, unk_3F10FD0);
        v21 = (__int64)v77;
        if ( v77 )
          sub_1113300((__int64)v77, v208, v75, v76, (__int64)&v238);
        return (_QWORD *)v21;
      }
      if ( v207 == 40 )
      {
        if ( sub_D94040((__int64)a4) )
        {
          v129 = *(_DWORD *)(a3 + 4);
          v242 = 257;
          v130 = v129 & 0x7FFFFFF;
          v131 = *(_QWORD *)(a3 - 32 * v130);
          v132 = *(_QWORD *)(a3 + 32 * (1 - v130));
          v133 = sub_BD2C40(72, unk_3F10FD0);
          v21 = (__int64)v133;
          if ( v133 )
            sub_1113300((__int64)v133, 41, v131, v132, (__int64)&v238);
          return (_QWORD *)v21;
        }
      }
      else if ( v207 == 38 && sub_986760((__int64)a4) )
      {
        v48 = *(_DWORD *)(a3 + 4);
        v242 = 257;
        v49 = v48 & 0x7FFFFFF;
        v50 = *(_QWORD *)(a3 - 32 * v49);
        v51 = *(_QWORD *)(a3 + 32 * (1 - v49));
        v52 = sub_BD2C40(72, unk_3F10FD0);
        v21 = (__int64)v52;
        if ( v52 )
          sub_1113300((__int64)v52, 39, v50, v51, (__int64)&v238);
        return (_QWORD *)v21;
      }
    }
    return 0;
  }
  if ( v13 == 65 )
  {
    if ( v207 != 34 )
    {
      if ( v207 != 36 )
        return 0;
      if ( (unsigned int)v15 > 0x40 )
      {
        v211 = *((_DWORD *)a4 + 2);
        v106 = sub_C444A0((__int64)a4);
        LODWORD(v15) = v211;
        if ( v211 - v106 > 0x40 )
          return 0;
        v53 = *(__int64 **)*a4;
        if ( !v53 || (unsigned __int64)v53 > v211 )
          return 0;
      }
      else
      {
        v53 = (__int64 *)*a4;
        if ( !*a4 || (unsigned __int64)v53 > (unsigned int)v15 )
          return 0;
      }
      v54 = &v233;
      sub_F0A5D0((__int64)&v233, v15, v15 - (_DWORD)v53);
      v242 = 257;
      v55 = sub_AD8D80(v14, (__int64)&v233);
      v57 = sub_B52500(
              53,
              34,
              *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)),
              v55,
              (__int64)&v238,
              v56,
              0,
              0);
LABEL_110:
      v21 = v57;
      sub_969240(v54);
      return (_QWORD *)v21;
    }
    v94 = (unsigned int)v15;
    if ( (unsigned int)v15 > 0x40 )
    {
      v212 = *((_DWORD *)a4 + 2);
      if ( v212 - (unsigned int)sub_C444A0((__int64)a4) > 0x40 )
        return 0;
      v108 = *(_QWORD *)*a4;
      if ( v94 <= v108 )
        return 0;
      v234 = v212;
      v109 = v212 - 1 - v108;
      sub_C43690((__int64)&v233, 0, 0);
      v96 = 1LL << v109;
      if ( v234 > 0x40 )
      {
        *(_QWORD *)(v233 + 8LL * (v109 >> 6)) |= v96;
        goto LABEL_157;
      }
    }
    else
    {
      v95 = (__int64 *)*a4;
      if ( (unsigned int)v15 <= *a4 )
        return 0;
      v234 = *((_DWORD *)a4 + 2);
      v233 = 0;
      v96 = 1LL << ((unsigned __int8)v15 - 1 - (unsigned __int8)v95);
    }
    v233 |= v96;
LABEL_157:
    v242 = 257;
    v97 = sub_AD8D80(v14, (__int64)&v233);
    v21 = sub_B52500(53, 36, *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), v97, (__int64)&v238, v98, 0, 0);
    sub_969240(&v233);
    return (_QWORD *)v21;
  }
  if ( v13 != 66 )
    return 0;
  v16 = (__int64 *)(unsigned int)(v15 - 1);
  v17 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  if ( (unsigned int)v15 > 0x40 )
  {
    v209 = *((_DWORD *)a4 + 2);
    v70 = sub_C444A0((__int64)a4);
    v15 = v209;
    if ( v209 - v70 <= 0x40 )
    {
      v18 = *(__int64 **)*a4;
      if ( v16 == v18 && v207 == 34 )
        goto LABEL_124;
      goto LABEL_20;
    }
    return 0;
  }
  v18 = (__int64 *)*a4;
  if ( v16 == (__int64 *)*a4 && v207 == 34 )
  {
LABEL_124:
    v242 = 257;
    v71 = sub_AD62B0(v14);
    return (_QWORD *)sub_B52500(53, 32, v17, v71, (__int64)&v238, v72, 0, 0);
  }
LABEL_20:
  if ( v207 != 36 || v18 != (__int64 *)v15 )
    return 0;
  v242 = 257;
  v19 = sub_AD62B0(v14);
  return (_QWORD *)sub_B52500(53, 33, v17, v19, (__int64)&v238, v20, 0, 0);
}
