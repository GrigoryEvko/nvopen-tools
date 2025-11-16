// Function: sub_1EFF2D0
// Address: 0x1eff2d0
//
unsigned __int64 __fastcall sub_1EFF2D0(
        _QWORD *a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rax
  unsigned __int8 *v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  unsigned __int8 *v18; // rsi
  unsigned __int8 *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r15
  __int64 v23; // r12
  unsigned __int8 *v24; // rsi
  _QWORD *v25; // r15
  _QWORD *v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 *v29; // r15
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  _QWORD *v34; // r8
  __int64 v35; // r9
  __int64 v36; // rsi
  __int64 v37; // rsi
  int v38; // eax
  __int64 v39; // rax
  int v40; // edx
  __int64 v41; // rdx
  _QWORD *v42; // rax
  __int64 v43; // rcx
  unsigned __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rdx
  __int64 *v54; // rax
  __int64 v55; // rcx
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // r14
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // r15
  __int64 v64; // rax
  _QWORD *v65; // r14
  __int64 v66; // rax
  unsigned __int8 *v67; // rsi
  _QWORD *v68; // rax
  unsigned __int8 *v69; // rsi
  __int64 v70; // rax
  __int64 v71; // r15
  __int64 v72; // r10
  _QWORD *v73; // rax
  __int64 v74; // r15
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rdx
  unsigned __int8 *v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rax
  _QWORD *v82; // rbx
  __int64 v83; // rax
  unsigned __int8 *v84; // rsi
  _QWORD *v85; // rax
  _QWORD *v86; // rax
  unsigned __int64 *v87; // r15
  __int64 v88; // rax
  unsigned __int64 v89; // rcx
  __int64 v90; // rsi
  __int64 v91; // rdx
  unsigned __int8 *v92; // rsi
  _QWORD *v93; // rax
  __int64 v94; // r10
  __int64 *v95; // r15
  __int64 v96; // rcx
  __int64 v97; // rax
  __int64 v98; // rsi
  __int64 v99; // rdx
  unsigned __int8 *v100; // rsi
  _QWORD *v101; // rax
  __int64 v102; // r9
  _QWORD **v103; // rax
  __int64 *v104; // rax
  __int64 v105; // rax
  __int64 v106; // r9
  unsigned __int64 v107; // rsi
  __int64 v108; // rax
  __int64 v109; // rsi
  __int64 v110; // rdx
  unsigned __int8 *v111; // rsi
  _QWORD *v112; // rax
  __int64 *v113; // r14
  __int64 v114; // rax
  __int64 v115; // rcx
  __int64 v116; // rsi
  unsigned __int8 *v117; // rsi
  __int64 v118; // rax
  __int64 v119; // r15
  __int64 v120; // r14
  __int64 v121; // rsi
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rcx
  _QWORD *v125; // r8
  __int64 v126; // r9
  __int64 v127; // rsi
  __int64 v128; // rsi
  int v129; // eax
  __int64 v130; // rax
  int v131; // edx
  __int64 v132; // rdx
  _QWORD *v133; // rax
  __int64 v134; // rcx
  unsigned __int64 v135; // rdx
  __int64 v136; // rdx
  __int64 v137; // rdx
  __int64 v138; // rax
  __int64 v139; // rcx
  __int64 v140; // rdx
  int v141; // eax
  __int64 v142; // rax
  int v143; // edx
  __int64 v144; // rdx
  __int64 *v145; // rax
  __int64 v146; // rcx
  unsigned __int64 v147; // rdx
  __int64 v148; // rdx
  __int64 v149; // rdx
  __int64 v150; // r14
  __int64 v151; // rcx
  __int64 v152; // rdx
  __int64 v153; // rax
  double v154; // xmm4_8
  double v155; // xmm5_8
  unsigned __int64 result; // rax
  __int64 v157; // rbx
  _QWORD *v158; // r15
  _QWORD *v159; // r12
  unsigned __int64 v160; // rsi
  __int64 v161; // rax
  __int64 v162; // rsi
  __int64 v163; // rdx
  unsigned __int8 *v164; // rsi
  __int64 v165; // rax
  __int64 v166; // rax
  __int64 v167; // rdx
  __int64 v168; // rax
  __int64 v169; // rax
  __int64 v170; // r12
  _QWORD *v171; // rax
  unsigned __int64 *v172; // r12
  __int64 v173; // rax
  unsigned __int64 v174; // rsi
  __int64 v175; // rsi
  __int64 v176; // rdx
  unsigned __int8 *v177; // rsi
  __int64 v178; // [rsp+10h] [rbp-170h]
  __int64 v179; // [rsp+18h] [rbp-168h]
  __int64 v180; // [rsp+20h] [rbp-160h]
  __int64 v181; // [rsp+28h] [rbp-158h]
  unsigned int v182; // [rsp+34h] [rbp-14Ch]
  __int64 v183; // [rsp+38h] [rbp-148h]
  __int64 v184; // [rsp+40h] [rbp-140h]
  __int64 *v185; // [rsp+48h] [rbp-138h]
  __int64 v186; // [rsp+50h] [rbp-130h]
  __int64 v188; // [rsp+70h] [rbp-110h]
  __int64 *v189; // [rsp+70h] [rbp-110h]
  __int64 v190; // [rsp+70h] [rbp-110h]
  __int64 v191; // [rsp+70h] [rbp-110h]
  _QWORD *v192; // [rsp+70h] [rbp-110h]
  __int64 v193; // [rsp+70h] [rbp-110h]
  __int64 v194; // [rsp+70h] [rbp-110h]
  __int64 v195; // [rsp+70h] [rbp-110h]
  _QWORD *v196; // [rsp+70h] [rbp-110h]
  unsigned __int64 *v197; // [rsp+70h] [rbp-110h]
  __int64 v198; // [rsp+70h] [rbp-110h]
  _QWORD *v199; // [rsp+78h] [rbp-108h]
  __int64 v200; // [rsp+78h] [rbp-108h]
  __int64 v201; // [rsp+78h] [rbp-108h]
  unsigned __int64 *v202; // [rsp+78h] [rbp-108h]
  _QWORD *v203; // [rsp+80h] [rbp-100h]
  __int64 v204; // [rsp+80h] [rbp-100h]
  __int64 i; // [rsp+88h] [rbp-F8h]
  __int64 *v206; // [rsp+88h] [rbp-F8h]
  __int64 v207; // [rsp+88h] [rbp-F8h]
  unsigned __int8 *v208; // [rsp+98h] [rbp-E8h] BYREF
  _QWORD v209[4]; // [rsp+A0h] [rbp-E0h] BYREF
  char *v210; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v211; // [rsp+C8h] [rbp-B8h]
  __int16 v212; // [rsp+D0h] [rbp-B0h]
  char *v213; // [rsp+E0h] [rbp-A0h] BYREF
  char *v214; // [rsp+E8h] [rbp-98h]
  __int16 v215; // [rsp+F0h] [rbp-90h]
  unsigned __int8 *v216; // [rsp+100h] [rbp-80h] BYREF
  _QWORD *v217; // [rsp+108h] [rbp-78h]
  __int64 *v218; // [rsp+110h] [rbp-70h]
  _QWORD *v219; // [rsp+118h] [rbp-68h]
  __int64 v220; // [rsp+120h] [rbp-60h]
  int v221; // [rsp+128h] [rbp-58h]
  __int64 v222; // [rsp+130h] [rbp-50h]
  __int64 v223; // [rsp+138h] [rbp-48h]

  v10 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
  v186 = a1[-3 * v10];
  v11 = a1[3 * (1 - v10)];
  v184 = a1[3 * (2 - v10)];
  v12 = *a1;
  v178 = a1[3 * (3 - v10)];
  v13 = 0;
  if ( *(_BYTE *)(*a1 + 8LL) == 16 )
    v13 = v12;
  v180 = v13;
  v14 = (_QWORD *)sub_16498A0((__int64)a1);
  v15 = (unsigned __int8 *)a1[6];
  v216 = 0;
  v219 = v14;
  v16 = (_QWORD *)a1[5];
  v220 = 0;
  v203 = v16;
  v217 = v16;
  v221 = 0;
  v222 = 0;
  v223 = 0;
  v185 = a1 + 3;
  v218 = a1 + 3;
  v213 = (char *)v15;
  if ( v15 )
  {
    sub_1623A60((__int64)&v213, (__int64)v15, 2);
    if ( v216 )
      sub_161E7C0((__int64)&v216, (__int64)v216);
    v216 = (unsigned __int8 *)v213;
    if ( v213 )
      sub_1623210((__int64)&v213, (unsigned __int8 *)v213, (__int64)&v216);
  }
  v17 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  v182 = (unsigned int)v17;
  v18 = (unsigned __int8 *)a1[6];
  v213 = (char *)v18;
  if ( v18 )
  {
    sub_1623A60((__int64)&v213, (__int64)v18, 2);
    v19 = v216;
    if ( !v216 )
      goto LABEL_13;
  }
  else
  {
    v19 = v216;
    if ( !v216 )
      goto LABEL_15;
  }
  sub_161E7C0((__int64)&v216, (__int64)v19);
LABEL_13:
  v19 = (unsigned __int8 *)v213;
  v216 = (unsigned __int8 *)v213;
  if ( v213 )
    sub_1623210((__int64)&v213, (unsigned __int8 *)v213, (__int64)&v216);
LABEL_15:
  v20 = sub_1599EF0((__int64 **)v180);
  v22 = *(_QWORD *)(v180 + 32);
  v23 = v20;
  if ( *(_BYTE *)(v184 + 16) == 8 )
  {
    if ( (_DWORD)v22 )
    {
      v207 = v20;
      v157 = 0;
      v204 = (unsigned int)v22;
      do
      {
        v167 = v157 - (*(_DWORD *)(v184 + 20) & 0xFFFFFFF);
        if ( !sub_1593BB0(*(_QWORD *)(v184 + 24 * v167), (__int64)v19, v167, v21) )
        {
          LODWORD(v209[0]) = v157;
          v210 = "Ptr";
          v212 = 2307;
          v211 = v209[0];
          v168 = sub_1643350(v219);
          v169 = sub_159C470(v168, v157, 0);
          v170 = v169;
          if ( *(_BYTE *)(v186 + 16) > 0x10u || *(_BYTE *)(v169 + 16) > 0x10u )
          {
            v215 = 257;
            v171 = sub_1648A60(56, 2u);
            v158 = v171;
            if ( v171 )
              sub_15FA320((__int64)v171, (_QWORD *)v186, v170, (__int64)&v213, 0);
            if ( v217 )
            {
              v172 = (unsigned __int64 *)v218;
              sub_157E9D0((__int64)(v217 + 5), (__int64)v158);
              v173 = v158[3];
              v174 = *v172;
              v158[4] = v172;
              v174 &= 0xFFFFFFFFFFFFFFF8LL;
              v158[3] = v174 | v173 & 7;
              *(_QWORD *)(v174 + 8) = v158 + 3;
              *v172 = *v172 & 7 | (unsigned __int64)(v158 + 3);
            }
            sub_164B780((__int64)v158, (__int64 *)&v210);
            if ( v216 )
            {
              v208 = v216;
              sub_1623A60((__int64)&v208, (__int64)v216, 2);
              v175 = v158[6];
              v176 = (__int64)(v158 + 6);
              if ( v175 )
              {
                sub_161E7C0((__int64)(v158 + 6), v175);
                v176 = (__int64)(v158 + 6);
              }
              v177 = v208;
              v158[6] = v208;
              if ( v177 )
                sub_1623210((__int64)&v208, v177, v176);
            }
          }
          else
          {
            v158 = (_QWORD *)sub_15A37D0((_BYTE *)v186, v169, 0);
          }
          LODWORD(v210) = v157;
          v213 = "Load";
          v215 = 2307;
          v214 = v210;
          v159 = sub_1648A60(64, 1u);
          if ( v159 )
            sub_15F9210((__int64)v159, *(_QWORD *)(*v158 + 24LL), (__int64)v158, 0, 0, 0);
          if ( v217 )
          {
            v202 = (unsigned __int64 *)v218;
            sub_157E9D0((__int64)(v217 + 5), (__int64)v159);
            v160 = *v202;
            v161 = v159[3] & 7LL;
            v159[4] = v202;
            v160 &= 0xFFFFFFFFFFFFFFF8LL;
            v159[3] = v160 | v161;
            *(_QWORD *)(v160 + 8) = v159 + 3;
            *v202 = *v202 & 7 | (unsigned __int64)(v159 + 3);
          }
          sub_164B780((__int64)v159, (__int64 *)&v213);
          if ( v216 )
          {
            v209[0] = v216;
            sub_1623A60((__int64)v209, (__int64)v216, 2);
            v162 = v159[6];
            v163 = (__int64)(v159 + 6);
            if ( v162 )
            {
              sub_161E7C0((__int64)(v159 + 6), v162);
              v163 = (__int64)(v159 + 6);
            }
            v164 = (unsigned __int8 *)v209[0];
            v159[6] = v209[0];
            if ( v164 )
              sub_1623210((__int64)v209, v164, v163);
          }
          sub_15F8F50((__int64)v159, v182);
          LODWORD(v210) = v157;
          v213 = "Res";
          v215 = 2307;
          v214 = v210;
          v165 = sub_1643350(v219);
          v166 = sub_159C470(v165, v157, 0);
          v19 = (unsigned __int8 *)v207;
          v207 = sub_156D8B0((__int64 *)&v216, v207, (__int64)v159, v166, (__int64)&v213);
        }
        ++v157;
      }
      while ( v204 != v157 );
      v23 = v207;
    }
    v152 = v23;
    v215 = 257;
    v151 = v178;
  }
  else
  {
    if ( (_DWORD)v22 )
    {
      v183 = v20;
      v181 = (unsigned int)(v22 - 1);
      for ( i = 0; ; ++i )
      {
        v210 = "Mask";
        LODWORD(v209[0]) = i;
        v211 = v209[0];
        v212 = 2307;
        v60 = sub_1643350(v219);
        v61 = sub_159C470(v60, i, 0);
        v62 = v61;
        if ( *(_BYTE *)(v184 + 16) > 0x10u || *(_BYTE *)(v61 + 16) > 0x10u )
        {
          v215 = 257;
          v112 = sub_1648A60(56, 2u);
          v63 = (__int64)v112;
          if ( v112 )
            sub_15FA320((__int64)v112, (_QWORD *)v184, v62, (__int64)&v213, 0);
          if ( v217 )
          {
            v113 = v218;
            sub_157E9D0((__int64)(v217 + 5), v63);
            v114 = *(_QWORD *)(v63 + 24);
            v115 = *v113;
            *(_QWORD *)(v63 + 32) = v113;
            v115 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v63 + 24) = v115 | v114 & 7;
            *(_QWORD *)(v115 + 8) = v63 + 24;
            *v113 = *v113 & 7 | (v63 + 24);
          }
          sub_164B780(v63, (__int64 *)&v210);
          if ( v216 )
          {
            v208 = v216;
            sub_1623A60((__int64)&v208, (__int64)v216, 2);
            v116 = *(_QWORD *)(v63 + 48);
            if ( v116 )
              sub_161E7C0(v63 + 48, v116);
            v117 = v208;
            *(_QWORD *)(v63 + 48) = v208;
            if ( v117 )
              sub_1623210((__int64)&v208, v117, v63 + 48);
          }
        }
        else
        {
          v63 = sub_15A37D0((_BYTE *)v184, v61, 0);
        }
        LODWORD(v209[0]) = i;
        v210 = "ToLoad";
        v211 = v209[0];
        v212 = 2307;
        v64 = sub_15A0680(*(_QWORD *)v63, 1, 0);
        if ( *(_BYTE *)(v63 + 16) > 0x10u || *(_BYTE *)(v64 + 16) > 0x10u )
        {
          v200 = v64;
          v215 = 257;
          v101 = sub_1648A60(56, 2u);
          v102 = v200;
          v65 = v101;
          if ( v101 )
          {
            v201 = (__int64)v101;
            v103 = *(_QWORD ***)v63;
            if ( *(_BYTE *)(*(_QWORD *)v63 + 8LL) == 16 )
            {
              v179 = v102;
              v196 = v103[4];
              v104 = (__int64 *)sub_1643320(*v103);
              v105 = (__int64)sub_16463B0(v104, (unsigned int)v196);
              v106 = v179;
            }
            else
            {
              v198 = v102;
              v105 = sub_1643320(*v103);
              v106 = v198;
            }
            sub_15FEC10((__int64)v65, v105, 51, 32, v63, v106, (__int64)&v213, 0);
          }
          else
          {
            v201 = 0;
          }
          if ( v217 )
          {
            v197 = (unsigned __int64 *)v218;
            sub_157E9D0((__int64)(v217 + 5), (__int64)v65);
            v107 = *v197;
            v108 = v65[3] & 7LL;
            v65[4] = v197;
            v107 &= 0xFFFFFFFFFFFFFFF8LL;
            v65[3] = v107 | v108;
            *(_QWORD *)(v107 + 8) = v65 + 3;
            *v197 = *v197 & 7 | (unsigned __int64)(v65 + 3);
          }
          sub_164B780(v201, (__int64 *)&v210);
          if ( v216 )
          {
            v208 = v216;
            sub_1623A60((__int64)&v208, (__int64)v216, 2);
            v109 = v65[6];
            v110 = (__int64)(v65 + 6);
            if ( v109 )
            {
              sub_161E7C0((__int64)(v65 + 6), v109);
              v110 = (__int64)(v65 + 6);
            }
            v111 = v208;
            v65[6] = v208;
            if ( v111 )
              sub_1623210((__int64)&v208, v111, v110);
          }
        }
        else
        {
          v65 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v63, (_QWORD *)v64, 0);
        }
        v213 = "cond.load";
        v215 = 259;
        v66 = sub_157FBF0(v203, v185, (__int64)&v213);
        v218 = v185;
        v199 = (_QWORD *)v66;
        v67 = (unsigned __int8 *)a1[6];
        v68 = (_QWORD *)a1[5];
        v213 = (char *)v67;
        v217 = v68;
        if ( v67 )
        {
          sub_1623A60((__int64)&v213, (__int64)v67, 2);
          v69 = v216;
          if ( !v216 )
            goto LABEL_70;
        }
        else
        {
          v69 = v216;
          if ( !v216 )
            goto LABEL_72;
        }
        sub_161E7C0((__int64)&v216, (__int64)v69);
LABEL_70:
        v216 = (unsigned __int8 *)v213;
        if ( v213 )
          sub_1623210((__int64)&v213, (unsigned __int8 *)v213, (__int64)&v216);
LABEL_72:
        LODWORD(v209[0]) = i;
        v210 = "Ptr";
        v211 = v209[0];
        v212 = 2307;
        v70 = sub_1643350(v219);
        v71 = sub_159C470(v70, i, 0);
        if ( *(_BYTE *)(v186 + 16) > 0x10u || *(_BYTE *)(v71 + 16) > 0x10u )
        {
          v215 = 257;
          v93 = sub_1648A60(56, 2u);
          v94 = (__int64)v93;
          if ( v93 )
          {
            v192 = v93;
            sub_15FA320((__int64)v93, (_QWORD *)v186, v71, (__int64)&v213, 0);
            v94 = (__int64)v192;
          }
          if ( v217 )
          {
            v95 = v218;
            v193 = v94;
            sub_157E9D0((__int64)(v217 + 5), v94);
            v94 = v193;
            v96 = *v95;
            v97 = *(_QWORD *)(v193 + 24);
            *(_QWORD *)(v193 + 32) = v95;
            v96 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v193 + 24) = v96 | v97 & 7;
            *(_QWORD *)(v96 + 8) = v193 + 24;
            *v95 = *v95 & 7 | (v193 + 24);
          }
          v194 = v94;
          sub_164B780(v94, (__int64 *)&v210);
          v72 = v194;
          if ( v216 )
          {
            v208 = v216;
            sub_1623A60((__int64)&v208, (__int64)v216, 2);
            v72 = v194;
            v98 = *(_QWORD *)(v194 + 48);
            v99 = v194 + 48;
            if ( v98 )
            {
              sub_161E7C0(v194 + 48, v98);
              v72 = v194;
              v99 = v194 + 48;
            }
            v100 = v208;
            *(_QWORD *)(v72 + 48) = v208;
            if ( v100 )
            {
              v195 = v72;
              sub_1623210((__int64)&v208, v100, v99);
              v72 = v195;
            }
          }
        }
        else
        {
          v72 = sub_15A37D0((_BYTE *)v186, v71, 0);
        }
        LODWORD(v210) = i;
        v213 = "Load";
        v188 = v72;
        v214 = v210;
        v215 = 2307;
        v73 = sub_1648A60(64, 1u);
        v74 = (__int64)v73;
        if ( v73 )
          sub_15F9210((__int64)v73, *(_QWORD *)(*(_QWORD *)v188 + 24LL), v188, 0, 0, 0);
        if ( v217 )
        {
          v189 = v218;
          sub_157E9D0((__int64)(v217 + 5), v74);
          v75 = *v189;
          v76 = *(_QWORD *)(v74 + 24) & 7LL;
          *(_QWORD *)(v74 + 32) = v189;
          v75 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v74 + 24) = v75 | v76;
          *(_QWORD *)(v75 + 8) = v74 + 24;
          *v189 = *v189 & 7 | (v74 + 24);
        }
        sub_164B780(v74, (__int64 *)&v213);
        if ( v216 )
        {
          v209[0] = v216;
          sub_1623A60((__int64)v209, (__int64)v216, 2);
          v77 = *(_QWORD *)(v74 + 48);
          v78 = v74 + 48;
          if ( v77 )
          {
            sub_161E7C0(v74 + 48, v77);
            v78 = v74 + 48;
          }
          v79 = (unsigned __int8 *)v209[0];
          *(_QWORD *)(v74 + 48) = v209[0];
          if ( v79 )
            sub_1623210((__int64)v209, v79, v78);
        }
        sub_15F8F50(v74, v182);
        LODWORD(v209[0]) = i;
        v210 = "Res";
        v212 = 2307;
        v211 = v209[0];
        v80 = sub_1643350(v219);
        v81 = sub_159C470(v80, i, 0);
        if ( *(_BYTE *)(v23 + 16) > 0x10u || *(_BYTE *)(v74 + 16) > 0x10u || *(_BYTE *)(v81 + 16) > 0x10u )
        {
          v191 = v81;
          v215 = 257;
          v86 = sub_1648A60(56, 3u);
          v82 = v86;
          if ( v86 )
            sub_15FA480((__int64)v86, (__int64 *)v23, v74, v191, (__int64)&v213, 0);
          if ( v217 )
          {
            v87 = (unsigned __int64 *)v218;
            sub_157E9D0((__int64)(v217 + 5), (__int64)v82);
            v88 = v82[3];
            v89 = *v87;
            v82[4] = v87;
            v89 &= 0xFFFFFFFFFFFFFFF8LL;
            v82[3] = v89 | v88 & 7;
            *(_QWORD *)(v89 + 8) = v82 + 3;
            *v87 = *v87 & 7 | (unsigned __int64)(v82 + 3);
          }
          sub_164B780((__int64)v82, (__int64 *)&v210);
          if ( v216 )
          {
            v208 = v216;
            sub_1623A60((__int64)&v208, (__int64)v216, 2);
            v90 = v82[6];
            v91 = (__int64)(v82 + 6);
            if ( v90 )
            {
              sub_161E7C0((__int64)(v82 + 6), v90);
              v91 = (__int64)(v82 + 6);
            }
            v92 = v208;
            v82[6] = v208;
            if ( v92 )
              sub_1623210((__int64)&v208, v92, v91);
          }
        }
        else
        {
          v82 = (_QWORD *)sub_15A3890((__int64 *)v23, v74, v81, 0);
        }
        v213 = "else";
        v215 = 259;
        v83 = sub_157FBF0(v199, v185, (__int64)&v213);
        v218 = v185;
        v190 = v83;
        v84 = (unsigned __int8 *)a1[6];
        v85 = (_QWORD *)a1[5];
        v213 = (char *)v84;
        v217 = v85;
        if ( v84 )
        {
          sub_1623A60((__int64)&v213, (__int64)v84, 2);
          v24 = v216;
          if ( !v216 )
            goto LABEL_20;
        }
        else
        {
          v24 = v216;
          if ( !v216 )
            goto LABEL_22;
        }
        sub_161E7C0((__int64)&v216, (__int64)v24);
LABEL_20:
        v216 = (unsigned __int8 *)v213;
        if ( v213 )
          sub_1623210((__int64)&v213, (unsigned __int8 *)v213, (__int64)&v216);
LABEL_22:
        v25 = (_QWORD *)sub_157EBA0((__int64)v203);
        v26 = sub_1648A60(56, 3u);
        if ( v26 )
          sub_15F83E0((__int64)v26, (__int64)v199, v190, (__int64)v65, (__int64)v25);
        sub_15F20C0(v25);
        if ( v181 == i )
          goto LABEL_137;
        v210 = "res.phi.else";
        v212 = 259;
        v215 = 257;
        v27 = sub_1648B60(64);
        v23 = v27;
        if ( v27 )
        {
          v28 = v27;
          sub_15F1EA0(v27, v180, 53, 0, 0, 0);
          *(_DWORD *)(v23 + 56) = 2;
          sub_164B780(v23, (__int64 *)&v213);
          sub_1648880(v23, *(_DWORD *)(v23 + 56), 1);
        }
        else
        {
          v28 = 0;
        }
        if ( v217 )
        {
          v29 = v218;
          sub_157E9D0((__int64)(v217 + 5), v23);
          v30 = *(_QWORD *)(v23 + 24);
          v31 = *v29;
          *(_QWORD *)(v23 + 32) = v29;
          v31 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v23 + 24) = v31 | v30 & 7;
          *(_QWORD *)(v31 + 8) = v23 + 24;
          *v29 = *v29 & 7 | (v23 + 24);
        }
        sub_164B780(v28, (__int64 *)&v210);
        v36 = (__int64)v216;
        if ( v216 )
        {
          v209[0] = v216;
          sub_1623A60((__int64)v209, (__int64)v216, 2);
          v37 = *(_QWORD *)(v23 + 48);
          v34 = v209;
          if ( v37 )
          {
            sub_161E7C0(v23 + 48, v37);
            v34 = v209;
          }
          v36 = v209[0];
          *(_QWORD *)(v23 + 48) = v209[0];
          if ( v36 )
            sub_1623210((__int64)v209, (unsigned __int8 *)v36, v23 + 48);
        }
        v38 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        if ( v38 == *(_DWORD *)(v23 + 56) )
        {
          sub_15F55D0(v23, v36, v32, v33, (__int64)v34, v35);
          v38 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        }
        v39 = (v38 + 1) & 0xFFFFFFF;
        v40 = v39 | *(_DWORD *)(v23 + 20) & 0xF0000000;
        *(_DWORD *)(v23 + 20) = v40;
        if ( (v40 & 0x40000000) != 0 )
          v41 = *(_QWORD *)(v23 - 8);
        else
          v41 = v28 - 24 * v39;
        v42 = (_QWORD *)(v41 + 24LL * (unsigned int)(v39 - 1));
        if ( *v42 )
        {
          v43 = v42[1];
          v44 = v42[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v44 = v43;
          if ( v43 )
          {
            v36 = *(_QWORD *)(v43 + 16) & 3LL;
            *(_QWORD *)(v43 + 16) = v36 | v44;
          }
        }
        *v42 = v82;
        if ( v82 )
        {
          v45 = v82[1];
          v36 = (__int64)(v82 + 1);
          v42[1] = v45;
          if ( v45 )
            *(_QWORD *)(v45 + 16) = (unsigned __int64)(v42 + 1) | *(_QWORD *)(v45 + 16) & 3LL;
          v42[2] = v36 | v42[2] & 3LL;
          v82[1] = v42;
        }
        v46 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        v47 = (unsigned int)(v46 - 1);
        if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
          v48 = *(_QWORD *)(v23 - 8);
        else
          v48 = v28 - 24 * v46;
        v49 = 3LL * *(unsigned int *)(v23 + 56);
        *(_QWORD *)(v48 + 8 * v47 + 24LL * *(unsigned int *)(v23 + 56) + 8) = v199;
        v50 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        if ( v50 == *(_DWORD *)(v23 + 56) )
        {
          sub_15F55D0(v23, v36, v49, v48, (__int64)v34, v35);
          v50 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        }
        v51 = (v50 + 1) & 0xFFFFFFF;
        v52 = v51 | *(_DWORD *)(v23 + 20) & 0xF0000000;
        *(_DWORD *)(v23 + 20) = v52;
        if ( (v52 & 0x40000000) != 0 )
          v53 = *(_QWORD *)(v23 - 8);
        else
          v53 = v28 - 24 * v51;
        v54 = (__int64 *)(v53 + 24LL * (unsigned int)(v51 - 1));
        if ( *v54 )
        {
          v55 = v54[1];
          v56 = v54[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v56 = v55;
          if ( v55 )
            *(_QWORD *)(v55 + 16) = *(_QWORD *)(v55 + 16) & 3LL | v56;
        }
        *v54 = v183;
        if ( v183 )
        {
          v57 = *(_QWORD *)(v183 + 8);
          v54[1] = v57;
          if ( v57 )
            *(_QWORD *)(v57 + 16) = (unsigned __int64)(v54 + 1) | *(_QWORD *)(v57 + 16) & 3LL;
          v54[2] = (v183 + 8) | v54[2] & 3;
          *(_QWORD *)(v183 + 8) = v54;
        }
        v58 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
          v59 = *(_QWORD *)(v23 - 8);
        else
          v59 = v28 - 24 * v58;
        v183 = v23;
        *(_QWORD *)(v59 + 8LL * (unsigned int)(v58 - 1) + 24LL * *(unsigned int *)(v23 + 56) + 8) = v203;
        v203 = (_QWORD *)v190;
      }
    }
    v82 = (_QWORD *)v20;
    v199 = 0;
LABEL_137:
    v210 = "res.phi.select";
    v212 = 259;
    v215 = 257;
    v118 = sub_1648B60(64);
    v119 = v118;
    if ( v118 )
    {
      v120 = v118;
      sub_15F1EA0(v118, v180, 53, 0, 0, 0);
      *(_DWORD *)(v119 + 56) = 2;
      sub_164B780(v119, (__int64 *)&v213);
      sub_1648880(v119, *(_DWORD *)(v119 + 56), 1);
    }
    else
    {
      v120 = 0;
    }
    if ( v217 )
    {
      v206 = v218;
      sub_157E9D0((__int64)(v217 + 5), v119);
      v121 = *v206;
      v122 = *(_QWORD *)(v119 + 24) & 7LL;
      *(_QWORD *)(v119 + 32) = v206;
      v121 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v119 + 24) = v121 | v122;
      *(_QWORD *)(v121 + 8) = v119 + 24;
      *v206 = *v206 & 7 | (v119 + 24);
    }
    sub_164B780(v120, (__int64 *)&v210);
    v127 = (__int64)v216;
    if ( v216 )
    {
      v209[0] = v216;
      sub_1623A60((__int64)v209, (__int64)v216, 2);
      v128 = *(_QWORD *)(v119 + 48);
      v125 = v209;
      v123 = v119 + 48;
      if ( v128 )
      {
        sub_161E7C0(v119 + 48, v128);
        v125 = v209;
        v123 = v119 + 48;
      }
      v127 = v209[0];
      *(_QWORD *)(v119 + 48) = v209[0];
      if ( v127 )
        sub_1623210((__int64)v209, (unsigned __int8 *)v127, v123);
    }
    v129 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    if ( v129 == *(_DWORD *)(v119 + 56) )
    {
      sub_15F55D0(v119, v127, v123, v124, (__int64)v125, v126);
      v129 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    }
    v130 = (v129 + 1) & 0xFFFFFFF;
    v131 = v130 | *(_DWORD *)(v119 + 20) & 0xF0000000;
    *(_DWORD *)(v119 + 20) = v131;
    if ( (v131 & 0x40000000) != 0 )
      v132 = *(_QWORD *)(v119 - 8);
    else
      v132 = v120 - 24 * v130;
    v133 = (_QWORD *)(v132 + 24LL * (unsigned int)(v130 - 1));
    if ( *v133 )
    {
      v134 = v133[1];
      v135 = v133[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v135 = v134;
      if ( v134 )
      {
        v127 = *(_QWORD *)(v134 + 16) & 3LL;
        *(_QWORD *)(v134 + 16) = v127 | v135;
      }
    }
    *v133 = v82;
    if ( v82 )
    {
      v136 = v82[1];
      v127 = (__int64)(v82 + 1);
      v133[1] = v136;
      if ( v136 )
        *(_QWORD *)(v136 + 16) = (unsigned __int64)(v133 + 1) | *(_QWORD *)(v136 + 16) & 3LL;
      v133[2] = v127 | v133[2] & 3LL;
      v82[1] = v133;
    }
    v137 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    v138 = (unsigned int)(v137 - 1);
    if ( (*(_BYTE *)(v119 + 23) & 0x40) != 0 )
      v139 = *(_QWORD *)(v119 - 8);
    else
      v139 = v120 - 24 * v137;
    v140 = 3LL * *(unsigned int *)(v119 + 56);
    *(_QWORD *)(v139 + 8 * v138 + 24LL * *(unsigned int *)(v119 + 56) + 8) = v199;
    v141 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    if ( v141 == *(_DWORD *)(v119 + 56) )
    {
      sub_15F55D0(v119, v127, v140, v139, (__int64)v125, v126);
      v141 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    }
    v142 = (v141 + 1) & 0xFFFFFFF;
    v143 = v142 | *(_DWORD *)(v119 + 20) & 0xF0000000;
    *(_DWORD *)(v119 + 20) = v143;
    if ( (v143 & 0x40000000) != 0 )
      v144 = *(_QWORD *)(v119 - 8);
    else
      v144 = v120 - 24 * v142;
    v145 = (__int64 *)(v144 + 24LL * (unsigned int)(v142 - 1));
    if ( *v145 )
    {
      v146 = v145[1];
      v147 = v145[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v147 = v146;
      if ( v146 )
        *(_QWORD *)(v146 + 16) = *(_QWORD *)(v146 + 16) & 3LL | v147;
    }
    *v145 = v23;
    if ( v23 )
    {
      v148 = *(_QWORD *)(v23 + 8);
      v145[1] = v148;
      if ( v148 )
        *(_QWORD *)(v148 + 16) = (unsigned __int64)(v145 + 1) | *(_QWORD *)(v148 + 16) & 3LL;
      v145[2] = (v23 + 8) | v145[2] & 3;
      *(_QWORD *)(v23 + 8) = v145;
    }
    v149 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v119 + 23) & 0x40) != 0 )
      v150 = *(_QWORD *)(v119 - 8);
    else
      v150 = v120 - 24 * v149;
    v151 = v178;
    *(_QWORD *)(v150 + 8LL * (unsigned int)(v149 - 1) + 24LL * *(unsigned int *)(v119 + 56) + 8) = v203;
    v215 = 257;
    v152 = v119;
  }
  v153 = sub_156B790((__int64 *)&v216, v184, v152, v151, (__int64)&v213, 0);
  sub_164D160((__int64)a1, v153, a2, a3, a4, a5, v154, v155, a8, a9);
  result = sub_15F20C0(a1);
  if ( v216 )
    return sub_161E7C0((__int64)&v216, (__int64)v216);
  return result;
}
