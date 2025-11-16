// Function: sub_150F8E0
// Address: 0x150f8e0
//
__int64 __fastcall sub_150F8E0(__int64 a1, int a2)
{
  __int64 v2; // r13
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  __int64 v5; // rdx
  int *v6; // rsi
  unsigned __int8 v7; // al
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  char v11; // dl
  unsigned int v12; // r8d
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // r14
  __int64 v15; // r11
  unsigned int v16; // r10d
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rsi
  unsigned int v19; // ebx
  unsigned int v20; // edi
  unsigned __int64 v21; // r11
  int v22; // ebx
  char v23; // r10
  __int64 v24; // r12
  unsigned __int64 *v25; // r8
  unsigned __int64 v26; // rsi
  unsigned int v27; // r11d
  unsigned __int64 v28; // rax
  char v29; // cl
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  char v33; // cl
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r8
  unsigned int v37; // eax
  unsigned __int64 *v38; // r9
  unsigned __int64 v39; // rdi
  unsigned int v40; // r14d
  unsigned int v42; // r11d
  __int64 v43; // rax
  __int64 v44; // rdx
  char v45; // cl
  unsigned int v46; // edi
  __int64 v47; // r12
  unsigned __int64 v48; // r11
  unsigned __int64 v49; // r8
  unsigned int v50; // ebx
  unsigned __int64 *v51; // r10
  unsigned __int64 v52; // rsi
  unsigned int v53; // r8d
  unsigned __int64 v54; // r12
  char v55; // bl
  unsigned int v56; // r11d
  __int64 v57; // r12
  unsigned __int64 v58; // r10
  unsigned __int64 v59; // rax
  unsigned int v60; // r13d
  unsigned __int64 *v61; // r9
  unsigned __int64 v62; // rsi
  int v63; // eax
  unsigned __int64 v64; // r12
  unsigned int v65; // edi
  unsigned __int64 v66; // rax
  unsigned int v67; // eax
  __int64 v68; // r8
  __int64 m; // rax
  __int64 v70; // rdx
  char v71; // cl
  char v72; // al
  __int64 v73; // rcx
  __int64 v74; // rax
  unsigned __int64 v75; // rcx
  unsigned __int64 v76; // rdx
  unsigned int v77; // eax
  unsigned __int64 v78; // r9
  unsigned __int64 *v79; // r11
  unsigned __int64 v80; // r8
  unsigned int v81; // r9d
  __int64 v82; // r9
  unsigned int v83; // edi
  unsigned __int64 v84; // r11
  unsigned __int64 v85; // r12
  unsigned int v86; // ebx
  unsigned __int64 v87; // r8
  _QWORD *v88; // r10
  unsigned int v89; // eax
  unsigned __int64 v90; // rax
  unsigned int v91; // ebx
  __int64 v92; // rax
  __int64 v93; // rdx
  char v94; // cl
  unsigned __int64 v95; // rax
  unsigned int v96; // r9d
  __int64 v97; // r8
  unsigned int v98; // ebx
  __int64 v99; // r14
  unsigned __int64 v100; // r12
  unsigned __int64 v101; // rdi
  unsigned int v102; // r13d
  unsigned __int64 v103; // r10
  _QWORD *v104; // r11
  unsigned int v105; // edi
  unsigned __int64 v106; // rsi
  unsigned __int64 v107; // r14
  int v108; // r14d
  __int64 v109; // r13
  unsigned __int64 v110; // r11
  unsigned __int64 v111; // rdi
  unsigned int v112; // r12d
  unsigned __int64 v113; // r9
  _QWORD *v114; // r10
  unsigned int v115; // edi
  unsigned __int64 v116; // rsi
  unsigned int v117; // r8d
  unsigned __int64 v118; // rax
  unsigned int v119; // edi
  __int64 v120; // r9
  __int64 v121; // rax
  __int64 v122; // rsi
  __int64 v123; // rdx
  char v124; // cl
  unsigned __int64 v125; // rax
  unsigned int v126; // edi
  __int64 v127; // r10
  __int64 v128; // rax
  __int64 v129; // rsi
  __int64 v130; // rdx
  char v131; // cl
  __int64 v132; // rdx
  __int64 v133; // rax
  __int64 v134; // rcx
  unsigned __int64 v135; // rcx
  unsigned __int64 v136; // rdx
  unsigned int v137; // eax
  unsigned __int64 v138; // r9
  unsigned __int64 *v139; // r11
  unsigned __int64 v140; // r8
  unsigned int v141; // r9d
  unsigned int v142; // r8d
  __int64 v143; // rax
  __int64 v144; // rdx
  char v145; // cl
  unsigned __int64 v146; // r9
  unsigned int v147; // r14d
  __int64 v148; // rdx
  __int64 v149; // rsi
  char v150; // cl
  unsigned __int64 v151; // r8
  unsigned int v152; // r8d
  unsigned __int64 v153; // r10
  unsigned __int64 v154; // rdi
  unsigned int v155; // ebx
  unsigned __int64 *v156; // r11
  unsigned int v157; // edi
  __int64 v158; // rax
  unsigned __int64 v159; // rsi
  __int64 v160; // rdx
  char v161; // cl
  unsigned __int64 v162; // r9
  unsigned int v163; // edi
  unsigned int v164; // r8d
  __int64 v165; // r9
  unsigned __int64 v166; // rdi
  unsigned __int64 v167; // rax
  unsigned int v168; // r12d
  unsigned __int64 *v169; // r10
  unsigned __int64 v170; // rsi
  unsigned int v171; // ebx
  unsigned int v172; // edi
  unsigned __int64 v173; // rax
  char v174; // bl
  unsigned int v175; // r11d
  __int64 v176; // r12
  unsigned __int64 v177; // r10
  unsigned __int64 v178; // rax
  unsigned int v179; // r14d
  unsigned __int64 *v180; // r9
  unsigned __int64 v181; // rsi
  int v182; // eax
  int v183; // edx
  unsigned __int64 v184; // rax
  char v185; // cl
  unsigned __int64 v186; // r12
  unsigned __int64 v187; // rax
  unsigned int v188; // r12d
  __int64 v189; // rax
  __int64 v190; // rsi
  __int64 v191; // rdx
  char v192; // cl
  char v193; // al
  unsigned int v194; // eax
  __int64 v195; // r8
  __int64 i; // rax
  __int64 v197; // rdx
  char v198; // cl
  __int64 v199; // r9
  unsigned __int64 v200; // r8
  unsigned __int64 v201; // rax
  unsigned int v202; // r12d
  unsigned __int64 *v203; // r10
  unsigned __int64 v204; // rsi
  unsigned int v205; // ebx
  unsigned __int64 v206; // rax
  char v207; // bl
  int v208; // r11d
  __int64 v209; // r12
  unsigned __int64 v210; // r10
  unsigned __int64 v211; // rax
  unsigned int v212; // r14d
  unsigned __int64 *v213; // r9
  unsigned __int64 v214; // rsi
  int v215; // eax
  unsigned __int64 v216; // r12
  unsigned int v217; // edi
  unsigned __int64 v218; // rax
  unsigned int v219; // eax
  __int64 v220; // r8
  __int64 j; // rax
  __int64 v222; // rdx
  char v223; // cl
  unsigned int v224; // edi
  __int64 v225; // r11
  unsigned __int64 v226; // r10
  unsigned __int64 v227; // r15
  unsigned int v228; // r14d
  unsigned __int64 *v229; // r9
  unsigned __int64 v230; // rsi
  unsigned int v231; // r15d
  unsigned __int64 v232; // r11
  unsigned int v233; // edi
  __int64 v234; // r14
  unsigned __int64 v235; // r11
  unsigned __int64 v236; // r8
  unsigned int v237; // r15d
  unsigned __int64 *v238; // r10
  unsigned __int64 v239; // rsi
  unsigned int v240; // r8d
  unsigned __int64 v241; // r14
  unsigned __int64 v242; // rax
  unsigned int v243; // r8d
  __int64 v244; // rax
  __int64 v245; // rdx
  char v246; // cl
  unsigned __int64 v247; // r9
  unsigned __int64 v248; // rax
  unsigned int v249; // r15d
  __int64 v250; // rax
  __int64 v251; // rdx
  char v252; // cl
  unsigned __int64 v253; // r8
  unsigned int v254; // r9d
  __int64 v255; // rdi
  __int64 v256; // rsi
  char v257; // cl
  unsigned __int64 v258; // rdx
  unsigned int v259; // r9d
  __int64 v260; // rsi
  __int64 v261; // rdi
  char v262; // cl
  unsigned __int64 v263; // rdx
  unsigned int v264; // eax
  __int64 v265; // r11
  unsigned __int64 v266; // rdi
  __int64 v267; // rax
  __int64 v268; // rdx
  char v269; // cl
  unsigned int v270; // eax
  __int64 v271; // r11
  unsigned __int64 v272; // r8
  __int64 v273; // rax
  __int64 v274; // rdx
  char v275; // cl
  unsigned __int64 v276; // rdx
  unsigned int v277; // [rsp+4h] [rbp-5Ch]
  __int64 v278; // [rsp+8h] [rbp-58h]
  unsigned __int64 v279; // [rsp+10h] [rbp-50h]
  unsigned int v280; // [rsp+1Ch] [rbp-44h]
  __int64 v281; // [rsp+20h] [rbp-40h]
  unsigned int v282; // [rsp+28h] [rbp-38h]
  unsigned int v283; // [rsp+28h] [rbp-38h]
  unsigned int v284; // [rsp+28h] [rbp-38h]
  int v285; // [rsp+28h] [rbp-38h]
  unsigned int v286; // [rsp+2Ch] [rbp-34h]
  unsigned int v287; // [rsp+2Ch] [rbp-34h]
  unsigned int v288; // [rsp+2Ch] [rbp-34h]
  int k; // [rsp+2Ch] [rbp-34h]

  v2 = a1;
  if ( a2 == 3 )
  {
    v164 = *(_DWORD *)(a1 + 32);
    if ( v164 > 5 )
    {
      v193 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 24) >>= 6;
      v172 = v164 - 6;
      LODWORD(v173) = v193 & 0x3F;
      *(_DWORD *)(v2 + 32) = v164 - 6;
    }
    else
    {
      v165 = 0;
      if ( v164 )
        v165 = *(_QWORD *)(a1 + 24);
      v166 = *(_QWORD *)(a1 + 16);
      v167 = *(_QWORD *)(v2 + 8);
      v168 = 6 - v164;
      if ( v166 >= v167 )
        goto LABEL_44;
      v169 = (unsigned __int64 *)(v166 + *(_QWORD *)v2);
      if ( v167 < v166 + 8 )
      {
        v264 = v167 - v166;
        *(_QWORD *)(v2 + 24) = 0;
        v265 = v264;
        v171 = 8 * v264;
        v266 = v264 + v166;
        if ( !v264 )
        {
          *(_QWORD *)(v2 + 16) = v266;
          *(_DWORD *)(v2 + 32) = 0;
          goto LABEL_44;
        }
        v267 = 0;
        v170 = 0;
        do
        {
          v268 = *((unsigned __int8 *)v169 + v267);
          v269 = 8 * v267++;
          v170 |= v268 << v269;
          *(_QWORD *)(v2 + 24) = v170;
        }
        while ( v265 != v267 );
        *(_QWORD *)(v2 + 16) = v266;
        *(_DWORD *)(v2 + 32) = v171;
        if ( v168 > v171 )
          goto LABEL_44;
      }
      else
      {
        v170 = *v169;
        *(_QWORD *)(v2 + 16) = v166 + 8;
        v171 = 64;
      }
      v172 = v164 + v171 - 6;
      *(_DWORD *)(v2 + 32) = v172;
      *(_QWORD *)(v2 + 24) = v170 >> v168;
      v173 = v165 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v164 + 58)) & v170) << v164);
    }
    v277 = v173;
    if ( (v173 & 0x20) != 0 )
    {
      v174 = 0;
      v175 = v173 & 0x1F;
      do
      {
        v174 += 5;
        if ( v172 <= 5 )
        {
          v176 = 0;
          if ( v172 )
            v176 = *(_QWORD *)(v2 + 24);
          v177 = *(_QWORD *)(v2 + 16);
          v178 = *(_QWORD *)(v2 + 8);
          v179 = 6 - v172;
          if ( v177 >= v178 )
            goto LABEL_44;
          v180 = (unsigned __int64 *)(v177 + *(_QWORD *)v2);
          if ( v178 < v177 + 8 )
          {
            v194 = v178 - v177;
            *(_QWORD *)(v2 + 24) = 0;
            v287 = v194;
            if ( !v194 )
              goto LABEL_129;
            v195 = v194;
            v181 = 0;
            for ( i = 0; i != v195; ++i )
            {
              v197 = *((unsigned __int8 *)v180 + i);
              v198 = 8 * i;
              v181 |= v197 << v198;
              *(_QWORD *)(v2 + 24) = v181;
            }
            *(_QWORD *)(v2 + 16) = v177 + v195;
            v182 = 8 * v287;
            *(_DWORD *)(v2 + 32) = 8 * v287;
            if ( v179 > 8 * v287 )
              goto LABEL_44;
          }
          else
          {
            v181 = *v180;
            *(_QWORD *)(v2 + 16) = v177 + 8;
            v182 = 64;
          }
          *(_QWORD *)(v2 + 24) = v181 >> v179;
          v183 = v182 + v172 - 6;
          v184 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v172 + 58);
          *(_DWORD *)(v2 + 32) = v183;
          v185 = v172;
          v172 = v183;
          v186 = ((v184 & v181) << v185) | v176;
        }
        else
        {
          v187 = *(_QWORD *)(v2 + 24);
          v172 -= 6;
          *(_DWORD *)(v2 + 32) = v172;
          *(_QWORD *)(v2 + 24) = v187 >> 6;
          LOBYTE(v186) = v187 & 0x3F;
        }
        v175 |= (v186 & 0x1F) << v174;
      }
      while ( (v186 & 0x20) != 0 );
      v277 = v175;
    }
    if ( v172 > 5 )
    {
      v276 = *(_QWORD *)(v2 + 24);
      *(_DWORD *)(v2 + 32) = v172 - 6;
      *(_QWORD *)(v2 + 24) = v276 >> 6;
      LODWORD(v206) = v276 & 0x3F;
    }
    else
    {
      v199 = 0;
      if ( v172 )
        v199 = *(_QWORD *)(v2 + 24);
      v200 = *(_QWORD *)(v2 + 16);
      v201 = *(_QWORD *)(v2 + 8);
      v202 = 6 - v172;
      if ( v200 >= v201 )
        goto LABEL_44;
      v203 = (unsigned __int64 *)(v200 + *(_QWORD *)v2);
      if ( v201 < v200 + 8 )
      {
        v270 = v201 - v200;
        *(_QWORD *)(v2 + 24) = 0;
        v271 = v270;
        v205 = 8 * v270;
        v272 = v270 + v200;
        if ( !v270 )
        {
          *(_QWORD *)(v2 + 16) = v272;
          goto LABEL_129;
        }
        v273 = 0;
        v204 = 0;
        do
        {
          v274 = *((unsigned __int8 *)v203 + v273);
          v275 = 8 * v273++;
          v204 |= v274 << v275;
          *(_QWORD *)(v2 + 24) = v204;
        }
        while ( v271 != v273 );
        *(_QWORD *)(v2 + 16) = v272;
        *(_DWORD *)(v2 + 32) = v205;
        if ( v202 > v205 )
          goto LABEL_44;
      }
      else
      {
        v204 = *v203;
        *(_QWORD *)(v2 + 16) = v200 + 8;
        v205 = 64;
      }
      *(_QWORD *)(v2 + 24) = v204 >> v202;
      *(_DWORD *)(v2 + 32) = v205 + v172 - 6;
      v206 = v199 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v172 + 58)) & v204) << v172);
    }
    v285 = v206;
    if ( (v206 & 0x20) != 0 )
    {
      v207 = 0;
      v208 = v206 & 0x1F;
      do
      {
        v217 = *(_DWORD *)(v2 + 32);
        v207 += 5;
        if ( v217 <= 5 )
        {
          v209 = 0;
          if ( v217 )
            v209 = *(_QWORD *)(v2 + 24);
          v210 = *(_QWORD *)(v2 + 16);
          v211 = *(_QWORD *)(v2 + 8);
          v212 = 6 - v217;
          if ( v210 >= v211 )
            goto LABEL_44;
          v213 = (unsigned __int64 *)(v210 + *(_QWORD *)v2);
          if ( v211 < v210 + 8 )
          {
            v219 = v211 - v210;
            *(_QWORD *)(v2 + 24) = 0;
            v288 = v219;
            if ( !v219 )
              goto LABEL_129;
            v220 = v219;
            v214 = 0;
            for ( j = 0; j != v220; ++j )
            {
              v222 = *((unsigned __int8 *)v213 + j);
              v223 = 8 * j;
              v214 |= v222 << v223;
              *(_QWORD *)(v2 + 24) = v214;
            }
            *(_QWORD *)(v2 + 16) = v210 + v220;
            v215 = 8 * v288;
            *(_DWORD *)(v2 + 32) = 8 * v288;
            if ( v212 > 8 * v288 )
              goto LABEL_44;
          }
          else
          {
            v214 = *v213;
            *(_QWORD *)(v2 + 16) = v210 + 8;
            v215 = 64;
          }
          *(_DWORD *)(v2 + 32) = v217 + v215 - 6;
          *(_QWORD *)(v2 + 24) = v214 >> v212;
          v216 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v217 + 58)) & v214) << v217) | v209;
        }
        else
        {
          v218 = *(_QWORD *)(v2 + 24);
          *(_DWORD *)(v2 + 32) = v217 - 6;
          *(_QWORD *)(v2 + 24) = v218 >> 6;
          LOBYTE(v216) = v218 & 0x3F;
        }
        v208 |= (v216 & 0x1F) << v207;
      }
      while ( (v216 & 0x20) != 0 );
      v285 = v208;
    }
    if ( v285 )
    {
      for ( k = 0; k != v285; ++k )
      {
        v224 = *(_DWORD *)(v2 + 32);
        if ( v224 > 5 )
        {
          v248 = *(_QWORD *)(v2 + 24);
          *(_DWORD *)(v2 + 32) = v224 - 6;
          *(_QWORD *)(v2 + 24) = v248 >> 6;
          LOBYTE(v232) = v248 & 0x3F;
        }
        else
        {
          v225 = 0;
          if ( v224 )
            v225 = *(_QWORD *)(v2 + 24);
          v226 = *(_QWORD *)(v2 + 16);
          v227 = *(_QWORD *)(v2 + 8);
          v228 = 6 - v224;
          if ( v226 >= v227 )
            goto LABEL_44;
          v229 = (unsigned __int64 *)(v226 + *(_QWORD *)v2);
          if ( v227 < v226 + 8 )
          {
            *(_QWORD *)(v2 + 24) = 0;
            v249 = v227 - v226;
            if ( !v249 )
              goto LABEL_129;
            v250 = 0;
            v230 = 0;
            do
            {
              v251 = *((unsigned __int8 *)v229 + v250);
              v252 = 8 * v250++;
              v230 |= v251 << v252;
              *(_QWORD *)(v2 + 24) = v230;
            }
            while ( v249 != v250 );
            v253 = v226 + v249;
            v231 = 8 * v249;
            *(_QWORD *)(v2 + 16) = v253;
            *(_DWORD *)(v2 + 32) = v231;
            if ( v228 > v231 )
              goto LABEL_44;
          }
          else
          {
            v230 = *v229;
            *(_QWORD *)(v2 + 16) = v226 + 8;
            v231 = 64;
          }
          *(_QWORD *)(v2 + 24) = v230 >> v228;
          *(_DWORD *)(v2 + 32) = v224 + v231 - 6;
          v232 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v224 + 58)) & v230) << v224) | v225;
        }
        if ( (v232 & 0x20) != 0 )
        {
          v233 = *(_DWORD *)(v2 + 32);
          if ( v233 <= 5 )
          {
LABEL_224:
            v234 = 0;
            if ( v233 )
              v234 = *(_QWORD *)(v2 + 24);
            v235 = *(_QWORD *)(v2 + 16);
            v236 = *(_QWORD *)(v2 + 8);
            v237 = 6 - v233;
            if ( v235 >= v236 )
              goto LABEL_44;
            v238 = (unsigned __int64 *)(v235 + *(_QWORD *)v2);
            if ( v236 >= v235 + 8 )
            {
              v239 = *v238;
              *(_QWORD *)(v2 + 16) = v235 + 8;
              v240 = 64;
LABEL_229:
              *(_QWORD *)(v2 + 24) = v239 >> v237;
              *(_DWORD *)(v2 + 32) = v233 + v240 - 6;
              v241 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v233 + 58)) & v239) << v233) | v234;
              goto LABEL_230;
            }
            *(_QWORD *)(v2 + 24) = 0;
            v243 = v236 - v235;
            if ( !v243 )
              goto LABEL_129;
            v244 = 0;
            v239 = 0;
            do
            {
              v245 = *((unsigned __int8 *)v238 + v244);
              v246 = 8 * v244++;
              v239 |= v245 << v246;
              *(_QWORD *)(v2 + 24) = v239;
            }
            while ( v243 != v244 );
            v247 = v235 + v243;
            v240 = 8 * v243;
            *(_QWORD *)(v2 + 16) = v247;
            *(_DWORD *)(v2 + 32) = v240;
            if ( v237 <= v240 )
              goto LABEL_229;
LABEL_44:
            sub_16BD130("Unexpected end of file", 1);
          }
          while ( 1 )
          {
            v242 = *(_QWORD *)(v2 + 24);
            *(_DWORD *)(v2 + 32) = v233 - 6;
            *(_QWORD *)(v2 + 24) = v242 >> 6;
            LOBYTE(v241) = v242 & 0x3F;
LABEL_230:
            if ( (v241 & 0x20) == 0 )
              break;
            v233 = *(_DWORD *)(v2 + 32);
            if ( v233 <= 5 )
              goto LABEL_224;
          }
        }
      }
    }
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 40);
    v4 = (unsigned int)(a2 - 4);
    if ( v4 >= (*(_QWORD *)(a1 + 48) - v3) >> 4 )
      sub_16BD130("Invalid abbrev number", 1);
    v5 = 16 * v4;
    v6 = **(int ***)(v3 + v5);
    v281 = *(_QWORD *)(v3 + v5);
    v7 = *((_BYTE *)v6 + 8);
    if ( (v7 & 1) != 0 )
    {
      v277 = *v6;
    }
    else
    {
      if ( ((((v7 >> 1) & 7) - 3) & 0xFD) == 0 )
        sub_16BD130("Abbreviation starts with an Array or a Blob", 1);
      v277 = sub_150F620(a1, (__int64 *)v6);
    }
    v280 = *(_DWORD *)(v281 + 8);
    if ( v280 > 1 )
    {
      v286 = 1;
      while ( 1 )
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)v281 + 16LL * v286;
          v10 = *(_BYTE *)(v9 + 8);
          if ( (v10 & 1) != 0 )
            goto LABEL_38;
          v11 = (v10 >> 1) & 7;
          if ( v11 != 3 )
            break;
          v46 = *(_DWORD *)(a1 + 32);
          if ( v46 > 5 )
          {
            v95 = *(_QWORD *)(a1 + 24);
            *(_DWORD *)(a1 + 32) = v46 - 6;
            *(_QWORD *)(a1 + 24) = v95 >> 6;
            LODWORD(v54) = v95 & 0x3F;
          }
          else
          {
            v47 = 0;
            if ( v46 )
              v47 = *(_QWORD *)(a1 + 24);
            v48 = *(_QWORD *)(a1 + 16);
            v49 = *(_QWORD *)(a1 + 8);
            v50 = 6 - v46;
            if ( v48 >= v49 )
              goto LABEL_44;
            v51 = (unsigned __int64 *)(v48 + *(_QWORD *)a1);
            if ( v49 < v48 + 8 )
            {
              *(_QWORD *)(a1 + 24) = 0;
              v142 = v49 - v48;
              if ( !v142 )
                goto LABEL_128;
              v143 = 0;
              v52 = 0;
              do
              {
                v144 = *((unsigned __int8 *)v51 + v143);
                v145 = 8 * v143++;
                v52 |= v144 << v145;
                *(_QWORD *)(a1 + 24) = v52;
              }
              while ( v142 != v143 );
              v146 = v48 + v142;
              v53 = 8 * v142;
              *(_QWORD *)(a1 + 16) = v146;
              *(_DWORD *)(a1 + 32) = v53;
              if ( v50 > v53 )
                goto LABEL_44;
            }
            else
            {
              v52 = *v51;
              *(_QWORD *)(a1 + 16) = v48 + 8;
              v53 = 64;
            }
            *(_QWORD *)(a1 + 24) = v52 >> v50;
            *(_DWORD *)(a1 + 32) = v46 + v53 - 6;
            v54 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v46 + 58)) & v52) << v46) | v47;
          }
          v283 = v54;
          if ( (v54 & 0x20) != 0 )
          {
            v55 = 0;
            v56 = v54 & 0x1F;
            do
            {
              v65 = *(_DWORD *)(a1 + 32);
              v55 += 5;
              if ( v65 <= 5 )
              {
                v57 = 0;
                if ( v65 )
                  v57 = *(_QWORD *)(a1 + 24);
                v58 = *(_QWORD *)(a1 + 16);
                v59 = *(_QWORD *)(a1 + 8);
                v60 = 6 - v65;
                if ( v58 >= v59 )
                  goto LABEL_44;
                v61 = (unsigned __int64 *)(v58 + *(_QWORD *)a1);
                if ( v59 < v58 + 8 )
                {
                  v67 = v59 - v58;
                  *(_QWORD *)(a1 + 24) = 0;
                  v284 = v67;
                  if ( !v67 )
                    goto LABEL_271;
                  v68 = v67;
                  v62 = 0;
                  for ( m = 0; m != v68; ++m )
                  {
                    v70 = *((unsigned __int8 *)v61 + m);
                    v71 = 8 * m;
                    v62 |= v70 << v71;
                    *(_QWORD *)(a1 + 24) = v62;
                  }
                  *(_QWORD *)(a1 + 16) = v58 + v68;
                  v63 = 8 * v284;
                  *(_DWORD *)(a1 + 32) = 8 * v284;
                  if ( v60 > 8 * v284 )
                    goto LABEL_44;
                }
                else
                {
                  v62 = *v61;
                  *(_QWORD *)(a1 + 16) = v58 + 8;
                  v63 = 64;
                }
                *(_DWORD *)(a1 + 32) = v65 + v63 - 6;
                *(_QWORD *)(a1 + 24) = v62 >> v60;
                v64 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v65 + 58)) & v62) << v65) | v57;
              }
              else
              {
                v66 = *(_QWORD *)(a1 + 24);
                *(_DWORD *)(a1 + 32) = v65 - 6;
                *(_QWORD *)(a1 + 24) = v66 >> 6;
                LOBYTE(v64) = v66 & 0x3F;
              }
              v56 |= (v64 & 0x1F) << v55;
            }
            while ( (v64 & 0x20) != 0 );
            v283 = v56;
          }
          v278 = *(_QWORD *)v281 + 16LL * ++v286;
          v72 = (*(_BYTE *)(v278 + 8) >> 1) & 7;
          switch ( v72 )
          {
            case 2:
              if ( v283 )
              {
                while ( 1 )
                {
                  v96 = *(_DWORD *)(a1 + 32);
                  v97 = *(_QWORD *)v278;
                  v98 = *(_QWORD *)v278;
                  if ( v98 <= v96 )
                  {
                    v125 = *(_QWORD *)(a1 + 24);
                    *(_DWORD *)(a1 + 32) = v96 - v97;
                    v107 = v125 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v97));
                    *(_QWORD *)(a1 + 24) = v125 >> v97;
                  }
                  else
                  {
                    v99 = 0;
                    if ( v96 )
                      v99 = *(_QWORD *)(a1 + 24);
                    v100 = *(_QWORD *)(a1 + 16);
                    v101 = *(_QWORD *)(a1 + 8);
                    v102 = v97 - v96;
                    if ( v100 >= v101 )
                      goto LABEL_44;
                    v103 = v100 + 8;
                    v104 = (_QWORD *)(v100 + *(_QWORD *)a1);
                    if ( v101 < v100 + 8 )
                    {
                      *(_QWORD *)(a1 + 24) = 0;
                      v126 = v101 - v100;
                      if ( !v126 )
                        goto LABEL_128;
                      v127 = v126;
                      v128 = 0;
                      v129 = 0;
                      do
                      {
                        v130 = *((unsigned __int8 *)v104 + v128);
                        v131 = 8 * v128++;
                        v129 |= v130 << v131;
                        *(_QWORD *)(a1 + 24) = v129;
                      }
                      while ( v126 != v128 );
                      v105 = 8 * v126;
                      v103 = v100 + v127;
                    }
                    else
                    {
                      v105 = 64;
                      *(_QWORD *)(a1 + 24) = *v104;
                    }
                    *(_QWORD *)(a1 + 16) = v103;
                    *(_DWORD *)(a1 + 32) = v105;
                    if ( v105 < v102 )
                      goto LABEL_44;
                    v106 = *(_QWORD *)(a1 + 24);
                    *(_DWORD *)(a1 + 32) = v96 - v97 + v105;
                    *(_QWORD *)(a1 + 24) = v106 >> v102;
                    v107 = ((v106 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v96 - (unsigned __int8)v97 + 64))) << v96)
                         | v99;
                  }
                  if ( _bittest((const int *)&v107, v97 - 1) )
                    break;
LABEL_117:
                  if ( !--v283 )
                    goto LABEL_38;
                }
                v279 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v97);
                v108 = 1 << (v97 - 1);
                while ( 1 )
                {
                  while ( 1 )
                  {
                    v117 = *(_DWORD *)(a1 + 32);
                    if ( v98 > v117 )
                      break;
                    v118 = *(_QWORD *)(a1 + 24);
                    *(_DWORD *)(a1 + 32) = v117 - v98;
                    *(_QWORD *)(a1 + 24) = v118 >> (v98 & 0x3F);
                    if ( ((unsigned int)v118 & (unsigned int)v279 & v108) == 0 )
                      goto LABEL_117;
                  }
                  LODWORD(v109) = 0;
                  if ( v117 )
                    v109 = *(_QWORD *)(a1 + 24);
                  v110 = *(_QWORD *)(a1 + 16);
                  v111 = *(_QWORD *)(a1 + 8);
                  v112 = v98 - v117;
                  if ( v110 >= v111 )
                    goto LABEL_44;
                  v113 = v110 + 8;
                  v114 = (_QWORD *)(v110 + *(_QWORD *)a1);
                  if ( v111 < v110 + 8 )
                  {
                    *(_QWORD *)(a1 + 24) = 0;
                    v119 = v111 - v110;
                    if ( !v119 )
                      goto LABEL_128;
                    v120 = v119;
                    v121 = 0;
                    v122 = 0;
                    do
                    {
                      v123 = *((unsigned __int8 *)v114 + v121);
                      v124 = 8 * v121++;
                      v122 |= v123 << v124;
                      *(_QWORD *)(a1 + 24) = v122;
                    }
                    while ( v119 != v121 );
                    v115 = 8 * v119;
                    v113 = v110 + v120;
                  }
                  else
                  {
                    v115 = 64;
                    *(_QWORD *)(a1 + 24) = *v114;
                  }
                  *(_QWORD *)(a1 + 16) = v113;
                  *(_DWORD *)(a1 + 32) = v115;
                  if ( v115 < v112 )
                    goto LABEL_44;
                  v116 = *(_QWORD *)(a1 + 24);
                  *(_DWORD *)(a1 + 32) = v117 - v98 + v115;
                  *(_QWORD *)(a1 + 24) = v116 >> v112;
                  if ( (((unsigned int)((v116
                                       & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v117 - (unsigned __int8)v98 + 64))) << v117)
                       | (unsigned int)v109)
                      & v108) == 0 )
                    goto LABEL_117;
                }
              }
LABEL_38:
              if ( v280 <= ++v286 )
                return v277;
              break;
            case 4:
              v73 = 8LL * *(_QWORD *)(a1 + 16);
              v74 = *(unsigned int *)(a1 + 32);
              *(_DWORD *)(a1 + 32) = 0;
              v75 = 6 * v283 + v73 - v74;
              v76 = (v75 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(a1 + 16) = v76;
              v77 = v75 & 0x3F;
              if ( (v75 & 0x3F) == 0 )
                goto LABEL_38;
              v78 = *(_QWORD *)(a1 + 8);
              if ( v76 >= v78 )
                goto LABEL_44;
              v79 = (unsigned __int64 *)(v76 + *(_QWORD *)a1);
              if ( v78 < v76 + 8 )
              {
                *(_QWORD *)(a1 + 24) = 0;
                v254 = v78 - v76;
                if ( !v254 )
                  goto LABEL_44;
                v255 = 0;
                v80 = 0;
                do
                {
                  v256 = *((unsigned __int8 *)v79 + v255);
                  v257 = 8 * v255++;
                  v80 |= v256 << v257;
                  *(_QWORD *)(a1 + 24) = v80;
                }
                while ( v254 != v255 );
                v258 = v254 + v76;
                v81 = 8 * v254;
                *(_QWORD *)(a1 + 16) = v258;
                *(_DWORD *)(a1 + 32) = v81;
                if ( v77 > v81 )
                  goto LABEL_44;
              }
              else
              {
                v80 = *v79;
                *(_QWORD *)(a1 + 16) = v76 + 8;
                v81 = 64;
              }
              ++v286;
              *(_DWORD *)(a1 + 32) = v81 - v77;
              *(_QWORD *)(a1 + 24) = v80 >> v77;
              if ( v280 <= v286 )
                return v277;
              break;
            case 1:
              v132 = *(unsigned int *)(a1 + 32);
              v133 = 8LL * *(_QWORD *)(a1 + 16);
              v134 = *(_QWORD *)v278 * v283;
              *(_DWORD *)(a1 + 32) = 0;
              v135 = v133 - v132 + v134;
              v136 = (v135 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(a1 + 16) = v136;
              v137 = v135 & 0x3F;
              if ( (v135 & 0x3F) == 0 )
                goto LABEL_38;
              v138 = *(_QWORD *)(a1 + 8);
              if ( v136 >= v138 )
                goto LABEL_44;
              v139 = (unsigned __int64 *)(v136 + *(_QWORD *)a1);
              if ( v138 < v136 + 8 )
              {
                *(_QWORD *)(a1 + 24) = 0;
                v259 = v138 - v136;
                if ( !v259 )
                  goto LABEL_44;
                v260 = 0;
                v140 = 0;
                do
                {
                  v261 = *((unsigned __int8 *)v139 + v260);
                  v262 = 8 * v260++;
                  v140 |= v261 << v262;
                  *(_QWORD *)(a1 + 24) = v140;
                }
                while ( v259 != v260 );
                v263 = v259 + v136;
                v141 = 8 * v259;
                *(_QWORD *)(a1 + 16) = v263;
                *(_DWORD *)(a1 + 32) = v141;
                if ( v137 > v141 )
                  goto LABEL_44;
              }
              else
              {
                v140 = *v139;
                *(_QWORD *)(a1 + 16) = v136 + 8;
                v141 = 64;
              }
              ++v286;
              *(_DWORD *)(a1 + 32) = v141 - v137;
              *(_QWORD *)(a1 + 24) = v140 >> v137;
              if ( v280 <= v286 )
                return v277;
              break;
            default:
              sub_16BD130("Array element type can't be an Array or a Blob", 1);
          }
        }
        switch ( v11 )
        {
          case 5:
            v12 = *(_DWORD *)(a1 + 32);
            v13 = *(_QWORD *)(a1 + 16);
            v14 = *(_QWORD *)(a1 + 8);
            if ( v12 > 5 )
            {
              v90 = *(_QWORD *)(a1 + 24);
              v20 = v12 - 6;
              *(_DWORD *)(a1 + 32) = v12 - 6;
              *(_QWORD *)(a1 + 24) = v90 >> 6;
              LODWORD(v21) = v90 & 0x3F;
            }
            else
            {
              v15 = 0;
              if ( v12 )
                v15 = *(_QWORD *)(a1 + 24);
              v16 = 6 - v12;
              if ( v13 >= v14 )
                goto LABEL_44;
              v17 = (unsigned __int64 *)(v13 + *(_QWORD *)a1);
              if ( v13 + 8 > v14 )
              {
                *(_QWORD *)(a1 + 24) = 0;
                v91 = v14 - v13;
                if ( (_DWORD)v14 == (_DWORD)v13 )
                {
LABEL_271:
                  *(_DWORD *)(a1 + 32) = 0;
                  goto LABEL_44;
                }
                v92 = 0;
                v18 = 0;
                do
                {
                  v93 = *((unsigned __int8 *)v17 + v92);
                  v94 = 8 * v92++;
                  v18 |= v93 << v94;
                  *(_QWORD *)(a1 + 24) = v18;
                }
                while ( v91 != v92 );
                v13 += v91;
                v19 = 8 * v91;
                *(_QWORD *)(a1 + 16) = v13;
                *(_DWORD *)(a1 + 32) = v19;
                if ( v16 > v19 )
                  goto LABEL_44;
              }
              else
              {
                v18 = *v17;
                *(_QWORD *)(a1 + 16) = v13 + 8;
                v13 += 8LL;
                v19 = 64;
              }
              v20 = v12 + v19 - 6;
              *(_DWORD *)(a1 + 32) = v20;
              *(_QWORD *)(a1 + 24) = v18 >> v16;
              v21 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v12 + 58)) & v18) << v12) | v15;
            }
            v22 = v21;
            if ( (v21 & 0x20) != 0 )
            {
              v22 = v21 & 0x1F;
              v23 = 0;
              do
              {
                v23 += 5;
                if ( v20 <= 5 )
                {
                  v24 = 0;
                  if ( v20 )
                    v24 = *(_QWORD *)(a1 + 24);
                  v282 = 6 - v20;
                  if ( v13 >= v14 )
                    goto LABEL_44;
                  v25 = (unsigned __int64 *)(v13 + *(_QWORD *)a1);
                  if ( v13 + 8 > v14 )
                  {
                    *(_QWORD *)(a1 + 24) = 0;
                    v42 = v14 - v13;
                    if ( (_DWORD)v14 == (_DWORD)v13 )
                    {
                      *(_QWORD *)(a1 + 16) = v13;
                      *(_DWORD *)(a1 + 32) = 0;
                      goto LABEL_44;
                    }
                    v43 = 0;
                    v26 = 0;
                    do
                    {
                      v44 = *((unsigned __int8 *)v25 + v43);
                      v45 = 8 * v43++;
                      v26 |= v44 << v45;
                      *(_QWORD *)(a1 + 24) = v26;
                    }
                    while ( v42 != v43 );
                    v13 += v42;
                    v27 = 8 * v42;
                    *(_QWORD *)(a1 + 16) = v13;
                    *(_DWORD *)(a1 + 32) = v27;
                    if ( v282 > v27 )
                      goto LABEL_44;
                  }
                  else
                  {
                    v26 = *v25;
                    *(_QWORD *)(a1 + 16) = v13 + 8;
                    v13 += 8LL;
                    v27 = 64;
                  }
                  *(_DWORD *)(a1 + 32) = v20 + v27 - 6;
                  *(_QWORD *)(a1 + 24) = v26 >> v282;
                  v28 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v20 + 58);
                  v29 = v20;
                  v20 = v20 + v27 - 6;
                  v30 = v24 | ((v28 & v26) << v29);
                }
                else
                {
                  v31 = *(_QWORD *)(a1 + 24);
                  v20 -= 6;
                  *(_DWORD *)(a1 + 32) = v20;
                  *(_QWORD *)(a1 + 24) = v31 >> 6;
                  LOBYTE(v30) = v31 & 0x3F;
                }
                v22 |= (v30 & 0x1F) << v23;
              }
              while ( (v30 & 0x20) != 0 );
            }
            if ( v20 > 0x1F )
            {
              *(_DWORD *)(a1 + 32) = 32;
              v32 = 32;
              *(_QWORD *)(a1 + 24) >>= (unsigned __int8)v20 - 32;
            }
            else
            {
              *(_DWORD *)(a1 + 32) = 0;
              v32 = 0;
            }
            v33 = ((8 * v22 + 24) & 0xE0) + 8 * v13 - v32;
            v34 = (((8 * v22 + 24) & 0xFFFFFFE0) + 8 * v13 - v32) >> 3;
            if ( v34 > v14 )
            {
              *(_QWORD *)(a1 + 16) = v14;
              return v277;
            }
            v35 = v34 & 0xFFFFFFFFFFFFFFF8LL;
            *(_DWORD *)(a1 + 32) = 0;
            *(_QWORD *)(a1 + 16) = v35;
            v36 = v35;
            v37 = v33 & 0x3F;
            if ( (v33 & 0x3F) != 0 )
            {
              if ( v36 >= v14 )
                goto LABEL_44;
              v38 = (unsigned __int64 *)(v36 + *(_QWORD *)a1);
              if ( v36 + 8 > v14 )
              {
                *(_QWORD *)(a1 + 24) = 0;
                v147 = v14 - v36;
                if ( !v147 )
                  goto LABEL_44;
                v148 = 0;
                v39 = 0;
                do
                {
                  v149 = *((unsigned __int8 *)v38 + v148);
                  v150 = 8 * v148++;
                  v39 |= v149 << v150;
                  *(_QWORD *)(a1 + 24) = v39;
                }
                while ( v147 != v148 );
                v151 = v147 + v36;
                v40 = 8 * v147;
                *(_QWORD *)(a1 + 16) = v151;
                *(_DWORD *)(a1 + 32) = v40;
                if ( v37 > v40 )
                  goto LABEL_44;
              }
              else
              {
                v39 = *v38;
                *(_QWORD *)(a1 + 16) = v36 + 8;
                v40 = 64;
              }
              *(_DWORD *)(a1 + 32) = v40 - v37;
              *(_QWORD *)(a1 + 24) = v39 >> v37;
            }
            goto LABEL_38;
          case 2:
            sub_150F320(a1, *(_QWORD *)v9);
            if ( v280 <= ++v286 )
              return v277;
            break;
          case 4:
            v152 = *(_DWORD *)(a1 + 32);
            if ( v152 > 5 )
            {
              ++v286;
              *(_DWORD *)(a1 + 32) = v152 - 6;
              *(_QWORD *)(a1 + 24) >>= 6;
              if ( v280 <= v286 )
                return v277;
            }
            else
            {
              v153 = *(_QWORD *)(a1 + 16);
              v154 = *(_QWORD *)(a1 + 8);
              v155 = 6 - v152;
              if ( v153 >= v154 )
                goto LABEL_44;
              v156 = (unsigned __int64 *)(v153 + *(_QWORD *)a1);
              if ( v154 >= v153 + 8 )
              {
                v159 = *v156;
                *(_QWORD *)(a1 + 16) = v153 + 8;
                v163 = 64;
              }
              else
              {
                *(_QWORD *)(a1 + 24) = 0;
                v157 = v154 - v153;
                if ( !v157 )
                  goto LABEL_128;
                v158 = 0;
                v159 = 0;
                do
                {
                  v160 = *((unsigned __int8 *)v156 + v158);
                  v161 = 8 * v158++;
                  v159 |= v160 << v161;
                  *(_QWORD *)(a1 + 24) = v159;
                }
                while ( v157 != v158 );
                v162 = v153 + v157;
                v163 = 8 * v157;
                *(_QWORD *)(a1 + 16) = v162;
                *(_DWORD *)(a1 + 32) = v163;
                if ( v155 > v163 )
                  goto LABEL_44;
              }
              ++v286;
              *(_DWORD *)(a1 + 32) = v152 + v163 - 6;
              *(_QWORD *)(a1 + 24) = v159 >> v155;
              if ( v280 <= v286 )
                return v277;
            }
            break;
          case 1:
            v82 = *(_QWORD *)v9;
            v83 = *(_DWORD *)(a1 + 32);
            if ( (unsigned int)*(_QWORD *)v9 <= v83 )
            {
              ++v286;
              *(_DWORD *)(a1 + 32) = v83 - v82;
              *(_QWORD *)(a1 + 24) >>= v82;
              if ( v280 <= v286 )
                return v277;
            }
            else
            {
              v84 = *(_QWORD *)(a1 + 16);
              v85 = *(_QWORD *)(a1 + 8);
              v86 = v82 - v83;
              if ( v84 >= v85 )
                goto LABEL_44;
              v87 = v84 + 8;
              v88 = (_QWORD *)(v84 + *(_QWORD *)a1);
              if ( v85 < v84 + 8 )
              {
                *(_QWORD *)(a1 + 24) = 0;
                v188 = v85 - v84;
                if ( !v188 )
                {
LABEL_128:
                  v2 = a1;
LABEL_129:
                  *(_DWORD *)(v2 + 32) = 0;
                  goto LABEL_44;
                }
                v189 = 0;
                v190 = 0;
                do
                {
                  v191 = *((unsigned __int8 *)v88 + v189);
                  v192 = 8 * v189++;
                  v190 |= v191 << v192;
                  *(_QWORD *)(a1 + 24) = v190;
                }
                while ( v188 != v189 );
                v89 = 8 * v188;
                v87 = v84 + v188;
              }
              else
              {
                *(_QWORD *)(a1 + 24) = *v88;
                v89 = 64;
              }
              *(_QWORD *)(a1 + 16) = v87;
              *(_DWORD *)(a1 + 32) = v89;
              if ( v86 > v89 )
                goto LABEL_44;
              ++v286;
              *(_QWORD *)(a1 + 24) >>= v86;
              *(_DWORD *)(a1 + 32) = v83 - v82 + v89;
              if ( v280 <= v286 )
                return v277;
            }
            break;
          default:
            goto LABEL_38;
        }
      }
    }
  }
  return v277;
}
