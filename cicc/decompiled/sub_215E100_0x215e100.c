// Function: sub_215E100
// Address: 0x215e100
//
__int64 __fastcall sub_215E100(
        __int64 a1,
        _QWORD **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int8 v13; // al
  __int64 v14; // r13
  unsigned int v15; // esi
  __int64 v16; // rdi
  int v17; // edx
  __int64 v18; // rax
  _QWORD *v19; // rbx
  int v20; // ecx
  __int64 v21; // rdx
  unsigned __int64 *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // r8
  int v35; // r9d
  __int64 v36; // rax
  __int64 v37; // r8
  unsigned int v38; // edx
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r15
  _QWORD *v43; // rax
  _QWORD *v44; // r12
  __int64 v45; // r13
  __int64 v46; // rdi
  unsigned __int64 *v47; // rbx
  __int64 v48; // rax
  unsigned __int64 v49; // rsi
  __int64 v50; // rsi
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 v53; // rsi
  unsigned int v54; // ecx
  __int64 v55; // rdx
  __int64 v56; // r8
  __int64 **v57; // rdx
  int v58; // ecx
  int v59; // r9d
  _QWORD *v60; // r8
  __int64 v61; // rdi
  unsigned int v62; // edx
  __int64 v63; // rsi
  int v64; // edx
  int v65; // r9d
  unsigned int v66; // eax
  __int64 v67; // rdi
  __int64 v68; // rbx
  __int64 v69; // r14
  __int64 v70; // r8
  int v71; // r9d
  __int64 v72; // rax
  unsigned __int16 v73; // dx
  unsigned int v74; // r15d
  __int64 **v75; // rdx
  _DWORD *v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rdx
  unsigned int *v79; // r10
  __int64 v80; // r15
  __int64 *v81; // r14
  _QWORD *v82; // r14
  __int64 v83; // r15
  _QWORD *v84; // r14
  _BYTE *v85; // rdx
  __int64 v86; // r15
  __int64 v87; // r14
  __int64 v88; // r15
  unsigned __int16 v89; // ax
  __int64 v90; // r15
  __int64 v91; // r14
  unsigned __int64 v92; // r13
  bool v93; // zf
  __int64 *v94; // r14
  __int64 v95; // r15
  __int64 v96; // rax
  _QWORD *v97; // rax
  __int64 v98; // rax
  __int64 *v99; // rax
  __int64 *v100; // rdi
  __int64 *v101; // rcx
  __int64 *v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rax
  __int64 *v105; // r14
  __int64 v106; // rax
  __int64 v107; // rcx
  __int64 v108; // rdi
  __int64 v109; // rsi
  __int64 *v110; // r14
  __int64 v111; // r15
  __int64 v112; // rsi
  unsigned __int8 *v113; // rsi
  int v114; // r11d
  _QWORD *v115; // r10
  int v116; // ecx
  int v117; // edx
  int v118; // edx
  int v119; // r9d
  __int64 v120; // rdi
  unsigned int v121; // ecx
  __int64 v122; // rsi
  __int64 v123; // r15
  __int64 *v124; // r12
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // r14
  _QWORD *v128; // rax
  _QWORD *v129; // rbx
  __int64 v130; // rdi
  unsigned __int64 *v131; // r13
  __int64 v132; // rax
  unsigned __int64 v133; // rsi
  __int64 v134; // rsi
  __int64 v135; // rsi
  __int64 v136; // rdx
  unsigned __int8 *v137; // rsi
  __int64 v138; // rax
  __int64 *v139; // r14
  __int64 v140; // rax
  __int64 v141; // rcx
  __int64 v142; // rsi
  __int64 v143; // rsi
  unsigned __int8 *v144; // rsi
  int v145; // edx
  int v146; // r9d
  _QWORD *v147; // rax
  __int64 v148; // rax
  __int64 *v149; // r14
  __int64 v150; // rax
  __int64 v151; // rcx
  _QWORD *v152; // rax
  __int64 v153; // rdx
  unsigned __int64 v154; // rsi
  __int64 v155; // rdx
  __int64 v156; // rdx
  unsigned __int64 v157; // rsi
  __int64 v158; // rdx
  __int64 v159; // rdx
  unsigned __int64 v160; // rcx
  __int64 v161; // rdx
  __int64 v162; // rax
  __int64 *v163; // r14
  __int64 v164; // rax
  __int64 v165; // rcx
  __int64 v166; // rsi
  __int64 v167; // rsi
  __int64 v168; // rax
  _QWORD *v169; // rax
  __int64 v170; // rax
  __int64 *v171; // rax
  __int64 *v172; // rdi
  __int64 *v173; // rcx
  __int64 *v174; // rax
  __int64 v175; // rdx
  _QWORD *v176; // rax
  _QWORD *v177; // rax
  _QWORD **v178; // rax
  __int64 *v179; // rax
  __int64 v180; // rsi
  __int64 v181; // rax
  __int64 *v182; // r14
  __int64 v183; // rax
  __int64 v184; // rcx
  __int64 *v185; // rsi
  __int64 v186; // rdx
  __int64 v187; // rax
  __int64 v188; // rax
  __int64 v189; // rdx
  int v190; // r15d
  __int64 v191; // rax
  __int64 *v192; // r15
  __int64 v193; // rax
  __int64 v194; // rcx
  __int64 v195; // rax
  __int64 v196; // rdx
  unsigned __int64 v197; // rsi
  __int64 v198; // rdx
  __int64 v199; // rsi
  __int64 v200; // rax
  __int64 v201; // rax
  unsigned __int8 v202; // r8
  __int64 v203; // [rsp+0h] [rbp-F0h]
  unsigned int v204; // [rsp+0h] [rbp-F0h]
  int v205; // [rsp+8h] [rbp-E8h]
  __int64 v206; // [rsp+8h] [rbp-E8h]
  __int64 v207; // [rsp+8h] [rbp-E8h]
  int v208; // [rsp+8h] [rbp-E8h]
  unsigned int v209; // [rsp+18h] [rbp-D8h]
  __int64 v210; // [rsp+18h] [rbp-D8h]
  __int64 v211; // [rsp+18h] [rbp-D8h]
  __int64 v212; // [rsp+20h] [rbp-D0h]
  _QWORD **v213; // [rsp+20h] [rbp-D0h]
  _QWORD *v214; // [rsp+20h] [rbp-D0h]
  __int64 v216; // [rsp+20h] [rbp-D0h]
  _QWORD *v217; // [rsp+20h] [rbp-D0h]
  __int64 v218; // [rsp+28h] [rbp-C8h]
  __int64 v219; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v220; // [rsp+28h] [rbp-C8h]
  _QWORD *v221; // [rsp+28h] [rbp-C8h]
  _QWORD *v222; // [rsp+28h] [rbp-C8h]
  __int64 v223; // [rsp+28h] [rbp-C8h]
  char v224; // [rsp+30h] [rbp-C0h]
  char v226; // [rsp+30h] [rbp-C0h]
  __int16 v227; // [rsp+30h] [rbp-C0h]
  _BYTE *v228; // [rsp+30h] [rbp-C0h]
  _QWORD **v229; // [rsp+30h] [rbp-C0h]
  _QWORD *v230; // [rsp+30h] [rbp-C0h]
  __int64 v231; // [rsp+30h] [rbp-C0h]
  __int64 v232; // [rsp+30h] [rbp-C0h]
  unsigned int *v233; // [rsp+30h] [rbp-C0h]
  __int64 v235; // [rsp+38h] [rbp-B8h]
  int v236; // [rsp+44h] [rbp-ACh] BYREF
  __int64 v237; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v238[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v239; // [rsp+60h] [rbp-90h]
  __int64 v240[2]; // [rsp+70h] [rbp-80h] BYREF
  __int16 v241; // [rsp+80h] [rbp-70h]
  __int64 *v242; // [rsp+90h] [rbp-60h] BYREF
  __int64 v243; // [rsp+98h] [rbp-58h] BYREF
  __int64 v244; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v245; // [rsp+A8h] [rbp-48h]
  __int64 v246; // [rsp+B0h] [rbp-40h]

  v10 = a1;
  v11 = a4;
  v12 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v12 )
  {
    v24 = *(_QWORD *)(a1 + 248);
    v25 = (v12 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v26 = v24 + 48LL * v25;
    v27 = *(_QWORD *)(v26 + 24);
    if ( v27 == v11 )
    {
LABEL_24:
      if ( v26 != v24 + 48 * v12 )
        return *(_QWORD *)(v26 + 40);
    }
    else
    {
      v64 = 1;
      while ( v27 != -8 )
      {
        v65 = v64 + 1;
        v25 = (v12 - 1) & (v64 + v25);
        v26 = v24 + 48LL * v25;
        v27 = *(_QWORD *)(v26 + 24);
        if ( v27 == v11 )
          goto LABEL_24;
        v64 = v65;
      }
    }
  }
  v13 = *(_BYTE *)(v11 + 16);
  if ( v13 != 3 )
  {
    if ( (unsigned int)v13 - 6 > 2 )
    {
      v14 = v11;
      if ( v13 != 5 )
        goto LABEL_5;
      v242 = &v244;
      v243 = 0x400000000LL;
      v66 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
      v204 = v66;
      if ( !v66 )
        goto LABEL_5;
      v226 = 0;
      v220 = v66 - 1;
      v67 = v11;
      v68 = 0;
      v213 = a2;
      v14 = v67;
      while ( 1 )
      {
        v69 = *(_QWORD *)(v67 + 24 * (v68 - v66));
        v70 = sub_215E100(v10, v213, a3, v69, a5);
        v226 |= v69 != v70;
        v72 = (unsigned int)v243;
        if ( (unsigned int)v243 >= HIDWORD(v243) )
        {
          v206 = v70;
          sub_16CD150((__int64)&v242, &v244, 0, 8, v70, v71);
          v72 = (unsigned int)v243;
          v70 = v206;
        }
        v242[v72] = v70;
        LODWORD(v243) = v243 + 1;
        if ( v220 == v68 )
          break;
        ++v68;
        v66 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
      }
      v11 = v67;
      if ( !v226 )
        goto LABEL_60;
      v73 = *(_WORD *)(v67 + 18);
      switch ( v73 )
      {
        case ' ':
          v92 = (unsigned __int64)v242;
          v93 = (*(_BYTE *)(v67 + 17) & 2) == 0;
          v239 = 257;
          v94 = v242 + 1;
          v228 = (_BYTE *)*v242;
          if ( !v93 )
          {
            v95 = sub_16348C0(v67);
            if ( v228[16] > 0x10u )
            {
LABEL_114:
              v241 = 257;
              if ( !v95 )
              {
                v201 = *(_QWORD *)v228;
                if ( *(_BYTE *)(*(_QWORD *)v228 + 8LL) == 16 )
                  v201 = **(_QWORD **)(v201 + 16);
                v95 = *(_QWORD *)(v201 + 24);
              }
              v97 = sub_1648A60(72, v204);
              v14 = (__int64)v97;
              if ( v97 )
              {
                v214 = v97;
                v210 = (__int64)&v97[-3 * v204];
                v98 = *(_QWORD *)v228;
                if ( *(_BYTE *)(*(_QWORD *)v228 + 8LL) == 16 )
                  v98 = **(_QWORD **)(v98 + 16);
                v205 = *(_DWORD *)(v98 + 8) >> 8;
                v99 = (__int64 *)sub_15F9F50(v95, (__int64)v94, v220);
                v100 = (__int64 *)sub_1646BA0(v99, v205);
                if ( *(_BYTE *)(*(_QWORD *)v228 + 8LL) == 16 )
                {
                  v100 = sub_16463B0(v100, *(_QWORD *)(*(_QWORD *)v228 + 32LL));
                }
                else
                {
                  v101 = &v94[v220];
                  if ( v94 != v101 )
                  {
                    v102 = v94;
                    while ( 1 )
                    {
                      v103 = *(_QWORD *)*v102;
                      if ( *(_BYTE *)(v103 + 8) == 16 )
                        break;
                      if ( v101 == ++v102 )
                        goto LABEL_124;
                    }
                    v100 = sub_16463B0(v100, *(_QWORD *)(v103 + 32));
                  }
                }
LABEL_124:
                sub_15F1EA0(v14, (__int64)v100, 32, v210, v204, 0);
                *(_QWORD *)(v14 + 56) = v95;
                *(_QWORD *)(v14 + 64) = sub_15F9F50(v95, (__int64)v94, v220);
                sub_15F9CE0(v14, (__int64)v228, v94, v220, (__int64)v240);
              }
              else
              {
                v214 = 0;
              }
              goto LABEL_125;
            }
            if ( v220 )
            {
              v96 = 0;
              while ( *(_BYTE *)(*(_QWORD *)(v92 + 8 * v96 + 8) + 16LL) <= 0x10u )
              {
                if ( ++v96 == v220 )
                  goto LABEL_266;
              }
              goto LABEL_114;
            }
LABEL_266:
            BYTE4(v240[0]) = 0;
            v202 = 0;
            goto LABEL_267;
          }
          v95 = sub_16348C0(v67);
          if ( v228[16] <= 0x10u )
          {
            if ( !v220 )
            {
LABEL_268:
              BYTE4(v240[0]) = 0;
              v202 = 1;
LABEL_267:
              v14 = sub_15A2E80(v95, (__int64)v228, (__int64 **)v94, v220, v202, (__int64)v240, 0);
LABEL_60:
              if ( v242 != &v244 )
                _libc_free((unsigned __int64)v242);
              goto LABEL_5;
            }
            v168 = 0;
            while ( *(_BYTE *)(*(_QWORD *)(v92 + 8 * v168 + 8) + 16LL) <= 0x10u )
            {
              if ( ++v168 == v220 )
                goto LABEL_268;
            }
          }
          v241 = 257;
          if ( !v95 )
          {
            v200 = *(_QWORD *)v228;
            if ( *(_BYTE *)(*(_QWORD *)v228 + 8LL) == 16 )
              v200 = **(_QWORD **)(v200 + 16);
            v95 = *(_QWORD *)(v200 + 24);
          }
          v169 = sub_1648A60(72, v204);
          v14 = (__int64)v169;
          if ( v169 )
          {
            v214 = v169;
            v211 = (__int64)&v169[-3 * v204];
            v170 = *(_QWORD *)v228;
            if ( *(_BYTE *)(*(_QWORD *)v228 + 8LL) == 16 )
              v170 = **(_QWORD **)(v170 + 16);
            v208 = *(_DWORD *)(v170 + 8) >> 8;
            v171 = (__int64 *)sub_15F9F50(v95, (__int64)v94, v220);
            v172 = (__int64 *)sub_1646BA0(v171, v208);
            if ( *(_BYTE *)(*(_QWORD *)v228 + 8LL) == 16 )
            {
              v172 = sub_16463B0(v172, *(_QWORD *)(*(_QWORD *)v228 + 32LL));
            }
            else
            {
              v173 = &v94[v220];
              if ( v94 != v173 )
              {
                v174 = v94;
                while ( 1 )
                {
                  v175 = *(_QWORD *)*v174;
                  if ( *(_BYTE *)(v175 + 8) == 16 )
                    break;
                  if ( v173 == ++v174 )
                    goto LABEL_227;
                }
                v172 = sub_16463B0(v172, *(_QWORD *)(v175 + 32));
              }
            }
LABEL_227:
            sub_15F1EA0(v14, (__int64)v172, 32, v211, v204, 0);
            *(_QWORD *)(v14 + 56) = v95;
            *(_QWORD *)(v14 + 64) = sub_15F9F50(v95, (__int64)v94, v220);
            sub_15F9CE0(v14, (__int64)v228, v94, v220, (__int64)v240);
          }
          else
          {
            v214 = 0;
          }
          sub_15FA2E0(v14, 1);
LABEL_125:
          v104 = *(_QWORD *)(a5 + 8);
          if ( v104 )
          {
            v105 = *(__int64 **)(a5 + 16);
            sub_157E9D0(v104 + 40, v14);
            v106 = *(_QWORD *)(v14 + 24);
            v107 = *v105;
            *(_QWORD *)(v14 + 32) = v105;
            v107 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v14 + 24) = v107 | v106 & 7;
            *(_QWORD *)(v107 + 8) = v14 + 24;
            *v105 = *v105 & 7 | (v14 + 24);
          }
          v108 = (__int64)v214;
          goto LABEL_128;
        case '3':
          v239 = 257;
          v87 = *v242;
          v88 = v242[1];
          v89 = sub_1594720(v67);
          v227 = v89;
          if ( *(_BYTE *)(v87 + 16) <= 0x10u && *(_BYTE *)(v88 + 16) <= 0x10u )
          {
            v14 = sub_15A37B0(v89, (_QWORD *)v87, (_QWORD *)v88, 0);
            goto LABEL_60;
          }
          v241 = 257;
          v177 = sub_1648A60(56, 2u);
          v14 = (__int64)v177;
          if ( v177 )
          {
            v222 = v177;
            v178 = *(_QWORD ***)v87;
            if ( *(_BYTE *)(*(_QWORD *)v87 + 8LL) == 16 )
            {
              v217 = v178[4];
              v179 = (__int64 *)sub_1643320(*v178);
              v180 = (__int64)sub_16463B0(v179, (unsigned int)v217);
            }
            else
            {
              v180 = sub_1643320(*v178);
            }
            sub_15FEC10(v14, v180, 51, v227, v87, v88, (__int64)v240, 0);
          }
          else
          {
            v222 = 0;
          }
          v181 = *(_QWORD *)(a5 + 8);
          if ( v181 )
          {
            v182 = *(__int64 **)(a5 + 16);
            sub_157E9D0(v181 + 40, v14);
            v183 = *(_QWORD *)(v14 + 24);
            v184 = *v182;
            *(_QWORD *)(v14 + 32) = v182;
            v184 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v14 + 24) = v184 | v183 & 7;
            *(_QWORD *)(v184 + 8) = v14 + 24;
            *v182 = *v182 & 7 | (v14 + 24);
          }
          v108 = (__int64)v222;
          goto LABEL_128;
        case '7':
          v239 = 257;
          v90 = *v242;
          v91 = v242[1];
          if ( *(_BYTE *)(*v242 + 16) <= 0x10u && *(_BYTE *)(v91 + 16) <= 0x10u && *(_BYTE *)(v242[2] + 16) <= 0x10u )
          {
            v14 = sub_15A2DC0(*v242, (__int64 *)v242[1], v242[2], 0);
            goto LABEL_60;
          }
          v231 = v242[2];
          v241 = 257;
          v152 = sub_1648A60(56, 3u);
          v14 = (__int64)v152;
          if ( v152 )
          {
            v216 = v231;
            v221 = v152 - 9;
            v232 = (__int64)v152;
            sub_15F1EA0((__int64)v152, *(_QWORD *)v91, 55, (__int64)(v152 - 9), 3, 0);
            if ( *(_QWORD *)(v14 - 72) )
            {
              v153 = *(_QWORD *)(v14 - 64);
              v154 = *(_QWORD *)(v14 - 56) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v154 = v153;
              if ( v153 )
                *(_QWORD *)(v153 + 16) = v154 | *(_QWORD *)(v153 + 16) & 3LL;
            }
            *(_QWORD *)(v14 - 72) = v90;
            v155 = *(_QWORD *)(v90 + 8);
            *(_QWORD *)(v14 - 64) = v155;
            if ( v155 )
              *(_QWORD *)(v155 + 16) = (v14 - 64) | *(_QWORD *)(v155 + 16) & 3LL;
            *(_QWORD *)(v14 - 56) = *(_QWORD *)(v14 - 56) & 3LL | (v90 + 8);
            *(_QWORD *)(v90 + 8) = v221;
            if ( *(_QWORD *)(v14 - 48) )
            {
              v156 = *(_QWORD *)(v14 - 40);
              v157 = *(_QWORD *)(v14 - 32) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v157 = v156;
              if ( v156 )
                *(_QWORD *)(v156 + 16) = v157 | *(_QWORD *)(v156 + 16) & 3LL;
            }
            *(_QWORD *)(v14 - 48) = v91;
            v158 = *(_QWORD *)(v91 + 8);
            *(_QWORD *)(v14 - 40) = v158;
            if ( v158 )
              *(_QWORD *)(v158 + 16) = (v14 - 40) | *(_QWORD *)(v158 + 16) & 3LL;
            *(_QWORD *)(v14 - 32) = *(_QWORD *)(v14 - 32) & 3LL | (v91 + 8);
            *(_QWORD *)(v91 + 8) = v14 - 48;
            if ( *(_QWORD *)(v14 - 24) )
            {
              v159 = *(_QWORD *)(v14 - 16);
              v160 = *(_QWORD *)(v14 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v160 = v159;
              if ( v159 )
                *(_QWORD *)(v159 + 16) = v160 | *(_QWORD *)(v159 + 16) & 3LL;
            }
            *(_QWORD *)(v14 - 24) = v216;
            if ( v216 )
            {
              v161 = *(_QWORD *)(v216 + 8);
              *(_QWORD *)(v14 - 16) = v161;
              if ( v161 )
                *(_QWORD *)(v161 + 16) = (v14 - 16) | *(_QWORD *)(v161 + 16) & 3LL;
              *(_QWORD *)(v14 - 8) = (v216 + 8) | *(_QWORD *)(v14 - 8) & 3LL;
              *(_QWORD *)(v216 + 8) = v14 - 24;
            }
            sub_164B780(v14, v240);
          }
          else
          {
            v232 = 0;
          }
          v162 = *(_QWORD *)(a5 + 8);
          if ( v162 )
          {
            v163 = *(__int64 **)(a5 + 16);
            sub_157E9D0(v162 + 40, v14);
            v164 = *(_QWORD *)(v14 + 24);
            v165 = *v163;
            *(_QWORD *)(v14 + 32) = v163;
            v165 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v14 + 24) = v165 | v164 & 7;
            *(_QWORD *)(v165 + 8) = v14 + 24;
            *v163 = *v163 & 7 | (v14 + 24);
          }
          sub_164B780(v232, v238);
          v166 = *(_QWORD *)a5;
          if ( !*(_QWORD *)a5 )
            goto LABEL_60;
          v240[0] = *(_QWORD *)a5;
          v110 = v240;
          goto LABEL_203;
        case ';':
          v239 = 257;
          v82 = (_QWORD *)*v242;
          v83 = v242[1];
          if ( *(_BYTE *)(*v242 + 16) <= 0x10u && *(_BYTE *)(v83 + 16) <= 0x10u )
          {
            v14 = sub_15A37D0((_BYTE *)*v242, v242[1], 0);
            goto LABEL_60;
          }
          v241 = 257;
          v176 = sub_1648A60(56, 2u);
          v14 = (__int64)v176;
          if ( v176 )
            sub_15FA320((__int64)v176, v82, v83, (__int64)v240, 0);
          goto LABEL_177;
        case '<':
          v241 = 257;
          v14 = sub_156D8B0((__int64 *)a5, *v242, v242[1], v242[2], (__int64)v240);
          goto LABEL_60;
        case '=':
          v239 = 257;
          v84 = (_QWORD *)*v242;
          v85 = (_BYTE *)v242[2];
          v86 = v242[1];
          if ( *(_BYTE *)(*v242 + 16) <= 0x10u && *(_BYTE *)(v86 + 16) <= 0x10u && v85[16] <= 0x10u )
          {
            v14 = sub_15A3950(*v242, v242[1], v85, 0);
            goto LABEL_60;
          }
          v230 = (_QWORD *)v242[2];
          v241 = 257;
          v147 = sub_1648A60(56, 3u);
          v14 = (__int64)v147;
          if ( v147 )
            sub_15FA660((__int64)v147, v84, v86, v230, (__int64)v240, 0);
          goto LABEL_177;
        case '>':
          v239 = 257;
          v79 = (unsigned int *)sub_1594710(v67);
          v80 = v78;
          v81 = (__int64 *)*v242;
          if ( *(_BYTE *)(*v242 + 16) <= 0x10u )
          {
            v14 = sub_15A3AE0((_QWORD *)*v242, v79, v78, 0);
            goto LABEL_60;
          }
          v233 = v79;
          v241 = 257;
          v14 = (__int64)sub_1648A60(88, 1u);
          if ( v14 )
          {
            v195 = sub_15FB2A0(*v81, v233, v80);
            sub_15F1EA0(v14, v195, 62, v14 - 24, 1, 0);
            if ( *(_QWORD *)(v14 - 24) )
            {
              v196 = *(_QWORD *)(v14 - 16);
              v197 = *(_QWORD *)(v14 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v197 = v196;
              if ( v196 )
                *(_QWORD *)(v196 + 16) = v197 | *(_QWORD *)(v196 + 16) & 3LL;
            }
            *(_QWORD *)(v14 - 24) = v81;
            v198 = v81[1];
            *(_QWORD *)(v14 - 16) = v198;
            if ( v198 )
              *(_QWORD *)(v198 + 16) = (v14 - 16) | *(_QWORD *)(v198 + 16) & 3LL;
            *(_QWORD *)(v14 - 8) = *(_QWORD *)(v14 - 8) & 3LL | (unsigned __int64)(v81 + 1);
            v81[1] = v14 - 24;
            *(_QWORD *)(v14 + 56) = v14 + 72;
            *(_QWORD *)(v14 + 64) = 0x400000000LL;
            sub_15FB110(v14, v233, v80, (__int64)v240);
          }
          goto LABEL_177;
        case '?':
          v241 = 257;
          v76 = (_DWORD *)sub_1594710(v67);
          v14 = sub_17FE490((__int64 *)a5, *v242, v242[1], v76, v77, v240);
          goto LABEL_60;
        default:
          v74 = v73;
          v239 = 257;
          if ( (unsigned int)v73 - 11 <= 0x11 )
          {
            v185 = (__int64 *)*v242;
            v186 = v242[1];
            if ( *(_BYTE *)(*v242 + 16) <= 0x10u && *(_BYTE *)(v186 + 16) <= 0x10u )
            {
              v223 = v242[1];
              v187 = sub_15A2A30((__int64 *)v74, v185, v186, 0, 0, a6, a7, a8);
              v186 = v223;
              v14 = v187;
              if ( v187 )
                goto LABEL_60;
            }
            v110 = v240;
            v241 = 257;
            v14 = sub_15FB440(v74, v185, v186, (__int64)v240, 0);
            v188 = *(_QWORD *)v14;
            if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
              v188 = **(_QWORD **)(v188 + 16);
            if ( (unsigned __int8)(*(_BYTE *)(v188 + 8) - 1) <= 5u || *(_BYTE *)(v14 + 16) == 76 )
            {
              v189 = *(_QWORD *)(a5 + 32);
              v190 = *(_DWORD *)(a5 + 40);
              if ( v189 )
                sub_1625C10(v14, 3, v189);
              sub_15F2440(v14, v190);
            }
            v191 = *(_QWORD *)(a5 + 8);
            if ( v191 )
            {
              v192 = *(__int64 **)(a5 + 16);
              sub_157E9D0(v191 + 40, v14);
              v193 = *(_QWORD *)(v14 + 24);
              v194 = *v192;
              *(_QWORD *)(v14 + 32) = v192;
              v194 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v14 + 24) = v194 | v193 & 7;
              *(_QWORD *)(v194 + 8) = v14 + 24;
              *v192 = *v192 & 7 | (v14 + 24);
            }
            sub_164B780(v14, v238);
            v166 = *(_QWORD *)a5;
            if ( !*(_QWORD *)a5 )
              goto LABEL_60;
            v240[0] = *(_QWORD *)a5;
LABEL_203:
            v111 = v14 + 48;
            sub_1623A60((__int64)v240, v166, 2);
            v167 = *(_QWORD *)(v14 + 48);
            if ( v167 )
              sub_161E7C0(v14 + 48, v167);
            v113 = (unsigned __int8 *)v240[0];
            *(_QWORD *)(v14 + 48) = v240[0];
            if ( !v113 )
              goto LABEL_60;
          }
          else
          {
            v14 = *v242;
            v75 = *(__int64 ***)v67;
            if ( *(_QWORD *)v67 == *(_QWORD *)*v242 )
              goto LABEL_60;
            if ( *(_BYTE *)(v14 + 16) <= 0x10u )
            {
              v14 = sub_15A46C0(v74, (__int64 ***)*v242, v75, 0);
              goto LABEL_60;
            }
            v199 = *v242;
            v241 = 257;
            v14 = sub_15FDBD0(v74, v199, (__int64)v75, (__int64)v240, 0);
LABEL_177:
            v148 = *(_QWORD *)(a5 + 8);
            if ( v148 )
            {
              v149 = *(__int64 **)(a5 + 16);
              sub_157E9D0(v148 + 40, v14);
              v150 = *(_QWORD *)(v14 + 24);
              v151 = *v149;
              *(_QWORD *)(v14 + 32) = v149;
              v151 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v14 + 24) = v151 | v150 & 7;
              *(_QWORD *)(v151 + 8) = v14 + 24;
              *v149 = *v149 & 7 | (v14 + 24);
            }
            v108 = v14;
LABEL_128:
            sub_164B780(v108, v238);
            v109 = *(_QWORD *)a5;
            if ( !*(_QWORD *)a5 )
              goto LABEL_60;
            v110 = &v237;
            v237 = *(_QWORD *)a5;
            v111 = v14 + 48;
            sub_1623A60((__int64)&v237, v109, 2);
            v112 = *(_QWORD *)(v14 + 48);
            if ( v112 )
              sub_161E7C0(v14 + 48, v112);
            v113 = (unsigned __int8 *)v237;
            *(_QWORD *)(v14 + 48) = v237;
            if ( !v113 )
              goto LABEL_60;
          }
          sub_1623210((__int64)v110, v113, v111);
          goto LABEL_60;
      }
    }
    v242 = &v244;
    v243 = 0x400000000LL;
    v209 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
    if ( v209 )
    {
      v224 = 0;
      v212 = a3;
      v218 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
      v30 = v11;
      v31 = 0;
      v14 = v30;
      do
      {
        if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
          v32 = *(_QWORD *)(v14 - 8);
        else
          v32 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        v33 = *(_QWORD *)(v32 + v31);
        v34 = sub_215E100(a1, a2, v212, v33, a5);
        v224 |= v33 != v34;
        v36 = (unsigned int)v243;
        if ( (unsigned int)v243 >= HIDWORD(v243) )
        {
          v203 = v34;
          sub_16CD150((__int64)&v242, &v244, 0, 8, v34, v35);
          v36 = (unsigned int)v243;
          v34 = v203;
        }
        v31 += 24;
        v242[v36] = v34;
        LODWORD(v243) = v243 + 1;
      }
      while ( v218 != v31 );
      v11 = v14;
      if ( v224 )
      {
        v14 = sub_1599EF0(*(__int64 ***)v14);
        if ( *(_BYTE *)(v11 + 16) == 8 )
        {
          v123 = 0;
          v124 = (__int64 *)a5;
          v229 = a2;
          v207 = v11;
          do
          {
            v125 = sub_1643350(*v229);
            v126 = sub_159C470(v125, v123, 0);
            v239 = 257;
            v127 = v242[v123];
            if ( *(_BYTE *)(v14 + 16) > 0x10u || *(_BYTE *)(v127 + 16) > 0x10u || *(_BYTE *)(v126 + 16) > 0x10u )
            {
              v235 = v126;
              v241 = 257;
              v128 = sub_1648A60(56, 3u);
              v129 = v128;
              if ( v128 )
                sub_15FA480((__int64)v128, (__int64 *)v14, v127, v235, (__int64)v240, 0);
              v130 = v124[1];
              if ( v130 )
              {
                v131 = (unsigned __int64 *)v124[2];
                sub_157E9D0(v130 + 40, (__int64)v129);
                v132 = v129[3];
                v133 = *v131;
                v129[4] = v131;
                v133 &= 0xFFFFFFFFFFFFFFF8LL;
                v129[3] = v133 | v132 & 7;
                *(_QWORD *)(v133 + 8) = v129 + 3;
                *v131 = *v131 & 7 | (unsigned __int64)(v129 + 3);
              }
              sub_164B780((__int64)v129, v238);
              v134 = *v124;
              if ( *v124 )
              {
                v237 = *v124;
                sub_1623A60((__int64)&v237, v134, 2);
                v135 = v129[6];
                v136 = (__int64)(v129 + 6);
                if ( v135 )
                {
                  sub_161E7C0((__int64)(v129 + 6), v135);
                  v136 = (__int64)(v129 + 6);
                }
                v137 = (unsigned __int8 *)v237;
                v129[6] = v237;
                if ( v137 )
                  sub_1623210((__int64)&v237, v137, v136);
              }
              v14 = (__int64)v129;
            }
            else
            {
              v14 = sub_15A3890((__int64 *)v14, v242[v123], v126, 0);
            }
            ++v123;
          }
          while ( v209 > (unsigned int)v123 );
          v10 = a1;
          v11 = v207;
        }
        else
        {
          v219 = v11;
          v40 = 0;
          v41 = v14;
          v236 = 0;
          do
          {
            v239 = 257;
            v42 = v242[v40];
            if ( *(_BYTE *)(v41 + 16) > 0x10u || *(_BYTE *)(v42 + 16) > 0x10u )
            {
              v241 = 257;
              v43 = sub_1648A60(88, 2u);
              v44 = v43;
              if ( v43 )
              {
                v45 = (__int64)v43;
                sub_15F1EA0((__int64)v43, *(_QWORD *)v41, 63, (__int64)(v43 - 6), 2, 0);
                v44[7] = v44 + 9;
                v44[8] = 0x400000000LL;
                sub_15FAD90((__int64)v44, v41, v42, &v236, 1, (__int64)v240);
              }
              else
              {
                v45 = 0;
              }
              v46 = *(_QWORD *)(a5 + 8);
              if ( v46 )
              {
                v47 = *(unsigned __int64 **)(a5 + 16);
                sub_157E9D0(v46 + 40, (__int64)v44);
                v48 = v44[3];
                v49 = *v47;
                v44[4] = v47;
                v49 &= 0xFFFFFFFFFFFFFFF8LL;
                v44[3] = v49 | v48 & 7;
                *(_QWORD *)(v49 + 8) = v44 + 3;
                *v47 = *v47 & 7 | (unsigned __int64)(v44 + 3);
              }
              sub_164B780(v45, v238);
              v50 = *(_QWORD *)a5;
              if ( *(_QWORD *)a5 )
              {
                v237 = *(_QWORD *)a5;
                sub_1623A60((__int64)&v237, v50, 2);
                v51 = v44[6];
                if ( v51 )
                  sub_161E7C0((__int64)(v44 + 6), v51);
                v52 = (unsigned __int8 *)v237;
                v44[6] = v237;
                if ( v52 )
                  sub_1623210((__int64)&v237, v52, (__int64)(v44 + 6));
              }
              v41 = (__int64)v44;
            }
            else
            {
              v41 = sub_15A3A20((__int64 *)v41, (__int64 *)v242[v40], &v236, 1, 0);
            }
            v40 = (unsigned int)(v236 + 1);
            v236 = v40;
          }
          while ( v209 > (unsigned int)v40 );
          v14 = v41;
          v10 = a1;
          v11 = v219;
        }
      }
      goto LABEL_60;
    }
    goto LABEL_28;
  }
  v29 = *(unsigned int *)(a1 + 184);
  if ( !(_DWORD)v29 )
    goto LABEL_28;
  v53 = *(_QWORD *)(a1 + 168);
  v54 = (v29 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v55 = v53 + 48LL * v54;
  v56 = *(_QWORD *)(v55 + 24);
  if ( v56 != v11 )
  {
    v145 = 1;
    while ( v56 != -8 )
    {
      v146 = v145 + 1;
      v54 = (v29 - 1) & (v145 + v54);
      v55 = v53 + 48LL * v54;
      v56 = *(_QWORD *)(v55 + 24);
      if ( v56 == v11 )
        goto LABEL_63;
      v145 = v146;
    }
    goto LABEL_28;
  }
LABEL_63:
  if ( v55 == 48 * v29 + v53 )
  {
LABEL_28:
    v14 = v11;
    goto LABEL_5;
  }
  v14 = *(_QWORD *)(v55 + 40);
  v241 = 257;
  v57 = (__int64 **)sub_1646BA0(*(__int64 **)(v14 + 24), 0);
  if ( v57 != *(__int64 ***)v14 )
  {
    if ( *(_BYTE *)(v14 + 16) > 0x10u )
    {
      LOWORD(v244) = 257;
      v14 = sub_15FDBD0(48, v14, (__int64)v57, (__int64)&v242, 0);
      v138 = *(_QWORD *)(a5 + 8);
      if ( v138 )
      {
        v139 = *(__int64 **)(a5 + 16);
        sub_157E9D0(v138 + 40, v14);
        v140 = *(_QWORD *)(v14 + 24);
        v141 = *v139;
        *(_QWORD *)(v14 + 32) = v139;
        v141 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v14 + 24) = v141 | v140 & 7;
        *(_QWORD *)(v141 + 8) = v14 + 24;
        *v139 = *v139 & 7 | (v14 + 24);
      }
      sub_164B780(v14, v240);
      v142 = *(_QWORD *)a5;
      if ( *(_QWORD *)a5 )
      {
        v238[0] = *(_QWORD *)a5;
        sub_1623A60((__int64)v238, v142, 2);
        v143 = *(_QWORD *)(v14 + 48);
        if ( v143 )
          sub_161E7C0(v14 + 48, v143);
        v144 = (unsigned __int8 *)v238[0];
        *(_QWORD *)(v14 + 48) = v238[0];
        if ( v144 )
          sub_1623210((__int64)v238, v144, v14 + 48);
      }
    }
    else
    {
      v14 = sub_15A46C0(48, (__int64 ***)v14, v57, 0);
    }
  }
LABEL_5:
  v243 = 2;
  v244 = 0;
  v245 = v11;
  if ( v11 != -16 && v11 != -8 )
    sub_164C220((__int64)&v243);
  v15 = *(_DWORD *)(v10 + 264);
  v16 = v10 + 240;
  v246 = v10 + 240;
  v242 = (__int64 *)&unk_4A01B30;
  if ( !v15 )
  {
    ++*(_QWORD *)(v10 + 240);
    goto LABEL_10;
  }
  v18 = v245;
  v37 = *(_QWORD *)(v10 + 248);
  v38 = (v15 - 1) & (((unsigned int)v245 >> 9) ^ ((unsigned int)v245 >> 4));
  v19 = (_QWORD *)(v37 + 48LL * v38);
  v39 = v19[3];
  if ( v39 != v245 )
  {
    v114 = 1;
    v115 = 0;
    while ( v39 != -8 )
    {
      if ( !v115 && v39 == -16 )
        v115 = v19;
      v38 = (v15 - 1) & (v114 + v38);
      v19 = (_QWORD *)(v37 + 48LL * v38);
      v39 = v19[3];
      if ( v245 == v39 )
        goto LABEL_38;
      ++v114;
    }
    v116 = *(_DWORD *)(v10 + 256);
    if ( v115 )
      v19 = v115;
    ++*(_QWORD *)(v10 + 240);
    v20 = v116 + 1;
    if ( 4 * v20 < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(v10 + 260) - v20 > v15 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(v10 + 256) = v20;
        if ( v19[3] == -8 )
        {
          v22 = v19 + 1;
          if ( v18 != -8 )
          {
LABEL_18:
            v19[3] = v18;
            if ( v18 != -8 && v18 != 0 && v18 != -16 )
              sub_1649AC0(v22, v243 & 0xFFFFFFFFFFFFFFF8LL);
            v18 = v245;
          }
        }
        else
        {
          --*(_DWORD *)(v10 + 260);
          v21 = v19[3];
          if ( v21 != v18 )
          {
            v22 = v19 + 1;
            if ( v21 != 0 && v21 != -8 && v21 != -16 )
            {
              sub_1649B30(v19 + 1);
              v18 = v245;
            }
            goto LABEL_18;
          }
        }
        v23 = v246;
        v19[5] = 0;
        v19[4] = v23;
        goto LABEL_38;
      }
      sub_215DD00(v16, v15);
      v117 = *(_DWORD *)(v10 + 264);
      if ( !v117 )
        goto LABEL_11;
      v18 = v245;
      v118 = v117 - 1;
      v119 = 1;
      v60 = 0;
      v120 = *(_QWORD *)(v10 + 248);
      v121 = v118 & (((unsigned int)v245 >> 9) ^ ((unsigned int)v245 >> 4));
      v19 = (_QWORD *)(v120 + 48LL * v121);
      v122 = v19[3];
      if ( v245 == v122 )
        goto LABEL_12;
      while ( v122 != -8 )
      {
        if ( v122 == -16 && !v60 )
          v60 = v19;
        v121 = v118 & (v119 + v121);
        v19 = (_QWORD *)(v120 + 48LL * v121);
        v122 = v19[3];
        if ( v245 == v122 )
          goto LABEL_12;
        ++v119;
      }
      goto LABEL_69;
    }
LABEL_10:
    sub_215DD00(v16, 2 * v15);
    v17 = *(_DWORD *)(v10 + 264);
    if ( !v17 )
    {
LABEL_11:
      v18 = v245;
      v19 = 0;
LABEL_12:
      v20 = *(_DWORD *)(v10 + 256) + 1;
      goto LABEL_13;
    }
    v18 = v245;
    v58 = v17 - 1;
    v59 = 1;
    v60 = 0;
    v61 = *(_QWORD *)(v10 + 248);
    v62 = (v17 - 1) & (((unsigned int)v245 >> 9) ^ ((unsigned int)v245 >> 4));
    v19 = (_QWORD *)(v61 + 48LL * v62);
    v63 = v19[3];
    if ( v63 == v245 )
      goto LABEL_12;
    while ( v63 != -8 )
    {
      if ( !v60 && v63 == -16 )
        v60 = v19;
      v62 = v58 & (v59 + v62);
      v19 = (_QWORD *)(v61 + 48LL * v62);
      v63 = v19[3];
      if ( v245 == v63 )
        goto LABEL_12;
      ++v59;
    }
LABEL_69:
    if ( v60 )
      v19 = v60;
    goto LABEL_12;
  }
LABEL_38:
  v242 = (__int64 *)&unk_49EE2B0;
  if ( v18 != 0 && v18 != -8 && v18 != -16 )
    sub_1649B30(&v243);
  v19[5] = v14;
  return v14;
}
