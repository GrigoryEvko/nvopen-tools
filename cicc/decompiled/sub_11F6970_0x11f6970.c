// Function: sub_11F6970
// Address: 0x11f6970
//
_BYTE *__fastcall sub_11F6970(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r15
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _BYTE *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rsi
  _BYTE *v24; // rcx
  _BYTE *v25; // rax
  unsigned __int64 v26; // rax
  _BYTE *v27; // r15
  int v28; // eax
  int v29; // eax
  unsigned int v30; // r12d
  unsigned int v31; // r14d
  unsigned int v32; // ebx
  int v33; // r11d
  __int64 v34; // r13
  __int64 *v35; // rsi
  unsigned int v36; // edi
  __int64 *v37; // rdx
  __int64 v38; // r10
  __int64 v39; // rax
  __int64 *v40; // rsi
  int v41; // r8d
  __int64 v42; // rdx
  unsigned int v43; // r9d
  unsigned int v44; // ecx
  unsigned int v45; // r10d
  __int64 *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 *v49; // rsi
  int v50; // r8d
  __int64 v51; // rdx
  unsigned int v52; // r9d
  unsigned int v53; // ecx
  unsigned int v54; // r10d
  __int64 *v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rax
  int v58; // r8d
  unsigned int v59; // ecx
  unsigned int v60; // r10d
  __int64 *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rax
  unsigned int v64; // ebx
  int v65; // edi
  int v66; // edi
  __int64 v67; // r9
  unsigned int v68; // ecx
  int v69; // edx
  __int64 v70; // r8
  bool v71; // r13
  __int64 v72; // rcx
  unsigned int v73; // edx
  _QWORD *v74; // rax
  __int64 v75; // r9
  bool *v76; // rax
  bool v77; // zf
  int v78; // edi
  int v79; // edi
  int v80; // edi
  __int64 v81; // r9
  __int64 *v82; // r10
  int v83; // r12d
  unsigned int v84; // ecx
  __int64 v85; // r8
  int v86; // eax
  int v87; // edi
  int v88; // eax
  int v89; // eax
  __int64 v90; // rsi
  unsigned int v91; // ecx
  int v92; // edx
  __int64 *v93; // rdi
  __int64 v94; // r8
  int v95; // r10d
  __int64 *v96; // r9
  int v97; // eax
  int v98; // edi
  int v99; // eax
  int v100; // eax
  int v101; // eax
  __int64 v102; // r8
  unsigned int v103; // ecx
  __int64 v104; // rdi
  char v105; // bl
  unsigned int v106; // esi
  __int64 v107; // rcx
  unsigned int v108; // edx
  __int64 *v109; // rax
  __int64 v110; // r10
  _BYTE *result; // rax
  int v112; // eax
  int v113; // eax
  __int64 v114; // r8
  __int64 *v115; // r10
  int v116; // ebx
  unsigned int v117; // ecx
  __int64 v118; // rdi
  int v119; // eax
  int v120; // eax
  __int64 v121; // r8
  __int64 *v122; // r10
  int v123; // ebx
  unsigned int v124; // ecx
  __int64 v125; // rdi
  int v126; // eax
  int v127; // eax
  __int64 v128; // r8
  int v129; // ebx
  unsigned int v130; // ecx
  __int64 v131; // rdi
  __int64 *v132; // r11
  int v133; // eax
  int v134; // eax
  __int64 v135; // r8
  unsigned int v136; // ecx
  __int64 v137; // rdi
  int v138; // ebx
  __int64 *v139; // r11
  int v140; // eax
  int v141; // eax
  __int64 v142; // r8
  unsigned int v143; // ecx
  __int64 v144; // rdi
  int v145; // ebx
  __int64 *v146; // r11
  unsigned __int64 v147; // rax
  int v148; // eax
  int v149; // eax
  __int64 v150; // rsi
  unsigned int v151; // ecx
  int v152; // edx
  __int64 *v153; // r8
  __int64 v154; // rdi
  int v155; // r10d
  __int64 *v156; // r9
  unsigned int v157; // ebx
  int v158; // eax
  int v159; // r11d
  int v160; // eax
  int v161; // r10d
  int v162; // eax
  int v163; // eax
  int v164; // ecx
  __int64 v165; // rsi
  int v166; // r9d
  unsigned int v167; // r12d
  __int64 *v168; // rdi
  __int64 v169; // rax
  int v170; // eax
  int v171; // ecx
  __int64 v172; // rsi
  __int64 *v173; // r8
  unsigned int v174; // ebx
  int v175; // r9d
  __int64 v176; // rax
  int v177; // r11d
  int v178; // r10d
  unsigned int v179; // r11d
  __int64 *v180; // r12
  int v181; // ebx
  int v182; // [rsp+24h] [rbp-A7Ch]
  __int64 v183; // [rsp+28h] [rbp-A78h]
  __int64 v184; // [rsp+30h] [rbp-A70h]
  int v185; // [rsp+38h] [rbp-A68h]
  __int64 v186; // [rsp+38h] [rbp-A68h]
  __int64 v187; // [rsp+38h] [rbp-A68h]
  __int64 v188; // [rsp+38h] [rbp-A68h]
  __int64 v189; // [rsp+38h] [rbp-A68h]
  __int64 v190; // [rsp+38h] [rbp-A68h]
  __int64 v191; // [rsp+38h] [rbp-A68h]
  __int64 v192; // [rsp+40h] [rbp-A60h]
  unsigned int v193; // [rsp+40h] [rbp-A60h]
  unsigned int v194; // [rsp+40h] [rbp-A60h]
  unsigned int v195; // [rsp+40h] [rbp-A60h]
  unsigned int v196; // [rsp+40h] [rbp-A60h]
  unsigned int v197; // [rsp+40h] [rbp-A60h]
  unsigned int v198; // [rsp+40h] [rbp-A60h]
  unsigned int v199; // [rsp+40h] [rbp-A60h]
  __int64 v200; // [rsp+48h] [rbp-A58h]
  __int64 v201; // [rsp+48h] [rbp-A58h]
  unsigned int v202; // [rsp+48h] [rbp-A58h]
  unsigned int v203; // [rsp+48h] [rbp-A58h]
  unsigned int v204; // [rsp+48h] [rbp-A58h]
  unsigned int v205; // [rsp+48h] [rbp-A58h]
  unsigned int v206; // [rsp+48h] [rbp-A58h]
  unsigned int v207; // [rsp+48h] [rbp-A58h]
  _QWORD v208[54]; // [rsp+50h] [rbp-A50h] BYREF
  __int64 v209; // [rsp+200h] [rbp-8A0h] BYREF
  __int64 *v210; // [rsp+208h] [rbp-898h]
  int v211; // [rsp+210h] [rbp-890h]
  int v212; // [rsp+214h] [rbp-88Ch]
  int v213; // [rsp+218h] [rbp-888h]
  char v214; // [rsp+21Ch] [rbp-884h]
  __int64 v215; // [rsp+220h] [rbp-880h] BYREF
  __int64 *v216; // [rsp+260h] [rbp-840h]
  __int64 v217; // [rsp+268h] [rbp-838h]
  __int64 v218; // [rsp+270h] [rbp-830h] BYREF
  int v219; // [rsp+278h] [rbp-828h]
  __int64 v220; // [rsp+280h] [rbp-820h]
  int v221; // [rsp+288h] [rbp-818h]
  __int64 v222; // [rsp+290h] [rbp-810h]
  char v223[8]; // [rsp+3B0h] [rbp-6F0h] BYREF
  __int64 v224; // [rsp+3B8h] [rbp-6E8h]
  char v225; // [rsp+3CCh] [rbp-6D4h]
  char v226[64]; // [rsp+3D0h] [rbp-6D0h] BYREF
  _BYTE *v227; // [rsp+410h] [rbp-690h] BYREF
  __int64 v228; // [rsp+418h] [rbp-688h]
  _BYTE v229[320]; // [rsp+420h] [rbp-680h] BYREF
  char v230[8]; // [rsp+560h] [rbp-540h] BYREF
  __int64 v231; // [rsp+568h] [rbp-538h]
  char v232; // [rsp+57Ch] [rbp-524h]
  char v233[64]; // [rsp+580h] [rbp-520h] BYREF
  _BYTE *v234; // [rsp+5C0h] [rbp-4E0h] BYREF
  __int64 v235; // [rsp+5C8h] [rbp-4D8h]
  _BYTE v236[320]; // [rsp+5D0h] [rbp-4D0h] BYREF
  char v237[8]; // [rsp+710h] [rbp-390h] BYREF
  __int64 v238; // [rsp+718h] [rbp-388h]
  char v239; // [rsp+72Ch] [rbp-374h]
  _BYTE *v240; // [rsp+770h] [rbp-330h] BYREF
  unsigned int v241; // [rsp+778h] [rbp-328h]
  _BYTE v242[320]; // [rsp+780h] [rbp-320h] BYREF
  char v243[8]; // [rsp+8C0h] [rbp-1E0h] BYREF
  __int64 v244; // [rsp+8C8h] [rbp-1D8h]
  char v245; // [rsp+8DCh] [rbp-1C4h]
  char *v246; // [rsp+920h] [rbp-180h] BYREF
  int v247; // [rsp+928h] [rbp-178h]
  char v248; // [rsp+930h] [rbp-170h] BYREF

  v3 = *(_QWORD *)(a2 + 80);
  v211 = 8;
  v213 = 0;
  v214 = 1;
  v212 = 1;
  if ( v3 )
    v3 -= 24;
  memset(v208, 0, sizeof(v208));
  v208[12] = &v208[14];
  v208[1] = &v208[4];
  v210 = &v215;
  v216 = &v218;
  HIDWORD(v208[13]) = 8;
  v217 = 0x800000000LL;
  v215 = v3;
  v209 = 1;
  v4 = *(_QWORD *)(v3 + 48);
  LODWORD(v208[2]) = 8;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  BYTE4(v208[3]) = 1;
  if ( v5 == v3 + 48 )
    goto LABEL_223;
  if ( !v5 )
LABEL_180:
    BUG();
  v6 = v5 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
LABEL_223:
    v7 = 0;
    v8 = 0;
    v6 = 0;
  }
  else
  {
    v7 = sub_B46E30(v6);
    v8 = v6;
  }
  v220 = v6;
  v222 = v3;
  v218 = v8;
  v219 = v7;
  v221 = 0;
  LODWORD(v217) = 1;
  sub_FDEBC0((__int64)&v209);
  sub_FDEE20((__int64)v230, (__int64)v208);
  sub_FDEE20((__int64)v223, (__int64)&v209);
  sub_FDEE20((__int64)v237, (__int64)v223);
  sub_FDEE20((__int64)v243, (__int64)v230);
  if ( v227 != v229 )
    _libc_free(v227, v230);
  if ( !v225 )
    _libc_free(v224, v230);
  if ( v234 != v236 )
    _libc_free(v234, v230);
  if ( !v232 )
    _libc_free(v231, v230);
  if ( v216 != &v218 )
    _libc_free(v216, v230);
  if ( !v214 )
    _libc_free(v210, v230);
  if ( (_QWORD *)v208[12] != &v208[14] )
    _libc_free(v208[12], v230);
  if ( !BYTE4(v208[3]) )
    _libc_free(v208[1], v230);
  sub_C8CD80((__int64)v230, (__int64)v233, (__int64)v243, v9, v10, v11);
  v234 = v236;
  v235 = 0x800000000LL;
  if ( v247 )
    sub_11F6800((__int64)&v234, (__int64 *)&v246, v12, v13, v14, v15);
  sub_C8CD80((__int64)v223, (__int64)v226, (__int64)v237, v13, v14, v15);
  v20 = v229;
  v228 = 0x800000000LL;
  v21 = v241;
  v227 = v229;
  if ( v241 )
  {
    sub_11F6800((__int64)&v227, (__int64 *)&v240, v16, v17, v18, v19);
    v21 = (unsigned int)v228;
    v20 = v227;
  }
LABEL_27:
  v183 = a1 + 8;
  while ( 1 )
  {
    v22 = 40 * v21;
    if ( v21 == (unsigned int)v235 )
      break;
LABEL_32:
    v184 = *(_QWORD *)&v20[v22 - 8];
    v26 = *(_QWORD *)(v184 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v26 == v184 + 48 )
      goto LABEL_126;
    if ( !v26 )
      goto LABEL_180;
    v27 = (_BYTE *)(v26 - 24);
    if ( (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30 > 0xA )
    {
LABEL_126:
      v27 = 0;
LABEL_127:
      v105 = qword_4F92128;
      if ( (_BYTE)qword_4F92128 && *v27 == 36 || (v105 = qword_4F92048) == 0 )
      {
        v106 = *(_DWORD *)(a1 + 32);
        if ( v106 )
          goto LABEL_131;
LABEL_182:
        ++*(_QWORD *)(a1 + 8);
        goto LABEL_183;
      }
      v147 = sub_AA4F10(v184);
      v106 = *(_DWORD *)(a1 + 32);
      v105 = v147 != 0;
      if ( !v106 )
        goto LABEL_182;
LABEL_131:
      v107 = *(_QWORD *)(a1 + 16);
      v108 = (v106 - 1) & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
      v109 = (__int64 *)(v107 + 16LL * v108);
      v110 = *v109;
      if ( *v109 == v184 )
      {
LABEL_132:
        *((_BYTE *)v109 + 8) = v105;
        goto LABEL_59;
      }
      v159 = 1;
      v153 = 0;
      while ( v110 != -4096 )
      {
        if ( !v153 && v110 == -8192 )
          v153 = v109;
        v108 = (v106 - 1) & (v159 + v108);
        v109 = (__int64 *)(v107 + 16LL * v108);
        v110 = *v109;
        if ( v184 == *v109 )
          goto LABEL_132;
        ++v159;
      }
      if ( !v153 )
        v153 = v109;
      v160 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v152 = v160 + 1;
      if ( 4 * (v160 + 1) >= 3 * v106 )
      {
LABEL_183:
        sub_11F63E0(v183, 2 * v106);
        v148 = *(_DWORD *)(a1 + 32);
        if ( !v148 )
          goto LABEL_315;
        v149 = v148 - 1;
        v150 = *(_QWORD *)(a1 + 16);
        v151 = v149 & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
        v152 = *(_DWORD *)(a1 + 24) + 1;
        v153 = (__int64 *)(v150 + 16LL * v151);
        v154 = *v153;
        if ( v184 != *v153 )
        {
          v155 = 1;
          v156 = 0;
          while ( v154 != -4096 )
          {
            if ( !v156 && v154 == -8192 )
              v156 = v153;
            v151 = v149 & (v155 + v151);
            v153 = (__int64 *)(v150 + 16LL * v151);
            v154 = *v153;
            if ( v184 == *v153 )
              goto LABEL_211;
            ++v155;
          }
          if ( v156 )
            v153 = v156;
        }
      }
      else if ( v106 - *(_DWORD *)(a1 + 28) - v152 <= v106 >> 3 )
      {
        sub_11F63E0(v183, v106);
        v163 = *(_DWORD *)(a1 + 32);
        if ( !v163 )
          goto LABEL_315;
        v164 = v163 - 1;
        v165 = *(_QWORD *)(a1 + 16);
        v166 = 1;
        v167 = (v163 - 1) & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
        v152 = *(_DWORD *)(a1 + 24) + 1;
        v168 = 0;
        v153 = (__int64 *)(v165 + 16LL * v167);
        v169 = *v153;
        if ( v184 != *v153 )
        {
          while ( v169 != -4096 )
          {
            if ( !v168 && v169 == -8192 )
              v168 = v153;
            v167 = v164 & (v166 + v167);
            v153 = (__int64 *)(v165 + 16LL * v167);
            v169 = *v153;
            if ( v184 == *v153 )
              goto LABEL_211;
            ++v166;
          }
          if ( v168 )
            v153 = v168;
        }
      }
LABEL_211:
      *(_DWORD *)(a1 + 24) = v152;
      if ( *v153 != -4096 )
        --*(_DWORD *)(a1 + 28);
      *((_BYTE *)v153 + 8) = 0;
      *v153 = v184;
      *((_BYTE *)v153 + 8) = v105;
      goto LABEL_59;
    }
    if ( !(unsigned int)sub_B46E30((__int64)v27) )
      goto LABEL_127;
    v28 = sub_B46E30((__int64)v27);
    v209 = a1;
    v182 = v28;
    v29 = v28 >> 2;
    if ( v29 <= 0 )
    {
      v158 = v182;
      v157 = 0;
LABEL_191:
      if ( v158 != 2 )
      {
        if ( v158 != 3 )
        {
          if ( v158 != 1 )
            goto LABEL_194;
LABEL_202:
          if ( (unsigned __int8)sub_11F65C0(&v209, (__int64)v27, v157) )
            goto LABEL_203;
LABEL_194:
          v30 = *(_DWORD *)(a1 + 32);
          v71 = 1;
          goto LABEL_55;
        }
        if ( (unsigned __int8)sub_11F65C0(&v209, (__int64)v27, v157) )
          goto LABEL_203;
        ++v157;
      }
      if ( !(unsigned __int8)sub_11F65C0(&v209, (__int64)v27, v157) )
      {
        ++v157;
        goto LABEL_202;
      }
LABEL_203:
      v30 = *(_DWORD *)(a1 + 32);
      v71 = v182 == v157;
LABEL_55:
      if ( !v30 )
        goto LABEL_91;
      goto LABEL_56;
    }
    v30 = *(_DWORD *)(a1 + 32);
    v192 = a1;
    v185 = 4 * v29;
    v31 = 0;
    v32 = v30 - 1;
    while ( 1 )
    {
      v63 = sub_B46EC0((__int64)v27, v31);
      if ( !v30 )
      {
        v64 = v31;
        a1 = v192;
        ++*(_QWORD *)(v192 + 8);
        goto LABEL_49;
      }
      v33 = 1;
      v34 = *(_QWORD *)(v192 + 16);
      v35 = 0;
      v36 = v32 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
      v37 = (__int64 *)(v34 + 16LL * v36);
      v38 = *v37;
      if ( v63 != *v37 )
      {
        while ( v38 != -4096 )
        {
          if ( !v35 && v38 == -8192 )
            v35 = v37;
          v36 = v32 & (v33 + v36);
          v37 = (__int64 *)(v34 + 16LL * v36);
          v38 = *v37;
          if ( v63 == *v37 )
            goto LABEL_39;
          ++v33;
        }
        v64 = v31;
        a1 = v192;
        if ( !v35 )
          v35 = v37;
        v78 = *(_DWORD *)(v192 + 24);
        ++*(_QWORD *)(v192 + 8);
        v69 = v78 + 1;
        if ( 4 * (v78 + 1) >= 3 * v30 )
        {
LABEL_49:
          v200 = v63;
          sub_11F63E0(v183, 2 * v30);
          v65 = *(_DWORD *)(a1 + 32);
          if ( !v65 )
            goto LABEL_315;
          v63 = v200;
          v66 = v65 - 1;
          v67 = *(_QWORD *)(a1 + 16);
          v68 = v66 & (((unsigned int)v200 >> 9) ^ ((unsigned int)v200 >> 4));
          v69 = *(_DWORD *)(a1 + 24) + 1;
          v35 = (__int64 *)(v67 + 16LL * v68);
          v70 = *v35;
          if ( v200 != *v35 )
          {
            v177 = 1;
            v82 = 0;
            while ( v70 != -4096 )
            {
              if ( v70 != -8192 || v82 )
                v35 = v82;
              v178 = v177 + 1;
              v179 = v68 + v177;
              v68 = v66 & v179;
              v180 = (__int64 *)(v67 + 16LL * (v66 & v179));
              v70 = *v180;
              if ( v200 == *v180 )
              {
                v35 = (__int64 *)(v67 + 16LL * (v66 & v179));
                goto LABEL_51;
              }
              v177 = v178;
              v82 = v35;
              v35 = v180;
            }
            goto LABEL_74;
          }
        }
        else if ( v30 - *(_DWORD *)(v192 + 28) - v69 <= v30 >> 3 )
        {
          v193 = ((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4);
          v201 = v63;
          sub_11F63E0(v183, v30);
          v79 = *(_DWORD *)(a1 + 32);
          if ( !v79 )
            goto LABEL_315;
          v80 = v79 - 1;
          v81 = *(_QWORD *)(a1 + 16);
          v82 = 0;
          v83 = 1;
          v84 = v80 & v193;
          v69 = *(_DWORD *)(a1 + 24) + 1;
          v63 = v201;
          v35 = (__int64 *)(v81 + 16LL * (v80 & v193));
          v85 = *v35;
          if ( v201 != *v35 )
          {
            while ( v85 != -4096 )
            {
              if ( !v82 && v85 == -8192 )
                v82 = v35;
              v84 = v80 & (v83 + v84);
              v35 = (__int64 *)(v81 + 16LL * v84);
              v85 = *v35;
              if ( v201 == *v35 )
                goto LABEL_51;
              ++v83;
            }
LABEL_74:
            if ( v82 )
              v35 = v82;
          }
        }
LABEL_51:
        *(_DWORD *)(a1 + 24) = v69;
        if ( *v35 != -4096 )
          --*(_DWORD *)(a1 + 28);
        *v35 = v63;
        *((_BYTE *)v35 + 8) = 0;
        v30 = *(_DWORD *)(a1 + 32);
LABEL_54:
        v71 = v182 == v64;
        goto LABEL_55;
      }
LABEL_39:
      if ( !*((_BYTE *)v37 + 8) )
      {
        v64 = v31;
        a1 = v192;
        goto LABEL_54;
      }
      v39 = sub_B46EC0((__int64)v27, v31 + 1);
      v40 = 0;
      v41 = 1;
      v42 = v39;
      v43 = v31 + 1;
      v44 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
      v45 = v44 & v32;
      v46 = (__int64 *)(v34 + 16LL * (v44 & v32));
      v47 = *v46;
      if ( *v46 != v42 )
      {
        while ( v47 != -4096 )
        {
          if ( !v40 && v47 == -8192 )
            v40 = v46;
          v45 = v32 & (v41 + v45);
          v46 = (__int64 *)(v34 + 16LL * v45);
          v47 = *v46;
          if ( *v46 == v42 )
            goto LABEL_41;
          ++v41;
        }
        a1 = v192;
        if ( !v40 )
          v40 = v46;
        v86 = *(_DWORD *)(v192 + 24);
        ++*(_QWORD *)(v192 + 8);
        v87 = v86 + 1;
        if ( 4 * (v86 + 1) >= 3 * v30 )
        {
          v187 = v42;
          v195 = v44;
          v203 = v43;
          sub_11F63E0(v183, 2 * v30);
          v112 = *(_DWORD *)(a1 + 32);
          if ( !v112 )
            goto LABEL_315;
          v113 = v112 - 1;
          v114 = *(_QWORD *)(a1 + 16);
          v115 = 0;
          v42 = v187;
          v43 = v203;
          v116 = 1;
          v117 = v113 & v195;
          v40 = (__int64 *)(v114 + 16LL * (v113 & v195));
          v118 = *v40;
          if ( *v40 != v187 )
          {
            while ( v118 != -4096 )
            {
              if ( v115 || v118 != -8192 )
                v40 = v115;
              v117 = v113 & (v116 + v117);
              v132 = (__int64 *)(v114 + 16LL * v117);
              v118 = *v132;
              if ( *v132 == v187 )
              {
LABEL_167:
                v40 = v132;
                v87 = *(_DWORD *)(a1 + 24) + 1;
                goto LABEL_87;
              }
              ++v116;
              v115 = v40;
              v40 = (__int64 *)(v114 + 16LL * v117);
            }
            goto LABEL_156;
          }
        }
        else
        {
          if ( v30 - *(_DWORD *)(v192 + 28) - v87 > v30 >> 3 )
            goto LABEL_87;
          v190 = v42;
          v198 = v44;
          v206 = v43;
          sub_11F63E0(v183, v30);
          v133 = *(_DWORD *)(a1 + 32);
          if ( !v133 )
            goto LABEL_315;
          v134 = v133 - 1;
          v135 = *(_QWORD *)(a1 + 16);
          v42 = v190;
          v43 = v206;
          v136 = v134 & v198;
          v40 = (__int64 *)(v135 + 16LL * (v134 & v198));
          v137 = *v40;
          if ( *v40 != v190 )
          {
            v138 = 1;
            v139 = 0;
            while ( v137 != -4096 )
            {
              if ( v139 || v137 != -8192 )
                v40 = v139;
              v136 = v134 & (v138 + v136);
              v115 = (__int64 *)(v135 + 16LL * v136);
              v137 = *v115;
              if ( *v115 == v190 )
              {
LABEL_251:
                v87 = *(_DWORD *)(a1 + 24) + 1;
                goto LABEL_157;
              }
              ++v138;
              v139 = v40;
              v40 = (__int64 *)(v135 + 16LL * v136);
            }
            goto LABEL_172;
          }
        }
LABEL_125:
        v87 = *(_DWORD *)(a1 + 24) + 1;
        goto LABEL_87;
      }
LABEL_41:
      if ( !*((_BYTE *)v46 + 8) )
        goto LABEL_195;
      v48 = sub_B46EC0((__int64)v27, v31 + 2);
      v49 = 0;
      v50 = 1;
      v51 = v48;
      v52 = v31 + 2;
      v53 = ((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4);
      v54 = v53 & v32;
      v55 = (__int64 *)(v34 + 16LL * (v53 & v32));
      v56 = *v55;
      if ( *v55 != v51 )
      {
        while ( v56 != -4096 )
        {
          if ( !v49 && v56 == -8192 )
            v49 = v55;
          v54 = v32 & (v50 + v54);
          v55 = (__int64 *)(v34 + 16LL * v54);
          v56 = *v55;
          if ( *v55 == v51 )
            goto LABEL_43;
          ++v50;
        }
        a1 = v192;
        if ( !v49 )
          v49 = v55;
        v97 = *(_DWORD *)(v192 + 24);
        ++*(_QWORD *)(v192 + 8);
        v98 = v97 + 1;
        if ( 4 * (v97 + 1) < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(v192 + 28) - v98 > v30 >> 3 )
            goto LABEL_109;
          v191 = v51;
          v199 = v52;
          v207 = v53;
          sub_11F63E0(v183, v30);
          v140 = *(_DWORD *)(a1 + 32);
          if ( v140 )
          {
            v141 = v140 - 1;
            v142 = *(_QWORD *)(a1 + 16);
            v51 = v191;
            v52 = v199;
            v143 = v141 & v207;
            v49 = (__int64 *)(v142 + 16LL * (v141 & v207));
            v144 = *v49;
            if ( *v49 != v191 )
            {
              v145 = 1;
              v146 = 0;
              while ( v144 != -4096 )
              {
                if ( v146 || v144 != -8192 )
                  v49 = v146;
                v143 = v141 & (v145 + v143);
                v122 = (__int64 *)(v142 + 16LL * v143);
                v144 = *v122;
                if ( *v122 == v191 )
                {
                  v98 = *(_DWORD *)(a1 + 24) + 1;
                  goto LABEL_245;
                }
                ++v145;
                v146 = v49;
                v49 = (__int64 *)(v142 + 16LL * v143);
              }
              v98 = *(_DWORD *)(a1 + 24) + 1;
              if ( v146 )
                v49 = v146;
              goto LABEL_109;
            }
LABEL_160:
            v98 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_109;
          }
LABEL_315:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
        v188 = v51;
        v196 = v52;
        v204 = v53;
        sub_11F63E0(v183, 2 * v30);
        v119 = *(_DWORD *)(a1 + 32);
        if ( !v119 )
          goto LABEL_315;
        v120 = v119 - 1;
        v121 = *(_QWORD *)(a1 + 16);
        v122 = 0;
        v51 = v188;
        v52 = v196;
        v123 = 1;
        v124 = v120 & v204;
        v49 = (__int64 *)(v121 + 16LL * (v120 & v204));
        v125 = *v49;
        if ( *v49 == v188 )
          goto LABEL_160;
        while ( v125 != -4096 )
        {
          if ( v122 || v125 != -8192 )
            v49 = v122;
          v124 = v120 & (v123 + v124);
          v125 = *(_QWORD *)(v121 + 16LL * v124);
          if ( v125 == v188 )
          {
            v49 = (__int64 *)(v121 + 16LL * v124);
            v98 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_109;
          }
          ++v123;
          v122 = v49;
          v49 = (__int64 *)(v121 + 16LL * v124);
        }
        v98 = *(_DWORD *)(a1 + 24) + 1;
        if ( v122 )
LABEL_245:
          v49 = v122;
LABEL_109:
        *(_DWORD *)(a1 + 24) = v98;
        if ( *v49 != -4096 )
          --*(_DWORD *)(a1 + 28);
        *v49 = v51;
        *((_BYTE *)v49 + 8) = 0;
        v30 = *(_DWORD *)(a1 + 32);
LABEL_112:
        v71 = v182 == v52;
        goto LABEL_55;
      }
LABEL_43:
      if ( !*((_BYTE *)v55 + 8) )
      {
        a1 = v192;
        goto LABEL_112;
      }
      v57 = sub_B46EC0((__int64)v27, v31 + 3);
      v40 = 0;
      v58 = 1;
      v42 = v57;
      v43 = v31 + 3;
      v59 = ((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4);
      v60 = v59 & v32;
      v61 = (__int64 *)(v34 + 16LL * (v59 & v32));
      v62 = *v61;
      if ( *v61 != v42 )
        break;
LABEL_45:
      if ( !*((_BYTE *)v61 + 8) )
      {
LABEL_195:
        a1 = v192;
        goto LABEL_90;
      }
      v31 += 4;
      if ( v185 == v31 )
      {
        v157 = v31;
        a1 = v192;
        v158 = v182 - v157;
        goto LABEL_191;
      }
    }
    while ( v62 != -4096 )
    {
      if ( !v40 && v62 == -8192 )
        v40 = v61;
      v60 = v32 & (v58 + v60);
      v61 = (__int64 *)(v34 + 16LL * v60);
      v62 = *v61;
      if ( *v61 == v42 )
        goto LABEL_45;
      ++v58;
    }
    a1 = v192;
    if ( !v40 )
      v40 = v61;
    v99 = *(_DWORD *)(v192 + 24);
    ++*(_QWORD *)(v192 + 8);
    v87 = v99 + 1;
    if ( 4 * (v99 + 1) >= 3 * v30 )
    {
      v189 = v42;
      v197 = v59;
      v205 = v43;
      sub_11F63E0(v183, 2 * v30);
      v126 = *(_DWORD *)(a1 + 32);
      if ( !v126 )
        goto LABEL_315;
      v127 = v126 - 1;
      v128 = *(_QWORD *)(a1 + 16);
      v115 = 0;
      v42 = v189;
      v43 = v205;
      v129 = 1;
      v130 = v127 & v197;
      v40 = (__int64 *)(v128 + 16LL * (v127 & v197));
      v131 = *v40;
      if ( *v40 != v189 )
      {
        while ( v131 != -4096 )
        {
          if ( v115 || v131 != -8192 )
            v40 = v115;
          v130 = v127 & (v129 + v130);
          v132 = (__int64 *)(v128 + 16LL * v130);
          v131 = *v132;
          if ( *v132 == v189 )
            goto LABEL_167;
          ++v129;
          v115 = v40;
          v40 = (__int64 *)(v128 + 16LL * v130);
        }
LABEL_156:
        v87 = *(_DWORD *)(a1 + 24) + 1;
        if ( v115 )
LABEL_157:
          v40 = v115;
        goto LABEL_87;
      }
      goto LABEL_125;
    }
    if ( v30 - *(_DWORD *)(v192 + 28) - v87 <= v30 >> 3 )
    {
      v186 = v42;
      v194 = v59;
      v202 = v43;
      sub_11F63E0(v183, v30);
      v100 = *(_DWORD *)(a1 + 32);
      if ( !v100 )
        goto LABEL_315;
      v101 = v100 - 1;
      v102 = *(_QWORD *)(a1 + 16);
      v42 = v186;
      v43 = v202;
      v103 = v101 & v194;
      v40 = (__int64 *)(v102 + 16LL * (v101 & v194));
      v104 = *v40;
      if ( *v40 == v186 )
        goto LABEL_125;
      v181 = 1;
      v139 = 0;
      while ( v104 != -4096 )
      {
        if ( v139 || v104 != -8192 )
          v40 = v139;
        v103 = v101 & (v181 + v103);
        v115 = (__int64 *)(v102 + 16LL * v103);
        v104 = *v115;
        if ( *v115 == v186 )
          goto LABEL_251;
        ++v181;
        v139 = v40;
        v40 = (__int64 *)(v102 + 16LL * v103);
      }
LABEL_172:
      v87 = *(_DWORD *)(a1 + 24) + 1;
      if ( v139 )
        v40 = v139;
    }
LABEL_87:
    *(_DWORD *)(a1 + 24) = v87;
    if ( *v40 != -4096 )
      --*(_DWORD *)(a1 + 28);
    *v40 = v42;
    *((_BYTE *)v40 + 8) = 0;
    v30 = *(_DWORD *)(a1 + 32);
LABEL_90:
    v71 = v43 == v182;
    if ( !v30 )
    {
LABEL_91:
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_92;
    }
LABEL_56:
    v72 = *(_QWORD *)(a1 + 16);
    v73 = (v30 - 1) & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
    v74 = (_QWORD *)(v72 + 16LL * v73);
    v75 = *v74;
    if ( v184 == *v74 )
    {
LABEL_57:
      v76 = (bool *)(v74 + 1);
    }
    else
    {
      v161 = 1;
      v93 = 0;
      while ( v75 != -4096 )
      {
        if ( v75 == -8192 && !v93 )
          v93 = v74;
        v73 = (v30 - 1) & (v161 + v73);
        v74 = (_QWORD *)(v72 + 16LL * v73);
        v75 = *v74;
        if ( v184 == *v74 )
          goto LABEL_57;
        ++v161;
      }
      if ( !v93 )
        v93 = v74;
      v162 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v92 = v162 + 1;
      if ( 4 * (v162 + 1) >= 3 * v30 )
      {
LABEL_92:
        sub_11F63E0(v183, 2 * v30);
        v88 = *(_DWORD *)(a1 + 32);
        if ( !v88 )
          goto LABEL_315;
        v89 = v88 - 1;
        v90 = *(_QWORD *)(a1 + 16);
        v91 = v89 & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
        v92 = *(_DWORD *)(a1 + 24) + 1;
        v93 = (__int64 *)(v90 + 16LL * v91);
        v94 = *v93;
        if ( v184 != *v93 )
        {
          v95 = 1;
          v96 = 0;
          while ( v94 != -4096 )
          {
            if ( !v96 && v94 == -8192 )
              v96 = v93;
            v91 = v89 & (v95 + v91);
            v93 = (__int64 *)(v90 + 16LL * v91);
            v94 = *v93;
            if ( v184 == *v93 )
              goto LABEL_220;
            ++v95;
          }
          if ( v96 )
            v93 = v96;
        }
      }
      else if ( v30 - *(_DWORD *)(a1 + 28) - v92 <= v30 >> 3 )
      {
        sub_11F63E0(v183, v30);
        v170 = *(_DWORD *)(a1 + 32);
        if ( !v170 )
          goto LABEL_315;
        v171 = v170 - 1;
        v172 = *(_QWORD *)(a1 + 16);
        v173 = 0;
        v174 = (v170 - 1) & (((unsigned int)v184 >> 9) ^ ((unsigned int)v184 >> 4));
        v175 = 1;
        v92 = *(_DWORD *)(a1 + 24) + 1;
        v93 = (__int64 *)(v172 + 16LL * v174);
        v176 = *v93;
        if ( v184 != *v93 )
        {
          while ( v176 != -4096 )
          {
            if ( !v173 && v176 == -8192 )
              v173 = v93;
            v174 = v171 & (v175 + v174);
            v93 = (__int64 *)(v172 + 16LL * v174);
            v176 = *v93;
            if ( v184 == *v93 )
              goto LABEL_220;
            ++v175;
          }
          if ( v173 )
            v93 = v173;
        }
      }
LABEL_220:
      *(_DWORD *)(a1 + 24) = v92;
      if ( *v93 != -4096 )
        --*(_DWORD *)(a1 + 28);
      *((_BYTE *)v93 + 8) = 0;
      *v93 = v184;
      v76 = (bool *)(v93 + 1);
    }
    *v76 = v71;
LABEL_59:
    v77 = (_DWORD)v228 == 1;
    v21 = (unsigned int)(v228 - 1);
    LODWORD(v228) = v228 - 1;
    if ( !v77 )
    {
      sub_FDEBC0((__int64)v223);
      v21 = (unsigned int)v228;
      v20 = v227;
      goto LABEL_27;
    }
    v20 = v227;
  }
  v23 = &v20[v22];
  v24 = v234;
  if ( &v20[v22] != v20 )
  {
    v25 = v20;
    while ( *((_QWORD *)v25 + 4) == *((_QWORD *)v24 + 4)
         && *((_DWORD *)v25 + 6) == *((_DWORD *)v24 + 6)
         && *((_DWORD *)v25 + 2) == *((_DWORD *)v24 + 2) )
    {
      v25 += 40;
      v24 += 40;
      if ( v23 == v25 )
        goto LABEL_137;
    }
    goto LABEL_32;
  }
LABEL_137:
  if ( v20 != v229 )
    _libc_free(v20, v23);
  if ( !v225 )
    _libc_free(v224, v23);
  if ( v234 != v236 )
    _libc_free(v234, v23);
  if ( !v232 )
    _libc_free(v231, v23);
  if ( v246 != &v248 )
    _libc_free(v246, v23);
  if ( !v245 )
    _libc_free(v244, v23);
  result = v242;
  if ( v240 != v242 )
    result = (_BYTE *)_libc_free(v240, v23);
  if ( !v239 )
    return (_BYTE *)_libc_free(v238, v23);
  return result;
}
