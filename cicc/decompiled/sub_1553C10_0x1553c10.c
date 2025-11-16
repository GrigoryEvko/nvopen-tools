// Function: sub_1553C10
// Address: 0x1553c10
//
_BYTE *__fastcall sub_1553C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v10; // al
  _BYTE *result; // rax
  void *v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // ecx
  char v22; // r8
  const char *v23; // rsi
  size_t v24; // rdx
  void *v25; // rdx
  unsigned int v26; // ecx
  char v27; // r8
  const char *v28; // rsi
  size_t v29; // rdx
  unsigned __int8 *v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  const char *v34; // rax
  size_t v35; // rdx
  size_t v36; // r14
  void *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // r8
  const char *v45; // rsi
  size_t v46; // rdx
  __m128i *v47; // rdx
  __m128i v48; // xmm0
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // r8
  __m128i *v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // r8
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // r8
  char v66; // r8
  const char *v67; // rsi
  size_t v68; // rdx
  unsigned __int8 *v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rax
  unsigned __int8 *v72; // rcx
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // r8
  __m128i *v77; // rdx
  __m128i v78; // xmm0
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // rdx
  __int64 v82; // r8
  __m128i *v83; // rdx
  __m128i v84; // xmm0
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // rdx
  __int64 v88; // r8
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int64 v92; // r8
  char v93; // cl
  char v94; // cl
  __m128i *v95; // rdx
  __m128i v96; // xmm0
  bool v97; // zf
  __int64 v98; // rcx
  __int64 v99; // rdx
  __int64 v100; // r8
  __m128i *v101; // rdx
  __int64 v102; // rdx
  unsigned __int8 *v103; // rcx
  unsigned __int8 *v104; // r14
  __m128i *v105; // rdx
  __m128i v106; // xmm0
  __int64 v107; // rax
  __int64 v108; // rcx
  __int64 v109; // rdx
  __int64 v110; // r8
  void *v111; // rdx
  __int64 v112; // rax
  unsigned __int8 *v113; // rcx
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // rdx
  __int64 v117; // r8
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // rdx
  __int64 v121; // r8
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // rdx
  __int64 v125; // r8
  __int64 v126; // rdx
  __int64 v127; // rdx
  void *v128; // rdx
  __int64 v129; // rdx
  __int64 v130; // rcx
  __int64 v131; // rdx
  __int64 v132; // r8
  char v133; // cl
  __m128i *v134; // rdx
  __m128i si128; // xmm0
  __int64 v136; // rdx
  unsigned __int8 *v137; // rcx
  unsigned __int8 *v138; // r14
  void *v139; // rdx
  __int64 v140; // rax
  __int64 v141; // rcx
  __int64 v142; // rdx
  __int64 v143; // r8
  __int64 v144; // rcx
  __int64 v145; // rax
  unsigned __int8 *v146; // rcx
  void *v147; // rdx
  __int64 v148; // rdx
  __int64 v149; // rcx
  __int64 v150; // rdx
  __int64 v151; // r8
  __int64 v152; // rcx
  unsigned __int8 *v153; // rcx
  __int64 v154; // rdx
  __int64 v155; // rcx
  __int64 v156; // rdx
  __int64 v157; // r8
  char v158; // cl
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // rdx
  __int64 v162; // r8
  __int64 v163; // rdx
  __int64 v164; // rcx
  __int64 v165; // rdx
  __int64 v166; // r8
  unsigned int v167; // r14d
  __int64 v168; // rdi
  void *v169; // rdx
  const char *v170; // rax
  char v171; // cl
  char v172; // cl
  char v173; // cl
  __int64 v174; // rdx
  __int64 v175; // rcx
  __int64 v176; // rdx
  __int64 v177; // r8
  __int64 v178; // rdx
  __int64 v179; // rcx
  __int64 v180; // rdx
  __int64 v181; // r8
  unsigned __int8 *v182; // rcx
  char v183; // cl
  char v184; // cl
  __int64 v185; // rax
  unsigned __int8 *v186; // rcx
  unsigned int v187; // ecx
  int v188; // r14d
  __int64 v189; // rdi
  void *v190; // rdx
  __int64 v191; // rax
  char v192; // cl
  __int64 v193; // rax
  unsigned __int8 *v194; // rcx
  __int64 v195; // rax
  unsigned __int8 *v196; // rcx
  _QWORD *v197; // rdx
  __int64 v198; // rax
  __int64 v199; // rcx
  __int64 v200; // rdx
  __int64 v201; // r8
  __int64 v202; // rdx
  __int64 v203; // rcx
  __int64 v204; // rdx
  __int64 v205; // r8
  __int64 v206; // rdx
  __int64 v207; // rdi
  __int64 v208; // r14
  const char *v209; // rax
  size_t v210; // rdx
  __int64 v211; // rax
  unsigned int v212; // ebx
  __int64 v213; // rdx
  unsigned int v214; // [rsp+8h] [rbp-88h]
  __int64 v215; // [rsp+8h] [rbp-88h]
  __int64 v216; // [rsp+10h] [rbp-80h] BYREF
  __int64 v217; // [rsp+18h] [rbp-78h]
  __int64 v218; // [rsp+20h] [rbp-70h]
  __int64 v219; // [rsp+30h] [rbp-60h] BYREF
  char v220; // [rsp+38h] [rbp-58h]
  const char *v221; // [rsp+40h] [rbp-50h]
  __int64 v222; // [rsp+48h] [rbp-48h]
  __int64 v223; // [rsp+50h] [rbp-40h]
  __int64 v224; // [rsp+58h] [rbp-38h]

  v10 = *(_BYTE *)(a2 + 1);
  if ( v10 == 1 )
  {
    sub_1263B40(a1, "distinct ");
  }
  else if ( v10 == 2 )
  {
    sub_1263B40(a1, "<temporary!> ");
  }
  switch ( *(_BYTE *)a2 )
  {
    case 4:
      return (_BYTE *)sub_1553360(a1, a2, a3, a4, a5);
    case 5:
      return sub_154FAA0(a1, a2, a3, a4, a5);
    case 6:
      return sub_15499D0(a1, a2);
    case 7:
      sub_1263B40(a1, "!DIGlobalVariableExpression(");
      v223 = a4;
      v221 = ", ";
      v145 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v220 = 1;
      v146 = *(unsigned __int8 **)(a2 - 8 * v145);
      v222 = a3;
      v224 = a5;
      sub_154F950((__int64)&v219, "var", 3u, v146, 1);
      sub_154F950((__int64)&v219, "expr", 4u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
      return (_BYTE *)sub_1263B40(a1, ")");
    case 8:
      return sub_1553990(a1, a2, a3, a4, a5);
    case 9:
      return sub_154FC00(a1, a2, a3, a4, a5);
    case 0xA:
      v139 = *(void **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v139 <= 0xDu )
      {
        sub_16E7EE0(a1, "!DIEnumerator(", 14);
      }
      else
      {
        qmemcpy(v139, "!DIEnumerator(", 14);
        *(_QWORD *)(a1 + 24) += 14LL;
      }
      v219 = a1;
      v221 = ", ";
      v140 = *(unsigned int *)(a2 + 8);
      v220 = 1;
      v222 = 0;
      v141 = *(_QWORD *)(a2 - 8 * v140);
      v223 = 0;
      v224 = 0;
      if ( v141 )
      {
        v141 = sub_161E970(v141);
        v143 = v142;
      }
      else
      {
        v143 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v141, v143, 0);
      v144 = *(_QWORD *)(a2 + 24);
      if ( *(_DWORD *)(a2 + 4) )
      {
        sub_154B000((__int64)&v219, "value", 5u, v144, 0);
        BYTE1(v216) = 0;
        sub_154AA90((__int64)&v219, "isUnsigned", 0xAu, 1, &v216);
      }
      else
      {
        sub_154AEF0((__int64)&v219, "value", 5u, v144, 0);
      }
      goto LABEL_16;
    case 0xB:
      v147 = *(void **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v147 <= 0xCu )
      {
        sub_16E7EE0(a1, "!DIBasicType(", 13);
      }
      else
      {
        qmemcpy(v147, "!DIBasicType(", 13);
        *(_QWORD *)(a1 + 24) += 13LL;
      }
      v97 = *(_WORD *)(a2 + 2) == 36;
      v219 = a1;
      v220 = 1;
      v221 = ", ";
      v222 = 0;
      v223 = 0;
      v224 = 0;
      if ( !v97 )
        sub_1549850(&v219, a2);
      v148 = *(unsigned int *)(a2 + 8);
      v149 = *(_QWORD *)(a2 + 8 * (2 - v148));
      if ( v149 )
      {
        v149 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v148)));
        v151 = v150;
      }
      else
      {
        v151 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v149, v151, 1);
      sub_154B000((__int64)&v219, "size", 4u, *(_QWORD *)(a2 + 32), 1);
      sub_154ADE0((__int64)&v219, "align", 5u, *(_DWORD *)(a2 + 48), 1);
      sub_154B110(&v219, "encoding", 8u, *(unsigned int *)(a2 + 52), (__int64 (__fastcall *)(_QWORD))sub_14E6F20);
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
        goto LABEL_22;
      goto LABEL_17;
    case 0xC:
      return sub_154FDC0(a1, a2, a3, a4, a5);
    case 0xD:
      return sub_1550050(a1, a2, a3, a4, a5);
    case 0xE:
      return sub_1550360(a1, a2, a3, a4, a5);
    case 0xF:
      v197 = *(_QWORD **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v197 <= 7u )
      {
        sub_16E7EE0(a1, "!DIFile(", 8);
      }
      else
      {
        *v197 = 0x28656C6946494421LL;
        *(_QWORD *)(a1 + 24) += 8LL;
      }
      v219 = a1;
      v221 = ", ";
      v198 = *(unsigned int *)(a2 + 8);
      v220 = 1;
      v222 = 0;
      v199 = *(_QWORD *)(a2 - 8 * v198);
      v223 = 0;
      v224 = 0;
      if ( v199 )
      {
        v199 = sub_161E970(v199);
        v201 = v200;
      }
      else
      {
        v201 = 0;
      }
      sub_154AC80(&v219, "filename", 8u, v199, v201, 0);
      v202 = *(unsigned int *)(a2 + 8);
      v203 = *(_QWORD *)(a2 + 8 * (1 - v202));
      if ( v203 )
      {
        v203 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v202)));
        v205 = v204;
      }
      else
      {
        v205 = 0;
      }
      sub_154AC80(&v219, "directory", 9u, v203, v205, 0);
      if ( *(_BYTE *)(a2 + 40) )
      {
        sub_161E970(*(_QWORD *)(a2 + 32));
        if ( *(_BYTE *)(a2 + 40) )
        {
          v211 = sub_161E970(*(_QWORD *)(a2 + 32));
          v212 = *(_DWORD *)(a2 + 24);
          v217 = v211;
          v214 = v212;
          v218 = v213;
        }
        v207 = v219;
        if ( v220 )
          v220 = 0;
        else
          v207 = sub_1263B40(v219, v221);
        v208 = sub_1263B40(v207, "checksumkind: ");
        v209 = (const char *)sub_15B0D20(v214);
        sub_1549FF0(v208, v209, v210);
        sub_154AC80(&v219, "checksum", 8u, v217, v218, 0);
      }
      v44 = 0;
      v42 = 0;
      if ( *(_BYTE *)(a2 + 56) )
      {
        v216 = sub_161E970(*(_QWORD *)(a2 + 48));
        v42 = v216;
        v217 = v206;
        v44 = v206;
      }
      v45 = "source";
      v46 = 6;
      goto LABEL_35;
    case 0x10:
      sub_1263B40(a1, "!DICompileUnit(");
      v222 = a3;
      v152 = *(unsigned int *)(a2 + 24);
      v219 = a1;
      v220 = 1;
      v221 = ", ";
      v223 = a4;
      v224 = a5;
      sub_154B110(&v219, "language", 8u, v152, (__int64 (__fastcall *)(_QWORD))sub_14E77F0);
      v153 = (unsigned __int8 *)a2;
      if ( *(_BYTE *)a2 != 15 )
        v153 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      sub_154F950((__int64)&v219, "file", 4u, v153, 0);
      v154 = *(unsigned int *)(a2 + 8);
      v155 = *(_QWORD *)(a2 + 8 * (1 - v154));
      if ( v155 )
      {
        v155 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v154)));
        v157 = v156;
      }
      else
      {
        v157 = 0;
      }
      sub_154AC80(&v219, "producer", 8u, v155, v157, 1);
      v158 = *(_BYTE *)(a2 + 28);
      BYTE1(v216) = 0;
      sub_154AA90((__int64)&v219, "isOptimized", 0xBu, v158, &v216);
      v159 = *(unsigned int *)(a2 + 8);
      v160 = *(_QWORD *)(a2 + 8 * (2 - v159));
      if ( v160 )
      {
        v160 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v159)));
        v162 = v161;
      }
      else
      {
        v162 = 0;
      }
      sub_154AC80(&v219, "flags", 5u, v160, v162, 1);
      sub_154ADE0((__int64)&v219, "runtimeVersion", 0xEu, *(_DWORD *)(a2 + 32), 0);
      v163 = *(unsigned int *)(a2 + 8);
      v164 = *(_QWORD *)(a2 + 8 * (3 - v163));
      if ( v164 )
      {
        v164 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v163)));
        v166 = v165;
      }
      else
      {
        v166 = 0;
      }
      sub_154AC80(&v219, "splitDebugFilename", 0x12u, v164, v166, 1);
      v167 = *(_DWORD *)(a2 + 36);
      v168 = v219;
      if ( v220 )
        v220 = 0;
      else
        v168 = sub_1263B40(v219, v221);
      v169 = *(void **)(v168 + 24);
      if ( *(_QWORD *)(v168 + 16) - (_QWORD)v169 <= 0xBu )
      {
        v168 = sub_16E7EE0(v168, "emissionKind", 12);
      }
      else
      {
        qmemcpy(v169, "emissionKind", 12);
        *(_QWORD *)(v168 + 24) += 12LL;
      }
      v215 = sub_1263B40(v168, ": ");
      v170 = (const char *)sub_15B0FC0(v167);
      sub_1263B40(v215, v170);
      sub_154F950((__int64)&v219, "enums", 5u, *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154F950(
        (__int64)&v219,
        "retainedTypes",
        0xDu,
        *(unsigned __int8 **)(a2 + 8 * (5LL - *(unsigned int *)(a2 + 8))),
        1);
      sub_154F950((__int64)&v219, "globals", 7u, *(unsigned __int8 **)(a2 + 8 * (6LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154F950((__int64)&v219, "imports", 7u, *(unsigned __int8 **)(a2 + 8 * (7LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154F950((__int64)&v219, "macros", 6u, *(unsigned __int8 **)(a2 + 8 * (8LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154B000((__int64)&v219, "dwoId", 5u, *(_QWORD *)(a2 + 40), 1);
      v171 = *(_BYTE *)(a2 + 48);
      LOWORD(v216) = 257;
      sub_154AA90((__int64)&v219, "splitDebugInlining", 0x12u, v171, &v216);
      v172 = *(_BYTE *)(a2 + 49);
      LOWORD(v216) = 256;
      sub_154AA90((__int64)&v219, "debugInfoForProfiling", 0x15u, v172, &v216);
      v173 = *(_BYTE *)(a2 + 50);
      LOWORD(v216) = 256;
      sub_154AA90((__int64)&v219, "gnuPubnames", 0xBu, v173, &v216);
      return (_BYTE *)sub_1263B40(a1, ")");
    case 0x11:
      sub_1263B40(a1, "!DISubprogram(");
      v174 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v221 = ", ";
      v220 = 1;
      v175 = *(_QWORD *)(a2 + 8 * (2 - v174));
      v222 = a3;
      v223 = a4;
      v224 = a5;
      if ( v175 )
      {
        v175 = sub_161E970(v175);
        v177 = v176;
      }
      else
      {
        v177 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v175, v177, 1);
      v178 = *(unsigned int *)(a2 + 8);
      v179 = *(_QWORD *)(a2 + 8 * (3 - v178));
      if ( v179 )
      {
        v179 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v178)));
        v181 = v180;
      }
      else
      {
        v181 = 0;
      }
      sub_154AC80(&v219, "linkageName", 0xBu, v179, v181, 1);
      sub_154F950((__int64)&v219, "scope", 5u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 0);
      v182 = (unsigned __int8 *)a2;
      if ( *(_BYTE *)a2 != 15 )
        v182 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      sub_154F950((__int64)&v219, "file", 4u, v182, 1);
      sub_154ADE0((__int64)&v219, "line", 4u, *(_DWORD *)(a2 + 24), 1);
      sub_154F950((__int64)&v219, "type", 4u, *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))), 1);
      v183 = *(_BYTE *)(a2 + 40);
      BYTE1(v216) = 0;
      sub_154AA90((__int64)&v219, "isLocal", 7u, (v183 & 4) != 0, &v216);
      v184 = *(_BYTE *)(a2 + 40);
      BYTE1(v216) = 0;
      sub_154AA90((__int64)&v219, "isDefinition", 0xCu, (v184 & 8) != 0, &v216);
      sub_154ADE0((__int64)&v219, "scopeLine", 9u, *(_DWORD *)(a2 + 28), 1);
      v185 = *(unsigned int *)(a2 + 8);
      v186 = 0;
      if ( (unsigned int)v185 > 8 )
        v186 = *(unsigned __int8 **)(a2 + 8 * (8 - v185));
      sub_154F950((__int64)&v219, "containingType", 0xEu, v186, 1);
      sub_154B110(&v219, "virtuality", 0xAu, *(_BYTE *)(a2 + 40) & 3, (__int64 (__fastcall *)(_QWORD))sub_14E76C0);
      v187 = *(_DWORD *)(a2 + 32);
      if ( (*(_BYTE *)(a2 + 40) & 3) != 0 || v187 )
        sub_154ADE0((__int64)&v219, "virtualIndex", 0xCu, v187, 0);
      v188 = *(_DWORD *)(a2 + 36);
      if ( v188 )
      {
        v189 = v219;
        if ( v220 )
          v220 = 0;
        else
          v189 = sub_1263B40(v219, v221);
        v190 = *(void **)(v189 + 24);
        if ( *(_QWORD *)(v189 + 16) - (_QWORD)v190 <= 0xDu )
        {
          v189 = sub_16E7EE0(v189, "thisAdjustment", 14);
        }
        else
        {
          qmemcpy(v190, "thisAdjustment", 14);
          *(_QWORD *)(v189 + 24) += 14LL;
        }
        v191 = sub_1263B40(v189, ": ");
        sub_16E7AB0(v191, v188);
      }
      sub_154B2B0(&v219, "flags", 5u, *(_DWORD *)(a2 + 44));
      v192 = *(_BYTE *)(a2 + 40);
      BYTE1(v216) = 0;
      sub_154AA90((__int64)&v219, "isOptimized", 0xBu, (v192 & 0x10) != 0, &v216);
      sub_154F950((__int64)&v219, "unit", 4u, *(unsigned __int8 **)(a2 + 8 * (5LL - *(unsigned int *)(a2 + 8))), 1);
      v193 = *(unsigned int *)(a2 + 8);
      v194 = 0;
      if ( (unsigned int)v193 > 9 )
        v194 = *(unsigned __int8 **)(a2 + 8 * (9 - v193));
      sub_154F950((__int64)&v219, "templateParams", 0xEu, v194, 1);
      sub_154F950(
        (__int64)&v219,
        "declaration",
        0xBu,
        *(unsigned __int8 **)(a2 + 8 * (6LL - *(unsigned int *)(a2 + 8))),
        1);
      sub_154F950(
        (__int64)&v219,
        "retainedNodes",
        0xDu,
        *(unsigned __int8 **)(a2 + 8 * (7LL - *(unsigned int *)(a2 + 8))),
        1);
      v195 = *(unsigned int *)(a2 + 8);
      v196 = 0;
      if ( (unsigned int)v195 > 0xA )
        v196 = *(unsigned __int8 **)(a2 + 8 * (10 - v195));
      sub_154F950((__int64)&v219, "thrownTypes", 0xBu, v196, 1);
      return (_BYTE *)sub_1263B40(a1, ")");
    case 0x12:
      v101 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v101 <= 0xFu )
      {
        sub_16E7EE0(a1, "!DILexicalBlock(", 16);
      }
      else
      {
        *v101 = _mm_load_si128((const __m128i *)&xmmword_3F24B30);
        *(_QWORD *)(a1 + 24) += 16LL;
      }
      v102 = *(unsigned int *)(a2 + 8);
      v222 = a3;
      v221 = ", ";
      v223 = a4;
      v103 = *(unsigned __int8 **)(a2 + 8 * (1 - v102));
      v219 = a1;
      v104 = (unsigned __int8 *)a2;
      v220 = 1;
      v224 = a5;
      sub_154F950((__int64)&v219, "scope", 5u, v103, 0);
      if ( *(_BYTE *)a2 != 15 )
        v104 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      sub_154F950((__int64)&v219, "file", 4u, v104, 1);
      sub_154ADE0((__int64)&v219, "line", 4u, *(_DWORD *)(a2 + 24), 1);
      v21 = *(unsigned __int16 *)(a2 + 28);
      v22 = 1;
      v23 = "column";
      v24 = 6;
      goto LABEL_15;
    case 0x13:
      v134 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v134 <= 0x13u )
      {
        sub_16E7EE0(a1, "!DILexicalBlockFile(", 20);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F24B50);
        v134[1].m128i_i32[0] = 677735529;
        *v134 = si128;
        *(_QWORD *)(a1 + 24) += 20LL;
      }
      v136 = *(unsigned int *)(a2 + 8);
      v222 = a3;
      v221 = ", ";
      v223 = a4;
      v137 = *(unsigned __int8 **)(a2 + 8 * (1 - v136));
      v219 = a1;
      v138 = (unsigned __int8 *)a2;
      v220 = 1;
      v224 = a5;
      sub_154F950((__int64)&v219, "scope", 5u, v137, 0);
      if ( *(_BYTE *)a2 != 15 )
        v138 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      sub_154F950((__int64)&v219, "file", 4u, v138, 1);
      v21 = *(_DWORD *)(a2 + 24);
      v22 = 0;
      v23 = "discriminator";
      v24 = 13;
      goto LABEL_15;
    case 0x14:
      v128 = *(void **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v128 <= 0xCu )
      {
        sub_16E7EE0(a1, "!DINamespace(", 13);
      }
      else
      {
        qmemcpy(v128, "!DINamespace(", 13);
        *(_QWORD *)(a1 + 24) += 13LL;
      }
      v129 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v221 = ", ";
      v220 = 1;
      v130 = *(_QWORD *)(a2 + 8 * (2 - v129));
      v222 = a3;
      v223 = a4;
      v224 = a5;
      if ( v130 )
      {
        v130 = sub_161E970(v130);
        v132 = v131;
      }
      else
      {
        v132 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v130, v132, 1);
      sub_154F950((__int64)&v219, "scope", 5u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 0);
      v133 = *(_BYTE *)(a2 + 24);
      LOWORD(v216) = 256;
      sub_154AA90((__int64)&v219, "exportSymbols", 0xDu, v133 & 1, &v216);
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
        goto LABEL_22;
      goto LABEL_17;
    case 0x15:
      v111 = *(void **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v111 <= 9u )
      {
        sub_16E7EE0(a1, "!DIModule(", 10);
      }
      else
      {
        qmemcpy(v111, "!DIModule(", 10);
        *(_QWORD *)(a1 + 24) += 10LL;
      }
      v222 = a3;
      v221 = ", ";
      v112 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v220 = 1;
      v113 = *(unsigned __int8 **)(a2 - 8 * v112);
      v223 = a4;
      v224 = a5;
      sub_154F950((__int64)&v219, "scope", 5u, v113, 0);
      v114 = *(unsigned int *)(a2 + 8);
      v115 = *(_QWORD *)(a2 + 8 * (1 - v114));
      if ( v115 )
      {
        v115 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v114)));
        v117 = v116;
      }
      else
      {
        v117 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v115, v117, 1);
      v118 = *(unsigned int *)(a2 + 8);
      v119 = *(_QWORD *)(a2 + 8 * (2 - v118));
      if ( v119 )
      {
        v119 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v118)));
        v121 = v120;
      }
      else
      {
        v121 = 0;
      }
      sub_154AC80(&v219, "configMacros", 0xCu, v119, v121, 1);
      v122 = *(unsigned int *)(a2 + 8);
      v123 = *(_QWORD *)(a2 + 8 * (3 - v122));
      if ( v123 )
      {
        v123 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v122)));
        v125 = v124;
      }
      else
      {
        v125 = 0;
      }
      sub_154AC80(&v219, "includePath", 0xBu, v123, v125, 1);
      v126 = *(unsigned int *)(a2 + 8);
      v42 = *(_QWORD *)(a2 + 8 * (4 - v126));
      if ( v42 )
      {
        v42 = sub_161E970(*(_QWORD *)(a2 + 8 * (4 - v126)));
        v44 = v127;
      }
      else
      {
        v44 = 0;
      }
      v45 = "isysroot";
      v46 = 8;
      goto LABEL_35;
    case 0x16:
      v105 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v105 <= 0x18u )
      {
        sub_16E7EE0(a1, "!DITemplateTypeParameter(", 25);
      }
      else
      {
        v106 = _mm_load_si128((const __m128i *)&xmmword_3F24B60);
        v105[1].m128i_i8[8] = 40;
        v105[1].m128i_i64[0] = 0x726574656D617261LL;
        *v105 = v106;
        *(_QWORD *)(a1 + 24) += 25LL;
      }
      v219 = a1;
      v221 = ", ";
      v107 = *(unsigned int *)(a2 + 8);
      v220 = 1;
      v222 = a3;
      v108 = *(_QWORD *)(a2 - 8 * v107);
      v223 = a4;
      v224 = a5;
      if ( v108 )
      {
        v108 = sub_161E970(v108);
        v110 = v109;
      }
      else
      {
        v110 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v108, v110, 1);
      v27 = 0;
      v28 = "type";
      v29 = 4;
      v30 = *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)));
      goto LABEL_21;
    case 0x17:
      v95 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v95 <= 0x19u )
      {
        sub_16E7EE0(a1, "!DITemplateValueParameter(", 26);
      }
      else
      {
        v96 = _mm_load_si128((const __m128i *)&xmmword_3F24B70);
        qmemcpy(&v95[1], "Parameter(", 10);
        *v95 = v96;
        *(_QWORD *)(a1 + 24) += 26LL;
      }
      v97 = *(_WORD *)(a2 + 2) == 48;
      v222 = a3;
      v219 = a1;
      v220 = 1;
      v221 = ", ";
      v223 = a4;
      v224 = a5;
      if ( !v97 )
        sub_1549850(&v219, a2);
      v98 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      if ( v98 )
      {
        v98 = sub_161E970(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
        v100 = v99;
      }
      else
      {
        v100 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v98, v100, 1);
      sub_154F950((__int64)&v219, "type", 4u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
      v66 = 0;
      v67 = "value";
      v68 = 5;
      v69 = *(unsigned __int8 **)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
      goto LABEL_51;
    case 0x18:
      v83 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v83 <= 0x11u )
      {
        sub_16E7EE0(a1, "!DIGlobalVariable(", 18);
      }
      else
      {
        v84 = _mm_load_si128((const __m128i *)&xmmword_3F24B40);
        v83[1].m128i_i16[0] = 10341;
        *v83 = v84;
        *(_QWORD *)(a1 + 24) += 18LL;
      }
      v85 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v221 = ", ";
      v220 = 1;
      v86 = *(_QWORD *)(a2 + 8 * (1 - v85));
      v222 = a3;
      v223 = a4;
      v224 = a5;
      if ( v86 )
      {
        v86 = sub_161E970(v86);
        v88 = v87;
      }
      else
      {
        v88 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v86, v88, 1);
      v89 = *(unsigned int *)(a2 + 8);
      v90 = *(_QWORD *)(a2 + 8 * (5 - v89));
      if ( v90 )
      {
        v90 = sub_161E970(*(_QWORD *)(a2 + 8 * (5 - v89)));
        v92 = v91;
      }
      else
      {
        v92 = 0;
      }
      sub_154AC80(&v219, "linkageName", 0xBu, v90, v92, 1);
      sub_154F950((__int64)&v219, "scope", 5u, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)), 0);
      sub_154F950((__int64)&v219, "file", 4u, *(unsigned __int8 **)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154ADE0((__int64)&v219, "line", 4u, *(_DWORD *)(a2 + 24), 1);
      sub_154F950((__int64)&v219, "type", 4u, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 1);
      v93 = *(_BYTE *)(a2 + 32);
      BYTE1(v216) = 0;
      sub_154AA90((__int64)&v219, "isLocal", 7u, v93, &v216);
      v94 = *(_BYTE *)(a2 + 33);
      BYTE1(v216) = 0;
      sub_154AA90((__int64)&v219, "isDefinition", 0xCu, v94, &v216);
      sub_154F950(
        (__int64)&v219,
        "declaration",
        0xBu,
        *(unsigned __int8 **)(a2 + 8 * (6LL - *(unsigned int *)(a2 + 8))),
        1);
      v21 = *(_DWORD *)(a2 + 28);
      v22 = 1;
      v23 = "align";
      v24 = 5;
      goto LABEL_15;
    case 0x19:
      v77 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v77 <= 0x10u )
      {
        sub_16E7EE0(a1, "!DILocalVariable(", 17);
      }
      else
      {
        v78 = _mm_load_si128((const __m128i *)&xmmword_4293150);
        v77[1].m128i_i8[0] = 40;
        *v77 = v78;
        *(_QWORD *)(a1 + 24) += 17LL;
      }
      v79 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v221 = ", ";
      v220 = 1;
      v80 = *(_QWORD *)(a2 + 8 * (1 - v79));
      v222 = a3;
      v223 = a4;
      v224 = a5;
      if ( v80 )
      {
        v80 = sub_161E970(v80);
        v82 = v81;
      }
      else
      {
        v82 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v80, v82, 1);
      sub_154ADE0((__int64)&v219, "arg", 3u, *(unsigned __int16 *)(a2 + 32), 1);
      sub_154F950((__int64)&v219, "scope", 5u, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)), 0);
      sub_154F950((__int64)&v219, "file", 4u, *(unsigned __int8 **)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154ADE0((__int64)&v219, "line", 4u, *(_DWORD *)(a2 + 24), 1);
      sub_154F950((__int64)&v219, "type", 4u, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154B2B0(&v219, "flags", 5u, *(_DWORD *)(a2 + 36));
      sub_154ADE0((__int64)&v219, "align", 5u, *(_DWORD *)(a2 + 28), 1);
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
        goto LABEL_22;
      goto LABEL_17;
    case 0x1A:
      v70 = *(_QWORD *)(a1 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v70) <= 8 )
      {
        sub_16E7EE0(a1, "!DILabel(", 9);
      }
      else
      {
        *(_BYTE *)(v70 + 8) = 40;
        *(_QWORD *)v70 = 0x6C6562614C494421LL;
        *(_QWORD *)(a1 + 24) += 9LL;
      }
      v222 = a3;
      v221 = ", ";
      v71 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v220 = 1;
      v72 = *(unsigned __int8 **)(a2 - 8 * v71);
      v223 = a4;
      v224 = a5;
      sub_154F950((__int64)&v219, "scope", 5u, v72, 0);
      v73 = *(unsigned int *)(a2 + 8);
      v74 = *(_QWORD *)(a2 + 8 * (1 - v73));
      if ( v74 )
      {
        v74 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v73)));
        v76 = v75;
      }
      else
      {
        v76 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v74, v76, 1);
      v19 = *(unsigned int *)(a2 + 8);
      v20 = 2;
      goto LABEL_14;
    case 0x1B:
      v53 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v53 <= 0xFu )
      {
        sub_16E7EE0(a1, "!DIObjCProperty(", 16);
      }
      else
      {
        *v53 = _mm_load_si128((const __m128i *)&xmmword_3F24B80);
        *(_QWORD *)(a1 + 24) += 16LL;
      }
      v219 = a1;
      v221 = ", ";
      v54 = *(unsigned int *)(a2 + 8);
      v220 = 1;
      v222 = a3;
      v55 = *(_QWORD *)(a2 - 8 * v54);
      v223 = a4;
      v224 = a5;
      if ( v55 )
      {
        v55 = sub_161E970(v55);
        v57 = v56;
      }
      else
      {
        v57 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v55, v57, 1);
      sub_154F950((__int64)&v219, "file", 4u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
      sub_154ADE0((__int64)&v219, "line", 4u, *(_DWORD *)(a2 + 24), 1);
      v58 = *(unsigned int *)(a2 + 8);
      v59 = *(_QWORD *)(a2 + 8 * (3 - v58));
      if ( v59 )
      {
        v59 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v58)));
        v61 = v60;
      }
      else
      {
        v61 = 0;
      }
      sub_154AC80(&v219, "setter", 6u, v59, v61, 1);
      v62 = *(unsigned int *)(a2 + 8);
      v63 = *(_QWORD *)(a2 + 8 * (2 - v62));
      if ( v63 )
      {
        v63 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v62)));
        v65 = v64;
      }
      else
      {
        v65 = 0;
      }
      sub_154AC80(&v219, "getter", 6u, v63, v65, 1);
      sub_154ADE0((__int64)&v219, "attributes", 0xAu, *(_DWORD *)(a2 + 28), 1);
      v66 = 1;
      v67 = "type";
      v68 = 4;
      v69 = *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8)));
LABEL_51:
      sub_154F950((__int64)&v219, v67, v68, v69, v66);
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
        goto LABEL_22;
      goto LABEL_17;
    case 0x1C:
      v47 = *(__m128i **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v47 <= 0x11u )
      {
        sub_16E7EE0(a1, "!DIImportedEntity(", 18);
      }
      else
      {
        v48 = _mm_load_si128(xmmword_3F24B90);
        v47[1].m128i_i16[0] = 10361;
        *v47 = v48;
        *(_QWORD *)(a1 + 24) += 18LL;
      }
      v222 = a3;
      v221 = ", ";
      v219 = a1;
      v220 = 1;
      v223 = a4;
      v224 = a5;
      sub_1549850(&v219, a2);
      v49 = *(unsigned int *)(a2 + 8);
      v50 = *(_QWORD *)(a2 + 8 * (2 - v49));
      if ( v50 )
      {
        v50 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v49)));
        v52 = v51;
      }
      else
      {
        v52 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v50, v52, 1);
      sub_154F950((__int64)&v219, "scope", 5u, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)), 0);
      sub_154F950((__int64)&v219, "entity", 6u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
      goto LABEL_13;
    case 0x1D:
      v31 = *(_QWORD *)(a1 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v31) <= 8 )
      {
        sub_16E7EE0(a1, "!DIMacro(", 9);
        v32 = *(_QWORD *)(a1 + 24);
      }
      else
      {
        *(_BYTE *)(v31 + 8) = 40;
        *(_QWORD *)v31 = 0x6F7263614D494421LL;
        v32 = *(_QWORD *)(a1 + 24) + 9LL;
        *(_QWORD *)(a1 + 24) = v32;
      }
      v219 = a1;
      v221 = ", ";
      v33 = *(_QWORD *)(a1 + 16);
      v222 = a3;
      v223 = a4;
      v224 = a5;
      v220 = 0;
      if ( (unsigned __int64)(v33 - v32) <= 5 )
      {
        sub_16E7EE0(a1, "type: ", 6);
      }
      else
      {
        *(_DWORD *)v32 = 1701869940;
        *(_WORD *)(v32 + 4) = 8250;
        *(_QWORD *)(a1 + 24) += 6LL;
      }
      v34 = sub_14E9610(*(unsigned __int16 *)(a2 + 2));
      v36 = v35;
      if ( v35 )
      {
        v37 = *(void **)(v219 + 24);
        if ( *(_QWORD *)(v219 + 16) - (_QWORD)v37 < v35 )
        {
          sub_16E7EE0(v219, v34, v35);
        }
        else
        {
          memcpy(v37, v34, v35);
          *(_QWORD *)(v219 + 24) += v36;
        }
      }
      else
      {
        sub_16E7A90(v219, *(unsigned __int16 *)(a2 + 2));
      }
      sub_154ADE0((__int64)&v219, "line", 4u, *(_DWORD *)(a2 + 24), 1);
      v38 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
      if ( v38 )
      {
        v38 = sub_161E970(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
        v40 = v39;
      }
      else
      {
        v40 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v38, v40, 1);
      v41 = *(unsigned int *)(a2 + 8);
      v42 = *(_QWORD *)(a2 + 8 * (1 - v41));
      if ( v42 )
      {
        v42 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v41)));
        v44 = v43;
      }
      else
      {
        v44 = 0;
      }
      v45 = "value";
      v46 = 5;
LABEL_35:
      sub_154AC80(&v219, v45, v46, v42, v44, 1);
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
        goto LABEL_22;
      goto LABEL_17;
    case 0x1E:
      v25 = *(void **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v25 <= 0xCu )
      {
        sub_16E7EE0(a1, "!DIMacroFile(", 13);
      }
      else
      {
        qmemcpy(v25, "!DIMacroFile(", 13);
        *(_QWORD *)(a1 + 24) += 13LL;
      }
      v223 = a4;
      v26 = *(_DWORD *)(a2 + 24);
      v221 = ", ";
      v219 = a1;
      v220 = 1;
      v222 = a3;
      v224 = a5;
      sub_154ADE0((__int64)&v219, "line", 4u, v26, 1);
      sub_154F950((__int64)&v219, "file", 4u, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)), 0);
      v27 = 1;
      v28 = "nodes";
      v29 = 5;
      v30 = *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)));
LABEL_21:
      sub_154F950((__int64)&v219, v28, v29, v30, v27);
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
        goto LABEL_22;
      goto LABEL_17;
    case 0x1F:
      v12 = *(void **)(a1 + 24);
      if ( *(_QWORD *)(a1 + 16) - (_QWORD)v12 <= 0xEu )
      {
        sub_16E7EE0(a1, "!DICommonBlock(", 15);
      }
      else
      {
        qmemcpy(v12, "!DICommonBlock(", 15);
        *(_QWORD *)(a1 + 24) += 15LL;
      }
      v222 = a3;
      v221 = ", ";
      v13 = *(unsigned int *)(a2 + 8);
      v219 = a1;
      v220 = 1;
      v14 = *(unsigned __int8 **)(a2 - 8 * v13);
      v223 = a4;
      v224 = a5;
      sub_154F950((__int64)&v219, "scope", 5u, v14, 0);
      sub_154F950(
        (__int64)&v219,
        "declaration",
        0xBu,
        *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))),
        0);
      v15 = *(unsigned int *)(a2 + 8);
      v16 = *(_QWORD *)(a2 + 8 * (2 - v15));
      if ( v16 )
      {
        v16 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v15)));
        v18 = v17;
      }
      else
      {
        v18 = 0;
      }
      sub_154AC80(&v219, "name", 4u, v16, v18, 1);
LABEL_13:
      v19 = *(unsigned int *)(a2 + 8);
      v20 = 3;
LABEL_14:
      sub_154F950((__int64)&v219, "file", 4u, *(unsigned __int8 **)(a2 + 8 * (v20 - v19)), 1);
      v21 = *(_DWORD *)(a2 + 24);
      v22 = 1;
      v23 = "line";
      v24 = 4;
LABEL_15:
      sub_154ADE0((__int64)&v219, v23, v24, v21, v22);
LABEL_16:
      result = *(_BYTE **)(a1 + 24);
      if ( *(_BYTE **)(a1 + 16) == result )
      {
LABEL_22:
        result = (_BYTE *)sub_16E7EE0(a1, ")", 1);
      }
      else
      {
LABEL_17:
        *result = 41;
        ++*(_QWORD *)(a1 + 24);
      }
      break;
    case 0x20:
      result = sub_1550580(a1, a2, a3, a4, a5);
      break;
    case 0x21:
      result = sub_1550770(a1, a2, a3, a4, a5);
      break;
    case 0x22:
      result = sub_15509D0(a1, a2, a3, a4, a5);
      break;
  }
  return result;
}
