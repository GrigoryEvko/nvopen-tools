// Function: sub_A5F6B0
// Address: 0xa5f6b0
//
_BYTE *__fastcall sub_A5F6B0(__int64 a1, const char *a2, __int64 *a3)
{
  __int64 v5; // r12
  char v6; // al
  _BYTE *result; // rax
  void *v8; // rdx
  __int64 v9; // rbx
  unsigned __int8 v10; // al
  __int64 *v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r8
  unsigned __int8 v19; // al
  __int64 v20; // rbx
  __int64 v21; // rcx
  unsigned int v22; // ecx
  char v23; // r8
  char *v24; // rsi
  size_t v25; // rdx
  void *v26; // rdx
  unsigned int v27; // ecx
  __int64 v28; // rbx
  unsigned __int8 v29; // al
  __int64 *v30; // rdx
  unsigned __int8 v31; // al
  __int64 v32; // rbx
  __int64 v33; // rcx
  char v34; // r8
  char *v35; // rsi
  size_t v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rax
  size_t v40; // rdx
  const void *v41; // rsi
  __int64 v42; // rbx
  unsigned __int8 v43; // al
  __int64 *v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // r8
  unsigned __int8 v48; // al
  __int64 v49; // rbx
  __int64 v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // r8
  __m128i *v53; // rdx
  __m128i v54; // xmm0
  __int64 v55; // rbx
  unsigned __int8 v56; // al
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // r8
  unsigned __int8 v61; // al
  __int64 *v62; // rdx
  unsigned __int8 v63; // al
  __int64 v64; // rdx
  unsigned __int8 v65; // al
  __int64 v66; // rdx
  unsigned __int8 v67; // al
  __int64 v68; // rbx
  __m128i *v69; // rdx
  __int64 v70; // rbx
  unsigned __int8 v71; // al
  __int64 *v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // r8
  unsigned __int8 v76; // al
  __int64 v77; // rdx
  unsigned __int8 v78; // al
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // rdx
  __int64 v82; // r8
  unsigned __int8 v83; // al
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // rdx
  __int64 v87; // r8
  unsigned __int8 v88; // al
  __int64 v89; // rbx
  __int64 v90; // rcx
  size_t v91; // rdx
  char *v92; // rsi
  void *v93; // rdx
  unsigned __int8 v94; // al
  __int64 *v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // rdx
  __int64 v98; // r8
  int v99; // ebx
  __int64 v100; // rdi
  __int64 v101; // rdx
  __m128i *v102; // rdx
  __m128i si128; // xmm0
  __int64 v104; // rbx
  unsigned __int8 v105; // al
  __int64 *v106; // rdx
  unsigned __int8 v107; // al
  __int64 v108; // rbx
  __int64 v109; // rdx
  __int64 v110; // rbx
  unsigned __int8 v111; // al
  __int64 *v112; // rdx
  unsigned __int8 v113; // al
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // rdx
  __int64 v117; // r8
  unsigned __int8 v118; // al
  __int64 v119; // rbx
  __int64 v120; // rbx
  unsigned __int8 v121; // al
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // rdx
  __int64 v125; // r8
  unsigned __int8 v126; // al
  __int64 *v127; // rdx
  unsigned __int8 v128; // al
  __int64 v129; // rdx
  unsigned __int8 v130; // al
  __int64 v131; // rdx
  unsigned __int8 v132; // al
  __int64 v133; // rbx
  __m128i *v134; // rdx
  __m128i v135; // xmm0
  __int64 v136; // rbx
  unsigned __int8 v137; // al
  __int64 *v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // rdx
  __int64 v141; // r8
  unsigned __int8 v142; // al
  __int64 v143; // rbx
  char *v144; // rsi
  size_t v145; // rdx
  unsigned int v146; // ecx
  __int64 v147; // rbx
  unsigned __int8 v148; // al
  __int64 v149; // rdx
  unsigned __int8 v150; // al
  __int64 v151; // rdx
  __int64 v152; // rcx
  __int64 v153; // rdx
  __int64 v154; // r8
  unsigned __int8 v155; // al
  __int64 v156; // rdx
  __int64 v157; // rcx
  __int64 v158; // rdx
  __int64 v159; // r8
  unsigned __int8 v160; // al
  __int64 v161; // rdx
  __int64 v162; // rcx
  __int64 v163; // rdx
  __int64 v164; // r8
  unsigned __int8 v165; // al
  __int64 v166; // rdx
  __int64 v167; // rcx
  __int64 v168; // rdx
  __int64 v169; // r8
  __int64 v170; // rcx
  unsigned __int8 v171; // al
  __int64 *v172; // rbx
  void *v173; // rdx
  __int64 v174; // rbx
  unsigned __int8 v175; // al
  __int64 v176; // rdx
  __int64 v177; // rcx
  __int64 v178; // rdx
  __int64 v179; // r8
  unsigned __int8 v180; // al
  __int64 v181; // rbx
  __m128i *v182; // rdx
  __m128i v183; // xmm0
  __int64 v184; // rbx
  unsigned __int8 v185; // al
  __int64 v186; // rdx
  __int64 v187; // rcx
  unsigned __int8 v188; // al
  __int64 *v189; // rbx
  void *v190; // rdx
  unsigned __int8 v191; // al
  __int64 v192; // rdx
  __int64 v193; // rcx
  __int64 v194; // rdx
  __int64 v195; // r8
  unsigned int v196; // eax
  __int64 *v197; // rsi
  __int64 v198; // rbx
  unsigned __int8 v199; // al
  __int64 *v200; // rdx
  __int64 v201; // rcx
  __int64 v202; // rdx
  __int64 v203; // r8
  unsigned __int8 v204; // al
  __int64 v205; // rbx
  __int64 v206; // rcx
  __int64 v207; // rdx
  __int64 v208; // r8
  char *v209; // rsi
  __int64 v210; // rcx
  __int64 v211; // r8
  __int64 v212; // rdx
  __int64 v213; // rbx
  unsigned __int8 v214; // al
  __int64 v215; // rdx
  __int64 v216; // rcx
  __int64 v217; // rdx
  __int64 v218; // r8
  unsigned __int8 v219; // al
  __int64 v220; // rdx
  __int64 v221; // rcx
  __int64 v222; // rdx
  __int64 v223; // r8
  unsigned __int8 v224; // al
  __int64 *v225; // rdx
  unsigned __int8 v226; // al
  __int64 v227; // rdx
  unsigned __int8 v228; // al
  __int64 v229; // rdx
  unsigned __int8 v230; // al
  __int64 v231; // rdx
  unsigned __int8 v232; // al
  __int64 v233; // rdx
  unsigned __int8 v234; // al
  __int64 v235; // rbx
  __m128i *v236; // rdx
  __m128i v237; // xmm0
  unsigned __int8 v238; // al
  __int64 v239; // rbx
  __int64 *v240; // rdx
  __int64 v241; // rcx
  __int64 v242; // rdx
  __int64 v243; // r8
  unsigned __int8 v244; // al
  __int64 v245; // rdx
  unsigned __int8 v246; // al
  __int64 v247; // rbx
  unsigned int v248; // ebx
  bool v249; // zf
  __int64 v250; // rdi
  __int64 v251; // r15
  const void *v252; // rax
  size_t v253; // rdx
  __int64 v254; // rax
  __int64 v255; // rdx
  __int64 v256; // [rsp+0h] [rbp-70h] BYREF
  char v257; // [rsp+8h] [rbp-68h]
  const char *v258; // [rsp+10h] [rbp-60h]
  __int64 *v259; // [rsp+18h] [rbp-58h]
  __int128 v260; // [rsp+20h] [rbp-50h] BYREF
  __int128 v261; // [rsp+30h] [rbp-40h]

  v5 = (__int64)a2;
  v6 = a2[1] & 0x7F;
  if ( v6 == 1 )
  {
    a2 = "distinct ";
    sub_904010(a1, "distinct ");
  }
  else if ( v6 == 2 )
  {
    a2 = "<temporary!> ";
    sub_904010(a1, "<temporary!> ");
  }
  switch ( *(_BYTE *)v5 )
  {
    case 5:
      return (_BYTE *)sub_A5C310(a1, v5, a3);
    case 6:
      return sub_A5CD70(a1, v5, (__int64)a3);
    case 7:
      return (_BYTE *)sub_A52D40(a1, v5);
    case 8:
      v102 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v102 <= 0x1Bu )
      {
        sub_CB6200(a1, "!DIGlobalVariableExpression(", 28);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F24B40);
        qmemcpy(&v102[1], "eExpression(", 12);
        *v102 = si128;
        *(_QWORD *)(a1 + 32) += 28LL;
      }
      *(_QWORD *)&v260 = a1;
      v104 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v105 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v105 & 2) != 0 )
        v106 = *(__int64 **)(v5 - 32);
      else
        v106 = (__int64 *)(v104 - 8LL * ((v105 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "var", 3u, *v106, 1);
      v107 = *(_BYTE *)(v5 - 16);
      if ( (v107 & 2) != 0 )
        v108 = *(_QWORD *)(v5 - 32);
      else
        v108 = v104 - 8LL * ((v107 >> 2) & 0xF);
      v33 = *(_QWORD *)(v108 + 8);
      v34 = 1;
      v36 = 4;
      v35 = "expr";
      goto LABEL_33;
    case 9:
      return (_BYTE *)sub_A5F430(a1, v5, a3);
    case 0xA:
      return (_BYTE *)sub_A5CF50(a1, v5, (__int64)a3);
    case 0xB:
      v93 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v93 <= 0xDu )
      {
        a2 = "!DIEnumerator(";
        sub_CB6200(a1, "!DIEnumerator(", 14);
      }
      else
      {
        qmemcpy(v93, "!DIEnumerator(", 14);
        *(_QWORD *)(a1 + 32) += 14LL;
      }
      *(_QWORD *)&v260 = a1;
      BYTE8(v260) = 1;
      *(_QWORD *)&v261 = ", ";
      if ( !byte_4F80910 && (unsigned int)sub_2207590(&byte_4F80910) )
      {
        qword_4F80928 = 0;
        a2 = (const char *)&qword_4F80920;
        qword_4F80920 = (__int64)off_4979428;
        qword_4F80930 = 0;
        qword_4F80938 = 0;
        __cxa_atexit(nullsub_37, &qword_4F80920, &qword_4A427C0);
        sub_2207640(&byte_4F80910);
      }
      *((_QWORD *)&v261 + 1) = &qword_4F80920;
      v94 = *(_BYTE *)(v5 - 16);
      if ( (v94 & 2) != 0 )
        v95 = *(__int64 **)(v5 - 32);
      else
        v95 = (__int64 *)(v5 - 16 - 8LL * ((v94 >> 2) & 0xF));
      v96 = *v95;
      if ( *v95 )
      {
        v96 = sub_B91420(*v95, a2);
        v98 = v97;
      }
      else
      {
        v98 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v96, v98, 0);
      v99 = *(_DWORD *)(v5 + 4);
      v100 = v260;
      if ( BYTE8(v260) )
        BYTE8(v260) = 0;
      else
        v100 = sub_904010(v260, (const char *)v261);
      v101 = *(_QWORD *)(v100 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v100 + 24) - v101) <= 4 )
      {
        v100 = sub_CB6200(v100, "value", 5);
      }
      else
      {
        *(_DWORD *)v101 = 1970037110;
        *(_BYTE *)(v101 + 4) = 101;
        *(_QWORD *)(v100 + 32) += 5LL;
      }
      sub_904010(v100, ": ");
      sub_C49420(v5 + 16, v260, v99 == 0);
      if ( *(_DWORD *)(v5 + 4) )
        sub_A53370((__int64)&v260, "isUnsigned", 0xAu, 1, 0);
      goto LABEL_23;
    case 0xC:
      v190 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v190 <= 0xCu )
      {
        a2 = "!DIBasicType(";
        sub_CB6200(a1, "!DIBasicType(", 13);
      }
      else
      {
        qmemcpy(v190, "!DIBasicType(", 13);
        *(_QWORD *)(a1 + 32) += 13LL;
      }
      *(_QWORD *)&v260 = a1;
      BYTE8(v260) = 1;
      *(_QWORD *)&v261 = ", ";
      if ( !byte_4F80910 && (unsigned int)sub_2207590(&byte_4F80910) )
      {
        qword_4F80928 = 0;
        a2 = (const char *)&qword_4F80920;
        qword_4F80920 = (__int64)off_4979428;
        qword_4F80930 = 0;
        qword_4F80938 = 0;
        __cxa_atexit(nullsub_37, &qword_4F80920, &qword_4A427C0);
        sub_2207640(&byte_4F80910);
      }
      *((_QWORD *)&v261 + 1) = &qword_4F80920;
      if ( (unsigned __int16)sub_AF18C0(v5) != 36 )
      {
        a2 = (const char *)v5;
        sub_A53560(&v260, v5);
      }
      v191 = *(_BYTE *)(v5 - 16);
      if ( (v191 & 2) != 0 )
        v192 = *(_QWORD *)(v5 - 32);
      else
        v192 = v5 - 16 - 8LL * ((v191 >> 2) & 0xF);
      v193 = *(_QWORD *)(v192 + 16);
      if ( v193 )
      {
        v193 = sub_B91420(*(_QWORD *)(v192 + 16), a2);
        v195 = v194;
      }
      else
      {
        v195 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v193, v195, 1);
      sub_A539C0((__int64)&v260, "size", 4u, *(_QWORD *)(v5 + 24));
      v196 = sub_AF18D0(v5);
      sub_A537C0((__int64)&v260, "align", 5u, v196, 1);
      sub_A53AC0((__int64 *)&v260, "encoding", 8u, *(_DWORD *)(v5 + 44), sub_E09D50, 1);
      sub_A537C0((__int64)&v260, "num_extra_inhabitants", 0x15u, *(_DWORD *)(v5 + 40), 1);
      sub_A53C60((__int64 *)&v260, "flags", 5u, *(_DWORD *)(v5 + 20));
      result = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == result )
        goto LABEL_34;
      goto LABEL_24;
    case 0xD:
      return (_BYTE *)sub_A5D2A0(a1, v5, (__int64)a3);
    case 0xE:
      return (_BYTE *)sub_A5D640(a1, v5, (__int64)a3);
    case 0xF:
      return sub_A5DBD0(a1, v5, (__int64)a3);
    case 0x10:
      v197 = (__int64 *)"!DIFile(";
      sub_904010(a1, "!DIFile(");
      v256 = a1;
      v257 = 1;
      v258 = ", ";
      if ( !byte_4F80910 && (unsigned int)sub_2207590(&byte_4F80910) )
      {
        qword_4F80928 = 0;
        v197 = &qword_4F80920;
        qword_4F80920 = (__int64)off_4979428;
        qword_4F80930 = 0;
        qword_4F80938 = 0;
        __cxa_atexit(nullsub_37, &qword_4F80920, &qword_4A427C0);
        sub_2207640(&byte_4F80910);
      }
      v198 = v5 - 16;
      v259 = &qword_4F80920;
      v199 = *(_BYTE *)(v5 - 16);
      if ( (v199 & 2) != 0 )
        v200 = *(__int64 **)(v5 - 32);
      else
        v200 = (__int64 *)(v198 - 8LL * ((v199 >> 2) & 0xF));
      v201 = *v200;
      if ( *v200 )
      {
        v201 = sub_B91420(*v200, v197);
        v203 = v202;
      }
      else
      {
        v203 = 0;
      }
      sub_A53660(&v256, "filename", 8u, v201, v203, 0);
      v204 = *(_BYTE *)(v5 - 16);
      if ( (v204 & 2) != 0 )
        v205 = *(_QWORD *)(v5 - 32);
      else
        v205 = v198 - 8LL * ((v204 >> 2) & 0xF);
      v206 = *(_QWORD *)(v205 + 8);
      if ( v206 )
      {
        v206 = sub_B91420(*(_QWORD *)(v205 + 8), "filename");
        v208 = v207;
      }
      else
      {
        v208 = 0;
      }
      v209 = "directory";
      sub_A53660(&v256, "directory", 9u, v206, v208, 0);
      if ( *(_BYTE *)(v5 + 32) )
      {
        v248 = 0;
        sub_B91420(*(_QWORD *)(v5 + 24), "directory");
        v249 = *(_BYTE *)(v5 + 32) == 0;
        v260 = 0;
        v261 = 0;
        if ( !v249 )
        {
          v254 = sub_B91420(*(_QWORD *)(v5 + 24), "directory");
          v248 = *(_DWORD *)(v5 + 16);
          *((_QWORD *)&v260 + 1) = v254;
          *(_QWORD *)&v261 = v255;
        }
        v250 = v256;
        if ( v257 )
          v257 = 0;
        else
          v250 = sub_904010(v256, v258);
        v251 = sub_904010(v250, "checksumkind: ");
        v252 = (const void *)sub_AF2F50(v248);
        sub_A51340(v251, v252, v253);
        v209 = "checksum";
        sub_A53660(&v256, "checksum", 8u, *((__int64 *)&v260 + 1), v261, 0);
      }
      v210 = *(_QWORD *)(v5 + 40);
      v211 = 0;
      if ( v210 )
      {
        v210 = sub_B91420(*(_QWORD *)(v5 + 40), v209);
        v211 = v212;
      }
      sub_A53660(&v256, "source", 6u, v210, v211, 1);
      return (_BYTE *)sub_904010(a1, ")");
    case 0x11:
      return (_BYTE *)sub_A5DDD0(a1, v5, (__int64)a3);
    case 0x12:
      return (_BYTE *)sub_A5E2A0(a1, v5, (__int64)a3);
    case 0x13:
      return sub_A5E9B0(a1, v5, (__int64)a3);
    case 0x14:
      v182 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v182 <= 0x13u )
      {
        sub_CB6200(a1, "!DILexicalBlockFile(", 20);
      }
      else
      {
        v183 = _mm_load_si128((const __m128i *)&xmmword_3F24B50);
        v182[1].m128i_i32[0] = 677735529;
        *v182 = v183;
        *(_QWORD *)(a1 + 32) += 20LL;
      }
      *(_QWORD *)&v260 = a1;
      v184 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v185 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v185 & 2) != 0 )
        v186 = *(_QWORD *)(v5 - 32);
      else
        v186 = v184 - 8LL * ((v185 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "scope", 5u, *(_QWORD *)(v186 + 8), 0);
      v187 = v5;
      if ( *(_BYTE *)v5 != 16 )
      {
        v188 = *(_BYTE *)(v5 - 16);
        if ( (v188 & 2) != 0 )
          v189 = *(__int64 **)(v5 - 32);
        else
          v189 = (__int64 *)(v184 - 8LL * ((v188 >> 2) & 0xF));
        v187 = *v189;
      }
      sub_A5CC00((__int64)&v260, "file", 4u, v187, 1);
      v22 = *(_DWORD *)(v5 + 4);
      v23 = 0;
      v24 = "discriminator";
      v25 = 13;
      goto LABEL_22;
    case 0x15:
      v173 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v173 <= 0xCu )
      {
        a2 = "!DINamespace(";
        sub_CB6200(a1, "!DINamespace(", 13);
      }
      else
      {
        qmemcpy(v173, "!DINamespace(", 13);
        *(_QWORD *)(a1 + 32) += 13LL;
      }
      *(_QWORD *)&v260 = a1;
      v174 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v175 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v175 & 2) != 0 )
        v176 = *(_QWORD *)(v5 - 32);
      else
        v176 = v174 - 8LL * ((v175 >> 2) & 0xF);
      v177 = *(_QWORD *)(v176 + 16);
      if ( v177 )
      {
        v177 = sub_B91420(*(_QWORD *)(v176 + 16), a2);
        v179 = v178;
      }
      else
      {
        v179 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v177, v179, 1);
      v180 = *(_BYTE *)(v5 - 16);
      if ( (v180 & 2) != 0 )
        v181 = *(_QWORD *)(v5 - 32);
      else
        v181 = v174 - 8LL * ((v180 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "scope", 5u, *(_QWORD *)(v181 + 8), 0);
      v144 = "exportSymbols";
      v145 = 13;
      v146 = (unsigned int)*(char *)(v5 + 1) >> 31;
      goto LABEL_147;
    case 0x16:
      v147 = v5 - 16;
      sub_904010(a1, "!DIModule(");
      *(_QWORD *)&v260 = a1;
      *(_QWORD *)&v261 = ", ";
      v148 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v148 & 2) != 0 )
        v149 = *(_QWORD *)(v5 - 32);
      else
        v149 = v147 - 8LL * ((v148 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "scope", 5u, *(_QWORD *)(v149 + 8), 0);
      v150 = *(_BYTE *)(v5 - 16);
      if ( (v150 & 2) != 0 )
        v151 = *(_QWORD *)(v5 - 32);
      else
        v151 = v147 - 8LL * ((v150 >> 2) & 0xF);
      v152 = *(_QWORD *)(v151 + 16);
      if ( v152 )
      {
        v152 = sub_B91420(*(_QWORD *)(v151 + 16), "scope");
        v154 = v153;
      }
      else
      {
        v154 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v152, v154, 1);
      v155 = *(_BYTE *)(v5 - 16);
      if ( (v155 & 2) != 0 )
        v156 = *(_QWORD *)(v5 - 32);
      else
        v156 = v147 - 8LL * ((v155 >> 2) & 0xF);
      v157 = *(_QWORD *)(v156 + 24);
      if ( v157 )
      {
        v157 = sub_B91420(*(_QWORD *)(v156 + 24), "name");
        v159 = v158;
      }
      else
      {
        v159 = 0;
      }
      sub_A53660((__int64 *)&v260, "configMacros", 0xCu, v157, v159, 1);
      v160 = *(_BYTE *)(v5 - 16);
      if ( (v160 & 2) != 0 )
        v161 = *(_QWORD *)(v5 - 32);
      else
        v161 = v147 - 8LL * ((v160 >> 2) & 0xF);
      v162 = *(_QWORD *)(v161 + 32);
      if ( v162 )
      {
        v162 = sub_B91420(*(_QWORD *)(v161 + 32), "configMacros");
        v164 = v163;
      }
      else
      {
        v164 = 0;
      }
      sub_A53660((__int64 *)&v260, "includePath", 0xBu, v162, v164, 1);
      v165 = *(_BYTE *)(v5 - 16);
      if ( (v165 & 2) != 0 )
        v166 = *(_QWORD *)(v5 - 32);
      else
        v166 = v147 - 8LL * ((v165 >> 2) & 0xF);
      v167 = *(_QWORD *)(v166 + 40);
      if ( v167 )
      {
        v167 = sub_B91420(*(_QWORD *)(v166 + 40), "includePath");
        v169 = v168;
      }
      else
      {
        v169 = 0;
      }
      sub_A53660((__int64 *)&v260, "apinotes", 8u, v167, v169, 1);
      v170 = v5;
      if ( *(_BYTE *)v5 != 16 )
      {
        v171 = *(_BYTE *)(v5 - 16);
        if ( (v171 & 2) != 0 )
          v172 = *(__int64 **)(v5 - 32);
        else
          v172 = (__int64 *)(v147 - 8LL * ((v171 >> 2) & 0xF));
        v170 = *v172;
      }
      sub_A5CC00((__int64)&v260, "file", 4u, v170, 1);
      sub_A537C0((__int64)&v260, "line", 4u, *(_DWORD *)(v5 + 4), 1);
      sub_A53370((__int64)&v260, "isDecl", 6u, *(char *)(v5 + 1) < 0, 0x100u);
      return (_BYTE *)sub_904010(a1, ")");
    case 0x17:
      v134 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v134 <= 0x18u )
      {
        a2 = "!DITemplateTypeParameter(";
        sub_CB6200(a1, "!DITemplateTypeParameter(", 25);
      }
      else
      {
        v135 = _mm_load_si128((const __m128i *)&xmmword_3F24B60);
        v134[1].m128i_i8[8] = 40;
        v134[1].m128i_i64[0] = 0x726574656D617261LL;
        *v134 = v135;
        *(_QWORD *)(a1 + 32) += 25LL;
      }
      *(_QWORD *)&v260 = a1;
      v136 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v137 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v137 & 2) != 0 )
        v138 = *(__int64 **)(v5 - 32);
      else
        v138 = (__int64 *)(v136 - 8LL * ((v137 >> 2) & 0xF));
      v139 = *v138;
      if ( *v138 )
      {
        v139 = sub_B91420(*v138, a2);
        v141 = v140;
      }
      else
      {
        v141 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v139, v141, 1);
      v142 = *(_BYTE *)(v5 - 16);
      if ( (v142 & 2) != 0 )
        v143 = *(_QWORD *)(v5 - 32);
      else
        v143 = v136 - 8LL * ((v142 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "type", 4u, *(_QWORD *)(v143 + 8), 0);
      v144 = "defaulted";
      v145 = 9;
      v146 = (unsigned int)*(char *)(v5 + 1) >> 31;
LABEL_147:
      sub_A53370((__int64)&v260, v144, v145, v146, 0x100u);
      result = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == result )
        goto LABEL_34;
      goto LABEL_24;
    case 0x18:
      v236 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v236 <= 0x19u )
      {
        a2 = "!DITemplateValueParameter(";
        sub_CB6200(a1, "!DITemplateValueParameter(", 26);
      }
      else
      {
        v237 = _mm_load_si128((const __m128i *)&xmmword_3F24B70);
        qmemcpy(&v236[1], "Parameter(", 10);
        *v236 = v237;
        *(_QWORD *)(a1 + 32) += 26LL;
      }
      *((_QWORD *)&v261 + 1) = a3;
      *(_QWORD *)&v260 = a1;
      BYTE8(v260) = 1;
      *(_QWORD *)&v261 = ", ";
      if ( (unsigned __int16)sub_AF18C0(v5) != 48 )
      {
        a2 = (const char *)v5;
        sub_A53560(&v260, v5);
      }
      v238 = *(_BYTE *)(v5 - 16);
      v239 = v5 - 16;
      if ( (v238 & 2) != 0 )
        v240 = *(__int64 **)(v5 - 32);
      else
        v240 = (__int64 *)(v239 - 8LL * ((v238 >> 2) & 0xF));
      v241 = *v240;
      if ( *v240 )
      {
        v241 = sub_B91420(*v240, a2);
        v243 = v242;
      }
      else
      {
        v243 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v241, v243, 1);
      v244 = *(_BYTE *)(v5 - 16);
      if ( (v244 & 2) != 0 )
        v245 = *(_QWORD *)(v5 - 32);
      else
        v245 = v239 - 8LL * ((v244 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "type", 4u, *(_QWORD *)(v245 + 8), 1);
      sub_A53370((__int64)&v260, "defaulted", 9u, *(char *)(v5 + 1) < 0, 0x100u);
      v246 = *(_BYTE *)(v5 - 16);
      if ( (v246 & 2) != 0 )
        v247 = *(_QWORD *)(v5 - 32);
      else
        v247 = v239 - 8LL * ((v246 >> 2) & 0xF);
      v33 = *(_QWORD *)(v247 + 16);
      v34 = 0;
      v35 = "value";
      goto LABEL_32;
    case 0x19:
      v213 = v5 - 16;
      sub_904010(a1, "!DIGlobalVariable(");
      *(_QWORD *)&v260 = a1;
      *(_QWORD *)&v261 = ", ";
      v214 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v214 & 2) != 0 )
        v215 = *(_QWORD *)(v5 - 32);
      else
        v215 = v213 - 8LL * ((v214 >> 2) & 0xF);
      v216 = *(_QWORD *)(v215 + 8);
      if ( v216 )
      {
        v216 = sub_B91420(*(_QWORD *)(v215 + 8), "!DIGlobalVariable(");
        v218 = v217;
      }
      else
      {
        v218 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v216, v218, 1);
      v219 = *(_BYTE *)(v5 - 16);
      if ( (v219 & 2) != 0 )
        v220 = *(_QWORD *)(v5 - 32);
      else
        v220 = v213 - 8LL * ((v219 >> 2) & 0xF);
      v221 = *(_QWORD *)(v220 + 40);
      if ( v221 )
      {
        v221 = sub_B91420(*(_QWORD *)(v220 + 40), "name");
        v223 = v222;
      }
      else
      {
        v223 = 0;
      }
      sub_A53660((__int64 *)&v260, "linkageName", 0xBu, v221, v223, 1);
      v224 = *(_BYTE *)(v5 - 16);
      if ( (v224 & 2) != 0 )
        v225 = *(__int64 **)(v5 - 32);
      else
        v225 = (__int64 *)(v213 - 8LL * ((v224 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "scope", 5u, *v225, 0);
      v226 = *(_BYTE *)(v5 - 16);
      if ( (v226 & 2) != 0 )
        v227 = *(_QWORD *)(v5 - 32);
      else
        v227 = v213 - 8LL * ((v226 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "file", 4u, *(_QWORD *)(v227 + 16), 1);
      sub_A537C0((__int64)&v260, "line", 4u, *(_DWORD *)(v5 + 16), 1);
      v228 = *(_BYTE *)(v5 - 16);
      if ( (v228 & 2) != 0 )
        v229 = *(_QWORD *)(v5 - 32);
      else
        v229 = v213 - 8LL * ((v228 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "type", 4u, *(_QWORD *)(v229 + 24), 1);
      sub_A53370((__int64)&v260, "isLocal", 7u, *(_BYTE *)(v5 + 20), 0);
      sub_A53370((__int64)&v260, "isDefinition", 0xCu, *(_BYTE *)(v5 + 21), 0);
      v230 = *(_BYTE *)(v5 - 16);
      if ( (v230 & 2) != 0 )
        v231 = *(_QWORD *)(v5 - 32);
      else
        v231 = v213 - 8LL * ((v230 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "declaration", 0xBu, *(_QWORD *)(v231 + 48), 1);
      v232 = *(_BYTE *)(v5 - 16);
      if ( (v232 & 2) != 0 )
        v233 = *(_QWORD *)(v5 - 32);
      else
        v233 = v213 - 8LL * ((v232 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "templateParams", 0xEu, *(_QWORD *)(v233 + 56), 1);
      sub_A537C0((__int64)&v260, "align", 5u, *(_DWORD *)(v5 + 4), 1);
      v234 = *(_BYTE *)(v5 - 16);
      if ( (v234 & 2) != 0 )
        v235 = *(_QWORD *)(v5 - 32);
      else
        v235 = v213 - 8LL * ((v234 >> 2) & 0xF);
      v90 = *(_QWORD *)(v235 + 64);
      v91 = 11;
      v92 = "annotations";
      goto LABEL_86;
    case 0x1A:
      v120 = v5 - 16;
      sub_904010(a1, "!DILocalVariable(");
      *(_QWORD *)&v260 = a1;
      *(_QWORD *)&v261 = ", ";
      v121 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v121 & 2) != 0 )
        v122 = *(_QWORD *)(v5 - 32);
      else
        v122 = v120 - 8LL * ((v121 >> 2) & 0xF);
      v123 = *(_QWORD *)(v122 + 8);
      if ( v123 )
      {
        v123 = sub_B91420(*(_QWORD *)(v122 + 8), "!DILocalVariable(");
        v125 = v124;
      }
      else
      {
        v125 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v123, v125, 1);
      sub_A537C0((__int64)&v260, "arg", 3u, *(unsigned __int16 *)(v5 + 20), 1);
      v126 = *(_BYTE *)(v5 - 16);
      if ( (v126 & 2) != 0 )
        v127 = *(__int64 **)(v5 - 32);
      else
        v127 = (__int64 *)(v120 - 8LL * ((v126 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "scope", 5u, *v127, 0);
      v128 = *(_BYTE *)(v5 - 16);
      if ( (v128 & 2) != 0 )
        v129 = *(_QWORD *)(v5 - 32);
      else
        v129 = v120 - 8LL * ((v128 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "file", 4u, *(_QWORD *)(v129 + 16), 1);
      sub_A537C0((__int64)&v260, "line", 4u, *(_DWORD *)(v5 + 16), 1);
      v130 = *(_BYTE *)(v5 - 16);
      if ( (v130 & 2) != 0 )
        v131 = *(_QWORD *)(v5 - 32);
      else
        v131 = v120 - 8LL * ((v130 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "type", 4u, *(_QWORD *)(v131 + 24), 1);
      sub_A53C60((__int64 *)&v260, "flags", 5u, *(_DWORD *)(v5 + 24));
      sub_A537C0((__int64)&v260, "align", 5u, *(_DWORD *)(v5 + 4), 1);
      v132 = *(_BYTE *)(v5 - 16);
      if ( (v132 & 2) != 0 )
        v133 = *(_QWORD *)(v5 - 32);
      else
        v133 = v120 - 8LL * ((v132 >> 2) & 0xF);
      v33 = *(_QWORD *)(v133 + 32);
      v34 = 1;
      v36 = 11;
      v35 = "annotations";
      goto LABEL_33;
    case 0x1B:
      v109 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v109) <= 8 )
      {
        sub_CB6200(a1, "!DILabel(", 9);
      }
      else
      {
        *(_BYTE *)(v109 + 8) = 40;
        *(_QWORD *)v109 = 0x6C6562614C494421LL;
        *(_QWORD *)(a1 + 32) += 9LL;
      }
      *(_QWORD *)&v260 = a1;
      v110 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v111 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v111 & 2) != 0 )
        v112 = *(__int64 **)(v5 - 32);
      else
        v112 = (__int64 *)(v110 - 8LL * ((v111 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "scope", 5u, *v112, 0);
      v113 = *(_BYTE *)(v5 - 16);
      if ( (v113 & 2) != 0 )
        v114 = *(_QWORD *)(v5 - 32);
      else
        v114 = v110 - 8LL * ((v113 >> 2) & 0xF);
      v115 = *(_QWORD *)(v114 + 8);
      if ( v115 )
      {
        v115 = sub_B91420(*(_QWORD *)(v114 + 8), "scope");
        v117 = v116;
      }
      else
      {
        v117 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v115, v117, 1);
      v118 = *(_BYTE *)(v5 - 16);
      if ( (v118 & 2) != 0 )
        v119 = *(_QWORD *)(v5 - 32);
      else
        v119 = v110 - 8LL * ((v118 >> 2) & 0xF);
      v21 = *(_QWORD *)(v119 + 16);
      goto LABEL_21;
    case 0x1C:
      v69 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v69 <= 0xFu )
      {
        a2 = "!DIObjCProperty(";
        sub_CB6200(a1, "!DIObjCProperty(", 16);
      }
      else
      {
        *v69 = _mm_load_si128((const __m128i *)&xmmword_3F24B80);
        *(_QWORD *)(a1 + 32) += 16LL;
      }
      *(_QWORD *)&v260 = a1;
      v70 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v71 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v71 & 2) != 0 )
        v72 = *(__int64 **)(v5 - 32);
      else
        v72 = (__int64 *)(v70 - 8LL * ((v71 >> 2) & 0xF));
      v73 = *v72;
      if ( *v72 )
      {
        v73 = sub_B91420(*v72, a2);
        v75 = v74;
      }
      else
      {
        v75 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v73, v75, 1);
      v76 = *(_BYTE *)(v5 - 16);
      if ( (v76 & 2) != 0 )
        v77 = *(_QWORD *)(v5 - 32);
      else
        v77 = v70 - 8LL * ((v76 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "file", 4u, *(_QWORD *)(v77 + 8), 1);
      sub_A537C0((__int64)&v260, "line", 4u, *(_DWORD *)(v5 + 16), 1);
      v78 = *(_BYTE *)(v5 - 16);
      if ( (v78 & 2) != 0 )
        v79 = *(_QWORD *)(v5 - 32);
      else
        v79 = v70 - 8LL * ((v78 >> 2) & 0xF);
      v80 = *(_QWORD *)(v79 + 24);
      if ( v80 )
      {
        v80 = sub_B91420(*(_QWORD *)(v79 + 24), "line");
        v82 = v81;
      }
      else
      {
        v82 = 0;
      }
      sub_A53660((__int64 *)&v260, "setter", 6u, v80, v82, 1);
      v83 = *(_BYTE *)(v5 - 16);
      if ( (v83 & 2) != 0 )
        v84 = *(_QWORD *)(v5 - 32);
      else
        v84 = v70 - 8LL * ((v83 >> 2) & 0xF);
      v85 = *(_QWORD *)(v84 + 16);
      if ( v85 )
      {
        v85 = sub_B91420(*(_QWORD *)(v84 + 16), "setter");
        v87 = v86;
      }
      else
      {
        v87 = 0;
      }
      sub_A53660((__int64 *)&v260, "getter", 6u, v85, v87, 1);
      sub_A537C0((__int64)&v260, "attributes", 0xAu, *(_DWORD *)(v5 + 20), 1);
      v88 = *(_BYTE *)(v5 - 16);
      if ( (v88 & 2) != 0 )
        v89 = *(_QWORD *)(v5 - 32);
      else
        v89 = v70 - 8LL * ((v88 >> 2) & 0xF);
      v90 = *(_QWORD *)(v89 + 32);
      v91 = 4;
      v92 = "type";
LABEL_86:
      sub_A5CC00((__int64)&v260, v92, v91, v90, 1);
      return (_BYTE *)sub_904010(a1, ")");
    case 0x1D:
      v53 = *(__m128i **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v53 <= 0x11u )
      {
        sub_CB6200(a1, "!DIImportedEntity(", 18);
      }
      else
      {
        v54 = _mm_load_si128(xmmword_3F24B90);
        v53[1].m128i_i16[0] = 10361;
        *v53 = v54;
        *(_QWORD *)(a1 + 32) += 18LL;
      }
      *((_QWORD *)&v261 + 1) = a3;
      v55 = v5 - 16;
      *(_QWORD *)&v260 = a1;
      *(_QWORD *)&v261 = ", ";
      BYTE8(v260) = 1;
      sub_A53560(&v260, v5);
      v56 = *(_BYTE *)(v5 - 16);
      if ( (v56 & 2) != 0 )
        v57 = *(_QWORD *)(v5 - 32);
      else
        v57 = v55 - 8LL * ((v56 >> 2) & 0xF);
      v58 = *(_QWORD *)(v57 + 16);
      if ( v58 )
      {
        v58 = sub_B91420(*(_QWORD *)(v57 + 16), v5);
        v60 = v59;
      }
      else
      {
        v60 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v58, v60, 1);
      v61 = *(_BYTE *)(v5 - 16);
      if ( (v61 & 2) != 0 )
        v62 = *(__int64 **)(v5 - 32);
      else
        v62 = (__int64 *)(v55 - 8LL * ((v61 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "scope", 5u, *v62, 0);
      v63 = *(_BYTE *)(v5 - 16);
      if ( (v63 & 2) != 0 )
        v64 = *(_QWORD *)(v5 - 32);
      else
        v64 = v55 - 8LL * ((v63 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "entity", 6u, *(_QWORD *)(v64 + 8), 1);
      v65 = *(_BYTE *)(v5 - 16);
      if ( (v65 & 2) != 0 )
        v66 = *(_QWORD *)(v5 - 32);
      else
        v66 = v55 - 8LL * ((v65 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "file", 4u, *(_QWORD *)(v66 + 24), 1);
      sub_A537C0((__int64)&v260, "line", 4u, *(_DWORD *)(v5 + 4), 1);
      v67 = *(_BYTE *)(v5 - 16);
      if ( (v67 & 2) != 0 )
        v68 = *(_QWORD *)(v5 - 32);
      else
        v68 = v55 - 8LL * ((v67 >> 2) & 0xF);
      v33 = *(_QWORD *)(v68 + 32);
      v34 = 1;
      v36 = 8;
      v35 = "elements";
      goto LABEL_33;
    case 0x1E:
      return (_BYTE *)sub_904010(a1, "!DIAssignID()");
    case 0x1F:
      v37 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v37) <= 8 )
      {
        sub_CB6200(a1, "!DIMacro(", 9);
        v38 = *(_QWORD *)(a1 + 32);
      }
      else
      {
        *(_BYTE *)(v37 + 8) = 40;
        *(_QWORD *)v37 = 0x6F7263614D494421LL;
        v38 = *(_QWORD *)(a1 + 32) + 9LL;
        *(_QWORD *)(a1 + 32) = v38;
      }
      *(_QWORD *)&v260 = a1;
      *(_QWORD *)&v261 = ", ";
      v39 = *(_QWORD *)(a1 + 24);
      *((_QWORD *)&v261 + 1) = a3;
      BYTE8(v260) = 0;
      if ( (unsigned __int64)(v39 - v38) <= 5 )
      {
        sub_CB6200(a1, "type: ", 6);
      }
      else
      {
        *(_DWORD *)v38 = 1701869940;
        *(_WORD *)(v38 + 4) = 8250;
        *(_QWORD *)(a1 + 32) += 6LL;
      }
      v41 = (const void *)sub_E0C510(*(unsigned __int16 *)(v5 + 2));
      if ( v40 )
        sub_A51340(v260, v41, v40);
      else
        sub_CB59D0(v260, *(unsigned __int16 *)(v5 + 2));
      v42 = v5 - 16;
      sub_A537C0((__int64)&v260, "line", 4u, *(_DWORD *)(v5 + 4), 1);
      v43 = *(_BYTE *)(v5 - 16);
      if ( (v43 & 2) != 0 )
        v44 = *(__int64 **)(v5 - 32);
      else
        v44 = (__int64 *)(v42 - 8LL * ((v43 >> 2) & 0xF));
      v45 = *v44;
      if ( *v44 )
      {
        v45 = sub_B91420(*v44, "line");
        v47 = v46;
      }
      else
      {
        v47 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v45, v47, 1);
      v48 = *(_BYTE *)(v5 - 16);
      if ( (v48 & 2) != 0 )
        v49 = *(_QWORD *)(v5 - 32);
      else
        v49 = v42 - 8LL * ((v48 >> 2) & 0xF);
      v50 = *(_QWORD *)(v49 + 8);
      if ( v50 )
      {
        v50 = sub_B91420(*(_QWORD *)(v49 + 8), "name");
        v52 = v51;
      }
      else
      {
        v52 = 0;
      }
      sub_A53660((__int64 *)&v260, "value", 5u, v50, v52, 1);
      result = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == result )
        goto LABEL_34;
      goto LABEL_24;
    case 0x20:
      v26 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v26 <= 0xCu )
      {
        sub_CB6200(a1, "!DIMacroFile(", 13);
      }
      else
      {
        qmemcpy(v26, "!DIMacroFile(", 13);
        *(_QWORD *)(a1 + 32) += 13LL;
      }
      *((_QWORD *)&v261 + 1) = a3;
      v27 = *(_DWORD *)(v5 + 4);
      *(_QWORD *)&v261 = ", ";
      *(_QWORD *)&v260 = a1;
      v28 = v5 - 16;
      BYTE8(v260) = 1;
      sub_A537C0((__int64)&v260, "line", 4u, v27, 1);
      v29 = *(_BYTE *)(v5 - 16);
      if ( (v29 & 2) != 0 )
        v30 = *(__int64 **)(v5 - 32);
      else
        v30 = (__int64 *)(v28 - 8LL * ((v29 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "file", 4u, *v30, 0);
      v31 = *(_BYTE *)(v5 - 16);
      if ( (v31 & 2) != 0 )
        v32 = *(_QWORD *)(v5 - 32);
      else
        v32 = v28 - 8LL * ((v31 >> 2) & 0xF);
      v33 = *(_QWORD *)(v32 + 8);
      v34 = 1;
      v35 = "nodes";
LABEL_32:
      v36 = 5;
LABEL_33:
      sub_A5CC00((__int64)&v260, v35, v36, v33, v34);
      result = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == result )
        goto LABEL_34;
      goto LABEL_24;
    case 0x21:
      v8 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v8 <= 0xEu )
      {
        sub_CB6200(a1, "!DICommonBlock(", 15);
      }
      else
      {
        qmemcpy(v8, "!DICommonBlock(", 15);
        *(_QWORD *)(a1 + 32) += 15LL;
      }
      *(_QWORD *)&v260 = a1;
      v9 = v5 - 16;
      *(_QWORD *)&v261 = ", ";
      v10 = *(_BYTE *)(v5 - 16);
      BYTE8(v260) = 1;
      *((_QWORD *)&v261 + 1) = a3;
      if ( (v10 & 2) != 0 )
        v11 = *(__int64 **)(v5 - 32);
      else
        v11 = (__int64 *)(v9 - 8LL * ((v10 >> 2) & 0xF));
      sub_A5CC00((__int64)&v260, "scope", 5u, *v11, 0);
      v12 = *(_BYTE *)(v5 - 16);
      if ( (v12 & 2) != 0 )
        v13 = *(_QWORD *)(v5 - 32);
      else
        v13 = v9 - 8LL * ((v12 >> 2) & 0xF);
      sub_A5CC00((__int64)&v260, "declaration", 0xBu, *(_QWORD *)(v13 + 8), 0);
      v14 = *(_BYTE *)(v5 - 16);
      if ( (v14 & 2) != 0 )
        v15 = *(_QWORD *)(v5 - 32);
      else
        v15 = v9 - 8LL * ((v14 >> 2) & 0xF);
      v16 = *(_QWORD *)(v15 + 16);
      if ( v16 )
      {
        v16 = sub_B91420(*(_QWORD *)(v15 + 16), "declaration");
        v18 = v17;
      }
      else
      {
        v18 = 0;
      }
      sub_A53660((__int64 *)&v260, "name", 4u, v16, v18, 1);
      v19 = *(_BYTE *)(v5 - 16);
      if ( (v19 & 2) != 0 )
        v20 = *(_QWORD *)(v5 - 32);
      else
        v20 = v9 - 8LL * ((v19 >> 2) & 0xF);
      v21 = *(_QWORD *)(v20 + 24);
LABEL_21:
      sub_A5CC00((__int64)&v260, "file", 4u, v21, 1);
      v22 = *(_DWORD *)(v5 + 4);
      v23 = 1;
      v24 = "line";
      v25 = 4;
LABEL_22:
      sub_A537C0((__int64)&v260, v24, v25, v22, v23);
LABEL_23:
      result = *(_BYTE **)(a1 + 32);
      if ( *(_BYTE **)(a1 + 24) == result )
      {
LABEL_34:
        result = (_BYTE *)sub_CB6200(a1, ")", 1);
      }
      else
      {
LABEL_24:
        *result = 41;
        ++*(_QWORD *)(a1 + 32);
      }
      break;
    case 0x22:
      result = sub_A5EB50(a1, (const char *)v5, (__int64)a3);
      break;
    case 0x23:
      result = (_BYTE *)sub_A5EDF0(a1, v5, (__int64)a3);
      break;
    case 0x24:
      result = (_BYTE *)sub_A5F0F0(a1, v5, (__int64)a3);
      break;
    default:
      BUG();
  }
  return result;
}
