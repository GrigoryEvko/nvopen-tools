// Function: sub_2822E30
// Address: 0x2822e30
//
__int64 __fastcall sub_2822E30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int16 a5,
        unsigned __int16 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  unsigned __int64 v14; // rsi
  int v15; // eax
  __int64 v16; // r15
  __int64 v17; // rsi
  _QWORD **v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // rax
  unsigned int v21; // r14d
  _DWORD *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  unsigned int v25; // eax
  unsigned __int64 v26; // rdx
  char *v27; // rdx
  unsigned int v28; // edx
  int v29; // eax
  unsigned int v30; // edx
  unsigned __int64 v31; // r12
  int v32; // eax
  unsigned __int64 v33; // r12
  __int64 v34; // rax
  unsigned __int8 *v35; // rax
  size_t v36; // rax
  char v37; // al
  __int64 v38; // rdx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 *v45; // rdx
  __int64 *v46; // rax
  _QWORD *v47; // r15
  __int64 v48; // rax
  int v49; // r14d
  unsigned __int64 v50; // r12
  int v51; // eax
  unsigned __int64 v52; // r12
  __int64 v53; // rax
  unsigned __int8 *v54; // rax
  __int64 v55; // r14
  unsigned __int8 *v56; // r15
  unsigned __int8 *v57; // r15
  bool v58; // al
  __int64 *v59; // r14
  _QWORD *v60; // r15
  __int64 *v61; // r13
  unsigned __int64 v62; // rax
  _QWORD *v63; // r13
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __m128i v67; // xmm1
  __int64 v68; // rsi
  __int64 v69; // rdi
  __int64 *v70; // r13
  __int64 *v71; // r14
  __int64 v72; // rsi
  _QWORD *v73; // rdi
  __int64 *v74; // rsi
  __int64 *v75; // r15
  __int64 v76; // rax
  __int64 v77; // rcx
  __int64 v78; // r12
  __int64 v79; // r12
  unsigned __int8 *v80; // rax
  __int64 v81; // r12
  char *v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r12
  char *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // r12
  __m128i v92; // xmm5
  __m128i v93; // xmm7
  unsigned __int64 *v94; // r12
  __int64 v95; // r8
  unsigned __int64 *v96; // r14
  unsigned __int64 v97; // rdi
  __int64 v98; // r9
  unsigned __int64 *v99; // r12
  unsigned __int64 *v100; // r13
  unsigned __int64 v101; // rdi
  _QWORD *v102; // rdi
  int v103; // edx
  __int64 v104; // rcx
  int v105; // edx
  unsigned int v106; // esi
  __int64 *v107; // rax
  __int64 v108; // r8
  __int64 v109; // rsi
  __int64 *v110; // rax
  __int64 v111; // rax
  bool v112; // al
  char v113; // r15
  __int64 v114; // rax
  const char *v115; // rax
  __int64 *v116; // r15
  __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // r12
  unsigned __int8 *v120; // rax
  __int64 v121; // r12
  __int64 v122; // rcx
  __int64 v123; // rbx
  __int64 v124; // r8
  __int64 v125; // r9
  __m128i v126; // xmm2
  __m128i v127; // xmm4
  __int64 v128; // rdx
  unsigned __int64 *v129; // rbx
  unsigned __int64 *v130; // r12
  unsigned __int64 v131; // rdi
  unsigned __int64 *v132; // rbx
  unsigned __int64 v133; // r12
  unsigned __int64 v134; // rdi
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 *v137; // rdx
  __int64 *v138; // rax
  __int64 v139; // rsi
  unsigned __int8 *v140; // rsi
  bool v141; // al
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 *v144; // rax
  unsigned __int64 v145; // rax
  unsigned __int64 v146; // rax
  __int64 *v147; // r12
  __int64 v148; // r13
  __int64 v149; // rax
  __int64 v150; // r13
  unsigned __int8 *v151; // rax
  __int64 v152; // r13
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // rbx
  __int64 v156; // r8
  __int64 v157; // r9
  __m128i v158; // xmm5
  __m128i v159; // xmm7
  unsigned __int64 *v160; // r15
  unsigned __int64 *v161; // rbx
  unsigned __int64 v162; // rdi
  unsigned __int64 *v163; // rbx
  unsigned __int64 v164; // rdi
  __int64 *v165; // rax
  int v166; // eax
  __int64 v167; // rax
  __int64 v168; // rax
  __int64 v169; // [rsp-10h] [rbp-A90h]
  __int64 v170; // [rsp-8h] [rbp-A88h]
  __int64 v171; // [rsp-8h] [rbp-A88h]
  __int64 v172; // [rsp+0h] [rbp-A80h]
  __int64 v173; // [rsp+8h] [rbp-A78h]
  unsigned __int8 *v174; // [rsp+10h] [rbp-A70h]
  __int64 v175; // [rsp+20h] [rbp-A60h]
  __int64 v176; // [rsp+48h] [rbp-A38h]
  __int64 v177; // [rsp+50h] [rbp-A30h]
  bool v178; // [rsp+59h] [rbp-A27h]
  char v179; // [rsp+59h] [rbp-A27h]
  bool v182; // [rsp+5Fh] [rbp-A21h]
  __int64 v183; // [rsp+60h] [rbp-A20h]
  unsigned __int8 *v184; // [rsp+60h] [rbp-A20h]
  __int64 v186; // [rsp+68h] [rbp-A18h]
  unsigned __int8 *v187; // [rsp+70h] [rbp-A10h]
  __int64 v188; // [rsp+80h] [rbp-A00h]
  char *s; // [rsp+88h] [rbp-9F8h]
  char *sa; // [rsp+88h] [rbp-9F8h]
  unsigned int v191; // [rsp+90h] [rbp-9F0h]
  bool v192; // [rsp+90h] [rbp-9F0h]
  __int64 v193; // [rsp+90h] [rbp-9F0h]
  __m128i v194; // [rsp+A0h] [rbp-9E0h] BYREF
  __m128i v195; // [rsp+B0h] [rbp-9D0h] BYREF
  _BYTE *v196; // [rsp+C0h] [rbp-9C0h] BYREF
  char v197; // [rsp+C8h] [rbp-9B8h]
  unsigned __int64 v198; // [rsp+D0h] [rbp-9B0h] BYREF
  unsigned int v199; // [rsp+D8h] [rbp-9A8h]
  __m128i v200; // [rsp+E0h] [rbp-9A0h] BYREF
  __m128i v201; // [rsp+F0h] [rbp-990h] BYREF
  __m128i v202; // [rsp+100h] [rbp-980h]
  __int64 v203[4]; // [rsp+110h] [rbp-970h] BYREF
  __int64 v204; // [rsp+130h] [rbp-950h] BYREF
  __int64 *v205; // [rsp+138h] [rbp-948h]
  __int64 v206; // [rsp+140h] [rbp-940h]
  int v207; // [rsp+148h] [rbp-938h]
  char v208; // [rsp+14Ch] [rbp-934h]
  __int64 v209; // [rsp+150h] [rbp-930h] BYREF
  unsigned __int64 v210[2]; // [rsp+160h] [rbp-920h] BYREF
  __int64 v211; // [rsp+170h] [rbp-910h] BYREF
  __int64 *v212; // [rsp+180h] [rbp-900h]
  __int64 v213; // [rsp+190h] [rbp-8F0h] BYREF
  unsigned __int64 v214[2]; // [rsp+1B0h] [rbp-8D0h] BYREF
  __int64 v215; // [rsp+1C0h] [rbp-8C0h] BYREF
  __int64 *v216; // [rsp+1D0h] [rbp-8B0h]
  __int64 v217; // [rsp+1E0h] [rbp-8A0h] BYREF
  unsigned __int64 v218[2]; // [rsp+200h] [rbp-880h] BYREF
  _QWORD v219[2]; // [rsp+210h] [rbp-870h] BYREF
  _QWORD *v220; // [rsp+220h] [rbp-860h]
  _QWORD v221[4]; // [rsp+230h] [rbp-850h] BYREF
  unsigned __int64 v222[2]; // [rsp+250h] [rbp-830h] BYREF
  _QWORD v223[2]; // [rsp+260h] [rbp-820h] BYREF
  _QWORD *v224; // [rsp+270h] [rbp-810h]
  _QWORD v225[4]; // [rsp+280h] [rbp-800h] BYREF
  unsigned __int64 v226[2]; // [rsp+2A0h] [rbp-7E0h] BYREF
  _QWORD v227[2]; // [rsp+2B0h] [rbp-7D0h] BYREF
  _QWORD *v228; // [rsp+2C0h] [rbp-7C0h]
  _QWORD v229[4]; // [rsp+2D0h] [rbp-7B0h] BYREF
  unsigned __int64 v230[2]; // [rsp+2F0h] [rbp-790h] BYREF
  _BYTE v231[32]; // [rsp+300h] [rbp-780h] BYREF
  __int64 v232; // [rsp+320h] [rbp-760h]
  __int64 v233; // [rsp+328h] [rbp-758h]
  __int16 v234; // [rsp+330h] [rbp-750h]
  __int64 *v235; // [rsp+338h] [rbp-748h]
  void **v236; // [rsp+340h] [rbp-740h]
  void **v237; // [rsp+348h] [rbp-738h]
  __int64 v238; // [rsp+350h] [rbp-730h]
  int v239; // [rsp+358h] [rbp-728h]
  __int16 v240; // [rsp+35Ch] [rbp-724h]
  char v241; // [rsp+35Eh] [rbp-722h]
  __int64 v242; // [rsp+360h] [rbp-720h]
  __int64 v243; // [rsp+368h] [rbp-718h]
  void *v244; // [rsp+370h] [rbp-710h] BYREF
  void *v245; // [rsp+378h] [rbp-708h] BYREF
  char *v246; // [rsp+380h] [rbp-700h] BYREF
  unsigned int v247; // [rsp+388h] [rbp-6F8h]
  char v248; // [rsp+38Ch] [rbp-6F4h]
  __int64 v249; // [rsp+390h] [rbp-6F0h]
  __m128i v250; // [rsp+398h] [rbp-6E8h]
  __int64 v251; // [rsp+3A8h] [rbp-6D8h]
  __m128i v252; // [rsp+3B0h] [rbp-6D0h]
  __m128i v253; // [rsp+3C0h] [rbp-6C0h]
  unsigned __int64 *v254; // [rsp+3D0h] [rbp-6B0h] BYREF
  __int64 v255; // [rsp+3D8h] [rbp-6A8h]
  _BYTE v256[320]; // [rsp+3E0h] [rbp-6A0h] BYREF
  char v257; // [rsp+520h] [rbp-560h]
  int v258; // [rsp+524h] [rbp-55Ch]
  __int64 v259; // [rsp+528h] [rbp-558h]
  unsigned __int64 v260; // [rsp+530h] [rbp-550h] BYREF
  __int64 v261; // [rsp+538h] [rbp-548h]
  _QWORD v262[8]; // [rsp+540h] [rbp-540h] BYREF
  unsigned __int64 *v263; // [rsp+580h] [rbp-500h]
  unsigned int v264; // [rsp+588h] [rbp-4F8h]
  _BYTE v265[336]; // [rsp+590h] [rbp-4F0h] BYREF
  _BYTE v266[928]; // [rsp+6E0h] [rbp-3A0h] BYREF

  if ( *(_BYTE *)a7 == 85 )
  {
    v111 = *(_QWORD *)(a7 - 32);
    if ( v111 )
    {
      if ( !*(_BYTE *)v111
        && *(_QWORD *)(v111 + 24) == *(_QWORD *)(a7 + 80)
        && (*(_BYTE *)(v111 + 33) & 0x20) != 0
        && *(_DWORD *)(v111 + 36) == 240 )
      {
        return 0;
      }
    }
  }
  v188 = sub_D4B130(*(_QWORD *)a1);
  v183 = v188 + 48;
  v14 = *(_QWORD *)(v188 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v188 + 48 == v14 )
  {
    v16 = 0;
  }
  else
  {
    if ( !v14 )
      BUG();
    v15 = *(unsigned __int8 *)(v14 - 24);
    v16 = 0;
    v17 = v14 - 24;
    if ( (unsigned int)(v15 - 30) < 0xB )
      v16 = v17;
  }
  v235 = (__int64 *)sub_BD5C60(v16);
  v236 = &v244;
  v237 = &v245;
  v230[0] = (unsigned __int64)v231;
  v244 = &unk_49DA100;
  v230[1] = 0x200000000LL;
  v240 = 512;
  v234 = 0;
  v245 = &unk_49DA0B0;
  v238 = 0;
  v239 = 0;
  v241 = 7;
  v242 = 0;
  v243 = 0;
  v232 = 0;
  v233 = 0;
  sub_D5F1F0((__int64)v230, v16);
  sub_27C1C30((__int64)v266, *(__int64 **)(a1 + 32), *(_QWORD *)(a1 + 56), (__int64)"loop-idiom", 1);
  v18 = *(_QWORD ***)(a9 + 32);
  v197 = 0;
  v196 = v266;
  v19 = *v18;
  v20 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
    v20 = **(_QWORD **)(v20 + 16);
  v21 = *(_DWORD *)(v20 + 8) >> 8;
  v22 = sub_AE2980(*(_QWORD *)(a1 + 56), v21);
  v176 = sub_BCD140(v235, v22[3]);
  v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a9 + 32) + 8LL) + 32LL);
  v199 = *(_DWORD *)(v23 + 32);
  if ( v199 > 0x40 )
    sub_C43780((__int64)&v198, (const void **)(v23 + 24));
  else
    v198 = *(_QWORD *)(v23 + 24);
  if ( *(_WORD *)(a4 + 24) )
    BUG();
  v24 = *(_QWORD *)(a4 + 32);
  if ( *(_DWORD *)(v24 + 32) <= 0x40u )
    v177 = *(_QWORD *)(v24 + 24);
  else
    v177 = **(_QWORD **)(v24 + 24);
  v25 = v199;
  v247 = v199;
  if ( v199 > 0x40 )
  {
    sub_C43780((__int64)&v246, (const void **)&v198);
    v25 = v247;
    if ( v247 > 0x40 )
    {
      sub_C43D10((__int64)&v246);
      goto LABEL_18;
    }
    v26 = (unsigned __int64)v246;
  }
  else
  {
    v26 = v198;
  }
  v27 = (char *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v25) & ~v26);
  if ( !v25 )
    v27 = 0;
  v246 = v27;
LABEL_18:
  sub_C46250((__int64)&v246);
  v28 = v247;
  v247 = 0;
  LODWORD(v261) = v28;
  v260 = (unsigned __int64)v246;
  v191 = v28;
  if ( v28 > 0x40 )
  {
    s = v246;
    v29 = sub_C444A0((__int64)&v260);
    v30 = v191;
    v192 = 0;
    if ( v30 - v29 <= 0x40 )
      v192 = *(_QWORD *)s == v177;
    if ( s )
    {
      j_j___libc_free_0_0((unsigned __int64)s);
      if ( v247 > 0x40 )
      {
        if ( v246 )
          j_j___libc_free_0_0((unsigned __int64)v246);
      }
    }
    if ( !v192 )
      goto LABEL_26;
    goto LABEL_151;
  }
  v192 = v177 == (_QWORD)v246;
  if ( (char *)v177 == v246 )
LABEL_151:
    v19 = sub_281DF50((__int64)v19, a11, v176, a4, *(__int64 **)(a1 + 32));
LABEL_26:
  v31 = *(_QWORD *)(v188 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v183 == v31 )
  {
    v33 = 0;
  }
  else
  {
    if ( !v31 )
      BUG();
    v32 = *(unsigned __int8 *)(v31 - 24);
    v33 = v31 - 24;
    if ( (unsigned int)(v32 - 30) >= 0xB )
      v33 = 0;
  }
  v34 = sub_BCE3C0(v235, v21);
  v35 = (unsigned __int8 *)sub_F8DB90((__int64)v266, (__int64)v19, v34, v33 + 24, 0);
  v208 = 1;
  v187 = v35;
  v205 = &v209;
  v206 = 0x100000002LL;
  v207 = 0;
  v209 = a7;
  sa = "load and store";
  v204 = 1;
  v182 = 0;
  if ( *(_BYTE *)a7 == 85
    && (v114 = *(_QWORD *)(a7 - 32)) != 0
    && !*(_BYTE *)v114
    && *(_QWORD *)(v114 + 24) == *(_QWORD *)(a7 + 80)
    && (*(_BYTE *)(v114 + 33) & 0x20) != 0 )
  {
    v182 = ((*(_DWORD *)(v114 + 36) - 238) & 0xFFFFFFFD) == 0;
    v115 = "memcpy";
    if ( !v182 )
      v115 = "load and store";
    sa = (char *)v115;
    v36 = strlen(v115);
  }
  else
  {
    v36 = strlen("load and store");
  }
  v175 = v36;
  v37 = sub_281E0D0((__int64)v187, 3u, *(_QWORD *)a1, a11, a4, *(_QWORD **)(a1 + 8), (__int64)&v204);
  v41 = v169;
  v42 = v170;
  v178 = v37;
  if ( !v37 )
    goto LABEL_50;
  v43 = *(_QWORD *)(a8 + 16);
  if ( !v43 || *(_QWORD *)(v43 + 8) )
    goto LABEL_34;
  if ( !v208 )
    goto LABEL_42;
  v110 = v205;
  v41 = HIDWORD(v206);
  v38 = (__int64)&v205[HIDWORD(v206)];
  if ( v205 != (__int64 *)v38 )
  {
    while ( *v110 != a8 )
    {
      if ( (__int64 *)v38 == ++v110 )
        goto LABEL_259;
    }
    goto LABEL_43;
  }
LABEL_259:
  if ( HIDWORD(v206) < (unsigned int)v206 )
  {
    ++HIDWORD(v206);
    *(_QWORD *)v38 = a8;
    ++v204;
  }
  else
  {
LABEL_42:
    sub_C8CC70((__int64)&v204, a8, v38, v41, v39, v40);
  }
LABEL_43:
  v42 = 3;
  if ( (unsigned __int8)sub_281E0D0((__int64)v187, 3u, *(_QWORD *)a1, a11, a4, *(_QWORD **)(a1 + 8), (__int64)&v204) )
  {
    v147 = *(__int64 **)(a1 + 64);
    v148 = *v147;
    v149 = sub_B2BE50(*v147);
    if ( !sub_B6EA50(v149) )
    {
      v167 = sub_B2BE50(v148);
      v168 = sub_B6F970(v167);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v168 + 48LL))(v168) )
        goto LABEL_34;
    }
    sub_B176B0((__int64)&v260, (__int64)"loop-idiom", (__int64)"LoopMayAccessStore", 18, a7);
    sub_B16430((__int64)v226, "Inst", 4u, sa, v175);
    v150 = sub_2820EB0((__int64)&v260, (__int64)v226);
    sub_B18290(v150, " in ", 4u);
    v151 = (unsigned __int8 *)sub_B43CB0(a7);
    sub_B16080((__int64)v222, "Function", 8, v151);
    v152 = sub_2445430(v150, (__int64)v222);
    sub_B18290(v152, " function will not be hoisted: ", 0x1Fu);
    sub_B16430((__int64)v218, "Reason", 6u, "The loop may access store location", 34);
    v155 = sub_2445430(v152, (__int64)v218);
    v247 = *(_DWORD *)(v155 + 8);
    v248 = *(_BYTE *)(v155 + 12);
    v249 = *(_QWORD *)(v155 + 16);
    v158 = _mm_loadu_si128((const __m128i *)(v155 + 24));
    v246 = (char *)&unk_49D9D40;
    v250 = v158;
    v251 = *(_QWORD *)(v155 + 40);
    v252 = _mm_loadu_si128((const __m128i *)(v155 + 48));
    v159 = _mm_loadu_si128((const __m128i *)(v155 + 64));
    v254 = (unsigned __int64 *)v256;
    v255 = 0x400000000LL;
    v253 = v159;
    if ( *(_DWORD *)(v155 + 88) )
      sub_2822780((__int64)&v254, v155 + 80, v153, v154, v156, v157);
    v257 = *(_BYTE *)(v155 + 416);
    v258 = *(_DWORD *)(v155 + 420);
    v259 = *(_QWORD *)(v155 + 424);
    v246 = (char *)&unk_49D9DB0;
    if ( v220 != v221 )
      j_j___libc_free_0((unsigned __int64)v220);
    if ( (_QWORD *)v218[0] != v219 )
      j_j___libc_free_0(v218[0]);
    if ( v224 != v225 )
      j_j___libc_free_0((unsigned __int64)v224);
    if ( (_QWORD *)v222[0] != v223 )
      j_j___libc_free_0(v222[0]);
    if ( v228 != v229 )
      j_j___libc_free_0((unsigned __int64)v228);
    if ( (_QWORD *)v226[0] != v227 )
      j_j___libc_free_0(v226[0]);
    v160 = v263;
    v260 = (unsigned __int64)&unk_49D9D40;
    v161 = &v263[10 * v264];
    if ( v263 != v161 )
    {
      do
      {
        v161 -= 10;
        v162 = v161[4];
        if ( (unsigned __int64 *)v162 != v161 + 6 )
          j_j___libc_free_0(v162);
        if ( (unsigned __int64 *)*v161 != v161 + 2 )
          j_j___libc_free_0(*v161);
      }
      while ( v160 != v161 );
      v160 = v263;
    }
    if ( v160 != (unsigned __int64 *)v265 )
      _libc_free((unsigned __int64)v160);
    v42 = (__int64)&v246;
    sub_1049740(v147, (__int64)&v246);
    v133 = (unsigned __int64)v254;
    v246 = (char *)&unk_49D9D40;
    v163 = &v254[10 * (unsigned int)v255];
    if ( v254 != v163 )
    {
      do
      {
        v163 -= 10;
        v164 = v163[4];
        if ( (unsigned __int64 *)v164 != v163 + 6 )
        {
          v42 = v163[6] + 1;
          j_j___libc_free_0(v164);
        }
        if ( (unsigned __int64 *)*v163 != v163 + 2 )
        {
          v42 = v163[2] + 1;
          j_j___libc_free_0(*v163);
        }
      }
      while ( (unsigned __int64 *)v133 != v163 );
      goto LABEL_218;
    }
LABEL_219:
    if ( (_BYTE *)v133 != v256 )
      _libc_free(v133);
    goto LABEL_34;
  }
  if ( v208 )
  {
    v45 = &v205[HIDWORD(v206)];
    if ( v205 != v45 )
    {
      v46 = v205;
      while ( *v46 != a8 )
      {
        if ( v45 == ++v46 )
          goto LABEL_50;
      }
      --HIDWORD(v206);
      *v46 = v205[HIDWORD(v206)];
      ++v204;
    }
  }
  else
  {
    v165 = sub_C8CA60((__int64)&v204, a8);
    if ( v165 )
    {
      *v165 = -2;
      ++v207;
      ++v204;
    }
  }
LABEL_50:
  v47 = **(_QWORD ***)(a10 + 32);
  v48 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v48 + 8) - 17 <= 1 )
    v48 = **(_QWORD **)(v48 + 16);
  v49 = *(_DWORD *)(v48 + 8) >> 8;
  if ( v192 )
    v47 = sub_281DF50(**(_QWORD **)(a10 + 32), a11, v176, a4, *(__int64 **)(a1 + 32));
  v50 = *(_QWORD *)(v188 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v183 == v50 )
  {
    v52 = 0;
  }
  else
  {
    if ( !v50 )
      BUG();
    v51 = *(unsigned __int8 *)(v50 - 24);
    v52 = v50 - 24;
    if ( (unsigned int)(v51 - 30) >= 0xB )
      v52 = 0;
  }
  v53 = sub_BCE3C0(v235, v49);
  v54 = (unsigned __int8 *)sub_F8DB90((__int64)v266, (__int64)v47, v53, v52 + 24, 0);
  v55 = *(_QWORD *)(a1 + 56);
  v186 = (__int64)v54;
  v56 = sub_BD3990(v54, (__int64)v47);
  LODWORD(v261) = sub_AE43F0(v55, *((_QWORD *)v56 + 1));
  if ( (unsigned int)v261 > 0x40 )
    sub_C43690((__int64)&v260, 0, 0);
  else
    v260 = 0;
  v174 = sub_BD45C0(v56, v55, (__int64)&v260, 1, 0, 0, 0, 0);
  if ( (unsigned int)v261 > 0x40 )
  {
    v172 = *(_QWORD *)v260;
    j_j___libc_free_0_0(v260);
  }
  else
  {
    v172 = 0;
    if ( (_DWORD)v261 )
      v172 = (__int64)(v260 << (64 - (unsigned __int8)v261)) >> (64 - (unsigned __int8)v261);
  }
  v57 = sub_BD3990(v187, v55);
  LODWORD(v261) = sub_AE43F0(v55, *((_QWORD *)v57 + 1));
  if ( (unsigned int)v261 > 0x40 )
    sub_C43690((__int64)&v260, 0, 0);
  else
    v260 = 0;
  v184 = sub_BD45C0(v57, v55, (__int64)&v260, 1, 0, 0, 0, 0);
  if ( (unsigned int)v261 > 0x40 )
  {
    v173 = *(_QWORD *)v260;
    j_j___libc_free_0_0(v260);
  }
  else
  {
    v173 = 0;
    if ( (_DWORD)v261 )
      v173 = (__int64)(v260 << (64 - (unsigned __int8)v261)) >> (64 - (unsigned __int8)v261);
  }
  if ( v182 && v184 != v174 )
  {
    if ( v208 )
    {
      v137 = &v205[HIDWORD(v206)];
      if ( v205 != v137 )
      {
        v138 = v205;
        while ( *v138 != a7 )
        {
          if ( v137 == ++v138 )
            goto LABEL_70;
        }
        --HIDWORD(v206);
        *v138 = v205[HIDWORD(v206)];
        ++v204;
      }
    }
    else
    {
      v144 = sub_C8CA60((__int64)&v204, a7);
      if ( v144 )
      {
        *v144 = -2;
        ++v207;
        ++v204;
      }
    }
  }
LABEL_70:
  v42 = v171;
  if ( (unsigned __int8)sub_281E0D0(v186, 2u, *(_QWORD *)a1, a11, a4, *(_QWORD **)(a1 + 8), (__int64)&v204) )
  {
    v116 = *(__int64 **)(a1 + 64);
    v117 = *v116;
    v118 = sub_B2BE50(*v116);
    if ( !sub_B6EA50(v118) )
    {
      v135 = sub_B2BE50(v117);
      v136 = sub_B6F970(v135);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v136 + 48LL))(v136, v171) )
        goto LABEL_34;
    }
    sub_B176B0((__int64)&v260, (__int64)"loop-idiom", (__int64)"LoopMayAccessLoad", 17, a8);
    sub_B16430((__int64)v226, "Inst", 4u, sa, v175);
    v119 = sub_2820EB0((__int64)&v260, (__int64)v226);
    sub_B18290(v119, " in ", 4u);
    v120 = (unsigned __int8 *)sub_B43CB0(a7);
    sub_B16080((__int64)v222, "Function", 8, v120);
    v121 = sub_2445430(v119, (__int64)v222);
    sub_B18290(v121, " function will not be hoisted: ", 0x1Fu);
    sub_B16430((__int64)v218, "Reason", 6u, "The loop may access load location", 33);
    v123 = sub_2445430(v121, (__int64)v218);
    v247 = *(_DWORD *)(v123 + 8);
    v248 = *(_BYTE *)(v123 + 12);
    v249 = *(_QWORD *)(v123 + 16);
    v126 = _mm_loadu_si128((const __m128i *)(v123 + 24));
    v246 = (char *)&unk_49D9D40;
    v250 = v126;
    v251 = *(_QWORD *)(v123 + 40);
    v252 = _mm_loadu_si128((const __m128i *)(v123 + 48));
    v127 = _mm_loadu_si128((const __m128i *)(v123 + 64));
    v254 = (unsigned __int64 *)v256;
    v255 = 0x400000000LL;
    v253 = v127;
    v128 = *(unsigned int *)(v123 + 88);
    if ( (_DWORD)v128 )
      sub_2822780((__int64)&v254, v123 + 80, v128, v122, v124, v125);
    v257 = *(_BYTE *)(v123 + 416);
    v258 = *(_DWORD *)(v123 + 420);
    v259 = *(_QWORD *)(v123 + 424);
    v246 = (char *)&unk_49D9DB0;
    if ( v220 != v221 )
      j_j___libc_free_0((unsigned __int64)v220);
    if ( (_QWORD *)v218[0] != v219 )
      j_j___libc_free_0(v218[0]);
    if ( v224 != v225 )
      j_j___libc_free_0((unsigned __int64)v224);
    if ( (_QWORD *)v222[0] != v223 )
      j_j___libc_free_0(v222[0]);
    if ( v228 != v229 )
      j_j___libc_free_0((unsigned __int64)v228);
    if ( (_QWORD *)v226[0] != v227 )
      j_j___libc_free_0(v226[0]);
    v129 = v263;
    v260 = (unsigned __int64)&unk_49D9D40;
    v130 = &v263[10 * v264];
    if ( v263 != v130 )
    {
      do
      {
        v130 -= 10;
        v131 = v130[4];
        if ( (unsigned __int64 *)v131 != v130 + 6 )
          j_j___libc_free_0(v131);
        if ( (unsigned __int64 *)*v130 != v130 + 2 )
          j_j___libc_free_0(*v130);
      }
      while ( v129 != v130 );
      v130 = v263;
    }
    if ( v130 != (unsigned __int64 *)v265 )
      _libc_free((unsigned __int64)v130);
    v42 = (__int64)&v246;
    sub_1049740(v116, (__int64)&v246);
    v132 = v254;
    v246 = (char *)&unk_49D9D40;
    v133 = (unsigned __int64)&v254[10 * (unsigned int)v255];
    if ( v254 != (unsigned __int64 *)v133 )
    {
      do
      {
        v133 -= 80LL;
        v134 = *(_QWORD *)(v133 + 32);
        if ( v134 != v133 + 48 )
        {
          v42 = *(_QWORD *)(v133 + 48) + 1LL;
          j_j___libc_free_0(v134);
        }
        if ( *(_QWORD *)v133 != v133 + 16 )
        {
          v42 = *(_QWORD *)(v133 + 16) + 1LL;
          j_j___libc_free_0(*(_QWORD *)v133);
        }
      }
      while ( v132 != (unsigned __int64 *)v133 );
LABEL_218:
      v133 = (unsigned __int64)v254;
      goto LABEL_219;
    }
    goto LABEL_219;
  }
  if ( sub_B46500((unsigned __int8 *)a7) )
  {
    v112 = v178;
    if ( v182 )
      v112 = v184 == v174;
    v113 = v112;
    goto LABEL_175;
  }
  v58 = sub_B46500((unsigned __int8 *)a8);
  v38 = v178;
  if ( v182 )
  {
    LOBYTE(v57) = v184 == v174;
    v38 = (unsigned int)v57;
  }
  v179 = v38;
  if ( v58 )
  {
    v113 = v38;
LABEL_175:
    if ( v113 )
      goto LABEL_34;
    v41 = (unsigned __int8)a5;
    v38 = 1LL << a5;
    if ( v177 > (unsigned __int64)(1LL << a5) )
      goto LABEL_34;
    v41 = (unsigned __int8)a6;
    if ( v177 > (unsigned __int64)(1LL << a6) || (unsigned int)sub_DFDD80(*(_QWORD *)(a1 + 48)) < v177 )
      goto LABEL_34;
    v179 = 0;
    v192 = 1;
    goto LABEL_77;
  }
  if ( (_BYTE)v38 )
  {
    if ( v182 )
    {
      if ( !v192 )
      {
        v38 = v172;
        if ( v173 >= v172 )
          goto LABEL_34;
        v179 = v182;
        goto LABEL_77;
      }
      v41 = v172;
      if ( v173 <= v172 )
        goto LABEL_34;
    }
    else
    {
      v42 = *(_QWORD *)(a8 + 8);
      v145 = sub_9208B0(v55, v42);
      v41 = (__int64)v174;
      v260 = v145;
      v261 = v38;
      if ( v184 != v174 )
        goto LABEL_34;
      v38 = (unsigned int)v177;
      v146 = v145 >> 3;
      if ( v146 != (unsigned int)v177 )
        goto LABEL_34;
      if ( !v192 )
      {
        if ( (__int64)(v173 + v146) > v172 )
          goto LABEL_34;
        goto LABEL_77;
      }
      if ( v173 < (__int64)(v172 + v146) )
        goto LABEL_34;
    }
    v141 = v192;
    v192 = 0;
    v179 = v141;
  }
  else
  {
    v192 = 0;
  }
LABEL_77:
  v41 = *(_QWORD *)a1;
  if ( !*(_BYTE *)(a1 + 72)
    || (unsigned int)((__int64)(*(_QWORD *)(v41 + 40) - *(_QWORD *)(v41 + 32)) >> 3) <= 1
    || *(_QWORD *)v41 )
  {
    v59 = *(__int64 **)(a1 + 32);
    v60 = sub_DE5A20(v59, a11, v176, v41);
    v262[1] = sub_DC5760((__int64)v59, a4, v176, 0);
    v260 = (unsigned __int64)v262;
    v262[0] = v60;
    v261 = 0x200000002LL;
    v61 = sub_DC8BD0(v59, (__int64)&v260, 2u, 0);
    if ( (_QWORD *)v260 != v262 )
      _libc_free(v260);
    v62 = sub_986580(v188);
    v63 = sub_F8DB90((__int64)v266, (__int64)v61, v176, v62 + 24, 0);
    sub_B91FC0(v201.m128i_i64, a8);
    sub_B91FC0(v203, a7);
    sub_E01E30(v194.m128i_i64, v201.m128i_i64, v203, v64, v65, v66);
    v67 = _mm_loadu_si128(&v195);
    v201 = _mm_loadu_si128(&v194);
    v202 = v67;
    if ( *(_BYTE *)v63 == 17 )
    {
      v68 = v63[3];
      if ( *((_DWORD *)v63 + 8) > 0x40u )
        v68 = *(_QWORD *)v68;
      v69 = v201.m128i_i64[0];
      if ( v201.m128i_i64[0] )
        goto LABEL_86;
    }
    else
    {
      v69 = v201.m128i_i64[0];
      if ( v201.m128i_i64[0] )
      {
        v68 = -1;
LABEL_86:
        v69 = sub_E00A50(v69, v68);
      }
    }
    v201.m128i_i64[0] = v69;
    if ( v192 )
    {
      v70 = (__int64 *)sub_B34600(
                         (__int64)v230,
                         (__int64)v187,
                         (unsigned __int8)a5,
                         v186,
                         (unsigned __int8)a6,
                         (__int64)v63,
                         v177,
                         v69,
                         v201.m128i_i64[1],
                         v202.m128i_i64[0],
                         v202.m128i_i64[1]);
    }
    else if ( v179 )
    {
      v70 = (__int64 *)sub_B343C0(
                         (__int64)v230,
                         0xF1u,
                         (__int64)v187,
                         a5,
                         v186,
                         a6,
                         (__int64)v63,
                         0,
                         v69,
                         0,
                         v202.m128i_i64[0],
                         v202.m128i_i64[1]);
    }
    else
    {
      v70 = (__int64 *)sub_B343C0(
                         (__int64)v230,
                         0xEEu,
                         (__int64)v187,
                         a5,
                         v186,
                         a6,
                         (__int64)v63,
                         0,
                         v69,
                         v201.m128i_i64[1],
                         v202.m128i_i64[0],
                         v202.m128i_i64[1]);
    }
    v71 = v70 + 6;
    v72 = *(_QWORD *)(a7 + 48);
    v260 = v72;
    if ( v72 )
    {
      sub_B96E90((__int64)&v260, v72, 1);
      if ( v71 == (__int64 *)&v260 )
      {
        if ( v260 )
          sub_B91220((__int64)&v260, v260);
        goto LABEL_94;
      }
      v139 = v70[6];
      if ( !v139 )
      {
LABEL_236:
        v140 = (unsigned __int8 *)v260;
        v70[6] = v260;
        if ( v140 )
          sub_B976B0((__int64)&v260, v140, (__int64)(v70 + 6));
        goto LABEL_94;
      }
    }
    else if ( v71 == (__int64 *)&v260 || (v139 = v70[6]) == 0 )
    {
LABEL_94:
      v73 = *(_QWORD **)(a1 + 80);
      if ( v73 )
      {
        v74 = (__int64 *)sub_D694D0(v73, (__int64)v70, 0, v70[5], 2u, 1u);
        sub_D75120(*(__int64 **)(a1 + 80), v74, 1);
      }
      v75 = *(__int64 **)(a1 + 64);
      v193 = *v75;
      v76 = sub_B2BE50(*v75);
      if ( sub_B6EA50(v76)
        || (v142 = sub_B2BE50(v193),
            v143 = sub_B6F970(v142),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v143 + 48LL))(v143)) )
      {
        sub_B157E0((__int64)&v200, v70 + 6);
        sub_B17430((__int64)&v260, (__int64)"loop-idiom", (__int64)"ProcessLoopStoreOfLoopLoad", 26, &v200, v188);
        sub_B18290((__int64)&v260, "Formed a call to ", 0x11u);
        v77 = *(v70 - 4);
        if ( v77 )
        {
          if ( *(_BYTE *)v77 )
          {
            v77 = 0;
          }
          else if ( *(_QWORD *)(v77 + 24) != v70[10] )
          {
            v77 = 0;
          }
        }
        sub_B16080((__int64)v226, "NewFunction", 11, (unsigned __int8 *)v77);
        v78 = sub_23FD640((__int64)&v260, (__int64)v226);
        sub_B18290(v78, "() intrinsic from ", 0x12u);
        sub_B16430((__int64)v222, "Inst", 4u, sa, v175);
        v79 = sub_23FD640(v78, (__int64)v222);
        sub_B18290(v79, " instruction in ", 0x10u);
        v80 = (unsigned __int8 *)sub_B43CB0(a7);
        sub_B16080((__int64)v218, "Function", 8, v80);
        v81 = sub_23FD640(v79, (__int64)v218);
        sub_B18290(v81, " function", 9u);
        sub_B17B50(v81);
        v82 = (char *)sub_BD5D20(*(_QWORD *)(a7 + 40));
        sub_B16430((__int64)v214, "FromBlock", 9u, v82, v83);
        v84 = sub_23FD640(v81, (__int64)v214);
        v85 = (char *)sub_BD5D20(v188);
        sub_B16430((__int64)v210, "ToBlock", 7u, v85, v86);
        v91 = sub_23FD640(v84, (__int64)v210);
        v247 = *(_DWORD *)(v91 + 8);
        v248 = *(_BYTE *)(v91 + 12);
        v249 = *(_QWORD *)(v91 + 16);
        v92 = _mm_loadu_si128((const __m128i *)(v91 + 24));
        v246 = (char *)&unk_49D9D40;
        v250 = v92;
        v251 = *(_QWORD *)(v91 + 40);
        v252 = _mm_loadu_si128((const __m128i *)(v91 + 48));
        v93 = _mm_loadu_si128((const __m128i *)(v91 + 64));
        v254 = (unsigned __int64 *)v256;
        v255 = 0x400000000LL;
        v253 = v93;
        if ( *(_DWORD *)(v91 + 88) )
          sub_2822780((__int64)&v254, v91 + 80, v87, v88, v89, v90);
        v257 = *(_BYTE *)(v91 + 416);
        v258 = *(_DWORD *)(v91 + 420);
        v259 = *(_QWORD *)(v91 + 424);
        v246 = (char *)&unk_49D9D78;
        if ( v212 != &v213 )
          j_j___libc_free_0((unsigned __int64)v212);
        if ( (__int64 *)v210[0] != &v211 )
          j_j___libc_free_0(v210[0]);
        if ( v216 != &v217 )
          j_j___libc_free_0((unsigned __int64)v216);
        if ( (__int64 *)v214[0] != &v215 )
          j_j___libc_free_0(v214[0]);
        if ( v220 != v221 )
          j_j___libc_free_0((unsigned __int64)v220);
        if ( (_QWORD *)v218[0] != v219 )
          j_j___libc_free_0(v218[0]);
        if ( v224 != v225 )
          j_j___libc_free_0((unsigned __int64)v224);
        if ( (_QWORD *)v222[0] != v223 )
          j_j___libc_free_0(v222[0]);
        if ( v228 != v229 )
          j_j___libc_free_0((unsigned __int64)v228);
        if ( (_QWORD *)v226[0] != v227 )
          j_j___libc_free_0(v226[0]);
        v94 = v263;
        v260 = (unsigned __int64)&unk_49D9D40;
        v95 = 10LL * v264;
        v96 = &v263[v95];
        if ( v263 != &v263[v95] )
        {
          do
          {
            v96 -= 10;
            v97 = v96[4];
            if ( (unsigned __int64 *)v97 != v96 + 6 )
              j_j___libc_free_0(v97);
            if ( (unsigned __int64 *)*v96 != v96 + 2 )
              j_j___libc_free_0(*v96);
          }
          while ( v94 != v96 );
          v96 = v263;
        }
        if ( v96 != (unsigned __int64 *)v265 )
          _libc_free((unsigned __int64)v96);
        sub_1049740(v75, (__int64)&v246);
        v99 = v254;
        v246 = (char *)&unk_49D9D40;
        v100 = &v254[10 * (unsigned int)v255];
        if ( v254 != v100 )
        {
          do
          {
            v100 -= 10;
            v101 = v100[4];
            if ( (unsigned __int64 *)v101 != v100 + 6 )
              j_j___libc_free_0(v101);
            if ( (unsigned __int64 *)*v100 != v100 + 2 )
              j_j___libc_free_0(*v100);
          }
          while ( v99 != v100 );
          v100 = v254;
        }
        if ( v100 != (unsigned __int64 *)v256 )
          _libc_free((unsigned __int64)v100);
      }
      v102 = *(_QWORD **)(a1 + 80);
      if ( v102 )
      {
        v103 = *(_DWORD *)(*v102 + 56LL);
        v104 = *(_QWORD *)(*v102 + 40LL);
        if ( v103 )
        {
          v105 = v103 - 1;
          v106 = v105 & (((unsigned int)a7 >> 9) ^ ((unsigned int)a7 >> 4));
          v107 = (__int64 *)(v104 + 16LL * v106);
          v108 = *v107;
          if ( a7 == *v107 )
          {
LABEL_144:
            v109 = v107[1];
            if ( v109 )
              sub_D6E4B0(v102, v109, 1, v104, v108, v98);
          }
          else
          {
            v166 = 1;
            while ( v108 != -4096 )
            {
              v98 = (unsigned int)(v166 + 1);
              v106 = v105 & (v166 + v106);
              v107 = (__int64 *)(v104 + 16LL * v106);
              v108 = *v107;
              if ( a7 == *v107 )
                goto LABEL_144;
              v166 = v98;
            }
          }
        }
      }
      v42 = sub_ACADE0(*(__int64 ***)(a7 + 8));
      sub_BD84D0(a7, v42);
      sub_B43D60((_QWORD *)a7);
      if ( *(_QWORD *)(a1 + 80) )
      {
        v38 = (__int64)byte_4F8F8E8;
        if ( byte_4F8F8E8[0] )
        {
          v42 = 0;
          nullsub_390();
        }
      }
      v197 = 1;
      goto LABEL_34;
    }
    sub_B91220((__int64)(v70 + 6), v139);
    goto LABEL_236;
  }
LABEL_34:
  if ( !v208 )
    _libc_free((unsigned __int64)v205);
  if ( v199 > 0x40 && v198 )
    j_j___libc_free_0_0(v198);
  sub_F82D10((__int64)&v196, v42, v38, (void *)v41, v39, v40);
  sub_27C20B0((__int64)v266);
  nullsub_61();
  v244 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v230[0] != v231 )
    _libc_free(v230[0]);
  return 1;
}
