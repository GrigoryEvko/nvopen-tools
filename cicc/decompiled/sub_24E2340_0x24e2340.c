// Function: sub_24E2340
// Address: 0x24e2340
//
_QWORD *__fastcall sub_24E2340(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r12
  void *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 i; // rbx
  unsigned __int8 *v16; // r12
  int v17; // eax
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // ebx
  unsigned __int16 v24; // ax
  unsigned __int8 v25; // dl
  __int64 v26; // rax
  __int64 *v27; // rdi
  unsigned __int8 *v28; // r12
  __int64 v29; // rax
  __int64 v30; // r13
  char v31; // bl
  char v32; // r15
  int v33; // ecx
  int v34; // ecx
  __int64 v35; // rsi
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // rdi
  int v39; // ecx
  __int64 v40; // rsi
  int v41; // ecx
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // rdi
  _QWORD *v45; // r14
  __int64 v46; // rax
  __int64 v47; // r8
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 *v50; // rax
  __int64 *v51; // rax
  unsigned int v52; // ebx
  __int64 *v53; // r15
  _QWORD *v54; // rax
  _QWORD *v55; // r14
  int v56; // edx
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rbx
  __int64 v61; // rax
  int v62; // ebx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 v67; // r15
  unsigned __int8 *v68; // rbx
  __int64 v69; // r15
  unsigned __int64 v70; // r9
  __int64 *v71; // rax
  __int64 v72; // r15
  int v73; // eax
  __int64 v74; // rbx
  __int64 v75; // r14
  _QWORD *v76; // rax
  int v77; // edx
  __int64 v78; // r15
  bool v79; // zf
  __int64 v80; // rax
  __int64 v81; // rax
  void **v82; // r14
  void *v83; // rsi
  void *v84; // r14
  __int64 v85; // rax
  _BYTE *v86; // rsi
  __int64 v87; // rdx
  _BYTE *v88; // rbx
  __int64 *v89; // rax
  _QWORD *v90; // r15
  _QWORD *v91; // r12
  __int64 v92; // rax
  __int64 v93; // r12
  __int64 v94; // rax
  char *v95; // rax
  __int64 v96; // rdx
  __int64 v97; // r12
  char *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // r12
  __int64 v101; // rdx
  __int64 v102; // r8
  __int64 v103; // r9
  __m128i v104; // xmm3
  __m128i v105; // xmm5
  __int64 v106; // rcx
  unsigned __int64 *v107; // r12
  unsigned __int64 *v108; // r14
  unsigned __int64 v109; // rdi
  unsigned __int64 *v110; // rbx
  unsigned __int64 *v111; // r12
  unsigned __int64 v112; // rdi
  _QWORD *v113; // rsi
  _QWORD *v114; // rdx
  char v116; // al
  __int64 v117; // r12
  __int64 v118; // rax
  char *v119; // rax
  __int64 v120; // rdx
  __int64 v121; // r12
  char *v122; // rax
  __int64 v123; // rdx
  __int64 v124; // r14
  char *v125; // rbx
  size_t v126; // rax
  __int64 v127; // rbx
  char *v128; // r15
  __int64 v129; // rdx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rax
  __m128i v133; // xmm0
  __m128i v134; // xmm2
  __int64 v135; // rdx
  unsigned __int64 *v136; // r12
  unsigned __int64 *v137; // r13
  unsigned __int64 v138; // rdi
  unsigned __int64 *v139; // rbx
  unsigned __int64 *v140; // r12
  unsigned __int64 v141; // rdi
  __int64 v142; // rsi
  unsigned __int8 *v143; // rsi
  __int64 v144; // r14
  __int64 v145; // rbx
  _QWORD *v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  int v149; // eax
  int v150; // eax
  __int64 v151; // rax
  __int64 v152; // rax
  int v153; // r8d
  int v154; // r8d
  int v158; // [rsp+30h] [rbp-7E0h]
  unsigned int v159; // [rsp+34h] [rbp-7DCh]
  __int64 v160; // [rsp+38h] [rbp-7D8h]
  __int64 v161; // [rsp+40h] [rbp-7D0h]
  __int64 *v162; // [rsp+58h] [rbp-7B8h]
  __int64 v163; // [rsp+58h] [rbp-7B8h]
  __int64 *v164; // [rsp+60h] [rbp-7B0h]
  __int64 v165; // [rsp+68h] [rbp-7A8h]
  __int64 *v166; // [rsp+70h] [rbp-7A0h]
  unsigned int v167; // [rsp+90h] [rbp-780h]
  unsigned __int8 v168; // [rsp+97h] [rbp-779h]
  char v169; // [rsp+98h] [rbp-778h]
  __int64 v170; // [rsp+98h] [rbp-778h]
  __int64 v171; // [rsp+A0h] [rbp-770h]
  __int64 **v172; // [rsp+A8h] [rbp-768h]
  __int64 v174; // [rsp+D8h] [rbp-738h]
  __int64 v175; // [rsp+E0h] [rbp-730h]
  __int64 v176; // [rsp+E8h] [rbp-728h]
  __int64 v177; // [rsp+F0h] [rbp-720h]
  __int64 v178; // [rsp+F0h] [rbp-720h]
  __int64 v179; // [rsp+F0h] [rbp-720h]
  int v180; // [rsp+F0h] [rbp-720h]
  _QWORD *v181; // [rsp+F0h] [rbp-720h]
  __int64 v182; // [rsp+F0h] [rbp-720h]
  __int64 v183; // [rsp+F0h] [rbp-720h]
  unsigned __int8 *v184; // [rsp+F0h] [rbp-720h]
  __int64 *v185; // [rsp+F8h] [rbp-718h]
  __int64 v186; // [rsp+100h] [rbp-710h]
  _QWORD *v187; // [rsp+108h] [rbp-708h]
  __int64 *v188; // [rsp+118h] [rbp-6F8h]
  __int64 *v189; // [rsp+120h] [rbp-6F0h] BYREF
  __int64 v190; // [rsp+128h] [rbp-6E8h]
  _BYTE v191[32]; // [rsp+130h] [rbp-6E0h] BYREF
  __int64 v192[2]; // [rsp+150h] [rbp-6C0h] BYREF
  _QWORD v193[2]; // [rsp+160h] [rbp-6B0h] BYREF
  _QWORD *v194; // [rsp+170h] [rbp-6A0h] BYREF
  _QWORD v195[4]; // [rsp+180h] [rbp-690h] BYREF
  __int64 v196[2]; // [rsp+1A0h] [rbp-670h] BYREF
  _QWORD v197[2]; // [rsp+1B0h] [rbp-660h] BYREF
  _QWORD *v198; // [rsp+1C0h] [rbp-650h] BYREF
  _QWORD v199[4]; // [rsp+1D0h] [rbp-640h] BYREF
  unsigned __int64 v200[2]; // [rsp+1F0h] [rbp-620h] BYREF
  _QWORD v201[2]; // [rsp+200h] [rbp-610h] BYREF
  _QWORD *v202; // [rsp+210h] [rbp-600h]
  _QWORD v203[4]; // [rsp+220h] [rbp-5F0h] BYREF
  unsigned __int64 v204[2]; // [rsp+240h] [rbp-5D0h] BYREF
  _QWORD v205[2]; // [rsp+250h] [rbp-5C0h] BYREF
  _QWORD *v206; // [rsp+260h] [rbp-5B0h]
  _QWORD v207[4]; // [rsp+270h] [rbp-5A0h] BYREF
  __int64 *v208; // [rsp+290h] [rbp-580h] BYREF
  unsigned __int64 v209; // [rsp+298h] [rbp-578h]
  __int64 v210; // [rsp+2A0h] [rbp-570h] BYREF
  __m128i v211; // [rsp+2A8h] [rbp-568h]
  __int64 v212; // [rsp+2B8h] [rbp-558h]
  __m128i v213; // [rsp+2C0h] [rbp-550h]
  __m128i v214; // [rsp+2D0h] [rbp-540h]
  unsigned __int64 *v215; // [rsp+2E0h] [rbp-530h] BYREF
  __int64 v216; // [rsp+2E8h] [rbp-528h]
  _BYTE v217[320]; // [rsp+2F0h] [rbp-520h] BYREF
  char v218; // [rsp+430h] [rbp-3E0h]
  int v219; // [rsp+434h] [rbp-3DCh]
  __int64 v220; // [rsp+438h] [rbp-3D8h]
  void *v221[12]; // [rsp+440h] [rbp-3D0h] BYREF
  __int64 v222; // [rsp+4A0h] [rbp-370h] BYREF
  _BYTE v223[192]; // [rsp+4A8h] [rbp-368h] BYREF
  _BYTE *v224; // [rsp+568h] [rbp-2A8h]
  __int64 v225; // [rsp+570h] [rbp-2A0h]
  _BYTE v226[120]; // [rsp+578h] [rbp-298h] BYREF
  __int64 v227; // [rsp+5F0h] [rbp-220h] BYREF
  char *v228; // [rsp+5F8h] [rbp-218h]
  __int64 v229; // [rsp+600h] [rbp-210h]
  int v230; // [rsp+608h] [rbp-208h]
  char v231; // [rsp+60Ch] [rbp-204h]
  char v232; // [rsp+610h] [rbp-200h] BYREF
  _BYTE *v233; // [rsp+690h] [rbp-180h]
  __int64 v234; // [rsp+698h] [rbp-178h]
  _BYTE v235[128]; // [rsp+6A0h] [rbp-170h] BYREF
  _BYTE *v236; // [rsp+720h] [rbp-F0h]
  __int64 v237; // [rsp+728h] [rbp-E8h]
  _BYTE v238[128]; // [rsp+730h] [rbp-E0h] BYREF
  __int64 v239; // [rsp+7B0h] [rbp-60h]
  __int64 v240; // [rsp+7B8h] [rbp-58h]
  __int64 v241; // [rsp+7C0h] [rbp-50h]
  __int64 v242; // [rsp+7C8h] [rbp-48h]
  __int64 v243; // [rsp+7D0h] [rbp-40h]

  v228 = &v232;
  v233 = v235;
  v236 = v238;
  v241 = a4;
  v239 = a5;
  v242 = a6;
  v227 = 0;
  v229 = 16;
  v230 = 0;
  v231 = 1;
  v234 = 0x1000000000LL;
  v237 = 0x1000000000LL;
  v243 = 0;
  v240 = a3;
  v243 = *(_QWORD *)(sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5) + 8);
  v171 = *(_QWORD *)(sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5) + 8);
  v9 = *(_QWORD *)(a3 + 8);
  v161 = v9 + 8LL * *(unsigned int *)(a3 + 16);
  if ( v161 == v9 )
  {
    v113 = a1 + 4;
    v114 = a1 + 10;
  }
  else
  {
    v174 = *(_QWORD *)(a3 + 8);
    v169 = 0;
    do
    {
      v10 = *(_QWORD *)(*(_QWORD *)v174 + 8LL);
      v11 = *(_QWORD *)(v10 + 40);
      v186 = v10;
      v221[0] = (void *)sub_BD5D20(v10);
      LOWORD(v221[4]) = 773;
      v221[1] = v12;
      v221[2] = ".noalloc";
      sub_CA0F50((__int64 *)&v208, v221);
      v187 = sub_BA8CB0(v11, (__int64)v208, v209);
      if ( v208 != &v210 )
        j_j___libc_free_0((unsigned __int64)v208);
      if ( v187 )
      {
        v189 = (__int64 *)v191;
        v190 = 0x400000000LL;
        for ( i = *(_QWORD *)(v10 + 16); i; i = *(_QWORD *)(i + 8) )
        {
          while ( 1 )
          {
            v16 = *(unsigned __int8 **)(i + 24);
            v17 = *v16;
            if ( (unsigned __int8)v17 > 0x1Cu )
            {
              v18 = (unsigned int)(v17 - 34);
              if ( (unsigned __int8)v18 <= 0x33u )
              {
                v19 = 0x8000000000041LL;
                if ( _bittest64(&v19, v18) )
                {
                  v20 = *((_QWORD *)v16 - 4);
                  if ( v20 )
                  {
                    if ( !*(_BYTE *)v20 && *(_QWORD *)(v20 + 24) == *((_QWORD *)v16 + 10) && v10 == v20 )
                      break;
                  }
                }
              }
            }
            i = *(_QWORD *)(i + 8);
            if ( !i )
              goto LABEL_19;
          }
          v21 = (unsigned int)v190;
          v22 = (unsigned int)v190 + 1LL;
          if ( v22 > HIDWORD(v190) )
          {
            sub_C8D5F0((__int64)&v189, v191, v22, 8u, v13, v14);
            v21 = (unsigned int)v190;
          }
          v189[v21] = (__int64)v16;
          LODWORD(v190) = v190 + 1;
        }
LABEL_19:
        v23 = v187[13] - 1;
        v165 = sub_A745B0(v187 + 15, v23);
        v24 = sub_A74840(v187 + 15, v23);
        v25 = 0;
        if ( HIBYTE(v24) )
          v25 = v24;
        v168 = v25;
        v26 = sub_BC1CD0(v171, &unk_4F8FAE8, v10);
        v27 = v189;
        v175 = v26;
        v164 = (__int64 *)(v26 + 8);
        v185 = &v189[(unsigned int)v190];
        if ( v185 != v189 )
        {
          v188 = v189;
          while ( 1 )
          {
            v28 = (unsigned __int8 *)*v188;
            v29 = sub_B43CB0(*v188);
            v30 = v29;
            if ( v29 )
              break;
LABEL_131:
            if ( v185 == ++v188 )
            {
              v27 = v189;
              goto LABEL_133;
            }
          }
          v31 = sub_B2D610(v29, 49);
          v32 = sub_A73ED0((_QWORD *)v28 + 9, 8);
          if ( v32 )
          {
            if ( !v31 )
              goto LABEL_147;
          }
          else
          {
            v116 = sub_B49560((__int64)v28, 8);
            v32 = v116;
            if ( !v31 || !v116 )
            {
LABEL_147:
              v117 = *(_QWORD *)(v175 + 8);
              v118 = sub_B2BE50(v117);
              if ( sub_B6EA50(v118)
                || (v147 = sub_B2BE50(v117),
                    v148 = sub_B6F970(v147),
                    (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v148 + 48LL))(v148)) )
              {
                sub_B17770((__int64)v221, (__int64)"coro-annotation-elide", (__int64)"CoroAnnotationElide", 19, v30);
                sub_B18290((__int64)v221, "'", 1u);
                v119 = (char *)sub_BD5D20(v186);
                sub_B16430((__int64)v204, "callee", 6u, v119, v120);
                v121 = sub_2445430((__int64)v221, (__int64)v204);
                sub_B18290(v121, "' not elided in '", 0x11u);
                v122 = (char *)sub_BD5D20(v30);
                sub_B16430((__int64)v200, "caller", 6u, v122, v123);
                v124 = sub_2445430(v121, (__int64)v200);
                sub_B18290(v124, "' (caller_presplit=", 0x13u);
                v196[0] = (__int64)v197;
                sub_24E1F60(v196, "caller_presplit", (__int64)"");
                v79 = v31 == 0;
                v125 = "false";
                if ( !v79 )
                  v125 = "true";
                v198 = v199;
                v126 = strlen(v125);
                sub_24E1F60((__int64 *)&v198, v125, (__int64)&v125[v126]);
                v199[2] = 0;
                v199[3] = 0;
                v127 = sub_2445430(v124, (__int64)v196);
                sub_B18290(v127, ", elide_safe_attr=", 0x12u);
                v192[0] = (__int64)v193;
                sub_24E1F60(v192, "elide_safe_attr", (__int64)"");
                v79 = v32 == 0;
                v128 = "false";
                if ( !v79 )
                  v128 = "true";
                v194 = v195;
                v129 = (__int64)&v128[strlen(v128)];
                sub_24E1F60((__int64 *)&v194, v128, v129);
                v195[2] = 0;
                v195[3] = 0;
                v182 = sub_2445430(v127, (__int64)v192);
                sub_B18290(v182, ")", 1u);
                v132 = v182;
                LODWORD(v209) = *(_DWORD *)(v182 + 8);
                BYTE4(v209) = *(_BYTE *)(v182 + 12);
                v210 = *(_QWORD *)(v182 + 16);
                v133 = _mm_loadu_si128((const __m128i *)(v182 + 24));
                v208 = (__int64 *)&unk_49D9D40;
                v211 = v133;
                v212 = *(_QWORD *)(v182 + 40);
                v213 = _mm_loadu_si128((const __m128i *)(v182 + 48));
                v134 = _mm_loadu_si128((const __m128i *)(v182 + 64));
                v216 = 0x400000000LL;
                v215 = (unsigned __int64 *)v217;
                v214 = v134;
                v135 = *(unsigned int *)(v182 + 88);
                if ( (_DWORD)v135 )
                {
                  sub_24E20C0((__int64)&v215, v182 + 80, v135, (__int64)v195, v130, v131);
                  v132 = v182;
                }
                v218 = *(_BYTE *)(v132 + 416);
                v219 = *(_DWORD *)(v132 + 420);
                v220 = *(_QWORD *)(v132 + 424);
                v208 = (__int64 *)&unk_49D9DB0;
                if ( v194 != v195 )
                  j_j___libc_free_0((unsigned __int64)v194);
                if ( (_QWORD *)v192[0] != v193 )
                  j_j___libc_free_0(v192[0]);
                if ( v198 != v199 )
                  j_j___libc_free_0((unsigned __int64)v198);
                if ( (_QWORD *)v196[0] != v197 )
                  j_j___libc_free_0(v196[0]);
                if ( v202 != v203 )
                  j_j___libc_free_0((unsigned __int64)v202);
                if ( (_QWORD *)v200[0] != v201 )
                  j_j___libc_free_0(v200[0]);
                if ( v206 != v207 )
                  j_j___libc_free_0((unsigned __int64)v206);
                if ( (_QWORD *)v204[0] != v205 )
                  j_j___libc_free_0(v204[0]);
                v136 = (unsigned __int64 *)v221[10];
                v221[0] = &unk_49D9D40;
                v137 = (unsigned __int64 *)((char *)v221[10] + 80 * LODWORD(v221[11]));
                if ( v221[10] != v137 )
                {
                  do
                  {
                    v137 -= 10;
                    v138 = v137[4];
                    if ( (unsigned __int64 *)v138 != v137 + 6 )
                      j_j___libc_free_0(v138);
                    if ( (unsigned __int64 *)*v137 != v137 + 2 )
                      j_j___libc_free_0(*v137);
                  }
                  while ( v136 != v137 );
                  v137 = (unsigned __int64 *)v221[10];
                }
                if ( v137 != (unsigned __int64 *)&v222 )
                  _libc_free((unsigned __int64)v137);
                sub_1049740(v164, (__int64)&v208);
                v139 = v215;
                v208 = (__int64 *)&unk_49D9D40;
                v140 = &v215[10 * (unsigned int)v216];
                if ( v215 != v140 )
                {
                  do
                  {
                    v140 -= 10;
                    v141 = v140[4];
                    if ( (unsigned __int64 *)v141 != v140 + 6 )
                      j_j___libc_free_0(v141);
                    if ( (unsigned __int64 *)*v140 != v140 + 2 )
                      j_j___libc_free_0(*v140);
                  }
                  while ( v139 != v140 );
                  v140 = v215;
                }
                if ( v140 != (unsigned __int64 *)v217 )
                  _libc_free((unsigned __int64)v140);
              }
              goto LABEL_131;
            }
          }
          v33 = *(_DWORD *)(a5 + 120);
          if ( v33 )
          {
            v34 = v33 - 1;
            v35 = *(_QWORD *)(a5 + 104);
            v36 = v34 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v37 = (__int64 *)(v35 + 16LL * v36);
            v38 = *v37;
            if ( v30 == *v37 )
            {
LABEL_28:
              v176 = v37[1];
              if ( v176 )
              {
                v39 = *(_DWORD *)(a5 + 328);
                v40 = *(_QWORD *)(a5 + 312);
                if ( v39 )
                {
                  v41 = v39 - 1;
                  v42 = v41 & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
                  v43 = (__int64 *)(v40 + 16LL * v42);
                  v44 = *v43;
                  if ( v176 == *v43 )
                  {
LABEL_31:
                    v172 = (__int64 **)v43[1];
                    goto LABEL_32;
                  }
                  v149 = 1;
                  while ( v44 != -4096 )
                  {
                    v154 = v149 + 1;
                    v42 = v41 & (v149 + v42);
                    v43 = (__int64 *)(v40 + 16LL * v42);
                    v44 = *v43;
                    if ( v176 == *v43 )
                      goto LABEL_31;
                    v149 = v154;
                  }
                }
              }
              v172 = 0;
LABEL_32:
              v45 = (_QWORD *)sub_B2BE50(v30);
              v46 = *(_QWORD *)(v30 + 80);
              if ( !v46 )
                BUG();
              v47 = *(_QWORD *)(v46 + 32);
              v48 = v46 + 24;
              if ( v47 == v48 )
LABEL_218:
                BUG();
              while ( 1 )
              {
                if ( !v47 )
                  BUG();
                if ( *(_BYTE *)(v47 - 24) != 60 )
                  break;
                v47 = *(_QWORD *)(v47 + 8);
                if ( v48 == v47 )
                  goto LABEL_218;
              }
              v177 = v47;
              v49 = sub_B2BEC0(v30);
              v50 = (__int64 *)sub_BCB2B0(v45);
              v51 = sub_BCD420(v50, v165);
              v52 = *(_DWORD *)(v49 + 4);
              v53 = v51;
              LOWORD(v221[4]) = 257;
              v54 = sub_BD2C40(80, unk_3F10A14);
              v55 = v54;
              if ( v54 )
                sub_B4CE50((__int64)v54, v53, v52, (__int64)v221, v177, 0);
              *((_WORD *)v55 + 1) = v168 | *((_WORD *)v55 + 1) & 0xFFC0;
              v170 = (__int64)(v28 + 24);
              v208 = &v210;
              v209 = 0x400000000LL;
              v56 = *v28;
              if ( v56 == 40 )
              {
                v57 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v28);
              }
              else
              {
                v57 = -32;
                if ( v56 != 85 )
                {
                  if ( v56 != 34 )
                    BUG();
                  v57 = -96;
                }
              }
              if ( (v28[7] & 0x80u) != 0 )
              {
                v178 = v57;
                v58 = sub_BD2BC0((__int64)v28);
                v57 = v178;
                v60 = v58 + v59;
                v61 = 0;
                if ( (v28[7] & 0x80u) != 0 )
                {
                  v61 = sub_BD2BC0((__int64)v28);
                  v57 = v178;
                }
                if ( (unsigned int)((v60 - v61) >> 4) )
                {
                  v179 = v57;
                  if ( (v28[7] & 0x80u) == 0 )
                    BUG();
                  v62 = *(_DWORD *)(sub_BD2BC0((__int64)v28) + 8);
                  if ( (v28[7] & 0x80u) == 0 )
                    BUG();
                  v63 = sub_BD2BC0((__int64)v28);
                  v57 = v179 - 32LL * (unsigned int)(*(_DWORD *)(v63 + v64 - 4) - v62);
                }
              }
              v65 = (unsigned int)v209;
              v66 = (__int64)&v28[v57];
              v67 = 32LL * (*((_DWORD *)v28 + 1) & 0x7FFFFFF);
              v68 = &v28[-v67];
              v69 = (v57 + v67) >> 5;
              v70 = v69 + (unsigned int)v209;
              if ( v70 > HIDWORD(v209) )
              {
                v184 = &v28[v57];
                sub_C8D5F0((__int64)&v208, &v210, v69 + (unsigned int)v209, 8u, v66, v70);
                v65 = (unsigned int)v209;
                v66 = (__int64)v184;
              }
              v71 = &v208[v65];
              if ( v68 != (unsigned __int8 *)v66 )
              {
                do
                {
                  if ( v71 )
                    *v71 = *(_QWORD *)v68;
                  v68 += 32;
                  ++v71;
                }
                while ( v68 != (unsigned __int8 *)v66 );
                LODWORD(v65) = v209;
              }
              LODWORD(v209) = v65 + v69;
              v72 = (unsigned int)(v65 + v69);
              if ( v72 + 1 > (unsigned __int64)HIDWORD(v209) )
              {
                sub_C8D5F0((__int64)&v208, &v210, v72 + 1, 8u, v66, v70);
                v72 = (unsigned int)v209;
              }
              v208[v72] = (__int64)v55;
              v73 = v209;
              v74 = (unsigned int)(v209 + 1);
              LODWORD(v209) = v209 + 1;
              if ( *v28 == 85 )
              {
                LOWORD(v221[4]) = 257;
                v162 = v208;
                v180 = v73 + 2;
                v75 = v187[3];
                v76 = sub_BD2C40(88, v73 + 2);
                v77 = v180;
                v78 = (__int64)v76;
                if ( v76 )
                {
                  v181 = v76;
                  v167 = v77 & 0x7FFFFFF | v167 & 0xE0000000;
                  sub_B44260((__int64)v76, **(_QWORD **)(v75 + 16), 56, v167, v170, 0);
                  *(_QWORD *)(v78 + 72) = 0;
                  sub_B4A290(v78, v75, (__int64)v187, v162, v74, (__int64)v221, 0, 0);
                }
                else
                {
                  v181 = 0;
                }
                *(_WORD *)(v78 + 2) = *(_WORD *)(v78 + 2) & 0xFFFC | *((_WORD *)v28 + 1) & 3;
              }
              else
              {
                if ( *v28 != 34 )
                  goto LABEL_218;
                LOWORD(v221[4]) = 257;
                v144 = *((_QWORD *)v28 - 8);
                v183 = (__int64)v208;
                v163 = (unsigned int)v74;
                v158 = v73 + 4;
                v160 = *((_QWORD *)v28 - 12);
                v145 = v187[3];
                v146 = sub_BD2CC0(88, (unsigned int)(v73 + 4));
                v78 = (__int64)v146;
                if ( v146 )
                {
                  v166 = (__int64 *)v183;
                  v181 = v146;
                  v159 = v158 & 0x7FFFFFF | v159 & 0xE0000000;
                  sub_B44260((__int64)v146, **(_QWORD **)(v145 + 16), 5, v159, v170, 0);
                  *(_QWORD *)(v78 + 72) = 0;
                  sub_B4A9C0(v78, v145, (__int64)v187, v160, v144, (__int64)v221, v166, v163, 0, 0);
                }
                else
                {
                  v181 = 0;
                }
              }
              v79 = *(_QWORD *)(v78 - 32) == 0;
              *(_QWORD *)(v78 + 80) = v187[3];
              if ( !v79 )
              {
                v80 = *(_QWORD *)(v78 - 24);
                **(_QWORD **)(v78 - 16) = v80;
                if ( v80 )
                  *(_QWORD *)(v80 + 16) = *(_QWORD *)(v78 - 16);
              }
              *(_QWORD *)(v78 - 32) = v187;
              v81 = v187[2];
              *(_QWORD *)(v78 - 24) = v81;
              if ( v81 )
                *(_QWORD *)(v81 + 16) = v78 - 24;
              *(_QWORD *)(v78 - 16) = v187 + 2;
              v82 = (void **)(v78 + 48);
              v187[2] = v78 - 32;
              *(_WORD *)(v78 + 2) = *((_WORD *)v28 + 1) & 0xFFC | *(_WORD *)(v78 + 2) & 0xF003;
              *(_QWORD *)(v78 + 72) = *((_QWORD *)v28 + 9);
              v83 = (void *)*((_QWORD *)v28 + 6);
              v221[0] = v83;
              if ( v83 )
              {
                sub_B96E90((__int64)v221, (__int64)v83, 1);
                if ( v82 == v221 )
                {
                  if ( v221[0] )
                    sub_B91220((__int64)v221, (__int64)v221[0]);
                  goto LABEL_73;
                }
                v142 = *(_QWORD *)(v78 + 48);
                if ( !v142 )
                {
LABEL_191:
                  v143 = (unsigned __int8 *)v221[0];
                  *(void **)(v78 + 48) = v221[0];
                  if ( v143 )
                    sub_B976B0((__int64)v221, v143, v78 + 48);
                  goto LABEL_73;
                }
              }
              else if ( v82 == v221 || (v142 = *(_QWORD *)(v78 + 48)) == 0 )
              {
LABEL_73:
                v84 = 0;
                if ( *(char *)(v78 + 7) < 0 )
                  v84 = (void *)sub_BD2BC0((__int64)v181);
                if ( (v28[7] & 0x80u) != 0 )
                {
                  v85 = sub_BD2BC0((__int64)v28);
                  v86 = 0;
                  v88 = (_BYTE *)(v85 + v87);
                  if ( (v28[7] & 0x80u) != 0 )
                    v86 = (_BYTE *)sub_BD2BC0((__int64)v28);
                  if ( v86 != v88 )
                    memmove(v84, v86, v88 - v86);
                }
                v89 = (__int64 *)sub_BD5C60((__int64)v181);
                *(_QWORD *)(v78 + 72) = sub_A7B980((__int64 *)(v78 + 72), v89, -1, 8);
                sub_BD84D0((__int64)v28, v78);
                v221[6] = (void *)0x400000000LL;
                v224 = v226;
                memset(v221, 0, 40);
                v221[5] = &v221[7];
                v221[11] = v223;
                v222 = 0x800000000LL;
                v225 = 0x800000000LL;
                v226[64] = 1;
                if ( sub_29F2700(v78, v221, 0, 0, 1, 0) )
                {
                  sub_BD84D0((__int64)v181, (__int64)v28);
                  sub_B43D60(v181);
                }
                else
                {
                  sub_B43D60(v28);
                }
                if ( v224 != v226 )
                  _libc_free((unsigned __int64)v224);
                v90 = v221[11];
                v91 = (char *)v221[11] + 24 * (unsigned int)v222;
                if ( v221[11] != v91 )
                {
                  do
                  {
                    v92 = *(v91 - 1);
                    v91 -= 3;
                    if ( v92 != 0 && v92 != -4096 && v92 != -8192 )
                      sub_BD60C0(v91);
                  }
                  while ( v90 != v91 );
                  v91 = v221[11];
                }
                if ( v91 != (_QWORD *)v223 )
                  _libc_free((unsigned __int64)v91);
                if ( v221[5] != &v221[7] )
                  _libc_free((unsigned __int64)v221[5]);
                if ( v208 != &v210 )
                  _libc_free((unsigned __int64)v208);
                v93 = *(_QWORD *)(v175 + 8);
                v94 = sub_B2BE50(v93);
                if ( sub_B6EA50(v94)
                  || (v151 = sub_B2BE50(v93),
                      v152 = sub_B6F970(v151),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v152 + 48LL))(v152)) )
                {
                  sub_B17560((__int64)v221, (__int64)"coro-annotation-elide", (__int64)"CoroAnnotationElide", 19, v30);
                  sub_B18290((__int64)v221, "'", 1u);
                  v95 = (char *)sub_BD5D20(v186);
                  sub_B16430((__int64)v204, "callee", 6u, v95, v96);
                  v97 = sub_23FD640((__int64)v221, (__int64)v204);
                  sub_B18290(v97, "' elided in '", 0xDu);
                  v98 = (char *)sub_BD5D20(v30);
                  sub_B16430((__int64)v200, "caller", 6u, v98, v99);
                  v100 = sub_23FD640(v97, (__int64)v200);
                  sub_B18290(v100, "'", 1u);
                  LODWORD(v209) = *(_DWORD *)(v100 + 8);
                  BYTE4(v209) = *(_BYTE *)(v100 + 12);
                  v210 = *(_QWORD *)(v100 + 16);
                  v104 = _mm_loadu_si128((const __m128i *)(v100 + 24));
                  v208 = (__int64 *)&unk_49D9D40;
                  v211 = v104;
                  v212 = *(_QWORD *)(v100 + 40);
                  v213 = _mm_loadu_si128((const __m128i *)(v100 + 48));
                  v105 = _mm_loadu_si128((const __m128i *)(v100 + 64));
                  v215 = (unsigned __int64 *)v217;
                  v216 = 0x400000000LL;
                  v214 = v105;
                  v106 = *(unsigned int *)(v100 + 88);
                  if ( (_DWORD)v106 )
                    sub_24E20C0((__int64)&v215, v100 + 80, v101, v106, v102, v103);
                  v218 = *(_BYTE *)(v100 + 416);
                  v219 = *(_DWORD *)(v100 + 420);
                  v220 = *(_QWORD *)(v100 + 424);
                  v208 = (__int64 *)&unk_49D9D78;
                  if ( v202 != v203 )
                    j_j___libc_free_0((unsigned __int64)v202);
                  if ( (_QWORD *)v200[0] != v201 )
                    j_j___libc_free_0(v200[0]);
                  if ( v206 != v207 )
                    j_j___libc_free_0((unsigned __int64)v206);
                  if ( (_QWORD *)v204[0] != v205 )
                    j_j___libc_free_0(v204[0]);
                  v107 = (unsigned __int64 *)v221[10];
                  v221[0] = &unk_49D9D40;
                  v108 = (unsigned __int64 *)((char *)v221[10] + 80 * LODWORD(v221[11]));
                  if ( v221[10] != v108 )
                  {
                    do
                    {
                      v108 -= 10;
                      v109 = v108[4];
                      if ( (unsigned __int64 *)v109 != v108 + 6 )
                        j_j___libc_free_0(v109);
                      if ( (unsigned __int64 *)*v108 != v108 + 2 )
                        j_j___libc_free_0(*v108);
                    }
                    while ( v107 != v108 );
                    v108 = (unsigned __int64 *)v221[10];
                  }
                  if ( v108 != (unsigned __int64 *)&v222 )
                    _libc_free((unsigned __int64)v108);
                  sub_1049740(v164, (__int64)&v208);
                  v110 = v215;
                  v208 = (__int64 *)&unk_49D9D40;
                  v111 = &v215[10 * (unsigned int)v216];
                  if ( v215 != v111 )
                  {
                    do
                    {
                      v111 -= 10;
                      v112 = v111[4];
                      if ( (unsigned __int64 *)v112 != v111 + 6 )
                        j_j___libc_free_0(v112);
                      if ( (unsigned __int64 *)*v111 != v111 + 2 )
                        j_j___libc_free_0(*v111);
                    }
                    while ( v110 != v111 );
                    v111 = v215;
                  }
                  if ( v111 != (unsigned __int64 *)v217 )
                    _libc_free((unsigned __int64)v111);
                }
                memset(v221, 0, sizeof(v221));
                v221[1] = &v221[4];
                LODWORD(v221[2]) = 2;
                BYTE4(v221[3]) = 1;
                v221[7] = &v221[10];
                LODWORD(v221[8]) = 2;
                BYTE4(v221[9]) = 1;
                sub_BBE020(v171, v30, (__int64)v221, 0);
                if ( !BYTE4(v221[9]) )
                  _libc_free((unsigned __int64)v221[7]);
                if ( !BYTE4(v221[3]) )
                  _libc_free((unsigned __int64)v221[1]);
                v169 = 1;
                if ( v172 )
                  sub_2284030(a5, v172, v176, a4, a6, v171);
                goto LABEL_131;
              }
              sub_B91220(v78 + 48, v142);
              goto LABEL_191;
            }
            v150 = 1;
            while ( v38 != -4096 )
            {
              v153 = v150 + 1;
              v36 = v34 & (v150 + v36);
              v37 = (__int64 *)(v35 + 16LL * v36);
              v38 = *v37;
              if ( v30 == *v37 )
                goto LABEL_28;
              v150 = v153;
            }
          }
          v176 = 0;
          v172 = 0;
          goto LABEL_32;
        }
LABEL_133:
        if ( v27 != (__int64 *)v191 )
          _libc_free((unsigned __int64)v27);
      }
      v174 += 8;
    }
    while ( v161 != v174 );
    v113 = a1 + 4;
    v114 = a1 + 10;
    if ( v169 )
    {
      memset(a1, 0, 0x60u);
      a1[1] = v113;
      *((_DWORD *)a1 + 4) = 2;
      *((_BYTE *)a1 + 28) = 1;
      a1[7] = v114;
      *((_DWORD *)a1 + 16) = 2;
      *((_BYTE *)a1 + 76) = 1;
      goto LABEL_138;
    }
  }
  a1[2] = 0x100000002LL;
  a1[1] = v113;
  a1[6] = 0;
  a1[7] = v114;
  a1[8] = 2;
  *((_DWORD *)a1 + 18) = 0;
  *((_BYTE *)a1 + 76) = 1;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  *a1 = 1;
  a1[4] = &qword_4F82400;
LABEL_138:
  sub_29A2B10(&v227);
  if ( v236 != v238 )
    _libc_free((unsigned __int64)v236);
  if ( v233 != v235 )
    _libc_free((unsigned __int64)v233);
  if ( !v231 )
    _libc_free((unsigned __int64)v228);
  return a1;
}
