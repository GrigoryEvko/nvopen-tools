// Function: sub_2C80C90
// Address: 0x2c80c90
//
__int64 __fastcall sub_2C80C90(__int64 a1, char *a2, __int64 a3, char *a4)
{
  __int64 v5; // rbx
  char *i; // r14
  char *j; // r14
  __int64 v8; // rdx
  char *v9; // rcx
  _BYTE *v10; // r15
  size_t v11; // r14
  unsigned __int8 *v12; // rax
  const char *v13; // rsi
  __int64 k; // rbx
  unsigned int v15; // r15d
  int v16; // r12d
  __int64 v17; // r15
  char *v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdi
  __m128i *v26; // rax
  __m128i v27; // xmm0
  __int64 v28; // rdi
  __int64 v29; // r15
  unsigned __int64 v30; // rax
  unsigned __int8 *v31; // rsi
  __int64 v32; // rdi
  __m128i *v33; // rax
  __m128i v34; // xmm0
  __int64 v35; // r12
  _QWORD *v36; // rdx
  unsigned __int8 *v37; // r15
  size_t v38; // rax
  _BYTE *v39; // rdi
  size_t v40; // r14
  _BYTE *v41; // rax
  unsigned __int8 *v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // r15
  unsigned __int64 v45; // rax
  __int64 v46; // rdi
  __m128i *v47; // rax
  __m128i v48; // xmm0
  __int64 v49; // r12
  _QWORD *v50; // rdx
  char *v51; // r15
  size_t v52; // rax
  _BYTE *v53; // rdi
  size_t v54; // r14
  _BYTE *v55; // rax
  bool v56; // al
  __int64 v57; // rax
  char *v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rdi
  __m128i *v61; // rax
  __m128i *v62; // rdx
  __m128i v63; // xmm0
  __int64 v64; // rcx
  _BYTE *v65; // rax
  __int64 v66; // rdi
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  char *v71; // rcx
  unsigned __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rdx
  char *v75; // rcx
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // r12
  unsigned int v79; // eax
  __int64 v80; // rax
  char *v81; // rcx
  __m128i *v82; // rdx
  __int64 v83; // rdi
  __m128i v84; // xmm0
  _BYTE *v85; // rax
  _BYTE *v86; // rax
  __int64 v87; // rsi
  __int64 v88; // rdi
  char *v89; // rsi
  __int64 v90; // rax
  __m128i *v91; // rdx
  __int64 v92; // rdi
  __m128i v93; // xmm0
  _BYTE *v94; // rax
  _BYTE *v95; // rax
  char *v96; // rcx
  __int64 v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rdi
  _BYTE *v100; // rax
  __int64 v101; // rax
  char *v102; // rsi
  __int64 v103; // rax
  char *v104; // rcx
  __m128i *v105; // rdx
  __int64 v106; // rdi
  void *v107; // rdx
  __int64 v108; // rax
  _BYTE *v109; // rax
  __int64 v110; // rdi
  __int64 v111; // rax
  __int64 v112; // rax
  __m128i v113; // xmm0
  char *v114; // rsi
  __int64 v115; // rax
  const char *v116; // r12
  unsigned __int8 v117; // cl
  int v118; // edx
  __int64 v119; // rdi
  __int64 v120; // rdx
  __int64 v121; // rdi
  _WORD *v122; // rdx
  __int64 v123; // rdi
  __int64 v124; // rdx
  __int64 v125; // rdi
  __int64 v126; // rax
  __int64 v127; // rdi
  __m128i *v128; // rax
  __m128i v129; // xmm0
  __int64 v130; // rdi
  __m128i *v131; // rax
  __m128i v132; // xmm0
  __int64 v133; // rdi
  __int64 v134; // rax
  __int64 v135; // rdi
  _QWORD *v136; // rdx
  void *v137; // rax
  _BYTE *v138; // rax
  __int64 v139; // rdi
  __int64 v140; // rax
  __int64 v141; // rdi
  _QWORD *v142; // rdx
  void *v143; // rax
  unsigned __int64 v144; // rdx
  __int64 v145; // rcx
  _BYTE *v146; // rax
  _BYTE *v147; // rax
  __int64 v148; // rdi
  const char *v149; // rsi
  __int64 v150; // rax
  char *v151; // rcx
  __int64 v152; // rdx
  __m128i v153; // xmm0
  _BYTE *v154; // rax
  __int64 v155; // rdi
  unsigned int v156; // eax
  __int64 v157; // rax
  __int64 v158; // rdi
  __m128i *v159; // rax
  __m128i v160; // xmm0
  __int64 v161; // rax
  __m128i v162; // xmm0
  const char *v163; // rax
  _BYTE *v164; // rdi
  __int64 v165; // rax
  __int64 v166; // rdx
  __int64 v167; // rdi
  __int64 v168; // rax
  __int64 v169; // rdi
  __m128i *v170; // rax
  __m128i v171; // xmm0
  __int64 v172; // rdx
  __int64 v173; // rcx
  __int64 v174; // rdx
  __int64 v175; // rax
  unsigned __int64 v176; // rdx
  __int64 v177; // rax
  __int64 v178; // rdx
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // rdx
  char *v182; // rcx
  __int64 v183; // rax
  __int64 v184; // rax
  __int64 v185; // rax
  __int64 v186; // rax
  __int64 v187; // rax
  __int64 v188; // rax
  __int64 v189; // rax
  __int64 v190; // rdi
  __int64 v191; // rax
  __int64 v192; // rdi
  __m128i *v193; // rax
  unsigned __int64 v194; // rdx
  __m128i si128; // xmm0
  _BYTE *v196; // rax
  __int64 v197; // rdi
  __int64 v198; // rax
  __int64 v199; // rdx
  char *v200; // rcx
  __int64 v201; // rdi
  __int64 v202; // rax
  __int64 v203; // rdi
  __m128i *v204; // rax
  unsigned __int64 v205; // rdx
  _BYTE *v206; // rax
  __int64 v207; // rsi
  __int64 v208; // rdi
  __int64 v209; // r12
  __m128i *v210; // rax
  __m128i v211; // xmm0
  unsigned __int8 *v212; // r15
  size_t v213; // rax
  _BYTE *v214; // rdi
  size_t v215; // r14
  _BYTE *v216; // rax
  char *v217; // rsi
  char *v218; // rcx
  __int64 v219; // r12
  _QWORD *v220; // rdx
  char *v221; // r15
  size_t v222; // rax
  _BYTE *v223; // rdi
  size_t v224; // r14
  _BYTE *v225; // rax
  __int64 v226; // [rsp+20h] [rbp-2A0h]
  __int64 v227; // [rsp+20h] [rbp-2A0h]
  __int64 v228; // [rsp+20h] [rbp-2A0h]
  __int64 v229; // [rsp+20h] [rbp-2A0h]
  __int64 v230; // [rsp+20h] [rbp-2A0h]
  __int64 v231; // [rsp+20h] [rbp-2A0h]
  __int64 v232; // [rsp+20h] [rbp-2A0h]
  __int64 v233; // [rsp+20h] [rbp-2A0h]
  __int64 v234; // [rsp+20h] [rbp-2A0h]
  char *v235; // [rsp+28h] [rbp-298h]
  __int64 v236; // [rsp+30h] [rbp-290h]
  __int64 v237; // [rsp+30h] [rbp-290h]
  __int64 v238; // [rsp+30h] [rbp-290h]
  __int64 v239; // [rsp+30h] [rbp-290h]
  unsigned int v240; // [rsp+30h] [rbp-290h]
  __int64 v241; // [rsp+30h] [rbp-290h]
  __int64 v242; // [rsp+30h] [rbp-290h]
  __int64 v243; // [rsp+30h] [rbp-290h]
  __int64 v244; // [rsp+30h] [rbp-290h]
  __int64 v245; // [rsp+30h] [rbp-290h]
  __int64 v246; // [rsp+30h] [rbp-290h]
  __int64 v247; // [rsp+30h] [rbp-290h]
  __int64 v248; // [rsp+30h] [rbp-290h]
  __int64 v249; // [rsp+30h] [rbp-290h]
  __int64 v250; // [rsp+30h] [rbp-290h]
  __int64 v251; // [rsp+30h] [rbp-290h]
  __int64 v252; // [rsp+30h] [rbp-290h]
  __int64 v253; // [rsp+30h] [rbp-290h]
  __int64 v254; // [rsp+30h] [rbp-290h]
  __int64 v255; // [rsp+30h] [rbp-290h]
  __int64 v256; // [rsp+30h] [rbp-290h]
  __int64 v257; // [rsp+30h] [rbp-290h]
  char *v258; // [rsp+38h] [rbp-288h]
  char *v259; // [rsp+40h] [rbp-280h]
  char *v260; // [rsp+48h] [rbp-278h]
  __int64 v261; // [rsp+48h] [rbp-278h]
  _QWORD v262[2]; // [rsp+50h] [rbp-270h] BYREF
  unsigned __int64 v263[2]; // [rsp+60h] [rbp-260h] BYREF
  __int64 v264[2]; // [rsp+70h] [rbp-250h] BYREF
  unsigned __int8 *v265; // [rsp+80h] [rbp-240h] BYREF
  size_t v266; // [rsp+88h] [rbp-238h]
  _QWORD v267[2]; // [rsp+90h] [rbp-230h] BYREF
  _QWORD v268[68]; // [rsp+A0h] [rbp-220h] BYREF

  v5 = *((_QWORD *)a2 + 4);
  v235 = a2 + 24;
  v260 = a2;
  for ( i = a2 + 24; (char *)v5 != i; v5 = *(_QWORD *)(v5 + 8) )
  {
    a2 = (char *)(v5 - 56);
    if ( !v5 )
      a2 = 0;
    sub_2C797D0(a1, (__int64)a2);
  }
  for ( j = (char *)*((_QWORD *)v260 + 2); j != v260 + 8; j = (char *)*((_QWORD *)j + 1) )
  {
    a2 = j - 56;
    if ( !j )
      a2 = 0;
    sub_2C7A130(a1, a2, a3);
  }
  v8 = *((_QWORD *)v260 + 96);
  if ( !v8 )
  {
    v190 = *(_QWORD *)(a1 + 24);
    v191 = *(_QWORD *)(v190 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v190 + 24) - v191) <= 6 )
    {
      a2 = "Error: ";
      sub_CB6200(v190, (unsigned __int8 *)"Error: ", 7u);
    }
    else
    {
      *(_DWORD *)v191 = 1869771333;
      *(_WORD *)(v191 + 4) = 14962;
      *(_BYTE *)(v191 + 6) = 32;
      *(_QWORD *)(v190 + 32) += 7LL;
    }
    v192 = *(_QWORD *)(a1 + 24);
    v193 = *(__m128i **)(v192 + 32);
    v194 = *(_QWORD *)(v192 + 24) - (_QWORD)v193;
    if ( v194 <= 0x24 )
    {
      a2 = "Empty target data layout, must exist\n";
      sub_CB6200(v192, "Empty target data layout, must exist\n", 0x25u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42D0C90);
      v193[2].m128i_i32[0] = 1953720696;
      v193[2].m128i_i8[4] = 10;
      *v193 = si128;
      v193[1] = _mm_load_si128((const __m128i *)&xmmword_42D0CA0);
      *(_QWORD *)(v192 + 32) += 37LL;
    }
    v196 = *(_BYTE **)(a1 + 16);
    if ( v196 )
      *v196 = 0;
    if ( !*(_DWORD *)(a1 + 4) )
    {
      v197 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v197 + 32) != *(_QWORD *)(v197 + 16) )
      {
        sub_CB5AE0((__int64 *)v197);
        v197 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v197 + 48), (__int64)a2, v194, a4);
    }
    v8 = *((_QWORD *)v260 + 96);
  }
  sub_AE3F70(v268, *((_BYTE **)v260 + 95), v8);
  if ( !*(_DWORD *)a1 )
  {
    sub_2C74F70(v263, (__int64)v268, *(_QWORD **)v260, 0, 1, 1);
    if ( (v263[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      goto LABEL_12;
    v43 = *(_QWORD *)(a1 + 24);
    v263[0] = v263[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_904010(v43, "Error: ");
    v44 = *(_QWORD *)(a1 + 24);
    v45 = v263[0];
    v263[0] = 0;
    v264[0] = v45 | 1;
    sub_C64870((__int64)&v265, v264);
    v31 = v265;
    sub_CB6200(v44, v265, v266);
    if ( v265 != (unsigned __int8 *)v267 )
    {
      v31 = (unsigned __int8 *)(v267[0] + 1LL);
      j_j___libc_free_0((unsigned __int64)v265);
    }
    if ( (v264[0] & 1) == 0 && (v264[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
      v46 = *(_QWORD *)(a1 + 24);
      v47 = *(__m128i **)(v46 + 32);
      if ( *(_QWORD *)(v46 + 24) - (_QWORD)v47 <= 0x1Bu )
      {
        sub_CB6200(v46, "\nExample valid data layout:\n", 0x1Cu);
      }
      else
      {
        v48 = _mm_load_si128((const __m128i *)&xmmword_4281920);
        qmemcpy(&v47[1], "ata layout:\n", 12);
        *v47 = v48;
        *(_QWORD *)(v46 + 32) += 28LL;
      }
      sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
      v49 = *(_QWORD *)(a1 + 24);
      v50 = *(_QWORD **)(v49 + 32);
      if ( *(_QWORD *)(v49 + 24) - (_QWORD)v50 <= 7u )
      {
        v49 = sub_CB6200(*(_QWORD *)(a1 + 24), "32-bit: ", 8u);
      }
      else
      {
        *v50 = 0x203A7469622D3233LL;
        *(_QWORD *)(v49 + 32) += 8LL;
      }
      v51 = off_4C5D0A0[0];
      if ( off_4C5D0A0[0] )
      {
        v52 = strlen(off_4C5D0A0[0]);
        v53 = *(_BYTE **)(v49 + 32);
        v54 = v52;
        v55 = *(_BYTE **)(v49 + 24);
        if ( v54 <= v55 - v53 )
        {
          if ( v54 )
          {
            memcpy(v53, v51, v54);
            v55 = *(_BYTE **)(v49 + 24);
            v53 = (_BYTE *)(v54 + *(_QWORD *)(v49 + 32));
            *(_QWORD *)(v49 + 32) = v53;
          }
LABEL_296:
          if ( v53 == v55 )
          {
            sub_CB6200(v49, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v53 = 10;
            ++*(_QWORD *)(v49 + 32);
          }
          v217 = "Error: ";
          sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
          v219 = *(_QWORD *)(a1 + 24);
          v220 = *(_QWORD **)(v219 + 32);
          if ( *(_QWORD *)(v219 + 24) - (_QWORD)v220 <= 7u )
          {
            v217 = "64-bit: ";
            v219 = sub_CB6200(*(_QWORD *)(a1 + 24), "64-bit: ", 8u);
          }
          else
          {
            *v220 = 0x203A7469622D3436LL;
            *(_QWORD *)(v219 + 32) += 8LL;
          }
          v221 = off_4C5D0A8[0];
          if ( !off_4C5D0A8[0] )
            goto LABEL_310;
          goto LABEL_301;
        }
        v49 = sub_CB6200(v49, (unsigned __int8 *)v51, v54);
      }
      v55 = *(_BYTE **)(v49 + 24);
      v53 = *(_BYTE **)(v49 + 32);
      goto LABEL_296;
    }
LABEL_328:
    sub_C63C30(v264, (__int64)v31);
  }
  v23 = *(_QWORD *)(a1 + 24);
  v24 = *(_QWORD *)(v23 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v23 + 24) - v24) <= 6 )
  {
    sub_CB6200(v23, (unsigned __int8 *)"Error: ", 7u);
  }
  else
  {
    *(_DWORD *)v24 = 1869771333;
    *(_WORD *)(v24 + 4) = 14962;
    *(_BYTE *)(v24 + 6) = 32;
    *(_QWORD *)(v23 + 32) += 7LL;
  }
  v25 = *(_QWORD *)(a1 + 24);
  v26 = *(__m128i **)(v25 + 32);
  if ( *(_QWORD *)(v25 + 24) - (_QWORD)v26 <= 0x18u )
  {
    sub_CB6200(v25, "IR Kind is UnifiedNVVMIR\n", 0x19u);
  }
  else
  {
    v27 = _mm_load_si128((const __m128i *)&xmmword_42D0CB0);
    v26[1].m128i_i8[8] = 10;
    v26[1].m128i_i64[0] = 0x52494D56564E6465LL;
    *v26 = v27;
    *(_QWORD *)(v25 + 32) += 25LL;
  }
  sub_2C74F70(v263, (__int64)v268, *(_QWORD **)v260, 0, 0, 0);
  if ( (v263[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_12;
  v28 = *(_QWORD *)(a1 + 24);
  v263[0] = v263[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  sub_904010(v28, "Error: ");
  v29 = *(_QWORD *)(a1 + 24);
  v30 = v263[0];
  v263[0] = 0;
  v264[0] = v30 | 1;
  sub_C64870((__int64)&v265, v264);
  v31 = v265;
  sub_CB6200(v29, v265, v266);
  if ( v265 != (unsigned __int8 *)v267 )
  {
    v31 = (unsigned __int8 *)(v267[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v265);
  }
  if ( (v264[0] & 1) != 0 || (v264[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_328;
  sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
  v32 = *(_QWORD *)(a1 + 24);
  v33 = *(__m128i **)(v32 + 32);
  if ( *(_QWORD *)(v32 + 24) - (_QWORD)v33 <= 0x1Bu )
  {
    sub_CB6200(v32, "\nExample valid data layout:\n", 0x1Cu);
  }
  else
  {
    v34 = _mm_load_si128((const __m128i *)&xmmword_4281920);
    qmemcpy(&v33[1], "ata layout:\n", 12);
    *v33 = v34;
    *(_QWORD *)(v32 + 32) += 28LL;
  }
  sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
  v35 = *(_QWORD *)(a1 + 24);
  v36 = *(_QWORD **)(v35 + 32);
  if ( *(_QWORD *)(v35 + 24) - (_QWORD)v36 <= 7u )
  {
    v35 = sub_CB6200(*(_QWORD *)(a1 + 24), "32-bit: ", 8u);
  }
  else
  {
    *v36 = 0x203A7469622D3233LL;
    *(_QWORD *)(v35 + 32) += 8LL;
  }
  v37 = (unsigned __int8 *)off_4C5D080[0];
  if ( !off_4C5D080[0] )
    goto LABEL_284;
  v38 = strlen(off_4C5D080[0]);
  v39 = *(_BYTE **)(v35 + 32);
  v40 = v38;
  v41 = *(_BYTE **)(v35 + 24);
  if ( v40 > v41 - v39 )
  {
    v35 = sub_CB6200(v35, v37, v40);
LABEL_284:
    v41 = *(_BYTE **)(v35 + 24);
    v39 = *(_BYTE **)(v35 + 32);
    goto LABEL_285;
  }
  if ( v40 )
  {
    memcpy(v39, v37, v40);
    v41 = *(_BYTE **)(v35 + 24);
    v39 = (_BYTE *)(v40 + *(_QWORD *)(v35 + 32));
    *(_QWORD *)(v35 + 32) = v39;
  }
LABEL_285:
  if ( v41 == v39 )
  {
    sub_CB6200(v35, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v39 = 10;
    ++*(_QWORD *)(v35 + 32);
  }
  sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
  v209 = *(_QWORD *)(a1 + 24);
  v210 = *(__m128i **)(v209 + 32);
  if ( *(_QWORD *)(v209 + 24) - (_QWORD)v210 <= 0x18u )
  {
    v209 = sub_CB6200(*(_QWORD *)(a1 + 24), "64-bit (mixed pointers): ", 0x19u);
  }
  else
  {
    v211 = _mm_load_si128((const __m128i *)&xmmword_42D0CC0);
    v210[1].m128i_i8[8] = 32;
    v210[1].m128i_i64[0] = 0x3A29737265746E69LL;
    *v210 = v211;
    *(_QWORD *)(v209 + 32) += 25LL;
  }
  v212 = (unsigned __int8 *)off_4C5D070[0];
  if ( !off_4C5D070[0] )
    goto LABEL_313;
  v213 = strlen(off_4C5D070[0]);
  v214 = *(_BYTE **)(v209 + 32);
  v215 = v213;
  v216 = *(_BYTE **)(v209 + 24);
  if ( v215 > v216 - v214 )
  {
    v209 = sub_CB6200(v209, v212, v215);
LABEL_313:
    v216 = *(_BYTE **)(v209 + 24);
    v214 = *(_BYTE **)(v209 + 32);
    goto LABEL_314;
  }
  if ( v215 )
  {
    memcpy(v214, v212, v215);
    v216 = *(_BYTE **)(v209 + 24);
    v214 = (_BYTE *)(v215 + *(_QWORD *)(v209 + 32));
    *(_QWORD *)(v209 + 32) = v214;
  }
LABEL_314:
  if ( v214 == v216 )
  {
    sub_CB6200(v209, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v214 = 10;
    ++*(_QWORD *)(v209 + 32);
  }
  v217 = "Error: ";
  sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
  v219 = *(_QWORD *)(a1 + 24);
  v220 = *(_QWORD **)(v219 + 32);
  if ( *(_QWORD *)(v219 + 24) - (_QWORD)v220 <= 7u )
  {
    v217 = "64-bit: ";
    v219 = sub_CB6200(*(_QWORD *)(a1 + 24), "64-bit: ", 8u);
  }
  else
  {
    *v220 = 0x203A7469622D3436LL;
    *(_QWORD *)(v219 + 32) += 8LL;
  }
  v221 = off_4C5D078[0];
  if ( !off_4C5D078[0] )
    goto LABEL_310;
LABEL_301:
  v222 = strlen(v221);
  v223 = *(_BYTE **)(v219 + 32);
  v224 = v222;
  v225 = *(_BYTE **)(v219 + 24);
  v220 = (_QWORD *)(v225 - v223);
  if ( v224 <= v225 - v223 )
  {
    if ( v224 )
    {
      v217 = v221;
      memcpy(v223, v221, v224);
      v225 = *(_BYTE **)(v219 + 24);
      v223 = (_BYTE *)(v224 + *(_QWORD *)(v219 + 32));
      *(_QWORD *)(v219 + 32) = v223;
    }
    goto LABEL_304;
  }
  v217 = v221;
  v219 = sub_CB6200(v219, (unsigned __int8 *)v221, v224);
LABEL_310:
  v225 = *(_BYTE **)(v219 + 24);
  v223 = *(_BYTE **)(v219 + 32);
LABEL_304:
  if ( v223 == v225 )
  {
    v217 = "\n";
    sub_CB6200(v219, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v223 = 10;
    ++*(_QWORD *)(v219 + 32);
  }
  sub_2C76240(a1, (__int64)v217, (__int64)v220, v218);
  if ( (v263[0] & 1) != 0 || (v263[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(v263, (__int64)v217);
LABEL_12:
  v265 = (unsigned __int8 *)v267;
  v10 = (_BYTE *)*((_QWORD *)v260 + 29);
  v11 = *((_QWORD *)v260 + 30);
  if ( &v10[v11] && !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v264[0] = *((_QWORD *)v260 + 30);
  if ( v11 > 0xF )
  {
    v265 = (unsigned __int8 *)sub_22409D0((__int64)&v265, (unsigned __int64 *)v264, 0);
    v42 = v265;
    v267[0] = v264[0];
LABEL_51:
    memcpy(v42, v10, v11);
    v11 = v264[0];
    v12 = v265;
    goto LABEL_17;
  }
  if ( v11 == 1 )
  {
    LOBYTE(v267[0]) = *v10;
    v12 = (unsigned __int8 *)v267;
    goto LABEL_17;
  }
  if ( v11 )
  {
    v42 = (unsigned __int8 *)v267;
    goto LABEL_51;
  }
  v12 = (unsigned __int8 *)v267;
LABEL_17:
  v266 = v11;
  v12[v11] = 0;
  if ( !v266 )
  {
    v201 = *(_QWORD *)(a1 + 24);
    v202 = *(_QWORD *)(v201 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v201 + 24) - v202) <= 6 )
    {
      sub_CB6200(v201, (unsigned __int8 *)"Error: ", 7u);
    }
    else
    {
      *(_DWORD *)v202 = 1869771333;
      *(_WORD *)(v202 + 4) = 14962;
      *(_BYTE *)(v202 + 6) = 32;
      *(_QWORD *)(v201 + 32) += 7LL;
    }
    v203 = *(_QWORD *)(a1 + 24);
    v204 = *(__m128i **)(v203 + 32);
    v205 = *(_QWORD *)(v203 + 24) - (_QWORD)v204;
    if ( v205 <= 0x1F )
    {
      sub_CB6200(v203, "Empty target triple, must exist\n", 0x20u);
    }
    else
    {
      *v204 = _mm_load_si128((const __m128i *)&xmmword_42D0CD0);
      v204[1] = _mm_load_si128((const __m128i *)&xmmword_42D0CE0);
      *(_QWORD *)(v203 + 32) += 32LL;
    }
    v206 = *(_BYTE **)(a1 + 16);
    if ( v206 )
      *v206 = 0;
    v207 = *(unsigned int *)(a1 + 4);
    if ( !(_DWORD)v207 )
    {
      v208 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v208 + 32) != *(_QWORD *)(v208 + 16) )
      {
        sub_CB5AE0((__int64 *)v208);
        v208 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v208 + 48), v207, v205, v9);
    }
  }
  if ( *(_DWORD *)a1 == 1 )
  {
    v13 = "nvptx-nvidia-cuda";
    if ( sub_2241AC0((__int64)&v265, "nvptx-nvidia-cuda") )
    {
      v13 = "nvptx64-nvidia-cuda";
      if ( sub_2241AC0((__int64)&v265, "nvptx64-nvidia-cuda") )
      {
        v13 = "nvptx-nvidia-nvcl";
        if ( sub_2241AC0((__int64)&v265, "nvptx-nvidia-nvcl") )
        {
          v13 = "nvptx64-nvidia-nvcl";
          if ( sub_2241AC0((__int64)&v265, "nvptx64-nvidia-nvcl") )
          {
            v13 = "nvsass-nvidia-cuda";
            if ( sub_2241AC0((__int64)&v265, "nvsass-nvidia-cuda") )
            {
              v13 = "nvsass-nvidia-nvcl";
              if ( sub_2241AC0((__int64)&v265, "nvsass-nvidia-nvcl") )
              {
                v13 = "nvsass-nvidia-directx";
                if ( sub_2241AC0((__int64)&v265, "nvsass-nvidia-directx") )
                {
                  v13 = "nvsass-nvidia-spirv";
                  if ( sub_2241AC0((__int64)&v265, "nvsass-nvidia-spirv") )
                  {
                    sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
                    v13 = "Invalid target triple\n";
                    sub_904010(*(_QWORD *)(a1 + 24), "Invalid target triple\n");
                    sub_2C76240(a1, (__int64)"Invalid target triple\n", v199, v200);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  else if ( sub_22416F0((__int64 *)&v265, "nvptx-", 0, 6u) && sub_22416F0((__int64 *)&v265, "nvptx64-", 0, 8u)
         || (v13 = "-cuda", sub_2241820((__int64 *)&v265, "-cuda", 0xFFFFFFFFFFFFFFFFLL, 5u) == -1) )
  {
    v125 = *(_QWORD *)(a1 + 24);
    v126 = *(_QWORD *)(v125 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v125 + 24) - v126) <= 6 )
    {
      sub_CB6200(v125, (unsigned __int8 *)"Error: ", 7u);
    }
    else
    {
      *(_DWORD *)v126 = 1869771333;
      *(_WORD *)(v126 + 4) = 14962;
      *(_BYTE *)(v126 + 6) = 32;
      *(_QWORD *)(v125 + 32) += 7LL;
    }
    v127 = *(_QWORD *)(a1 + 24);
    v128 = *(__m128i **)(v127 + 32);
    if ( *(_QWORD *)(v127 + 24) - (_QWORD)v128 <= 0x16u )
    {
      v127 = sub_CB6200(v127, "Invalid target triple (", 0x17u);
    }
    else
    {
      v129 = _mm_load_si128((const __m128i *)&xmmword_42D0CF0);
      v128[1].m128i_i32[0] = 1819306354;
      v128[1].m128i_i16[2] = 8293;
      v128[1].m128i_i8[6] = 40;
      *v128 = v129;
      *(_QWORD *)(v127 + 32) += 23LL;
    }
    v13 = (const char *)v265;
    v130 = sub_CB6200(v127, v265, v266);
    v131 = *(__m128i **)(v130 + 32);
    if ( *(_QWORD *)(v130 + 24) - (_QWORD)v131 <= 0x12u )
    {
      v13 = "), must be one of:\n";
      sub_CB6200(v130, "), must be one of:\n", 0x13u);
    }
    else
    {
      v132 = _mm_load_si128((const __m128i *)&xmmword_42D0D00);
      v131[1].m128i_i8[2] = 10;
      v131[1].m128i_i16[0] = 14950;
      *v131 = v132;
      *(_QWORD *)(v130 + 32) += 19LL;
    }
    v133 = *(_QWORD *)(a1 + 24);
    v134 = *(_QWORD *)(v133 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v133 + 24) - v134) <= 6 )
    {
      v13 = "Error: ";
      sub_CB6200(v133, (unsigned __int8 *)"Error: ", 7u);
    }
    else
    {
      *(_DWORD *)v134 = 1869771333;
      *(_WORD *)(v134 + 4) = 14962;
      *(_BYTE *)(v134 + 6) = 32;
      *(_QWORD *)(v133 + 32) += 7LL;
    }
    v135 = *(_QWORD *)(a1 + 24);
    v136 = *(_QWORD **)(v135 + 32);
    if ( *(_QWORD *)(v135 + 24) - (_QWORD)v136 <= 7u )
    {
      v13 = "32-bit: ";
      v135 = sub_CB6200(v135, "32-bit: ", 8u);
      v137 = *(void **)(v135 + 32);
    }
    else
    {
      *v136 = 0x203A7469622D3233LL;
      v137 = (void *)(*(_QWORD *)(v135 + 32) + 8LL);
      *(_QWORD *)(v135 + 32) = v137;
    }
    if ( *(_QWORD *)(v135 + 24) - (_QWORD)v137 <= 0xBu )
    {
      v13 = "nvptx-*-cuda";
      v135 = sub_CB6200(v135, "nvptx-*-cuda", 0xCu);
      v138 = *(_BYTE **)(v135 + 32);
    }
    else
    {
      qmemcpy(v137, "nvptx-*-cuda", 12);
      v138 = (_BYTE *)(*(_QWORD *)(v135 + 32) + 12LL);
      *(_QWORD *)(v135 + 32) = v138;
    }
    if ( *(_BYTE **)(v135 + 24) == v138 )
    {
      v13 = "\n";
      sub_CB6200(v135, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v138 = 10;
      ++*(_QWORD *)(v135 + 32);
    }
    v139 = *(_QWORD *)(a1 + 24);
    v140 = *(_QWORD *)(v139 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v139 + 24) - v140) <= 6 )
    {
      v13 = "Error: ";
      sub_CB6200(v139, (unsigned __int8 *)"Error: ", 7u);
    }
    else
    {
      *(_DWORD *)v140 = 1869771333;
      *(_WORD *)(v140 + 4) = 14962;
      *(_BYTE *)(v140 + 6) = 32;
      *(_QWORD *)(v139 + 32) += 7LL;
    }
    v141 = *(_QWORD *)(a1 + 24);
    v142 = *(_QWORD **)(v141 + 32);
    if ( *(_QWORD *)(v141 + 24) - (_QWORD)v142 <= 7u )
    {
      v13 = "64-bit: ";
      v141 = sub_CB6200(v141, "64-bit: ", 8u);
      v143 = *(void **)(v141 + 32);
    }
    else
    {
      *v142 = 0x203A7469622D3436LL;
      v143 = (void *)(*(_QWORD *)(v141 + 32) + 8LL);
      *(_QWORD *)(v141 + 32) = v143;
    }
    v144 = *(_QWORD *)(v141 + 24) - (_QWORD)v143;
    if ( v144 <= 0xD )
    {
      v13 = "nvptx64-*-cuda";
      v141 = sub_CB6200(v141, "nvptx64-*-cuda", 0xEu);
      v146 = *(_BYTE **)(v141 + 32);
    }
    else
    {
      v145 = 0x2D3436787470766ELL;
      qmemcpy(v143, "nvptx64-*-cuda", 14);
      v146 = (_BYTE *)(*(_QWORD *)(v141 + 32) + 14LL);
      *(_QWORD *)(v141 + 32) = v146;
    }
    if ( *(_BYTE **)(v141 + 24) == v146 )
    {
      v13 = "\n";
      sub_CB6200(v141, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v146 = 10;
      ++*(_QWORD *)(v141 + 32);
    }
    v147 = *(_BYTE **)(a1 + 16);
    if ( v147 )
      *v147 = 0;
    if ( !*(_DWORD *)(a1 + 4) )
    {
      v148 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v148 + 32) != *(_QWORD *)(v148 + 16) )
      {
        sub_CB5AE0((__int64 *)v148);
        v148 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v148 + 48), (__int64)v13, v144, (char *)v145);
    }
  }
  for ( k = *((_QWORD *)v260 + 10); v260 + 72 != (char *)k; k = *(_QWORD *)(k + 8) )
  {
    v15 = 0;
    v16 = sub_B91A00(k);
    if ( v16 )
    {
      do
      {
        v13 = (const char *)sub_B91A10(k, v15);
        if ( v13 )
          sub_2C7AA20(a1, v13);
        ++v15;
      }
      while ( v16 != v15 );
    }
  }
  v17 = a1;
  v258 = (char *)*((_QWORD *)v260 + 4);
  if ( v235 != v258 )
  {
    while ( 1 )
    {
      v18 = v258;
      v13 = v258 - 56;
      v258 = (char *)*((_QWORD *)v258 + 1);
      sub_2C771D0(v17, v13);
      v259 = v18 + 16;
      v261 = *((_QWORD *)v18 + 3);
      if ( v18 + 16 != (char *)v261 )
        break;
LABEL_79:
      if ( v235 == v258 )
        goto LABEL_80;
    }
    while ( 1 )
    {
      v19 = *(_QWORD *)(v261 + 32);
      v20 = v261 + 24;
      v261 = *(_QWORD *)(v261 + 8);
      while ( v20 != v19 )
      {
LABEL_32:
        v21 = v19;
        v19 = *(_QWORD *)(v19 + 8);
        v22 = v21 - 24;
        switch ( *(_BYTE *)(v21 - 24) )
        {
          case 0x1E:
          case 0x20:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3F:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x5E:
          case 0x60:
            goto LABEL_76;
          case 0x1F:
            if ( !*(_QWORD *)(v21 + 24) && (*(_BYTE *)(v21 - 17) & 0x20) == 0 )
            {
              v13 = (const char *)(v21 - 24);
              sub_2C795F0(v17, v21 - 24);
              goto LABEL_77;
            }
            v114 = "pragma";
            v247 = v21 - 24;
            v115 = sub_B91F50(v21 - 24, "pragma", 6u);
            v22 = v21 - 24;
            v116 = (const char *)v115;
            if ( !v115 )
              goto LABEL_76;
            v117 = *(_BYTE *)(v115 - 16);
            if ( (v117 & 2) != 0 )
              v118 = *(_DWORD *)(v115 - 24);
            else
              v118 = (*(_WORD *)(v115 - 16) >> 6) & 0xF;
            if ( v118 != 2 )
            {
              v119 = *(_QWORD *)(v17 + 24);
              v120 = *(_QWORD *)(v119 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v119 + 24) - v120) <= 6 )
              {
                sub_CB6200(v119, (unsigned __int8 *)"Error: ", 7u);
                v22 = v247;
              }
              else
              {
                *(_DWORD *)v120 = 1869771333;
                *(_WORD *)(v120 + 4) = 14962;
                *(_BYTE *)(v120 + 6) = 32;
                *(_QWORD *)(v119 + 32) += 7LL;
              }
              v121 = *(_QWORD *)(v17 + 24);
              v122 = *(_WORD **)(v121 + 32);
              if ( *(_QWORD *)(v121 + 24) - (_QWORD)v122 <= 1u )
              {
                v255 = v22;
                sub_CB6200(v121, (unsigned __int8 *)": ", 2u);
                v22 = v255;
              }
              else
              {
                *v122 = 8250;
                *(_QWORD *)(v121 + 32) += 2LL;
              }
              v248 = v22;
              sub_A61DE0(v116, *(_QWORD *)(v17 + 24), 0);
              v123 = *(_QWORD *)(v17 + 24);
              v22 = v248;
              v124 = *(_QWORD *)(v123 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v123 + 24) - v124) <= 2 )
              {
                v58 = "\n  ";
                sub_CB6200(v123, "\n  ", 3u);
                v22 = v248;
              }
              else
              {
                v58 = (char *)8202;
                *(_BYTE *)(v124 + 2) = 32;
                *(_WORD *)v124 = 8202;
                *(_QWORD *)(v123 + 32) += 3LL;
              }
              v99 = *(_QWORD *)(v17 + 24);
              v62 = *(__m128i **)(v99 + 32);
              if ( *(_QWORD *)(v99 + 24) - (_QWORD)v62 <= 0x2Fu )
              {
                v58 = "branch pragma metadata does not have 2 operands?";
                v254 = v22;
                v188 = sub_CB6200(v99, "branch pragma metadata does not have 2 operands?", 0x30u);
                v22 = v254;
                v99 = v188;
                v100 = *(_BYTE **)(v188 + 32);
              }
              else
              {
                *v62 = _mm_load_si128((const __m128i *)&xmmword_42D0D10);
                v62[1] = _mm_load_si128((const __m128i *)&xmmword_42D0D20);
                v62[2] = _mm_load_si128((const __m128i *)&xmmword_42D0D30);
                v100 = (_BYTE *)(*(_QWORD *)(v99 + 32) + 48LL);
                *(_QWORD *)(v99 + 32) = v100;
              }
              goto LABEL_141;
            }
            if ( (*(_BYTE *)(v115 + 1) & 0x7F) == 1 )
            {
              if ( (v117 & 2) != 0 )
                v187 = *(_QWORD *)(v115 - 32);
              else
                v187 = v115 - 8LL * ((v117 >> 2) & 0xF) - 16;
              v116 = *(const char **)(v187 + 8);
              if ( (unsigned __int8)(*v116 - 5) > 0x1Fu )
                goto LABEL_76;
              if ( (*(v116 - 16) & 2) != 0 )
              {
LABEL_203:
                v163 = (const char *)*((_QWORD *)v116 - 4);
                goto LABEL_204;
              }
            }
            else if ( (v117 & 2) != 0 )
            {
              goto LABEL_203;
            }
            v163 = &v116[-8 * (((unsigned __int8)*(v116 - 16) >> 2) & 0xF) - 16];
LABEL_204:
            v164 = *(_BYTE **)v163;
            if ( !*(_QWORD *)v163 )
              goto LABEL_76;
            if ( *v164 )
              goto LABEL_76;
            v165 = sub_B91420((__int64)v164);
            v22 = v247;
            if ( v166 != 6 || *(_DWORD *)v165 != 1869770357 || *(_WORD *)(v165 + 4) != 27756 )
              goto LABEL_76;
            v167 = *(_QWORD *)(v17 + 24);
            v168 = *(_QWORD *)(v167 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v167 + 24) - v168) <= 6 )
            {
              v114 = "Error: ";
              sub_CB6200(v167, (unsigned __int8 *)"Error: ", 7u);
              v22 = v247;
            }
            else
            {
              *(_DWORD *)v168 = 1869771333;
              *(_WORD *)(v168 + 4) = 14962;
              *(_BYTE *)(v168 + 6) = 32;
              *(_QWORD *)(v167 + 32) += 7LL;
            }
            v169 = *(_QWORD *)(v17 + 24);
            v170 = *(__m128i **)(v169 + 32);
            if ( *(_QWORD *)(v169 + 24) - (_QWORD)v170 <= 0x49u )
            {
              v114 = "pragma unroll is not supported.  Please use llvm.loop.unroll.count instead";
              v257 = v22;
              sub_CB6200(v169, "pragma unroll is not supported.  Please use llvm.loop.unroll.count instead", 0x4Au);
              v22 = v257;
            }
            else
            {
              v171 = _mm_load_si128((const __m128i *)&xmmword_42D0D40);
              v172 = 25697;
              v173 = 0x6574736E6920746ELL;
              qmemcpy(&v170[4], "nt instead", 10);
              *v170 = v171;
              v170[1] = _mm_load_si128((const __m128i *)&xmmword_42D0D50);
              v170[2] = _mm_load_si128((const __m128i *)&xmmword_42D0D60);
              v170[3] = _mm_load_si128((const __m128i *)&xmmword_42D0D70);
              *(_QWORD *)(v169 + 32) += 74LL;
            }
            v252 = v22;
            sub_2C76240(v17, (__int64)v114, v172, (char *)v173);
            v13 = (const char *)v252;
            sub_2C795F0(v17, v252);
            goto LABEL_77;
          case 0x21:
            v13 = "indirectbr";
            sub_2C76F10(v17, "indirectbr", v21 - 24);
            continue;
          case 0x22:
            v13 = "invoke";
            sub_2C76F10(v17, "invoke", v21 - 24);
            continue;
          case 0x23:
            v13 = "resume";
            sub_2C76F10(v17, "resume", v21 - 24);
            continue;
          case 0x3C:
            _BitScanReverse64(&v72, 1LL << *(_WORD *)(v21 - 22));
            if ( 0x8000000000000000LL >> ((unsigned __int8)v72 ^ 0x3Fu) > 0x800000 )
            {
              v180 = sub_2C76A00(v17, v21 - 24, 0);
              sub_904010(v180, "alloca align must be <= 2^23\n");
              sub_2C76240(v17, (__int64)"alloca align must be <= 2^23\n", v181, v182);
              v22 = v21 - 24;
            }
            if ( !(*(_DWORD *)(*(_QWORD *)(v21 - 16) + 8LL) >> 8) )
              goto LABEL_76;
            v239 = v22;
            v73 = sub_2C76A00(v17, v22, 0);
            sub_904010(v73, "Allocas are not supported on address spaces except Generic\n");
            sub_2C76240(v17, (__int64)"Allocas are not supported on address spaces except Generic\n", v74, v75);
            v13 = (const char *)v239;
            sub_2C795F0(v17, v239);
            goto LABEL_77;
          case 0x3D:
          case 0x3E:
            v56 = sub_B46500((unsigned __int8 *)(v21 - 24));
            v22 = v21 - 24;
            if ( v56 )
            {
              v149 = (const char *)(v21 - 24);
              v150 = sub_2C76A00(v17, v21 - 24, 0);
              v22 = v21 - 24;
              v152 = *(_QWORD *)(v150 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v150 + 24) - v152) <= 0x25 )
              {
                v149 = "Atomic loads/stores are not supported\n";
                sub_CB6200(v150, "Atomic loads/stores are not supported\n", 0x26u);
                v22 = v21 - 24;
              }
              else
              {
                v153 = _mm_load_si128((const __m128i *)&xmmword_43A2660);
                *(_DWORD *)(v152 + 32) = 1702130287;
                *(_WORD *)(v152 + 36) = 2660;
                *(__m128i *)v152 = v153;
                *(__m128i *)(v152 + 16) = _mm_load_si128((const __m128i *)&xmmword_43A2670);
                *(_QWORD *)(v150 + 32) += 38LL;
              }
              v154 = *(_BYTE **)(v17 + 16);
              if ( v154 )
                *v154 = 0;
              if ( !*(_DWORD *)(v17 + 4) )
              {
                v155 = *(_QWORD *)(v17 + 24);
                if ( *(_QWORD *)(v155 + 32) != *(_QWORD *)(v155 + 16) )
                {
                  v249 = v22;
                  sub_CB5AE0((__int64 *)v155);
                  v155 = *(_QWORD *)(v17 + 24);
                  v22 = v249;
                }
                v250 = v22;
                sub_CEB520(*(_QWORD **)(v155 + 48), (__int64)v149, v152, v151);
                v22 = v250;
              }
            }
            v57 = *(_QWORD *)(*(_QWORD *)(v21 - 56) + 8LL);
            if ( (unsigned int)*(unsigned __int8 *)(v57 + 8) - 17 <= 1 )
              v57 = **(_QWORD **)(v57 + 16);
            if ( *(_DWORD *)(v57 + 8) >> 8 != 6 )
              goto LABEL_76;
            v58 = (char *)v22;
            v236 = v22;
            v59 = sub_2C76A00(v17, v22, 0);
            v22 = v236;
            v60 = v59;
            v61 = *(__m128i **)(v59 + 32);
            v62 = (__m128i *)(*(_QWORD *)(v60 + 24) - (_QWORD)v61);
            if ( (unsigned __int64)v62 <= 0x2C )
            {
              v58 = "Tensor Memory loads/stores are not supported\n";
              sub_CB6200(v60, "Tensor Memory loads/stores are not supported\n", 0x2Du);
              v22 = v236;
            }
            else
            {
              v63 = _mm_load_si128((const __m128i *)&xmmword_43A2680);
              v64 = 0x6F7070757320746FLL;
              qmemcpy(&v61[2], "ot supported\n", 13);
              *v61 = v63;
              v61[1] = _mm_load_si128((const __m128i *)&xmmword_43A2690);
              *(_QWORD *)(v60 + 32) += 45LL;
            }
            goto LABEL_70;
          case 0x40:
            if ( *(_DWORD *)v17 == 1 )
            {
              if ( (unsigned __int16)((*(_WORD *)(v21 - 22) & 7) - 6) <= 1u )
              {
LABEL_76:
                v13 = (const char *)v22;
                sub_2C795F0(v17, v22);
              }
              else
              {
                v68 = sub_2C76A00(v17, v21 - 24, 0);
                v69 = sub_904010(v68, "Invalid ordering for fence, only acq_rel and seq_cst are supported.");
                sub_904010(v69, "\n");
                sub_2C76240(v17, (__int64)"\n", v70, v71);
                v13 = (const char *)(v21 - 24);
                sub_2C795F0(v17, v21 - 24);
              }
LABEL_77:
              if ( v20 == v19 )
                goto LABEL_78;
              goto LABEL_32;
            }
            v13 = "fence";
            sub_2C76F10(v17, "fence", v21 - 24);
            break;
          case 0x41:
            v101 = *(_QWORD *)(v21 - 56);
            if ( *(_BYTE *)(*(_QWORD *)(v101 + 8) + 8LL) == 12 )
            {
              v232 = *(_QWORD *)(v101 + 8);
              v264[0] = sub_BCAE30(v232);
              v264[1] = v174;
              v175 = sub_CA1930(v264);
              v22 = v21 - 24;
              if ( v175 == 32 )
                goto LABEL_138;
              v263[0] = sub_BCAE30(v232);
              v263[1] = v176;
              v177 = sub_CA1930(v263);
              v22 = v21 - 24;
              if ( v177 == 64 )
                goto LABEL_138;
              v262[0] = sub_BCAE30(v232);
              v262[1] = v178;
              v179 = sub_CA1930(v262);
              v22 = v21 - 24;
              if ( v179 == 128 )
                goto LABEL_138;
            }
            v102 = (char *)v22;
            v242 = v22;
            v103 = sub_2C76A00(v17, v22, 0);
            v22 = v242;
            v105 = *(__m128i **)(v103 + 32);
            v106 = v103;
            if ( *(_QWORD *)(v103 + 24) - (_QWORD)v105 <= 0x2Fu )
            {
              v102 = "Atomic operations on non-i32/i64/i128 types are ";
              v183 = sub_CB6200(v103, "Atomic operations on non-i32/i64/i128 types are ", 0x30u);
              v22 = v242;
              v106 = v183;
              v107 = *(void **)(v183 + 32);
              if ( *(_QWORD *)(v183 + 24) - (_QWORD)v107 > 0xDu )
              {
LABEL_131:
                qmemcpy(v107, "not supported\n", 14);
                *(_QWORD *)(v106 + 32) += 14LL;
                goto LABEL_132;
              }
            }
            else
            {
              *v105 = _mm_load_si128((const __m128i *)&xmmword_42D0AF0);
              v105[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B00);
              v105[2] = _mm_load_si128((const __m128i *)&xmmword_42D0B10);
              v107 = (void *)(*(_QWORD *)(v103 + 32) + 48LL);
              v108 = *(_QWORD *)(v103 + 24);
              *(_QWORD *)(v106 + 32) = v107;
              if ( (unsigned __int64)(v108 - (_QWORD)v107) > 0xD )
                goto LABEL_131;
            }
            v102 = "not supported\n";
            v253 = v22;
            sub_CB6200(v106, (unsigned __int8 *)"not supported\n", 0xEu);
            v22 = v253;
LABEL_132:
            v109 = *(_BYTE **)(v17 + 16);
            if ( v109 )
              *v109 = 0;
            if ( !*(_DWORD *)(v17 + 4) )
            {
              v110 = *(_QWORD *)(v17 + 24);
              if ( *(_QWORD *)(v110 + 32) != *(_QWORD *)(v110 + 16) )
              {
                v243 = v22;
                sub_CB5AE0((__int64 *)v110);
                v110 = *(_QWORD *)(v17 + 24);
                v22 = v243;
              }
              v244 = v22;
              sub_CEB520(*(_QWORD **)(v110 + 48), (__int64)v102, (__int64)v107, v104);
              v22 = v244;
            }
LABEL_138:
            v111 = *(_QWORD *)(*(_QWORD *)(v21 - 120) + 8LL);
            if ( *(_BYTE *)(v111 + 8) == 14 )
            {
              v156 = *(_DWORD *)(v111 + 8);
              if ( v156 <= 0x1FF || v156 >> 8 == 3 )
                goto LABEL_76;
              v58 = (char *)v22;
              v251 = v22;
              v157 = sub_2C76A00(v17, v22, 0);
              v22 = v251;
              v158 = v157;
              v159 = *(__m128i **)(v157 + 32);
              if ( *(_QWORD *)(v158 + 24) - (_QWORD)v159 <= 0x2Du )
              {
                v58 = "cmpxchg pointer operand must point to generic,";
                v198 = sub_CB6200(v158, "cmpxchg pointer operand must point to generic,", 0x2Eu);
                v22 = v251;
                v158 = v198;
                v161 = *(_QWORD *)(v198 + 32);
              }
              else
              {
                v160 = _mm_load_si128((const __m128i *)&xmmword_42D0B20);
                v64 = 0x6567206F7420746ELL;
                qmemcpy(&v159[2], "nt to generic,", 14);
                *v159 = v160;
                v159[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B40);
                v161 = *(_QWORD *)(v158 + 32) + 46LL;
                *(_QWORD *)(v158 + 32) = v161;
              }
              v62 = (__m128i *)(*(_QWORD *)(v158 + 24) - v161);
              if ( (unsigned __int64)v62 <= 0x20 )
              {
                v58 = " global, or shared address space\n";
                v256 = v22;
                sub_CB6200(v158, (unsigned __int8 *)" global, or shared address space\n", 0x21u);
                v22 = v256;
              }
              else
              {
                v162 = _mm_load_si128((const __m128i *)&xmmword_42D0B50);
                *(_BYTE *)(v161 + 32) = 10;
                *(__m128i *)v161 = v162;
                *(__m128i *)(v161 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0B60);
                *(_QWORD *)(v158 + 32) += 33LL;
              }
            }
            else
            {
              v58 = (char *)v22;
              v245 = v22;
              v112 = sub_2C76A00(v17, v22, 0);
              v22 = v245;
              v62 = *(__m128i **)(v112 + 32);
              v99 = v112;
              if ( *(_QWORD *)(v112 + 24) - (_QWORD)v62 <= 0x25u )
              {
                v58 = "cmpxchg pointer operand not a pointer?";
                v186 = sub_CB6200(v112, "cmpxchg pointer operand not a pointer?", 0x26u);
                v22 = v245;
                v99 = v186;
                v100 = *(_BYTE **)(v186 + 32);
              }
              else
              {
                v113 = _mm_load_si128((const __m128i *)&xmmword_42D0B20);
                v62[2].m128i_i32[0] = 1702129257;
                v62[2].m128i_i16[2] = 16242;
                *v62 = v113;
                v62[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B30);
                v100 = (_BYTE *)(*(_QWORD *)(v112 + 32) + 38LL);
                *(_QWORD *)(v99 + 32) = v100;
              }
LABEL_141:
              if ( *(_BYTE **)(v99 + 24) == v100 )
              {
LABEL_142:
                v58 = "\n";
                v246 = v22;
                sub_CB6200(v99, (unsigned __int8 *)"\n", 1u);
                v22 = v246;
              }
              else
              {
LABEL_126:
                *v100 = 10;
                ++*(_QWORD *)(v99 + 32);
              }
            }
LABEL_70:
            v65 = *(_BYTE **)(v17 + 16);
            if ( v65 )
              *v65 = 0;
            if ( !*(_DWORD *)(v17 + 4) )
            {
              v66 = *(_QWORD *)(v17 + 24);
              if ( *(_QWORD *)(v66 + 32) != *(_QWORD *)(v66 + 16) )
              {
                v237 = v22;
                sub_CB5AE0((__int64 *)v66);
                v66 = *(_QWORD *)(v17 + 24);
                v22 = v237;
              }
              v238 = v22;
              sub_CEB520(*(_QWORD **)(v66 + 48), (__int64)v58, (__int64)v62, (char *)v64);
              v22 = v238;
            }
            goto LABEL_76;
          case 0x42:
            v13 = (const char *)(v21 - 24);
            sub_2C7AF00(v17, (char *)(v21 - 24));
            continue;
          case 0x4F:
            v76 = *(_QWORD *)(*(_QWORD *)(v21 - 56) + 8LL);
            v77 = *(_QWORD *)(v21 - 16);
            if ( (unsigned int)*(unsigned __int8 *)(v76 + 8) - 17 <= 1 )
              v76 = **(_QWORD **)(v76 + 16);
            v78 = *(_DWORD *)(v76 + 8) >> 8;
            if ( (unsigned int)*(unsigned __int8 *)(v77 + 8) - 17 <= 1 )
              v77 = **(_QWORD **)(v77 + 16);
            v79 = *(_DWORD *)(v77 + 8);
            v240 = v79 >> 8;
            if ( v79 > 0x1FF && (((unsigned int)&loc_FFFFFD + (v79 >> 8)) & 0xFFFFFF) > 2 )
            {
              v226 = v22;
              v80 = sub_2C76A00(v17, v22, 0);
              v22 = v226;
              v82 = *(__m128i **)(v80 + 32);
              v83 = v80;
              if ( *(_QWORD *)(v80 + 24) - (_QWORD)v82 <= 0x1Bu )
              {
                v184 = sub_CB6200(v80, "Invalid target address space", 0x1Cu);
                v22 = v226;
                v83 = v184;
                v85 = *(_BYTE **)(v184 + 32);
              }
              else
              {
                v84 = _mm_load_si128((const __m128i *)&xmmword_42D0E20);
                qmemcpy(&v82[1], "ddress space", 12);
                *v82 = v84;
                v85 = (_BYTE *)(*(_QWORD *)(v80 + 32) + 28LL);
                *(_QWORD *)(v83 + 32) = v85;
              }
              if ( v85 == *(_BYTE **)(v83 + 24) )
              {
                v233 = v22;
                sub_CB6200(v83, (unsigned __int8 *)"\n", 1u);
                v22 = v233;
              }
              else
              {
                *v85 = 10;
                ++*(_QWORD *)(v83 + 32);
              }
              v86 = *(_BYTE **)(v17 + 16);
              if ( v86 )
                *v86 = 0;
              v87 = *(unsigned int *)(v17 + 4);
              if ( !(_DWORD)v87 )
              {
                v88 = *(_QWORD *)(v17 + 24);
                if ( *(_QWORD *)(v88 + 32) != *(_QWORD *)(v88 + 16) )
                {
                  v227 = v22;
                  sub_CB5AE0((__int64 *)v88);
                  v88 = *(_QWORD *)(v17 + 24);
                  v22 = v227;
                }
                v228 = v22;
                sub_CEB520(*(_QWORD **)(v88 + 48), v87, (__int64)v82, v81);
                v22 = v228;
              }
            }
            if ( (unsigned int)v78 > 1 && ((unsigned int)((unsigned int)&loc_FFFFFD + v78) & 0xFFFFFF) > 2 )
            {
              v89 = (char *)v22;
              v229 = v22;
              v90 = sub_2C76A00(v17, v22, 0);
              v22 = v229;
              v91 = *(__m128i **)(v90 + 32);
              v92 = v90;
              if ( *(_QWORD *)(v90 + 24) - (_QWORD)v91 <= 0x1Bu )
              {
                v89 = "Invalid source address space";
                v185 = sub_CB6200(v90, "Invalid source address space", 0x1Cu);
                v22 = v229;
                v92 = v185;
                v94 = *(_BYTE **)(v185 + 32);
              }
              else
              {
                v93 = _mm_load_si128((const __m128i *)&xmmword_42D0E30);
                qmemcpy(&v91[1], "ddress space", 12);
                *v91 = v93;
                v94 = (_BYTE *)(*(_QWORD *)(v90 + 32) + 28LL);
                *(_QWORD *)(v92 + 32) = v94;
              }
              if ( *(_BYTE **)(v92 + 24) == v94 )
              {
                v89 = "\n";
                v234 = v22;
                sub_CB6200(v92, (unsigned __int8 *)"\n", 1u);
                v22 = v234;
              }
              else
              {
                *v94 = 10;
                ++*(_QWORD *)(v92 + 32);
              }
              v95 = *(_BYTE **)(v17 + 16);
              if ( v95 )
                *v95 = 0;
              v96 = (char *)*(unsigned int *)(v17 + 4);
              if ( !(_DWORD)v96 )
              {
                v97 = *(_QWORD *)(v17 + 24);
                if ( *(_QWORD *)(v97 + 32) != *(_QWORD *)(v97 + 16) )
                {
                  v230 = v22;
                  sub_CB5AE0((__int64 *)v97);
                  v97 = *(_QWORD *)(v17 + 24);
                  v22 = v230;
                }
                v231 = v22;
                sub_CEB520(*(_QWORD **)(v97 + 48), (__int64)v89, (__int64)v91, v96);
                v22 = v231;
              }
            }
            if ( !(_DWORD)v78 || !v240 )
              goto LABEL_76;
            v58 = (char *)v22;
            v241 = v22;
            v98 = sub_2C76A00(v17, v22, 0);
            v22 = v241;
            v62 = *(__m128i **)(v98 + 32);
            v99 = v98;
            if ( *(_QWORD *)(v98 + 24) - (_QWORD)v62 <= 0x3Fu )
            {
              v58 = "Cannot cast non-generic pointer to different non-generic pointer";
              v189 = sub_CB6200(v98, "Cannot cast non-generic pointer to different non-generic pointer", 0x40u);
              v22 = v241;
              v99 = v189;
              v100 = *(_BYTE **)(v189 + 32);
            }
            else
            {
              *v62 = _mm_load_si128((const __m128i *)&xmmword_42D0E40);
              v62[1] = _mm_load_si128((const __m128i *)&xmmword_42D0E50);
              v62[2] = _mm_load_si128((const __m128i *)&xmmword_42D0E60);
              v62[3] = _mm_load_si128((const __m128i *)&xmmword_42D0E70);
              v100 = (_BYTE *)(*(_QWORD *)(v98 + 32) + 64LL);
              *(_QWORD *)(v99 + 32) = v100;
            }
            if ( v100 != *(_BYTE **)(v99 + 24) )
              goto LABEL_126;
            goto LABEL_142;
          case 0x55:
            v13 = (const char *)(v21 - 24);
            sub_2C7B6A0(v17, v21 - 24);
            continue;
          case 0x5F:
            v13 = "landingpad";
            sub_2C76F10(v17, "landingpad", v21 - 24);
            continue;
          default:
            BUG();
        }
      }
LABEL_78:
      if ( v259 == (char *)v261 )
        goto LABEL_79;
    }
  }
LABEL_80:
  if ( v265 != (unsigned __int8 *)v267 )
  {
    v13 = (const char *)(v267[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v265);
  }
  return sub_AE4030(v268, (__int64)v13);
}
