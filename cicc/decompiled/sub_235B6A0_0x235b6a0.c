// Function: sub_235B6A0
// Address: 0x235b6a0
//
__int64 *__fastcall sub_235B6A0(__int64 *a1, __int64 a2, unsigned __int64 *a3, const __m128i *a4)
{
  __int64 v7; // r10
  __int64 v8; // rbx
  __m128i v9; // rdi
  bool v10; // al
  const char *v11; // rdx
  __int64 v12; // r10
  __int64 v13; // r15
  __int64 v14; // r13
  __m128i v15; // xmm0
  __int64 v17; // rdi
  unsigned __int64 *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // rdx
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  __int64 v120; // r9
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // r8
  __int64 v128; // r9
  __int64 v129; // rdx
  __int64 v130; // rcx
  __int64 v131; // r8
  __int64 v132; // r9
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 v135; // r8
  __int64 v136; // r9
  __int64 v137; // rdx
  __int64 v138; // rcx
  __int64 v139; // r8
  __int64 v140; // r9
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 v144; // r9
  __int64 v145; // rdx
  __int64 v146; // rcx
  __int64 v147; // r8
  __int64 v148; // r9
  __int64 v149; // rdx
  __int64 v150; // rcx
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // r8
  __int64 v156; // r9
  __int64 v157; // rdx
  __int64 v158; // rcx
  __int64 v159; // r8
  __int64 v160; // r9
  __int64 v161; // rdx
  __int64 v162; // rcx
  __int64 v163; // r8
  __int64 v164; // r9
  __int64 v165; // rdi
  unsigned __int64 *v166; // r12
  char *v167; // rax
  char *v168; // rdi
  unsigned int v169; // eax
  unsigned int v170; // ebx
  __int64 v171; // rdx
  __int64 v172; // r13
  char *v173; // rsi
  __int64 v174; // rdi
  unsigned __int64 *v175; // r12
  char *v176; // rax
  __int64 v177; // rdi
  __int64 v178; // rdi
  __int64 v179; // rdi
  __int64 v180; // rdi
  __int64 v181; // rdi
  __int64 v182; // rdi
  __int64 v183; // rdi
  __int64 v184; // rdi
  __int64 v185; // rdi
  __int64 v186; // rdi
  __int64 v187; // rdi
  __int64 v188; // rdi
  void *v189; // rbx
  __int64 v190; // rdx
  __int64 v191; // rcx
  __int64 v192; // r8
  __int64 v193; // r9
  __int64 v194; // rdi
  void *v195; // rbx
  __int64 v196; // rdx
  __int64 v197; // rcx
  __int64 v198; // r8
  __int64 v199; // r9
  __int64 v200; // rdi
  void *v201; // rbx
  __int64 v202; // rdx
  __int64 v203; // rcx
  __int64 v204; // r8
  __int64 v205; // r9
  __int64 v206; // rdi
  void *v207; // rbx
  __int64 v208; // rdx
  __int64 v209; // rcx
  __int64 v210; // r8
  __int64 v211; // r9
  void *v212; // rax
  __int64 v213; // rdx
  __int64 v214; // rcx
  __int64 v215; // r8
  __int64 v216; // r9
  __int64 v217; // rdi
  __int64 v218; // r8
  __int64 v219; // r9
  __int64 v220; // rdx
  __int64 v221; // rdi
  unsigned __int64 *v222; // r12
  const char *v223; // r15
  __int16 v224; // r13
  __int64 v225; // rax
  const char **v226; // rsi
  __int64 v227; // r8
  __int64 v228; // r9
  __int64 v229; // rdx
  __int64 v230; // rdi
  char v231; // dl
  __int16 v232; // r13
  __int64 v233; // rdx
  __int64 v234; // rcx
  __int64 v235; // r8
  __int64 v236; // r9
  unsigned __int64 *v237; // r12
  __int64 v238; // rax
  __int64 v239; // r15
  __int64 v240; // rbx
  __m128i v241; // xmm2
  __int64 v242; // rax
  char v243; // al
  __int64 v244; // rdi
  unsigned int v245; // eax
  __int64 v246; // rdx
  __int64 v247; // r8
  __int64 v248; // r9
  unsigned __int64 v249; // rdx
  __int64 v250; // rdi
  __int16 v251; // r13
  __int64 v252; // rax
  __int64 v253; // rdi
  __int64 v254; // rdi
  __int64 v255; // rdi
  __int64 v256; // rdi
  __int64 v257; // rdi
  __int64 v258; // rdi
  __int64 v259; // rdi
  __int64 v260; // rdi
  __int64 v261; // rdi
  __int64 v262; // rdi
  __int64 v263; // rdi
  __int64 v264; // rdi
  __int64 v265; // [rsp+0h] [rbp-170h]
  __m128i v267; // [rsp+10h] [rbp-160h] BYREF
  __m128i v268; // [rsp+20h] [rbp-150h] BYREF
  __int64 v269[2]; // [rsp+30h] [rbp-140h] BYREF
  __m128i v270; // [rsp+40h] [rbp-130h] BYREF
  char v271; // [rsp+50h] [rbp-120h] BYREF
  int v272; // [rsp+80h] [rbp-F0h]
  __int64 v273; // [rsp+88h] [rbp-E8h]
  __int64 v274; // [rsp+90h] [rbp-E0h]
  __int64 v275; // [rsp+98h] [rbp-D8h]
  __int64 v276; // [rsp+A0h] [rbp-D0h]
  __int64 v277; // [rsp+A8h] [rbp-C8h]
  __int64 v278; // [rsp+B0h] [rbp-C0h]
  unsigned __int64 v279; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int64 v280; // [rsp+C8h] [rbp-A8h] BYREF
  _QWORD v281[2]; // [rsp+D0h] [rbp-A0h] BYREF
  char v282; // [rsp+E0h] [rbp-90h]
  _QWORD v283[2]; // [rsp+E8h] [rbp-88h] BYREF
  _QWORD v284[15]; // [rsp+F8h] [rbp-78h] BYREF

  v7 = a4[1].m128i_i64[0];
  v8 = a4[1].m128i_i64[1];
  v265 = v7;
  v267 = _mm_loadu_si128(a4);
  if ( v7 == v8 )
  {
    if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-flatten", 12) )
    {
      sub_2332320((__int64)a3, 1, v33, v34, v35, v36);
      v168 = (char *)sub_22077B0(0x10u);
      if ( v168 )
        *(_QWORD *)v168 = &unk_4A13D38;
    }
    else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-interchange", 16) )
    {
      sub_2332320((__int64)a3, 1, v37, v38, v39, v40);
      v168 = (char *)sub_22077B0(0x10u);
      if ( v168 )
        *(_QWORD *)v168 = &unk_4A13D78;
    }
    else
    {
      if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-unroll-and-jam", 19) )
      {
        LODWORD(v279) = 2;
        sub_235B530(a3, (int *)&v279, v41, v42, v43, v44);
        goto LABEL_9;
      }
      if ( !sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "no-op-loopnest", 14) )
      {
        if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "require<ddg>", 12) )
        {
          v174 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v174, 0, v49, v50, v51, v52);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13E38;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "invalidate<ddg>", 15) )
        {
          v177 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v177, 0, v53, v54, v55, v56);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13E78;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "require<iv-users>", 17) )
        {
          v178 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v178, 0, v57, v58, v59, v60);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13EB8;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "invalidate<iv-users>", 20) )
        {
          v179 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v179, 0, v61, v62, v63, v64);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13EF8;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "require<no-op-loop>", 19) )
        {
          v180 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v180, 0, v65, v66, v67, v68);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13F38;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "invalidate<no-op-loop>", 22) )
        {
          v181 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v181, 0, v69, v70, v71, v72);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13F78;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "require<pass-instrumentation>", 29) )
        {
          v182 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v182, 0, v73, v74, v75, v76);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13FB8;
        }
        else if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "invalidate<pass-instrumentation>", 32) )
        {
          v183 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v183, 0, v77, v78, v79, v80);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A13FF8;
        }
        else if ( sub_9691B0(
                    (const void *)v267.m128i_i64[0],
                    v267.m128i_u64[1],
                    "require<should-run-extra-simple-loop-unswitch>",
                    46) )
        {
          v184 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v184, 0, v81, v82, v83, v84);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A14038;
        }
        else if ( sub_9691B0(
                    (const void *)v267.m128i_i64[0],
                    v267.m128i_u64[1],
                    "invalidate<should-run-extra-simple-loop-unswitch>",
                    49) )
        {
          v186 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v186, 0, v85, v86, v87, v88);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A14078;
        }
        else
        {
          if ( !sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "canon-freeze", 12) )
          {
            if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "dot-ddg", 7) )
            {
              v187 = (__int64)a3;
              v166 = a3 + 9;
              sub_2332320(v187, 0, v93, v94, v95, v96);
              v167 = (char *)sub_22077B0(0x10u);
              if ( v167 )
                *(_QWORD *)v167 = &unk_4A11EB8;
              goto LABEL_55;
            }
            if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "guard-widening", 14) )
            {
              v230 = (__int64)a3;
              v166 = a3 + 9;
              sub_2332320(v230, 0, v97, v98, v99, v100);
              v167 = (char *)sub_22077B0(0x10u);
              if ( v167 )
                *(_QWORD *)v167 = &unk_4A11EF8;
              goto LABEL_55;
            }
            if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "extra-simple-loop-unswitch-passes", 33) )
            {
              memset(&v284[1], 0, 56);
              v279 = (unsigned __int64)v281;
              v280 = 0x600000000LL;
              sub_235AE30((__int64)a3, (__int64)&v279, v101, v102, v103, v104);
              sub_2337B30(&v279);
            }
            else
            {
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "indvars", 7) )
              {
                v244 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v244, 0, v105, v106, v107, v108);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                {
                  v167[8] = 1;
                  *(_QWORD *)v167 = &unk_4A11F78;
                }
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "invalidate<all>", 15) )
              {
                v256 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v256, 0, v109, v110, v111, v112);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A11FB8;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-bound-split", 16) )
              {
                v255 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v255, 0, v113, v114, v115, v116);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A11FF8;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-deletion", 13) )
              {
                v254 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v254, 0, v117, v118, v119, v120);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A12038;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-idiom", 10) )
              {
                v253 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v253, 0, v121, v122, v123, v124);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A12078;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-idiom-vectorize", 20) )
              {
                v264 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v264, 0, v125, v126, v127, v128);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                {
                  *(_QWORD *)v167 = &unk_4A120B8;
                  *((_QWORD *)v167 + 1) = 0x1000000000LL;
                }
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-instsimplify", 17) )
              {
                v263 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v263, 0, v129, v130, v131, v132);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A120F8;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-predication", 16) )
              {
                v262 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v262, 0, v133, v134, v135, v136);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A12138;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-reduce", 11) )
              {
                v261 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v261, 0, v137, v138, v139, v140);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A12178;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-term-fold", 14) )
              {
                v260 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v260, 0, v141, v142, v143, v144);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A121B8;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-simplifycfg", 16) )
              {
                v259 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v259, 0, v145, v146, v147, v148);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A121F8;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-unroll-full", 16) )
              {
                v258 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v258, 0, v149, v150, v151, v152);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                {
                  *((_DWORD *)v167 + 2) = 2;
                  *((_WORD *)v167 + 6) = 0;
                  *(_QWORD *)v167 = &unk_4A12238;
                }
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-versioning-licm", 20) )
              {
                v257 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v257, 0, v153, v154, v155, v156);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A12278;
                goto LABEL_55;
              }
              if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "no-op-loop", 10) )
              {
                v217 = (__int64)a3;
                v166 = a3 + 9;
                sub_2332320(v217, 0, v157, v158, v159, v160);
                v167 = (char *)sub_22077B0(0x10u);
                if ( v167 )
                  *(_QWORD *)v167 = &unk_4A122B8;
                goto LABEL_55;
              }
              if ( !sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "print", 5) )
              {
                if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "print<ddg>", 10) )
                {
                  v206 = (__int64)a3;
                  v166 = a3 + 9;
                  v207 = sub_CB72A0();
                  sub_2332320(v206, 0, v208, v209, v210, v211);
                  v167 = (char *)sub_22077B0(0x10u);
                  if ( v167 )
                  {
                    *((_QWORD *)v167 + 1) = v207;
                    *(_QWORD *)v167 = &unk_4A12338;
                  }
                  goto LABEL_55;
                }
                if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "print<iv-users>", 15) )
                {
                  v200 = (__int64)a3;
                  v166 = a3 + 9;
                  v201 = sub_CB72A0();
                  sub_2332320(v200, 0, v202, v203, v204, v205);
                  v167 = (char *)sub_22077B0(0x10u);
                  if ( v167 )
                  {
                    *((_QWORD *)v167 + 1) = v201;
                    *(_QWORD *)v167 = &unk_4A12378;
                  }
                  goto LABEL_55;
                }
                if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "print<loop-cache-cost>", 22) )
                {
                  v194 = (__int64)a3;
                  v166 = a3 + 9;
                  v195 = sub_CB72A0();
                  sub_2332320(v194, 0, v196, v197, v198, v199);
                  v167 = (char *)sub_22077B0(0x10u);
                  if ( v167 )
                  {
                    *((_QWORD *)v167 + 1) = v195;
                    *(_QWORD *)v167 = &unk_4A123B8;
                  }
                  goto LABEL_55;
                }
                if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "print<loopnest>", 15) )
                {
                  v188 = (__int64)a3;
                  v166 = a3 + 9;
                  v189 = sub_CB72A0();
                  sub_2332320(v188, 0, v190, v191, v192, v193);
                  v167 = (char *)sub_22077B0(0x10u);
                  if ( v167 )
                  {
                    *((_QWORD *)v167 + 1) = v189;
                    *(_QWORD *)v167 = &unk_4A123F8;
                  }
                  goto LABEL_55;
                }
                if ( sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop-index-split", 16) )
                {
                  v165 = (__int64)a3;
                  v166 = a3 + 9;
                  sub_2332320(v165, 0, v161, v162, v163, v164);
                  v167 = (char *)sub_22077B0(0x10u);
                  if ( v167 )
                    *(_QWORD *)v167 = &unk_4A12438;
LABEL_55:
                  v279 = (unsigned __int64)v167;
                  sub_235ACD0(v166, &v279);
                  v168 = (char *)v279;
                  goto LABEL_56;
                }
                if ( (unsigned __int8)sub_2337DE0((char *)v267.m128i_i64[0], v267.m128i_i64[1], "licm", 4u) )
                {
                  sub_234D100(
                    (__int64)&v279,
                    (void (__fastcall *)(__int64, const void *, __int64))sub_2332EB0,
                    (const void *)v267.m128i_i64[0],
                    v267.m128i_i64[1],
                    "licm",
                    4u);
                  v220 = v281[0] & 1;
                  LOBYTE(v281[0]) = (2 * v220) | v281[0] & 0xFD;
                  if ( !(_BYTE)v220 )
                  {
                    v221 = (__int64)a3;
                    v222 = a3 + 9;
                    v223 = (const char *)v279;
                    v224 = v280;
                    sub_2332320(v221, 0, v220, (unsigned int)(2 * v220), v218, v219);
                    v225 = sub_22077B0(0x18u);
                    if ( v225 )
                    {
                      *(_QWORD *)(v225 + 8) = v223;
                      *(_WORD *)(v225 + 16) = v224;
                      *(_QWORD *)v225 = &unk_4A12478;
                    }
                    v270.m128i_i64[0] = v225;
                    v226 = (const char **)&v270;
                    sub_235ACD0(v222, (unsigned __int64 *)&v270);
                    if ( v270.m128i_i64[0] )
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v270.m128i_i64[0] + 8LL))(v270.m128i_i64[0]);
                    v270.m128i_i64[0] = 0;
                    *a1 = 1;
                    sub_9C66B0(v270.m128i_i64);
                    goto LABEL_119;
                  }
                  goto LABEL_120;
                }
                if ( (unsigned __int8)sub_2337DE0((char *)v267.m128i_i64[0], v267.m128i_i64[1], "lnicm", 5u) )
                {
                  sub_234D100(
                    (__int64)&v279,
                    (void (__fastcall *)(__int64, const void *, __int64))sub_2332EB0,
                    (const void *)v267.m128i_i64[0],
                    v267.m128i_i64[1],
                    "lnicm",
                    5u);
                  v229 = v281[0] & 1;
                  LOBYTE(v281[0]) = (2 * v229) | v281[0] & 0xFD;
                  if ( !(_BYTE)v229 )
                  {
                    v226 = (const char **)&v270;
                    v270.m128i_i64[0] = v279;
                    v270.m128i_i16[4] = v280;
                    sub_235B5E0(a3, v270.m128i_i64, v229, (unsigned int)(2 * v229), v227, v228);
                    v270.m128i_i64[0] = 0;
                    *a1 = 1;
                    sub_9C66B0(v270.m128i_i64);
                    goto LABEL_119;
                  }
LABEL_120:
                  v226 = (const char **)&v279;
                  sub_234D1A0(a1, (__int64 *)&v279);
LABEL_119:
                  sub_2352290(&v279, (__int64)v226);
                  return a1;
                }
                if ( (unsigned __int8)sub_2337DE0((char *)v267.m128i_i64[0], v267.m128i_i64[1], "loop-rotate", 0xBu) )
                {
                  sub_234D210(
                    (__int64)&v279,
                    (void (__fastcall *)(__int64, const void *, __int64))sub_2332C00,
                    (const void *)v267.m128i_i64[0],
                    v267.m128i_i64[1],
                    "loop-rotate",
                    0xBu);
                  v231 = v280 & 1;
                  LOBYTE(v280) = (2 * (v280 & 1)) | v280 & 0xFD;
                  if ( !v231 )
                  {
                    sub_28448C0(&v270, (unsigned __int8)v279, BYTE1(v279));
                    v232 = v270.m128i_i16[0];
                    sub_2332320((__int64)a3, 0, v233, v234, v235, v236);
                    v237 = a3 + 9;
                    v238 = sub_22077B0(0x10u);
                    if ( v238 )
                    {
                      *(_WORD *)(v238 + 8) = v232;
                      *(_QWORD *)v238 = &unk_4A124B8;
                    }
                    v270.m128i_i64[0] = v238;
LABEL_131:
                    sub_235ACD0(v237, (unsigned __int64 *)&v270);
                    if ( v270.m128i_i64[0] )
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v270.m128i_i64[0] + 8LL))(v270.m128i_i64[0]);
                    v270.m128i_i64[0] = 0;
                    *a1 = 1;
                    sub_9C66B0(v270.m128i_i64);
                    goto LABEL_134;
                  }
                }
                else
                {
                  v9.m128i_i64[1] = v267.m128i_i64[1];
                  if ( !(unsigned __int8)sub_2337DE0(
                                           (char *)v267.m128i_i64[0],
                                           v267.m128i_i64[1],
                                           "simple-loop-unswitch",
                                           0x14u) )
                  {
                    v239 = *(_QWORD *)(a2 + 1888);
                    v240 = v239 + 32LL * *(unsigned int *)(a2 + 1896);
                    while ( v240 != v239 )
                    {
                      v11 = (const char *)a4[1].m128i_i64[0];
                      v9.m128i_i64[0] = 0xCCCCCCCCCCCCCCCDLL;
                      v241 = _mm_loadu_si128(&v267);
                      v242 = a4[1].m128i_i64[1] - (_QWORD)v11;
                      v279 = (unsigned __int64)v11;
                      v270 = v241;
                      v280 = 0xCCCCCCCCCCCCCCCDLL * (v242 >> 3);
                      if ( !*(_QWORD *)(v239 + 16) )
                        goto LABEL_90;
                      v9.m128i_i64[1] = (__int64)&v270;
                      v243 = (*(__int64 (__fastcall **)(__int64, __m128i *, unsigned __int64 *, unsigned __int64 *))(v239 + 24))(
                               v239,
                               &v270,
                               a3,
                               &v279);
                      v239 += 32;
                      if ( v243 )
                        goto LABEL_9;
                    }
                    v245 = sub_C63BB0();
                    v280 = 23;
                    v170 = v245;
                    v172 = v246;
                    v279 = (unsigned __int64)"unknown loop pass '{0}'";
                    goto LABEL_59;
                  }
                  sub_234D210(
                    (__int64)&v279,
                    (void (__fastcall *)(__int64, const void *, __int64))sub_2332990,
                    (const void *)v267.m128i_i64[0],
                    v267.m128i_i64[1],
                    "simple-loop-unswitch",
                    0x14u);
                  v249 = v280 & 1;
                  LOBYTE(v280) = (2 * v249) | v280 & 0xFD;
                  if ( !(_BYTE)v249 )
                  {
                    v250 = (__int64)a3;
                    v251 = v279;
                    v237 = a3 + 9;
                    sub_2332320(v250, 0, v249, (unsigned int)(2 * v249), v247, v248);
                    v252 = sub_22077B0(0x10u);
                    if ( v252 )
                    {
                      *(_WORD *)(v252 + 8) = v251;
                      *(_QWORD *)v252 = &unk_4A124F8;
                    }
                    v270.m128i_i64[0] = v252;
                    goto LABEL_131;
                  }
                }
                sub_9C9930(a1, (__int64 *)&v279);
LABEL_134:
                sub_23521F0(&v279);
                return a1;
              }
              sub_230B630(v270.m128i_i64, byte_3F871B3);
              v212 = sub_CB72A0();
              sub_283D800(&v279, v212, &v270);
              sub_235AD10((__int64)a3, (__int64)&v279, v213, v214, v215, v216);
              sub_2240A30(&v280);
              sub_2240A30((unsigned __int64 *)&v270);
            }
LABEL_9:
            v279 = 0;
            *a1 = 1;
            sub_9C66B0((__int64 *)&v279);
            return a1;
          }
          v185 = (__int64)a3;
          v175 = a3 + 9;
          sub_2332320(v185, 0, v89, v90, v91, v92);
          v176 = (char *)sub_22077B0(0x10u);
          if ( v176 )
            *(_QWORD *)v176 = &unk_4A11E78;
        }
        v279 = (unsigned __int64)v176;
        sub_235ACD0(v175, &v279);
        sub_233F7D0((__int64 *)&v279);
        goto LABEL_9;
      }
      sub_2332320((__int64)a3, 1, v45, v46, v47, v48);
      v168 = (char *)sub_22077B0(0x10u);
      if ( v168 )
        *(_QWORD *)v168 = &unk_4A13DF8;
    }
    v279 = (unsigned __int64)v168;
    v173 = (char *)a3[13];
    if ( v173 == (char *)a3[14] )
    {
      sub_235B010(a3 + 12, v173, &v279);
      v168 = (char *)v279;
    }
    else
    {
      if ( v173 )
      {
        *(_QWORD *)v173 = v168;
        a3[13] += 8LL;
LABEL_67:
        v279 = 0;
        *a1 = 1;
        sub_9C66B0((__int64 *)&v279);
        return a1;
      }
      a3[13] = 8;
    }
LABEL_56:
    if ( v168 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v168 + 8LL))(v168);
    goto LABEL_67;
  }
  v9 = v267;
  v10 = sub_9691B0((const void *)v267.m128i_i64[0], v267.m128i_u64[1], "loop", 4);
  v12 = v265;
  if ( !v10 )
  {
    v13 = *(_QWORD *)(a2 + 1888);
    v14 = v13 + 32LL * *(unsigned int *)(a2 + 1896);
    if ( v13 != v14 )
    {
      while ( 1 )
      {
        v15 = _mm_loadu_si128(&v267);
        v269[0] = v12;
        v268 = v15;
        v269[1] = 0xCCCCCCCCCCCCCCCDLL * ((v8 - v12) >> 3);
        if ( !*(_QWORD *)(v13 + 16) )
          break;
        v9.m128i_i64[1] = (__int64)&v268;
        v9.m128i_i64[0] = v13;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, unsigned __int64 *, __int64 *))(v13 + 24))(
               v13,
               &v268,
               a3,
               v269) )
        {
          goto LABEL_9;
        }
        v13 += 32;
        if ( v14 == v13 )
          goto LABEL_58;
        v8 = a4[1].m128i_i64[1];
        v12 = a4[1].m128i_i64[0];
      }
LABEL_90:
      sub_4263D6(v9.m128i_i64[0], v9.m128i_i64[1], v11);
    }
LABEL_58:
    v169 = sub_C63BB0();
    v280 = 42;
    v170 = v169;
    v172 = v171;
    v279 = (unsigned __int64)"invalid use of '{0}' pass as loop pipeline";
LABEL_59:
    v281[1] = 1;
    v281[0] = v284;
    v282 = 1;
    v283[0] = &unk_49DB108;
    v283[1] = &v267;
    v284[0] = v283;
    sub_23328D0((__int64)&v270, (__int64)&v279);
    sub_23058C0(a1, (__int64)&v270, v170, v172);
    sub_2240A30((unsigned __int64 *)&v270);
    return a1;
  }
  v270.m128i_i64[0] = (__int64)&v271;
  v270.m128i_i64[1] = 0x600000000LL;
  v272 = 0;
  v273 = 0;
  v274 = 0;
  v275 = 0;
  v276 = 0;
  v277 = 0;
  v278 = 0;
  sub_235CCD0(&v279, a2, &v270, v265, 0xCCCCCCCCCCCCCCCDLL * ((v8 - v265) >> 3));
  if ( (v279 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v279 = v279 & 0xFFFFFFFFFFFFFFFELL | 1;
    *a1 = 0;
    sub_23055C0(a1, &v279);
    sub_9C66B0((__int64 *)&v279);
  }
  else
  {
    v279 = 0;
    sub_9C66B0((__int64 *)&v279);
    v17 = (__int64)a3;
    v18 = a3 + 9;
    sub_2332320(v17, 0, v19, v20, v21, v22);
    sub_2337A80((__int64)&v279, (__int64)&v270, v23, v24, v25, v26);
    v27 = (_QWORD *)sub_22077B0(0x80u);
    v32 = (__int64)v27;
    if ( v27 )
    {
      *v27 = &unk_4A0B4E8;
      sub_2337A80((__int64)(v27 + 1), (__int64)&v279, v28, v29, v30, v31);
    }
    v269[0] = v32;
    sub_235ACD0(v18, (unsigned __int64 *)v269);
    sub_233F7D0(v269);
    sub_2337B30(&v279);
    v279 = 0;
    *a1 = 1;
    sub_9C66B0((__int64 *)&v279);
  }
  sub_2337B30((unsigned __int64 *)&v270);
  return a1;
}
