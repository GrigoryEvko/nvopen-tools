// Function: sub_212CE20
// Address: 0x212ce20
//
unsigned __int64 __fastcall sub_212CE20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v7; // rsi
  int v8; // eax
  unsigned __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // rax
  __m128i v16; // xmm3
  unsigned __int8 v17; // r15
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int8 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // r15
  unsigned int v23; // r13d
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v27; // r8
  __int64 v28; // r15
  unsigned int v29; // r13d
  __int64 v30; // rbx
  __int64 v31; // r12
  unsigned __int8 v32; // r14
  __int64 i; // r15
  unsigned int v34; // ecx
  __int64 v35; // r15
  const void ***v36; // rax
  int v37; // edx
  __int64 v38; // r9
  __int64 *v39; // rdi
  const void ***v40; // r13
  int v41; // r12d
  int v42; // edx
  __int64 *v43; // rdi
  __int64 v44; // r9
  unsigned int v45; // edx
  unsigned __int64 result; // rax
  __int64 v47; // rax
  unsigned int v48; // eax
  __int64 v49; // rdx
  const void ***v50; // rax
  int v51; // edx
  __int64 v52; // r9
  __int64 *v53; // rdi
  const void ***v54; // r13
  int v55; // r12d
  int v56; // edx
  __int64 *v57; // rdi
  __int64 v58; // r9
  unsigned int v59; // edx
  unsigned int v60; // r13d
  __int64 v61; // rbx
  __int64 v62; // r12
  unsigned __int8 v63; // r14
  __int64 j; // r15
  __int64 v65; // r12
  __int64 v66; // rax
  const void **v67; // r13
  int v69; // edx
  bool v70; // cl
  _DWORD *v71; // r8
  bool v72; // r15
  int v73; // r15d
  __int64 *v74; // rdi
  int v75; // edx
  __int64 *v76; // rax
  int v77; // edx
  unsigned __int64 v78; // rdx
  __int64 *v79; // r12
  unsigned __int8 *v80; // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rax
  const void **v83; // rdx
  __int64 *v84; // rax
  __int64 v85; // rdx
  __int64 *v86; // r15
  __int64 v87; // rax
  __int64 v88; // rdx
  __int128 v89; // rax
  __int64 v90; // rax
  unsigned int v91; // edx
  unsigned int v92; // edx
  unsigned int v93; // eax
  unsigned int v94; // eax
  bool v95; // al
  unsigned int v96; // r8d
  unsigned int v97; // eax
  int v98; // edx
  __int64 *v99; // rdi
  __int64 v100; // r9
  int v101; // edx
  int v102; // edx
  __int64 *v103; // r12
  unsigned __int64 v104; // rax
  const void **v105; // rdx
  __int64 *v106; // rax
  __int64 v107; // rdx
  __int16 *v108; // r12
  __int64 *v109; // r15
  __int64 v110; // rax
  __int64 v111; // rdx
  __int128 v112; // rax
  __int64 *v113; // rax
  __int64 *v114; // r12
  __int64 v115; // rdx
  unsigned __int64 v116; // rax
  const void **v117; // rdx
  __int64 *v118; // rax
  __int64 *v119; // r12
  __int16 *v120; // rdx
  __int128 v121; // rax
  __int128 v122; // rax
  unsigned int v123; // edx
  int v124; // edx
  __int64 *v125; // rdi
  __int64 v126; // r9
  unsigned int v127; // eax
  __int64 v128; // rdx
  int v129; // edx
  const void ***v130; // rcx
  __int64 v131; // r9
  int v132; // edx
  __int64 *v133; // rax
  int v134; // edx
  unsigned int v135; // r10d
  __int64 *v136; // r15
  __int64 v137; // rax
  unsigned __int64 v138; // rdx
  __int64 *v139; // rax
  unsigned int v140; // edx
  unsigned int v141; // edx
  unsigned int v142; // edx
  int v143; // edx
  __int128 v144; // rax
  unsigned int v145; // edx
  unsigned int v146; // edx
  __int128 v147; // [rsp-10h] [rbp-330h]
  __int128 v148; // [rsp-10h] [rbp-330h]
  __int128 v149; // [rsp-10h] [rbp-330h]
  __int128 v150; // [rsp-10h] [rbp-330h]
  __int128 v151; // [rsp-10h] [rbp-330h]
  __int128 v152; // [rsp-10h] [rbp-330h]
  __int128 v153; // [rsp-10h] [rbp-330h]
  __int128 v154; // [rsp-10h] [rbp-330h]
  __int128 v155; // [rsp-10h] [rbp-330h]
  __int128 v156; // [rsp+0h] [rbp-320h]
  __int128 v157; // [rsp+0h] [rbp-320h]
  __int128 v158; // [rsp+0h] [rbp-320h]
  __int128 v159; // [rsp+0h] [rbp-320h]
  __int128 v160; // [rsp+0h] [rbp-320h]
  __int128 v161; // [rsp+0h] [rbp-320h]
  __int128 v162; // [rsp+0h] [rbp-320h]
  __int128 v163; // [rsp+0h] [rbp-320h]
  unsigned int v164; // [rsp+10h] [rbp-310h]
  unsigned int v165; // [rsp+18h] [rbp-308h]
  unsigned int v166; // [rsp+20h] [rbp-300h]
  unsigned int v167; // [rsp+28h] [rbp-2F8h]
  unsigned int v168; // [rsp+30h] [rbp-2F0h]
  __int64 v169; // [rsp+40h] [rbp-2E0h]
  unsigned int v170; // [rsp+40h] [rbp-2E0h]
  bool v171; // [rsp+40h] [rbp-2E0h]
  __int64 v173; // [rsp+48h] [rbp-2D8h]
  __int64 v174; // [rsp+48h] [rbp-2D8h]
  unsigned __int64 v175; // [rsp+48h] [rbp-2D8h]
  int v176; // [rsp+48h] [rbp-2D8h]
  unsigned int v177; // [rsp+48h] [rbp-2D8h]
  __int64 v178; // [rsp+50h] [rbp-2D0h]
  __int64 (__fastcall *v179)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+50h] [rbp-2D0h]
  __int64 v180; // [rsp+50h] [rbp-2D0h]
  _DWORD *v181; // [rsp+50h] [rbp-2D0h]
  unsigned __int64 v182; // [rsp+50h] [rbp-2D0h]
  unsigned int v183; // [rsp+50h] [rbp-2D0h]
  __int64 v186; // [rsp+60h] [rbp-2C0h]
  __int64 v187; // [rsp+60h] [rbp-2C0h]
  unsigned int v188; // [rsp+60h] [rbp-2C0h]
  unsigned int v189; // [rsp+60h] [rbp-2C0h]
  __int64 v190; // [rsp+68h] [rbp-2B8h]
  unsigned int v191; // [rsp+70h] [rbp-2B0h]
  __int64 v192; // [rsp+70h] [rbp-2B0h]
  __int64 (__fastcall *v193)(__int64, __int64, __int64, __int64, __int64); // [rsp+70h] [rbp-2B0h]
  __int16 *v194; // [rsp+70h] [rbp-2B0h]
  unsigned int v195; // [rsp+70h] [rbp-2B0h]
  unsigned __int64 v196; // [rsp+70h] [rbp-2B0h]
  const void **v197; // [rsp+70h] [rbp-2B0h]
  unsigned int v198; // [rsp+70h] [rbp-2B0h]
  unsigned int v199; // [rsp+70h] [rbp-2B0h]
  __int16 *v200; // [rsp+78h] [rbp-2A8h]
  unsigned int v201; // [rsp+80h] [rbp-2A0h]
  unsigned __int8 v202; // [rsp+80h] [rbp-2A0h]
  __int64 v203; // [rsp+80h] [rbp-2A0h]
  __int64 v204; // [rsp+80h] [rbp-2A0h]
  __int64 v205; // [rsp+80h] [rbp-2A0h]
  __int64 v206; // [rsp+80h] [rbp-2A0h]
  __int64 v207; // [rsp+80h] [rbp-2A0h]
  __int64 v208; // [rsp+80h] [rbp-2A0h]
  __int64 v209; // [rsp+88h] [rbp-298h]
  __int64 v210; // [rsp+88h] [rbp-298h]
  __int64 v211; // [rsp+88h] [rbp-298h]
  int v212; // [rsp+C8h] [rbp-258h]
  __int64 *v213; // [rsp+1C0h] [rbp-160h]
  __int64 *v214; // [rsp+1E0h] [rbp-140h]
  __int64 *v215; // [rsp+200h] [rbp-120h]
  __int64 *v216; // [rsp+220h] [rbp-100h]
  __int64 v217; // [rsp+230h] [rbp-F0h] BYREF
  int v218; // [rsp+238h] [rbp-E8h]
  __m128i v219; // [rsp+240h] [rbp-E0h] BYREF
  __m128i v220; // [rsp+250h] [rbp-D0h] BYREF
  __m128i v221; // [rsp+260h] [rbp-C0h] BYREF
  __m128i v222; // [rsp+270h] [rbp-B0h] BYREF
  char v223[8]; // [rsp+280h] [rbp-A0h] BYREF
  __int64 v224; // [rsp+288h] [rbp-98h]
  __int64 v225; // [rsp+290h] [rbp-90h]
  __m128 v226; // [rsp+2A0h] [rbp-80h] BYREF
  __int128 v227; // [rsp+2B0h] [rbp-70h]
  _OWORD v228[2]; // [rsp+2C0h] [rbp-60h] BYREF
  __int64 *v229; // [rsp+2E0h] [rbp-40h]
  int v230; // [rsp+2E8h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v217 = v7;
  if ( v7 )
  {
    v201 = a6;
    sub_1623A60((__int64)&v217, v7, 2);
    a6 = v201;
  }
  v8 = *(_DWORD *)(a2 + 64);
  v219.m128i_i32[2] = 0;
  v220.m128i_i32[2] = 0;
  v218 = v8;
  v9 = *(unsigned __int64 **)(a2 + 32);
  v221.m128i_i32[2] = 0;
  v222.m128i_i32[2] = 0;
  v10 = v9[1];
  v219.m128i_i64[0] = 0;
  v220.m128i_i64[0] = 0;
  v221.m128i_i64[0] = 0;
  v222.m128i_i64[0] = 0;
  v191 = a6;
  sub_20174B0((__int64)a1, *v9, v10, &v219, &v220);
  sub_20174B0(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
    &v221,
    &v222);
  v11 = *a1;
  v12 = (__m128)_mm_loadu_si128(&v219);
  v169 = a2;
  v13 = _mm_loadu_si128(&v221);
  v14 = _mm_loadu_si128(&v220);
  v15 = *(_QWORD *)(v219.m128i_i64[0] + 40) + 16LL * v219.m128i_u32[2];
  v16 = _mm_loadu_si128(&v222);
  v17 = *(_BYTE *)v15;
  v18 = *(_QWORD *)(v15 + 8);
  v229 = 0;
  v19 = a1[1];
  v226 = v12;
  v230 = 0;
  v20 = v17;
  v21 = *(_QWORD *)(v19 + 48);
  v202 = v17;
  v22 = v18;
  v178 = v18;
  v23 = v191;
  v227 = (__int128)v13;
  v24 = v21;
  v228[0] = v14;
  v228[1] = v16;
  while ( 1 )
  {
    LOBYTE(v23) = v20;
    sub_1F40D10((__int64)v223, v11, v24, v23, v22);
    if ( !v223[0] )
      break;
    v93 = v168;
    LOBYTE(v93) = v20;
    sub_1F40D10((__int64)v223, v11, v24, v93, v22);
    v20 = v224;
    v22 = v225;
  }
  v25 = v178;
  if ( v20 == 1 )
  {
    v27 = 1;
  }
  else if ( !v20 || (v27 = v20, !*(_QWORD *)(v11 + 8LL * v20 + 120)) )
  {
    v28 = a1[1];
    goto LABEL_9;
  }
  v28 = a1[1];
  v192 = *(_QWORD *)(v28 + 48);
  if ( (*(_BYTE *)((unsigned int)(*(_WORD *)(v169 + 24) != 52) + 68 + 259 * v27 + v11 + 2422) & 0xFB) == 0 )
  {
    v173 = *a1;
    v179 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 264LL);
    v47 = sub_1E0A0C0(*(_QWORD *)(v28 + 32));
    v48 = v179(v173, v47, v192, v202, v25);
    v50 = (const void ***)sub_1D252B0(v28, v202, v25, v48, v49);
    v53 = (__int64 *)a1[1];
    v54 = v50;
    v55 = v51;
    *((_QWORD *)&v157 + 1) = 2;
    *(_QWORD *)&v157 = &v226;
    if ( *(_WORD *)(v169 + 24) == 52 )
    {
      v216 = sub_1D36D80(
               v53,
               71,
               (__int64)&v217,
               v50,
               v51,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v52,
               v157);
      *(_QWORD *)a3 = v216;
      v229 = v216;
      *(_DWORD *)(a3 + 8) = v124;
      v125 = (__int64 *)a1[1];
      *((_QWORD *)&v152 + 1) = 3;
      *(_QWORD *)&v152 = v228;
      v230 = 1;
      *(_QWORD *)a4 = sub_1D36D80(
                        v125,
                        68,
                        (__int64)&v217,
                        v54,
                        v55,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        v126,
                        v152);
    }
    else
    {
      v215 = sub_1D36D80(
               v53,
               73,
               (__int64)&v217,
               v50,
               v51,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v52,
               v157);
      *(_QWORD *)a3 = v215;
      v229 = v215;
      *(_DWORD *)(a3 + 8) = v56;
      v57 = (__int64 *)a1[1];
      *((_QWORD *)&v148 + 1) = 3;
      *(_QWORD *)&v148 = v228;
      v230 = 1;
      *(_QWORD *)a4 = sub_1D36D80(
                        v57,
                        69,
                        (__int64)&v217,
                        v54,
                        v55,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        v58,
                        v148);
    }
    result = v59;
    *(_DWORD *)(a4 + 8) = v59;
    goto LABEL_22;
  }
LABEL_9:
  v29 = v167;
  v30 = *a1;
  v31 = *(_QWORD *)(v28 + 48);
  v32 = v202;
  for ( i = v178; ; i = v225 )
  {
    LOBYTE(v29) = v32;
    sub_1F40D10((__int64)v223, v30, v31, v29, i);
    if ( !v223[0] )
      break;
    v94 = v166;
    LOBYTE(v94) = v32;
    sub_1F40D10((__int64)v223, v30, v31, v94, i);
    v32 = v224;
  }
  if ( v32 == 1 )
  {
    v34 = 1;
  }
  else if ( !v32 || (v34 = v32, !*(_QWORD *)(v30 + 8LL * v32 + 120)) )
  {
    v35 = a1[1];
    goto LABEL_27;
  }
  v35 = a1[1];
  if ( (*(_BYTE *)((unsigned int)(*(_WORD *)(v169 + 24) != 52) + 64 + 259LL * v34 + v30 + 2422) & 0xFB) == 0 )
  {
    v36 = (const void ***)sub_1D252B0(a1[1], v202, v178, 111, 0);
    v39 = (__int64 *)a1[1];
    v40 = v36;
    v41 = v37;
    *((_QWORD *)&v156 + 1) = 2;
    *(_QWORD *)&v156 = &v226;
    if ( *(_WORD *)(v169 + 24) == 52 )
    {
      v214 = sub_1D36D80(
               v39,
               64,
               (__int64)&v217,
               v36,
               v37,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v38,
               v156);
      *(_QWORD *)a3 = v214;
      v229 = v214;
      *(_DWORD *)(a3 + 8) = v98;
      v99 = (__int64 *)a1[1];
      *((_QWORD *)&v150 + 1) = 3;
      *(_QWORD *)&v150 = v228;
      v230 = 1;
      *(_QWORD *)a4 = sub_1D36D80(
                        v99,
                        66,
                        (__int64)&v217,
                        v40,
                        v41,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        v100,
                        v150);
    }
    else
    {
      v213 = sub_1D36D80(
               v39,
               65,
               (__int64)&v217,
               v36,
               v37,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v38,
               v156);
      *(_QWORD *)a3 = v213;
      v229 = v213;
      *(_DWORD *)(a3 + 8) = v42;
      v43 = (__int64 *)a1[1];
      *((_QWORD *)&v147 + 1) = 3;
      *(_QWORD *)&v147 = v228;
      v230 = 1;
      *(_QWORD *)a4 = sub_1D36D80(
                        v43,
                        67,
                        (__int64)&v217,
                        v40,
                        v41,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        v44,
                        v147);
    }
    result = v45;
    *(_DWORD *)(a4 + 8) = v45;
    goto LABEL_22;
  }
LABEL_27:
  v60 = v165;
  v61 = *a1;
  v62 = *(_QWORD *)(v35 + 48);
  v63 = v202;
  for ( j = v178; ; j = v225 )
  {
    LOBYTE(v60) = v63;
    sub_1F40D10((__int64)v223, v61, v62, v60, j);
    if ( !v223[0] )
      break;
    v97 = v164;
    LOBYTE(v97) = v63;
    sub_1F40D10((__int64)v223, v61, v62, v97, j);
    v63 = v224;
  }
  v65 = v169;
  v66 = v61;
  v67 = (const void **)v178;
  v69 = *(unsigned __int16 *)(v169 + 24);
  if ( v63 == 1 )
  {
    v96 = 1;
LABEL_46:
    v70 = (*(_BYTE *)(2 * (unsigned int)(v69 != 52) + 71 + 259LL * v96 + v66 + 2422) & 0xFB) == 0;
    goto LABEL_31;
  }
  v70 = 0;
  if ( v63 )
  {
    v96 = v63;
    if ( *(_QWORD *)(v66 + 8LL * v63 + 120) )
      goto LABEL_46;
  }
LABEL_31:
  v71 = (_DWORD *)*a1;
  v224 = v178;
  v223[0] = v202;
  if ( v202 )
  {
    if ( (unsigned __int8)(v202 - 14) > 0x5Fu )
    {
      v72 = (unsigned __int8)(v202 - 86) <= 0x17u || (unsigned __int8)(v202 - 8) <= 5u;
      goto LABEL_34;
    }
  }
  else
  {
    v171 = v70;
    v176 = v69;
    v181 = v71;
    v72 = sub_1F58CD0((__int64)v223);
    v95 = sub_1F58D20((__int64)v223);
    v71 = v181;
    v69 = v176;
    v70 = v171;
    if ( !v95 )
    {
LABEL_34:
      if ( v72 )
        v73 = v71[16];
      else
        v73 = v71[15];
      goto LABEL_36;
    }
  }
  v73 = v71[17];
LABEL_36:
  v74 = (__int64 *)a1[1];
  if ( v70 )
  {
    v127 = sub_21278D0((__int64)v71, a1[1], v202, (__int64)v67);
    v197 = (const void **)v128;
    v183 = v127;
    v130 = (const void ***)sub_1D252B0(a1[1], v202, (__int64)v67, v127, v128);
    *((_QWORD *)&v161 + 1) = 2;
    *(_QWORD *)&v161 = &v226;
    if ( *(_WORD *)(v65 + 24) == 52 )
    {
      *(_QWORD *)a3 = sub_1D36D80(
                        (__int64 *)a1[1],
                        71,
                        (__int64)&v217,
                        v130,
                        v129,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        v131,
                        v161);
      *(_DWORD *)(a3 + 8) = v143;
      *((_QWORD *)&v155 + 1) = 2;
      *(_QWORD *)&v155 = v228;
      v133 = sub_1D359D0(
               (__int64 *)a1[1],
               52,
               (__int64)&v217,
               v202,
               v67,
               0,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v155);
      v135 = v202;
      v177 = 53;
    }
    else
    {
      *(_QWORD *)a3 = sub_1D36D80(
                        (__int64 *)a1[1],
                        73,
                        (__int64)&v217,
                        v130,
                        v129,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        v131,
                        v161);
      *(_DWORD *)(a3 + 8) = v132;
      *((_QWORD *)&v153 + 1) = 2;
      *(_QWORD *)&v153 = v228;
      v133 = sub_1D359D0(
               (__int64 *)a1[1],
               53,
               (__int64)&v217,
               v202,
               v67,
               0,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v153);
      v135 = v202;
      v177 = 52;
    }
    *(_QWORD *)a4 = v133;
    *(_DWORD *)(a4 + 8) = v134;
    result = *(_QWORD *)a3;
    v206 = *(_QWORD *)a3;
    v211 = 1;
    if ( v73 != 1 )
    {
      if ( v73 == 2 )
      {
        v199 = v135;
        v208 = sub_1D322C0(
                 (__int64 *)a1[1],
                 v206,
                 1,
                 (__int64)&v217,
                 v135,
                 v67,
                 *(double *)v12.m128_u64,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64);
        *((_QWORD *)&v163 + 1) = v145;
        *(_QWORD *)&v163 = v208;
        *(_QWORD *)a4 = sub_1D332F0(
                          (__int64 *)a1[1],
                          v177,
                          (__int64)&v217,
                          v199,
                          v67,
                          0,
                          *(double *)v12.m128_u64,
                          *(double *)v13.m128i_i64,
                          v14,
                          *(_QWORD *)a4,
                          *(_QWORD *)(a4 + 8),
                          v163);
        result = v146;
        *(_DWORD *)(a4 + 8) = v146;
        goto LABEL_22;
      }
      if ( v73 )
        goto LABEL_22;
      v136 = (__int64 *)a1[1];
      v189 = v135;
      v137 = sub_1D38BB0(
               (__int64)v136,
               1,
               (__int64)&v217,
               v183,
               v197,
               0,
               (__m128i)v12,
               *(double *)v13.m128i_i64,
               v14,
               0);
      *((_QWORD *)&v154 + 1) = 1;
      *(_QWORD *)&v154 = v206;
      v139 = sub_1D332F0(
               v136,
               118,
               (__int64)&v217,
               v183,
               v197,
               0,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               v14,
               v137,
               v138,
               v154);
      v135 = v189;
      v206 = (__int64)v139;
      v211 = v140;
    }
    v198 = v135;
    v207 = sub_1D323C0(
             (__int64 *)a1[1],
             v206,
             v211,
             (__int64)&v217,
             v135,
             v67,
             *(double *)v12.m128_u64,
             *(double *)v13.m128i_i64,
             *(double *)v14.m128i_i64);
    *((_QWORD *)&v162 + 1) = v141 | v211 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v162 = v207;
    *(_QWORD *)a4 = sub_1D332F0(
                      (__int64 *)a1[1],
                      *(unsigned __int16 *)(v65 + 24),
                      (__int64)&v217,
                      v198,
                      v67,
                      0,
                      *(double *)v12.m128_u64,
                      *(double *)v13.m128i_i64,
                      v14,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      v162);
    result = v142;
    *(_DWORD *)(a4 + 8) = v142;
    goto LABEL_22;
  }
  if ( v69 == 52 )
  {
    *((_QWORD *)&v160 + 1) = 2;
    *(_QWORD *)&v160 = &v226;
    *(_QWORD *)a3 = sub_1D359D0(
                      v74,
                      52,
                      (__int64)&v217,
                      v202,
                      v67,
                      0,
                      *(double *)v12.m128_u64,
                      *(double *)v13.m128i_i64,
                      v14,
                      v160);
    *(_DWORD *)(a3 + 8) = v101;
    *((_QWORD *)&v151 + 1) = 2;
    *(_QWORD *)&v151 = v228;
    *(_QWORD *)a4 = sub_1D359D0(
                      (__int64 *)a1[1],
                      52,
                      (__int64)&v217,
                      v202,
                      v67,
                      0,
                      *(double *)v12.m128_u64,
                      *(double *)v13.m128i_i64,
                      v14,
                      v151);
    *(_DWORD *)(a4 + 8) = v102;
    v103 = (__int64 *)a1[1];
    v104 = sub_21278D0(*a1, (__int64)v103, v202, (__int64)v67);
    v106 = sub_1F81070(
             v103,
             (__int64)&v217,
             v104,
             v105,
             *(_QWORD *)a3,
             *(__int16 **)(a3 + 8),
             v12,
             *(double *)v13.m128i_i64,
             v14,
             *(_OWORD *)&v226,
             0xCu);
    v108 = (__int16 *)v107;
    if ( v73 == 1 )
    {
      *(_QWORD *)&v144 = sub_1D323C0(
                           (__int64 *)a1[1],
                           (__int64)v106,
                           v107,
                           (__int64)&v217,
                           v202,
                           v67,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)v14.m128i_i64);
      *(_QWORD *)a4 = sub_1D332F0(
                        (__int64 *)a1[1],
                        52,
                        (__int64)&v217,
                        v202,
                        v67,
                        0,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        *(_QWORD *)a4,
                        *(_QWORD *)(a4 + 8),
                        v144);
    }
    else
    {
      v109 = (__int64 *)a1[1];
      v195 = v202;
      v182 = (unsigned __int64)v106;
      v110 = sub_1D38BB0((__int64)v109, 0, (__int64)&v217, v202, v67, 0, (__m128i)v12, *(double *)v13.m128i_i64, v14, 0);
      v209 = v111;
      v204 = v110;
      *(_QWORD *)&v112 = sub_1D38BB0(
                           a1[1],
                           1,
                           (__int64)&v217,
                           v195,
                           v67,
                           0,
                           (__m128i)v12,
                           *(double *)v13.m128i_i64,
                           v14,
                           0);
      v113 = sub_1F810E0(
               v109,
               (__int64)&v217,
               v195,
               v67,
               v182,
               v108,
               v12,
               *(double *)v13.m128i_i64,
               v14,
               v112,
               v204,
               v209);
      v114 = (__int64 *)a1[1];
      v210 = v115;
      v205 = (__int64)v113;
      v116 = sub_21278D0(*a1, (__int64)v114, v195, (__int64)v67);
      v118 = sub_1F81070(
               v114,
               (__int64)&v217,
               v116,
               v117,
               *(_QWORD *)a3,
               *(__int16 **)(a3 + 8),
               v12,
               *(double *)v13.m128i_i64,
               v14,
               v227,
               0xCu);
      v119 = (__int64 *)a1[1];
      v200 = v120;
      v188 = v195;
      v196 = (unsigned __int64)v118;
      *(_QWORD *)&v121 = sub_1D38BB0(
                           (__int64)v119,
                           1,
                           (__int64)&v217,
                           v188,
                           v67,
                           0,
                           (__m128i)v12,
                           *(double *)v13.m128i_i64,
                           v14,
                           0);
      *(_QWORD *)&v122 = sub_1F810E0(
                           v119,
                           (__int64)&v217,
                           v188,
                           v67,
                           v196,
                           v200,
                           v12,
                           *(double *)v13.m128i_i64,
                           v14,
                           v121,
                           v205,
                           v210);
      *(_QWORD *)a4 = sub_1D332F0(
                        (__int64 *)a1[1],
                        52,
                        (__int64)&v217,
                        v188,
                        v67,
                        0,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        v14,
                        *(_QWORD *)a4,
                        *(_QWORD *)(a4 + 8),
                        v122);
    }
    result = v123;
    *(_DWORD *)(a4 + 8) = v123;
  }
  else
  {
    *((_QWORD *)&v158 + 1) = 2;
    *(_QWORD *)&v158 = &v226;
    *(_QWORD *)a3 = sub_1D359D0(
                      v74,
                      53,
                      (__int64)&v217,
                      v202,
                      v67,
                      0,
                      *(double *)v12.m128_u64,
                      *(double *)v13.m128i_i64,
                      v14,
                      v158);
    v170 = v202;
    *(_DWORD *)(a3 + 8) = v75;
    *((_QWORD *)&v149 + 1) = 2;
    *(_QWORD *)&v149 = v228;
    v76 = sub_1D359D0(
            (__int64 *)a1[1],
            53,
            (__int64)&v217,
            v202,
            v67,
            0,
            *(double *)v12.m128_u64,
            *(double *)v13.m128i_i64,
            v14,
            v149);
    v212 = v77;
    v78 = v226.m128_u64[0];
    *(_QWORD *)a4 = v76;
    *(_DWORD *)(a4 + 8) = v212;
    v79 = (__int64 *)a1[1];
    v80 = (unsigned __int8 *)(*(_QWORD *)(v78 + 40) + 16LL * v226.m128_u32[2]);
    v186 = *a1;
    v174 = *((_QWORD *)v80 + 1);
    v180 = *v80;
    v203 = v79[6];
    v193 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 264LL);
    v81 = sub_1E0A0C0(v79[4]);
    v82 = v193(v186, v81, v203, v180, v174);
    v84 = sub_1F81070(
            v79,
            (__int64)&v217,
            v82,
            v83,
            v226.m128_u64[0],
            (__int16 *)v226.m128_u64[1],
            v12,
            *(double *)v13.m128i_i64,
            v14,
            v227,
            0xCu);
    v194 = (__int16 *)v85;
    if ( v73 == 1 )
    {
      v90 = sub_1D323C0(
              (__int64 *)a1[1],
              (__int64)v84,
              v85,
              (__int64)&v217,
              v170,
              v67,
              *(double *)v12.m128_u64,
              *(double *)v13.m128i_i64,
              *(double *)v14.m128i_i64);
    }
    else
    {
      v86 = (__int64 *)a1[1];
      v175 = (unsigned __int64)v84;
      v87 = sub_1D38BB0((__int64)v86, 0, (__int64)&v217, v170, v67, 0, (__m128i)v12, *(double *)v13.m128i_i64, v14, 0);
      v190 = v88;
      v187 = v87;
      *(_QWORD *)&v89 = sub_1D38BB0(
                          a1[1],
                          1,
                          (__int64)&v217,
                          v170,
                          v67,
                          0,
                          (__m128i)v12,
                          *(double *)v13.m128i_i64,
                          v14,
                          0);
      v90 = (__int64)sub_1F810E0(
                       v86,
                       (__int64)&v217,
                       v170,
                       v67,
                       v175,
                       v194,
                       v12,
                       *(double *)v13.m128i_i64,
                       v14,
                       v89,
                       v187,
                       v190);
    }
    *((_QWORD *)&v159 + 1) = v91;
    *(_QWORD *)&v159 = v90;
    *(_QWORD *)a4 = sub_1D332F0(
                      (__int64 *)a1[1],
                      53,
                      (__int64)&v217,
                      v170,
                      v67,
                      0,
                      *(double *)v12.m128_u64,
                      *(double *)v13.m128i_i64,
                      v14,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      v159);
    result = v92;
    *(_DWORD *)(a4 + 8) = v92;
  }
LABEL_22:
  if ( v217 )
    return sub_161E7C0((__int64)&v217, v217);
  return result;
}
