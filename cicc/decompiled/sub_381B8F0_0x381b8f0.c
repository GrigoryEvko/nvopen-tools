// Function: sub_381B8F0
// Address: 0x381b8f0
//
void __fastcall sub_381B8F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r15d
  __int64 v6; // rsi
  int v7; // eax
  unsigned __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r9
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __int64 v14; // rax
  __m128i v15; // xmm3
  __int64 v16; // r14
  unsigned __int16 v17; // r11
  __int64 v18; // rax
  __int64 v19; // r8
  unsigned __int16 v20; // bx
  __int64 v21; // r10
  __int64 v22; // r13
  __int64 v23; // r12
  unsigned int v24; // r8d
  unsigned __int16 v25; // r11
  char v27; // al
  __int64 v28; // rdi
  unsigned int v29; // r15d
  unsigned __int16 v30; // ax
  unsigned int v31; // r8d
  char v32; // al
  __int64 *v33; // rdi
  unsigned __int16 v34; // ax
  char v35; // al
  __int64 v36; // r9
  _DWORD *v37; // r8
  char v38; // dl
  unsigned __int16 v39; // cx
  _QWORD *v40; // rdi
  int v41; // edx
  __int64 v42; // r9
  unsigned __int8 *v43; // rax
  int v44; // edx
  __int64 v45; // rdx
  unsigned __int16 *v46; // rax
  unsigned int v47; // eax
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // r13
  __int128 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  __int128 v54; // rax
  __int128 v55; // rax
  unsigned int v56; // edx
  __int64 v57; // r9
  __int64 v58; // r10
  int v59; // edx
  __int64 (__fastcall *v60)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r9
  _QWORD *v64; // rdi
  unsigned int *v65; // r13
  __int64 v66; // r12
  int v67; // edx
  _QWORD *v68; // rdi
  __int64 v69; // r9
  int v70; // edx
  unsigned int v71; // r15d
  unsigned int v72; // eax
  __int64 v73; // rdx
  unsigned int *v74; // rax
  __int64 v75; // rdx
  _QWORD *v76; // rdi
  int v77; // edx
  unsigned int *v78; // r9
  int v79; // r12d
  int v80; // eax
  bool v81; // al
  unsigned __int8 *v82; // rax
  int v83; // edx
  int v84; // ecx
  unsigned __int8 *v85; // rbx
  bool v86; // cc
  bool v87; // al
  __int64 v88; // rdx
  unsigned __int8 *v89; // rax
  int v90; // edx
  int v91; // edx
  _QWORD *v92; // rdi
  __int64 v93; // r9
  unsigned __int8 *v94; // rax
  __int64 v95; // rdi
  int v96; // edx
  _QWORD *v97; // r12
  __int128 v98; // rax
  unsigned int v99; // eax
  __int64 v100; // rdx
  __int64 v101; // r13
  __int128 v102; // kr00_16
  unsigned int v103; // esi
  __int128 v104; // rax
  unsigned int v105; // edx
  __int64 v106; // r10
  __int128 v107; // rax
  __int128 v108; // rax
  unsigned int v109; // edx
  __int64 v110; // r12
  __int64 v111; // r13
  __int64 v112; // r9
  int v113; // edx
  __int64 v114; // r9
  int v115; // edx
  int v116; // edx
  unsigned int *v117; // r9
  int v118; // r12d
  int v119; // eax
  bool v120; // al
  unsigned __int8 *v121; // rax
  int v122; // edx
  int v123; // ecx
  unsigned __int8 *v124; // rbx
  unsigned int v125; // eax
  __int64 v126; // rdx
  __int64 v127; // r13
  unsigned int *v128; // rax
  __int64 v129; // rdx
  __int64 v130; // r9
  _QWORD *v131; // rdi
  int v132; // edx
  __int64 v133; // r9
  int v134; // edx
  unsigned __int8 *v135; // r10
  __int64 v136; // r11
  __int128 v137; // rax
  __int64 v138; // r9
  unsigned int v139; // edx
  unsigned int v140; // edx
  __int64 v141; // r9
  int v142; // edx
  bool v143; // al
  __int128 v144; // rax
  unsigned int v145; // eax
  __int64 v146; // rdx
  int v147; // edx
  unsigned int v148; // eax
  __int64 v149; // rdx
  __int64 v150; // r12
  __int64 v151; // r13
  __int128 v152; // rax
  unsigned int v153; // edx
  int v154; // edx
  int v155; // edx
  __int64 v156; // r9
  unsigned int v157; // edx
  __int64 v158; // r9
  int v159; // edx
  __int128 v160; // rax
  unsigned int v161; // eax
  __int64 v162; // rdx
  __int128 v163; // [rsp-20h] [rbp-340h]
  __int128 v164; // [rsp-20h] [rbp-340h]
  __int128 v165; // [rsp-10h] [rbp-330h]
  __int128 v166; // [rsp-10h] [rbp-330h]
  __int128 v167; // [rsp-10h] [rbp-330h]
  __int128 v168; // [rsp-10h] [rbp-330h]
  __int128 v169; // [rsp-10h] [rbp-330h]
  __int128 v170; // [rsp-10h] [rbp-330h]
  __int128 v171; // [rsp-10h] [rbp-330h]
  __int128 v172; // [rsp+0h] [rbp-320h]
  __int128 v173; // [rsp+0h] [rbp-320h]
  __int128 v174; // [rsp+0h] [rbp-320h]
  __int128 v175; // [rsp+0h] [rbp-320h]
  __int128 v176; // [rsp+0h] [rbp-320h]
  __int128 v177; // [rsp+0h] [rbp-320h]
  __int128 v178; // [rsp+0h] [rbp-320h]
  __int128 v179; // [rsp+0h] [rbp-320h]
  __int128 v180; // [rsp+0h] [rbp-320h]
  __int128 v181; // [rsp+0h] [rbp-320h]
  __int128 v182; // [rsp+0h] [rbp-320h]
  __int128 v183; // [rsp+0h] [rbp-320h]
  __int128 v184; // [rsp+0h] [rbp-320h]
  __int128 v185; // [rsp+0h] [rbp-320h]
  __int128 v186; // [rsp+0h] [rbp-320h]
  _QWORD *v187; // [rsp+10h] [rbp-310h]
  __int64 v188; // [rsp+18h] [rbp-308h]
  __int64 v189; // [rsp+18h] [rbp-308h]
  __int64 v190; // [rsp+18h] [rbp-308h]
  unsigned int v192; // [rsp+20h] [rbp-300h]
  char v193; // [rsp+20h] [rbp-300h]
  __int128 v194; // [rsp+20h] [rbp-300h]
  __int64 v195; // [rsp+20h] [rbp-300h]
  unsigned int v196; // [rsp+20h] [rbp-300h]
  unsigned int v197; // [rsp+20h] [rbp-300h]
  int v198; // [rsp+30h] [rbp-2F0h]
  __int128 v199; // [rsp+30h] [rbp-2F0h]
  __int64 v200; // [rsp+30h] [rbp-2F0h]
  unsigned int *v201; // [rsp+30h] [rbp-2F0h]
  unsigned int v202; // [rsp+30h] [rbp-2F0h]
  int v203; // [rsp+30h] [rbp-2F0h]
  unsigned int *v204; // [rsp+30h] [rbp-2F0h]
  unsigned int v205; // [rsp+30h] [rbp-2F0h]
  __int128 v206; // [rsp+30h] [rbp-2F0h]
  _QWORD *v208; // [rsp+40h] [rbp-2E0h]
  __int64 v209; // [rsp+40h] [rbp-2E0h]
  __int128 v210; // [rsp+40h] [rbp-2E0h]
  __int64 v211; // [rsp+40h] [rbp-2E0h]
  _QWORD *v212; // [rsp+40h] [rbp-2E0h]
  unsigned __int16 v213; // [rsp+50h] [rbp-2D0h]
  int v214; // [rsp+50h] [rbp-2D0h]
  __int128 v215; // [rsp+50h] [rbp-2D0h]
  __int64 v216; // [rsp+50h] [rbp-2D0h]
  char v217; // [rsp+50h] [rbp-2D0h]
  __int128 v218; // [rsp+50h] [rbp-2D0h]
  __int64 v219; // [rsp+50h] [rbp-2D0h]
  __int64 v220; // [rsp+58h] [rbp-2C8h]
  __int64 v222; // [rsp+68h] [rbp-2B8h]
  __int64 v223; // [rsp+68h] [rbp-2B8h]
  __int64 v224; // [rsp+68h] [rbp-2B8h]
  __int64 *v225; // [rsp+68h] [rbp-2B8h]
  _DWORD *v226; // [rsp+68h] [rbp-2B8h]
  int v227; // [rsp+A8h] [rbp-278h]
  unsigned __int8 *v228; // [rsp+130h] [rbp-1F0h]
  unsigned __int8 *v229; // [rsp+150h] [rbp-1D0h]
  unsigned __int8 *v230; // [rsp+1C0h] [rbp-160h]
  unsigned __int8 *v231; // [rsp+1E0h] [rbp-140h]
  unsigned __int8 *v232; // [rsp+200h] [rbp-120h]
  unsigned __int8 *v233; // [rsp+220h] [rbp-100h]
  __int64 v234; // [rsp+230h] [rbp-F0h] BYREF
  int v235; // [rsp+238h] [rbp-E8h]
  __m128i v236; // [rsp+240h] [rbp-E0h] BYREF
  __m128i v237; // [rsp+250h] [rbp-D0h] BYREF
  __m128i v238; // [rsp+260h] [rbp-C0h] BYREF
  __m128i v239; // [rsp+270h] [rbp-B0h] BYREF
  __int128 v240; // [rsp+280h] [rbp-A0h] BYREF
  __m128i v241; // [rsp+290h] [rbp-90h]
  unsigned __int64 v242; // [rsp+2A0h] [rbp-80h] BYREF
  __int64 v243; // [rsp+2A8h] [rbp-78h]
  unsigned __int64 v244; // [rsp+2B0h] [rbp-70h] BYREF
  unsigned int v245; // [rsp+2B8h] [rbp-68h]
  __int128 v246; // [rsp+2C0h] [rbp-60h] BYREF
  __m128i v247; // [rsp+2D0h] [rbp-50h]
  unsigned __int8 *v248; // [rsp+2E0h] [rbp-40h]
  __int64 v249; // [rsp+2E8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v234 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v234, v6, 1);
  v7 = *(_DWORD *)(a2 + 72);
  v236.m128i_i32[2] = 0;
  v237.m128i_i32[2] = 0;
  v235 = v7;
  v8 = *(unsigned __int64 **)(a2 + 40);
  v238.m128i_i32[2] = 0;
  v239.m128i_i32[2] = 0;
  v9 = v8[1];
  v236.m128i_i64[0] = 0;
  v237.m128i_i64[0] = 0;
  v238.m128i_i64[0] = 0;
  v239.m128i_i64[0] = 0;
  sub_375E510((__int64)a1, *v8, v9, (__int64)&v236, (__int64)&v237);
  sub_375E510(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v238,
    (__int64)&v239);
  v10 = *a1;
  v11 = _mm_loadu_si128(&v236);
  v188 = a2;
  v12 = _mm_loadu_si128(&v238);
  v13 = _mm_loadu_si128(&v237);
  v14 = *(_QWORD *)(v236.m128i_i64[0] + 48) + 16LL * v236.m128i_u32[2];
  v15 = _mm_loadu_si128(&v239);
  v16 = *(_QWORD *)(v14 + 8);
  v17 = *(_WORD *)v14;
  v248 = 0;
  v18 = a1[1];
  LODWORD(v249) = 0;
  v213 = v17;
  v19 = v16;
  v20 = v17;
  v21 = *(_QWORD *)(v18 + 64);
  v22 = v10;
  v240 = (__int128)v11;
  v241 = v12;
  v23 = v21;
  v246 = (__int128)v13;
  v247 = v15;
  while ( 1 )
  {
    LOWORD(v4) = v20;
    v222 = v19;
    sub_2FE6CC0((__int64)&v242, v22, v23, v4, v19);
    if ( !(_BYTE)v242 )
      break;
    if ( (_BYTE)v242 != 2 )
      BUG();
    v60 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v22 + 592LL);
    if ( v60 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v242, v22, v23, v4, v222);
      v20 = v243;
      v19 = v244;
    }
    else
    {
      v4 = v60(v22, v23, v4, v222);
      v20 = v4;
      v19 = v88;
    }
  }
  v25 = v20;
  v27 = sub_3813820(v22, (unsigned int)(*(_DWORD *)(v188 + 24) != 56) + 72, v25, 0, v24);
  v28 = *a1;
  if ( v27 )
  {
    v225 = (__int64 *)a1[1];
    v71 = v213;
    v72 = sub_38137B0(v28, (__int64)v225, v213, v16);
    v74 = (unsigned int *)sub_33E5110(v225, v213, v16, v72, v73);
    v76 = (_QWORD *)a1[1];
    v216 = v75;
    if ( *(_DWORD *)(v188 + 24) == 56 )
    {
      *((_QWORD *)&v180 + 1) = 2;
      *(_QWORD *)&v180 = &v240;
      v204 = v74;
      v233 = sub_3411630(v76, 77, (__int64)&v234, v74, v75, (__int64)v74, v180);
      *(_QWORD *)a3 = v233;
      v248 = v233;
      *(_DWORD *)(a3 + 8) = v116;
      LODWORD(v249) = 1;
      sub_33DD090((__int64)&v242, a1[1], (__int64)v248, v249, 0);
      v118 = v243;
      if ( !(_DWORD)v243
        || ((v117 = v204, (unsigned int)v243 <= 0x40)
          ? (v120 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v243) == v242)
          : (v119 = sub_C445E0((__int64)&v242), v117 = v204, v120 = v118 == v119),
            v120) )
      {
        *((_QWORD *)&v184 + 1) = 2;
        *(_QWORD *)&v184 = &v246;
        v124 = sub_33FC220((_QWORD *)a1[1], 56, (__int64)&v234, v71, v16, (__int64)v117, v184);
        v123 = v147;
      }
      else
      {
        *((_QWORD *)&v181 + 1) = 3;
        *(_QWORD *)&v181 = &v246;
        v121 = sub_3411630((_QWORD *)a1[1], 72, (__int64)&v234, v117, v216, (__int64)v117, v181);
        v123 = v122;
        v124 = v121;
      }
      *(_QWORD *)a4 = v124;
      *(_DWORD *)(a4 + 8) = v123;
      sub_969240((__int64 *)&v244);
      sub_969240((__int64 *)&v242);
    }
    else
    {
      *((_QWORD *)&v175 + 1) = 2;
      *(_QWORD *)&v175 = &v240;
      v201 = v74;
      v232 = sub_3411630(v76, 79, (__int64)&v234, v74, v75, (__int64)v74, v175);
      *(_QWORD *)a3 = v232;
      v248 = v232;
      *(_DWORD *)(a3 + 8) = v77;
      LODWORD(v249) = 1;
      sub_33DD090((__int64)&v242, a1[1], (__int64)v248, v249, 0);
      v79 = v243;
      if ( !(_DWORD)v243
        || ((v78 = v201, (unsigned int)v243 <= 0x40)
          ? (v81 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v243) == v242)
          : (v80 = sub_C445E0((__int64)&v242), v78 = v201, v81 = v79 == v80),
            v81) )
      {
        *((_QWORD *)&v177 + 1) = 2;
        *(_QWORD *)&v177 = &v246;
        v89 = sub_33FC220((_QWORD *)a1[1], 57, (__int64)&v234, v71, v16, (__int64)v78, v177);
        v84 = v90;
        v85 = v89;
      }
      else
      {
        *((_QWORD *)&v176 + 1) = 3;
        *(_QWORD *)&v176 = &v246;
        v82 = sub_3411630((_QWORD *)a1[1], 73, (__int64)&v234, v78, v216, (__int64)v78, v176);
        v84 = v83;
        v85 = v82;
      }
      v86 = v245 <= 0x40;
      *(_QWORD *)a4 = v85;
      *(_DWORD *)(a4 + 8) = v84;
      if ( !v86 && v244 )
        j_j___libc_free_0_0(v244);
      if ( (unsigned int)v243 > 0x40 && v242 )
        j_j___libc_free_0_0(v242);
    }
  }
  else
  {
    v29 = v213;
    v223 = *a1;
    v30 = sub_3814400(v28, *(_QWORD *)(a1[1] + 64), v213, v16);
    v32 = sub_3813820(v223, (unsigned int)(*(_DWORD *)(v188 + 24) != 56) + 68, v30, 0, v31);
    v33 = (__int64 *)a1[1];
    if ( !v32 )
    {
      v224 = *a1;
      v34 = sub_3814400(*a1, v33[8], v213, v16);
      v198 = *(_DWORD *)(v188 + 24);
      v35 = sub_3813820(v224, 2 * (unsigned int)(v198 != 56) + 77, v34, 0, v224);
      v37 = (_DWORD *)*a1;
      v243 = v16;
      v38 = v35;
      LOWORD(v242) = v213;
      if ( v213 )
      {
        v39 = v213 - 17;
        if ( (unsigned __int16)(v213 - 10) > 6u && (unsigned __int16)(v213 - 126) > 0x31u )
        {
          if ( v39 > 0xD3u )
          {
LABEL_11:
            v214 = v37[15];
            goto LABEL_12;
          }
          goto LABEL_45;
        }
        if ( v39 <= 0xD3u )
        {
LABEL_45:
          v214 = v37[17];
LABEL_12:
          v40 = (_QWORD *)a1[1];
          if ( v38 )
          {
            v125 = sub_38137B0((__int64)v37, a1[1], v29, v16);
            v127 = v126;
            v205 = v125;
            v128 = (unsigned int *)sub_33E5110((__int64 *)a1[1], v29, v16, v125, v126);
            v131 = (_QWORD *)a1[1];
            *((_QWORD *)&v182 + 1) = 2;
            *(_QWORD *)&v182 = &v240;
            if ( *(_DWORD *)(v188 + 24) == 56 )
            {
              *(_QWORD *)a3 = sub_3411630(v131, 77, (__int64)&v234, v128, v129, v130, v182);
              *(_DWORD *)(a3 + 8) = v155;
              *((_QWORD *)&v171 + 1) = 2;
              *(_QWORD *)&v171 = &v246;
              v196 = 57;
              *(_QWORD *)a4 = sub_33FC220((_QWORD *)a1[1], 56, (__int64)&v234, v29, v16, v156, v171);
            }
            else
            {
              *(_QWORD *)a3 = sub_3411630(v131, 79, (__int64)&v234, v128, v129, v130, v182);
              *(_DWORD *)(a3 + 8) = v132;
              *((_QWORD *)&v169 + 1) = 2;
              *(_QWORD *)&v169 = &v246;
              v196 = 56;
              *(_QWORD *)a4 = sub_33FC220((_QWORD *)a1[1], 57, (__int64)&v234, v29, v16, v133, v169);
            }
            *(_DWORD *)(a4 + 8) = v134;
            v135 = *(unsigned __int8 **)a3;
            v136 = 1;
            if ( v214 != 1 )
            {
              if ( v214 == 2 )
              {
                v228 = sub_33FB160(a1[1], *(_QWORD *)a3, 1u, (__int64)&v234, v29, v16, v11);
                *((_QWORD *)&v186 + 1) = v157;
                *(_QWORD *)&v186 = v228;
                *(_QWORD *)a4 = sub_3406EB0((_QWORD *)a1[1], v196, (__int64)&v234, v29, v16, v158, *(_OWORD *)a4, v186);
                *(_DWORD *)(a4 + 8) = v159;
                goto LABEL_17;
              }
              if ( v214 )
                goto LABEL_17;
              v219 = *(_QWORD *)a3;
              v212 = (_QWORD *)a1[1];
              *(_QWORD *)&v137 = sub_3400BD0((__int64)v212, 1, (__int64)&v234, v205, v127, 0, v11, 0);
              *((_QWORD *)&v170 + 1) = 1;
              *(_QWORD *)&v170 = v219;
              v135 = sub_3406EB0(v212, 0xBAu, (__int64)&v234, v205, v127, v138, v137, v170);
              v136 = v139;
            }
            v220 = v136;
            v229 = sub_33FB310(a1[1], (__int64)v135, v136, (__int64)&v234, v29, v16, v11);
            *((_QWORD *)&v183 + 1) = v140 | v220 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v183 = v229;
            *(_QWORD *)a4 = sub_3406EB0(
                              (_QWORD *)a1[1],
                              *(_DWORD *)(v188 + 24),
                              (__int64)&v234,
                              v29,
                              v16,
                              v141,
                              *(_OWORD *)a4,
                              v183);
            *(_DWORD *)(a4 + 8) = v142;
            goto LABEL_17;
          }
          if ( v198 != 56 )
          {
            *((_QWORD *)&v172 + 1) = 2;
            *(_QWORD *)&v172 = &v240;
            *(_QWORD *)a3 = sub_33FC220(v40, 57, (__int64)&v234, v29, v16, v36, v172);
            *(_DWORD *)(a3 + 8) = v41;
            *((_QWORD *)&v165 + 1) = 2;
            *(_QWORD *)&v165 = &v246;
            v43 = sub_33FC220((_QWORD *)a1[1], 57, (__int64)&v234, v29, v16, v42, v165);
            v227 = v44;
            v45 = v240;
            *(_QWORD *)a4 = v43;
            *(_DWORD *)(a4 + 8) = v227;
            v46 = (unsigned __int16 *)(*(_QWORD *)(v45 + 48) + 16LL * DWORD2(v240));
            v208 = (_QWORD *)a1[1];
            v47 = sub_38137B0(*a1, (__int64)v208, *v46, *((_QWORD *)v46 + 1));
            v48 = v240;
            v189 = v49;
            v192 = v47;
            v50 = *((_QWORD *)&v240 + 1);
            v199 = (__int128)v241;
            *(_QWORD *)&v51 = sub_33ED040(v208, 0xCu);
            *((_QWORD *)&v163 + 1) = v50;
            *(_QWORD *)&v163 = v48;
            v209 = sub_340F900(v208, 0xD0u, (__int64)&v234, v192, v189, (__int64)v208, v163, v199, v51);
            v200 = v52;
            if ( v214 == 1 )
            {
              v58 = (__int64)sub_33FB310(a1[1], v209, v52, (__int64)&v234, v29, v16, v11);
            }
            else
            {
              v53 = a1[1];
              *(_QWORD *)&v54 = sub_3400BD0(v53, 0, (__int64)&v234, v29, v16, 0, v11, 0);
              v215 = v54;
              *(_QWORD *)&v55 = sub_3400BD0(a1[1], 1, (__int64)&v234, v29, v16, 0, v11, 0);
              v58 = sub_3288B20(v53, (int)&v234, v29, v16, v209, v200, v55, v215, 0);
            }
            *((_QWORD *)&v173 + 1) = v56;
            *(_QWORD *)&v173 = v58;
            *(_QWORD *)a4 = sub_3406EB0((_QWORD *)a1[1], 0x39u, (__int64)&v234, v29, v16, v57, *(_OWORD *)a4, v173);
            *(_DWORD *)(a4 + 8) = v59;
            goto LABEL_17;
          }
          *((_QWORD *)&v178 + 1) = 2;
          *(_QWORD *)&v178 = &v240;
          v94 = sub_33FC220(v40, 56, (__int64)&v234, v29, v16, v36, v178);
          v95 = v241.m128i_i64[0];
          *(_QWORD *)a3 = v94;
          *(_DWORD *)(a3 + 8) = v96;
          if ( sub_33CF4D0(v95) )
          {
            v97 = (_QWORD *)a1[1];
            *(_QWORD *)&v98 = sub_3400BD0((__int64)v97, 0, (__int64)&v234, v29, v16, 0, v11, 0);
            v194 = v98;
            v99 = sub_38137B0(*a1, a1[1], v29, v16);
            v101 = v100;
            v202 = v99;
            v102 = *(_OWORD *)a3;
          }
          else
          {
            if ( !sub_33CF460(v241.m128i_i64[0]) )
            {
              v187 = (_QWORD *)a1[1];
              v148 = sub_38137B0(*a1, (__int64)v187, v29, v16);
              v190 = v149;
              v197 = v148;
              v150 = *(_QWORD *)a3;
              v151 = *(_QWORD *)(a3 + 8);
              v206 = v240;
              *(_QWORD *)&v152 = sub_33ED040(v187, 0xCu);
              *((_QWORD *)&v164 + 1) = v151;
              *(_QWORD *)&v164 = v150;
              v211 = sub_340F900(v187, 0xD0u, (__int64)&v234, v197, v190, (__int64)v187, v164, v206, v152);
              v106 = v153;
              goto LABEL_50;
            }
            v143 = sub_33CF460(v247.m128i_i64[0]);
            v97 = (_QWORD *)a1[1];
            if ( !v143 )
            {
              *(_QWORD *)&v160 = sub_3400BD0((__int64)v97, 0, (__int64)&v234, v29, v16, 0, v11, 0);
              v194 = v160;
              v161 = sub_38137B0(*a1, a1[1], v29, v16);
              v103 = 22;
              v101 = v162;
              v202 = v161;
              v210 = v240;
              goto LABEL_49;
            }
            *(_QWORD *)&v144 = sub_3400BD0((__int64)v97, 0, (__int64)&v234, v29, v16, 0, v11, 0);
            v194 = v144;
            v145 = sub_38137B0(*a1, a1[1], v29, v16);
            v102 = v240;
            v101 = v146;
            v202 = v145;
          }
          v210 = v102;
          v103 = 17;
LABEL_49:
          *(_QWORD *)&v104 = sub_33ED040(v97, v103);
          v211 = sub_340F900(v97, 0xD0u, (__int64)&v234, v202, v101, *((__int64 *)&v194 + 1), v210, v194, v104);
          v106 = v105;
LABEL_50:
          if ( v214 == 1 )
          {
            v110 = (__int64)sub_33FB310(a1[1], v211, v106, (__int64)&v234, v29, v16, v11);
          }
          else
          {
            v195 = v106;
            v203 = a1[1];
            *(_QWORD *)&v107 = sub_3400BD0(a1[1], 0, (__int64)&v234, v29, v16, 0, v11, 0);
            v218 = v107;
            *(_QWORD *)&v108 = sub_3400BD0(a1[1], 1, (__int64)&v234, v29, v16, 0, v11, 0);
            v110 = sub_3288B20(v203, (int)&v234, v29, v16, v211, v195, v108, v218, 0);
          }
          v111 = v109;
          if ( sub_33CF460(v241.m128i_i64[0]) && sub_33CF460(v247.m128i_i64[0]) )
          {
            *((_QWORD *)&v185 + 1) = v111;
            *(_QWORD *)&v185 = v110;
            *(_QWORD *)a4 = sub_3406EB0((_QWORD *)a1[1], 0x39u, (__int64)&v234, v29, v16, v112, v246, v185);
            *(_DWORD *)(a4 + 8) = v154;
          }
          else
          {
            *((_QWORD *)&v179 + 1) = 2;
            *(_QWORD *)&v179 = &v246;
            *(_QWORD *)a4 = sub_33FC220((_QWORD *)a1[1], 56, (__int64)&v234, v29, v16, v112, v179);
            *(_DWORD *)(a4 + 8) = v113;
            *((_QWORD *)&v168 + 1) = v111;
            *(_QWORD *)&v168 = v110;
            *(_QWORD *)a4 = sub_3406EB0((_QWORD *)a1[1], 0x38u, (__int64)&v234, v29, v16, v114, *(_OWORD *)a4, v168);
            *(_DWORD *)(a4 + 8) = v115;
          }
          goto LABEL_17;
        }
      }
      else
      {
        v226 = v37;
        v217 = v35;
        v193 = sub_3007030((__int64)&v242);
        v87 = sub_30070B0((__int64)&v242);
        v37 = v226;
        v38 = v217;
        if ( v87 )
          goto LABEL_45;
        if ( !v193 )
          goto LABEL_11;
      }
      v214 = v37[16];
      goto LABEL_12;
    }
    v61 = (unsigned int *)sub_33E5110(v33, v213, v16, 262, 0);
    v64 = (_QWORD *)a1[1];
    v65 = v61;
    v66 = v62;
    *((_QWORD *)&v174 + 1) = 2;
    *(_QWORD *)&v174 = &v240;
    if ( *(_DWORD *)(v188 + 24) == 56 )
    {
      v231 = sub_3411630(v64, 68, (__int64)&v234, v61, v62, v63, v174);
      *(_QWORD *)a3 = v231;
      v248 = v231;
      *(_DWORD *)(a3 + 8) = v91;
      v92 = (_QWORD *)a1[1];
      *((_QWORD *)&v167 + 1) = 3;
      *(_QWORD *)&v167 = &v246;
      LODWORD(v249) = 1;
      *(_QWORD *)a4 = sub_3411630(v92, 70, (__int64)&v234, v65, v66, v93, v167);
    }
    else
    {
      v230 = sub_3411630(v64, 69, (__int64)&v234, v61, v62, v63, v174);
      *(_QWORD *)a3 = v230;
      v248 = v230;
      *(_DWORD *)(a3 + 8) = v67;
      v68 = (_QWORD *)a1[1];
      *((_QWORD *)&v166 + 1) = 3;
      *(_QWORD *)&v166 = &v246;
      LODWORD(v249) = 1;
      *(_QWORD *)a4 = sub_3411630(v68, 71, (__int64)&v234, v65, v66, v69, v166);
    }
    *(_DWORD *)(a4 + 8) = v70;
  }
LABEL_17:
  if ( v234 )
    sub_B91220((__int64)&v234, v234);
}
