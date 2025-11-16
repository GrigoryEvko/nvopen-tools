// Function: sub_330BE50
// Address: 0x330be50
//
__int64 __fastcall sub_330BE50(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r12
  const __m128i *v8; // rax
  __int16 *v9; // rdx
  __int64 v10; // rsi
  __m128i v11; // xmm0
  __int16 v12; // ax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  int v16; // r9d
  int v18; // eax
  __int64 v19; // rsi
  __int128 *v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rcx
  __int64 *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r10
  __int64 v28; // r11
  __int64 v29; // rdx
  __int64 v30; // rax
  __int32 v31; // r9d
  unsigned __int64 v32; // r8
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int); // rax
  __int64 (*v34)(); // rax
  unsigned __int16 *v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rsi
  char v38; // al
  int v39; // eax
  __int128 v40; // rax
  int v41; // r9d
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r15
  __int64 v45; // r8
  __int64 v46; // r9
  const __m128i *v47; // roff
  __int64 v48; // rdi
  unsigned int v49; // eax
  int v50; // r9d
  __int64 v51; // rdi
  __int64 v52; // r12
  __int128 v53; // rax
  int v54; // r9d
  __int64 v55; // rax
  unsigned int v56; // esi
  _QWORD *v57; // rax
  _DWORD *v58; // r9
  __m128i v59; // xmm3
  __int64 v60; // rax
  __int64 *v61; // rdi
  __int64 v62; // r9
  __int64 v63; // r15
  __int128 v64; // rax
  int v65; // r9d
  int v66; // r9d
  __int64 v67; // rax
  __int32 v68; // edx
  __int64 v69; // rdi
  __int64 v70; // rax
  int v71; // r9d
  __int64 v72; // rax
  __int64 v73; // r15
  __m128i v74; // rax
  __int64 v75; // r12
  __int128 v76; // rax
  int v77; // r9d
  __int64 v78; // rax
  __int64 v79; // r15
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 *v82; // rdx
  __int64 v83; // r15
  __m128i v84; // xmm5
  unsigned int v85; // ecx
  __int64 v86; // rcx
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // rdx
  __int128 v90; // rdi
  __int64 v91; // rdx
  int v92; // edx
  __int64 v93; // rax
  __int128 v94; // kr20_16
  unsigned __int16 v95; // dx
  unsigned __int16 *v96; // rdx
  int v97; // eax
  __int64 v98; // r15
  __m128i v99; // rax
  __int64 v100; // rdi
  __int64 v101; // rdx
  __int128 v102; // rax
  __int32 v103; // edx
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 *v107; // rdi
  __int64 (__fastcall *v108)(__int64, __int64, __int64, __int64, __int64); // r10
  __int64 v109; // rdx
  __int64 v110; // rsi
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // rax
  unsigned __int64 v114; // rdi
  __int64 v115; // rsi
  __int128 v116; // rax
  int v117; // r9d
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // r8
  __int64 v123; // rdx
  __m128i v124; // rax
  __int64 v125; // r12
  __int128 v126; // rax
  int v127; // r9d
  bool v128; // zf
  __int64 v129; // rdi
  __int64 v130; // rdx
  __int64 v131; // rdi
  unsigned __int64 v132; // r10
  __int64 v133; // r11
  __int64 v134; // r9
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 *v137; // rcx
  __int128 v138; // rax
  __m128i v139; // rax
  unsigned int v140; // eax
  __int128 v141; // rax
  int v142; // r9d
  __m128i v143; // rax
  char v144; // al
  char v145; // r15
  int v146; // r9d
  __int64 v147; // r10
  __int64 v148; // rbx
  __int64 v149; // r8
  int v150; // r15d
  __int32 v151; // eax
  int v152; // ecx
  _QWORD *v153; // r15
  __m128i v154; // rax
  unsigned __int16 *v155; // rax
  __m128i v156; // rax
  __int64 v157; // rax
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rdx
  __int64 v161; // rdx
  __int64 v162; // rax
  int v163; // edx
  bool v164; // al
  int v165; // eax
  char v166; // al
  __int16 v167; // ax
  __int64 v168; // rax
  __int64 i; // rax
  __int64 v170; // rcx
  char v171; // al
  __int16 v172; // ax
  __int64 v173; // rax
  __int64 j; // rax
  __int64 v175; // rcx
  __int64 v176; // rdi
  __m128i v177; // rax
  __int64 v178; // rdi
  __int128 v179; // rax
  __int64 v180; // rcx
  __int64 v181; // rdx
  char v182; // r8
  __int64 v183; // rax
  __int64 v184; // rdx
  __int64 v185; // rdi
  __int64 v186; // rdx
  char v187; // al
  int v188; // eax
  __int64 v189; // rdx
  __int64 v190; // rdi
  __int128 v191; // rax
  int v192; // r9d
  __int64 v193; // rax
  __int64 v194; // rdx
  int v195; // r15d
  __int128 v196; // rax
  int v197; // r9d
  __int128 v198; // [rsp+0h] [rbp-1D0h]
  _DWORD *v199; // [rsp+10h] [rbp-1C0h]
  __int64 (__fastcall *v200)(_DWORD *, __int64, __int64, _QWORD, _QWORD); // [rsp+18h] [rbp-1B8h]
  _DWORD *v201; // [rsp+18h] [rbp-1B8h]
  __int64 v202; // [rsp+20h] [rbp-1B0h]
  __int64 v203; // [rsp+20h] [rbp-1B0h]
  __int64 v204; // [rsp+28h] [rbp-1A8h]
  char v205; // [rsp+28h] [rbp-1A8h]
  unsigned int v206; // [rsp+30h] [rbp-1A0h]
  __int64 (__fastcall *v207)(__int64, __int64, __int64, __int64, __int64); // [rsp+30h] [rbp-1A0h]
  __int64 v208; // [rsp+38h] [rbp-198h]
  __int64 v209; // [rsp+38h] [rbp-198h]
  __int128 v210; // [rsp+40h] [rbp-190h]
  __int64 v211; // [rsp+40h] [rbp-190h]
  unsigned __int64 v212; // [rsp+40h] [rbp-190h]
  __int64 v213; // [rsp+40h] [rbp-190h]
  __int64 v214; // [rsp+40h] [rbp-190h]
  __int64 v215; // [rsp+40h] [rbp-190h]
  int v216; // [rsp+40h] [rbp-190h]
  __int64 v217; // [rsp+48h] [rbp-188h]
  int v218; // [rsp+50h] [rbp-180h]
  __int64 v219; // [rsp+50h] [rbp-180h]
  __int128 v220; // [rsp+50h] [rbp-180h]
  int v221; // [rsp+50h] [rbp-180h]
  unsigned int v222; // [rsp+60h] [rbp-170h]
  __int128 v223; // [rsp+60h] [rbp-170h]
  int v224; // [rsp+60h] [rbp-170h]
  __m128i v225; // [rsp+70h] [rbp-160h] BYREF
  __int64 v226; // [rsp+80h] [rbp-150h]
  unsigned __int64 v227; // [rsp+88h] [rbp-148h]
  __int128 v228; // [rsp+90h] [rbp-140h]
  __m128i v229; // [rsp+A0h] [rbp-130h]
  __m128i v230; // [rsp+B0h] [rbp-120h] BYREF
  unsigned int v231; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v232; // [rsp+C8h] [rbp-108h]
  __int64 v233; // [rsp+D0h] [rbp-100h] BYREF
  int v234; // [rsp+D8h] [rbp-F8h]
  __m128i v235; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v236; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v237; // [rsp+F8h] [rbp-D8h]
  __int128 v238; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v239; // [rsp+110h] [rbp-C0h] BYREF
  int v240; // [rsp+118h] [rbp-B8h]
  __int64 v241; // [rsp+120h] [rbp-B0h]
  __int64 v242; // [rsp+128h] [rbp-A8h]
  __int64 v243; // [rsp+130h] [rbp-A0h]
  __int64 v244; // [rsp+138h] [rbp-98h]
  __int16 v245; // [rsp+140h] [rbp-90h] BYREF
  __int64 v246; // [rsp+148h] [rbp-88h]
  __int64 v247; // [rsp+150h] [rbp-80h] BYREF
  __int64 v248; // [rsp+158h] [rbp-78h]
  __m128i v249; // [rsp+160h] [rbp-70h] BYREF
  _QWORD *v250; // [rsp+170h] [rbp-60h] BYREF
  __int64 v251; // [rsp+178h] [rbp-58h]
  _QWORD v252[10]; // [rsp+180h] [rbp-50h] BYREF

  v7 = (__int64)a2;
  v8 = (const __m128i *)a2[5];
  v9 = (__int16 *)a2[6];
  v10 = a2[10];
  v11 = _mm_loadu_si128(v8);
  v12 = *v9;
  v13 = *((_QWORD *)v9 + 1);
  v233 = v10;
  LOWORD(v231) = v12;
  v232 = v13;
  v230 = v11;
  if ( v10 )
  {
    sub_B96E90((__int64)&v233, v10, 1);
    v12 = v231;
  }
  v234 = *(_DWORD *)(v7 + 72);
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
      goto LABEL_5;
  }
  else if ( !sub_30070B0((__int64)&v231) )
  {
    goto LABEL_5;
  }
  v15 = sub_32730B0(a1, v7, (__int64)&v233, a4, a5);
  if ( v15 )
    goto LABEL_7;
LABEL_5:
  v14 = v230.m128i_i64[0];
  if ( *(_DWORD *)(v230.m128i_i64[0] + 24) == 51 )
  {
    v7 = sub_3400BD0(*a1, 0, (unsigned int)&v233, v231, v232, 0, 0);
    goto LABEL_8;
  }
  v15 = sub_32788C0(v7, (int)&v233, a1[1], *a1, *((_BYTE *)a1 + 34), a6);
  if ( v15 )
  {
LABEL_7:
    v7 = v15;
    goto LABEL_8;
  }
  v18 = *(_DWORD *)(v230.m128i_i64[0] + 24);
  if ( (v18 & 0xFFFFFFFD) == 0xD5 )
  {
    v22 = sub_33FAF80(*a1, 213, (unsigned int)&v233, v231, v232, v16, *(_OWORD *)*(_QWORD *)(v230.m128i_i64[0] + 40));
    goto LABEL_22;
  }
  if ( (unsigned int)(v18 - 223) <= 1 )
  {
    v19 = *(_QWORD *)(v7 + 80);
    v20 = *(__int128 **)(v230.m128i_i64[0] + 40);
    v21 = *a1;
    v250 = (_QWORD *)v19;
    if ( v19 )
      sub_B96E90((__int64)&v250, v19, 1);
    LODWORD(v251) = *(_DWORD *)(v7 + 72);
    v7 = sub_33FAF80(v21, 224, (unsigned int)&v250, v231, v232, v16, *v20);
    if ( v250 )
      sub_B91220((__int64)&v250, (__int64)v250);
    goto LABEL_8;
  }
  if ( v18 == 222 )
  {
    v25 = *(__int64 **)(v230.m128i_i64[0] + 40);
    v26 = *v25;
    v27 = *v25;
    v28 = v25[1];
    v29 = *((unsigned int *)v25 + 2);
    v30 = v25[5];
    v31 = *(unsigned __int16 *)(v30 + 96);
    v32 = *(_QWORD *)(v30 + 104);
    if ( *(_DWORD *)(v26 + 24) == 216 )
      goto LABEL_243;
    v23 = a1[1];
    v33 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v23 + 1408LL);
    if ( v33 == sub_2FE3A30 )
    {
      v34 = *(__int64 (**)())(*(_QWORD *)v23 + 1392LL);
      v35 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16 * v29);
      v36 = *((_QWORD *)v35 + 1);
      v37 = *v35;
      if ( v34 == sub_2FE3480 )
        goto LABEL_26;
      v225.m128i_i64[0] = v27;
      v225.m128i_i64[1] = v28;
      LODWORD(v227) = v31;
      *(_QWORD *)&v228 = v32;
      v38 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v34)(v23, v37, v36, (unsigned __int16)v31);
      LODWORD(v32) = v228;
      v31 = v227;
      v28 = v225.m128i_i64[1];
      v27 = v225.m128i_i64[0];
    }
    else
    {
      v225.m128i_i32[0] = v31;
      v227 = v32;
      *(_QWORD *)&v228 = v27;
      *((_QWORD *)&v228 + 1) = v28;
      v229.m128i_i64[0] = (unsigned __int16)v31;
      v38 = v33(v23, v27, v28);
      v31 = v225.m128i_i32[0];
      LODWORD(v32) = v227;
      v28 = *((_QWORD *)&v228 + 1);
      v27 = v228;
    }
    if ( v38 )
    {
LABEL_243:
      if ( !*((_BYTE *)a1 + 34) || (_WORD)v31 && *(_QWORD *)(a1[1] + 8LL * (unsigned __int16)v31 + 112) )
      {
        HIWORD(v39) = v229.m128i_i16[1];
        *((_QWORD *)&v198 + 1) = v28;
        *(_QWORD *)&v198 = v27;
        LOWORD(v39) = v31;
        *(_QWORD *)&v40 = sub_33FAF80(*a1, 216, (unsigned int)&v233, v39, v32, v31, v198);
        v7 = sub_33FAF80(*a1, 213, (unsigned int)&v233, v231, v232, v41, v40);
        goto LABEL_8;
      }
    }
    v18 = *(_DWORD *)(v230.m128i_i64[0] + 24);
  }
  if ( v18 != 216 )
  {
    v23 = a1[1];
    goto LABEL_26;
  }
  v42 = sub_32B3F40(a1, v230.m128i_i64[0]);
  if ( v42 )
  {
    if ( v230.m128i_i64[0] != v42 )
    {
      v44 = **(_QWORD **)(v230.m128i_i64[0] + 40);
      v251 = v43;
      v250 = (_QWORD *)v42;
      sub_32EB790((__int64)a1, v230.m128i_i64[0], (__int64 *)&v250, 1, 1);
      sub_32B3E80((__int64)a1, v44, 1, 0, v45, v46);
    }
    goto LABEL_8;
  }
  v47 = *(const __m128i **)(v230.m128i_i64[0] + 40);
  v48 = v47->m128i_i64[0];
  v49 = v47->m128i_u32[2];
  v229 = _mm_loadu_si128(v47);
  v227 = v48;
  v225.m128i_i32[0] = v49;
  *(_QWORD *)&v228 = sub_3263630(v48, v49);
  v218 = sub_3263630(v230.m128i_i64[0], v230.m128i_u32[2]);
  v222 = sub_32844A0((unsigned __int16 *)&v231, v230.m128i_u32[2]);
  if ( (*(_BYTE *)(v230.m128i_i64[0] + 28) & 2) != 0
    || (unsigned int)sub_33D4D80(*a1, v229.m128i_i64[0], v229.m128i_i64[1], 0) > (int)v228 - v218 )
  {
    if ( (_DWORD)v228 == v222 )
    {
      v7 = v229.m128i_i64[0];
    }
    else
    {
      v51 = *a1;
      if ( (unsigned int)v228 < v222 )
        v7 = sub_33FAF80(v51, 213, (unsigned int)&v233, v231, v232, v50, *(_OWORD *)&v229);
      else
        v7 = sub_33FA050(
               v51,
               216,
               (unsigned int)&v233,
               v231,
               v232,
               *(_DWORD *)(v230.m128i_i64[0] + 28) & 1 | 2u,
               v229.m128i_i64[0],
               v229.m128i_i64[1]);
    }
    goto LABEL_8;
  }
  if ( !*((_BYTE *)a1 + 33)
    || (v23 = a1[1], sub_328D6E0(v23, 0xDEu, *(_WORD *)(*(_QWORD *)(v230.m128i_i64[0] + 48) + 16LL * v230.m128i_u32[2]))) )
  {
    v52 = *a1;
    if ( (unsigned int)v228 < v222 )
    {
      sub_3285E70((__int64)&v250, v230.m128i_i64[0]);
      v67 = sub_33FAF80(v52, 215, (unsigned int)&v250, v231, v232, v66, *(_OWORD *)&v229);
    }
    else
    {
      if ( (unsigned int)v228 <= v222 )
      {
LABEL_54:
        *(_QWORD *)&v53 = sub_33F7D60(
                            v52,
                            *(unsigned __int16 *)(*(_QWORD *)(v230.m128i_i64[0] + 48) + 16LL * v230.m128i_u32[2]),
                            *(_QWORD *)(*(_QWORD *)(v230.m128i_i64[0] + 48) + 16LL * v230.m128i_u32[2] + 8));
        v229.m128i_i64[0] = v227;
        v229.m128i_i64[1] = v225.m128i_u32[0] | v229.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v55 = sub_3406EB0(v52, 222, (unsigned int)&v233, v231, v232, v54, __PAIR128__(v229.m128i_u64[1], v227), v53);
LABEL_55:
        v7 = v55;
        goto LABEL_8;
      }
      sub_3285E70((__int64)&v250, v230.m128i_i64[0]);
      v67 = sub_33FAF80(v52, 216, (unsigned int)&v250, v231, v232, v71, *(_OWORD *)&v229);
    }
    v227 = v67;
    v225.m128i_i32[0] = v68;
    sub_9C6650(&v250);
    v52 = *a1;
    goto LABEL_54;
  }
LABEL_26:
  v15 = sub_330B820(
          *a1,
          a1,
          v23,
          v231,
          v232,
          *((_BYTE *)a1 + 33),
          v7,
          v230.m128i_i64[0],
          v230.m128i_i64[1],
          2,
          0xD5u,
          0);
  if ( v15 )
    goto LABEL_7;
  v15 = sub_326BA10(*a1, a1[1], v231, v232, *((_BYTE *)a1 + 33), v7, v230.m128i_i64[0], v230.m128i_i32[2], 2, 213);
  if ( v15 )
    goto LABEL_7;
  v24 = sub_3304AC0(a1, v7);
  if ( v24 )
    goto LABEL_29;
  v24 = sub_32FE700(
          *a1,
          (__int64)a1,
          a1[1],
          v231,
          v232,
          *((_BYTE *)a1 + 33),
          v7,
          v230.m128i_i64[0],
          v230.m128i_i64[1],
          2);
  if ( v24 )
    goto LABEL_29;
  v24 = sub_3271140(*a1, a1[1], v231, v232, v230.m128i_i64[0], 2);
  if ( v24 )
    goto LABEL_29;
  v56 = *(_DWORD *)(v230.m128i_i64[0] + 24);
  if ( v56 - 186 <= 2 )
  {
    v57 = *(_QWORD **)(v230.m128i_i64[0] + 40);
    if ( *(_DWORD *)(*v57 + 24LL) == 298 && *(_DWORD *)(v57[5] + 24LL) == 11 )
    {
      v128 = *((_BYTE *)a1 + 33) == 0;
      v225.m128i_i64[0] = *v57;
      if ( v128 )
      {
        v129 = a1[1];
        LODWORD(v227) = v231;
        v229.m128i_i64[0] = v232;
        *(_QWORD *)&v228 = v129;
        if ( sub_328D6E0(v129, v56, v231) )
        {
          v130 = *(unsigned __int16 *)(v225.m128i_i64[0] + 96);
          if ( (_WORD)v130 )
          {
            if ( (_WORD)v231 )
            {
              v131 = v228;
              if ( (*(_BYTE *)(v228 + 2 * (v130 + 274LL * (unsigned __int16)v231 + 71704) + 7) & 0xF) == 0
                && ((*(_BYTE *)(v225.m128i_i64[0] + 33) ^ 0xC) & 0xC) != 0
                && (*(_WORD *)(v225.m128i_i64[0] + 32) & 0x380) == 0 )
              {
                v250 = v252;
                *(_QWORD *)&v228 = v225.m128i_i64[0];
                v251 = 0x400000000LL;
                if ( (unsigned __int8)sub_32611B0(
                                        v227,
                                        v229.m128i_i64[0],
                                        v230.m128i_i64[0],
                                        **(_QWORD **)(v230.m128i_i64[0] + 40),
                                        *(_DWORD *)(*(_QWORD *)(v230.m128i_i64[0] + 40) + 8LL),
                                        213,
                                        (__int64)&v250,
                                        v131) )
                {
                  v132 = v228;
                  v133 = *a1;
                  v229.m128i_i64[0] = (__int64)&v249;
                  v134 = *(_QWORD *)(v228 + 112);
                  v135 = *(unsigned __int16 *)(v228 + 96);
                  v136 = *(_QWORD *)(v228 + 104);
                  v137 = *(__int64 **)(v228 + 40);
                  v249.m128i_i64[0] = *(_QWORD *)(v228 + 80);
                  if ( v249.m128i_i64[0] )
                  {
                    v211 = v135;
                    v217 = v136;
                    v224 = v134;
                    v225.m128i_i64[0] = (__int64)v137;
                    v227 = v228;
                    *(_QWORD *)&v228 = v133;
                    sub_325F5D0(v249.m128i_i64);
                    v135 = v211;
                    v136 = v217;
                    LODWORD(v134) = v224;
                    v137 = (__int64 *)v225.m128i_i64[0];
                    v132 = v227;
                    LODWORD(v133) = v228;
                  }
                  v249.m128i_i32[2] = *(_DWORD *)(v132 + 72);
                  v212 = v132;
                  *(_QWORD *)&v138 = sub_33F1B30(
                                       v133,
                                       2,
                                       v229.m128i_i32[0],
                                       v231,
                                       v232,
                                       v134,
                                       *v137,
                                       v137[1],
                                       v137[5],
                                       v137[6],
                                       v135,
                                       v136);
                  v228 = v138;
                  v227 = v138;
                  sub_9C6650(v229.m128i_i64[0]);
                  v225.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v230.m128i_i64[0] + 40) + 40LL) + 96LL) + 24LL;
                  v139.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v231);
                  v249 = v139;
                  v140 = sub_CA1930(v229.m128i_i64[0]);
                  sub_C44830((__int64)&v247, v225.m128i_i64[0], v140);
                  v225.m128i_i64[0] = *a1;
                  *(_QWORD *)&v141 = sub_34007B0(
                                       v225.m128i_i32[0],
                                       (unsigned int)&v247,
                                       (unsigned int)&v233,
                                       v231,
                                       v232,
                                       0,
                                       0);
                  v143.m128i_i64[0] = sub_3406EB0(
                                        v225.m128i_i32[0],
                                        *(_DWORD *)(v230.m128i_i64[0] + 24),
                                        (unsigned int)&v233,
                                        v231,
                                        v232,
                                        v142,
                                        v228,
                                        v141);
                  v225 = v143;
                  sub_3304760(
                    a1,
                    (__int64)&v250,
                    **(_QWORD **)(v230.m128i_i64[0] + 40),
                    *(_DWORD *)(*(_QWORD *)(v230.m128i_i64[0] + 40) + 8LL),
                    v228,
                    *((__int64 *)&v228 + 1),
                    213);
                  v144 = sub_3286E00(&v230);
                  v249.m128i_i32[2] = 0;
                  LOBYTE(v226) = v144;
                  v249.m128i_i64[0] = v212;
                  v145 = sub_3286E00(v229.m128i_i64[0]);
                  v249 = _mm_load_si128(&v225);
                  sub_32EB790((__int64)a1, v7, (__int64 *)v229.m128i_i64[0], 1, 1);
                  v146 = (unsigned __int8)v226;
                  v147 = v212;
                  if ( !(_BYTE)v226 )
                  {
                    v155 = (unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16LL * v230.m128i_u32[2]);
                    v156.m128i_i64[0] = sub_33FAF80(
                                          *a1,
                                          216,
                                          (unsigned int)&v233,
                                          *v155,
                                          *((_QWORD *)v155 + 1),
                                          0,
                                          *(_OWORD *)&v225);
                    v249 = v156;
                    sub_32EB790((__int64)a1, v14, (__int64 *)v229.m128i_i64[0], 1, 1);
                    v147 = v212;
                  }
                  v148 = *a1;
                  if ( v145 )
                  {
                    sub_34161C0(*a1, v147, 1, v227, 1);
                  }
                  else
                  {
                    v149 = *(_QWORD *)(*(_QWORD *)(v147 + 48) + 8LL);
                    v150 = **(unsigned __int16 **)(v147 + 48);
                    v249.m128i_i64[0] = *(_QWORD *)(v147 + 80);
                    if ( v249.m128i_i64[0] )
                    {
                      v213 = v147;
                      v225.m128i_i64[0] = v149;
                      sub_325F5D0((__int64 *)v229.m128i_i64[0]);
                      v147 = v213;
                      LODWORD(v149) = v225.m128i_i32[0];
                    }
                    v151 = *(_DWORD *)(v147 + 72);
                    v152 = v150;
                    v153 = (_QWORD *)v229.m128i_i64[0];
                    v225.m128i_i64[0] = v147;
                    v249.m128i_i32[2] = v151;
                    v154.m128i_i64[0] = sub_33FAF80(v148, 216, v229.m128i_i32[0], v152, v149, v146, v228);
                    v229 = v154;
                    sub_9C6650(v153);
                    sub_32EFDE0((__int64)a1, v225.m128i_i64[0], v229.m128i_i64[0], v229.m128i_i64[1], v227, 1, 1);
                  }
                  sub_969240(&v247);
                  if ( v250 != v252 )
                    _libc_free((unsigned __int64)v250);
                  goto LABEL_8;
                }
                if ( v250 != v252 )
                  _libc_free((unsigned __int64)v250);
              }
            }
          }
        }
      }
    }
  }
  v24 = sub_32815D0(v7, *a1, *((_BYTE *)a1 + 33));
  v219 = v24;
  if ( v24 )
  {
LABEL_29:
    v7 = v24;
    goto LABEL_8;
  }
  v59 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
  v235 = v59;
  v60 = v59.m128i_i64[0];
  if ( *(_DWORD *)(v59.m128i_i64[0] + 24) == 208 )
  {
    v82 = *(__int64 **)(v59.m128i_i64[0] + 40);
    *((_QWORD *)&v90 + 1) = *(_QWORD *)(v7 + 80);
    v83 = *v82;
    v84 = _mm_loadu_si128((const __m128i *)(v82 + 5));
    v208 = v82[5];
    v85 = *((_DWORD *)v82 + 12);
    v225 = _mm_loadu_si128((const __m128i *)v82);
    v206 = v85;
    v86 = v82[10];
    v87 = *((unsigned int *)v82 + 2);
    v223 = (__int128)v84;
    LODWORD(v227) = *(_DWORD *)(v86 + 96);
    v88 = *(_QWORD *)(v7 + 48);
    v226 = 16 * v87;
    v89 = *(_QWORD *)(v83 + 48) + 16 * v87;
    LOWORD(v90) = *(_WORD *)v88;
    v237 = *(_QWORD *)(v88 + 8);
    LOWORD(v88) = *(_WORD *)v89;
    v91 = *(_QWORD *)(v89 + 8);
    v229.m128i_i16[0] = v90;
    LOWORD(v236) = v90;
    LOWORD(v238) = v88;
    *((_QWORD *)&v238 + 1) = v91;
    v239 = *((_QWORD *)&v90 + 1);
    if ( *((_QWORD *)&v90 + 1) )
    {
      sub_B96E90((__int64)&v239, *((__int64 *)&v90 + 1), 1);
      v60 = v235.m128i_i64[0];
    }
    v240 = *(_DWORD *)(v7 + 72);
    v92 = *(_DWORD *)(v60 + 28);
    v93 = *a1;
    LODWORD(v251) = v92;
    v250 = (_QWORD *)v93;
    v252[0] = *(_QWORD *)(v93 + 1024);
    *(_QWORD *)(v93 + 1024) = &v250;
    if ( v229.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v229.m128i_i16[0] - 17) > 0xD3u )
        goto LABEL_110;
    }
    else if ( !sub_30070B0((__int64)&v236) )
    {
      goto LABEL_110;
    }
    if ( *((_BYTE *)a1 + 33) )
      goto LABEL_110;
    v58 = (_DWORD *)a1[1];
    v94 = v238;
    v249 = (__m128i)v238;
    if ( (_WORD)v238 )
    {
      v95 = v238 - 17;
      if ( (unsigned __int16)(v238 - 10) > 6u && (unsigned __int16)(v238 - 126) > 0x31u )
      {
        if ( v95 > 0xD3u )
        {
          if ( (unsigned __int16)(v238 - 208) > 0x14u )
          {
LABEL_108:
            LODWORD(v228) = v58[15];
            goto LABEL_109;
          }
          goto LABEL_177;
        }
LABEL_178:
        LODWORD(v228) = v58[17];
LABEL_109:
        if ( (_DWORD)v228 != 2 )
          goto LABEL_110;
        v199 = v58;
        v200 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v58 + 528LL);
        v202 = *(_QWORD *)(*a1 + 64LL);
        v157 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
        v158 = v200(v199, v157, v202, v94, *((_QWORD *)&v94 + 1));
        v249.m128i_i16[0] = v158;
        *((_QWORD *)&v90 + 1) = v158;
        v159 = v235.m128i_i64[0];
        v249.m128i_i64[1] = v160;
        v214 = v160;
        v161 = *(_QWORD *)(v235.m128i_i64[0] + 48) + 16LL * v235.m128i_u32[2];
        if ( WORD4(v90) == *(_WORD *)v161 )
        {
          if ( WORD4(v90) || *(_QWORD *)(v161 + 8) == v214 )
            goto LABEL_170;
        }
        else if ( WORD4(v90) )
        {
          if ( WORD4(v90) == 1 || (unsigned __int16)(WORD4(v90) - 504) <= 7u )
            goto LABEL_240;
          v180 = *(_QWORD *)&byte_444C4A0[16 * WORD4(v90) - 16];
          v182 = byte_444C4A0[16 * WORD4(v90) - 8];
          goto LABEL_222;
        }
        v243 = sub_3007260((__int64)&v249);
        v180 = v243;
        v244 = v181;
        v182 = v181;
LABEL_222:
        if ( v229.m128i_i16[0] )
        {
          if ( v229.m128i_i16[0] == 1 || (unsigned __int16)(v229.m128i_i16[0] - 504) <= 7u )
            goto LABEL_240;
          v186 = *(_QWORD *)&byte_444C4A0[16 * v229.m128i_u16[0] - 16];
          v187 = byte_444C4A0[16 * v229.m128i_u16[0] - 8];
        }
        else
        {
          v203 = v180;
          v205 = v182;
          v183 = sub_3007260((__int64)&v236);
          v180 = v203;
          v185 = v184;
          v241 = v183;
          v186 = v183;
          v182 = v205;
          v242 = v185;
          v187 = v185;
        }
        if ( v180 == v186 && v187 == v182 )
        {
          v195 = v236;
          *(_QWORD *)&v228 = *a1;
          v229.m128i_i64[0] = v237;
          *(_QWORD *)&v196 = sub_33ED040(v228, (unsigned int)v227);
          *(_QWORD *)&v228 = sub_340F900(
                               v228,
                               208,
                               (unsigned int)&v239,
                               v195,
                               v229.m128i_i32[0],
                               v197,
                               *(_OWORD *)&v225,
                               v223,
                               v196);
          goto LABEL_119;
        }
        v188 = sub_327FDF0((unsigned __int16 *)&v238, *((__int64 *)&v90 + 1));
        if ( WORD4(v90) == (_WORD)v188 && (WORD4(v90) || v214 == v189) )
        {
          v190 = *a1;
          v221 = v188;
          *(_QWORD *)&v228 = v189;
          v229.m128i_i64[0] = v190;
          *(_QWORD *)&v191 = sub_33ED040(v190, (unsigned int)v227);
          v193 = sub_340F900(v190, 208, (unsigned int)&v239, v221, v228, v192, *(_OWORD *)&v225, v223, v191);
          *(_QWORD *)&v228 = sub_33FB160(*a1, v193, v194, &v239, (unsigned int)v236, v237);
          goto LABEL_119;
        }
        v159 = v235.m128i_i64[0];
LABEL_170:
        v162 = *(_QWORD *)(v159 + 56);
        v163 = 1;
        while ( v162 )
        {
          if ( v235.m128i_i32[2] == *(_DWORD *)(v162 + 8) )
          {
            if ( !v163 )
              goto LABEL_110;
            v163 = 0;
          }
          v162 = *(_QWORD *)(v162 + 32);
        }
        if ( v163 != 1 )
        {
          *((_QWORD *)&v90 + 1) = 208;
          v215 = a1[1];
          if ( (unsigned __int8)sub_328A020(v215, 0xD0u, v236, v237, 0) )
          {
            *((_QWORD *)&v90 + 1) = 208;
            if ( !(unsigned __int8)sub_328A020(v215, 0xD0u, v249.m128i_u16[0], v249.m128i_i64[1], 0) )
            {
              *((_QWORD *)&v90 + 1) = v225.m128i_i64[1];
              v165 = 3;
              if ( (unsigned int)(v227 - 18) <= 3 )
                v165 = v228;
              v216 = 214 - ((unsigned int)(v227 - 18) < 4);
              LODWORD(v228) = v165;
              v166 = sub_326A930(v225.m128i_i64[0], v225.m128i_u32[2], 1u);
              LODWORD(v58) = v216;
              if ( !v166 )
              {
                if ( *(_DWORD *)(v83 + 24) != 298 )
                  goto LABEL_110;
                if ( (*(_BYTE *)(v83 + 33) & 0xC) != 0 )
                  goto LABEL_110;
                v167 = *(_WORD *)(v83 + 32);
                if ( (v167 & 0x380) != 0 )
                  goto LABEL_110;
                if ( (*(_BYTE *)(*(_QWORD *)(v83 + 112) + 37LL) & 0xF) != 0 )
                  goto LABEL_110;
                if ( (v167 & 8) != 0 )
                  goto LABEL_110;
                *((_QWORD *)&v90 + 1) = a1[1];
                v168 = *(unsigned __int16 *)(*(_QWORD *)(v83 + 48) + v226);
                if ( !v229.m128i_i16[0]
                  || !(_WORD)v168
                  || (((int)*(unsigned __int16 *)(*((_QWORD *)&v90 + 1)
                                                + 2 * (v168 + 274LL * v229.m128i_u16[0] + 71704)
                                                + 6) >> (4 * v228))
                    & 0xF) != 0 )
                {
                  goto LABEL_110;
                }
                for ( i = *(_QWORD *)(v83 + 56); i; i = *(_QWORD *)(i + 32) )
                {
                  v170 = *(_QWORD *)(i + 16);
                  if ( !*(_DWORD *)(i + 8)
                    && v170 != v235.m128i_i64[0]
                    && (v216 != *(_DWORD *)(v170 + 24) || v229.m128i_i16[0] != **(_WORD **)(v170 + 48)) )
                  {
                    goto LABEL_110;
                  }
                }
              }
              v90 = v223;
              v171 = sub_326A930(v90, DWORD2(v90), 1u);
              LODWORD(v58) = 214 - ((unsigned int)(v227 - 18) < 4);
              if ( v171 )
              {
LABEL_219:
                v176 = *a1;
                LODWORD(v228) = 214 - ((unsigned int)(v227 - 18) < 4);
                v177.m128i_i64[0] = sub_33FAF80(v176, v216, (unsigned int)&v239, v236, v237, v216, *(_OWORD *)&v225);
                v178 = *a1;
                v229 = v177;
                *(_QWORD *)&v179 = sub_33FAF80(v178, v228, (unsigned int)&v239, v236, v237, v228, v223);
                *(_QWORD *)&v228 = sub_32889F0(
                                     *a1,
                                     (int)&v239,
                                     (unsigned int)v236,
                                     v237,
                                     v229.m128i_i64[0],
                                     v229.m128i_i64[1],
                                     v179,
                                     (unsigned int)v227,
                                     0);
                goto LABEL_119;
              }
              if ( *(_DWORD *)(v208 + 24) == 298 && (*(_BYTE *)(v208 + 33) & 0xC) == 0 )
              {
                v172 = *(_WORD *)(v208 + 32);
                if ( (v172 & 0x380) == 0 && (*(_BYTE *)(*(_QWORD *)(v208 + 112) + 37LL) & 0xF) == 0 && (v172 & 8) == 0 )
                {
                  *((_QWORD *)&v90 + 1) = v229.m128i_u16[0];
                  v173 = *(unsigned __int16 *)(*(_QWORD *)(v208 + 48) + 16LL * v206);
                  if ( v229.m128i_i16[0] )
                  {
                    if ( (_WORD)v173 )
                    {
                      *((_QWORD *)&v90 + 1) = 274LL * v229.m128i_u16[0];
                      if ( (((int)*(unsigned __int16 *)(a1[1] + 2 * (v173 + *((_QWORD *)&v90 + 1) + 71704) + 6) >> (4 * v228))
                          & 0xF) == 0 )
                      {
                        for ( j = *(_QWORD *)(v208 + 56); j; j = *(_QWORD *)(j + 32) )
                        {
                          v175 = *(_QWORD *)(j + 16);
                          if ( !*(_DWORD *)(j + 8)
                            && v235.m128i_i64[0] != v175
                            && (v216 != *(_DWORD *)(v175 + 24) || v229.m128i_i16[0] != **(_WORD **)(v175 + 48)) )
                          {
                            goto LABEL_110;
                          }
                        }
                        goto LABEL_219;
                      }
                    }
                  }
                }
              }
            }
          }
        }
LABEL_110:
        v96 = (unsigned __int16 *)(*(_QWORD *)(v235.m128i_i64[0] + 48) + 16LL * v235.m128i_u32[2]);
        v97 = *v96;
        v98 = *((_QWORD *)v96 + 1);
        v245 = v97;
        v246 = v98;
        if ( (_WORD)v97 )
        {
          if ( (unsigned __int16)(v97 - 17) > 0xD3u )
          {
            LOWORD(v247) = v97;
            v248 = v98;
LABEL_113:
            if ( (_WORD)v97 != 1 && (unsigned __int16)(v97 - 504) > 7u )
            {
              v99.m128i_i64[0] = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v97 - 16];
              goto LABEL_116;
            }
LABEL_240:
            BUG();
          }
          LOWORD(v97) = word_4456580[v97 - 1];
        }
        else
        {
          *(_QWORD *)&v228 = &v245;
          if ( !sub_30070B0((__int64)&v245) )
          {
            LOWORD(v247) = 0;
            v248 = v98;
            goto LABEL_138;
          }
          LOWORD(v97) = sub_3009970((__int64)&v245, *((__int64 *)&v90 + 1), v120, v121, v122);
          v219 = v123;
        }
        LOWORD(v247) = v97;
        v248 = v219;
        if ( (_WORD)v97 )
          goto LABEL_113;
LABEL_138:
        v99.m128i_i64[0] = sub_3007260((__int64)&v247);
        v249 = v99;
LABEL_116:
        v100 = *a1;
        if ( v99.m128i_i32[0] == 1 )
          *(_QWORD *)&v220 = sub_34015B0(v100, &v239, (unsigned int)v236, v237, 0, 0);
        else
          *(_QWORD *)&v220 = sub_3401740(v100, 1, (unsigned int)&v239, v236, v237, (_DWORD)v58, v238);
        *((_QWORD *)&v220 + 1) = v101;
        *(_QWORD *)&v102 = sub_3400BD0(*a1, 0, (unsigned int)&v239, v236, v237, 0, 0, v100);
        v210 = v102;
        *(_QWORD *)&v228 = sub_32C7250(
                             a1,
                             (__int64)&v239,
                             v225.m128i_i64[0],
                             v225.m128i_u64[1],
                             v223,
                             *((unsigned __int64 *)&v223 + 1),
                             v220,
                             v102,
                             v227,
                             1);
        if ( (_QWORD)v228 )
        {
LABEL_119:
          v104 = v239;
          v250[128] = v252[0];
          if ( v104 )
          {
            v229.m128i_i32[0] = v103;
            sub_B91220((__int64)&v239, v104);
          }
          if ( (_QWORD)v228 )
          {
            v7 = v228;
            goto LABEL_8;
          }
          goto LABEL_64;
        }
        if ( v229.m128i_i16[0] )
        {
          if ( (unsigned __int16)(v229.m128i_i16[0] - 17) <= 0xD3u )
            goto LABEL_125;
        }
        else if ( sub_30070B0((__int64)&v236) )
        {
          goto LABEL_125;
        }
        if ( !(unsigned __int8)sub_326C7E0(v235.m128i_i64, v236, v237, a1[1]) )
        {
          v105 = *a1;
          v106 = *(_QWORD *)a1[1];
          v204 = a1[1];
          v226 = *((_QWORD *)&v238 + 1);
          v107 = *(__int64 **)(v105 + 40);
          v108 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(v106 + 528);
          v109 = *(_QWORD *)(v105 + 64);
          v229.m128i_i64[0] = v238;
          v207 = v108;
          v209 = v109;
          v110 = sub_2E79000(v107);
          LOWORD(v247) = v207(v204, v110, v209, v229.m128i_i64[0], v226);
          v248 = v111;
          if ( sub_32844A0((unsigned __int16 *)&v247, v110) != 1 )
          {
            if ( !*((_BYTE *)a1 + 33)
              || ((v112 = a1[1], v113 = 1, (_WORD)v238 == 1)
               || (_WORD)v238 && (v113 = (unsigned __int16)v238, *(_QWORD *)(v112 + 8LL * (unsigned __int16)v238 + 112)))
              && !*(_BYTE *)(v112 + 500 * v113 + 6622) )
            {
              v114 = *a1;
              *(_QWORD *)&v228 = v248;
              v115 = (unsigned int)v227;
              v229.m128i_i64[0] = v247;
              v227 = v114;
              *(_QWORD *)&v116 = sub_33ED040(v114, v115);
              v118 = sub_340F900(
                       v114,
                       208,
                       (unsigned int)&v239,
                       v229.m128i_i32[0],
                       v228,
                       v117,
                       *(_OWORD *)&v225,
                       v223,
                       v116);
              *(_QWORD *)&v228 = sub_3288B20(*a1, (int)&v239, v236, v237, v118, v119, v220, v210, 0);
              goto LABEL_119;
            }
          }
        }
LABEL_125:
        v103 = 0;
        goto LABEL_119;
      }
      if ( v95 <= 0xD3u )
        goto LABEL_178;
    }
    else
    {
      v201 = v58;
      LOBYTE(v228) = sub_3007030((__int64)&v249);
      v164 = sub_30070B0((__int64)&v249);
      v58 = v201;
      if ( v164 )
        goto LABEL_178;
      if ( !(_BYTE)v228 )
        goto LABEL_108;
    }
LABEL_177:
    LODWORD(v228) = v58[16];
    goto LABEL_109;
  }
LABEL_64:
  v61 = (__int64 *)a1[1];
  v62 = *v61;
  v229.m128i_i64[0] = 16LL * v230.m128i_u32[2];
  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, _QWORD, _QWORD, _QWORD, __int64))(v62 + 1456))(
          v61,
          *(unsigned __int16 *)(*(_QWORD *)(v14 + 48) + v229.m128i_i64[0]),
          *(_QWORD *)(*(_QWORD *)(v14 + 48) + v229.m128i_i64[0] + 8),
          v231,
          v232)
    && (!*((_BYTE *)a1 + 33) || sub_328D6E0(a1[1], 0xD6u, v231))
    && (unsigned __int8)sub_33DD2A0(*a1, v230.m128i_i64[0], v230.m128i_i64[1], 0) )
  {
    v7 = sub_33FA050(*a1, 214, (unsigned int)&v233, v231, v232, 16, v230.m128i_i64[0], v230.m128i_i64[1]);
    goto LABEL_8;
  }
  v24 = sub_327CF00(a1, v7);
  if ( v24 )
    goto LABEL_29;
  if ( *(_DWORD *)(v14 + 24) != 57 )
    goto LABEL_70;
  if ( (unsigned __int8)sub_3286E00(&v230) )
  {
    if ( (unsigned __int8)sub_33E0720(**(_QWORD **)(v14 + 40), *(_QWORD *)(*(_QWORD *)(v14 + 40) + 8LL), 0) )
    {
      v78 = *(_QWORD *)(v14 + 40);
      if ( *(_DWORD *)(*(_QWORD *)(v78 + 40) + 24LL) == 214 )
      {
        v79 = a1[1];
        v225.m128i_i64[0] = *(_QWORD *)(v78 + 40);
        *(_QWORD *)&v228 = v232;
        LODWORD(v227) = v231;
        if ( (unsigned __int8)sub_328A020(v79, 0x39u, v231, v232, 0) )
        {
          v80 = sub_33FB310(
                  *a1,
                  **(_QWORD **)(v225.m128i_i64[0] + 40),
                  *(_QWORD *)(*(_QWORD *)(v225.m128i_i64[0] + 40) + 8LL),
                  &v233,
                  (unsigned int)v227,
                  v228);
          v15 = sub_3407430(*a1, v80, v81, &v233, v231, v232);
          goto LABEL_7;
        }
      }
    }
LABEL_70:
    if ( *(_DWORD *)(v14 + 24) == 56 )
    {
      if ( (unsigned __int8)sub_3286E00(&v230) )
      {
        if ( (unsigned __int8)sub_33E07E0(
                                *(_QWORD *)(*(_QWORD *)(v14 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v14 + 40) + 48LL),
                                0) )
        {
          v72 = *(_QWORD *)(v14 + 40);
          if ( *(_DWORD *)(*(_QWORD *)v72 + 24LL) == 214 )
          {
            v73 = a1[1];
            v225.m128i_i64[0] = *(_QWORD *)v72;
            *(_QWORD *)&v228 = v232;
            LODWORD(v227) = v231;
            if ( (unsigned __int8)sub_328A020(v73, 0x38u, v231, v232, 0) )
            {
              v74.m128i_i64[0] = sub_33FB310(
                                   *a1,
                                   **(_QWORD **)(v225.m128i_i64[0] + 40),
                                   *(_QWORD *)(*(_QWORD *)(v225.m128i_i64[0] + 40) + 8LL),
                                   &v233,
                                   (unsigned int)v227,
                                   v228);
              v75 = *a1;
              v229 = v74;
              *(_QWORD *)&v76 = sub_34015B0(v75, &v233, v231, v232, 0, 0);
              v55 = sub_3406EB0(v75, 56, (unsigned int)&v233, v231, v232, v77, *(_OWORD *)&v229, v76);
              goto LABEL_55;
            }
          }
        }
      }
    }
  }
  if ( *(_WORD *)(*(_QWORD *)(v14 + 48) + v229.m128i_i64[0]) == 2
    && (unsigned __int8)sub_33DFCF0(v230.m128i_i64[0], v230.m128i_i64[1], 0)
    && (unsigned __int8)sub_3286E00(&v230) )
  {
    if ( !*((_BYTE *)a1 + 33)
      || (v63 = a1[1], LODWORD(v228) = v231, v229.m128i_i64[0] = v232, sub_328D6E0(v63, 0xD6u, v231))
      && sub_328D6E0(v63, 0x38u, v228) )
    {
      *(_QWORD *)&v64 = sub_32FA5C0(a1, v14);
      if ( !(_QWORD)v64 )
      {
        v124.m128i_i64[0] = sub_33FAF80(
                              *a1,
                              214,
                              (unsigned int)&v233,
                              v231,
                              v232,
                              v65,
                              *(_OWORD *)*(_QWORD *)(v14 + 40));
        v125 = *a1;
        v229 = v124;
        *(_QWORD *)&v126 = sub_34015B0(v125, &v233, v231, v232, 0, 0);
        v7 = sub_3406EB0(v125, 56, (unsigned int)&v233, v231, v232, v127, *(_OWORD *)&v229, v126);
        goto LABEL_8;
      }
      if ( v14 == (_QWORD)v64 )
      {
        v7 = 0;
        goto LABEL_8;
      }
      v22 = sub_33FAF80(*a1, 213, (unsigned int)&v233, v231, v232, v65, v64);
LABEL_22:
      v7 = v22;
      goto LABEL_8;
    }
  }
  else
  {
    v63 = a1[1];
  }
  v69 = v7;
  v7 = 0;
  v70 = sub_32735C0(v69, v63, *a1, (int)&v233);
  if ( v70 )
    v7 = v70;
LABEL_8:
  if ( v233 )
    sub_B91220((__int64)&v233, v233);
  return v7;
}
