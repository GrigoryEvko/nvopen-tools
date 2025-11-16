// Function: sub_2036AE0
// Address: 0x2036ae0
//
__int64 *__fastcall sub_2036AE0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int16 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 *v9; // rdx
  unsigned int v10; // eax
  __m128 v11; // xmm0
  __int64 v12; // r15
  const void **v13; // rdx
  unsigned int v14; // r12d
  unsigned __int16 v15; // ax
  unsigned __int16 v16; // r14
  unsigned __int64 *v17; // rax
  __m128i v18; // xmm1
  unsigned __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  int v24; // r8d
  int v25; // r9d
  unsigned __int64 v26; // rdx
  __int64 v27; // rdx
  __int8 v28; // al
  __int64 v29; // rdx
  unsigned __int32 v30; // r15d
  __int64 v31; // r12
  __int64 v32; // r13
  char v33; // al
  __int128 v34; // rax
  __int64 *v35; // rax
  __int64 *v36; // r15
  unsigned __int64 v37; // rdx
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // r11
  unsigned int v41; // edx
  char v42; // al
  __int128 v43; // rax
  __int128 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 *v47; // r15
  __int64 v48; // rax
  unsigned int v49; // edx
  _QWORD *v50; // r14
  unsigned __int8 v51; // al
  __int64 v52; // rdx
  __int64 v53; // r14
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // r15
  __int128 v56; // rax
  __int64 *v57; // r14
  unsigned int v59; // r13d
  _QWORD *v60; // r12
  unsigned __int8 v61; // al
  __int64 *v62; // r12
  __int64 *v63; // r8
  __int64 v64; // r13
  char v65; // al
  __int64 v66; // rcx
  __int128 v67; // rax
  __int64 *v68; // rax
  __int64 *v69; // r15
  unsigned __int64 v70; // rdx
  __int64 v71; // rax
  unsigned int v72; // edx
  char v73; // al
  __int64 v74; // rsi
  __int64 v75; // rcx
  __int128 v76; // rax
  __int128 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 *v80; // r15
  __int64 v81; // rax
  unsigned int v82; // edx
  __int64 v83; // rdx
  unsigned int *v84; // r12
  __int64 **v85; // r15
  int v86; // ebx
  __int64 v87; // rax
  unsigned int v88; // r13d
  unsigned int v89; // eax
  __int64 v90; // rcx
  __int64 v91; // r9
  unsigned int v92; // r13d
  __int64 *v93; // rax
  __int64 v94; // rax
  __int32 v95; // r8d
  __int8 v96; // cl
  __int64 v97; // rsi
  unsigned int *v98; // rdx
  __int64 v99; // rax
  unsigned __int8 v100; // cl
  unsigned int v101; // eax
  __int64 v102; // r14
  __int64 **v103; // r12
  unsigned int v104; // r15d
  _QWORD *v105; // r13
  __int64 v106; // rax
  __int64 v107; // r9
  __int64 *v108; // rcx
  const void **v109; // rdx
  unsigned int v110; // r13d
  unsigned int v111; // r14d
  __int64 *v112; // rdi
  __m128i v113; // rax
  __int128 *v114; // r14
  __int64 v115; // r12
  char v116; // al
  __int64 v117; // rsi
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rdx
  __int64 *v121; // r15
  __int64 v122; // rax
  unsigned int v123; // edx
  __int64 v124; // rax
  __int64 *v125; // rdi
  __int64 v126; // rdx
  int v127; // r8d
  _QWORD *v128; // r12
  __int64 v129; // r9
  __int64 v130; // rcx
  unsigned int v131; // r14d
  __int64 v132; // rdx
  __int64 v133; // rdi
  __int64 v134; // rcx
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 *v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rdi
  __int64 *v141; // rdx
  __int64 v142; // rax
  __int64 *v143; // r12
  unsigned int v144; // eax
  __int64 v145; // rcx
  __int64 v146; // r8
  __int64 v147; // r9
  unsigned int v148; // eax
  _QWORD *v149; // rdi
  __int64 v150; // rdx
  __int64 v151; // r9
  __int64 v152; // rcx
  __int64 v153; // rdx
  __int64 v154; // rax
  __int8 v155; // dl
  __int64 v156; // rax
  __int64 v157; // rax
  __int8 v158; // dl
  __int64 v159; // rax
  __int128 v160; // [rsp-10h] [rbp-420h]
  __int128 v161; // [rsp-10h] [rbp-420h]
  int v162; // [rsp+8h] [rbp-408h]
  __int64 v163; // [rsp+18h] [rbp-3F8h]
  unsigned __int64 v164; // [rsp+40h] [rbp-3D0h]
  __int64 *v165; // [rsp+48h] [rbp-3C8h]
  __int64 v166; // [rsp+50h] [rbp-3C0h]
  __int64 v167; // [rsp+58h] [rbp-3B8h]
  const void **v168; // [rsp+60h] [rbp-3B0h]
  unsigned int v169; // [rsp+68h] [rbp-3A8h]
  __int64 v170; // [rsp+70h] [rbp-3A0h]
  __int32 v171; // [rsp+70h] [rbp-3A0h]
  unsigned __int64 v172; // [rsp+78h] [rbp-398h]
  __int64 v173; // [rsp+80h] [rbp-390h]
  unsigned __int64 v174; // [rsp+88h] [rbp-388h]
  __int64 v175; // [rsp+90h] [rbp-380h]
  unsigned __int16 v176; // [rsp+98h] [rbp-378h]
  __int64 **v177; // [rsp+98h] [rbp-378h]
  __int64 v178; // [rsp+A0h] [rbp-370h]
  __int64 (__fastcall *v179)(__int64, __int64); // [rsp+A0h] [rbp-370h]
  const void **v180; // [rsp+A0h] [rbp-370h]
  unsigned __int64 v181; // [rsp+A8h] [rbp-368h]
  unsigned __int32 v182; // [rsp+B0h] [rbp-360h]
  __int64 (__fastcall *v183)(__int64, __int64); // [rsp+B0h] [rbp-360h]
  __int64 *v184; // [rsp+B0h] [rbp-360h]
  __int32 v185; // [rsp+B0h] [rbp-360h]
  __int64 *v186; // [rsp+B8h] [rbp-358h]
  unsigned int v187; // [rsp+B8h] [rbp-358h]
  char v188; // [rsp+B8h] [rbp-358h]
  __int64 v189; // [rsp+C0h] [rbp-350h]
  __int64 v190; // [rsp+C0h] [rbp-350h]
  __int64 (__fastcall *v191)(__int64, __int64); // [rsp+C0h] [rbp-350h]
  __int64 v192; // [rsp+C0h] [rbp-350h]
  __int64 v193; // [rsp+C0h] [rbp-350h]
  unsigned __int64 v194; // [rsp+C8h] [rbp-348h]
  __int64 v195; // [rsp+D0h] [rbp-340h]
  __int64 (__fastcall *v196)(__int64, __int64); // [rsp+D0h] [rbp-340h]
  __int64 *v197; // [rsp+D0h] [rbp-340h]
  unsigned __int32 v198; // [rsp+D0h] [rbp-340h]
  int v199; // [rsp+D8h] [rbp-338h]
  __int64 (__fastcall *v200)(__int64, __int64); // [rsp+D8h] [rbp-338h]
  __int64 v201; // [rsp+D8h] [rbp-338h]
  __int64 v202; // [rsp+D8h] [rbp-338h]
  unsigned int v203; // [rsp+E4h] [rbp-32Ch]
  unsigned int v204; // [rsp+E8h] [rbp-328h]
  unsigned int v205; // [rsp+E8h] [rbp-328h]
  __m128i v206; // [rsp+F0h] [rbp-320h] BYREF
  _QWORD *v207; // [rsp+100h] [rbp-310h]
  __int64 v208; // [rsp+108h] [rbp-308h]
  _QWORD *v209; // [rsp+110h] [rbp-300h]
  __int64 v210; // [rsp+118h] [rbp-2F8h]
  __int64 *v211; // [rsp+120h] [rbp-2F0h]
  __int64 v212; // [rsp+128h] [rbp-2E8h]
  __int64 *v213; // [rsp+130h] [rbp-2E0h]
  __int64 v214; // [rsp+138h] [rbp-2D8h]
  __m128i v215; // [rsp+140h] [rbp-2D0h]
  __int64 *v216; // [rsp+150h] [rbp-2C0h]
  __int64 v217; // [rsp+158h] [rbp-2B8h]
  __int64 *v218; // [rsp+160h] [rbp-2B0h]
  __int64 v219; // [rsp+168h] [rbp-2A8h]
  __int64 v220; // [rsp+170h] [rbp-2A0h] BYREF
  int v221; // [rsp+178h] [rbp-298h]
  __m128i v222; // [rsp+180h] [rbp-290h] BYREF
  __m128i v223; // [rsp+190h] [rbp-280h] BYREF
  __m128i v224; // [rsp+1A0h] [rbp-270h] BYREF
  __m128i v225; // [rsp+1B0h] [rbp-260h] BYREF
  _QWORD v226[2]; // [rsp+1C0h] [rbp-250h] BYREF
  _BYTE v227[256]; // [rsp+1D0h] [rbp-240h] BYREF
  __m128i v228; // [rsp+2D0h] [rbp-140h] BYREF
  _QWORD v229[38]; // [rsp+2E0h] [rbp-130h] BYREF

  v6 = *(_WORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 72);
  v220 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v220, v7, 2);
  v8 = (__int64)*a1;
  v9 = a1[1];
  v221 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v228, v8, v9[6], **(unsigned __int8 **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v222.m128i_i8[0] = v228.m128i_i8[8];
  v222.m128i_i64[1] = v229[0];
  LOBYTE(v10) = sub_1F7E0F0((__int64)&v222);
  v11 = (__m128)_mm_loadu_si128(&v222);
  v12 = v222.m128i_u8[0];
  v168 = v13;
  v169 = v10;
  v223 = (__m128i)v11;
  if ( v222.m128i_i8[0] )
    v14 = word_4305480[(unsigned __int8)(v222.m128i_i8[0] - 14)];
  else
    v14 = sub_1F58D30((__int64)&v223);
  v15 = *(_WORD *)(a2 + 80);
  v206.m128i_i16[0] = v6;
  v176 = v15;
  while ( !(_BYTE)v12 || !(*a1)[v12 + 15] )
  {
    if ( v14 == 1 )
    {
      v143 = a1[1];
      goto LABEL_153;
    }
    v14 >>= 1;
    v50 = (_QWORD *)a1[1][6];
    v51 = sub_1D15020(v169, v14);
    v52 = 0;
    v12 = v51;
    if ( !v51 )
      v12 = (unsigned __int8)sub_1F593D0(v50, v169, (__int64)v168, v14);
    v223.m128i_i8[0] = v12;
    v223.m128i_i64[1] = v52;
  }
  v16 = v206.m128i_i16[0];
  if ( v14 == 1 )
  {
    v143 = a1[1];
LABEL_153:
    v144 = sub_1D15970(&v222);
    v57 = sub_1D40890(v143, a2, v144, v145, v146, v147, (__m128i)v11, a4, a5);
    goto LABEL_39;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(**a1 + 344))(
          *a1,
          *(unsigned __int16 *)(a2 + 24),
          v223.m128i_u32[0],
          v223.m128i_i64[1]) )
  {
    v53 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
    v55 = v54;
    *(_QWORD *)&v56 = sub_20363F0(
                        (__int64)a1,
                        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
    v57 = sub_1D332F0(
            a1[1],
            *(unsigned __int16 *)(a2 + 24),
            (__int64)&v220,
            v222.m128i_u32[0],
            (const void **)v222.m128i_i64[1],
            v176,
            *(double *)v11.m128_u64,
            a4,
            a5,
            v53,
            v55,
            v56);
    goto LABEL_39;
  }
  v17 = *(unsigned __int64 **)(a2 + 32);
  v18 = _mm_loadu_si128(&v223);
  v19 = *v17;
  v20 = v17[1];
  v224 = v18;
  v173 = sub_20363F0((__int64)a1, v19, v20);
  v174 = v21;
  v22 = sub_20363F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v172 = v26;
  v27 = *(_QWORD *)(a2 + 40);
  v170 = v22;
  v28 = *(_BYTE *)v27;
  v29 = *(_QWORD *)(v27 + 8);
  v228.m128i_i8[0] = v28;
  v228.m128i_i64[1] = v29;
  if ( v28 )
    v206.m128i_i32[0] = word_4305480[(unsigned __int8)(v28 - 14)];
  else
    v206.m128i_i32[0] = sub_1F58D30((__int64)&v228);
  v30 = v14;
  v226[0] = v227;
  v228.m128i_i64[0] = 0;
  v228.m128i_i32[2] = 0;
  v226[1] = 0x1000000000LL;
  sub_202F910((__int64)v226, v206.m128i_u32[0], &v228, v23, v24, v25);
  v199 = 0;
  v203 = 0;
LABEL_13:
  if ( v206.m128i_i32[0] )
  {
    if ( v30 > v206.m128i_i32[0] )
    {
      v59 = v169;
    }
    else
    {
      v182 = v30;
      v31 = v189;
      v32 = v195;
      while ( 1 )
      {
        v47 = a1[1];
        v197 = *a1;
        v191 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
        v48 = sub_1E0A0C0(v47[4]);
        if ( v191 == sub_1D13A20 )
        {
          v49 = 8 * sub_15A9520(v48, 0);
          if ( v49 == 32 )
          {
            v33 = 5;
          }
          else if ( v49 <= 0x20 )
          {
            v33 = 3;
            if ( v49 != 8 )
              v33 = 4 * (v49 == 16);
          }
          else
          {
            v33 = 6;
            if ( v49 != 64 )
            {
              v33 = 0;
              if ( v49 == 128 )
                v33 = 7;
            }
          }
        }
        else
        {
          v33 = v191((__int64)v197, v48);
        }
        LOBYTE(v32) = v33;
        *(_QWORD *)&v34 = sub_1D38BB0(
                            (__int64)v47,
                            v199,
                            (__int64)&v220,
                            (unsigned int)v32,
                            0,
                            0,
                            (__m128i)v11,
                            *(double *)v18.m128i_i64,
                            a5,
                            0);
        v35 = sub_1D332F0(
                v47,
                109,
                (__int64)&v220,
                v223.m128i_u32[0],
                (const void **)v223.m128i_i64[1],
                0,
                *(double *)v11.m128_u64,
                *(double *)v18.m128i_i64,
                a5,
                v173,
                v174,
                v34);
        v36 = a1[1];
        v190 = (__int64)v35;
        v194 = v37;
        v186 = *a1;
        v196 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
        v38 = sub_1E0A0C0(v36[4]);
        if ( v196 == sub_1D13A20 )
        {
          v39 = sub_15A9520(v38, 0);
          v40 = v199;
          v41 = 8 * v39;
          if ( 8 * v39 == 32 )
          {
            v42 = 5;
          }
          else if ( v41 > 0x20 )
          {
            v42 = 6;
            if ( v41 != 64 )
            {
              v42 = 0;
              if ( v41 == 128 )
                v42 = 7;
            }
          }
          else
          {
            v42 = 3;
            if ( v41 != 8 )
              v42 = 4 * (v41 == 16);
          }
        }
        else
        {
          v42 = v196((__int64)v186, v38);
          v40 = v199;
        }
        LOBYTE(v31) = v42;
        *(_QWORD *)&v43 = sub_1D38BB0(
                            (__int64)v36,
                            v40,
                            (__int64)&v220,
                            (unsigned int)v31,
                            0,
                            0,
                            (__m128i)v11,
                            *(double *)v18.m128i_i64,
                            a5,
                            0);
        *(_QWORD *)&v44 = sub_1D332F0(
                            v36,
                            109,
                            (__int64)&v220,
                            v223.m128i_u32[0],
                            (const void **)v223.m128i_i64[1],
                            0,
                            *(double *)v11.m128_u64,
                            *(double *)v18.m128i_i64,
                            a5,
                            v170,
                            v172,
                            v43);
        v218 = sub_1D332F0(
                 a1[1],
                 v16,
                 (__int64)&v220,
                 v223.m128i_u32[0],
                 (const void **)v223.m128i_i64[1],
                 v176,
                 *(double *)v11.m128_u64,
                 *(double *)v18.m128i_i64,
                 a5,
                 v190,
                 v194,
                 v44);
        v45 = v226[0] + 16LL * v203;
        v219 = v46;
        *(_QWORD *)v45 = v218;
        *(_DWORD *)(v45 + 8) = v219;
        v206.m128i_i32[0] -= v182;
        v199 += v182;
        if ( v182 > v206.m128i_i32[0] )
          break;
        ++v203;
      }
      v189 = v31;
      v30 = v182;
      ++v203;
      v195 = v32;
      v59 = v169;
    }
    while ( 1 )
    {
      v30 >>= 1;
      v60 = (_QWORD *)a1[1][6];
      v61 = sub_1D15020(v59, v30);
      if ( v61 )
      {
        v223.m128i_i8[0] = v61;
        v62 = *a1;
        v223.m128i_i64[1] = 0;
      }
      else
      {
        v61 = sub_1F593D0(v60, v59, (__int64)v168, v30);
        v62 = *a1;
        v223.m128i_i8[0] = v61;
        v223.m128i_i64[1] = v83;
        if ( !v61 )
          goto LABEL_74;
      }
      if ( v62[v61 + 15] )
      {
        if ( v30 != 1 )
          goto LABEL_13;
LABEL_55:
        if ( v206.m128i_i32[0] )
        {
          v63 = v62;
          v64 = v199;
          v164 = v199 + (unsigned __int64)(unsigned int)(v206.m128i_i32[0] - 1) + 1;
          while ( 1 )
          {
            v80 = a1[1];
            v184 = v63;
            v187 = v64 + v203 - v199;
            v179 = *(__int64 (__fastcall **)(__int64, __int64))(*v63 + 48);
            v81 = sub_1E0A0C0(v80[4]);
            if ( v179 == sub_1D13A20 )
            {
              v82 = 8 * sub_15A9520(v81, 0);
              if ( v82 == 32 )
              {
                v65 = 5;
              }
              else if ( v82 <= 0x20 )
              {
                v65 = 3;
                if ( v82 != 8 )
                  v65 = 4 * (v82 == 16);
              }
              else
              {
                v65 = 6;
                if ( v82 != 64 )
                {
                  v65 = 0;
                  if ( v82 == 128 )
                    v65 = 7;
                }
              }
            }
            else
            {
              v65 = v179((__int64)v184, v81);
            }
            v66 = v166;
            LOBYTE(v66) = v65;
            *(_QWORD *)&v67 = sub_1D38BB0(
                                (__int64)v80,
                                v64,
                                (__int64)&v220,
                                v66,
                                0,
                                0,
                                (__m128i)v11,
                                *(double *)v18.m128i_i64,
                                a5,
                                0);
            v68 = sub_1D332F0(
                    v80,
                    106,
                    (__int64)&v220,
                    v169,
                    v168,
                    0,
                    *(double *)v11.m128_u64,
                    *(double *)v18.m128i_i64,
                    a5,
                    v173,
                    v174,
                    v67);
            v69 = a1[1];
            v178 = (__int64)v68;
            v181 = v70;
            v165 = *a1;
            v183 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
            v71 = sub_1E0A0C0(v69[4]);
            if ( v183 == sub_1D13A20 )
            {
              v72 = 8 * sub_15A9520(v71, 0);
              if ( v72 == 32 )
              {
                v73 = 5;
              }
              else if ( v72 > 0x20 )
              {
                v73 = 6;
                if ( v72 != 64 )
                {
                  v73 = 0;
                  if ( v72 == 128 )
                    v73 = 7;
                }
              }
              else
              {
                v73 = 3;
                if ( v72 != 8 )
                  v73 = 4 * (v72 == 16);
              }
            }
            else
            {
              v73 = v183((__int64)v165, v71);
            }
            v74 = v64;
            v75 = v167;
            ++v64;
            LOBYTE(v75) = v73;
            *(_QWORD *)&v76 = sub_1D38BB0(
                                (__int64)v69,
                                v74,
                                (__int64)&v220,
                                v75,
                                0,
                                0,
                                (__m128i)v11,
                                *(double *)v18.m128i_i64,
                                a5,
                                0);
            *(_QWORD *)&v77 = sub_1D332F0(
                                v69,
                                106,
                                (__int64)&v220,
                                v169,
                                v168,
                                0,
                                *(double *)v11.m128_u64,
                                *(double *)v18.m128i_i64,
                                a5,
                                v170,
                                v172,
                                v76);
            v216 = sub_1D332F0(
                     a1[1],
                     v16,
                     (__int64)&v220,
                     v169,
                     v168,
                     v176,
                     *(double *)v11.m128_u64,
                     *(double *)v18.m128i_i64,
                     a5,
                     v178,
                     v181,
                     v77);
            v78 = v226[0] + 16LL * v187;
            v217 = v79;
            *(_QWORD *)v78 = v216;
            *(_DWORD *)(v78 + 8) = v217;
            if ( v164 == v64 )
              break;
            v63 = *a1;
          }
          v203 += v206.m128i_i32[0];
          v199 += v206.m128i_i32[0];
        }
        v206.m128i_i32[0] = 0;
        v30 = 1;
        goto LABEL_13;
      }
LABEL_74:
      if ( v30 == 1 )
        goto LABEL_55;
    }
  }
  v84 = (unsigned int *)v226[0];
  if ( v203 == 1 )
  {
    v154 = *(_QWORD *)(*(_QWORD *)v226[0] + 40LL) + 16LL * *(unsigned int *)(v226[0] + 8LL);
    v155 = *(_BYTE *)v154;
    v156 = *(_QWORD *)(v154 + 8);
    v223.m128i_i8[0] = v155;
    v223.m128i_i64[1] = v156;
    if ( v222.m128i_i8[0] == v155 && (v222.m128i_i64[1] == v156 || v155) )
    {
LABEL_162:
      v57 = *(__int64 **)v84;
      goto LABEL_91;
    }
  }
  v85 = a1;
  v86 = v162;
  v185 = v203 - 1;
LABEL_85:
  v198 = v185;
  v87 = *(_QWORD *)(*(_QWORD *)&v84[4 * v185] + 40LL) + 16LL * v84[4 * v185 + 2];
  if ( v224.m128i_i8[0] != *(_BYTE *)v87 || *(_QWORD *)(v87 + 8) != v224.m128i_i64[1] && !*(_BYTE *)v87 )
  {
    v171 = v185;
    v94 = *(_QWORD *)(*(_QWORD *)&v84[4 * v185] + 40LL) + 16LL * v84[4 * v185 + 2];
    v95 = v203 - 2;
    v96 = *(_BYTE *)v94;
    v97 = *(_QWORD *)(v94 + 8);
    v206.m128i_i32[0] = v203 - 2;
    v223.m128i_i8[0] = v96;
    v223.m128i_i64[1] = v97;
    if ( (int)(v203 - 2) < 0 )
    {
      v204 = 1;
LABEL_97:
      if ( !v96 )
        goto LABEL_133;
      v100 = v96 - 14;
      if ( v100 <= 0x5Fu )
      {
        v101 = word_4305480[v100];
LABEL_100:
        v102 = v163;
        v103 = v85;
        v104 = v101;
        goto LABEL_101;
      }
    }
    else
    {
      v98 = &v84[4 * v95];
      while ( 1 )
      {
        v99 = *(_QWORD *)(*(_QWORD *)v98 + 40LL) + 16LL * v98[2];
        if ( v96 != *(_BYTE *)v99 )
        {
          v206.m128i_i32[0] = v95;
          v171 = v95 + 1;
          v198 = v95 + 1;
          v204 = v185 - v95;
          v203 = v95 + 2;
          v185 = v95 + 1;
          goto LABEL_97;
        }
        if ( v97 != *(_QWORD *)(v99 + 8) && !v96 )
          break;
        v98 -= 4;
        if ( !v95 )
        {
          v198 = 0;
          v171 = 0;
          v204 = v203;
          v206.m128i_i32[0] = -1;
          v203 = 1;
          v185 = 0;
          goto LABEL_97;
        }
        --v95;
      }
      v206.m128i_i32[0] = v95;
      v171 = v95 + 1;
      v198 = v95 + 1;
      v204 = v185 - v95;
      v203 = v95 + 2;
      v185 = v95 + 1;
LABEL_133:
      if ( sub_1F58D20((__int64)&v223) )
      {
        v101 = sub_1F58D30((__int64)&v223);
        goto LABEL_100;
      }
    }
    v103 = v85;
    v104 = 1;
    v102 = v163;
    while ( 1 )
    {
LABEL_101:
      while ( 1 )
      {
        v104 *= 2;
        v105 = (_QWORD *)v103[1][6];
        LOBYTE(v106) = sub_1D15020(v169, v104);
        if ( !(_BYTE)v106 )
          break;
        LOBYTE(v102) = v106;
        v108 = *v103;
        v109 = 0;
        v110 = v102;
LABEL_103:
        if ( v108[(unsigned __int8)v106 + 15] )
        {
          v188 = v106;
          v163 = v102;
          v111 = v104;
          v85 = v103;
          v180 = v109;
          if ( v223.m128i_i8[0] )
          {
            if ( (unsigned __int8)(v223.m128i_i8[0] - 14) > 0x5Fu )
              goto LABEL_106;
LABEL_139:
            v125 = v103[1];
            v228.m128i_i64[0] = 0;
            v228.m128i_i32[2] = 0;
            v128 = sub_1D2B300(v125, 0x30u, (__int64)&v228, v223.m128i_u32[0], v223.m128i_i64[1], v107);
            v129 = v126;
            if ( v228.m128i_i64[0] )
            {
              v201 = v126;
              sub_161E7C0((__int64)&v228, v228.m128i_i64[0]);
              v129 = v201;
            }
            if ( v223.m128i_i8[0] )
            {
              v130 = word_4305480[(unsigned __int8)(v223.m128i_i8[0] - 14)];
            }
            else
            {
              v202 = v129;
              v148 = sub_1F58D30((__int64)&v223);
              v129 = v202;
              v130 = v148;
            }
            v193 = v129;
            v225.m128i_i64[0] = 0;
            v225.m128i_i32[2] = 0;
            v131 = v111 / (unsigned int)v130;
            v228.m128i_i64[0] = (__int64)v229;
            v228.m128i_i64[1] = 0x1000000000LL;
            sub_202F910((__int64)&v228, v131, &v225, v130, v127, v129);
            if ( v204 )
            {
              v132 = 0;
              v133 = 16 * (v206.m128i_i32[0] + 1LL);
              do
              {
                v134 = v228.m128i_i64[0];
                v135 = v132 + v133 + v226[0];
                *(_QWORD *)(v228.m128i_i64[0] + v132) = *(_QWORD *)v135;
                *(_DWORD *)(v134 + v132 + 8) = *(_DWORD *)(v135 + 8);
                v132 += 16;
              }
              while ( 16LL * v204 != v132 );
            }
            if ( v131 > v204 )
            {
              v136 = 16LL * v204;
              do
              {
                v137 = v228.m128i_i64[0];
                v210 = v193;
                v209 = v128;
                *(_QWORD *)(v228.m128i_i64[0] + v136) = v128;
                *(_DWORD *)(v137 + v136 + 8) = v210;
                v136 += 16;
              }
              while ( 16 * (v204 + (unsigned __int64)(v131 - 1 - v204) + 1) != v136 );
            }
            LOBYTE(v110) = v188;
            *((_QWORD *)&v161 + 1) = v228.m128i_u32[2];
            *(_QWORD *)&v161 = v228.m128i_i64[0];
            v138 = sub_1D359D0(
                     v85[1],
                     107,
                     (__int64)&v220,
                     v110,
                     v180,
                     0,
                     *(double *)v11.m128_u64,
                     *(double *)v18.m128i_i64,
                     a5,
                     v161);
            v140 = v139;
            v141 = v138;
            v212 = v140;
            v142 = v226[0] + 16LL * v198;
            v211 = v141;
            *(_QWORD *)v142 = v141;
            *(_DWORD *)(v142 + 8) = v212;
            if ( (_QWORD *)v228.m128i_i64[0] != v229 )
              _libc_free(v228.m128i_u64[0]);
          }
          else
          {
            if ( sub_1F58D20((__int64)&v223) )
              goto LABEL_139;
LABEL_106:
            LOBYTE(v110) = v188;
            v112 = v103[1];
            v228.m128i_i64[0] = 0;
            v228.m128i_i32[2] = 0;
            v113.m128i_i64[0] = (__int64)sub_1D2B300(v112, 0x30u, (__int64)&v228, v110, (__int64)v180, v107);
            if ( v228.m128i_i64[0] )
            {
              v206 = v113;
              sub_161E7C0((__int64)&v228, v228.m128i_i64[0]);
              v113 = v206;
            }
            v206 = v113;
            if ( v204 )
            {
              LODWORD(v114) = v86;
              v115 = 0;
              v177 = v85;
              v175 = v204;
              do
              {
                v205 = v115 + v198;
                v121 = v177[1];
                v192 = (__int64)*v177;
                v200 = *(__int64 (__fastcall **)(__int64, __int64))(**v177 + 48);
                v122 = sub_1E0A0C0(v121[4]);
                if ( v200 == sub_1D13A20 )
                {
                  v123 = 8 * sub_15A9520(v122, 0);
                  if ( v123 == 32 )
                  {
                    v116 = 5;
                  }
                  else if ( v123 <= 0x20 )
                  {
                    v116 = 3;
                    if ( v123 != 8 )
                      v116 = 4 * (v123 == 16);
                  }
                  else
                  {
                    v116 = 6;
                    if ( v123 != 64 )
                    {
                      v116 = 0;
                      if ( v123 == 128 )
                        v116 = 7;
                    }
                  }
                }
                else
                {
                  v116 = v200(v192, v122);
                }
                LOBYTE(v114) = v116;
                v117 = v115++;
                v118 = sub_1D38BB0(
                         (__int64)v121,
                         v117,
                         (__int64)&v220,
                         (unsigned int)v114,
                         0,
                         0,
                         (__m128i)v11,
                         *(double *)v18.m128i_i64,
                         a5,
                         0);
                LOBYTE(v110) = v188;
                v114 = (__int128 *)(v226[0] + 16LL * v205);
                v213 = sub_1D3A900(
                         v121,
                         0x69u,
                         (__int64)&v220,
                         v110,
                         v180,
                         0,
                         v11,
                         *(double *)v18.m128i_i64,
                         a5,
                         v206.m128i_u64[0],
                         (__int16 *)v206.m128i_i64[1],
                         *v114,
                         v118,
                         v119);
                v206.m128i_i64[0] = (__int64)v213;
                v214 = v120;
                v206.m128i_i64[1] = (unsigned int)v120 | v206.m128i_i64[1] & 0xFFFFFFFF00000000LL;
              }
              while ( v175 != v115 );
              v85 = v177;
              v86 = (int)v114;
            }
            a5 = _mm_load_si128(&v206);
            v124 = v226[0] + 16LL * v171;
            v215 = a5;
            *(_QWORD *)v124 = v206.m128i_i64[0];
            *(_DWORD *)(v124 + 8) = v215.m128i_i32[2];
          }
          v84 = (unsigned int *)v226[0];
          goto LABEL_85;
        }
      }
      v106 = sub_1F593D0(v105, v169, (__int64)v168, v104);
      v102 = v106;
      v110 = v106;
      if ( (_BYTE)v106 )
      {
        v108 = *v103;
        goto LABEL_103;
      }
    }
  }
  if ( v203 == 1 )
  {
    v157 = *(_QWORD *)(*(_QWORD *)v84 + 40LL) + 16LL * v84[2];
    v158 = *(_BYTE *)v157;
    v159 = *(_QWORD *)(v157 + 8);
    v223.m128i_i8[0] = v158;
    v223.m128i_i64[1] = v159;
    if ( v222.m128i_i8[0] == v158 && (v222.m128i_i64[1] == v159 || v158) )
      goto LABEL_162;
  }
  v88 = sub_1D15970(&v222);
  v89 = sub_1D15970(&v224);
  v92 = v88 / v89;
  if ( v203 != v92 )
  {
    v149 = sub_1D2B530(v85[1], v224.m128i_u32[0], v224.m128i_i64[1], v90, v89, v91);
    v151 = v150;
    if ( v203 < v92 )
    {
      v152 = 16LL * v203;
      do
      {
        v153 = v226[0];
        v208 = v151;
        v207 = v149;
        *(_QWORD *)(v226[0] + v152) = v149;
        *(_DWORD *)(v153 + v152 + 8) = v208;
        v152 += 16;
      }
      while ( 16 * (v203 + (unsigned __int64)(v92 + ~v203) + 1) != v152 );
    }
    v84 = (unsigned int *)v226[0];
  }
  *((_QWORD *)&v160 + 1) = v92;
  *(_QWORD *)&v160 = v84;
  v93 = sub_1D359D0(
          v85[1],
          107,
          (__int64)&v220,
          v222.m128i_u32[0],
          (const void **)v222.m128i_i64[1],
          0,
          *(double *)v11.m128_u64,
          *(double *)v18.m128i_i64,
          a5,
          v160);
  v84 = (unsigned int *)v226[0];
  v57 = v93;
LABEL_91:
  if ( v84 != (unsigned int *)v227 )
    _libc_free((unsigned __int64)v84);
LABEL_39:
  if ( v220 )
    sub_161E7C0((__int64)&v220, v220);
  return v57;
}
