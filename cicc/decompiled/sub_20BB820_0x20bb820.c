// Function: sub_20BB820
// Address: 0x20bb820
//
__int64 __fastcall sub_20BB820(__int64 *a1, _QWORD *a2, __int64 *a3, double a4, double a5, __m128i a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // rdi
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  char v16; // dl
  const void **v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rbx
  _QWORD *v22; // r13
  __int8 v23; // al
  int i; // r13d
  unsigned int v25; // esi
  unsigned int v26; // esi
  bool v27; // cc
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  const void **v31; // r8
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rdx
  const void **v37; // rdx
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // rax
  unsigned __int8 v41; // bl
  int v42; // eax
  unsigned int v43; // esi
  __int64 v44; // r9
  _QWORD *v45; // rax
  unsigned int v46; // edi
  unsigned __int64 v47; // rdx
  __int64 v48; // rbx
  _QWORD *v49; // r14
  unsigned __int8 *v50; // rax
  __int64 v51; // rdx
  __int128 v52; // rax
  __int128 v53; // rax
  unsigned int v54; // r15d
  int v55; // ecx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 *v62; // rdx
  __int64 v63; // rbx
  unsigned int v64; // edx
  __int64 v65; // rdx
  __int64 *v66; // rax
  unsigned int v67; // edx
  __int64 v68; // rsi
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdi
  unsigned int v72; // eax
  __int64 v73; // r11
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rdi
  __int64 v76; // r10
  char v77; // r8
  __int64 v78; // rcx
  __int64 v79; // rbx
  __int64 v80; // rdx
  __int64 v81; // r14
  __int64 v82; // r10
  __int64 v83; // r14
  unsigned int v84; // esi
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rdi
  __int64 v90; // rbx
  __int64 v91; // rdx
  __m128i v92; // xmm2
  unsigned int v93; // eax
  __int64 v94; // rcx
  unsigned __int64 v95; // r14
  unsigned __int64 v96; // rsi
  int v97; // eax
  __int64 v98; // rax
  int v99; // r8d
  int v100; // r9d
  __int64 v101; // rdx
  __int64 v102; // r15
  __int64 v103; // r14
  __int64 v104; // rdx
  __int64 *v105; // rdx
  char v107; // di
  __int64 v108; // r15
  __int64 v109; // rax
  _QWORD *v110; // rdi
  int v111; // ebx
  __int64 v112; // rax
  __int64 v113; // rax
  const void **v114; // rdx
  __int128 v115; // rax
  __m128i v116; // rax
  __int64 v117; // rbx
  __int64 v118; // rdi
  __int16 v119; // r14
  _BYTE *v120; // r8
  __m128i *v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rdi
  __int64 v124; // rbx
  __int128 v125; // rax
  __int64 *v126; // rax
  unsigned int v127; // edx
  unsigned int v128; // ebx
  __int64 v129; // rdx
  unsigned __int64 v130; // r14
  int v131; // ecx
  unsigned __int64 v132; // rsi
  int v133; // eax
  __int64 v134; // rdi
  _BYTE *v135; // r8
  __m128i *v136; // rax
  bool v137; // zf
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 v140; // rax
  __int64 v141; // rdx
  char v142; // di
  __int64 v143; // r10
  __int64 v144; // rax
  __int64 v145; // rdx
  __int128 v146; // [rsp+0h] [rbp-310h]
  __int128 v147; // [rsp+0h] [rbp-310h]
  __int64 v148; // [rsp+10h] [rbp-300h]
  __int128 v149; // [rsp+30h] [rbp-2E0h]
  __int128 v150; // [rsp+40h] [rbp-2D0h]
  __int128 v151; // [rsp+50h] [rbp-2C0h]
  unsigned int v152; // [rsp+68h] [rbp-2A8h]
  unsigned int v153; // [rsp+6Ch] [rbp-2A4h]
  unsigned int v154; // [rsp+70h] [rbp-2A0h]
  unsigned int v155; // [rsp+74h] [rbp-29Ch]
  __int64 v156; // [rsp+78h] [rbp-298h]
  unsigned int v157; // [rsp+88h] [rbp-288h]
  __int64 v158; // [rsp+88h] [rbp-288h]
  __int64 v159; // [rsp+90h] [rbp-280h]
  const void **v160; // [rsp+90h] [rbp-280h]
  unsigned int v161; // [rsp+90h] [rbp-280h]
  __int64 v162; // [rsp+90h] [rbp-280h]
  __int64 v163; // [rsp+98h] [rbp-278h]
  unsigned int v164; // [rsp+A0h] [rbp-270h]
  __int64 v165; // [rsp+A0h] [rbp-270h]
  int v166; // [rsp+A8h] [rbp-268h]
  __int64 v167; // [rsp+B0h] [rbp-260h]
  _QWORD *v168; // [rsp+B0h] [rbp-260h]
  unsigned __int64 v169; // [rsp+B8h] [rbp-258h]
  unsigned int v170; // [rsp+B8h] [rbp-258h]
  int v171; // [rsp+B8h] [rbp-258h]
  __int64 v172; // [rsp+C0h] [rbp-250h]
  unsigned int v173; // [rsp+C0h] [rbp-250h]
  __int64 v174; // [rsp+C8h] [rbp-248h]
  const void **v175; // [rsp+C8h] [rbp-248h]
  __int64 v176; // [rsp+C8h] [rbp-248h]
  __int64 v177; // [rsp+C8h] [rbp-248h]
  __m128i v178; // [rsp+D0h] [rbp-240h] BYREF
  __m128i *v179; // [rsp+E0h] [rbp-230h]
  __int64 v180; // [rsp+E8h] [rbp-228h]
  __int64 v181; // [rsp+F0h] [rbp-220h]
  __int64 v182; // [rsp+F8h] [rbp-218h]
  __int64 *v183; // [rsp+100h] [rbp-210h]
  unsigned __int64 v184; // [rsp+108h] [rbp-208h]
  __m128i v185; // [rsp+110h] [rbp-200h]
  __int64 v186; // [rsp+120h] [rbp-1F0h]
  __int64 v187; // [rsp+128h] [rbp-1E8h]
  __int64 *v188; // [rsp+130h] [rbp-1E0h]
  __int64 v189; // [rsp+138h] [rbp-1D8h]
  __int64 v190; // [rsp+140h] [rbp-1D0h]
  __int64 v191; // [rsp+148h] [rbp-1C8h]
  unsigned int v192; // [rsp+150h] [rbp-1C0h] BYREF
  const void **v193; // [rsp+158h] [rbp-1B8h]
  __int64 v194; // [rsp+160h] [rbp-1B0h] BYREF
  int v195; // [rsp+168h] [rbp-1A8h]
  __int64 v196; // [rsp+170h] [rbp-1A0h] BYREF
  __int64 v197; // [rsp+178h] [rbp-198h]
  __m128i v198; // [rsp+180h] [rbp-190h] BYREF
  __m128i v199; // [rsp+190h] [rbp-180h] BYREF
  __int64 v200; // [rsp+1A0h] [rbp-170h]
  __int128 v201; // [rsp+1B0h] [rbp-160h] BYREF
  __int64 v202; // [rsp+1C0h] [rbp-150h]
  __int128 v203; // [rsp+1D0h] [rbp-140h] BYREF
  __int64 v204; // [rsp+1E0h] [rbp-130h]
  __int128 v205; // [rsp+1F0h] [rbp-120h] BYREF
  __int64 v206; // [rsp+200h] [rbp-110h]
  __int128 v207; // [rsp+210h] [rbp-100h] BYREF
  __int64 v208; // [rsp+220h] [rbp-F0h]
  __m128i v209; // [rsp+230h] [rbp-E0h] BYREF
  __int64 v210; // [rsp+240h] [rbp-D0h]
  __int128 v211; // [rsp+250h] [rbp-C0h] BYREF
  __int64 v212[22]; // [rsp+260h] [rbp-B0h] BYREF

  v6 = (__int64)a2;
  v8 = a2[4];
  v183 = a1;
  v9 = a2[13];
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 80));
  v11 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v174 = *(_QWORD *)v8;
  v12 = *(_QWORD *)(v8 + 8);
  v13 = 16LL * *(unsigned int *)(v8 + 48);
  v185 = v10;
  v172 = v12;
  v14 = *(_QWORD *)(v8 + 80);
  v178 = v11;
  v182 = v14;
  LODWORD(v181) = *(_DWORD *)(v8 + 88);
  v159 = *(_QWORD *)(v8 + 40);
  v15 = v13 + *(_QWORD *)(v159 + 40);
  v16 = *(_BYTE *)v15;
  v17 = *(const void ***)(v15 + 8);
  LOBYTE(v192) = v16;
  v193 = v17;
  v18 = sub_1E34390(v9);
  v19 = a2[9];
  v180 = v18;
  v20 = a3[4];
  v194 = v19;
  v156 = v20;
  if ( v19 )
    sub_1623A60((__int64)&v194, v19, 2);
  v21 = *(_QWORD *)(v6 + 96);
  v22 = (_QWORD *)a3[6];
  v195 = *(_DWORD *)(v6 + 64);
  v23 = *(_BYTE *)(v6 + 88);
  v209.m128i_i64[1] = v21;
  v209.m128i_i8[0] = v23;
  if ( v23 )
  {
    if ( (unsigned __int8)(v23 - 8) > 0x65u )
    {
      v179 = &v209;
      v170 = sub_1F3E310(&v209);
      goto LABEL_6;
    }
LABEL_16:
    if ( (_BYTE)v192 )
    {
      v26 = sub_1F3E310(&v192);
      v27 = v26 <= 0x20;
      if ( v26 != 32 )
      {
LABEL_18:
        if ( v27 )
        {
          if ( v26 == 8 )
          {
            v28 = 3;
            LOBYTE(v29) = 3;
          }
          else
          {
            v28 = 4;
            LOBYTE(v29) = 4;
            if ( v26 != 16 )
            {
              v28 = 2;
              LOBYTE(v29) = 2;
              if ( v26 != 1 )
                goto LABEL_28;
            }
          }
        }
        else if ( v26 == 64 )
        {
          v28 = 6;
          LOBYTE(v29) = 6;
        }
        else
        {
          if ( v26 != 128 )
          {
LABEL_28:
            v29 = sub_1F58CC0(v22, v26);
            v30 = v29;
            v31 = v37;
            if ( !(_BYTE)v29 )
              goto LABEL_29;
            v28 = (unsigned __int8)v29;
LABEL_22:
            if ( v183[v28 + 15] )
            {
              if ( (*((_BYTE *)v183 + 259 * (unsigned __int8)v29 + 2608) & 0xFB) != 0 )
              {
                v35 = sub_20B91E0(v183, v6, (__int64)a3, v10);
              }
              else
              {
                LOBYTE(v30) = v29;
                v32 = sub_1D309E0(
                        a3,
                        158,
                        (__int64)&v194,
                        v30,
                        v31,
                        0,
                        *(double *)v10.m128i_i64,
                        *(double *)v11.m128i_i64,
                        *(double *)a6.m128i_i64,
                        *(_OWORD *)&v178);
                v33 = *(_QWORD *)(v6 + 104);
                v211 = 0u;
                v212[0] = 0;
                v190 = sub_1D2BF40(
                         a3,
                         v174,
                         v172,
                         (__int64)&v194,
                         v32,
                         v34,
                         v185.m128i_i64[0],
                         v185.m128i_i64[1],
                         *(_OWORD *)v33,
                         *(_QWORD *)(v33 + 16),
                         v180,
                         *(unsigned __int16 *)(v33 + 32),
                         (__int64)&v211);
                v35 = v190;
                v191 = v36;
              }
              goto LABEL_73;
            }
LABEL_29:
            v38 = *(_QWORD *)(v6 + 96);
            v198.m128i_i8[0] = *(_BYTE *)(v6 + 88);
            v198.m128i_i64[1] = v38;
            if ( v198.m128i_i8[0] )
              v39 = sub_1F3E310(&v198);
            else
              v39 = sub_1F58D40((__int64)&v198);
            if ( v39 == 32 )
            {
              v40 = 5;
              goto LABEL_35;
            }
            if ( v39 > 0x20 )
            {
              if ( v39 == 64 )
              {
                v40 = 6;
                goto LABEL_35;
              }
              if ( v39 == 128 )
              {
                v40 = 7;
                goto LABEL_35;
              }
            }
            else
            {
              if ( v39 == 8 )
              {
                v40 = 3;
                goto LABEL_35;
              }
              v40 = 4;
              if ( v39 == 16 )
                goto LABEL_35;
              v40 = 2;
              if ( v39 == 1 )
                goto LABEL_35;
            }
            LOBYTE(v40) = sub_1F58CC0((_QWORD *)a3[6], v39);
            v79 = a3[6];
            v81 = v80;
            v40 = (unsigned __int8)v40;
            v209.m128i_i8[0] = v40;
            v209.m128i_i64[1] = v80;
            if ( !(_BYTE)v40 )
            {
              v179 = &v209;
              if ( sub_1F58D20((__int64)&v209) )
              {
                LOBYTE(v211) = 0;
                *((_QWORD *)&v211 + 1) = 0;
                LOBYTE(v205) = 0;
                sub_1F426C0((__int64)v183, v79, v209.m128i_u32[0], v81, (__int64)&v211, (unsigned int *)&v207, &v205);
                v41 = v205;
              }
              else
              {
                sub_1F40D10((__int64)&v211, (__int64)v183, v79, v209.m128i_i64[0], v209.m128i_i64[1]);
                v41 = sub_1D5E9F0((__int64)v183, v79, BYTE8(v211), v212[0]);
              }
              goto LABEL_36;
            }
LABEL_35:
            v41 = *((_BYTE *)v183 + v40 + 1155);
            v179 = &v209;
LABEL_36:
            LOBYTE(v196) = v41;
            v180 = (unsigned int)v181;
            v160 = *(const void ***)(*(_QWORD *)(v182 + 40) + 16LL * (unsigned int)v181 + 8);
            v183 = (__int64 *)*(unsigned __int8 *)(*(_QWORD *)(v182 + 40) + 16LL * (unsigned int)v181);
            if ( v198.m128i_i8[0] )
              v42 = sub_1F3E310(&v198);
            else
              v42 = sub_1F58D40((__int64)&v198);
            v164 = (unsigned int)v183;
            v43 = (unsigned int)(v42 + 7) >> 3;
            v152 = v43;
            v155 = (unsigned int)sub_1F3E310(&v196) >> 3;
            v154 = (v43 + v155 - 1) / v155;
            v45 = sub_1D29D50(a3, v198.m128i_u32[0], v198.m128i_i64[1], v41, 0, v44);
            v46 = *((_DWORD *)v45 + 21);
            v184 = v47;
            v48 = (unsigned int)v47;
            v49 = v45;
            v181 = (__int64)v45;
            v153 = v46;
            v183 = v45;
            v211 = 0u;
            v212[0] = 0;
            sub_1E341E0((__int64)&v201, v156, v46, 0);
            *(_QWORD *)&v151 = sub_1D2C750(
                                 a3,
                                 v174,
                                 v172,
                                 (__int64)&v194,
                                 v178.m128i_i64[0],
                                 v178.m128i_i64[1],
                                 (__int64)v183,
                                 v184,
                                 v201,
                                 v202,
                                 v198.m128i_i64[0],
                                 v198.m128i_i64[1],
                                 0,
                                 0,
                                 (__int64)&v211);
            v50 = (unsigned __int8 *)(v49[5] + 16 * v48);
            LODWORD(v49) = *v50;
            *((_QWORD *)&v151 + 1) = v51;
            v175 = (const void **)*((_QWORD *)v50 + 1);
            v178.m128i_i64[0] = v155;
            *(_QWORD *)&v52 = sub_1D38BB0(
                                (__int64)a3,
                                v155,
                                (__int64)&v194,
                                v164,
                                v160,
                                0,
                                v10,
                                *(double *)v11.m128i_i64,
                                a6,
                                0);
            v150 = v52;
            *(_QWORD *)&v53 = sub_1D38BB0(
                                (__int64)a3,
                                v155,
                                (__int64)&v194,
                                (unsigned int)v49,
                                v175,
                                0,
                                v10,
                                *(double *)v11.m128i_i64,
                                a6,
                                0);
            *(_QWORD *)&v211 = v212;
            *((_QWORD *)&v211 + 1) = 0x800000000LL;
            v149 = v53;
            if ( v154 <= 1 )
            {
              v83 = 0;
              v82 = 0;
            }
            else
            {
              v176 = v6;
              v54 = v157;
              v178.m128i_i32[0] = 1;
              v173 = 0;
              while ( 1 )
              {
                v209 = 0u;
                v210 = 0;
                sub_1E341E0((__int64)&v203, v156, v153, v173);
                v68 = v167;
                LOBYTE(v68) = v196;
                v183 = (__int64 *)v181;
                v184 = v48 | v184 & 0xFFFFFFFF00000000LL;
                v69 = sub_1D2B730(
                        a3,
                        v68,
                        0,
                        (__int64)&v194,
                        v151,
                        *((__int64 *)&v151 + 1),
                        v181,
                        v184,
                        v203,
                        v204,
                        0,
                        0,
                        (__int64)v179,
                        0);
                v209 = 0u;
                v158 = v69;
                v165 = v70;
                v71 = *(_QWORD *)(v176 + 104);
                v210 = 0;
                v161 = *(unsigned __int16 *)(v71 + 32);
                v72 = sub_1E34390(v71);
                v73 = *(_QWORD *)(v176 + 104);
                v74 = -(__int64)(v173 | (unsigned __int64)v72) & (v173 | (unsigned __int64)v72);
                v75 = *(_QWORD *)v73 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v75 )
                {
                  v76 = *(_QWORD *)(v73 + 8) + v173;
                  v77 = *(_BYTE *)(v73 + 16);
                  if ( (*(_QWORD *)v73 & 4) != 0 )
                  {
                    *((_QWORD *)&v205 + 1) = *(_QWORD *)(v73 + 8) + v173;
                    LOBYTE(v206) = v77;
                    *(_QWORD *)&v205 = v75 | 4;
                    HIDWORD(v206) = *(_DWORD *)(v75 + 12);
                  }
                  else
                  {
                    *(_QWORD *)&v205 = *(_QWORD *)v73 & 0xFFFFFFFFFFFFFFF8LL;
                    *((_QWORD *)&v205 + 1) = v76;
                    LOBYTE(v206) = v77;
                    v78 = *(_QWORD *)v75;
                    if ( *(_BYTE *)(*(_QWORD *)v75 + 8LL) == 16 )
                      v78 = **(_QWORD **)(v78 + 16);
                    HIDWORD(v206) = *(_DWORD *)(v78 + 8) >> 8;
                  }
                }
                else
                {
                  v55 = *(_DWORD *)(v73 + 20);
                  LODWORD(v206) = 0;
                  v205 = 0u;
                  HIDWORD(v206) = v55;
                }
                v185.m128i_i64[0] = v182;
                v169 = v169 & 0xFFFFFFFF00000000LL | 1;
                v185.m128i_i64[1] = v180 | v185.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                v56 = sub_1D2BF40(
                        a3,
                        v158,
                        v169,
                        (__int64)&v194,
                        v158,
                        v165,
                        v182,
                        v185.m128i_i64[1],
                        v205,
                        v206,
                        v74,
                        v161,
                        (__int64)v179);
                v58 = v57;
                v59 = v56;
                v60 = DWORD2(v211);
                if ( DWORD2(v211) >= HIDWORD(v211) )
                {
                  v162 = v56;
                  v163 = v58;
                  sub_16CD150((__int64)&v211, v212, 0, 16, v56, v58);
                  v60 = DWORD2(v211);
                  v59 = v162;
                  v58 = v163;
                }
                v61 = v181;
                v62 = (__int64 *)(v211 + 16 * v60);
                v62[1] = v58;
                *v62 = v59;
                ++DWORD2(v211);
                v63 = *(_QWORD *)(v61 + 40) + 16 * v48;
                LODWORD(v61) = v166;
                LOBYTE(v61) = *(_BYTE *)v63;
                v173 += v155;
                v181 = (__int64)sub_1D332F0(
                                  a3,
                                  52,
                                  (__int64)&v194,
                                  (unsigned int)v61,
                                  *(const void ***)(v63 + 8),
                                  3u,
                                  *(double *)v10.m128i_i64,
                                  *(double *)v11.m128i_i64,
                                  a6,
                                  (__int64)v183,
                                  v184,
                                  v149);
                v48 = v64;
                v65 = *(_QWORD *)(v182 + 40) + 16 * v180;
                LOBYTE(v54) = *(_BYTE *)v65;
                v66 = sub_1D332F0(
                        a3,
                        52,
                        (__int64)&v194,
                        v54,
                        *(const void ***)(v65 + 8),
                        3u,
                        *(double *)v10.m128i_i64,
                        *(double *)v11.m128i_i64,
                        a6,
                        v185.m128i_i64[0],
                        v185.m128i_u64[1],
                        v150);
                ++v178.m128i_i32[0];
                v182 = (__int64)v66;
                if ( v154 == v178.m128i_i32[0] )
                  break;
                v180 = v67;
              }
              v6 = v176;
              v152 -= v155 * (v154 - 1);
              v82 = v155 * (v154 - 1);
              v180 = v67;
              v83 = v82;
            }
            v84 = 8 * v152;
            if ( 8 * v152 == 32 )
            {
              LOBYTE(v85) = 5;
              goto LABEL_66;
            }
            if ( v84 > 0x20 )
            {
              if ( v84 == 64 )
              {
                LOBYTE(v85) = 6;
                goto LABEL_66;
              }
              if ( v84 == 128 )
              {
                LOBYTE(v85) = 7;
                goto LABEL_66;
              }
            }
            else
            {
              if ( v84 == 8 )
              {
                LOBYTE(v85) = 3;
                goto LABEL_66;
              }
              LOBYTE(v85) = 4;
              if ( v84 == 16 )
              {
LABEL_66:
                v86 = 0;
LABEL_67:
                v87 = v148;
                v178.m128i_i64[1] = v86;
                v177 = v82;
                v209 = 0u;
                LOBYTE(v87) = v85;
                v178.m128i_i64[0] = v87;
                v210 = 0;
                sub_1E341E0((__int64)&v207, v156, v153, v82);
                v183 = (__int64 *)v181;
                v184 = v48 | v184 & 0xFFFFFFFF00000000LL;
                v88 = sub_1D2B810(
                        a3,
                        1u,
                        (__int64)&v194,
                        (unsigned __int8)v196,
                        0,
                        0,
                        v151,
                        v181,
                        v184,
                        v207,
                        v208,
                        v87,
                        v178.m128i_i64[1],
                        0,
                        (__int64)v179);
                v89 = *(_QWORD *)(v6 + 104);
                v90 = v88;
                v181 = v91;
                v92 = _mm_loadu_si128((const __m128i *)(v89 + 40));
                v199 = v92;
                v200 = *(_QWORD *)(v89 + 56);
                LODWORD(v183) = *(unsigned __int16 *)(v89 + 32);
                v93 = sub_1E34390(v89);
                v94 = *(_QWORD *)(v6 + 104);
                v95 = -(__int64)(v93 | (unsigned __int64)v83) & (v93 | (unsigned __int64)v83);
                v96 = *(_QWORD *)v94 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v96 )
                {
                  v107 = *(_BYTE *)(v94 + 16);
                  v108 = v177 + *(_QWORD *)(v94 + 8);
                  if ( (*(_QWORD *)v94 & 4) != 0 )
                  {
                    v209.m128i_i64[1] = v177 + *(_QWORD *)(v94 + 8);
                    LOBYTE(v210) = v107;
                    v209.m128i_i64[0] = v96 | 4;
                    HIDWORD(v210) = *(_DWORD *)(v96 + 12);
                  }
                  else
                  {
                    v209.m128i_i64[0] = *(_QWORD *)v94 & 0xFFFFFFFFFFFFFFF8LL;
                    v209.m128i_i64[1] = v108;
                    LOBYTE(v210) = v107;
                    v109 = *(_QWORD *)v96;
                    if ( *(_BYTE *)(*(_QWORD *)v96 + 8LL) == 16 )
                      v109 = **(_QWORD **)(v109 + 16);
                    HIDWORD(v210) = *(_DWORD *)(v109 + 8) >> 8;
                  }
                }
                else
                {
                  v97 = *(_DWORD *)(v94 + 20);
                  LODWORD(v210) = 0;
                  v209 = 0u;
                  HIDWORD(v210) = v97;
                }
                v185.m128i_i64[0] = v182;
                v185.m128i_i64[1] = v180 | v185.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                v98 = sub_1D2C750(
                        a3,
                        v90,
                        1,
                        (__int64)&v194,
                        v90,
                        v181,
                        v182,
                        v185.m128i_i64[1],
                        *(_OWORD *)&v209,
                        v210,
                        v178.m128i_i64[0],
                        v178.m128i_i64[1],
                        v95,
                        (__int16)v183,
                        (__int64)&v199);
                v102 = v101;
                v103 = v98;
                v104 = DWORD2(v211);
                if ( DWORD2(v211) >= HIDWORD(v211) )
                {
                  sub_16CD150((__int64)&v211, v212, 0, 16, v99, v100);
                  v104 = DWORD2(v211);
                }
                v105 = (__int64 *)(v211 + 16 * v104);
                *v105 = v103;
                v105[1] = v102;
                ++DWORD2(v211);
                *((_QWORD *)&v146 + 1) = DWORD2(v211);
                *(_QWORD *)&v146 = v211;
                v35 = (__int64)sub_1D359D0(
                                 a3,
                                 2,
                                 (__int64)&v194,
                                 1,
                                 0,
                                 0,
                                 *(double *)v10.m128i_i64,
                                 *(double *)v11.m128i_i64,
                                 v92,
                                 v146);
                if ( (__int64 *)v211 != v212 )
                  _libc_free(v211);
                goto LABEL_73;
              }
            }
            v110 = (_QWORD *)a3[6];
            v178.m128i_i64[0] = v82;
            v85 = sub_1F58CC0(v110, v84);
            v82 = v178.m128i_i64[0];
            v148 = v85;
            goto LABEL_67;
          }
          v28 = 7;
          LOBYTE(v29) = 7;
        }
LABEL_21:
        v30 = (unsigned __int8)v29;
        v31 = 0;
        goto LABEL_22;
      }
    }
    else
    {
      v26 = sub_1F58D40((__int64)&v192);
      v27 = v26 <= 0x20;
      if ( v26 != 32 )
        goto LABEL_18;
    }
    v28 = 5;
    LOBYTE(v29) = 5;
    goto LABEL_21;
  }
  v179 = &v209;
  if ( sub_1F58CD0((__int64)&v209) )
    goto LABEL_16;
  LOBYTE(v211) = 0;
  *((_QWORD *)&v211 + 1) = v21;
  if ( sub_1F58D20((__int64)&v211) )
    goto LABEL_16;
  v170 = sub_1F58D40((__int64)v179);
LABEL_6:
  v168 = v22;
  for ( i = 2; i != 8; ++i )
  {
    LOBYTE(v211) = i;
    *((_QWORD *)&v211 + 1) = 0;
    if ( 2 * (unsigned int)sub_1F3E310(&v211) >= v170 )
    {
      LOBYTE(v196) = i;
      v197 = 0;
      goto LABEL_98;
    }
  }
  v25 = (v170 + 1) >> 1;
  if ( v25 == 32 )
  {
    LOBYTE(v196) = 5;
    v197 = 0;
    goto LABEL_98;
  }
  if ( v25 > 0x20 )
  {
    if ( v25 == 64 )
    {
      LOBYTE(v196) = 6;
      v197 = 0;
      goto LABEL_98;
    }
    if ( v25 == 128 )
    {
      LOBYTE(v196) = 7;
      v197 = 0;
      goto LABEL_98;
    }
  }
  else
  {
    switch ( v25 )
    {
      case 8u:
        LOBYTE(v196) = 3;
        v197 = 0;
        goto LABEL_98;
      case 0x10u:
        LOBYTE(v196) = 4;
        v197 = 0;
LABEL_98:
        v111 = sub_1F3E310(&v196);
        goto LABEL_99;
      case 1u:
        LOBYTE(v196) = 2;
        v197 = 0;
        goto LABEL_98;
    }
  }
  LOBYTE(v196) = sub_1F58CC0(v168, v25);
  v197 = v145;
  if ( (_BYTE)v196 )
    goto LABEL_98;
  v111 = sub_1F58D40((__int64)&v196);
LABEL_99:
  v171 = v111 / 8;
  v112 = sub_1E0A0C0(a3[4]);
  v113 = sub_1F40B60(
           (__int64)v183,
           *(unsigned __int8 *)(*(_QWORD *)(v159 + 40) + v13),
           *(_QWORD *)(*(_QWORD *)(v159 + 40) + v13 + 8),
           v112,
           1);
  *(_QWORD *)&v115 = sub_1D38BB0((__int64)a3, v111, (__int64)&v194, v113, v114, 0, v10, *(double *)v11.m128i_i64, a6, 0);
  v198 = _mm_load_si128(&v178);
  v116.m128i_i64[0] = (__int64)sub_1D332F0(
                                 a3,
                                 124,
                                 (__int64)&v194,
                                 v192,
                                 v193,
                                 0,
                                 *(double *)v10.m128i_i64,
                                 *(double *)v11.m128i_i64,
                                 a6,
                                 v178.m128i_i64[0],
                                 v178.m128i_u64[1],
                                 v115);
  v117 = *(_QWORD *)(v6 + 104);
  v118 = a3[4];
  v199 = v116;
  v211 = 0u;
  v212[0] = 0;
  v119 = *(_WORD *)(v117 + 32);
  v120 = (_BYTE *)sub_1E0A0C0(v118);
  v121 = &v199;
  if ( !*v120 )
    v121 = &v198;
  v188 = (__int64 *)sub_1D2C750(
                      a3,
                      v174,
                      v172,
                      (__int64)&v194,
                      v121->m128i_i64[0],
                      v121->m128i_i64[1],
                      v185.m128i_i64[0],
                      v185.m128i_i64[1],
                      *(_OWORD *)v117,
                      *(_QWORD *)(v117 + 16),
                      v196,
                      v197,
                      v180,
                      v119,
                      (__int64)&v211);
  v183 = v188;
  v184 = 0;
  v189 = v122;
  v123 = *(_QWORD *)(v182 + 40);
  v178.m128i_i64[0] = (unsigned int)v181;
  v181 = v171;
  v184 = (unsigned int)v122;
  v124 = 16 * v178.m128i_i64[0];
  *(_QWORD *)&v125 = sub_1D38BB0(
                       (__int64)a3,
                       v171,
                       (__int64)&v194,
                       *(unsigned __int8 *)(16 * v178.m128i_i64[0] + v123),
                       *(const void ***)(16 * v178.m128i_i64[0] + v123 + 8),
                       0,
                       v10,
                       *(double *)v11.m128i_i64,
                       a6,
                       0);
  v126 = sub_1D332F0(
           a3,
           52,
           (__int64)&v194,
           *(unsigned __int8 *)(*(_QWORD *)(v182 + 40) + v124),
           *(const void ***)(*(_QWORD *)(v182 + 40) + v124 + 8),
           3u,
           *(double *)v10.m128i_i64,
           *(double *)v11.m128i_i64,
           a6,
           v182,
           v178.m128i_u64[0],
           v125);
  v128 = v127;
  v129 = *(_QWORD *)(v6 + 104);
  v182 = (__int64)v126;
  v209 = _mm_loadu_si128((const __m128i *)(v129 + 40));
  v130 = (int)(v180 | v171) & (unsigned __int64)-(__int64)(int)(v180 | v171);
  v210 = *(_QWORD *)(v129 + 56);
  v131 = *(unsigned __int16 *)(v129 + 32);
  v132 = *(_QWORD *)v129 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v132 )
  {
    v142 = *(_BYTE *)(v129 + 16);
    v143 = *(_QWORD *)(v129 + 8) + v181;
    if ( (*(_QWORD *)v129 & 4) != 0 )
    {
      *((_QWORD *)&v211 + 1) = *(_QWORD *)(v129 + 8) + v181;
      LOBYTE(v212[0]) = v142;
      *(_QWORD *)&v211 = v132 | 4;
      HIDWORD(v212[0]) = *(_DWORD *)(v132 + 12);
    }
    else
    {
      *(_QWORD *)&v211 = *(_QWORD *)v129 & 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)&v211 + 1) = v143;
      LOBYTE(v212[0]) = v142;
      v144 = *(_QWORD *)v132;
      if ( *(_BYTE *)(*(_QWORD *)v132 + 8LL) == 16 )
        v144 = **(_QWORD **)(v144 + 16);
      HIDWORD(v212[0]) = *(_DWORD *)(v144 + 8) >> 8;
    }
  }
  else
  {
    v133 = *(_DWORD *)(v129 + 20);
    LODWORD(v212[0]) = 0;
    v211 = 0u;
    HIDWORD(v212[0]) = v133;
  }
  v134 = a3[4];
  LODWORD(v181) = v131;
  v135 = (_BYTE *)sub_1E0A0C0(v134);
  v136 = &v199;
  v137 = *v135 == 0;
  v185.m128i_i64[0] = v182;
  if ( !v137 )
    v136 = &v198;
  v138 = v136->m128i_i64[0];
  v139 = v136->m128i_i64[1];
  v185.m128i_i64[1] = v128 | v185.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v140 = sub_1D2C750(
           a3,
           v174,
           v172,
           (__int64)&v194,
           v138,
           v139,
           v185.m128i_i64[0],
           v185.m128i_i64[1],
           v211,
           v212[0],
           v196,
           v197,
           v130,
           v181,
           (__int64)v179);
  v187 = v141;
  v186 = v140;
  *((_QWORD *)&v147 + 1) = (unsigned int)v141;
  *(_QWORD *)&v147 = v140;
  v35 = (__int64)sub_1D332F0(
                   a3,
                   2,
                   (__int64)&v194,
                   1,
                   0,
                   0,
                   *(double *)v10.m128i_i64,
                   *(double *)v11.m128i_i64,
                   a6,
                   (__int64)v183,
                   v184,
                   v147);
LABEL_73:
  if ( v194 )
    sub_161E7C0((__int64)&v194, v194);
  return v35;
}
