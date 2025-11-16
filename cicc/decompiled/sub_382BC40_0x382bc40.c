// Function: sub_382BC40
// Address: 0x382bc40
//
void __fastcall sub_382BC40(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int8 **a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned __int64 v8; // rbx
  unsigned __int16 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  bool v13; // zf
  unsigned int *v14; // rax
  __int64 *v15; // r8
  __int64 v16; // rax
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int); // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int16 v23; // ax
  __int64 v24; // rax
  __int64 v25; // rax
  _WORD *v26; // r15
  const char *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  const char *v31; // rdi
  size_t v32; // rax
  __int128 v33; // rax
  _QWORD *v34; // r15
  unsigned __int8 *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  unsigned __int64 v38; // rdx
  unsigned __int16 *v39; // rax
  __int64 v40; // r9
  unsigned __int8 v41; // al
  __m128i *v42; // rax
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned int *v45; // rcx
  __int64 v46; // rax
  unsigned int *v47; // r15
  __int64 v48; // rdx
  unsigned int *v49; // rax
  __m128i *v50; // rcx
  unsigned int *v51; // rbx
  unsigned int *v52; // r15
  __m128i *v53; // rsi
  unsigned __int16 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rax
  __int32 v58; // edx
  __m128i si128; // xmm3
  __int64 v60; // rax
  __m128i *v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r15
  __int64 v64; // rax
  __int64 v65; // rdx
  __m128i *v66; // rdx
  unsigned __int64 v67; // rdi
  unsigned int v68; // r8d
  __int64 v69; // rax
  void (***v70)(); // rdi
  void (*v71)(); // rax
  _WORD *v72; // rsi
  __int64 *v73; // rdi
  __m128i *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r15
  __m128i *v77; // r14
  __m128i v78; // rax
  __int64 v79; // rcx
  __int128 v80; // rax
  __int64 v81; // r9
  unsigned __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rsi
  _QWORD *v86; // r15
  __int64 v87; // rax
  __int128 v88; // rax
  __int64 v89; // r9
  unsigned __int8 *v90; // r14
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // r15
  __int64 v94; // r8
  unsigned int *v95; // rcx
  __int128 v96; // rax
  __int64 v97; // r9
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // r14
  unsigned __int64 v101; // r15
  unsigned __int64 *v102; // rax
  unsigned __int64 v103; // r15
  __int64 v104; // r14
  unsigned __int64 v105; // rsi
  __int64 v106; // rdx
  __int64 *v107; // rdi
  unsigned __int16 *v108; // rax
  unsigned int v109; // r10d
  __int64 v110; // rdx
  __int64 v111; // rax
  __int64 v112; // r15
  unsigned int *v113; // rax
  __int64 v114; // rdi
  __int64 v115; // rdx
  __int128 v116; // rax
  _QWORD *v117; // r15
  __int128 v118; // rax
  __int64 v119; // r9
  __int64 v120; // rdx
  __int128 v121; // rax
  __int64 v122; // r9
  __int128 v123; // rax
  __int64 v124; // r9
  unsigned __int8 *v125; // rax
  __int64 v126; // rdx
  __int64 v127; // r15
  unsigned __int8 *v128; // r14
  __int64 v129; // r9
  __int64 v130; // rdx
  unsigned __int8 *v131; // r14
  _QWORD *v132; // rdi
  __int64 v133; // rdx
  unsigned __int64 v134; // r15
  __int64 v135; // r9
  __int64 v136; // rdx
  unsigned __int8 *v137; // r14
  _QWORD *v138; // rdi
  __int64 v139; // rdx
  unsigned __int64 v140; // r15
  __int64 v141; // r9
  __int128 v142; // rax
  __int128 v143; // rax
  int v144; // r9d
  __int128 v145; // rax
  unsigned __int8 *v146; // rax
  __int64 v147; // rdx
  __int64 v148; // r9
  unsigned __int8 *v149; // rax
  _QWORD *v150; // r8
  __int64 v151; // rdx
  __int64 v152; // r10
  unsigned __int8 **v153; // rdx
  __int64 v154; // r9
  __int64 v155; // rdx
  __int128 v156; // [rsp-40h] [rbp-1380h]
  __int128 v157; // [rsp-20h] [rbp-1360h]
  __int128 v158; // [rsp-20h] [rbp-1360h]
  __int128 v159; // [rsp-20h] [rbp-1360h]
  __int128 v160; // [rsp-20h] [rbp-1360h]
  __int128 v161; // [rsp-10h] [rbp-1350h]
  __int128 v162; // [rsp-10h] [rbp-1350h]
  __int128 v163; // [rsp-10h] [rbp-1350h]
  __int64 v164; // [rsp+0h] [rbp-1340h]
  unsigned int v165; // [rsp+0h] [rbp-1340h]
  __int64 v166; // [rsp+8h] [rbp-1338h]
  __m128i *v167; // [rsp+10h] [rbp-1330h]
  _QWORD *v168; // [rsp+10h] [rbp-1330h]
  __int64 v169; // [rsp+18h] [rbp-1328h]
  unsigned __int64 v170; // [rsp+20h] [rbp-1320h]
  __int128 v171; // [rsp+20h] [rbp-1320h]
  unsigned __int64 s1a; // [rsp+30h] [rbp-1310h]
  unsigned __int64 s1; // [rsp+30h] [rbp-1310h]
  __m128i s1b; // [rsp+30h] [rbp-1310h]
  const char *s2; // [rsp+40h] [rbp-1300h]
  void *s2a; // [rsp+40h] [rbp-1300h]
  void *s2b; // [rsp+40h] [rbp-1300h]
  __int128 s2c; // [rsp+40h] [rbp-1300h]
  __int64 **v179; // [rsp+50h] [rbp-12F0h]
  __int64 v180; // [rsp+50h] [rbp-12F0h]
  unsigned int v181; // [rsp+50h] [rbp-12F0h]
  __int128 v182; // [rsp+50h] [rbp-12F0h]
  __int128 v183; // [rsp+50h] [rbp-12F0h]
  __int128 v184; // [rsp+60h] [rbp-12E0h] BYREF
  unsigned int *v185; // [rsp+70h] [rbp-12D0h]
  __int64 v186; // [rsp+78h] [rbp-12C8h]
  __int64 v187; // [rsp+80h] [rbp-12C0h]
  unsigned __int8 **v188; // [rsp+88h] [rbp-12B8h]
  __int64 v189; // [rsp+90h] [rbp-12B0h]
  __int64 v190; // [rsp+98h] [rbp-12A8h]
  __m128i *v191; // [rsp+A0h] [rbp-12A0h]
  __int64 v192; // [rsp+A8h] [rbp-1298h]
  __m128i v193; // [rsp+B0h] [rbp-1290h]
  unsigned __int8 *v194; // [rsp+C0h] [rbp-1280h]
  __int64 v195; // [rsp+C8h] [rbp-1278h]
  unsigned __int8 *v196; // [rsp+D0h] [rbp-1270h]
  __int64 v197; // [rsp+D8h] [rbp-1268h]
  unsigned __int8 *v198; // [rsp+E0h] [rbp-1260h]
  __int64 v199; // [rsp+E8h] [rbp-1258h]
  unsigned __int8 *v200; // [rsp+F0h] [rbp-1250h]
  __int64 v201; // [rsp+F8h] [rbp-1248h]
  unsigned int v202; // [rsp+100h] [rbp-1240h] BYREF
  __int64 v203; // [rsp+108h] [rbp-1238h]
  __int64 v204; // [rsp+110h] [rbp-1230h] BYREF
  int v205; // [rsp+118h] [rbp-1228h]
  unsigned int v206; // [rsp+120h] [rbp-1220h] BYREF
  __int64 v207; // [rsp+128h] [rbp-1218h]
  unsigned __int64 v208; // [rsp+130h] [rbp-1210h] BYREF
  __m128i *v209; // [rsp+138h] [rbp-1208h]
  const __m128i *v210; // [rsp+140h] [rbp-1200h]
  __int64 v211; // [rsp+150h] [rbp-11F0h]
  __int64 v212; // [rsp+158h] [rbp-11E8h]
  __int64 v213; // [rsp+160h] [rbp-11E0h]
  __m128i v214; // [rsp+170h] [rbp-11D0h] BYREF
  __int64 v215; // [rsp+180h] [rbp-11C0h]
  __int64 v216; // [rsp+188h] [rbp-11B8h]
  __int128 v217; // [rsp+190h] [rbp-11B0h] BYREF
  __int64 v218; // [rsp+1A0h] [rbp-11A0h]
  __int64 v219; // [rsp+1A8h] [rbp-1198h]
  __m128i v220; // [rsp+1B0h] [rbp-1190h] BYREF
  __m128i v221; // [rsp+1C0h] [rbp-1180h] BYREF
  __m128i v222; // [rsp+1D0h] [rbp-1170h] BYREF
  __int128 v223; // [rsp+1E0h] [rbp-1160h] BYREF
  unsigned int *v224; // [rsp+1F0h] [rbp-1150h]
  unsigned __int64 v225; // [rsp+1F8h] [rbp-1148h]
  __int64 v226; // [rsp+200h] [rbp-1140h]
  __int64 v227; // [rsp+208h] [rbp-1138h]
  __int64 v228; // [rsp+210h] [rbp-1130h]
  unsigned __int64 v229; // [rsp+218h] [rbp-1128h] BYREF
  __m128i *v230; // [rsp+220h] [rbp-1120h]
  const __m128i *v231; // [rsp+228h] [rbp-1118h]
  __int64 v232; // [rsp+230h] [rbp-1110h]
  __int64 v233; // [rsp+238h] [rbp-1108h] BYREF
  int v234; // [rsp+240h] [rbp-1100h]
  __int64 v235; // [rsp+248h] [rbp-10F8h]
  _BYTE *v236; // [rsp+250h] [rbp-10F0h]
  __int64 v237; // [rsp+258h] [rbp-10E8h]
  _BYTE v238[1792]; // [rsp+260h] [rbp-10E0h] BYREF
  _BYTE *v239; // [rsp+960h] [rbp-9E0h]
  __int64 v240; // [rsp+968h] [rbp-9D8h]
  _BYTE v241[512]; // [rsp+970h] [rbp-9D0h] BYREF
  _BYTE *v242; // [rsp+B70h] [rbp-7D0h]
  __int64 v243; // [rsp+B78h] [rbp-7C8h]
  _BYTE v244[1792]; // [rsp+B80h] [rbp-7C0h] BYREF
  _BYTE *v245; // [rsp+1280h] [rbp-C0h]
  __int64 v246; // [rsp+1288h] [rbp-B8h]
  _BYTE v247[64]; // [rsp+1290h] [rbp-B0h] BYREF
  __int64 v248; // [rsp+12D0h] [rbp-70h]
  __int64 v249; // [rsp+12D8h] [rbp-68h]
  int v250; // [rsp+12E0h] [rbp-60h]
  char v251; // [rsp+1300h] [rbp-40h]

  v8 = a2;
  v9 = *(unsigned __int16 **)(a2 + 48);
  v187 = a3;
  v10 = *(_QWORD *)(a2 + 80);
  v188 = a4;
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v204 = v10;
  LOWORD(v202) = v11;
  v203 = v12;
  if ( v10 )
    sub_B96E90((__int64)&v204, v10, 1);
  v13 = *(_DWORD *)(v8 + 24) == 81;
  v205 = *(_DWORD *)(v8 + 72);
  if ( v13 )
  {
    v102 = *(unsigned __int64 **)(v8 + 40);
    v103 = v102[5];
    v104 = v102[6];
    v105 = *v102;
    v106 = v102[1];
    v214.m128i_i64[0] = 0;
    v214.m128i_i32[2] = 0;
    *(_QWORD *)&v217 = 0;
    DWORD2(v217) = 0;
    v220.m128i_i64[0] = 0;
    v220.m128i_i32[2] = 0;
    *(_QWORD *)&v223 = 0;
    DWORD2(v223) = 0;
    sub_375E510((__int64)a1, v105, v106, (__int64)&v217, (__int64)&v214);
    sub_375E510((__int64)a1, v103, v104, (__int64)&v223, (__int64)&v220);
    v107 = (__int64 *)a1[1];
    v108 = (unsigned __int16 *)(*(_QWORD *)(v217 + 48) + 16LL * DWORD2(v217));
    v109 = *v108;
    v110 = *((_QWORD *)v108 + 1);
    v111 = *(_QWORD *)(v8 + 48);
    v112 = v110;
    v181 = v109;
    LOWORD(v104) = *(_WORD *)(v111 + 16);
    v186 = *(_QWORD *)(v111 + 24);
    *(_QWORD *)&v184 = (unsigned __int16)v104;
    v113 = (unsigned int *)sub_33E5110(v107, v109, v110, (unsigned __int16)v104, v186);
    v114 = a1[1];
    v169 = v115;
    v165 = v181;
    s2b = (void *)v112;
    v185 = v113;
    *(_QWORD *)&v116 = sub_3400BD0(v114, 0, (__int64)&v204, v181, v112, 0, a7, 0);
    v117 = (_QWORD *)a1[1];
    v182 = v116;
    s1b = _mm_loadu_si128(&v220);
    *(_QWORD *)&v118 = sub_33ED040(v117, 0x16u);
    s1b.m128i_i64[0] = sub_340F900(
                         v117,
                         0xD0u,
                         (__int64)&v204,
                         (unsigned __int16)v104,
                         v186,
                         v119,
                         *(_OWORD *)&s1b,
                         v182,
                         v118);
    v171 = (__int128)_mm_loadu_si128(&v214);
    v168 = (_QWORD *)a1[1];
    s1b.m128i_i64[1] = v120;
    *(_QWORD *)&v121 = sub_33ED040(v168, 0x16u);
    *(_QWORD *)&v123 = sub_340F900(v168, 0xD0u, (__int64)&v204, (unsigned __int16)v104, v186, v122, v171, v182, v121);
    v125 = sub_3406EB0(v117, 0xBAu, (__int64)&v204, (unsigned __int16)v104, v186, v124, v123, *(_OWORD *)&s1b);
    v127 = v126;
    v128 = v125;
    *(_QWORD *)&v182 = sub_3411F20((_QWORD *)a1[1], 81, (__int64)&v204, v185, v169, v129, *(_OWORD *)&v214, v223);
    *((_QWORD *)&v182 + 1) = v130;
    *((_QWORD *)&v161 + 1) = 1;
    *(_QWORD *)&v161 = v182;
    *((_QWORD *)&v158 + 1) = v127;
    *(_QWORD *)&v158 = v128;
    v131 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, (__int64)&v204, (unsigned int)v184, v186, v182, v158, v161);
    v132 = (_QWORD *)a1[1];
    v200 = v131;
    v201 = v133;
    v134 = (unsigned int)v133 | v127 & 0xFFFFFFFF00000000LL;
    s1b.m128i_i64[0] = (__int64)sub_3411F20(v132, 81, (__int64)&v204, v185, v169, v135, *(_OWORD *)&v220, v217);
    s1b.m128i_i64[1] = v136;
    *((_QWORD *)&v162 + 1) = 1;
    *(_QWORD *)&v162 = s1b.m128i_i64[0];
    *((_QWORD *)&v159 + 1) = v134;
    *(_QWORD *)&v159 = v131;
    v137 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, (__int64)&v204, (unsigned int)v184, v186, s1b.m128i_i64[0], v159, v162);
    v138 = (_QWORD *)a1[1];
    v198 = v137;
    v199 = v139;
    v140 = (unsigned int)v139 | v134 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v142 = sub_3406EB0(v138, 0x38u, (__int64)&v204, v165, (__int64)s2b, v141, v182, *(_OWORD *)&s1b);
    s1b.m128i_i64[0] = a1[1];
    s2c = v142;
    *(_QWORD *)&v143 = sub_33FAF80(s1b.m128i_i64[0], 214, (__int64)&v204, v202, v203, s1b.m128i_i32[0], a7);
    v183 = v143;
    *(_QWORD *)&v145 = sub_33FAF80(a1[1], 214, (__int64)&v204, v202, v203, v144, a7);
    v146 = sub_3406EB0(s1b.m128i_i64[0], 0x3Au, (__int64)&v204, v202, v203, s1b.m128i_i64[0], v145, v183);
    sub_375BC20(a1, (__int64)v146, v147, v187, (__int64)v188, a7);
    v149 = sub_3411F20((_QWORD *)a1[1], 77, (__int64)&v204, v185, v169, v148, *(_OWORD *)v188, s2c);
    v150 = (_QWORD *)v186;
    v152 = v151;
    v153 = v188;
    v196 = v149;
    v197 = v152;
    *v188 = v149;
    *((_DWORD *)v153 + 2) = v197;
    *((_QWORD *)&v163 + 1) = 1;
    *(_QWORD *)&v163 = v149;
    *((_QWORD *)&v160 + 1) = v140;
    *(_QWORD *)&v160 = v137;
    v194 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, (__int64)&v204, (unsigned int)v184, (__int64)v150, v154, v160, v163);
    v195 = v155;
    sub_3760E70((__int64)a1, v8, 1, (unsigned __int64)v194, (unsigned int)v155 | v140 & 0xFFFFFFFF00000000LL);
LABEL_54:
    if ( v204 )
      sub_B91220((__int64)&v204, v204);
    return;
  }
  v14 = (unsigned int *)sub_3007410((__int64)&v202, *(__int64 **)(a1[1] + 64), v11, (__int64)a4, a5, a6);
  v15 = (__int64 *)*a1;
  v185 = v14;
  v16 = *v15;
  v186 = (__int64)v15;
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v16 + 32);
  v18 = sub_2E79000(*(__int64 **)(a1[1] + 40));
  if ( v17 == sub_2D42F30 )
  {
    v19 = (unsigned int)sub_AE2980(v18, 0)[1];
    v23 = 2;
    if ( (_DWORD)v19 != 1 )
    {
      v23 = 3;
      if ( (_DWORD)v19 != 2 )
      {
        v23 = 4;
        if ( (_DWORD)v19 != 4 )
        {
          v23 = 5;
          if ( (_DWORD)v19 != 8 )
          {
            v23 = 6;
            if ( (_DWORD)v19 != 16 )
            {
              v23 = 7;
              if ( (_DWORD)v19 != 32 )
              {
                v23 = 8;
                if ( (_DWORD)v19 != 64 )
                  v23 = 9 * ((_DWORD)v19 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v23 = v17(v186, v18, 0);
  }
  LOWORD(v206) = v23;
  v24 = a1[1];
  v207 = 0;
  v179 = (__int64 **)sub_3007410((__int64)&v206, *(__int64 **)(v24 + 64), v19, v20, v21, v22);
  switch ( (_WORD)v202 )
  {
    case 7:
      v25 = 17;
      break;
    case 8:
      v25 = 18;
      break;
    case 9:
      v25 = 19;
      break;
    default:
      v26 = (_WORD *)*a1;
      goto LABEL_53;
  }
  v26 = (_WORD *)*a1;
  v186 = v25;
  if ( !*(_QWORD *)&v26[4 * v25 + 262644] )
    goto LABEL_53;
  v27 = sub_2E791E0(*(__int64 **)(a1[1] + 40));
  v26 = (_WORD *)*a1;
  s2 = v27;
  v31 = *(const char **)(*a1 + 8 * v186 + 525288);
  if ( v31 )
  {
    *(_QWORD *)&v184 = v28;
    v32 = strlen(v31);
    if ( v32 != (_QWORD)v184 || v32 && memcmp(v31, s2, v32) )
      goto LABEL_18;
    goto LABEL_53;
  }
  if ( !v28 )
  {
LABEL_53:
    v84 = *(_QWORD *)(v8 + 40);
    v85 = a1[1];
    v220.m128i_i64[0] = 0;
    v220.m128i_i32[2] = 0;
    *(_QWORD *)&v223 = 0;
    DWORD2(v223) = 0;
    sub_3495B70(
      v26,
      v85,
      (__int64)&v204,
      1,
      *(_QWORD *)v84,
      *(_QWORD *)(v84 + 8),
      a7,
      *(_OWORD *)(v84 + 40),
      &v220,
      &v223);
    v86 = (_QWORD *)a1[1];
    v87 = sub_32844A0((unsigned __int16 *)&v202, v85);
    *(_QWORD *)&v88 = sub_3400BD0((__int64)v86, v87 - 1, (__int64)&v204, v202, v203, 0, a7, 0);
    v90 = sub_3406EB0(v86, 0xBFu, (__int64)&v204, v202, v203, v89, *(_OWORD *)&v220, v88);
    v91 = *(_QWORD *)(v8 + 48);
    v93 = v92;
    v186 = a1[1];
    v94 = *(_QWORD *)(v91 + 24);
    v95 = (unsigned int *)*(unsigned __int16 *)(v91 + 16);
    v184 = v223;
    v180 = v94;
    v185 = v95;
    *(_QWORD *)&v96 = sub_33ED040((_QWORD *)v186, 0x16u);
    *((_QWORD *)&v157 + 1) = v93;
    *(_QWORD *)&v157 = v90;
    v98 = sub_340F900((_QWORD *)v186, 0xD0u, (__int64)&v204, (unsigned int)v185, v180, v97, v184, v157, v96);
    v100 = v99;
    v101 = v98;
    sub_375BC20(a1, v220.m128i_i64[0], v220.m128i_i64[1], v187, (__int64)v188, a7);
    sub_3760E70((__int64)a1, v8, 1, v101, v100);
    goto LABEL_54;
  }
LABEL_18:
  *(_QWORD *)&v33 = sub_33EDFE0(a1[1], v206, v207, 1, v29, v30);
  v34 = (_QWORD *)a1[1];
  v184 = v33;
  v223 = 0u;
  v224 = 0;
  v225 = 0;
  v35 = sub_3400BD0((__int64)v34, 0, (__int64)&v204, v206, v207, 0, a7, 0);
  v36 = a1[1];
  v37 = (__int64)v35;
  v220 = 0u;
  v221.m128i_i32[0] = 0;
  v221.m128i_i8[4] = 0;
  v39 = (unsigned __int16 *)(*((_QWORD *)v35 + 6) + 16LL * (unsigned int)v38);
  s1a = v38;
  v170 = v36 + 288;
  s2a = (void *)v37;
  v41 = sub_33CC4A0((__int64)v34, *v39, *((_QWORD *)v39 + 1), v36, v37, v40);
  v42 = sub_33F4560(
          v34,
          v170,
          0,
          (__int64)&v204,
          (unsigned __int64)s2a,
          s1a,
          v184,
          *((unsigned __int64 *)&v184 + 1),
          *(_OWORD *)&v220,
          v221.m128i_i64[0],
          v41,
          0,
          (__int64)&v223);
  v45 = *(unsigned int **)(v8 + 40);
  v208 = 0;
  v167 = v42;
  v46 = *(unsigned int *)(v8 + 64);
  v47 = v45;
  v166 = v48;
  v209 = 0;
  v210 = 0;
  v49 = &v45[10 * v46];
  v50 = &v220;
  v220 = 0u;
  v221 = 0u;
  v222 = 0u;
  if ( v47 != v49 )
  {
    s1 = v8;
    v51 = v47;
    v52 = v49;
    do
    {
      while ( 1 )
      {
        v54 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v51 + 48LL) + 16LL * v51[2]);
        v55 = *v54;
        *((_QWORD *)&v223 + 1) = *((_QWORD *)v54 + 1);
        v56 = a1[1];
        LOWORD(v223) = v55;
        v57 = sub_3007410((__int64)&v223, *(__int64 **)(v56 + 64), v55, (__int64)v50, v43, v44);
        v53 = v209;
        v220.m128i_i64[1] = *(_QWORD *)v51;
        v58 = v51[2];
        v221.m128i_i64[1] = v57;
        v221.m128i_i32[0] = v58;
        v222.m128i_i8[0] = v222.m128i_i8[0] & 0xFC | 1;
        if ( v209 != v210 )
          break;
        v51 += 10;
        sub_332CDC0(&v208, v209, &v220);
        if ( v52 == v51 )
          goto LABEL_25;
      }
      if ( v209 )
      {
        a7 = _mm_loadu_si128(&v220);
        *v209 = a7;
        v53[1] = _mm_loadu_si128(&v221);
        v53[2] = _mm_loadu_si128(&v222);
        v53 = v209;
      }
      v51 += 10;
      v209 = v53 + 3;
    }
    while ( v52 != v51 );
LABEL_25:
    v8 = s1;
  }
  si128 = _mm_load_si128((const __m128i *)&v184);
  v220.m128i_i64[1] = v184;
  v193 = si128;
  v221.m128i_i32[0] = si128.m128i_i32[2];
  v60 = sub_BCE3C0(*v179, 0);
  v61 = v209;
  v221.m128i_i64[1] = v60;
  v222.m128i_i8[0] = v222.m128i_i8[0] & 0xFC | 1;
  if ( v209 == v210 )
  {
    sub_332CDC0(&v208, v209, &v220);
  }
  else
  {
    if ( v209 )
    {
      *v209 = _mm_loadu_si128(&v220);
      v61[1] = _mm_loadu_si128(&v221);
      v61[2] = _mm_loadu_si128(&v222);
      v61 = v209;
    }
    v209 = v61 + 3;
  }
  v62 = sub_33EED90(a1[1], *(const char **)(*a1 + 8 * v186 + 525288), v206, v207);
  v248 = 0;
  v63 = v62;
  v64 = a1[1];
  v225 = 0xFFFFFFFF00000020LL;
  v232 = v64;
  v236 = v238;
  v237 = 0x2000000000LL;
  v240 = 0x2000000000LL;
  v243 = 0x2000000000LL;
  v245 = v247;
  v246 = 0x400000000LL;
  v239 = v241;
  v164 = v65;
  v242 = v244;
  v223 = 0u;
  v224 = 0;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  v249 = 0;
  v250 = 0;
  v251 = 0;
  sub_9C6650(&v233);
  v233 = v204;
  if ( v204 )
    sub_3813810(&v233);
  v66 = v209;
  v67 = v229;
  v209 = 0;
  v234 = v205;
  v191 = v167;
  *(_QWORD *)&v223 = v167;
  v192 = v166;
  DWORD2(v223) = v166;
  v68 = *(_DWORD *)(*a1 + 4 * v186 + 531128);
  v189 = v63;
  v227 = v63;
  v224 = v185;
  v190 = v164;
  LODWORD(v226) = v68;
  LODWORD(v228) = v164;
  v229 = v208;
  v69 = (__int64)((__int64)v66->m128i_i64 - v208) >> 4;
  v230 = v66;
  v208 = 0;
  HIDWORD(v225) = -1431655765 * v69;
  v231 = v210;
  v210 = 0;
  if ( v67 )
  {
    LODWORD(v186) = v68;
    j_j___libc_free_0(v67);
    v68 = v186;
  }
  v70 = *(void (****)())(v232 + 16);
  v71 = **v70;
  if ( v71 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v71)(
      v70,
      *(_QWORD *)(v232 + 40),
      v68,
      &v229);
  v72 = (_WORD *)*a1;
  LOBYTE(v225) = v225 | 1;
  sub_3377410((__int64)&v214, v72, (__int64)&v223);
  sub_375BC20(a1, v214.m128i_i64[0], v214.m128i_i64[1], v187, (__int64)v188, a7);
  v73 = (__int64 *)a1[1];
  v212 = 0;
  LODWORD(v213) = 0;
  BYTE4(v213) = 0;
  v217 = 0u;
  v218 = 0;
  v219 = 0;
  v211 = 0;
  v74 = sub_33F1F00(
          v73,
          v206,
          v207,
          (__int64)&v204,
          v215,
          v216,
          v184,
          *((__int64 *)&v184 + 1),
          0,
          v213,
          0,
          0,
          (__int64)&v217,
          0);
  v76 = v75;
  v188 = (unsigned __int8 **)a1[1];
  v77 = v74;
  v78.m128i_i64[0] = (__int64)sub_3400BD0((__int64)v188, 0, (__int64)&v204, v206, v207, 0, a7, 0);
  v184 = (__int128)v78;
  v78.m128i_i64[0] = *(_QWORD *)(v8 + 48);
  v79 = *(unsigned __int16 *)(v78.m128i_i64[0] + 16);
  v186 = *(_QWORD *)(v78.m128i_i64[0] + 24);
  v187 = v79;
  *(_QWORD *)&v80 = sub_33ED040(v188, 0x16u);
  *((_QWORD *)&v156 + 1) = v76;
  *(_QWORD *)&v156 = v77;
  v82 = sub_340F900(v188, 0xD0u, (__int64)&v204, v187, v186, v81, v156, v184, v80);
  sub_3760E70((__int64)a1, v8, 1, v82, v83);
  if ( v245 != v247 )
    _libc_free((unsigned __int64)v245);
  if ( v242 != v244 )
    _libc_free((unsigned __int64)v242);
  if ( v239 != v241 )
    _libc_free((unsigned __int64)v239);
  if ( v236 != v238 )
    _libc_free((unsigned __int64)v236);
  sub_9C6650(&v233);
  if ( v229 )
    j_j___libc_free_0(v229);
  if ( v208 )
    j_j___libc_free_0(v208);
  sub_9C6650(&v204);
}
