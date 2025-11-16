// Function: sub_1D3F1E0
// Address: 0x1d3f1e0
//
__int64 __fastcall sub_1D3F1E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int128 a9,
        __int128 a10,
        unsigned int a11,
        unsigned __int8 a12,
        char a13,
        __int128 a14,
        __int64 a15,
        __int128 a16,
        __int64 a17)
{
  int v19; // eax
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i v27; // xmm4
  __m128i *v28; // rsi
  __m128i *v29; // rax
  __m128i v30; // xmm5
  __m128i *v31; // rsi
  __m128i *v32; // rax
  __m128i v33; // xmm6
  __m128i *v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // edx
  unsigned __int8 v39; // al
  __int64 v40; // rdx
  __int64 v41; // rax
  char v42; // si
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rdi
  unsigned int v47; // r15d
  __int64 v48; // rax
  __m128i *v49; // rsi
  __m128i *v50; // rax
  void (***v51)(); // rdi
  void (*v52)(); // rax
  __int64 v53; // rsi
  __int64 v54; // rbx
  __int64 v55; // r12
  unsigned int v56; // r13d
  __int64 v58; // rax
  unsigned __int64 v59; // r10
  __int64 v60; // r13
  __m128i v61; // xmm2
  bool v62; // zf
  __int64 *v63; // r12
  __int64 v64; // r12
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 *v67; // rax
  char v68; // al
  unsigned __int64 v69; // r10
  char v70; // r14
  int v71; // eax
  unsigned int v72; // eax
  unsigned int v73; // r13d
  unsigned int v74; // eax
  unsigned int v75; // eax
  unsigned __int64 v76; // r10
  __m128i v77; // xmm3
  __m128i v78; // xmm2
  unsigned int v79; // eax
  __int64 v80; // rsi
  char v81; // al
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r14
  __int64 v87; // r13
  __int64 v88; // rdx
  __m128i *v89; // rcx
  unsigned int v90; // r12d
  __int64 v91; // r12
  unsigned __int64 v92; // rdx
  char v93; // al
  unsigned __int16 v94; // r9
  unsigned __int64 v95; // rdx
  __int64 *v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rdi
  __int64 v101; // rax
  __int64 *v102; // rax
  __int64 v103; // rax
  __int64 *v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  __m128i *v108; // rdx
  __int64 *v109; // rax
  __int64 v110; // r8
  __int64 v111; // r9
  _QWORD *v112; // rsi
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r13
  __int64 v116; // r14
  unsigned int v117; // r12d
  __int64 v118; // r12
  unsigned __int64 v119; // rdx
  __int64 *v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rdi
  __int64 v125; // rax
  __int64 *v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // rax
  __int128 v130; // [rsp-10h] [rbp-1280h]
  __int128 v131; // [rsp-10h] [rbp-1280h]
  _QWORD *v132; // [rsp-8h] [rbp-1278h]
  __int64 v133; // [rsp+8h] [rbp-1268h]
  unsigned int v134; // [rsp+10h] [rbp-1260h]
  __int64 v135; // [rsp+38h] [rbp-1238h]
  unsigned __int64 v136; // [rsp+40h] [rbp-1230h]
  unsigned int v137; // [rsp+40h] [rbp-1230h]
  int v138; // [rsp+4Ch] [rbp-1224h]
  unsigned __int16 v139; // [rsp+4Ch] [rbp-1224h]
  __int64 v140; // [rsp+50h] [rbp-1220h]
  __int64 v141; // [rsp+50h] [rbp-1220h]
  __int64 v142; // [rsp+58h] [rbp-1218h]
  unsigned __int64 v143; // [rsp+60h] [rbp-1210h]
  __int128 v144; // [rsp+68h] [rbp-1208h]
  __int64 v145; // [rsp+70h] [rbp-1200h]
  __int64 v146; // [rsp+78h] [rbp-11F8h]
  __int64 v147; // [rsp+80h] [rbp-11F0h]
  unsigned __int16 v148; // [rsp+80h] [rbp-11F0h]
  unsigned __int64 v149; // [rsp+80h] [rbp-11F0h]
  __int64 v150; // [rsp+80h] [rbp-11F0h]
  unsigned __int64 v151; // [rsp+88h] [rbp-11E8h]
  int v152; // [rsp+88h] [rbp-11E8h]
  unsigned __int64 v153; // [rsp+88h] [rbp-11E8h]
  unsigned int v154; // [rsp+90h] [rbp-11E0h]
  char v155; // [rsp+94h] [rbp-11DCh]
  unsigned __int8 v156; // [rsp+98h] [rbp-11D8h]
  __m128i v157; // [rsp+A0h] [rbp-11D0h] BYREF
  __int64 *v158; // [rsp+B0h] [rbp-11C0h]
  __int64 v159; // [rsp+B8h] [rbp-11B8h]
  __int64 v160; // [rsp+C0h] [rbp-11B0h]
  __int64 v161; // [rsp+C8h] [rbp-11A8h]
  __int64 v162; // [rsp+D0h] [rbp-11A0h]
  __int64 v163; // [rsp+D8h] [rbp-1198h]
  __m128i v164; // [rsp+E0h] [rbp-1190h]
  __m128i v165; // [rsp+F0h] [rbp-1180h]
  __m128i v166; // [rsp+100h] [rbp-1170h]
  __m128i v167; // [rsp+110h] [rbp-1160h]
  __m128i v168; // [rsp+120h] [rbp-1150h]
  __int64 v169; // [rsp+130h] [rbp-1140h]
  __int64 v170; // [rsp+138h] [rbp-1138h]
  __int64 *v171; // [rsp+140h] [rbp-1130h]
  __int64 v172; // [rsp+148h] [rbp-1128h]
  __int64 v173; // [rsp+150h] [rbp-1120h]
  __int64 v174; // [rsp+158h] [rbp-1118h]
  __m128 v175; // [rsp+160h] [rbp-1110h] BYREF
  __m128i v176; // [rsp+170h] [rbp-1100h] BYREF
  __int64 v177; // [rsp+180h] [rbp-10F0h]
  __m128i v178; // [rsp+190h] [rbp-10E0h] BYREF
  __int64 v179; // [rsp+1A0h] [rbp-10D0h]
  __int64 v180; // [rsp+1B0h] [rbp-10C0h] BYREF
  __int64 v181; // [rsp+1B8h] [rbp-10B8h]
  __int64 v182; // [rsp+1C0h] [rbp-10B0h]
  __int128 v183; // [rsp+1D0h] [rbp-10A0h]
  __int64 v184; // [rsp+1E0h] [rbp-1090h]
  __m128i v185; // [rsp+1F0h] [rbp-1080h] BYREF
  __m128i *v186; // [rsp+200h] [rbp-1070h]
  __int128 v187; // [rsp+210h] [rbp-1060h] BYREF
  __int64 v188; // [rsp+220h] [rbp-1050h]
  __int64 *v189; // [rsp+230h] [rbp-1040h] BYREF
  __int64 v190; // [rsp+238h] [rbp-1038h]
  __int64 v191; // [rsp+240h] [rbp-1030h] BYREF
  __m128i v192; // [rsp+2C0h] [rbp-FB0h] BYREF
  __m128i v193; // [rsp+2D0h] [rbp-FA0h] BYREF
  __int64 v194; // [rsp+2E0h] [rbp-F90h]
  __int64 *v195; // [rsp+350h] [rbp-F20h] BYREF
  __int64 v196; // [rsp+358h] [rbp-F18h]
  __int64 v197; // [rsp+360h] [rbp-F10h] BYREF
  unsigned __int64 v198; // [rsp+368h] [rbp-F08h]
  __int64 v199; // [rsp+370h] [rbp-F00h]
  __int64 v200; // [rsp+378h] [rbp-EF8h]
  __int64 v201; // [rsp+380h] [rbp-EF0h]
  __m128i v202; // [rsp+388h] [rbp-EE8h] BYREF
  __m128i *v203; // [rsp+398h] [rbp-ED8h]
  __int64 *v204; // [rsp+3A0h] [rbp-ED0h]
  __int64 v205; // [rsp+3A8h] [rbp-EC8h] BYREF
  int v206; // [rsp+3B0h] [rbp-EC0h]
  __int64 v207; // [rsp+3B8h] [rbp-EB8h]
  _BYTE *v208; // [rsp+3C0h] [rbp-EB0h]
  __int64 v209; // [rsp+3C8h] [rbp-EA8h]
  _BYTE v210[1536]; // [rsp+3D0h] [rbp-EA0h] BYREF
  _BYTE *v211; // [rsp+9D0h] [rbp-8A0h]
  __int64 v212; // [rsp+9D8h] [rbp-898h]
  _BYTE v213[512]; // [rsp+9E0h] [rbp-890h] BYREF
  _BYTE *v214; // [rsp+BE0h] [rbp-690h]
  __int64 v215; // [rsp+BE8h] [rbp-688h]
  _BYTE v216[1536]; // [rsp+BF0h] [rbp-680h] BYREF
  _BYTE *v217; // [rsp+11F0h] [rbp-80h]
  __int64 v218; // [rsp+11F8h] [rbp-78h]
  _BYTE v219[112]; // [rsp+1200h] [rbp-70h] BYREF

  v159 = a3;
  v158 = (__int64 *)a2;
  v156 = a12;
  v157.m128i_i64[0] = a5;
  v155 = a13;
  v19 = *(unsigned __int16 *)(a10 + 24);
  v157.m128i_i64[1] = a6;
  if ( v19 != 32 && v19 != 10 )
    goto LABEL_3;
  v55 = *(_QWORD *)(a10 + 88);
  v56 = *(_DWORD *)(v55 + 32);
  if ( v56 <= 0x40 )
  {
    if ( !*(_QWORD *)(v55 + 24) )
      return (__int64)v158;
    v59 = *(_QWORD *)(v55 + 24);
  }
  else
  {
    if ( v56 == (unsigned int)sub_16A57B0(v55 + 24) )
      return (__int64)v158;
    v59 = **(_QWORD **)(v55 + 24);
  }
  v60 = a9;
  v61 = _mm_loadu_si128((const __m128i *)&a14);
  v146 = v157.m128i_i64[0];
  v62 = *(_WORD *)(a9 + 24) == 48;
  v143 = v157.m128i_u64[1];
  v144 = a9;
  v177 = a15;
  v176 = v61;
  v179 = a17;
  v178 = _mm_loadu_si128((const __m128i *)&a16);
  if ( v62 )
  {
    v63 = v158;
    goto LABEL_47;
  }
  v64 = a1[2];
  v151 = v59;
  v65 = sub_1E0A0C0(a1[4]);
  v181 = 0;
  v142 = v65;
  v66 = a1[6];
  v182 = 0;
  v140 = v66;
  v67 = (__int64 *)a1[4];
  v180 = 0;
  v147 = v67[7];
  v68 = sub_1D139C0(*v67, v67[1]);
  v69 = v151;
  v70 = v68;
  v71 = *(unsigned __int16 *)(v157.m128i_i64[0] + 24);
  if ( v71 != 36 && v71 != 14 )
  {
    v72 = sub_1D1FC50((__int64)a1, a9);
    if ( a11 >= v72 )
      v72 = a11;
    v154 = v72;
    if ( v70 )
      v73 = *(_DWORD *)(v64 + 81532);
    else
      v73 = *(_DWORD *)(v64 + 81528);
    sub_1E340A0(&v178);
    v75 = sub_1E340A0(&v176);
    v76 = v151;
    goto LABEL_82;
  }
  v152 = *(_DWORD *)(v157.m128i_i64[0] + 84);
  if ( v152 < 0 )
  {
    v136 = v69;
    v138 = -*(_DWORD *)(v147 + 32);
    v79 = sub_1D1FC50((__int64)a1, v60);
    if ( a11 >= v79 )
      v79 = a11;
    v73 = *(_DWORD *)(v64 + 81532);
    if ( !v70 )
      v73 = *(_DWORD *)(v64 + 81528);
    v154 = v79;
    sub_1E340A0(&v178);
    v75 = sub_1E340A0(&v176);
    v76 = v136;
    if ( v152 >= v138 )
    {
LABEL_82:
      v80 = v73;
      v81 = sub_1D26C30((__int64)&v180, v73, v76, a11, v154, 0, 0, 0, 0, v75, a1[4], v64);
      v137 = a11;
      if ( v81 )
      {
LABEL_83:
        v192.m128i_i64[0] = (__int64)&v193;
        v139 = 4 * (v156 != 0);
        v189 = &v191;
        v190 = 0x800000000LL;
        v192.m128i_i64[1] = 0x800000000LL;
        v196 = 0x800000000LL;
        v84 = v180;
        v195 = &v197;
        v85 = (v181 - v180) >> 4;
        v134 = v85;
        if ( (_DWORD)v85 )
        {
          v86 = 0;
          v87 = 0;
          v88 = 16LL * (unsigned int)(v85 - 1);
          v89 = &v185;
          v135 = v88;
          while ( 1 )
          {
            a7 = _mm_loadu_si128((const __m128i *)(v84 + v86));
            v185 = a7;
            v90 = a7.m128i_i8[0] ? sub_1D13440(a7.m128i_i8[0]) : sub_1F58D40(&v185, v80, v88, v89, v82, v83);
            v91 = v90 >> 3;
            v92 = v178.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v178.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              if ( (v178.m128i_i8[0] & 4) != 0 )
              {
                *((_QWORD *)&v187 + 1) = v87 + v178.m128i_i64[1];
                LOBYTE(v188) = v179;
                *(_QWORD *)&v187 = v92 | 4;
                HIDWORD(v188) = *(_DWORD *)(v92 + 12);
              }
              else
              {
                *(_QWORD *)&v187 = v178.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                *((_QWORD *)&v187 + 1) = v87 + v178.m128i_i64[1];
                LOBYTE(v188) = v179;
                v107 = *(_QWORD *)v92;
                if ( *(_BYTE *)(*(_QWORD *)v92 + 8LL) == 16 )
                  v107 = **(_QWORD **)(v107 + 16);
                HIDWORD(v188) = *(_DWORD *)(v107 + 8) >> 8;
              }
            }
            else
            {
              LODWORD(v188) = 0;
              v187 = 0u;
              HIDWORD(v188) = HIDWORD(v179);
            }
            v93 = sub_1E340B0(&v187, (unsigned int)v91, v140, v142);
            v187 = 0u;
            v188 = 0;
            v94 = v139 | 0x10;
            if ( !v93 )
              v94 = 4 * (v156 != 0);
            v95 = v178.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v178.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              if ( (v178.m128i_i8[0] & 4) != 0 )
              {
                *((_QWORD *)&v183 + 1) = v87 + v178.m128i_i64[1];
                LOBYTE(v184) = v179;
                *(_QWORD *)&v183 = v95 | 4;
                HIDWORD(v184) = *(_DWORD *)(v95 + 12);
              }
              else
              {
                v106 = *(_QWORD *)v95;
                *(_QWORD *)&v183 = v178.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                *((_QWORD *)&v183 + 1) = v87 + v178.m128i_i64[1];
                v62 = *(_BYTE *)(v106 + 8) == 16;
                LOBYTE(v184) = v179;
                if ( v62 )
                  v106 = **(_QWORD **)(v106 + 16);
                HIDWORD(v184) = *(_DWORD *)(v106 + 8) >> 8;
              }
            }
            else
            {
              LODWORD(v184) = 0;
              v183 = 0u;
              HIDWORD(v184) = HIDWORD(v179);
            }
            v148 = v94;
            v96 = sub_1D3D250(a1, v144, *((unsigned __int64 *)&v144 + 1), v87, a4, a7, *(double *)a8.m128i_i64, v61);
            v80 = v185.m128i_u32[0];
            v98 = sub_1D2B730(
                    a1,
                    v185.m128i_u32[0],
                    v185.m128i_i64[1],
                    a4,
                    (__int64)v158,
                    v159,
                    (__int64)v96,
                    v97,
                    v183,
                    v184,
                    v154,
                    v148,
                    (__int64)&v187,
                    0);
            v100 = v99;
            v173 = v98;
            v88 = v98;
            v82 = v98;
            v174 = v100;
            v101 = (unsigned int)v190;
            v83 = (unsigned int)v100;
            if ( (unsigned int)v190 >= HIDWORD(v190) )
            {
              v80 = (__int64)&v191;
              v133 = v88;
              sub_16CD150((__int64)&v189, &v191, 0, 16, v82, v100);
              v101 = (unsigned int)v190;
              v82 = v133;
              v83 = (unsigned int)v100;
            }
            v102 = &v189[2 * v101];
            *v102 = v82;
            v102[1] = v83;
            v103 = v192.m128i_u32[2];
            LODWORD(v190) = v190 + 1;
            if ( v192.m128i_i32[2] >= (unsigned __int32)v192.m128i_i32[3] )
            {
              v80 = (__int64)&v193;
              v150 = v82;
              sub_16CD150((__int64)&v192, &v193, 0, 16, v82, v83);
              v103 = v192.m128i_u32[2];
              v82 = v150;
            }
            v104 = (__int64 *)(v192.m128i_i64[0] + 16 * v103);
            v87 += v91;
            *v104 = v82;
            v104[1] = 1;
            v105 = (unsigned int)++v192.m128i_i32[2];
            if ( v135 == v86 )
              break;
            v84 = v180;
            v86 += 16;
          }
          v108 = (__m128i *)v192.m128i_i64[0];
        }
        else
        {
          v108 = &v193;
          v105 = 0;
        }
        *((_QWORD *)&v130 + 1) = v105;
        *(_QWORD *)&v130 = v108;
        v109 = sub_1D359D0(a1, 2, a4, 1, 0, 0, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, v61, v130);
        v112 = v132;
        LODWORD(v196) = 0;
        v145 = (__int64)v109;
        v172 = v113;
        v113 = (unsigned int)v113;
        v171 = v109;
        v149 = (unsigned int)v113 | v159 & 0xFFFFFFFF00000000LL;
        if ( v134 )
        {
          v114 = v139;
          v115 = 0;
          v116 = 0;
          do
          {
            a8 = _mm_loadu_si128((const __m128i *)(v180 + v115 * 8));
            v175 = (__m128)a8;
            if ( a8.m128i_i8[0] )
              v117 = sub_1D13440(a8.m128i_i8[0]);
            else
              v117 = sub_1F58D40(&v175, v112, v113, v114, v110, v111);
            v118 = v117 >> 3;
            v185 = 0u;
            v186 = 0;
            v119 = v176.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v176.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              if ( (v176.m128i_i8[0] & 4) != 0 )
              {
                *((_QWORD *)&v187 + 1) = v116 + v176.m128i_i64[1];
                LOBYTE(v188) = v177;
                *(_QWORD *)&v187 = v119 | 4;
                HIDWORD(v188) = *(_DWORD *)(v119 + 12);
              }
              else
              {
                *(_QWORD *)&v187 = v176.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                *((_QWORD *)&v187 + 1) = v116 + v176.m128i_i64[1];
                LOBYTE(v188) = v177;
                v128 = *(_QWORD *)v119;
                if ( *(_BYTE *)(*(_QWORD *)v119 + 8LL) == 16 )
                  v128 = **(_QWORD **)(v128 + 16);
                HIDWORD(v188) = *(_DWORD *)(v128 + 8) >> 8;
              }
            }
            else
            {
              LODWORD(v188) = 0;
              v187 = 0u;
              HIDWORD(v188) = HIDWORD(v177);
            }
            v120 = sub_1D3D250(a1, v146, v143, v116, a4, a7, *(double *)a8.m128i_i64, v61);
            v112 = (_QWORD *)v145;
            v122 = sub_1D2BF40(
                     a1,
                     v145,
                     v149,
                     a4,
                     v189[v115],
                     v189[v115 + 1],
                     (__int64)v120,
                     v121,
                     v187,
                     v188,
                     v137,
                     v139,
                     (__int64)&v185);
            v124 = v123;
            v169 = v122;
            v113 = v122;
            v110 = v122;
            v170 = v124;
            v125 = (unsigned int)v196;
            v111 = (unsigned int)v124;
            if ( (unsigned int)v196 >= HIDWORD(v196) )
            {
              v112 = &v197;
              v141 = v113;
              sub_16CD150((__int64)&v195, &v197, 0, 16, v110, v124);
              v125 = (unsigned int)v196;
              v110 = v141;
              v111 = (unsigned int)v124;
            }
            v126 = &v195[2 * v125];
            v116 += v118;
            v115 += 2;
            *v126 = v110;
            v126[1] = v111;
            v127 = (unsigned int)(v196 + 1);
            LODWORD(v196) = v196 + 1;
          }
          while ( v115 != 2LL * v134 );
        }
        else
        {
          v127 = 0;
        }
        *((_QWORD *)&v131 + 1) = v127;
        *(_QWORD *)&v131 = v195;
        v63 = sub_1D359D0(a1, 2, a4, 1, 0, 0, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, v61, v131);
        if ( v195 != &v197 )
          _libc_free((unsigned __int64)v195);
        if ( (__m128i *)v192.m128i_i64[0] != &v193 )
          _libc_free(v192.m128i_u64[0]);
        if ( v189 != &v191 )
          _libc_free((unsigned __int64)v189);
        goto LABEL_68;
      }
      goto LABEL_67;
    }
  }
  else
  {
    v153 = v69;
    v74 = sub_1D1FC50((__int64)a1, v60);
    if ( a11 >= v74 )
      v74 = a11;
    v154 = v74;
    if ( v70 )
      v73 = *(_DWORD *)(v64 + 81532);
    else
      v73 = *(_DWORD *)(v64 + 81528);
    sub_1E340A0(&v178);
    v75 = sub_1E340A0(&v176);
    v76 = v153;
  }
  if ( (unsigned __int8)sub_1D26C30((__int64)&v180, v73, v76, 0, v154, 0, 0, 0, 0, v75, a1[4], v64) )
  {
    v80 = sub_1F58E60(v180, v140);
    v137 = sub_15A9FE0(v142, v80);
    if ( a11 >= v137 )
    {
      v137 = a11;
    }
    else
    {
      v80 = v147;
      v129 = *(_QWORD *)(v147 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v147 + 32) + *(_DWORD *)(v157.m128i_i64[0] + 84));
      if ( v137 > *(_DWORD *)(v129 + 16) )
      {
        *(_DWORD *)(v129 + 16) = v137;
        v80 = v137;
        sub_1E08740(v147, v137);
      }
    }
    goto LABEL_83;
  }
LABEL_67:
  v63 = 0;
LABEL_68:
  if ( v180 )
    j_j___libc_free_0(v180, v182 - v180);
LABEL_47:
  if ( v63 )
    return (__int64)v63;
LABEL_3:
  v20 = a1[1];
  if ( v20 )
  {
    v21 = *(__int64 (**)())(*(_QWORD *)v20 + 24LL);
    if ( v21 != sub_1D12E60 )
    {
      v58 = ((__int64 (__fastcall *)(__int64, __int64 *, __int64, __int64 *, __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD, __int64))v21)(
              v20,
              a1,
              a4,
              v158,
              v159,
              a11,
              v157.m128i_i64[0],
              v157.m128i_i64[1],
              a9,
              *((_QWORD *)&a9 + 1),
              a10,
              *((_QWORD *)&a10 + 1),
              v156,
              a14,
              *((_QWORD *)&a14 + 1),
              a15,
              a16,
              *((_QWORD *)&a16 + 1),
              a17);
      if ( v58 )
        return v58;
    }
  }
  v22 = sub_1E340A0(&a14);
  sub_1D13830(a1[2], v22);
  v23 = sub_1E340A0(&a16);
  sub_1D13830(a1[2], v23);
  v24 = a1[4];
  v185 = 0u;
  v186 = 0;
  v192 = 0u;
  v193 = 0u;
  LODWORD(v194) = 0;
  v25 = sub_1E0A0C0(v24);
  v26 = sub_15A9620(v25, a1[6], 0);
  v27 = _mm_load_si128(&v157);
  v28 = (__m128i *)v185.m128i_i64[1];
  v193.m128i_i64[1] = v26;
  v168 = v27;
  v192.m128i_i64[1] = v157.m128i_i64[0];
  v193.m128i_i32[0] = v27.m128i_i32[2];
  v29 = v186;
  if ( (__m128i *)v185.m128i_i64[1] != v186 )
  {
    if ( v185.m128i_i64[1] )
    {
      *(__m128i *)v185.m128i_i64[1] = _mm_load_si128(&v192);
      v28[1] = _mm_load_si128(&v193);
      v28[2].m128i_i64[0] = v194;
      v28 = (__m128i *)v185.m128i_i64[1];
      v29 = v186;
    }
    v30 = _mm_loadu_si128((const __m128i *)&a9);
    v31 = (__m128i *)((char *)v28 + 40);
    v185.m128i_i64[1] = (__int64)v31;
    v192.m128i_i64[1] = a9;
    v167 = v30;
    v193.m128i_i32[0] = v30.m128i_i32[2];
    if ( v31 != v29 )
      goto LABEL_9;
LABEL_71:
    sub_1D27190((const __m128i **)&v185, v31, &v192);
    v77 = _mm_loadu_si128((const __m128i *)&a10);
    v34 = (__m128i *)v185.m128i_i64[1];
    v192.m128i_i64[1] = a10;
    v164 = v77;
    v193.m128i_i32[0] = v77.m128i_i32[2];
    if ( (__m128i *)v185.m128i_i64[1] != v186 )
    {
      if ( !v185.m128i_i64[1] )
        goto LABEL_12;
      goto LABEL_11;
    }
LABEL_70:
    sub_1D27190((const __m128i **)&v185, v34, &v192);
    goto LABEL_13;
  }
  sub_1D27190((const __m128i **)&v185, (const __m128i *)v185.m128i_i64[1], &v192);
  v78 = _mm_loadu_si128((const __m128i *)&a9);
  v31 = (__m128i *)v185.m128i_i64[1];
  v32 = v186;
  v192.m128i_i64[1] = a9;
  v166 = v78;
  v193.m128i_i32[0] = v78.m128i_i32[2];
  if ( (__m128i *)v185.m128i_i64[1] == v186 )
    goto LABEL_71;
  if ( v185.m128i_i64[1] )
  {
LABEL_9:
    *v31 = _mm_load_si128(&v192);
    v31[1] = _mm_load_si128(&v193);
    v31[2].m128i_i64[0] = v194;
    v31 = (__m128i *)v185.m128i_i64[1];
    v32 = v186;
  }
  v33 = _mm_loadu_si128((const __m128i *)&a10);
  v34 = (__m128i *)((char *)v31 + 40);
  v185.m128i_i64[1] = (__int64)v34;
  v192.m128i_i64[1] = a10;
  v165 = v33;
  v193.m128i_i32[0] = v33.m128i_i32[2];
  if ( v34 == v32 )
    goto LABEL_70;
LABEL_11:
  *v34 = _mm_load_si128(&v192);
  v34[1] = _mm_load_si128(&v193);
  v34[2].m128i_i64[0] = v194;
  v34 = (__m128i *)v185.m128i_i64[1];
LABEL_12:
  v185.m128i_i64[1] = (__int64)&v34[2].m128i_i64[1];
LABEL_13:
  v35 = *(_QWORD *)a4;
  v198 = 0xFFFFFFFF00000020LL;
  v209 = 0x2000000000LL;
  v212 = 0x2000000000LL;
  v215 = 0x2000000000LL;
  v217 = v219;
  v195 = 0;
  v196 = 0;
  v197 = 0;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v202 = 0u;
  v203 = 0;
  v204 = a1;
  v206 = 0;
  v207 = 0;
  v208 = v210;
  v211 = v213;
  v214 = v216;
  v218 = 0x400000000LL;
  v205 = v35;
  if ( v35 )
    sub_1623A60((__int64)&v205, v35, 2);
  v206 = *(_DWORD *)(a4 + 8);
  v163 = v159;
  v162 = (__int64)v158;
  v36 = a1[4];
  LODWORD(v196) = v159;
  v195 = v158;
  v37 = sub_1E0A0C0(v36);
  v38 = 8 * sub_15A9520(v37, 0);
  if ( v38 == 32 )
  {
    v39 = 5;
  }
  else if ( v38 > 0x20 )
  {
    v39 = 6;
    if ( v38 != 64 )
    {
      v39 = 0;
      if ( v38 == 128 )
        v39 = 7;
    }
  }
  else
  {
    v39 = 3;
    if ( v38 != 8 )
      v39 = 4 * (v38 == 16);
  }
  v159 = sub_1D27640((__int64)a1, *(char **)(a1[2] + 76712), v39, 0);
  v158 = (__int64 *)v40;
  v41 = *(_QWORD *)(v157.m128i_i64[0] + 40) + 16LL * v157.m128i_u32[2];
  v42 = *(_BYTE *)v41;
  v43 = *(_QWORD *)(v41 + 8);
  LOBYTE(v189) = v42;
  v44 = a1[6];
  v190 = v43;
  v45 = sub_1F58E60(&v189, v44);
  v46 = v202.m128i_i64[0];
  v47 = *(_DWORD *)(a1[2] + 80956);
  v197 = v45;
  v160 = v159;
  v200 = v159;
  v161 = (__int64)v158;
  LODWORD(v199) = v47;
  LODWORD(v201) = (_DWORD)v158;
  v202 = v185;
  v48 = (v185.m128i_i64[1] - v185.m128i_i64[0]) >> 3;
  v185 = 0u;
  v49 = v203;
  HIDWORD(v198) = -858993459 * v48;
  v50 = v186;
  v186 = 0;
  v203 = v50;
  if ( v46 )
    j_j___libc_free_0(v46, (char *)v49 - v46);
  v51 = (void (***)())v204[2];
  v52 = **v51;
  if ( v52 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), __int64, _QWORD, __m128i *))v52)(v51, v204[4], v47, &v202);
  v53 = a1[2];
  LOBYTE(v198) = v198 & 0xDF;
  BYTE1(v198) = v155;
  sub_2056920(&v189, v53, &v195);
  v54 = v191;
  if ( v217 != v219 )
    _libc_free((unsigned __int64)v217);
  if ( v214 != v216 )
    _libc_free((unsigned __int64)v214);
  if ( v211 != v213 )
    _libc_free((unsigned __int64)v211);
  if ( v208 != v210 )
    _libc_free((unsigned __int64)v208);
  if ( v205 )
    sub_161E7C0((__int64)&v205, v205);
  if ( v202.m128i_i64[0] )
    j_j___libc_free_0(v202.m128i_i64[0], (char *)v203 - v202.m128i_i64[0]);
  if ( v185.m128i_i64[0] )
    j_j___libc_free_0(v185.m128i_i64[0], (char *)v186 - v185.m128i_i64[0]);
  return v54;
}
