// Function: sub_1D43030
// Address: 0x1d43030
//
__int64 __fastcall sub_1D43030(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __int128 a10,
        __int128 a11,
        unsigned int a12,
        unsigned __int8 a13,
        char a14,
        __int128 a15,
        __int64 a16)
{
  __int64 *v16; // r14
  int v17; // eax
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i v23; // xmm5
  __int64 v24; // rbx
  __m128i v25; // xmm6
  __int64 v26; // rsi
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // rax
  __int64 v30; // rax
  __m128i *v31; // rsi
  __m128i *v32; // rax
  __m128i v33; // xmm7
  __m128i *v34; // rsi
  __int64 v35; // rsi
  int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // rax
  unsigned int v39; // edx
  unsigned __int8 v40; // al
  __int64 v41; // rdx
  __int64 v42; // rax
  char v43; // si
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdi
  unsigned int v48; // r9d
  __int64 v49; // rsi
  void (***v50)(); // rdi
  void (*v51)(); // r10
  __int64 v52; // rsi
  __int64 *v53; // r15
  __int64 v54; // rbx
  unsigned int v55; // r12d
  __int64 v57; // r13
  bool v58; // zf
  __int128 v59; // kr00_16
  unsigned int *v60; // rax
  __int64 *v61; // rax
  __int64 v62; // r15
  char v63; // r12
  char v64; // bl
  int v65; // eax
  __int64 v66; // rdi
  unsigned int v67; // edx
  _BOOL4 v68; // edx
  unsigned int v69; // eax
  unsigned int v70; // ecx
  unsigned int v71; // edx
  __int64 v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rbx
  __int64 v77; // rcx
  const __m128i *v78; // r13
  const __m128i *v79; // r15
  char v80; // r14
  char v81; // di
  __int64 v82; // rdx
  __int64 v83; // rcx
  unsigned int v84; // ebx
  __int64 v85; // r8
  __int64 v86; // r9
  unsigned int v87; // eax
  __int64 *v88; // rsi
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  unsigned __int64 v93; // r15
  unsigned int v94; // eax
  unsigned int v95; // eax
  __int64 v96; // rdx
  __int64 v97; // r9
  __int64 v98; // r12
  __int64 v99; // rbx
  unsigned __int64 v100; // rdx
  __int64 *v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rcx
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // rdx
  __int64 v108; // r13
  __int64 v109; // r12
  __int64 v110; // rdx
  __int64 *v111; // rdx
  char v112; // di
  unsigned int v113; // r12d
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  __int64 v118; // r9
  unsigned int v119; // eax
  __int64 (*v120)(); // rax
  __int64 v121; // rdx
  unsigned int v122; // eax
  __m128i v123; // xmm7
  bool v124; // sf
  bool v125; // of
  int v126; // eax
  __int64 v127; // r12
  __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // rdx
  __int128 v131; // [rsp-10h] [rbp-1130h]
  unsigned __int8 v132; // [rsp+Fh] [rbp-1111h]
  unsigned __int64 v133; // [rsp+10h] [rbp-1110h]
  unsigned int *v134; // [rsp+18h] [rbp-1108h]
  unsigned int v135; // [rsp+44h] [rbp-10DCh]
  unsigned int v136; // [rsp+48h] [rbp-10D8h]
  unsigned int v137; // [rsp+4Ch] [rbp-10D4h]
  __int128 v138; // [rsp+50h] [rbp-10D0h]
  unsigned __int64 v139; // [rsp+60h] [rbp-10C0h]
  __int64 v140; // [rsp+68h] [rbp-10B8h]
  __int64 v141; // [rsp+70h] [rbp-10B0h]
  __int64 v142; // [rsp+78h] [rbp-10A8h]
  unsigned __int64 v143; // [rsp+80h] [rbp-10A0h]
  char v144; // [rsp+88h] [rbp-1098h]
  unsigned __int8 v145; // [rsp+8Ch] [rbp-1094h]
  unsigned __int64 v146; // [rsp+90h] [rbp-1090h]
  unsigned __int64 v147; // [rsp+98h] [rbp-1088h]
  _BOOL4 v149; // [rsp+A8h] [rbp-1078h]
  __int64 *v150; // [rsp+A8h] [rbp-1078h]
  __int64 v151; // [rsp+A8h] [rbp-1078h]
  __m128i v152; // [rsp+B0h] [rbp-1070h] BYREF
  __int64 *v153; // [rsp+C0h] [rbp-1060h]
  __int64 v154; // [rsp+C8h] [rbp-1058h]
  __int64 v155; // [rsp+D0h] [rbp-1050h]
  __int64 v156; // [rsp+D8h] [rbp-1048h]
  __int64 v157; // [rsp+E0h] [rbp-1040h]
  unsigned __int64 v158; // [rsp+E8h] [rbp-1038h]
  __m128i v159; // [rsp+F0h] [rbp-1030h]
  __m128i v160; // [rsp+100h] [rbp-1020h]
  __m128i v161; // [rsp+110h] [rbp-1010h]
  __m128i v162; // [rsp+120h] [rbp-1000h]
  __int64 v163; // [rsp+130h] [rbp-FF0h]
  __int64 v164; // [rsp+138h] [rbp-FE8h]
  _QWORD *v165; // [rsp+140h] [rbp-FE0h]
  __int64 v166; // [rsp+148h] [rbp-FD8h]
  __m128i v167; // [rsp+150h] [rbp-FD0h] BYREF
  __m128 v168; // [rsp+160h] [rbp-FC0h] BYREF
  __m128i v169; // [rsp+170h] [rbp-FB0h] BYREF
  __int64 v170; // [rsp+180h] [rbp-FA0h]
  __int128 v171; // [rsp+190h] [rbp-F90h]
  __int64 v172; // [rsp+1A0h] [rbp-F80h]
  const __m128i *v173; // [rsp+1B0h] [rbp-F70h] BYREF
  __int64 v174; // [rsp+1B8h] [rbp-F68h]
  __int64 *v175; // [rsp+1C0h] [rbp-F60h]
  __m128i v176; // [rsp+1D0h] [rbp-F50h] BYREF
  __m128i v177; // [rsp+1E0h] [rbp-F40h] BYREF
  __int64 v178; // [rsp+1F0h] [rbp-F30h]
  __int64 *v179; // [rsp+200h] [rbp-F20h] BYREF
  __int64 v180; // [rsp+208h] [rbp-F18h]
  __int64 v181; // [rsp+210h] [rbp-F10h] BYREF
  unsigned __int64 v182; // [rsp+218h] [rbp-F08h]
  __int64 v183; // [rsp+220h] [rbp-F00h]
  __int64 v184; // [rsp+228h] [rbp-EF8h]
  __int64 v185; // [rsp+230h] [rbp-EF0h]
  __m128i v186; // [rsp+238h] [rbp-EE8h] BYREF
  __int64 v187; // [rsp+248h] [rbp-ED8h]
  __int64 *v188; // [rsp+250h] [rbp-ED0h]
  __int64 v189; // [rsp+258h] [rbp-EC8h] BYREF
  int v190; // [rsp+260h] [rbp-EC0h]
  __int64 v191; // [rsp+268h] [rbp-EB8h]
  _BYTE *v192; // [rsp+270h] [rbp-EB0h]
  __int64 v193; // [rsp+278h] [rbp-EA8h]
  _BYTE v194[1536]; // [rsp+280h] [rbp-EA0h] BYREF
  _BYTE *v195; // [rsp+880h] [rbp-8A0h]
  __int64 v196; // [rsp+888h] [rbp-898h]
  _BYTE v197[512]; // [rsp+890h] [rbp-890h] BYREF
  _BYTE *v198; // [rsp+A90h] [rbp-690h]
  __int64 v199; // [rsp+A98h] [rbp-688h]
  _BYTE v200[1536]; // [rsp+AA0h] [rbp-680h] BYREF
  _BYTE *v201; // [rsp+10A0h] [rbp-80h]
  __int64 v202; // [rsp+10A8h] [rbp-78h]
  _BYTE v203[112]; // [rsp+10B0h] [rbp-70h] BYREF

  v16 = (__int64 *)a1;
  v153 = (__int64 *)a2;
  v145 = a13;
  v154 = a4;
  v144 = a14;
  v17 = *(unsigned __int16 *)(a11 + 24);
  v152.m128i_i64[0] = a5;
  v152.m128i_i64[1] = a6;
  if ( v17 != 32 && v17 != 10 )
    goto LABEL_3;
  v54 = *(_QWORD *)(a11 + 88);
  v55 = *(_DWORD *)(v54 + 32);
  if ( v55 <= 0x40 )
  {
    if ( !*(_QWORD *)(v54 + 24) )
      return (__int64)v153;
    v147 = *(_QWORD *)(v54 + 24);
  }
  else
  {
    if ( v55 == (unsigned int)sub_16A57B0(v54 + 24) )
      return (__int64)v153;
    v147 = **(_QWORD **)(v54 + 24);
  }
  v57 = a10;
  v146 = a3;
  v58 = *(_WORD *)(a10 + 24) == 48;
  v139 = v152.m128i_u64[1];
  v53 = v153;
  v140 = v152.m128i_i64[0];
  v59 = a10;
  v169 = _mm_loadu_si128((const __m128i *)&a15);
  v170 = a16;
  if ( !v58 )
  {
    v173 = 0;
    v60 = *(unsigned int **)(a1 + 16);
    v174 = 0;
    v134 = v60;
    v61 = *(__int64 **)(a1 + 32);
    v175 = 0;
    v62 = v61[7];
    v63 = sub_1D139C0(*v61, v61[1]);
    v64 = *(_WORD *)(v152.m128i_i64[0] + 24) == 36 || *(_WORD *)(v152.m128i_i64[0] + 24) == 14;
    if ( v64 )
    {
      v65 = *(_DWORD *)(v152.m128i_i64[0] + 84);
      if ( v65 >= 0 )
      {
        if ( *(_WORD *)(v57 + 24) == 32 || *(_WORD *)(v57 + 24) == 10 )
        {
          v142 = v152.m128i_i64[0];
          v64 = *(_WORD *)(v57 + 24) == 32 || *(_WORD *)(v57 + 24) == 10;
          goto LABEL_48;
        }
        v69 = sub_1E340A0(&v169);
        v71 = 0;
        v142 = v152.m128i_i64[0];
        v70 = 0;
LABEL_52:
        v72 = v134[20376];
        if ( !v63 )
          v72 = v134[20375];
        if ( !(unsigned __int8)sub_1D26C30(
                                 (__int64)&v173,
                                 v72,
                                 v147,
                                 v70,
                                 0,
                                 1u,
                                 v71,
                                 0,
                                 1u,
                                 v69,
                                 v16[4],
                                 (__int64)v134) )
        {
          v53 = 0;
LABEL_69:
          if ( v173 )
            j_j___libc_free_0(v173, (char *)v175 - (char *)v173);
          goto LABEL_71;
        }
        if ( v64
          && (v127 = sub_1F58E60(v173, v16[6]),
              v128 = sub_1E0A0C0(v16[4]),
              v72 = v127,
              v136 = sub_15A9FE0(v128, v127),
              a12 < v136) )
        {
          v73 = 5LL * (unsigned int)(*(_DWORD *)(v62 + 32) + *(_DWORD *)(v142 + 84));
          v129 = *(_QWORD *)(v62 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v62 + 32) + *(_DWORD *)(v142 + 84));
          if ( v136 > *(_DWORD *)(v129 + 16) )
          {
            *(_DWORD *)(v129 + 16) = v136;
            v72 = v136;
            sub_1E08740(v62, v136);
          }
        }
        else
        {
          v136 = a12;
        }
        v179 = &v181;
        v180 = 0x800000000LL;
        v76 = (v174 - (__int64)v173) >> 4;
        v135 = v76;
        v167 = _mm_loadu_si128(v173);
        if ( (unsigned int)v76 > 1 )
        {
          v141 = v57;
          v77 = (unsigned int)(v76 - 2);
          v78 = v173 + 1;
          v150 = v16;
          v79 = &v173[v77 + 2];
          while ( 1 )
          {
            a7 = _mm_load_si128(&v167);
            v80 = v167.m128i_i8[0];
            v176 = a7;
            v81 = v78->m128i_i8[0];
            if ( v167.m128i_i8[0] != v78->m128i_i8[0] )
              break;
            if ( !v167.m128i_i8[0] && v176.m128i_i64[1] != v78->m128i_i64[1] )
              goto LABEL_80;
LABEL_60:
            if ( v79 == ++v78 )
            {
              v16 = v150;
              v88 = (__int64 *)*((_QWORD *)&v59 + 1);
              *(_QWORD *)&v138 = sub_1D42590(
                                   v141,
                                   *((unsigned __int64 *)&v59 + 1),
                                   v167.m128i_u32[0],
                                   (const void **)v167.m128i_i64[1],
                                   v150,
                                   v154,
                                   a7,
                                   *(double *)a8.m128i_i64,
                                   a9);
              *((_QWORD *)&v138 + 1) = v89;
              goto LABEL_83;
            }
          }
          if ( v81 )
          {
            v84 = sub_1D13440(v81);
            if ( v80 )
            {
LABEL_64:
              v87 = sub_1D13440(v80);
              goto LABEL_65;
            }
          }
          else
          {
LABEL_80:
            v84 = sub_1F58D40(v78, v72, v73, v77 * 16, v74, v75);
            if ( v80 )
              goto LABEL_64;
          }
          v87 = sub_1F58D40(&v176, v72, v82, v83, v85, v86);
LABEL_65:
          if ( v87 < v84 )
            v167 = _mm_loadu_si128(v78);
          goto LABEL_60;
        }
        v88 = (__int64 *)*((_QWORD *)&v59 + 1);
        *(_QWORD *)&v138 = sub_1D42590(
                             v57,
                             *((unsigned __int64 *)&v59 + 1),
                             v167.m128i_u32[0],
                             (const void **)v167.m128i_i64[1],
                             v16,
                             v154,
                             a7,
                             *(double *)a8.m128i_i64,
                             a9);
        *((_QWORD *)&v138 + 1) = v89;
        if ( !(_DWORD)v76 )
        {
          v113 = v180;
LABEL_124:
          *((_QWORD *)&v131 + 1) = v113;
          *(_QWORD *)&v131 = v179;
          v53 = sub_1D359D0(v16, 2, v154, 1, 0, 0, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v131);
          if ( v179 != &v181 )
            _libc_free((unsigned __int64)v179);
          goto LABEL_69;
        }
LABEL_83:
        v151 = 0;
        v93 = 0;
        v133 = *((_QWORD *)&v138 + 1) & 0xFFFFFFFF00000000LL;
        while ( 1 )
        {
          a8 = _mm_loadu_si128(&v173[v151]);
          v168 = (__m128)a8;
          if ( a8.m128i_i8[0] )
            v95 = sub_1D13440(a8.m128i_i8[0]);
          else
            v95 = sub_1F58D40(&v168, v88, v89, v90, v91, v92);
          a9 = _mm_load_si128(&v167);
          v98 = v138;
          v143 = v95 >> 3;
          v176 = a9;
          v99 = *((_QWORD *)&v138 + 1);
          if ( v147 < v143 )
            v93 = v147 + v93 - v143;
          if ( a8.m128i_i8[0] == v167.m128i_i8[0] )
          {
            if ( a8.m128i_i8[0] || v176.m128i_i64[1] == v168.m128_u64[1] )
              goto LABEL_92;
          }
          else if ( a8.m128i_i8[0] )
          {
            v137 = sub_1D13440(a8.m128i_i8[0]);
            goto LABEL_104;
          }
          v132 = v167.m128i_i8[0];
          v122 = sub_1F58D40(&v168, v88, v96, v147, v167.m128i_u8[0], v97);
          v117 = v132;
          v137 = v122;
LABEL_104:
          if ( (_BYTE)v117 )
            v119 = sub_1D13440(v117);
          else
            v119 = sub_1F58D40(&v176, v88, v115, v116, v117, v118);
          if ( v119 <= v137 )
            goto LABEL_92;
          if ( v167.m128i_i8[0] )
          {
            if ( (unsigned __int8)(v167.m128i_i8[0] - 14) <= 0x5Fu )
              goto LABEL_112;
          }
          else if ( (unsigned __int8)sub_1F58D20(&v167) )
          {
            goto LABEL_112;
          }
          if ( a8.m128i_i8[0] )
          {
            if ( (unsigned __int8)(a8.m128i_i8[0] - 14) <= 0x5Fu )
              goto LABEL_112;
          }
          else if ( (unsigned __int8)sub_1F58D20(&v168) )
          {
            goto LABEL_112;
          }
          v120 = *(__int64 (**)())(*(_QWORD *)v134 + 800LL);
          if ( v120 != sub_1D12DF0
            && ((unsigned __int8 (__fastcall *)(unsigned int *, _QWORD, __int64, _QWORD, unsigned __int64))v120)(
                 v134,
                 v167.m128i_u32[0],
                 v167.m128i_i64[1],
                 v168.m128_u32[0],
                 v168.m128_u64[1]) )
          {
            v163 = sub_1D309E0(
                     v16,
                     145,
                     v154,
                     v168.m128_u32[0],
                     (const void **)v168.m128_u64[1],
                     0,
                     *(double *)a7.m128i_i64,
                     *(double *)a8.m128i_i64,
                     *(double *)a9.m128i_i64,
                     v138);
            v98 = v163;
            v164 = v130;
            v99 = v133 | (unsigned int)v130;
            goto LABEL_92;
          }
LABEL_112:
          v165 = sub_1D42590(
                   v59,
                   *((unsigned __int64 *)&v59 + 1),
                   v168.m128_u32[0],
                   (const void **)v168.m128_u64[1],
                   v16,
                   v154,
                   a7,
                   *(double *)a8.m128i_i64,
                   a9);
          v98 = (__int64)v165;
          v166 = v121;
          v99 = v133 | (unsigned int)v121;
LABEL_92:
          v176 = 0u;
          v177.m128i_i64[0] = 0;
          v100 = v169.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v169.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v169.m128i_i8[0] & 4) != 0 )
            {
              *((_QWORD *)&v171 + 1) = v93 + v169.m128i_i64[1];
              LOBYTE(v172) = v170;
              *(_QWORD *)&v171 = v100 | 4;
              HIDWORD(v172) = *(_DWORD *)(v100 + 12);
            }
            else
            {
              v114 = *(_QWORD *)v100;
              *(_QWORD *)&v171 = v169.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
              *((_QWORD *)&v171 + 1) = v93 + v169.m128i_i64[1];
              v58 = *(_BYTE *)(v114 + 8) == 16;
              LOBYTE(v172) = v170;
              if ( v58 )
                v114 = **(_QWORD **)(v114 + 16);
              HIDWORD(v172) = *(_DWORD *)(v114 + 8) >> 8;
            }
          }
          else
          {
            LODWORD(v172) = 0;
            v171 = 0u;
            HIDWORD(v172) = HIDWORD(v170);
          }
          v101 = sub_1D3D250(v16, v140, v139, v93, v154, a7, *(double *)a8.m128i_i64, a9);
          v88 = v153;
          v146 = v146 & 0xFFFFFFFF00000000LL | (unsigned int)a3;
          v103 = sub_1D2BF40(
                   v16,
                   (__int64)v153,
                   v146,
                   v154,
                   v98,
                   v99,
                   (__int64)v101,
                   v102,
                   v171,
                   v172,
                   v136,
                   v145 != 0 ? 4 : 0,
                   (__int64)&v176);
          v108 = v107;
          v109 = v103;
          v110 = (unsigned int)v180;
          if ( (unsigned int)v180 >= HIDWORD(v180) )
          {
            v88 = &v181;
            sub_16CD150((__int64)&v179, &v181, 0, 16, v105, v106);
            v110 = (unsigned int)v180;
          }
          v111 = &v179[2 * v110];
          *v111 = v109;
          v112 = v168.m128_i8[0];
          v111[1] = v108;
          v113 = v180 + 1;
          LODWORD(v180) = v180 + 1;
          if ( v112 )
            v94 = sub_1D13440(v112);
          else
            v94 = sub_1F58D40(&v168, v88, v111, v104, v105, v106);
          ++v151;
          v93 += v94 >> 3;
          v147 -= v143;
          if ( v135 <= (unsigned int)v151 )
            goto LABEL_124;
        }
      }
      v125 = __OFSUB__(v65, -*(_DWORD *)(v62 + 32));
      v124 = v65 + *(_DWORD *)(v62 + 32) < 0;
      v126 = *(unsigned __int16 *)(v57 + 24);
      v64 = v124 ^ v125;
      if ( v126 != 32 && v126 != 10 )
      {
        v68 = 0;
        v142 = v152.m128i_i64[0];
        goto LABEL_50;
      }
      v142 = v152.m128i_i64[0];
    }
    else
    {
      v142 = 0;
      if ( *(_WORD *)(v57 + 24) != 32 && *(_WORD *)(v57 + 24) != 10 )
      {
        v69 = sub_1E340A0(&v169);
        v70 = a12;
        v71 = 0;
        goto LABEL_52;
      }
    }
LABEL_48:
    v66 = *(_QWORD *)(v59 + 88);
    v67 = *(_DWORD *)(v66 + 32);
    if ( v67 <= 0x40 )
      v68 = *(_QWORD *)(v66 + 24) == 0;
    else
      v68 = v67 == sub_16A57B0(v66 + 24);
LABEL_50:
    v149 = v68;
    v69 = sub_1E340A0(&v169);
    v70 = 0;
    v71 = v149;
    if ( !v64 )
      v70 = a12;
    goto LABEL_52;
  }
LABEL_71:
  if ( v53 )
    return (__int64)v53;
LABEL_3:
  v18 = v16[1];
  if ( v18 )
  {
    v19 = *(__int64 (**)())(*(_QWORD *)v18 + 32LL);
    if ( v19 != sub_1D12E70 )
    {
      v53 = (__int64 *)((__int64 (__fastcall *)(__int64, __int64 *, __int64, __int64 *, unsigned __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64))v19)(
                         v18,
                         v16,
                         v154,
                         v153,
                         a3,
                         a12,
                         v152.m128i_i64[0],
                         v152.m128i_i64[1],
                         a10,
                         *((_QWORD *)&a10 + 1),
                         a11,
                         *((_QWORD *)&a11 + 1),
                         v145,
                         a15,
                         *((_QWORD *)&a15 + 1),
                         a16);
      if ( v53 )
        return (__int64)v53;
    }
  }
  v20 = sub_1E340A0(&a15);
  sub_1D13830(v16[2], v20);
  v21 = sub_1E0A0C0(v16[4]);
  v22 = sub_15A9620(v21, v16[6], 0);
  v23 = _mm_load_si128(&v152);
  v169 = 0u;
  v24 = v22;
  v162 = v23;
  v176.m128i_i64[1] = v152.m128i_i64[0];
  v177.m128i_i32[0] = v23.m128i_i32[2];
  v170 = 0;
  v176.m128i_i64[0] = 0;
  LODWORD(v178) = 0;
  v177.m128i_i64[1] = v22;
  sub_1D27190((const __m128i **)&v169, 0, &v176);
  v25 = _mm_loadu_si128((const __m128i *)&a10);
  v176.m128i_i64[1] = a10;
  v161 = v25;
  v26 = v16[6];
  v177.m128i_i32[0] = v25.m128i_i32[2];
  v27 = *(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10);
  v28 = *(_BYTE *)v27;
  v29 = *(_QWORD *)(v27 + 8);
  LOBYTE(v179) = v28;
  v180 = v29;
  v30 = sub_1F58E60(&v179, v26);
  v31 = (__m128i *)v169.m128i_i64[1];
  v177.m128i_i64[1] = v30;
  v32 = (__m128i *)v170;
  if ( v169.m128i_i64[1] == v170 )
  {
    sub_1D27190((const __m128i **)&v169, (const __m128i *)v169.m128i_i64[1], &v176);
    v123 = _mm_loadu_si128((const __m128i *)&a11);
    v177.m128i_i64[1] = v24;
    v34 = (__m128i *)v169.m128i_i64[1];
    v176.m128i_i64[1] = a11;
    v159 = v123;
    v177.m128i_i32[0] = v123.m128i_i32[2];
    if ( v170 != v169.m128i_i64[1] )
    {
      if ( !v169.m128i_i64[1] )
        goto LABEL_10;
      goto LABEL_9;
    }
LABEL_137:
    sub_1D27190((const __m128i **)&v169, v34, &v176);
    goto LABEL_11;
  }
  if ( v169.m128i_i64[1] )
  {
    *(__m128i *)v169.m128i_i64[1] = _mm_load_si128(&v176);
    v31[1] = _mm_load_si128(&v177);
    v31[2].m128i_i64[0] = v178;
    v31 = (__m128i *)v169.m128i_i64[1];
    v32 = (__m128i *)v170;
  }
  v33 = _mm_loadu_si128((const __m128i *)&a11);
  v34 = (__m128i *)((char *)v31 + 40);
  v177.m128i_i64[1] = v24;
  v169.m128i_i64[1] = (__int64)v34;
  v176.m128i_i64[1] = a11;
  v160 = v33;
  v177.m128i_i32[0] = v33.m128i_i32[2];
  if ( v32 == v34 )
    goto LABEL_137;
LABEL_9:
  *v34 = _mm_load_si128(&v176);
  v34[1] = _mm_load_si128(&v177);
  v34[2].m128i_i64[0] = v178;
  v34 = (__m128i *)v169.m128i_i64[1];
LABEL_10:
  v169.m128i_i64[1] = (__int64)&v34[2].m128i_i64[1];
LABEL_11:
  v179 = 0;
  v182 = 0xFFFFFFFF00000020LL;
  v193 = 0x2000000000LL;
  v196 = 0x2000000000LL;
  v199 = 0x2000000000LL;
  v201 = v203;
  v202 = 0x400000000LL;
  v180 = 0;
  v35 = *(_QWORD *)v154;
  v181 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0u;
  v187 = 0;
  v188 = v16;
  v190 = 0;
  v191 = 0;
  v192 = v194;
  v195 = v197;
  v198 = v200;
  v189 = v35;
  if ( v35 )
    sub_1623A60((__int64)&v189, v35, 2);
  v36 = *(_DWORD *)(v154 + 8);
  v158 = a3;
  LODWORD(v180) = a3;
  v190 = v36;
  v157 = (__int64)v153;
  v37 = v16[4];
  v179 = v153;
  v38 = sub_1E0A0C0(v37);
  v39 = 8 * sub_15A9520(v38, 0);
  if ( v39 == 32 )
  {
    v40 = 5;
  }
  else if ( v39 > 0x20 )
  {
    v40 = 6;
    if ( v39 != 64 )
    {
      v40 = 0;
      if ( v39 == 128 )
        v40 = 7;
    }
  }
  else
  {
    v40 = 3;
    if ( v39 != 8 )
      v40 = 4 * (v39 == 16);
  }
  v154 = sub_1D27640((__int64)v16, *(char **)(v16[2] + 76720), v40, 0);
  v153 = (__int64 *)v41;
  v42 = *(_QWORD *)(v152.m128i_i64[0] + 40) + 16LL * v152.m128i_u32[2];
  v43 = *(_BYTE *)v42;
  v44 = *(_QWORD *)(v42 + 8);
  LOBYTE(v173) = v43;
  v45 = v16[6];
  v174 = v44;
  v46 = sub_1F58E60(&v173, v45);
  v47 = v186.m128i_i64[0];
  v48 = *(_DWORD *)(v16[2] + 80960);
  v156 = (__int64)v153;
  v181 = v46;
  LODWORD(v183) = v48;
  LODWORD(v185) = (_DWORD)v153;
  v155 = v154;
  v184 = v154;
  LODWORD(v46) = -858993459 * ((v169.m128i_i64[1] - v169.m128i_i64[0]) >> 3);
  v186 = v169;
  v49 = v187;
  v169 = 0u;
  HIDWORD(v182) = v46;
  v187 = v170;
  v170 = 0;
  if ( v47 )
  {
    LODWORD(v154) = v48;
    j_j___libc_free_0(v47, v49 - v47);
    v48 = v154;
  }
  v50 = (void (***)())v188[2];
  v51 = **v50;
  if ( v51 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), __int64, _QWORD, __m128i *))v51)(v50, v188[4], v48, &v186);
  v52 = v16[2];
  LOBYTE(v182) = v182 & 0xDF;
  BYTE1(v182) = v144;
  sub_2056920(&v173, v52, &v179);
  v53 = v175;
  if ( v201 != v203 )
    _libc_free((unsigned __int64)v201);
  if ( v198 != v200 )
    _libc_free((unsigned __int64)v198);
  if ( v195 != v197 )
    _libc_free((unsigned __int64)v195);
  if ( v192 != v194 )
    _libc_free((unsigned __int64)v192);
  if ( v189 )
    sub_161E7C0((__int64)&v189, v189);
  if ( v186.m128i_i64[0] )
    j_j___libc_free_0(v186.m128i_i64[0], v187 - v186.m128i_i64[0]);
  if ( v169.m128i_i64[0] )
    j_j___libc_free_0(v169.m128i_i64[0], v170 - v169.m128i_i64[0]);
  return (__int64)v53;
}
