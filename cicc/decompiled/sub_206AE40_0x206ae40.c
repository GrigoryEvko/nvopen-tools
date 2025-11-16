// Function: sub_206AE40
// Address: 0x206ae40
//
void __fastcall sub_206AE40(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rax
  __m128i *v9; // r14
  __int64 v10; // rax
  unsigned int v11; // edx
  unsigned __int8 v12; // al
  int v13; // edx
  unsigned int v14; // ebx
  __int64 v15; // rax
  int v16; // r9d
  __int64 v17; // rax
  __int64 v18; // rsi
  __m128i v19; // rax
  __int64 v20; // r13
  _QWORD **v21; // r12
  __int64 v22; // rax
  int v23; // eax
  _QWORD *v24; // r10
  __int64 v25; // rsi
  __m128i v26; // rax
  __int64 v27; // rax
  __int64 (*v28)(); // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 (*v33)(); // rdx
  __int64 *v34; // r12
  __int64 *v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // r9
  __int64 v39; // r8
  __int32 v40; // edx
  __int64 v41; // rax
  unsigned int v42; // edx
  __int64 v43; // rsi
  __int64 *v44; // rdi
  const void **v45; // r13
  unsigned int v46; // ebx
  __int64 *v47; // rax
  __int64 *v48; // r12
  __int16 *v49; // rdx
  __int64 v50; // rdx
  __int64 (__fastcall *v51)(__m128i *, __int64, __int64, __int64, __int64); // rbx
  unsigned __int8 *v52; // rax
  __int64 v53; // r8
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 *v64; // rax
  __int64 *v65; // r14
  __int64 *v66; // r12
  __int64 v67; // rdx
  __int64 v68; // r13
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  _QWORD *v72; // rax
  __int64 v73; // rdx
  __int64 *v74; // rax
  __int64 *v75; // r14
  __int64 v76; // r12
  __int64 v77; // rdx
  unsigned __int64 v78; // r13
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  __int128 v82; // rax
  __int64 *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r12
  __int64 *v86; // rbx
  __int64 v87; // r13
  __int64 v88; // rbx
  __m128i v89; // xmm0
  __int64 v90; // rax
  __int64 v91; // rax
  char v92; // al
  __int64 *v93; // rcx
  __m128i *v94; // rsi
  __int64 v95; // rax
  int v96; // esi
  __int64 v97; // rdx
  __int64 v98; // rsi
  __int64 v99; // rsi
  __int64 *v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rdi
  __int64 *v103; // rdx
  __int64 v104; // rsi
  __int64 v105; // rdi
  const __m128i *v106; // rdi
  signed __int64 v107; // rax
  const __m128i *v108; // rsi
  const __m128i *v109; // rax
  __int64 v110; // r12
  __int64 v111; // r15
  __int64 v112; // rdx
  __int128 v113; // [rsp-20h] [rbp-10A0h]
  __int128 v114; // [rsp-10h] [rbp-1090h]
  __int128 v115; // [rsp+0h] [rbp-1080h]
  __int64 v116; // [rsp+10h] [rbp-1070h]
  unsigned int v117; // [rsp+18h] [rbp-1068h]
  __int64 v118; // [rsp+18h] [rbp-1068h]
  const void **v119; // [rsp+18h] [rbp-1068h]
  __int64 *v120; // [rsp+18h] [rbp-1068h]
  __int64 *v121; // [rsp+18h] [rbp-1068h]
  __int64 *v122; // [rsp+18h] [rbp-1068h]
  __int64 v123; // [rsp+18h] [rbp-1068h]
  int v125; // [rsp+28h] [rbp-1058h]
  __int16 *v126; // [rsp+28h] [rbp-1058h]
  __m128i v127; // [rsp+30h] [rbp-1050h] BYREF
  _QWORD *v128; // [rsp+40h] [rbp-1040h]
  __int64 *v129; // [rsp+48h] [rbp-1038h]
  __int64 v130; // [rsp+50h] [rbp-1030h]
  __int64 v131; // [rsp+58h] [rbp-1028h]
  __int64 v132; // [rsp+60h] [rbp-1020h]
  __int64 v133; // [rsp+68h] [rbp-1018h]
  __int64 *v134; // [rsp+70h] [rbp-1010h]
  __int64 v135; // [rsp+78h] [rbp-1008h]
  __int64 v136; // [rsp+80h] [rbp-1000h]
  __int64 v137; // [rsp+88h] [rbp-FF8h]
  __int64 *v138; // [rsp+90h] [rbp-FF0h]
  __int64 v139; // [rsp+98h] [rbp-FE8h]
  __m128i v140; // [rsp+A0h] [rbp-FE0h]
  _QWORD *v141; // [rsp+B0h] [rbp-FD0h]
  __int64 v142; // [rsp+B8h] [rbp-FC8h]
  __int64 v143; // [rsp+C0h] [rbp-FC0h] BYREF
  int v144; // [rsp+C8h] [rbp-FB8h]
  const __m128i *v145; // [rsp+D0h] [rbp-FB0h] BYREF
  __m128i *v146; // [rsp+D8h] [rbp-FA8h]
  const __m128i *v147; // [rsp+E0h] [rbp-FA0h]
  __int128 v148; // [rsp+F0h] [rbp-F90h] BYREF
  __int64 v149; // [rsp+100h] [rbp-F80h]
  __int128 v150; // [rsp+110h] [rbp-F70h] BYREF
  __int64 v151; // [rsp+120h] [rbp-F60h]
  __int64 v152; // [rsp+128h] [rbp-F58h]
  __m128i v153; // [rsp+130h] [rbp-F50h] BYREF
  __m128i v154; // [rsp+140h] [rbp-F40h] BYREF
  __int64 v155; // [rsp+150h] [rbp-F30h]
  __m128i v156; // [rsp+160h] [rbp-F20h] BYREF
  __int64 v157; // [rsp+170h] [rbp-F10h]
  unsigned __int64 v158; // [rsp+178h] [rbp-F08h]
  __int64 v159; // [rsp+180h] [rbp-F00h]
  __int64 *v160; // [rsp+188h] [rbp-EF8h]
  __int64 v161; // [rsp+190h] [rbp-EF0h]
  const __m128i *v162; // [rsp+198h] [rbp-EE8h]
  __m128i *v163; // [rsp+1A0h] [rbp-EE0h]
  const __m128i *v164; // [rsp+1A8h] [rbp-ED8h]
  __int64 v165; // [rsp+1B0h] [rbp-ED0h]
  __int64 v166; // [rsp+1B8h] [rbp-EC8h] BYREF
  int v167; // [rsp+1C0h] [rbp-EC0h]
  __int64 v168; // [rsp+1C8h] [rbp-EB8h]
  _QWORD *v169; // [rsp+1D0h] [rbp-EB0h]
  __int64 v170; // [rsp+1D8h] [rbp-EA8h]
  _BYTE v171[1536]; // [rsp+1E0h] [rbp-EA0h] BYREF
  _BYTE *v172; // [rsp+7E0h] [rbp-8A0h]
  __int64 v173; // [rsp+7E8h] [rbp-898h]
  _BYTE v174[512]; // [rsp+7F0h] [rbp-890h] BYREF
  _BYTE *v175; // [rsp+9F0h] [rbp-690h]
  __int64 v176; // [rsp+9F8h] [rbp-688h]
  _BYTE v177[1536]; // [rsp+A00h] [rbp-680h] BYREF
  _BYTE *v178; // [rsp+1000h] [rbp-80h]
  __int64 v179; // [rsp+1008h] [rbp-78h]
  _BYTE v180[112]; // [rsp+1010h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a1 + 552);
  v9 = *(__m128i **)(v8 + 16);
  v10 = sub_1E0A0C0(*(_QWORD *)(v8 + 32));
  v11 = 8 * sub_15A9520(v10, 0);
  if ( v11 == 32 )
  {
    v12 = 5;
  }
  else if ( v11 > 0x20 )
  {
    v12 = 6;
    if ( v11 != 64 )
    {
      v12 = 0;
      if ( v11 == 128 )
        v12 = 7;
    }
  }
  else
  {
    v12 = 3;
    if ( v11 != 8 )
      v12 = 4 * (v11 == 16);
  }
  v13 = *(_DWORD *)(a1 + 536);
  v143 = 0;
  v14 = v12;
  v15 = *(_QWORD *)(a3 + 56);
  v144 = v13;
  v16 = *(_DWORD *)(*(_QWORD *)(v15 + 56) + 68LL);
  v17 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v129 = &v143;
    if ( &v143 != (__int64 *)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v143 = v18;
      if ( v18 )
      {
        v127.m128i_i32[0] = v16;
        sub_1623A60((__int64)&v143, v18, 2);
        v16 = v127.m128i_i32[0];
      }
    }
  }
  else
  {
    v129 = &v143;
  }
  v117 = v16;
  v19.m128i_i64[0] = (__int64)sub_1D299D0(*(_QWORD **)(a1 + 552), v16, v14, 0, 0);
  v20 = *(_QWORD *)(a1 + 560);
  v127 = v19;
  v21 = *(_QWORD ***)(**(_QWORD **)(a3 + 56) + 40LL);
  v22 = sub_16471D0(*v21, 0);
  v23 = sub_15AAE50(v20, v22);
  v24 = *(_QWORD **)(a1 + 552);
  v125 = v23;
  v156 = 0u;
  v157 = 0;
  v25 = v24[4];
  v128 = v24;
  sub_1E341E0((__int64)&v148, v25, v117, 0);
  v26.m128i_i64[0] = sub_1D2B730(
                       v128,
                       v14,
                       0,
                       (__int64)v129,
                       *(_QWORD *)(a1 + 552) + 88LL,
                       0,
                       v127.m128i_i64[0],
                       v127.m128i_i64[1],
                       v148,
                       v149,
                       v125,
                       4u,
                       (__int64)&v156,
                       0);
  v127 = v26;
  v128 = (_QWORD *)v26.m128i_i64[0];
  v27 = v9->m128i_i64[0];
  v28 = *(__int64 (**)())(v9->m128i_i64[0] + 544);
  if ( v28 != sub_1F2AB40 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__m128i *))v28)(v9) )
    {
      v128 = (_QWORD *)(*(__int64 (__fastcall **)(__m128i *, _QWORD, __int64, __int64, __int64 *))(v9->m128i_i64[0]
                                                                                                 + 1480))(
                         v9,
                         *(_QWORD *)(a1 + 552),
                         v127.m128i_i64[0],
                         v127.m128i_i64[1],
                         v129);
      v141 = v128;
      v127.m128i_i64[0] = (__int64)v128;
      v142 = v112;
      v127.m128i_i64[1] = (unsigned int)v112 | v127.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    v27 = v9->m128i_i64[0];
  }
  v29 = (*(__int64 (__fastcall **)(__m128i *, _QWORD **))(v27 + 552))(v9, v21);
  v30 = v29;
  if ( !v29 )
  {
    v31 = *(_QWORD *)(a1 + 552);
    DWORD2(v150) = 0;
    *(_QWORD *)&v150 = v31 + 88;
    v32 = v9->m128i_i64[0];
    v33 = *(__int64 (**)())(v9->m128i_i64[0] + 1472);
    if ( v33 != sub_2043CB0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__m128i *, _QWORD **, __int64 (*)(), _QWORD))v33)(v9, v21, v33, 0) )
      {
        v43 = sub_2046710(*(_QWORD **)(a1 + 552), (__int64)v129, &v150);
LABEL_17:
        v44 = *(__int64 **)(a1 + 552);
        v45 = *(const void ***)(*(_QWORD *)(v43 + 40) + 16LL * v42 + 8);
        v46 = *(unsigned __int8 *)(*(_QWORD *)(v43 + 40) + 16LL * v42);
        *((_QWORD *)&v114 + 1) = v127.m128i_i64[1];
        v127.m128i_i64[0] = (__int64)v128;
        *(_QWORD *)&v114 = v128;
        v47 = sub_1D332F0(
                v44,
                53,
                (__int64)v129,
                v46,
                v45,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                a6,
                v43,
                v42,
                v114);
        v48 = *(__int64 **)(a1 + 552);
        v126 = v49;
        v127.m128i_i64[0] = (__int64)v47;
        *(_QWORD *)&v115 = sub_1D38BB0((__int64)v48, 0, (__int64)v129, v46, v45, 0, a4, *(double *)a5.m128i_i64, a6, 0);
        *((_QWORD *)&v115 + 1) = v50;
        v51 = *(__int64 (__fastcall **)(__m128i *, __int64, __int64, __int64, __int64))(v9->m128i_i64[0] + 264);
        v52 = (unsigned __int8 *)(*(_QWORD *)(v127.m128i_i64[0] + 40) + 16LL * (unsigned int)v126);
        v53 = *((_QWORD *)v52 + 1);
        v54 = *v52;
        v55 = *(_QWORD *)(a1 + 552);
        v116 = v53;
        v56 = *(_QWORD *)(v55 + 48);
        v118 = v54;
        v57 = sub_1E0A0C0(*(_QWORD *)(v55 + 32));
        LODWORD(v51) = v51(v9, v57, v56, v118, v116);
        v119 = (const void **)v58;
        v62 = sub_1D28D50(v48, 0x16u, v58, v59, v60, v61);
        v64 = sub_1D3A900(
                v48,
                0x89u,
                (__int64)v129,
                (unsigned int)v51,
                v119,
                0,
                (__m128)a4,
                *(double *)a5.m128i_i64,
                a6,
                v127.m128i_u64[0],
                v126,
                v115,
                v62,
                v63);
        v65 = *(__int64 **)(a1 + 552);
        v66 = v64;
        v68 = v67;
        v72 = sub_1D2A490(v65, *(_QWORD *)(a2 + 16), v67, v69, v70, v71);
        *((_QWORD *)&v113 + 1) = v68;
        *(_QWORD *)&v113 = v66;
        v74 = sub_1D3A900(
                v65,
                0xBFu,
                (__int64)v129,
                1u,
                0,
                0,
                (__m128)a4,
                *(double *)a5.m128i_i64,
                a6,
                *(_QWORD *)v128[4],
                *(__int16 **)(v128[4] + 8LL),
                v113,
                (__int64)v72,
                v73);
        v75 = *(__int64 **)(a1 + 552);
        v76 = (__int64)v74;
        v78 = v77;
        *(_QWORD *)&v82 = sub_1D2A490(v75, *(_QWORD *)(a2 + 8), v77, v79, v80, v81);
        v83 = sub_1D332F0(
                v75,
                188,
                (__int64)v129,
                1,
                0,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                a6,
                v76,
                v78,
                v82);
        v85 = *(_QWORD *)(a1 + 552);
        v86 = v83;
        v87 = v84;
        if ( v83 )
        {
          nullsub_686();
          v135 = v87;
          v134 = v86;
          *(_QWORD *)(v85 + 176) = v86;
          *(_DWORD *)(v85 + 184) = v135;
          sub_1D23870();
        }
        else
        {
          v131 = v84;
          v130 = 0;
          *(_QWORD *)(v85 + 176) = 0;
          *(_DWORD *)(v85 + 184) = v131;
        }
        goto LABEL_19;
      }
      v32 = v9->m128i_i64[0];
    }
    v34 = (__int64 *)(*(__int64 (__fastcall **)(__m128i *, _QWORD **, __int64 (*)(), __int64))(v32 + 536))(
                       v9,
                       v21,
                       v33,
                       v30);
    v35 = sub_20685E0(a1, v34, a4, a5, a6);
    v153 = (__m128i)(unsigned __int64)v34;
    v36 = *(_QWORD **)(a1 + 552);
    v38 = v37;
    v39 = (__int64)v35;
    v40 = 0;
    v156 = 0u;
    v157 = 0;
    v154.m128i_i8[0] = 0;
    if ( v34 )
    {
      v41 = *v34;
      if ( *(_BYTE *)(*v34 + 8) == 16 )
        v41 = **(_QWORD **)(v41 + 16);
      v40 = *(_DWORD *)(v41 + 8) >> 8;
    }
    v154.m128i_i32[1] = v40;
    v43 = sub_1D2B730(
            v36,
            v14,
            0,
            (__int64)v129,
            v150,
            *((__int64 *)&v150 + 1),
            v39,
            v38,
            *(_OWORD *)&v153,
            v154.m128i_i64[0],
            v125,
            4u,
            (__int64)&v156,
            0);
    goto LABEL_17;
  }
  v88 = *(_QWORD *)(v29 + 24);
  v154.m128i_i64[1] = 0;
  v127.m128i_i64[0] = (__int64)v128;
  v89 = _mm_load_si128(&v127);
  v153.m128i_i64[1] = (__int64)v128;
  v140 = v89;
  v153.m128i_i64[0] = 0;
  LODWORD(v155) = 0;
  v154.m128i_i32[0] = v89.m128i_i32[2];
  v147 = 0;
  v90 = *(_QWORD *)(v88 + 16);
  v146 = 0;
  v91 = *(_QWORD *)(v90 + 8);
  v127.m128i_i64[0] = v30;
  v145 = 0;
  v154.m128i_i64[1] = v91;
  v156.m128i_i64[0] = *(_QWORD *)(v30 + 112);
  v92 = sub_1560260(&v156, 1, 12);
  v93 = (__int64 *)v127.m128i_i64[0];
  if ( v92 )
    LOBYTE(v155) = v155 | 4;
  v94 = v146;
  if ( v146 == v147 )
  {
    sub_1D27190(&v145, v146, &v153);
    v93 = (__int64 *)v127.m128i_i64[0];
  }
  else
  {
    if ( v146 )
    {
      a5 = _mm_loadu_si128(&v153);
      *v146 = a5;
      a6 = _mm_loadu_si128(&v154);
      v94[1] = a6;
      v94[2].m128i_i64[0] = v155;
      v94 = v146;
    }
    v146 = (__m128i *)((char *)v94 + 40);
  }
  v95 = *(_QWORD *)(a1 + 552);
  v158 = 0xFFFFFFFF00000020LL;
  v170 = 0x2000000000LL;
  v173 = 0x2000000000LL;
  v176 = 0x2000000000LL;
  v128 = v171;
  v169 = v171;
  v127.m128i_i64[0] = (__int64)v180;
  v178 = v180;
  v175 = v177;
  v96 = *(_DWORD *)(a1 + 536);
  v179 = 0x400000000LL;
  v97 = *(_QWORD *)a1;
  v156 = 0u;
  v157 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v165 = v95;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v172 = v174;
  *(_QWORD *)&v150 = 0;
  DWORD2(v150) = v96;
  if ( v97 )
  {
    if ( &v150 != (__int128 *)(v97 + 48) )
    {
      v98 = *(_QWORD *)(v97 + 48);
      *(_QWORD *)&v150 = v98;
      if ( v98 )
      {
        v120 = v93;
        sub_1623A60((__int64)&v150, v98, 2);
        v93 = v120;
        if ( v166 )
        {
          sub_161E7C0((__int64)&v166, v166);
          v99 = v150;
          v93 = v120;
        }
        else
        {
          v99 = v150;
        }
        v166 = v99;
        if ( v99 )
        {
          v121 = v93;
          sub_1623A60((__int64)&v166, v99, 2);
          v95 = *(_QWORD *)(a1 + 552);
          v93 = v121;
        }
        else
        {
          v95 = *(_QWORD *)(a1 + 552);
        }
      }
    }
  }
  v122 = v93;
  v167 = DWORD2(v150);
  v156.m128i_i64[0] = v95 + 88;
  v156.m128i_i32[2] = 0;
  v100 = sub_20685E0(a1, v93, v89, a5, a6);
  v102 = v101;
  v103 = v100;
  v104 = v102;
  v105 = **(_QWORD **)(v88 + 16);
  LOWORD(v100) = *((_WORD *)v122 + 9);
  v139 = v104;
  v138 = v103;
  v160 = v103;
  v157 = v105;
  v106 = v162;
  LODWORD(v161) = v104;
  LODWORD(v159) = ((unsigned __int16)v100 >> 4) & 0x3FF;
  v162 = v145;
  v107 = (char *)v146 - (char *)v145;
  v163 = v146;
  v145 = 0;
  v146 = 0;
  v108 = v164;
  HIDWORD(v158) = -858993459 * (v107 >> 3);
  v109 = v147;
  v147 = 0;
  v164 = v109;
  if ( v106 )
    j_j___libc_free_0(v106, (char *)v108 - (char *)v106);
  if ( (_QWORD)v150 )
    sub_161E7C0((__int64)&v150, v150);
  sub_2056920((__int64)&v150, v9, &v156, v89, a5, a6);
  v110 = *(_QWORD *)(a1 + 552);
  v111 = v151;
  if ( v151 )
  {
    v123 = v152;
    nullsub_686();
    v136 = v111;
    v137 = v123;
    *(_QWORD *)(v110 + 176) = v111;
    *(_DWORD *)(v110 + 184) = v137;
    sub_1D23870();
  }
  else
  {
    v133 = v152;
    v132 = 0;
    *(_QWORD *)(v110 + 176) = 0;
    *(_DWORD *)(v110 + 184) = v133;
  }
  if ( v178 != (_BYTE *)v127.m128i_i64[0] )
    _libc_free((unsigned __int64)v178);
  if ( v175 != v177 )
    _libc_free((unsigned __int64)v175);
  if ( v172 != v174 )
    _libc_free((unsigned __int64)v172);
  if ( v169 != v128 )
    _libc_free((unsigned __int64)v169);
  if ( v166 )
    sub_161E7C0((__int64)&v166, v166);
  if ( v162 )
    j_j___libc_free_0(v162, (char *)v164 - (char *)v162);
  if ( v145 )
    j_j___libc_free_0(v145, (char *)v147 - (char *)v145);
LABEL_19:
  if ( v143 )
    sub_161E7C0((__int64)v129, v143);
}
