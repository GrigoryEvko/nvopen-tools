// Function: sub_20688C0
// Address: 0x20688c0
//
void __fastcall sub_20688C0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  __int64 *v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 (*v15)(); // rax
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int16 v19; // r13
  unsigned int v20; // edx
  __int64 (__fastcall *v21)(__int64, __int64 *, unsigned __int64, _QWORD, bool, _BYTE **, _BYTE **, __m128i *, __int64); // rbx
  __m128i *v22; // rax
  bool v23; // r14
  unsigned int v24; // r13d
  __int64 v25; // rsi
  __int64 v26; // r12
  int v27; // edx
  int v28; // ebx
  __int64 v29; // r13
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  __int64 *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rax
  __int64 *v37; // r12
  __int64 *v38; // rbx
  __int64 *v39; // r15
  __int64 v40; // rsi
  __int64 v41; // r13
  __int64 v42; // rsi
  __int64 v43; // rbx
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // r8
  __int64 v47; // r9
  __m128i v48; // rax
  unsigned int v49; // edx
  unsigned int v50; // ebx
  unsigned int v51; // r13d
  __int64 *v52; // rax
  int v53; // edx
  int v54; // r12d
  int v55; // r8d
  int v56; // r9d
  __m128i *v57; // rdx
  __m128i *v58; // rax
  unsigned int v59; // r14d
  unsigned int v60; // r12d
  unsigned int v61; // r13d
  __int64 v62; // r10
  __int64 v63; // r11
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned int v70; // edx
  _QWORD *v71; // r10
  int v72; // edx
  _QWORD *v73; // r10
  __int64 v74; // r9
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rax
  __int64 v78; // rsi
  int v79; // edx
  __int64 v80; // r13
  __int64 v81; // r14
  __int64 *v82; // r12
  __int64 v83; // rax
  __int64 v84; // rsi
  __int64 *v85; // rax
  unsigned int v86; // edx
  unsigned int v87; // ebx
  unsigned __int8 v88; // r12
  int v89; // edx
  _BYTE *v90; // r13
  unsigned __int8 v91; // al
  __int64 v92; // r15
  char v93; // r12
  __int64 (__fastcall *v95)(__int64, __int64, __int64, __int64); // rax
  char v96; // bl
  __int64 v97; // rax
  __int64 (__fastcall *v98)(__int64, __int64, __int64, __int64, __int64); // rax
  int v99; // r12d
  unsigned int v100; // r15d
  __int64 (__fastcall *v101)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v102; // rbx
  int v103; // r9d
  __m128i *v104; // r8
  __m128i *v105; // rax
  __int32 v106; // edx
  __m128i *v107; // rax
  __int64 v108; // rsi
  const __m128i *v109; // r9
  __int64 v110; // rbx
  int v111; // r8d
  __int64 v112; // r12
  __int64 v113; // r15
  __int64 v114; // rax
  __int8 v115; // r13
  __int8 v116; // cl
  __int64 v117; // rdx
  __m128i *v118; // rdx
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rbx
  unsigned int v122; // eax
  __int8 v123; // bl
  unsigned int v124; // edx
  int v125; // r8d
  int v126; // r9d
  __int8 v127; // al
  __int64 v128; // rax
  __m128i *v129; // rax
  __m128i v130; // xmm7
  _QWORD *v131; // r12
  unsigned int v132; // edx
  unsigned __int8 v133; // al
  unsigned int v134; // ebx
  int v135; // eax
  __int64 v136; // r8
  __int64 v137; // r9
  _QWORD *v138; // rax
  int v139; // r8d
  int v140; // r9d
  __int64 v141; // rdx
  __int64 v142; // r13
  _QWORD *v143; // r12
  __int64 v144; // rdx
  _QWORD *v145; // rdx
  unsigned int v146; // esi
  __int64 v147; // r14
  __int64 v148; // rdx
  __int128 v149; // [rsp-20h] [rbp-4F0h]
  __int128 v150; // [rsp-10h] [rbp-4E0h]
  __int128 v151; // [rsp-10h] [rbp-4E0h]
  __int64 v152; // [rsp+0h] [rbp-4D0h]
  unsigned __int8 v153; // [rsp+18h] [rbp-4B8h]
  unsigned __int8 v154; // [rsp+20h] [rbp-4B0h]
  int v155; // [rsp+28h] [rbp-4A8h]
  __int64 v156; // [rsp+30h] [rbp-4A0h]
  __int64 v157; // [rsp+38h] [rbp-498h]
  __int64 *v158; // [rsp+38h] [rbp-498h]
  int v159; // [rsp+40h] [rbp-490h]
  __int64 v160; // [rsp+40h] [rbp-490h]
  __int64 v161; // [rsp+48h] [rbp-488h]
  __int64 v162; // [rsp+48h] [rbp-488h]
  __int32 v163; // [rsp+50h] [rbp-480h]
  __int64 v164; // [rsp+50h] [rbp-480h]
  _BYTE *v165; // [rsp+58h] [rbp-478h]
  __int64 v166; // [rsp+60h] [rbp-470h]
  __int64 v167; // [rsp+60h] [rbp-470h]
  unsigned int v168; // [rsp+90h] [rbp-440h]
  __int64 v169; // [rsp+90h] [rbp-440h]
  _QWORD *v170; // [rsp+90h] [rbp-440h]
  unsigned __int16 v171; // [rsp+90h] [rbp-440h]
  char v172; // [rsp+90h] [rbp-440h]
  const __m128i *v173; // [rsp+90h] [rbp-440h]
  __int64 v174; // [rsp+90h] [rbp-440h]
  char v175; // [rsp+90h] [rbp-440h]
  unsigned int v176; // [rsp+90h] [rbp-440h]
  unsigned int v177; // [rsp+98h] [rbp-438h]
  __m128i *v178; // [rsp+98h] [rbp-438h]
  _BYTE *v179; // [rsp+98h] [rbp-438h]
  __int64 *v180; // [rsp+A0h] [rbp-430h]
  __int64 v181; // [rsp+A0h] [rbp-430h]
  const void ***v182; // [rsp+A8h] [rbp-428h]
  __int64 *v183; // [rsp+A8h] [rbp-428h]
  unsigned __int64 v185; // [rsp+B8h] [rbp-418h]
  int v186; // [rsp+C0h] [rbp-410h]
  unsigned int v188; // [rsp+D0h] [rbp-400h]
  unsigned __int64 v189; // [rsp+D8h] [rbp-3F8h]
  int v190; // [rsp+E0h] [rbp-3F0h]
  __int64 v191; // [rsp+E0h] [rbp-3F0h]
  __int64 *v192; // [rsp+E0h] [rbp-3F0h]
  __int64 *v193; // [rsp+E0h] [rbp-3F0h]
  __int64 v194; // [rsp+E0h] [rbp-3F0h]
  __int64 v195; // [rsp+E8h] [rbp-3E8h]
  __int64 v196; // [rsp+130h] [rbp-3A0h]
  __int64 v197; // [rsp+140h] [rbp-390h] BYREF
  int v198; // [rsp+148h] [rbp-388h]
  __int64 v199; // [rsp+150h] [rbp-380h] BYREF
  __int64 v200; // [rsp+158h] [rbp-378h]
  __int64 v201; // [rsp+160h] [rbp-370h]
  __int128 v202; // [rsp+170h] [rbp-360h] BYREF
  __int64 v203; // [rsp+180h] [rbp-350h]
  __m128i v204; // [rsp+190h] [rbp-340h] BYREF
  _BYTE v205[16]; // [rsp+1A0h] [rbp-330h] BYREF
  __m128i v206; // [rsp+1B0h] [rbp-320h] BYREF
  __m128i v207; // [rsp+1C0h] [rbp-310h] BYREF
  __m128i v208; // [rsp+1D0h] [rbp-300h] BYREF
  _BYTE *v209; // [rsp+1E0h] [rbp-2F0h] BYREF
  __int64 v210; // [rsp+1E8h] [rbp-2E8h]
  _BYTE v211[64]; // [rsp+1F0h] [rbp-2E0h] BYREF
  __m128i v212; // [rsp+230h] [rbp-2A0h] BYREF
  __m128i v213; // [rsp+240h] [rbp-290h] BYREF
  __m128i v214; // [rsp+250h] [rbp-280h] BYREF
  _BYTE *v215; // [rsp+280h] [rbp-250h] BYREF
  __int64 v216; // [rsp+288h] [rbp-248h]
  _BYTE v217[128]; // [rsp+290h] [rbp-240h] BYREF
  _BYTE *v218; // [rsp+310h] [rbp-1C0h] BYREF
  __int64 v219; // [rsp+318h] [rbp-1B8h]
  _BYTE v220[432]; // [rsp+320h] [rbp-1B0h] BYREF

  v5 = *(_QWORD *)(a1 + 552);
  v165 = *(_BYTE **)(v5 + 16);
  v156 = sub_1E0A0C0(*(_QWORD *)(v5 + 32));
  v10 = sub_2051DF0((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, a2, v6, v7, v8, v9);
  v11 = *(_QWORD *)(a2 + 40);
  v166 = (__int64)v10;
  v180 = v10;
  v218 = v220;
  v185 = v12;
  v219 = 0x800000000LL;
  v215 = v217;
  v216 = 0x800000000LL;
  if ( !sub_157ECB0(v11) )
  {
    v13 = *(_QWORD *)(a1 + 712);
    if ( *(_BYTE *)(v13 + 40) )
    {
      v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( !v14 )
        goto LABEL_4;
      v209 = v211;
      v210 = 0x400000000LL;
      sub_20C7CE0(v165, v156, **(_QWORD **)(a2 - 24LL * v14), &v209, 0, 0);
      v87 = v210;
      if ( (_DWORD)v210 )
      {
        v88 = 0;
        v158 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a3, a4, a5);
        v155 = v89;
        v162 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
        v212.m128i_i64[0] = *(_QWORD *)(v162 + 112);
        v188 = 142;
        if ( !(unsigned __int8)sub_1560260(&v212, 0, 40) )
        {
          v212.m128i_i64[0] = *(_QWORD *)(v162 + 112);
          v88 = sub_1560260(&v212, 0, 58);
          v188 = (v88 == 0) + 143;
        }
        v167 = sub_15E0530(v162);
        v212.m128i_i64[0] = *(_QWORD *)(v162 + 112);
        v90 = v165;
        v194 = 0;
        v153 = sub_1560260(&v212, 0, 12);
        v91 = 0;
        if ( v188 != 142 )
          v91 = v88;
        v154 = v91;
        v160 = v87;
        while ( 1 )
        {
          v204 = _mm_loadu_si128((const __m128i *)&v209[16 * v194]);
          v92 = *(_QWORD *)v90;
          if ( v188 == 144 )
            goto LABEL_77;
          v93 = v204.m128i_i8[0];
          if ( !(v204.m128i_i8[0]
               ? (unsigned __int8)(v204.m128i_i8[0] - 14) <= 0x47u || (unsigned __int8)(v204.m128i_i8[0] - 2) <= 5u
               : sub_1F58CF0((__int64)&v204)) )
            goto LABEL_77;
          v95 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(v92 + 1264);
          if ( v95 != sub_2046680 )
          {
            v204.m128i_i32[0] = ((__int64 (__fastcall *)(_BYTE *, __int64, _QWORD, __int64, _QWORD))v95)(
                                  v90,
                                  v167,
                                  v204.m128i_u32[0],
                                  v204.m128i_i64[1],
                                  v188);
            v204.m128i_i64[1] = v148;
            v92 = *(_QWORD *)v90;
            goto LABEL_77;
          }
          v206 = _mm_load_si128(&v204);
          v96 = v90[1160];
          v212.m128i_i64[1] = 0;
          v212.m128i_i8[0] = v96;
          if ( v96 == v93 )
            break;
          if ( !v93 )
            goto LABEL_153;
          v176 = sub_2045180(v93);
LABEL_148:
          if ( v96 )
            v146 = sub_2045180(v96);
          else
            v146 = sub_1F58D40((__int64)&v212);
          v97 = 0;
          if ( v146 > v176 )
            goto LABEL_76;
          v97 = v206.m128i_i64[1];
LABEL_75:
          v96 = v93;
LABEL_76:
          v204.m128i_i8[0] = v96;
          v204.m128i_i64[1] = v97;
LABEL_77:
          v171 = *(_WORD *)(v162 + 18);
          v98 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(v92 + 392);
          v99 = (v171 >> 4) & 0x3FF;
          if ( v98 == sub_1F42F80 )
            v100 = sub_1FDDD20((__int64)v90, v167, v204.m128i_i64[0], v204.m128i_i64[1]);
          else
            v100 = v98((__int64)v90, v167, (v171 >> 4) & 0x3FF, v204.m128i_u32[0], v204.m128i_i64[1]);
          v101 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v90 + 384LL);
          if ( v101 == sub_1F42DB0 )
          {
            v102 = v204.m128i_i64[1];
            v206.m128i_i8[0] = v204.m128i_i8[0];
            v206.m128i_i64[1] = v204.m128i_i64[1];
            if ( v204.m128i_i8[0] )
            {
              v103 = (unsigned __int8)v90[v204.m128i_u8[0] + 1155];
            }
            else if ( sub_1F58D20((__int64)&v206) )
            {
              v212.m128i_i8[0] = 0;
              v212.m128i_i64[1] = 0;
              LOBYTE(v199) = 0;
              sub_1F426C0((__int64)v90, v167, v206.m128i_u32[0], v102, (__int64)&v212, (unsigned int *)&v202, &v199);
              v103 = (unsigned __int8)v199;
            }
            else
            {
              sub_1F40D10((__int64)&v212, (__int64)v90, v167, v206.m128i_i64[0], v206.m128i_i64[1]);
              v120 = v152;
              LOBYTE(v120) = v212.m128i_i8[8];
              v152 = v120;
              v103 = sub_1D5E9F0((__int64)v90, v167, (unsigned int)v120, v213.m128i_i64[0]);
            }
          }
          else
          {
            v103 = v101((__int64)v90, v167, (v171 >> 4) & 0x3FF, v204.m128i_u32[0], v204.m128i_i64[1]);
          }
          v104 = &v213;
          v212.m128i_i64[1] = 0x400000000LL;
          v212.m128i_i64[0] = (__int64)&v213;
          if ( v100 > 4 )
          {
            v175 = v103;
            sub_16CD150((__int64)&v212, &v213, v100, 16, (int)&v213, v103);
            v104 = (__m128i *)v212.m128i_i64[0];
            LOBYTE(v103) = v175;
          }
          v212.m128i_i32[2] = v100;
          v105 = &v104[v100];
          if ( v105 != v104 )
          {
            do
            {
              if ( v104 )
              {
                v104->m128i_i64[0] = 0;
                v104->m128i_i32[2] = 0;
              }
              ++v104;
            }
            while ( v105 != v104 );
            v104 = (__m128i *)v212.m128i_i64[0];
          }
          LODWORD(v202) = v99;
          BYTE4(v202) = 1;
          v106 = *(_DWORD *)(a1 + 536);
          v107 = *(__m128i **)a1;
          v206.m128i_i64[0] = 0;
          v206.m128i_i32[2] = v106;
          if ( v107 )
          {
            if ( &v206 != &v107[3] )
            {
              v108 = v107[3].m128i_i64[0];
              v206.m128i_i64[0] = v108;
              if ( v108 )
              {
                v172 = v103;
                v178 = v104;
                sub_1623A60((__int64)&v206, v108, 2);
                LOBYTE(v103) = v172;
                v104 = v178;
              }
            }
          }
          sub_204A2F0(
            *(_QWORD *)(a1 + 552),
            (__int64)&v206,
            (__int64)v158,
            (unsigned int)(v155 + v194),
            (unsigned __int64)v104,
            v100,
            a3,
            *(double *)a4.m128i_i64,
            a5,
            v103,
            a2,
            (__int64)&v202,
            v188);
          if ( v206.m128i_i64[0] )
            sub_161E7C0((__int64)&v206, v206.m128i_i64[0]);
          if ( v100 )
          {
            LODWORD(v109) = v204.m128i_i32[2];
            v179 = v90;
            v110 = 16LL * v100;
            v111 = v204.m128i_u8[0];
            v112 = 0;
            v113 = v204.m128i_i64[1];
            v114 = (4LL * v153) | (2LL * (v188 == 142)) | v154;
            v115 = v204.m128i_i8[0];
            do
            {
              v116 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v112 + v212.m128i_i64[0]) + 40LL)
                              + 16LL * *(unsigned int *)(v112 + v212.m128i_i64[0] + 8));
              v208.m128i_i8[0] = 1;
              *(__int64 *)((char *)v208.m128i_i64 + 4) = 0;
              v206.m128i_i8[8] = v116;
              v207.m128i_i8[0] = v115;
              v206.m128i_i64[0] = v114 | v206.m128i_i32[0] & 0xF8000000;
              v117 = (unsigned int)v219;
              v207.m128i_i64[1] = v113;
              if ( (unsigned int)v219 >= HIDWORD(v219) )
              {
                v174 = v114;
                sub_16CD150((__int64)&v218, v220, 0, 48, v111, (int)v109);
                v117 = (unsigned int)v219;
                v114 = v174;
              }
              a4 = _mm_load_si128(&v206);
              v118 = (__m128i *)&v218[48 * v117];
              *v118 = a4;
              a5 = _mm_load_si128(&v207);
              LODWORD(v219) = v219 + 1;
              v118[1] = a5;
              v118[2] = _mm_load_si128(&v208);
              v119 = (unsigned int)v216;
              v109 = (const __m128i *)(v112 + v212.m128i_i64[0]);
              if ( (unsigned int)v216 >= HIDWORD(v216) )
              {
                v164 = v114;
                v173 = (const __m128i *)(v112 + v212.m128i_i64[0]);
                sub_16CD150((__int64)&v215, v217, 0, 16, v111, (int)v109);
                v119 = (unsigned int)v216;
                v114 = v164;
                v109 = v173;
              }
              a3 = _mm_loadu_si128(v109);
              v112 += 16;
              *(__m128i *)&v215[16 * v119] = a3;
              LODWORD(v216) = v216 + 1;
            }
            while ( v112 != v110 );
            v90 = v179;
          }
          if ( (__m128i *)v212.m128i_i64[0] != &v213 )
            _libc_free(v212.m128i_u64[0]);
          if ( v160 == ++v194 )
            goto LABEL_106;
        }
        v97 = v206.m128i_i64[1];
        if ( v93 || !v206.m128i_i64[1] )
          goto LABEL_75;
LABEL_153:
        v176 = sub_1F58D40((__int64)&v206);
        goto LABEL_148;
      }
LABEL_106:
      if ( v209 != v211 )
        _libc_free((unsigned __int64)v209);
LABEL_4:
      v15 = *(__int64 (**)())(*(_QWORD *)v165 + 1160LL);
      if ( v15 != sub_1D45FE0 )
      {
        v121 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
        if ( ((unsigned __int8 (__fastcall *)(_BYTE *))v15)(v165) )
        {
          v212.m128i_i64[0] = *(_QWORD *)(v121 + 112);
          if ( (unsigned __int8)sub_1560490(&v212, 54, 0) )
          {
            v122 = 8 * sub_15A9520(v156, 0);
            if ( v122 == 32 )
            {
              v123 = 5;
            }
            else if ( v122 > 0x20 )
            {
              v123 = 6;
              if ( v122 != 64 )
              {
                v123 = 0;
                if ( v122 == 128 )
                  v123 = 7;
              }
            }
            else
            {
              v123 = 3;
              if ( v122 != 8 )
                v123 = 4 * (v122 == 16);
            }
            v124 = 8 * sub_15A9520(v156, 0);
            if ( v124 == 32 )
            {
              v127 = 5;
            }
            else if ( v124 > 0x20 )
            {
              v127 = 6;
              if ( v124 != 64 )
              {
                v127 = 0;
                if ( v124 == 128 )
                  v127 = 7;
              }
            }
            else
            {
              v127 = 3;
              if ( v124 != 8 )
                v127 = 4 * (v124 == 16);
            }
            v212.m128i_i8[8] = v127;
            v128 = (unsigned int)v219;
            v212.m128i_i64[0] = 2048;
            v214.m128i_i64[0] = 0x100000001LL;
            v214.m128i_i32[2] = 0;
            v213.m128i_i8[0] = v123;
            v213.m128i_i64[1] = 0;
            if ( (unsigned int)v219 >= HIDWORD(v219) )
            {
              sub_16CD150((__int64)&v218, v220, 0, 48, v125, v126);
              v128 = (unsigned int)v219;
            }
            v129 = (__m128i *)&v218[48 * v128];
            *v129 = _mm_load_si128(&v212);
            v130 = _mm_load_si128(&v213);
            LODWORD(v219) = v219 + 1;
            v129[1] = v130;
            v129[2] = _mm_load_si128(&v214);
            v131 = *(_QWORD **)(a1 + 552);
            v132 = 8 * sub_15A9520(v156, 0);
            if ( v132 == 32 )
            {
              v133 = 5;
            }
            else if ( v132 > 0x20 )
            {
              v133 = 6;
              if ( v132 != 64 )
              {
                v133 = 0;
                if ( v132 == 128 )
                  v133 = 7;
              }
            }
            else
            {
              v133 = 3;
              if ( v132 != 8 )
                v133 = 4 * (v132 == 16);
            }
            v134 = v133;
            v135 = sub_1FE6270(
                     *(_QWORD *)(a1 + 712),
                     a2,
                     *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL),
                     *(_QWORD *)(*(_QWORD *)(a1 + 712) + 176LL));
            v138 = sub_1D2A660(v131, v135, v134, 0, v136, v137);
            v142 = v141;
            v143 = v138;
            v144 = (unsigned int)v216;
            if ( (unsigned int)v216 >= HIDWORD(v216) )
            {
              sub_16CD150((__int64)&v215, v217, 0, 16, v139, v140);
              v144 = (unsigned int)v216;
            }
            v145 = &v215[16 * v144];
            *v145 = v143;
            v145[1] = v142;
            LODWORD(v216) = v216 + 1;
          }
        }
      }
      v16 = *(_QWORD *)(a1 + 552);
      v17 = *(_QWORD *)(v16 + 16);
      v18 = **(_QWORD **)(v16 + 32);
      v19 = *(_WORD *)(v18 + 18);
      v20 = *(_DWORD *)(*(_QWORD *)(v18 + 24) + 8LL);
      v21 = *(__int64 (__fastcall **)(__int64, __int64 *, unsigned __int64, _QWORD, bool, _BYTE **, _BYTE **, __m128i *, __int64))(*(_QWORD *)v17 + 1224LL);
      v22 = *(__m128i **)a1;
      v212.m128i_i64[0] = 0;
      v23 = v20 >> 8 != 0;
      v212.m128i_i32[2] = *(_DWORD *)(a1 + 536);
      v24 = (v19 >> 4) & 0x3FF;
      if ( v22 )
      {
        if ( &v212 != &v22[3] )
        {
          v25 = v22[3].m128i_i64[0];
          v212.m128i_i64[0] = v25;
          if ( v25 )
            sub_1623A60((__int64)&v212, v25, 2);
        }
      }
      v26 = v21(v17, v180, v185, v24, v23, &v218, &v215, &v212, v16);
      v28 = v27;
      if ( v212.m128i_i64[0] )
        sub_161E7C0((__int64)&v212, v212.m128i_i64[0]);
      v29 = *(_QWORD *)(a1 + 552);
      if ( v26 )
      {
        nullsub_686();
        *(_QWORD *)(v29 + 176) = v26;
        *(_DWORD *)(v29 + 184) = v28;
        sub_1D23870();
      }
      else
      {
        *(_QWORD *)(v29 + 176) = 0;
        *(_DWORD *)(v29 + 184) = v28;
      }
      v30 = (unsigned __int64)v215;
      if ( v215 != v217 )
        goto LABEL_14;
      goto LABEL_15;
    }
    v190 = *(_DWORD *)(v13 + 44);
    v31 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
    v204.m128i_i64[0] = (__int64)v205;
    v204.m128i_i64[1] = 0x100000000LL;
    v32 = **(__int64 ***)(*(_QWORD *)(v31 + 24) + 16LL);
    v33 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
    v34 = sub_1647190(v32, *(_DWORD *)(v33 + 4));
    sub_20C7CE0(v165, v156, v34, &v204, 0, 0);
    v35 = *(_DWORD *)(a1 + 536);
    v36 = *(_QWORD *)a1;
    v209 = 0;
    v37 = *(__int64 **)(a1 + 552);
    v38 = (__int64 *)v204.m128i_i64[0];
    LODWORD(v210) = v35;
    if ( v36 )
    {
      v39 = v37;
      if ( &v209 != (_BYTE **)(v36 + 48) )
      {
        v40 = *(_QWORD *)(v36 + 48);
        v209 = (_BYTE *)v40;
        if ( v40 )
        {
          sub_1623A60((__int64)&v209, v40, 2);
          v39 = *(__int64 **)(a1 + 552);
        }
      }
    }
    else
    {
      v39 = v37;
    }
    v41 = v38[1];
    v42 = (unsigned int)*v38;
    v43 = *v38;
    v44 = sub_1D252B0((__int64)v37, v42, v41, 1, 0);
    v186 = v45;
    v182 = (const void ***)v44;
    v212.m128i_i64[0] = (__int64)(v39 + 11);
    v212.m128i_i32[2] = 0;
    v48.m128i_i64[0] = (__int64)sub_1D2A660(v37, v190, v43, v41, v46, v47);
    v213 = v48;
    *((_QWORD *)&v150 + 1) = 2;
    *(_QWORD *)&v150 = &v212;
    v183 = sub_1D36D80(
             v37,
             47,
             (__int64)&v209,
             v182,
             v186,
             *(double *)a3.m128i_i64,
             *(double *)a4.m128i_i64,
             a5,
             (__int64)v182,
             v150);
    v50 = v49;
    v51 = v49;
    if ( v209 )
      sub_161E7C0((__int64)&v209, (__int64)v209);
    v52 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a3, a4, a5);
    v210 = 0x400000000LL;
    v161 = (__int64)v52;
    v54 = v53;
    v209 = v211;
    v206.m128i_i64[0] = (__int64)&v207;
    LODWORD(v52) = *(_DWORD *)(a2 + 20);
    v206.m128i_i64[1] = 0x400000000LL;
    sub_20C7CE0(v165, v156, **(_QWORD **)(a2 - 24LL * ((unsigned int)v52 & 0xFFFFFFF)), &v209, &v206, 0);
    v212.m128i_i64[1] = 0x400000000LL;
    v163 = v210;
    v212.m128i_i64[0] = (__int64)&v213;
    if ( (unsigned int)v210 > 4 )
    {
      v147 = 16LL * (unsigned int)v210;
      sub_16CD150((__int64)&v212, &v213, (unsigned int)v210, 16, v55, v56);
      v212.m128i_i32[2] = v163;
      v58 = (__m128i *)v212.m128i_i64[0];
      v57 = (__m128i *)(v212.m128i_i64[0] + v147);
    }
    else
    {
      v57 = &v213 + (unsigned int)v210;
      v212.m128i_i32[2] = v210;
      v58 = &v213;
      if ( v57 == &v213 )
      {
LABEL_29:
        if ( v163 )
        {
          v159 = v54;
          v59 = 0;
          v157 = v51;
          v60 = v177;
          v61 = v168;
          v181 = 16LL * v50;
          do
          {
            v62 = *(_QWORD *)(a1 + 552);
            v63 = *(_QWORD *)(v206.m128i_i64[0] + 8LL * v59);
            v64 = *(_QWORD *)a1;
            LODWORD(v200) = *(_DWORD *)(a1 + 536);
            v199 = 0;
            if ( v64 )
            {
              if ( &v199 != (__int64 *)(v64 + 48) )
              {
                v65 = *(_QWORD *)(v64 + 48);
                v199 = v65;
                if ( v65 )
                {
                  v169 = v63;
                  v191 = v62;
                  sub_1623A60((__int64)&v199, v65, 2);
                  v63 = v169;
                  v62 = v191;
                }
              }
            }
            v192 = (__int64 *)v62;
            v66 = v183[5] + v181;
            LOBYTE(v60) = *(_BYTE *)v66;
            v67 = sub_1D38BB0(
                    v62,
                    v63,
                    (__int64)&v199,
                    v60,
                    *(const void ***)(v66 + 8),
                    0,
                    a3,
                    *(double *)a4.m128i_i64,
                    a5,
                    0);
            v68 = v183[5] + v181;
            LOBYTE(v61) = *(_BYTE *)v68;
            v189 = v157 | v189 & 0xFFFFFFFF00000000LL;
            *((_QWORD *)&v149 + 1) = v69;
            *(_QWORD *)&v149 = v67;
            v193 = sub_1D332F0(
                     v192,
                     52,
                     (__int64)&v199,
                     v61,
                     *(const void ***)(v68 + 8),
                     3u,
                     *(double *)a3.m128i_i64,
                     *(double *)a4.m128i_i64,
                     a5,
                     (__int64)v183,
                     v189,
                     v149);
            v195 = v70;
            if ( v199 )
              sub_161E7C0((__int64)&v199, v199);
            v71 = *(_QWORD **)(a1 + 552);
            v199 = 0;
            v200 = 0;
            v201 = 0;
            v170 = v71;
            sub_1E34280((__int64)&v202, v71[4]);
            v72 = *(_DWORD *)(a1 + 536);
            v197 = 0;
            v73 = v170;
            v74 = v59 + v159;
            v75 = *(_QWORD *)a1;
            v198 = v72;
            if ( v75 )
            {
              if ( &v197 != (__int64 *)(v75 + 48) )
              {
                v76 = *(_QWORD *)(v75 + 48);
                v197 = v76;
                if ( v76 )
                {
                  sub_1623A60((__int64)&v197, v76, 2);
                  v74 = v59 + v159;
                  v73 = v170;
                }
              }
            }
            v196 = sub_1D2BF40(
                     v73,
                     v166,
                     v185,
                     (__int64)&v197,
                     v161,
                     v74,
                     (__int64)v193,
                     v195,
                     v202,
                     v203,
                     0,
                     0,
                     (__int64)&v199);
            v77 = v212.m128i_i64[0] + 16LL * v59;
            *(_QWORD *)v77 = v196;
            v78 = v197;
            *(_DWORD *)(v77 + 8) = v79;
            if ( v78 )
              sub_161E7C0((__int64)&v197, v78);
            ++v59;
          }
          while ( v59 != v163 );
        }
        v80 = v212.m128i_i64[0];
        v199 = 0;
        v81 = v212.m128i_u32[2];
        v82 = *(__int64 **)(a1 + 552);
        v83 = *(_QWORD *)a1;
        LODWORD(v200) = *(_DWORD *)(a1 + 536);
        if ( v83 )
        {
          if ( &v199 != (__int64 *)(v83 + 48) )
          {
            v84 = *(_QWORD *)(v83 + 48);
            v199 = v84;
            if ( v84 )
              sub_1623A60((__int64)&v199, v84, 2);
          }
        }
        *((_QWORD *)&v151 + 1) = v81;
        *(_QWORD *)&v151 = v80;
        v85 = sub_1D359D0(v82, 2, (__int64)&v199, 1, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v151);
        v185 = v86 | v185 & 0xFFFFFFFF00000000LL;
        v180 = v85;
        if ( v199 )
          sub_161E7C0((__int64)&v199, v199);
        if ( (__m128i *)v212.m128i_i64[0] != &v213 )
          _libc_free(v212.m128i_u64[0]);
        if ( (__m128i *)v206.m128i_i64[0] != &v207 )
          _libc_free(v206.m128i_u64[0]);
        if ( v209 != v211 )
          _libc_free((unsigned __int64)v209);
        if ( (_BYTE *)v204.m128i_i64[0] != v205 )
          _libc_free(v204.m128i_u64[0]);
        goto LABEL_4;
      }
    }
    do
    {
      if ( v58 )
      {
        v58->m128i_i64[0] = 0;
        v58->m128i_i32[2] = 0;
      }
      ++v58;
    }
    while ( v57 != v58 );
    goto LABEL_29;
  }
  sub_2098CA0(a1);
  v30 = (unsigned __int64)v215;
  if ( v215 != v217 )
LABEL_14:
    _libc_free(v30);
LABEL_15:
  if ( v218 != v220 )
    _libc_free((unsigned __int64)v218);
}
