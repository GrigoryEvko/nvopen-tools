// Function: sub_3765180
// Address: 0x3765180
//
void __fastcall sub_3765180(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // ebx
  int v9; // r12d
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int16 *v15; // rdx
  __int16 v16; // ax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int16 *v19; // rcx
  unsigned __int32 v20; // r12d
  int v21; // kr00_4
  __int64 v22; // rdx
  unsigned __int64 *v23; // rax
  __m128i v24; // xmm0
  __int16 v25; // r14d^2
  __int128 v26; // xmm2
  unsigned __int16 *v27; // rdx
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rsi
  unsigned int v31; // ebx
  int v32; // eax
  __int64 v33; // rcx
  __m128i *v34; // rax
  __m128i *v35; // rdx
  __m128i *v36; // rsi
  __m128i *i; // rdx
  unsigned int v38; // r13d
  __int64 v39; // r12
  bool v40; // al
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rax
  _QWORD *v46; // r14
  __int128 v47; // rax
  __int64 v48; // r9
  __int128 v49; // rax
  _QWORD *v50; // r14
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned __int8 *v53; // r14
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // rax
  unsigned int v57; // eax
  __int64 v58; // r14
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 *v65; // rcx
  _QWORD *v66; // r15
  __int128 v67; // rax
  __int64 v68; // r9
  unsigned __int8 *v69; // r10
  __int64 v70; // rdx
  __int64 v71; // r11
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rax
  __int16 v75; // dx
  __int64 v76; // rax
  char v77; // si
  __int64 v78; // rcx
  unsigned __int16 *v79; // rdx
  __int64 v80; // r8
  unsigned __int16 v81; // dx
  unsigned __int8 v82; // cl
  __int64 v83; // r13
  __int64 v84; // rbx
  __int64 v85; // rsi
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // r9
  unsigned __int16 *v89; // r11
  _QWORD *v90; // rdi
  __int64 v91; // rdx
  __m128i v92; // xmm0
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rsi
  __m128i v97; // xmm4
  __m128i v98; // xmm5
  __m128i v99; // xmm6
  __int64 v100; // r8
  unsigned __int8 *v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rdx
  unsigned __int8 *v104; // rax
  __int64 v105; // r8
  unsigned __int8 *v106; // r12
  unsigned int v107; // edx
  unsigned int v108; // r13d
  __int64 v109; // r9
  __int64 v110; // rax
  unsigned __int8 **v111; // rax
  __int64 *v112; // rsi
  __int64 *v113; // rax
  __m128i v114; // xmm0
  unsigned int v115; // ebx
  unsigned __int16 *v116; // rax
  unsigned __int8 *v117; // r12
  __int64 *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // r13
  __int64 v121; // r9
  __int128 v122; // rax
  unsigned __int8 *v123; // rax
  __int64 v124; // rdx
  __int64 v125; // rdx
  __m128i v126; // xmm7
  __m128i v127; // xmm3
  __m128i v128; // xmm7
  unsigned __int8 *v129; // rax
  __int64 v130; // rdx
  __int64 v131; // rdx
  __int128 v132; // [rsp-48h] [rbp-2D8h]
  __int128 v133; // [rsp-38h] [rbp-2C8h]
  __int128 v134; // [rsp-20h] [rbp-2B0h]
  __int128 v135; // [rsp-20h] [rbp-2B0h]
  __int128 v136; // [rsp-20h] [rbp-2B0h]
  __int128 v137; // [rsp-20h] [rbp-2B0h]
  __int128 v138; // [rsp-20h] [rbp-2B0h]
  __int128 v139; // [rsp-10h] [rbp-2A0h]
  __int64 v140; // [rsp-10h] [rbp-2A0h]
  __int128 v141; // [rsp-10h] [rbp-2A0h]
  __int128 v142; // [rsp-10h] [rbp-2A0h]
  __int64 v143; // [rsp-8h] [rbp-298h]
  __int128 v145; // [rsp+20h] [rbp-270h]
  __int64 v146; // [rsp+30h] [rbp-260h]
  unsigned __int64 v147; // [rsp+38h] [rbp-258h]
  __int64 v148; // [rsp+40h] [rbp-250h]
  __int16 v149; // [rsp+4Eh] [rbp-242h]
  __int64 v150; // [rsp+50h] [rbp-240h]
  __int64 (__fastcall *v151)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+58h] [rbp-238h]
  __int64 v152; // [rsp+60h] [rbp-230h]
  __int64 v153; // [rsp+60h] [rbp-230h]
  __int64 v154; // [rsp+68h] [rbp-228h]
  _QWORD *v155; // [rsp+70h] [rbp-220h]
  unsigned __int8 *v156; // [rsp+70h] [rbp-220h]
  __int64 v157; // [rsp+78h] [rbp-218h]
  __int128 v158; // [rsp+80h] [rbp-210h]
  __int128 v159; // [rsp+80h] [rbp-210h]
  unsigned __int32 v160; // [rsp+80h] [rbp-210h]
  __int64 v161; // [rsp+90h] [rbp-200h]
  unsigned int v162; // [rsp+90h] [rbp-200h]
  __int64 v163; // [rsp+98h] [rbp-1F8h]
  char v164; // [rsp+98h] [rbp-1F8h]
  __int64 v165; // [rsp+A0h] [rbp-1F0h]
  __int16 v166; // [rsp+A2h] [rbp-1EEh]
  unsigned __int8 v167; // [rsp+B0h] [rbp-1E0h]
  _QWORD *v168; // [rsp+B0h] [rbp-1E0h]
  unsigned __int64 v169; // [rsp+B8h] [rbp-1D8h]
  __m128i v170; // [rsp+C0h] [rbp-1D0h] BYREF
  unsigned __int8 *v171; // [rsp+D0h] [rbp-1C0h]
  __int64 v172; // [rsp+D8h] [rbp-1B8h]
  unsigned __int8 *v173; // [rsp+E0h] [rbp-1B0h]
  __int64 v174; // [rsp+E8h] [rbp-1A8h]
  unsigned __int8 *v175; // [rsp+F0h] [rbp-1A0h]
  __int64 v176; // [rsp+F8h] [rbp-198h]
  __int64 v177; // [rsp+100h] [rbp-190h]
  __int64 v178; // [rsp+108h] [rbp-188h]
  unsigned __int8 *v179; // [rsp+110h] [rbp-180h]
  __int64 v180; // [rsp+118h] [rbp-178h]
  unsigned __int8 *v181; // [rsp+120h] [rbp-170h]
  __int64 v182; // [rsp+128h] [rbp-168h]
  __int64 v183; // [rsp+130h] [rbp-160h]
  __int64 v184; // [rsp+138h] [rbp-158h]
  __int64 v185; // [rsp+140h] [rbp-150h]
  __int64 v186; // [rsp+148h] [rbp-148h]
  bool v187; // [rsp+15Fh] [rbp-131h] BYREF
  __m128i v188; // [rsp+160h] [rbp-130h] BYREF
  __m128i v189; // [rsp+170h] [rbp-120h] BYREF
  __m128i v190; // [rsp+180h] [rbp-110h] BYREF
  __m128i v191; // [rsp+190h] [rbp-100h] BYREF
  __int128 v192; // [rsp+1A0h] [rbp-F0h] BYREF
  __int64 v193; // [rsp+1B0h] [rbp-E0h] BYREF
  int v194; // [rsp+1B8h] [rbp-D8h]
  __int64 v195; // [rsp+1C0h] [rbp-D0h] BYREF
  __int64 v196; // [rsp+1C8h] [rbp-C8h]
  __m128i v197; // [rsp+1D0h] [rbp-C0h] BYREF
  __m128i v198; // [rsp+1E0h] [rbp-B0h] BYREF
  __m128i v199; // [rsp+1F0h] [rbp-A0h]
  __m128i v200; // [rsp+200h] [rbp-90h]
  __int64 v201; // [rsp+210h] [rbp-80h]
  unsigned int v202; // [rsp+218h] [rbp-78h]

  v9 = *(_DWORD *)(a2 + 24);
  v170.m128i_i64[0] = a1;
  v10 = *(__int64 **)(a2 + 40);
  v11 = *(_QWORD *)(a1 + 8);
  v187 = 0;
  if ( v9 == 147 )
  {
    v77 = 0;
    goto LABEL_25;
  }
  if ( v9 == 148 )
  {
    v77 = 1;
LABEL_25:
    v78 = v10[5];
    v189 = _mm_loadu_si128((const __m128i *)(v10 + 5));
    v188 = _mm_loadu_si128((const __m128i *)v10);
    v79 = (unsigned __int16 *)(*(_QWORD *)(v78 + 48) + 16LL * v189.m128i_u32[2]);
    v191 = _mm_loadu_si128((const __m128i *)(v10 + 15));
    v190 = _mm_loadu_si128((const __m128i *)v10 + 5);
    v80 = *(int *)(v191.m128i_i64[0] + 96);
    v81 = *v79;
    if ( ((*(_DWORD *)(v11 + 4 * ((v81 >> 3) + 36 * v80 - v80) + 521536) >> (4 * (v81 & 7))) & 0xF) != 2 )
    {
      sub_3763F80((__int64 *)v170.m128i_i64[0], a2, a3, 4 * (v81 & 7u), v80, 9 * v80, a4);
      return;
    }
    v164 = 1;
    v82 = v77;
LABEL_30:
    v162 = 0;
    v83 = 0;
    v84 = 0;
    v160 = 0;
    goto LABEL_31;
  }
  v188.m128i_i64[0] = 0;
  v188.m128i_i32[2] = 0;
  v12 = *v10;
  v190 = _mm_loadu_si128((const __m128i *)(v10 + 5));
  v189 = _mm_loadu_si128((const __m128i *)v10);
  v191 = _mm_loadu_si128((const __m128i *)v10 + 5);
  LOWORD(v12) = *(_WORD *)(*(_QWORD *)(v12 + 48) + 16LL * v189.m128i_u32[2]);
  v13 = *(int *)(v191.m128i_i64[0] + 96);
  v14 = 9 * v13;
  if ( ((*(_DWORD *)(v11 + 4 * (((unsigned __int16)v12 >> 3) + 36 * v13 - v13) + 521536) >> (4 * (v12 & 7))) & 0xF) != 2 )
  {
    v15 = *(__int16 **)(a2 + 48);
    v16 = *v15;
    v17 = *((_QWORD *)v15 + 1);
    LOWORD(v192) = v16;
    *((_QWORD *)&v192 + 1) = v17;
    if ( v16 )
    {
      if ( (unsigned __int16)(v16 - 176) > 0x34u )
        goto LABEL_44;
    }
    else if ( !sub_3007100((__int64)&v192) )
    {
LABEL_6:
      v20 = sub_3007130((__int64)&v192, v13);
LABEL_7:
      v21 = sub_3009970((__int64)&v192, v13, v18, (__int64)v19, v14);
      HIWORD(v7) = HIWORD(v21);
      v149 = v21;
      v161 = v22;
LABEL_8:
      v23 = *(unsigned __int64 **)(a2 + 40);
      v24 = _mm_loadu_si128((const __m128i *)v23);
      v25 = HIWORD(v7);
      v26 = (__int128)_mm_loadu_si128((const __m128i *)v23 + 5);
      v169 = v24.m128i_u64[1];
      v147 = *v23;
      v146 = *((unsigned int *)v23 + 2);
      v27 = (unsigned __int16 *)(*(_QWORD *)(*v23 + 48) + 16 * v146);
      v145 = (__int128)_mm_loadu_si128((const __m128i *)(v23 + 5));
      v28 = *v27;
      v29 = *((_QWORD *)v27 + 1);
      v197.m128i_i16[0] = v28;
      v197.m128i_i64[1] = v29;
      if ( (_WORD)v28 )
      {
        v163 = 0;
        LOWORD(v28) = word_4456580[v28 - 1];
      }
      else
      {
        v28 = sub_3009970((__int64)&v197, v13, v29, (__int64)v19, v14);
        v166 = HIWORD(v28);
        v163 = v125;
      }
      HIWORD(v31) = v166;
      v30 = *(_QWORD *)(a2 + 80);
      LOWORD(v31) = v28;
      v193 = v30;
      if ( v30 )
        sub_B96E90((__int64)&v193, v30, 1);
      v32 = *(_DWORD *)(a2 + 72);
      v197.m128i_i64[1] = 0x800000000LL;
      v33 = v20;
      v194 = v32;
      v34 = &v198;
      v148 = v20;
      v35 = &v198;
      v36 = &v198;
      v197.m128i_i64[0] = (__int64)&v198;
      if ( v20 )
      {
        if ( v20 > 8uLL )
        {
          sub_C8D5F0((__int64)&v197, &v198, v20, 0x10u, v14, a7);
          v35 = (__m128i *)v197.m128i_i64[0];
          v34 = (__m128i *)(v197.m128i_i64[0] + 16LL * v197.m128i_u32[2]);
        }
        for ( i = &v35[v20]; i != v34; ++v34 )
        {
          if ( v34 )
          {
            v34->m128i_i64[0] = 0;
            v34->m128i_i32[2] = 0;
          }
        }
        v197.m128i_i32[2] = v20;
        HIWORD(v38) = v25;
        v39 = 0;
        do
        {
          v46 = *(_QWORD **)v170.m128i_i64[0];
          *(_QWORD *)&v47 = sub_3400EE0(*(_QWORD *)v170.m128i_i64[0], v39, (__int64)&v193, 0, v24);
          v169 = v146 | v169 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v49 = sub_3406EB0(v46, 0x9Eu, (__int64)&v193, v31, v163, v48, __PAIR128__(v169, v147), v47);
          v50 = *(_QWORD **)v170.m128i_i64[0];
          v158 = v49;
          *(_QWORD *)&v51 = sub_3400EE0(*(_QWORD *)v170.m128i_i64[0], v39, (__int64)&v193, 0, v24);
          v53 = sub_3406EB0(v50, 0x9Eu, (__int64)&v193, v31, v163, v52, v145, v51);
          v55 = v54;
          v155 = *(_QWORD **)v170.m128i_i64[0];
          v150 = *(_QWORD *)(v170.m128i_i64[0] + 8);
          v152 = *(_QWORD *)(*(_QWORD *)v170.m128i_i64[0] + 64LL);
          v151 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v150 + 528LL);
          v56 = sub_2E79000(*(__int64 **)(*(_QWORD *)v170.m128i_i64[0] + 40LL));
          v57 = v151(v150, v56, v152, v31, v163);
          *((_QWORD *)&v135 + 1) = v55;
          *(_QWORD *)&v135 = v53;
          v58 = 16 * v39;
          v60 = sub_340F900(v155, 0xD0u, (__int64)&v193, v57, v59, (__int64)v155, v158, v135, v26);
          v62 = v61;
          v63 = v60;
          v64 = v197.m128i_i64[0];
          v65 = (__int64 *)v170.m128i_i64[0];
          v186 = v62;
          LOWORD(v38) = v149;
          v185 = v63;
          *(_QWORD *)(v197.m128i_i64[0] + 16 * v39) = v63;
          *(_DWORD *)(v64 + v58 + 8) = v186;
          v66 = (_QWORD *)*v65;
          *(_QWORD *)&v67 = sub_3400BD0(*v65, 0, (__int64)&v193, v38, v161, 0, v24, 0);
          v159 = v67;
          v69 = sub_3401740(*(_QWORD *)v170.m128i_i64[0], 1, (__int64)&v193, v38, v161, v68, v192);
          v71 = v70;
          v72 = *(_QWORD *)(16 * v39 + v197.m128i_i64[0]);
          v73 = *(_QWORD *)(16 * v39 + v197.m128i_i64[0] + 8);
          v74 = *(_QWORD *)(*(_QWORD *)(v58 + v197.m128i_i64[0]) + 48LL)
              + 16LL * *(unsigned int *)(v58 + v197.m128i_i64[0] + 8);
          v75 = *(_WORD *)v74;
          v76 = *(_QWORD *)(v74 + 8);
          LOWORD(v195) = v75;
          v196 = v76;
          if ( v75 )
          {
            v40 = (unsigned __int16)(v75 - 17) <= 0xD3u;
          }
          else
          {
            v153 = v72;
            v154 = v73;
            v156 = v69;
            v157 = v71;
            v40 = sub_30070B0((__int64)&v195);
            v72 = v153;
            v73 = v154;
            v69 = v156;
            v71 = v157;
          }
          *((_QWORD *)&v134 + 1) = v71;
          ++v39;
          *(_QWORD *)&v134 = v69;
          v41 = sub_340EC60(v66, 205 - ((unsigned int)!v40 - 1), (__int64)&v193, v38, v161, 0, v72, v73, v134, v159);
          v43 = v42;
          v44 = v41;
          v45 = v197.m128i_i64[0];
          v183 = v44;
          v184 = v43;
          *(_QWORD *)(v197.m128i_i64[0] + v58) = v44;
          *(_DWORD *)(v45 + v58 + 8) = v184;
        }
        while ( v39 != v148 );
        v36 = (__m128i *)v197.m128i_i64[0];
        v33 = v197.m128i_u32[2];
      }
      *((_QWORD *)&v139 + 1) = v33;
      *(_QWORD *)&v139 = v36;
      v104 = sub_33FC220(*(_QWORD **)v170.m128i_i64[0], 156, (__int64)&v193, v192, *((__int64 *)&v192 + 1), a7, v139);
      v105 = v140;
      v106 = v104;
      v108 = v107;
      v109 = v143;
      if ( (__m128i *)v197.m128i_i64[0] != &v198 )
        _libc_free(v197.m128i_u64[0]);
      if ( v193 )
        sub_B91220((__int64)&v193, v193);
      v110 = *(unsigned int *)(a3 + 8);
      if ( v110 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v110 + 1, 0x10u, v105, v109);
        v110 = *(unsigned int *)(a3 + 8);
      }
      v111 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v110);
      *v111 = v106;
      v111[1] = (unsigned __int8 *)v108;
      ++*(_DWORD *)(a3 + 8);
      return;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v192 )
      goto LABEL_6;
    if ( (unsigned __int16)(v192 - 176) > 0x34u )
    {
      v95 = (unsigned __int16)v192 - 1;
      v20 = word_4456340[v95];
      goto LABEL_45;
    }
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_44:
    v19 = word_4456340;
    v18 = (unsigned __int16)v192;
    v95 = (unsigned __int16)v192 - 1;
    v20 = word_4456340[v95];
    if ( !(_WORD)v192 )
      goto LABEL_7;
LABEL_45:
    v161 = 0;
    v149 = word_4456580[v95];
    goto LABEL_8;
  }
  v164 = 0;
  v82 = 0;
  if ( v9 != 463 )
    goto LABEL_30;
  v112 = v10 + 15;
  v113 = v10 + 20;
  v84 = *v112;
  v83 = *v113;
  v160 = *((_DWORD *)v112 + 2);
  v162 = *((_DWORD *)v113 + 2);
LABEL_31:
  v85 = *(_QWORD *)(a2 + 80);
  v195 = v85;
  if ( v85 )
  {
    v167 = v82;
    sub_B96E90((__int64)&v195, v85, 1);
    v82 = v167;
    v11 = *(_QWORD *)(v170.m128i_i64[0] + 8);
  }
  LODWORD(v196) = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v133 + 1) = v162;
  *(_QWORD *)&v133 = v83;
  *((_QWORD *)&v132 + 1) = v160;
  *(_QWORD *)&v132 = v84;
  if ( !(unsigned __int8)sub_3470350(
                           v11,
                           *(_QWORD **)v170.m128i_i64[0],
                           **(unsigned __int16 **)(a2 + 48),
                           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                           &v189,
                           v190.m128i_i64,
                           a4,
                           &v191,
                           v132,
                           v133,
                           &v187,
                           (__int64)&v195,
                           (unsigned __int8 **)&v188,
                           v82) )
  {
    v115 = **(unsigned __int16 **)(a2 + 48);
    v165 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v116 = (unsigned __int16 *)(*(_QWORD *)(v189.m128i_i64[0] + 48) + 16LL * v189.m128i_u32[2]);
    v168 = *(_QWORD **)v170.m128i_i64[0];
    *((_QWORD *)&v141 + 1) = *((_QWORD *)v116 + 1);
    *(_QWORD *)&v141 = *v116;
    v117 = sub_3401740((__int64)v168, 0, (__int64)&v195, **(unsigned __int16 **)(a2 + 48), v165, (__int64)v168, v141);
    v118 = (__int64 *)v170.m128i_i64[0];
    v120 = v119;
    v170.m128i_i64[0] = v165;
    *((_QWORD *)&v136 + 1) = *(_QWORD *)(*(_QWORD *)(v189.m128i_i64[0] + 48) + 16LL * v189.m128i_u32[2] + 8);
    *(_QWORD *)&v136 = *(unsigned __int16 *)(*(_QWORD *)(v189.m128i_i64[0] + 48) + 16LL * v189.m128i_u32[2]);
    *(_QWORD *)&v122 = sub_3401740(*v118, 1, (__int64)&v195, v115, v165, v121, v136);
    *((_QWORD *)&v137 + 1) = v120;
    *(_QWORD *)&v137 = v117;
    v123 = sub_33FC1D0(
             v168,
             207,
             (__int64)&v195,
             v115,
             v165,
             (__int64)v168,
             *(_OWORD *)&v189,
             *(_OWORD *)&v190,
             v122,
             v137,
             *(_OWORD *)&v191);
    v172 = v124;
    v171 = v123;
    v189.m128i_i32[2] = v124;
    LODWORD(v124) = *(_DWORD *)(a2 + 28);
    v189.m128i_i64[0] = (__int64)v123;
    *((_DWORD *)v123 + 7) = v124;
    goto LABEL_38;
  }
  if ( !v191.m128i_i64[0] )
  {
LABEL_47:
    if ( !v187 )
      goto LABEL_38;
    if ( v9 != 463 )
    {
LABEL_49:
      v175 = sub_3407510(
               *(_QWORD **)v170.m128i_i64[0],
               (__int64)&v195,
               v189.m128i_i64[0],
               v189.m128i_i64[1],
               **(unsigned __int16 **)(v189.m128i_i64[0] + 48),
               *(_QWORD *)(*(_QWORD *)(v189.m128i_i64[0] + 48) + 8LL));
      v176 = v103;
      v189.m128i_i64[0] = (__int64)v175;
      v189.m128i_i32[2] = v103;
      goto LABEL_38;
    }
LABEL_68:
    *((_QWORD *)&v142 + 1) = *(_QWORD *)(*(_QWORD *)(v189.m128i_i64[0] + 48) + 8LL);
    *(_QWORD *)&v142 = **(unsigned __int16 **)(v189.m128i_i64[0] + 48);
    *((_QWORD *)&v138 + 1) = v162;
    *(_QWORD *)&v138 = v83;
    v173 = sub_3401870(
             *(_QWORD **)v170.m128i_i64[0],
             (__int64)&v195,
             v189.m128i_i64[0],
             v189.m128i_i64[1],
             v84,
             v160,
             v138,
             v142);
    v174 = v131;
    v189.m128i_i64[0] = (__int64)v173;
    v189.m128i_i32[2] = v131;
    goto LABEL_38;
  }
  v88 = *(unsigned int *)(a2 + 28);
  v89 = *(unsigned __int16 **)(a2 + 48);
  v90 = *(_QWORD **)v170.m128i_i64[0];
  if ( v164 )
  {
    v96 = *(unsigned int *)(a2 + 24);
    v97 = _mm_load_si128(&v189);
    v98 = _mm_load_si128(&v190);
    v99 = _mm_load_si128(&v191);
    v100 = *(unsigned int *)(a2 + 68);
    v197 = _mm_load_si128(&v188);
    v198 = v97;
    v199 = v98;
    v200 = v99;
    v101 = sub_3410740(v90, v96, (__int64)&v195, (unsigned int *)v89, v100, v88, a4, &v197, 4);
    v188.m128i_i32[2] = 1;
    v182 = v102;
    v181 = v101;
    v189.m128i_i64[0] = (__int64)v101;
    v189.m128i_i32[2] = v102;
    v188.m128i_i64[0] = (__int64)v101;
    goto LABEL_47;
  }
  if ( v9 == 463 )
  {
    v201 = v83;
    v126 = _mm_load_si128(&v189);
    v127 = _mm_load_si128(&v191);
    v200.m128i_i64[0] = v84;
    v200.m128i_i32[2] = v160;
    v197 = v126;
    v128 = _mm_load_si128(&v190);
    v202 = v162;
    v198 = v128;
    v199 = v127;
    v129 = sub_33FBA10(v90, 463, (__int64)&v195, *v89, *((_QWORD *)v89 + 1), v88, (__int64)&v197, 5);
    v180 = v130;
    v179 = v129;
    v189.m128i_i64[0] = (__int64)v129;
    v189.m128i_i32[2] = v130;
    if ( !v187 )
      goto LABEL_38;
    goto LABEL_68;
  }
  v177 = sub_340EC60(
           v90,
           0xD0u,
           (__int64)&v195,
           *v89,
           *((_QWORD *)v89 + 1),
           v88,
           v189.m128i_i64[0],
           v189.m128i_i64[1],
           *(_OWORD *)&v190,
           *(_OWORD *)&v191);
  v178 = v91;
  v189.m128i_i64[0] = v177;
  v189.m128i_i32[2] = v91;
  if ( v187 )
    goto LABEL_49;
LABEL_38:
  v92 = _mm_load_si128(&v189);
  v93 = *(unsigned int *)(a3 + 8);
  if ( v93 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v170 = v92;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v93 + 1, 0x10u, v86, v87);
    v93 = *(unsigned int *)(a3 + 8);
    v92 = _mm_load_si128(&v170);
  }
  *(__m128i *)(*(_QWORD *)a3 + 16 * v93) = v92;
  v170.m128i_i32[0] = *(_DWORD *)(a3 + 8);
  v94 = (unsigned int)(v170.m128i_i32[0] + 1);
  *(_DWORD *)(a3 + 8) = v94;
  if ( v164 )
  {
    v114 = _mm_load_si128(&v188);
    if ( v94 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v170 = v114;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v94 + 1, 0x10u, v86, v87);
      v94 = *(unsigned int *)(a3 + 8);
      v114 = _mm_load_si128(&v170);
    }
    *(__m128i *)(*(_QWORD *)a3 + 16 * v94) = v114;
    ++*(_DWORD *)(a3 + 8);
  }
  if ( v195 )
    sub_B91220((__int64)&v195, v195);
}
