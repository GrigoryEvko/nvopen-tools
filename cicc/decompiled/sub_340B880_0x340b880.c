// Function: sub_340B880
// Address: 0x340b880
//
unsigned __int8 *__fastcall sub_340B880(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        unsigned __int64 a10,
        unsigned __int8 a11,
        char a12,
        char a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        const __m128i *a17)
{
  unsigned __int8 *v17; // r13
  __int64 v19; // r12
  char v20; // bl
  bool v21; // r15
  int v22; // eax
  char v23; // r14
  bool v24; // al
  int v25; // edx
  unsigned __int8 (__fastcall *v26)(_DWORD *, const __m128i **, _QWORD, _BYTE **, _QWORD, __int64, __int64 *); // rbx
  unsigned int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rbx
  const __m128i *v33; // r13
  const __m128i *v34; // rbx
  unsigned __int16 v35; // r12
  unsigned __int64 v36; // r15
  __int64 v37; // rdx
  char v38; // r14
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __m128i v43; // xmm1
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // r12
  unsigned __int64 v47; // rcx
  unsigned __int8 *v48; // rax
  unsigned __int64 v49; // rdx
  __m128i *v50; // rax
  __int64 v51; // r9
  __m128i *v52; // rdx
  __m128i *v53; // r15
  __int64 v54; // rdx
  __m128i *v55; // r14
  unsigned __int64 v56; // r8
  __m128i **v57; // rdx
  unsigned __int16 v58; // ax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r9
  __m128i v62; // xmm0
  __int64 v63; // rdx
  __int64 v64; // rdx
  char v65; // al
  unsigned __int64 v66; // rax
  unsigned __int16 v67; // r15
  unsigned __int16 v68; // cx
  unsigned __int64 v69; // r14
  unsigned __int64 v70; // rbx
  __int64 v71; // rax
  unsigned __int64 v72; // rsi
  __int64 v73; // rdx
  char v74; // r8
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rdx
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rdx
  unsigned __int64 v82; // rbx
  __int64 v83; // rax
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  unsigned __int16 v86; // r14
  unsigned __int64 v87; // rbx
  unsigned __int16 v88; // r14
  __int64 v89; // rax
  __int16 v90; // bx
  __int64 v91; // rax
  int v92; // edx
  int v93; // ebx
  unsigned __int64 v94; // rdx
  __int64 (*v95)(); // rax
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  unsigned __int16 v100; // ax
  __int64 v101; // rdx
  __int64 v102; // rdx
  int v103; // eax
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  __int64 v106; // rdx
  __int64 (*v107)(); // rbx
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rdx
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // rax
  __int64 v115; // rdx
  int v116; // r9d
  __int64 v117; // rdx
  unsigned __int8 *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // r15
  unsigned __int8 *v121; // r14
  __int128 v122; // rax
  __int64 v123; // r9
  unsigned __int8 *v124; // rax
  int v125; // edx
  int v126; // ebx
  unsigned __int8 *v127; // rdx
  __int64 v128; // rsi
  __int64 v129; // r14
  unsigned __int8 v130; // bl
  __int64 v131; // r13
  int v132; // eax
  __int64 v133; // rdx
  int v134; // r9d
  unsigned __int8 *v135; // rax
  int v136; // edx
  int v137; // ebx
  unsigned __int8 *v138; // rdx
  __int128 v139; // [rsp-30h] [rbp-340h]
  __int128 v140; // [rsp-10h] [rbp-320h]
  __int64 v141; // [rsp+10h] [rbp-300h]
  unsigned __int64 v142; // [rsp+18h] [rbp-2F8h]
  _DWORD *v143; // [rsp+20h] [rbp-2F0h]
  __int64 *v144; // [rsp+28h] [rbp-2E8h]
  __int16 v145; // [rsp+30h] [rbp-2E0h]
  char v146; // [rsp+30h] [rbp-2E0h]
  unsigned int v147; // [rsp+30h] [rbp-2E0h]
  unsigned int v148; // [rsp+30h] [rbp-2E0h]
  unsigned __int64 v149; // [rsp+38h] [rbp-2D8h]
  __int64 v150; // [rsp+48h] [rbp-2C8h]
  __int64 v151; // [rsp+50h] [rbp-2C0h]
  unsigned int v152; // [rsp+50h] [rbp-2C0h]
  bool v153; // [rsp+68h] [rbp-2A8h]
  unsigned int v154; // [rsp+70h] [rbp-2A0h]
  unsigned __int64 v155; // [rsp+70h] [rbp-2A0h]
  unsigned __int64 v156; // [rsp+78h] [rbp-298h]
  unsigned __int8 v157; // [rsp+87h] [rbp-289h]
  unsigned __int64 v162; // [rsp+A0h] [rbp-270h]
  __int64 v164; // [rsp+B8h] [rbp-258h]
  unsigned int v165; // [rsp+FCh] [rbp-214h] BYREF
  __m128i v166; // [rsp+100h] [rbp-210h] BYREF
  __m128i v167; // [rsp+110h] [rbp-200h] BYREF
  unsigned int v168; // [rsp+120h] [rbp-1F0h] BYREF
  __int64 v169; // [rsp+128h] [rbp-1E8h]
  __int64 v170; // [rsp+130h] [rbp-1E0h] BYREF
  __int64 v171; // [rsp+138h] [rbp-1D8h]
  __int64 v172; // [rsp+140h] [rbp-1D0h]
  __int64 v173; // [rsp+148h] [rbp-1C8h]
  __int64 v174; // [rsp+150h] [rbp-1C0h] BYREF
  __int64 v175; // [rsp+158h] [rbp-1B8h]
  __int64 v176; // [rsp+160h] [rbp-1B0h]
  __int64 v177; // [rsp+168h] [rbp-1A8h]
  __int64 v178; // [rsp+170h] [rbp-1A0h]
  __int64 v179; // [rsp+178h] [rbp-198h]
  __int64 v180; // [rsp+180h] [rbp-190h]
  __int64 v181; // [rsp+188h] [rbp-188h]
  __int64 v182; // [rsp+190h] [rbp-180h] BYREF
  __int64 v183; // [rsp+198h] [rbp-178h]
  __int64 v184; // [rsp+1A0h] [rbp-170h]
  __int64 v185; // [rsp+1A8h] [rbp-168h]
  __int64 v186; // [rsp+1B0h] [rbp-160h]
  __int64 v187; // [rsp+1B8h] [rbp-158h]
  __int64 v188; // [rsp+1C0h] [rbp-150h]
  __int64 v189; // [rsp+1C8h] [rbp-148h]
  __int64 v190; // [rsp+1D0h] [rbp-140h]
  __int64 v191; // [rsp+1D8h] [rbp-138h]
  __int64 v192; // [rsp+1E0h] [rbp-130h]
  __int64 v193; // [rsp+1E8h] [rbp-128h]
  const __m128i *v194; // [rsp+1F0h] [rbp-120h] BYREF
  __int64 v195; // [rsp+1F8h] [rbp-118h]
  __int64 v196; // [rsp+200h] [rbp-110h]
  __int128 v197; // [rsp+210h] [rbp-100h]
  __int64 v198; // [rsp+220h] [rbp-F0h]
  __int64 v199; // [rsp+230h] [rbp-E0h] BYREF
  __int64 v200; // [rsp+238h] [rbp-D8h]
  __m128i v201; // [rsp+240h] [rbp-D0h]
  _BYTE *v202; // [rsp+250h] [rbp-C0h] BYREF
  __int64 v203; // [rsp+258h] [rbp-B8h]
  _BYTE v204[176]; // [rsp+260h] [rbp-B0h] BYREF

  if ( *(_DWORD *)(a8 + 24) == 51 )
    return (unsigned __int8 *)a3;
  v194 = 0;
  v195 = 0;
  v19 = *(_QWORD *)(a1 + 40);
  v157 = a11;
  v196 = 0;
  v143 = *(_DWORD **)(a1 + 16);
  v150 = *(_QWORD *)(v19 + 48);
  v20 = sub_33CC5F0((__int64 *)v19, a1);
  v151 = a5;
  v21 = *(_DWORD *)(a5 + 24) == 15 || *(_DWORD *)(a5 + 24) == 39;
  if ( v21 )
  {
    v22 = *(_DWORD *)(a5 + 96);
    v23 = 1;
    if ( v22 < 0 )
    {
      v21 = v22 < -*(_DWORD *)(v150 + 32);
      v23 = v21;
    }
  }
  else
  {
    v151 = 0;
    v23 = 0;
  }
  v24 = sub_33CF170(a8);
  v25 = -1;
  if ( !a13 )
  {
    if ( v20 )
      v25 = v143[134243];
    else
      v25 = v143[134242];
  }
  v153 = v24;
  v154 = v25;
  v26 = *(unsigned __int8 (__fastcall **)(_DWORD *, const __m128i **, _QWORD, _BYTE **, _QWORD, __int64, __int64 *))(*(_QWORD *)v143 + 1984LL);
  v199 = *(_QWORD *)(*(_QWORD *)v19 + 120LL);
  v27 = sub_2EAC1E0((__int64)&a14);
  LOBYTE(v203) = v23;
  BYTE3(v203) = 1;
  v202 = (_BYTE *)a10;
  BYTE4(v203) = v153;
  BYTE1(v203) = a11;
  *(_WORD *)((char *)&v203 + 5) = 0;
  BYTE2(v203) = a12 ^ 1;
  if ( v26(v143, &v194, v154, &v202, v27, 0xFFFFFFFFLL, &v199) )
  {
    if ( v21 )
    {
      v128 = sub_3007410((__int64)v194, *(__int64 **)(a1 + 64), v28, v29, v30, v31);
      v129 = sub_2E79000(*(__int64 **)(a1 + 40));
      v130 = sub_AE5020(v129, v128);
      v131 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v19 + 16) + 200LL))(*(_QWORD *)(v19 + 16));
      if ( (!(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v131 + 544LL))(v131, v19)
         || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v131 + 536LL))(v131, v19))
        && *(_BYTE *)(v129 + 17)
        && v130 > *(_BYTE *)(v129 + 16) )
      {
        v130 = *(_BYTE *)(v129 + 16);
      }
      if ( v130 > a11 )
      {
        v132 = *(_DWORD *)(v151 + 96);
        v133 = *(_QWORD *)(v150 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v150 + 32) + v132);
        if ( *(_BYTE *)(v133 + 16) < v130 )
        {
          *(_BYTE *)(v133 + 16) = v130;
          if ( (*(_BYTE *)(*(_QWORD *)(v150 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v150 + 32) + v132) + 20) & 0xFD) == 0 )
            sub_2E76F70(v150, v130);
        }
        v157 = v130;
      }
    }
    v202 = v204;
    v203 = 0x800000000LL;
    v32 = (v195 - (__int64)v194) >> 4;
    v152 = v32;
    v166 = _mm_loadu_si128(v194);
    if ( (unsigned int)v32 > 1 )
    {
      v33 = v194 + 1;
      v34 = &v194[(unsigned int)(v32 - 2) + 2];
      while ( 1 )
      {
        v35 = v33->m128i_i16[0];
        if ( v166.m128i_i16[0] == v33->m128i_i16[0] )
        {
          if ( v35 || v33->m128i_i64[1] == v166.m128i_i64[1] )
            goto LABEL_18;
          v175 = v166.m128i_i64[1];
          LOWORD(v174) = 0;
        }
        else
        {
          LOWORD(v174) = v166.m128i_i16[0];
          v175 = v166.m128i_i64[1];
          if ( v166.m128i_i16[0] )
          {
            if ( v166.m128i_i16[0] == 1 || (unsigned __int16)(v166.m128i_i16[0] - 504) <= 7u )
LABEL_138:
              BUG();
            v36 = *(_QWORD *)&byte_444C4A0[16 * v166.m128i_u16[0] - 16];
            v38 = byte_444C4A0[16 * v166.m128i_u16[0] - 8];
            goto LABEL_22;
          }
        }
        v178 = sub_3007260((__int64)&v174);
        v36 = v178;
        v179 = v37;
        v38 = v37;
LABEL_22:
        if ( v35 )
        {
          if ( v35 == 1 || (unsigned __int16)(v35 - 504) <= 7u )
            goto LABEL_138;
          v42 = *(_QWORD *)&byte_444C4A0[16 * v35 - 16];
          LOBYTE(v41) = byte_444C4A0[16 * v35 - 8];
        }
        else
        {
          v39 = sub_3007260((__int64)v33);
          v41 = v40;
          v176 = v39;
          v42 = v39;
          v177 = v41;
        }
        if ( !(_BYTE)v41 && v38 || v42 <= v36 )
        {
LABEL_18:
          if ( v34 == ++v33 )
            goto LABEL_28;
        }
        else
        {
          v43 = _mm_loadu_si128(v33++);
          v166 = v43;
          if ( v34 == v33 )
          {
LABEL_28:
            v44 = sub_3408940(a8, a9, v166.m128i_u32[0], v166.m128i_i64[1], a1, a2, a7);
            v200 = 0;
            v155 = v44;
            v156 = v45;
            v199 = 0;
            v201 = _mm_loadu_si128(a17 + 1);
            goto LABEL_29;
          }
        }
      }
    }
    v104 = sub_3408940(a8, a9, v166.m128i_u32[0], v166.m128i_i64[1], a1, a2, a7);
    v200 = 0;
    v155 = v104;
    v156 = v105;
    v199 = 0;
    v201 = _mm_loadu_si128(a17 + 1);
    if ( !(_DWORD)v32 )
    {
LABEL_101:
      *((_QWORD *)&v140 + 1) = (unsigned int)v203;
      *(_QWORD *)&v140 = v202;
      v17 = sub_33FC220((_QWORD *)a1, 2, a2, 1, 0, v61, v140);
      if ( v202 != v204 )
        _libc_free((unsigned __int64)v202);
      goto LABEL_12;
    }
LABEL_29:
    v164 = 0;
    v46 = 0;
    v142 = v156 & 0xFFFFFFFF00000000LL;
    v149 = a3;
    while ( 1 )
    {
      v62 = _mm_loadu_si128(&v194[v164]);
      v167 = v62;
      if ( v62.m128i_i16[0] )
      {
        if ( v62.m128i_i16[0] == 1 || (unsigned __int16)(v62.m128i_i16[0] - 504) <= 7u )
          goto LABEL_138;
        v96 = 16LL * (v62.m128i_u16[0] - 1);
        v64 = *(_QWORD *)&byte_444C4A0[v96];
        v65 = byte_444C4A0[v96 + 8];
      }
      else
      {
        v180 = sub_3007260((__int64)&v167);
        v181 = v63;
        v64 = v180;
        v65 = v181;
      }
      v182 = v64;
      LOBYTE(v183) = v65;
      v66 = sub_CA1930(&v182);
      v67 = v166.m128i_i16[0];
      v68 = v167.m128i_i16[0];
      v69 = v156;
      v162 = (unsigned int)(v66 >> 3);
      v70 = v155;
      if ( v162 > a10 )
        v46 = v46 + a10 - v162;
      if ( v167.m128i_i16[0] == v166.m128i_i16[0] )
        break;
      LOWORD(v182) = v166.m128i_i16[0];
      v183 = v166.m128i_i64[1];
      if ( !v166.m128i_i16[0] )
        goto LABEL_44;
      if ( v166.m128i_i16[0] == 1 || (unsigned __int16)(v166.m128i_i16[0] - 504) <= 7u )
        goto LABEL_138;
      v72 = *(_QWORD *)&byte_444C4A0[16 * v166.m128i_u16[0] - 16];
      v74 = byte_444C4A0[16 * v166.m128i_u16[0] - 8];
LABEL_45:
      if ( v68 )
      {
        if ( v68 == 1 || (unsigned __int16)(v68 - 504) <= 7u )
          goto LABEL_138;
        v79 = *(_QWORD *)&byte_444C4A0[16 * v68 - 16];
        LOBYTE(v78) = byte_444C4A0[16 * v68 - 8];
      }
      else
      {
        v146 = v74;
        v75 = sub_3007260((__int64)&v167);
        v74 = v146;
        v76 = v75;
        v78 = v77;
        v184 = v76;
        v79 = v76;
        v185 = v78;
      }
      if ( (_BYTE)v78 && !v74 || v72 <= v79 )
        goto LABEL_31;
      if ( v67 )
      {
        if ( v67 == 1 || (unsigned __int16)(v67 - 504) <= 7u )
          goto LABEL_138;
        v81 = 16LL * (v67 - 1);
        v80 = *(_QWORD *)&byte_444C4A0[v81];
        LOBYTE(v81) = byte_444C4A0[v81 + 8];
      }
      else
      {
        v80 = sub_3007260((__int64)&v166);
        v190 = v80;
        v191 = v81;
      }
      v170 = v80;
      LOBYTE(v171) = v81;
      v82 = sub_CA1930(&v170);
      if ( v167.m128i_i16[0] )
      {
        if ( v167.m128i_i16[0] == 1 || (unsigned __int16)(v167.m128i_i16[0] - 504) <= 7u )
          goto LABEL_138;
        v84 = 16LL * (v167.m128i_u16[0] - 1);
        v83 = *(_QWORD *)&byte_444C4A0[v84];
        LOBYTE(v84) = byte_444C4A0[v84 + 8];
      }
      else
      {
        v83 = sub_3007260((__int64)&v167);
        v188 = v83;
        v189 = v84;
      }
      LOBYTE(v183) = v84;
      v182 = v83;
      v85 = sub_CA1930(&v182);
      v86 = v167.m128i_i16[0];
      v87 = v82 / v85;
      if ( v167.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v167.m128i_i16[0] - 17) > 0xD3u )
          goto LABEL_56;
        v141 = 0;
        v86 = word_4456580[v167.m128i_u16[0] - 1];
      }
      else
      {
        if ( !sub_30070B0((__int64)&v167) )
        {
LABEL_56:
          v141 = v167.m128i_i64[1];
          goto LABEL_57;
        }
        v100 = sub_3009970((__int64)&v167, v72, v97, v98, v99);
        v141 = v101;
        v86 = v100;
      }
LABEL_57:
      v147 = v86;
      v144 = *(__int64 **)(a1 + 64);
      v88 = sub_2D43050(v86, v87);
      v89 = 0;
      if ( !v88 )
      {
        v88 = sub_3009400(v144, v147, v141, (unsigned int)v87, 0);
        v89 = v106;
      }
      v90 = v166.m128i_i16[0];
      LOWORD(v168) = v88;
      v169 = v89;
      if ( v166.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v166.m128i_i16[0] - 17) <= 0xD3u )
          goto LABEL_61;
      }
      else if ( sub_30070B0((__int64)&v166) )
      {
        goto LABEL_61;
      }
      if ( v167.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v167.m128i_i16[0] - 17) > 0xD3u )
        {
LABEL_75:
          v95 = *(__int64 (**)())(*(_QWORD *)v143 + 1392LL);
          if ( v95 != sub_2FE3480 )
          {
            if ( ((unsigned __int8 (__fastcall *)(_DWORD *, _QWORD, __int64, _QWORD, __int64))v95)(
                   v143,
                   v166.m128i_u32[0],
                   v166.m128i_i64[1],
                   v167.m128i_u32[0],
                   v167.m128i_i64[1]) )
            {
              v135 = sub_33FAF80(a1, 216, a2, v167.m128i_u32[0], v167.m128i_i64[1], v134, v62);
              v137 = v136;
              v138 = v135;
              LODWORD(v135) = v137;
              v70 = (unsigned __int64)v138;
              v69 = v142 | (unsigned int)v135;
              goto LABEL_31;
            }
            v90 = v166.m128i_i16[0];
          }
        }
      }
      else if ( !sub_30070B0((__int64)&v167) )
      {
        goto LABEL_75;
      }
      if ( v90 )
      {
        if ( (unsigned __int16)(v90 - 17) > 0xD3u )
          goto LABEL_63;
      }
      else if ( !sub_30070B0((__int64)&v166) )
      {
        goto LABEL_63;
      }
LABEL_61:
      if ( v167.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v167.m128i_i16[0] - 17) <= 0xD3u )
          goto LABEL_63;
      }
      else if ( sub_30070B0((__int64)&v167) )
      {
        goto LABEL_63;
      }
      v107 = *(__int64 (**)())(*(_QWORD *)v143 + 496LL);
      v108 = sub_2D5B750((unsigned __int16 *)&v167);
      v171 = v109;
      v170 = v108;
      v148 = sub_CA1930(&v170);
      v114 = sub_3007410((__int64)&v166, *(__int64 **)(a1 + 64), v110, v111, v112, v113);
      if ( v107 == sub_2FE30D0
        || !((unsigned __int8 (__fastcall *)(_DWORD *, __int64, _QWORD, unsigned int *))v107)(v143, v114, v148, &v165)
        || !v88
        || !*(_QWORD *)&v143[2 * v88 + 28]
        || (v182 = sub_2D5B750((unsigned __int16 *)&v168),
            v183 = v115,
            v174 = sub_2D5B750((unsigned __int16 *)&v166),
            v175 = v117,
            v174 != v182)
        || (_BYTE)v175 != (_BYTE)v183 )
      {
LABEL_63:
        v91 = sub_3408940(a8, a9, v167.m128i_u32[0], v167.m128i_i64[1], a1, a2, v62);
        v93 = v92;
        v94 = v91;
        LODWORD(v91) = v93;
        v70 = v94;
        v69 = v142 | (unsigned int)v91;
        goto LABEL_31;
      }
      v118 = sub_33FAF80(a1, 234, a2, v168, v169, v116, v62);
      v120 = v119;
      v121 = v118;
      *(_QWORD *)&v122 = sub_3400EE0(a1, v165, a2, 0, v62);
      *((_QWORD *)&v139 + 1) = v120;
      *(_QWORD *)&v139 = v121;
      v124 = sub_3406EB0((_QWORD *)a1, 0x9Eu, a2, v167.m128i_u32[0], v167.m128i_i64[1], v123, v139, v122);
      v126 = v125;
      v127 = v124;
      LODWORD(v124) = v126;
      v70 = (unsigned __int64)v127;
      v69 = v142 | (unsigned int)v124;
LABEL_31:
      v47 = a14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a14 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (a14 & 4) != 0 )
        {
          *((_QWORD *)&v197 + 1) = v46 + a15;
          BYTE4(v198) = BYTE4(a16);
          *(_QWORD *)&v197 = v47 | 4;
          LODWORD(v198) = *(_DWORD *)(v47 + 12);
        }
        else
        {
          *((_QWORD *)&v197 + 1) = v46 + a15;
          v102 = *(_QWORD *)(v47 + 8);
          *(_QWORD *)&v197 = a14 & 0xFFFFFFFFFFFFFFF8LL;
          v103 = *(unsigned __int8 *)(v102 + 8);
          BYTE4(v198) = BYTE4(a16);
          if ( (unsigned int)(v103 - 17) <= 1 )
            v102 = **(_QWORD **)(v102 + 16);
          LODWORD(v198) = *(_DWORD *)(v102 + 8) >> 8;
        }
      }
      else
      {
        *((_QWORD *)&v197 + 1) = v46 + a15;
        *(_QWORD *)&v197 = 0;
        LODWORD(v198) = a16;
        BYTE4(v198) = 0;
      }
      LOBYTE(v173) = 0;
      v172 = v46;
      v48 = sub_3409320((_QWORD *)a1, a5, a6, v46, 0, a2, v62, 0);
      v50 = sub_33F4560(
              (_QWORD *)a1,
              v149,
              a4,
              a2,
              v70,
              v69,
              (unsigned __int64)v48,
              v49,
              v197,
              v198,
              v157,
              a12 == 0 ? 0 : 4,
              (__int64)&v199);
      v53 = v52;
      v54 = (unsigned int)v203;
      v55 = v50;
      v56 = (unsigned int)v203 + 1LL;
      if ( v56 > HIDWORD(v203) )
      {
        sub_C8D5F0((__int64)&v202, v204, (unsigned int)v203 + 1LL, 0x10u, v56, v51);
        v54 = (unsigned int)v203;
      }
      v57 = (__m128i **)&v202[16 * v54];
      *v57 = v55;
      v58 = v167.m128i_i16[0];
      v57[1] = v53;
      LODWORD(v203) = v203 + 1;
      if ( v58 )
      {
        if ( v58 == 1 || (unsigned __int16)(v58 - 504) <= 7u )
          goto LABEL_138;
        v60 = 16LL * (v58 - 1);
        v59 = *(_QWORD *)&byte_444C4A0[v60];
        LOBYTE(v60) = byte_444C4A0[v60 + 8];
      }
      else
      {
        v59 = sub_3007260((__int64)&v167);
        v192 = v59;
        v193 = v60;
      }
      v182 = v59;
      LOBYTE(v183) = v60;
      ++v164;
      v46 += (unsigned __int64)sub_CA1930(&v182) >> 3;
      a10 -= v162;
      if ( v152 <= (unsigned int)v164 )
        goto LABEL_101;
    }
    if ( v167.m128i_i16[0] || v166.m128i_i64[1] == v167.m128i_i64[1] )
      goto LABEL_31;
    v183 = v166.m128i_i64[1];
    LOWORD(v182) = 0;
LABEL_44:
    v145 = v167.m128i_i16[0];
    v71 = sub_3007260((__int64)&v182);
    v68 = v145;
    v186 = v71;
    v72 = v71;
    v187 = v73;
    v74 = v73;
    goto LABEL_45;
  }
  v17 = 0;
LABEL_12:
  if ( v194 )
    j_j___libc_free_0((unsigned __int64)v194);
  return v17;
}
