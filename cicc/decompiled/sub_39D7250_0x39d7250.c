// Function: sub_39D7250
// Address: 0x39d7250
//
__int64 __fastcall sub_39D7250(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 (*v6)(void); // rax
  int v7; // esi
  int v8; // eax
  int v9; // ebx
  __int64 v10; // rdx
  __int32 v11; // r15d
  __int64 v12; // r9
  unsigned int v13; // edi
  int *v14; // rax
  int v15; // r8d
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r14
  __m128i si128; // xmm4
  __m128i v20; // xmm5
  __int64 v21; // rax
  unsigned int v22; // esi
  int v23; // r11d
  int v24; // eax
  int v25; // r11d
  __int64 v26; // r8
  int v27; // r9d
  unsigned int v28; // r10d
  int *v29; // rdx
  int v30; // edi
  int v31; // esi
  int *v32; // r15
  int v33; // r15d
  __int32 v34; // r14d
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // r10
  unsigned int v38; // edi
  _DWORD *v39; // rdx
  int v40; // r9d
  __int64 v41; // rax
  __int64 v42; // rdi
  char *v43; // rsi
  __int64 v44; // rdx
  _BYTE *v45; // rdi
  __int8 *v46; // rax
  __int64 v47; // rdi
  int v48; // edx
  __int64 v49; // rbx
  char v50; // al
  __m128i v51; // xmm4
  __m128i v52; // xmm5
  __int64 v53; // rdx
  __int64 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // r8
  int v57; // esi
  unsigned int v58; // edi
  int *v59; // r14
  int v60; // eax
  int v61; // r10d
  __int64 v62; // rdx
  int v63; // ebx
  int v64; // ecx
  int v65; // r8d
  unsigned int v66; // r10d
  int *v67; // rax
  int v68; // r15d
  __int64 v69; // rax
  bool v70; // zf
  __int64 v71; // rdi
  __int64 v72; // r9
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r8
  __int64 v76; // r15
  __int64 result; // rax
  __int64 v78; // r14
  __int64 v79; // r13
  __int64 v80; // rdx
  __int64 v81; // rdi
  int v82; // ecx
  unsigned int v83; // esi
  int *v84; // rax
  int v85; // r10d
  __int64 v86; // rdx
  unsigned __int8 *v87; // rdi
  __int64 v88; // rsi
  __int64 v89; // rcx
  __int64 v90; // r15
  __m128i *v91; // rdx
  __int64 m128i_i64; // r9
  __int64 v93; // rax
  __int64 v94; // r14
  __int64 v95; // r13
  __int64 v96; // r15
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rax
  int v100; // eax
  int v101; // eax
  int v102; // r14d
  int v103; // eax
  _DWORD *v104; // rcx
  int v105; // eax
  int v106; // edi
  int v107; // r9d
  int v108; // r15d
  int v109; // eax
  int v110; // r8d
  int v111; // eax
  int v112; // esi
  __int64 v113; // r10
  unsigned int v114; // r11d
  int v115; // edx
  int v116; // eax
  _DWORD *v117; // r14
  int v118; // eax
  int v119; // esi
  __int64 v120; // r10
  int v121; // eax
  unsigned int v122; // r11d
  int v123; // edx
  int v124; // r11d
  int v125; // r11d
  __int64 v126; // r8
  int v127; // esi
  unsigned int v128; // r10d
  __int64 v131; // [rsp+20h] [rbp-220h]
  int v132; // [rsp+28h] [rbp-218h]
  __int64 v133; // [rsp+30h] [rbp-210h]
  __int64 v134; // [rsp+38h] [rbp-208h]
  __int64 v135; // [rsp+48h] [rbp-1F8h]
  __int64 v136; // [rsp+50h] [rbp-1F0h]
  __int64 v137; // [rsp+50h] [rbp-1F0h]
  __int32 v138; // [rsp+58h] [rbp-1E8h]
  __int32 v139; // [rsp+58h] [rbp-1E8h]
  __int64 v140; // [rsp+58h] [rbp-1E8h]
  __int64 v141; // [rsp+58h] [rbp-1E8h]
  __m128i *v143; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v144; // [rsp+78h] [rbp-1C8h]
  __m128i v145; // [rsp+80h] [rbp-1C0h] BYREF
  __int32 v146; // [rsp+90h] [rbp-1B0h]
  char v147; // [rsp+94h] [rbp-1ACh]
  unsigned __int64 v148; // [rsp+A0h] [rbp-1A0h] BYREF
  __m128i n; // [rsp+A8h] [rbp-198h] BYREF
  __m128i v150; // [rsp+B8h] [rbp-188h] BYREF
  _QWORD *v151; // [rsp+C8h] [rbp-178h]
  __m128i v152; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v153; // [rsp+E0h] [rbp-160h] BYREF
  void *dest; // [rsp+E8h] [rbp-158h]
  __m128i v155; // [rsp+F0h] [rbp-150h] BYREF
  int v156; // [rsp+100h] [rbp-140h] BYREF
  __int16 v157; // [rsp+104h] [rbp-13Ch]
  char v158; // [rsp+106h] [rbp-13Ah]
  __m128i v159; // [rsp+108h] [rbp-138h] BYREF
  __int64 v160; // [rsp+118h] [rbp-128h] BYREF
  __int64 v161; // [rsp+120h] [rbp-120h]
  __m128i v162; // [rsp+128h] [rbp-118h] BYREF
  unsigned __int64 v163; // [rsp+138h] [rbp-108h]
  __int64 *v164; // [rsp+140h] [rbp-100h]
  __int64 v165; // [rsp+148h] [rbp-F8h] BYREF
  __int64 v166; // [rsp+150h] [rbp-F0h] BYREF
  __m128i v167; // [rsp+158h] [rbp-E8h] BYREF
  __int64 v168; // [rsp+168h] [rbp-D8h]
  __m128i **v169; // [rsp+170h] [rbp-D0h]
  __int64 v170; // [rsp+178h] [rbp-C8h]
  __m128i *v171; // [rsp+180h] [rbp-C0h] BYREF
  __int64 v172; // [rsp+188h] [rbp-B8h]
  __m128i v173; // [rsp+190h] [rbp-B0h] BYREF
  __m128i v174; // [rsp+1A0h] [rbp-A0h] BYREF
  __m128i *v175; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v176; // [rsp+1B8h] [rbp-88h]
  __m128i v177; // [rsp+1C0h] [rbp-80h] BYREF
  __m128i v178; // [rsp+1D0h] [rbp-70h] BYREF
  _BYTE *v179; // [rsp+1E0h] [rbp-60h]
  __int64 v180; // [rsp+1E8h] [rbp-58h]
  _BYTE v181[16]; // [rsp+1F0h] [rbp-50h] BYREF
  __m128i v182[4]; // [rsp+200h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a3 + 56);
  v133 = 0;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(a3 + 16) + 112LL);
  if ( v6 != sub_1D00B10 )
    v133 = v6();
  v7 = *(_DWORD *)(v5 + 32);
  v8 = v7;
  v9 = -v7;
  if ( v7 > 0 )
  {
    v10 = *(_QWORD *)(v5 + 8);
    v11 = 0;
    while ( *(_QWORD *)(v10 + 40LL * (unsigned int)(v7 + v9) + 8) == -1 )
    {
LABEL_19:
      v8 = v7;
      if ( !++v9 )
        goto LABEL_40;
    }
    v152.m128i_i64[1] = 0;
    v153 = 0;
    v159 = (__m128i)(unsigned __int64)&v160;
    v164 = &v166;
    v169 = &v171;
    v174 = (__m128i)(unsigned __int64)&v175;
    LODWORD(dest) = 0;
    v155 = 0u;
    v156 = 0;
    v157 = 0;
    v158 = 0;
    LOBYTE(v160) = 0;
    v162 = 0u;
    LOBYTE(v163) = 1;
    v165 = 0;
    LOBYTE(v166) = 0;
    v167.m128i_i64[1] = 0;
    v168 = 0;
    v170 = 0;
    LOBYTE(v171) = 0;
    v173 = 0u;
    LOBYTE(v175) = 0;
    v177 = 0u;
    v152.m128i_i32[0] = v11;
    v16 = 5LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9);
    LODWORD(dest) = *(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9) + 21);
    v155.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 8 * v16);
    v155.m128i_i64[1] = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9) + 8);
    v156 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9) + 16);
    LOBYTE(v157) = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9) + 23);
    v17 = 0;
    if ( !*(_BYTE *)(v5 + 657) )
      v17 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9) + 20);
    HIBYTE(v157) = v17;
    v158 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v9) + 33);
    v18 = a2[39];
    if ( v18 == a2[40] )
    {
      sub_39D63D0(a2 + 38, a2[39], (__int64)&v152);
    }
    else
    {
      if ( v18 )
      {
        *(__m128i *)v18 = _mm_load_si128(&v152);
        *(_QWORD *)(v18 + 16) = v153;
        *(_DWORD *)(v18 + 24) = (_DWORD)dest;
        *(__m128i *)(v18 + 32) = v155;
        *(_DWORD *)(v18 + 48) = v156;
        *(_WORD *)(v18 + 52) = v157;
        *(_BYTE *)(v18 + 54) = v158;
        *(_QWORD *)(v18 + 56) = v18 + 72;
        sub_39CF630((__int64 *)(v18 + 56), v159.m128i_i64[0], v159.m128i_i64[0] + v159.m128i_i64[1]);
        *(__m128i *)(v18 + 88) = _mm_loadu_si128(&v162);
        *(_BYTE *)(v18 + 104) = v163;
        *(_QWORD *)(v18 + 112) = v18 + 128;
        sub_39CF630((__int64 *)(v18 + 112), v164, (__int64)v164 + v165);
        si128 = _mm_load_si128((const __m128i *)&v167.m128i_u64[1]);
        *(_QWORD *)(v18 + 160) = v18 + 176;
        *(__m128i *)(v18 + 144) = si128;
        sub_39CF630((__int64 *)(v18 + 160), v169, (__int64)v169 + v170);
        v20 = _mm_load_si128(&v173);
        *(_QWORD *)(v18 + 208) = v18 + 224;
        *(__m128i *)(v18 + 192) = v20;
        sub_39CF630((__int64 *)(v18 + 208), v174.m128i_i64[0], v174.m128i_i64[0] + v174.m128i_i64[1]);
        *(__m128i *)(v18 + 240) = _mm_load_si128(&v177);
        v18 = a2[39];
      }
      a2[39] = v18 + 256;
    }
    v134 = a1 + 40;
    v138 = v11 + 1;
    v143 = &v145;
    sub_39CF540((__int64 *)&v143, byte_3F871B3, (__int64)byte_3F871B3);
    v146 = v11;
    v147 = 1;
    LODWORD(v148) = v9;
    n.m128i_i64[0] = (__int64)&v150;
    if ( v143 == &v145 )
    {
      v150 = _mm_load_si128(&v145);
    }
    else
    {
      n.m128i_i64[0] = (__int64)v143;
      v150.m128i_i64[0] = v145.m128i_i64[0];
    }
    v21 = v144;
    v22 = *(_DWORD *)(a1 + 64);
    v143 = &v145;
    v144 = 0;
    n.m128i_i64[1] = v21;
    v145.m128i_i8[0] = 0;
    LODWORD(v151) = v11;
    BYTE4(v151) = 1;
    if ( v22 )
    {
      v12 = *(_QWORD *)(a1 + 48);
      v13 = (v22 - 1) & (37 * v9);
      v14 = (int *)(v12 + 48LL * v13);
      v15 = *v14;
      if ( *v14 == v9 )
      {
LABEL_6:
        if ( (__m128i *)n.m128i_i64[0] != &v150 )
          j_j___libc_free_0(n.m128i_u64[0]);
        goto LABEL_8;
      }
      v108 = 1;
      v29 = 0;
      while ( v15 != 0x7FFFFFFF )
      {
        if ( v15 != 0x80000000 || v29 )
          v14 = v29;
        v13 = (v22 - 1) & (v108 + v13);
        v15 = *(_DWORD *)(v12 + 48LL * v13);
        if ( v15 == v9 )
          goto LABEL_6;
        ++v108;
        v29 = v14;
        v14 = (int *)(v12 + 48LL * v13);
      }
      if ( !v29 )
        v29 = v14;
      v109 = *(_DWORD *)(a1 + 56);
      ++*(_QWORD *)(a1 + 40);
      v27 = v109 + 1;
      if ( 4 * (v109 + 1) < 3 * v22 )
      {
        v30 = v9;
        if ( v22 - *(_DWORD *)(a1 + 60) - v27 > v22 >> 3 )
          goto LABEL_155;
        sub_39CF780(v134, v22);
        v124 = *(_DWORD *)(a1 + 64);
        if ( !v124 )
        {
LABEL_213:
          ++*(_DWORD *)(a1 + 56);
          BUG();
        }
        v125 = v124 - 1;
        v126 = *(_QWORD *)(a1 + 48);
        v32 = 0;
        v24 = v148;
        v127 = 1;
        v27 = *(_DWORD *)(a1 + 56) + 1;
        v128 = v125 & (37 * v148);
        v29 = (int *)(v126 + 48LL * v128);
        v30 = *v29;
        if ( (_DWORD)v148 == *v29 )
          goto LABEL_155;
        while ( v30 != 0x7FFFFFFF )
        {
          if ( !v32 && v30 == 0x80000000 )
            v32 = v29;
          v128 = v125 & (v127 + v128);
          v29 = (int *)(v126 + 48LL * v128);
          v30 = *v29;
          if ( (_DWORD)v148 == *v29 )
            goto LABEL_155;
          ++v127;
        }
        goto LABEL_35;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 40);
    }
    sub_39CF780(v134, 2 * v22);
    v23 = *(_DWORD *)(a1 + 64);
    if ( !v23 )
      goto LABEL_213;
    v24 = v148;
    v25 = v23 - 1;
    v26 = *(_QWORD *)(a1 + 48);
    v27 = *(_DWORD *)(a1 + 56) + 1;
    v28 = v25 & (37 * v148);
    v29 = (int *)(v26 + 48LL * v28);
    v30 = *v29;
    if ( (_DWORD)v148 == *v29 )
      goto LABEL_155;
    v31 = 1;
    v32 = 0;
    while ( v30 != 0x7FFFFFFF )
    {
      if ( !v32 && v30 == 0x80000000 )
        v32 = v29;
      v28 = v25 & (v31 + v28);
      v29 = (int *)(v26 + 48LL * v28);
      v30 = *v29;
      if ( (_DWORD)v148 == *v29 )
        goto LABEL_155;
      ++v31;
    }
LABEL_35:
    v30 = v24;
    if ( v32 )
      v29 = v32;
LABEL_155:
    *(_DWORD *)(a1 + 56) = v27;
    if ( *v29 != 0x7FFFFFFF )
      --*(_DWORD *)(a1 + 60);
    *v29 = v30;
    *((_QWORD *)v29 + 1) = v29 + 6;
    if ( (__m128i *)n.m128i_i64[0] == &v150 )
    {
      *(__m128i *)(v29 + 6) = _mm_loadu_si128(&v150);
    }
    else
    {
      *((_QWORD *)v29 + 1) = n.m128i_i64[0];
      *((_QWORD *)v29 + 3) = v150.m128i_i64[0];
    }
    *((_QWORD *)v29 + 2) = n.m128i_i64[1];
    v29[10] = (int)v151;
    *((_BYTE *)v29 + 44) = BYTE4(v151);
LABEL_8:
    if ( v143 != &v145 )
      j_j___libc_free_0((unsigned __int64)v143);
    if ( (__m128i **)v174.m128i_i64[0] != &v175 )
      j_j___libc_free_0(v174.m128i_u64[0]);
    if ( v169 != &v171 )
      j_j___libc_free_0((unsigned __int64)v169);
    if ( v164 != &v166 )
      j_j___libc_free_0((unsigned __int64)v164);
    if ( (__int64 *)v159.m128i_i64[0] != &v160 )
      j_j___libc_free_0(v159.m128i_u64[0]);
    v11 = v138;
    v7 = *(_DWORD *)(v5 + 32);
    v10 = *(_QWORD *)(v5 + 8);
    goto LABEL_19;
  }
  v10 = *(_QWORD *)(v5 + 8);
LABEL_40:
  v132 = -858993459 * ((*(_QWORD *)(v5 + 16) - v10) >> 3) - v7;
  if ( v132 > 0 )
  {
    v33 = 0;
    v34 = 0;
    while ( 1 )
    {
      if ( *(_QWORD *)(v10 + 40LL * (unsigned int)(v33 + v8) + 8) == -1 )
        goto LABEL_61;
      v177.m128i_i8[0] = 0;
      dest = &v155.m128i_u64[1];
      v163 = (unsigned __int64)&v165;
      v171 = &v173;
      v175 = &v177;
      v152.m128i_i64[1] = 0;
      v153 = 0;
      v155.m128i_i64[0] = 0;
      v155.m128i_i8[8] = 0;
      v159 = 0u;
      LODWORD(v160) = 0;
      v161 = 0;
      v162.m128i_i64[0] = 0;
      v162.m128i_i32[2] = 0;
      v162.m128i_i8[12] = 0;
      v164 = 0;
      LOBYTE(v165) = 0;
      v167 = 0u;
      LOBYTE(v168) = 1;
      LOBYTE(v170) = 0;
      v172 = 0;
      v173.m128i_i8[0] = 0;
      v174 = 0u;
      v176 = 0;
      v178 = 0u;
      v179 = v181;
      v180 = 0;
      v181[0] = 0;
      v182[0] = 0u;
      v152.m128i_i32[0] = v34;
      v41 = *(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v33);
      v42 = *(_QWORD *)(v41 + 24);
      if ( v42 )
        break;
LABEL_73:
      v48 = 1;
      if ( !*(_BYTE *)(v41 + 21) )
        v48 = 2 * (*(_QWORD *)(v41 + 8) == 0);
      LODWORD(v160) = v48;
      v161 = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v33));
      v162.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v33) + 8);
      v162.m128i_i32[2] = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v33) + 16);
      v162.m128i_i8[12] = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v33) + 23);
      v49 = a2[42];
      if ( v49 == a2[43] )
      {
        sub_39D6B60(a2 + 41, (_QWORD *)a2[42], &v152);
      }
      else
      {
        if ( v49 )
        {
          *(__m128i *)v49 = _mm_load_si128(&v152);
          *(_QWORD *)(v49 + 16) = v153;
          *(_QWORD *)(v49 + 24) = v49 + 40;
          sub_39CF630((__int64 *)(v49 + 24), dest, (__int64)dest + v155.m128i_i64[0]);
          *(__m128i *)(v49 + 56) = _mm_loadu_si128(&v159);
          *(_DWORD *)(v49 + 72) = v160;
          *(_QWORD *)(v49 + 80) = v161;
          *(_QWORD *)(v49 + 88) = v162.m128i_i64[0];
          *(_DWORD *)(v49 + 96) = v162.m128i_i32[2];
          *(_BYTE *)(v49 + 100) = v162.m128i_i8[12];
          *(_QWORD *)(v49 + 104) = v49 + 120;
          sub_39CF630((__int64 *)(v49 + 104), (_BYTE *)v163, (__int64)v164 + v163);
          *(__m128i *)(v49 + 136) = _mm_loadu_si128(&v167);
          *(_BYTE *)(v49 + 152) = v168;
          v50 = v170;
          *(_BYTE *)(v49 + 168) = v170;
          if ( v50 )
            *(_QWORD *)(v49 + 160) = v169;
          *(_QWORD *)(v49 + 176) = v49 + 192;
          sub_39CF630((__int64 *)(v49 + 176), v171, (__int64)v171->m128i_i64 + v172);
          v51 = _mm_load_si128(&v174);
          *(_QWORD *)(v49 + 224) = v49 + 240;
          *(__m128i *)(v49 + 208) = v51;
          sub_39CF630((__int64 *)(v49 + 224), v175, (__int64)v175->m128i_i64 + v176);
          v52 = _mm_load_si128(&v178);
          *(_QWORD *)(v49 + 272) = v49 + 288;
          *(__m128i *)(v49 + 256) = v52;
          sub_39CF630((__int64 *)(v49 + 272), v179, (__int64)&v179[v180]);
          *(__m128i *)(v49 + 304) = _mm_load_si128(v182);
          v49 = a2[42];
        }
        a2[42] = v49 + 320;
      }
      v131 = a1 + 40;
      v139 = v34 + 1;
      if ( dest )
      {
        v143 = &v145;
        sub_39CF540((__int64 *)&v143, dest, (__int64)dest + v155.m128i_i64[0]);
        v146 = v34;
        v147 = 0;
        v35 = v144;
        LODWORD(v148) = v33;
        n.m128i_i64[0] = (__int64)&v150;
        if ( v143 != &v145 )
        {
          n.m128i_i64[0] = (__int64)v143;
          v150.m128i_i64[0] = v145.m128i_i64[0];
          goto LABEL_44;
        }
      }
      else
      {
        v145.m128i_i8[0] = 0;
        v35 = 0;
        v146 = v34;
        v147 = 0;
        LODWORD(v148) = v33;
        n.m128i_i64[0] = (__int64)&v150;
      }
      v150 = _mm_load_si128(&v145);
LABEL_44:
      v36 = *(_DWORD *)(a1 + 64);
      n.m128i_i64[1] = v35;
      v143 = &v145;
      v144 = 0;
      v145.m128i_i8[0] = 0;
      LODWORD(v151) = v34;
      BYTE4(v151) = 0;
      if ( !v36 )
      {
        ++*(_QWORD *)(a1 + 40);
        goto LABEL_164;
      }
      v37 = *(_QWORD *)(a1 + 48);
      v38 = (v36 - 1) & (37 * v33);
      v39 = (_DWORD *)(v37 + 48LL * v38);
      v40 = *v39;
      if ( *v39 != v33 )
      {
        v103 = 1;
        v104 = 0;
        while ( v40 != 0x7FFFFFFF )
        {
          if ( v40 != 0x80000000 || v104 )
            v39 = v104;
          v38 = (v36 - 1) & (v103 + v38);
          v40 = *(_DWORD *)(v37 + 48LL * v38);
          if ( v40 == v33 )
            goto LABEL_46;
          ++v103;
          v104 = v39;
          v39 = (_DWORD *)(v37 + 48LL * v38);
        }
        v105 = *(_DWORD *)(a1 + 56);
        if ( !v104 )
          v104 = v39;
        ++*(_QWORD *)(a1 + 40);
        v106 = v105 + 1;
        if ( 4 * (v105 + 1) < 3 * v36 )
        {
          v107 = v33;
          if ( v36 - *(_DWORD *)(a1 + 60) - v106 <= v36 >> 3 )
          {
            sub_39CF780(v131, v36);
            v118 = *(_DWORD *)(a1 + 64);
            if ( !v118 )
            {
LABEL_212:
              ++*(_DWORD *)(a1 + 56);
              BUG();
            }
            v107 = v148;
            v119 = v118 - 1;
            v120 = *(_QWORD *)(a1 + 48);
            v117 = 0;
            v106 = *(_DWORD *)(a1 + 56) + 1;
            v121 = 1;
            v122 = v119 & (37 * v148);
            v104 = (_DWORD *)(v120 + 48LL * v122);
            v123 = *v104;
            if ( *v104 != (_DWORD)v148 )
            {
              while ( v123 != 0x7FFFFFFF )
              {
                if ( v123 == 0x80000000 && !v117 )
                  v117 = v104;
                v122 = v119 & (v121 + v122);
                v104 = (_DWORD *)(v120 + 48LL * v122);
                v123 = *v104;
                if ( (_DWORD)v148 == *v104 )
                  goto LABEL_143;
                ++v121;
              }
              goto LABEL_168;
            }
          }
          goto LABEL_143;
        }
LABEL_164:
        sub_39CF780(v131, 2 * v36);
        v111 = *(_DWORD *)(a1 + 64);
        if ( !v111 )
          goto LABEL_212;
        v107 = v148;
        v112 = v111 - 1;
        v113 = *(_QWORD *)(a1 + 48);
        v106 = *(_DWORD *)(a1 + 56) + 1;
        v114 = (v111 - 1) & (37 * v148);
        v104 = (_DWORD *)(v113 + 48LL * v114);
        v115 = *v104;
        if ( *v104 != (_DWORD)v148 )
        {
          v116 = 1;
          v117 = 0;
          while ( v115 != 0x7FFFFFFF )
          {
            if ( v115 == 0x80000000 && !v117 )
              v117 = v104;
            v114 = v112 & (v116 + v114);
            v104 = (_DWORD *)(v113 + 48LL * v114);
            v115 = *v104;
            if ( (_DWORD)v148 == *v104 )
              goto LABEL_143;
            ++v116;
          }
LABEL_168:
          if ( v117 )
            v104 = v117;
        }
LABEL_143:
        *(_DWORD *)(a1 + 56) = v106;
        if ( *v104 != 0x7FFFFFFF )
          --*(_DWORD *)(a1 + 60);
        *v104 = v107;
        *((_QWORD *)v104 + 1) = v104 + 6;
        if ( (__m128i *)n.m128i_i64[0] == &v150 )
        {
          *(__m128i *)(v104 + 6) = _mm_loadu_si128(&v150);
        }
        else
        {
          *((_QWORD *)v104 + 1) = n.m128i_i64[0];
          *((_QWORD *)v104 + 3) = v150.m128i_i64[0];
        }
        *((_QWORD *)v104 + 2) = n.m128i_i64[1];
        v104[10] = (_DWORD)v151;
        *((_BYTE *)v104 + 44) = BYTE4(v151);
        goto LABEL_48;
      }
LABEL_46:
      if ( (__m128i *)n.m128i_i64[0] != &v150 )
        j_j___libc_free_0(n.m128i_u64[0]);
LABEL_48:
      if ( v143 != &v145 )
        j_j___libc_free_0((unsigned __int64)v143);
      if ( v179 != v181 )
        j_j___libc_free_0((unsigned __int64)v179);
      if ( v175 != &v177 )
        j_j___libc_free_0((unsigned __int64)v175);
      if ( v171 != &v173 )
        j_j___libc_free_0((unsigned __int64)v171);
      if ( (__int64 *)v163 != &v165 )
        j_j___libc_free_0(v163);
      if ( dest != &v155.m128i_u64[1] )
        j_j___libc_free_0((unsigned __int64)dest);
      v34 = v139;
LABEL_61:
      if ( ++v33 == v132 )
        goto LABEL_87;
      v8 = *(_DWORD *)(v5 + 32);
      v10 = *(_QWORD *)(v5 + 8);
    }
    v43 = "<unnamed alloca>";
    v44 = 16;
    if ( (*(_BYTE *)(v42 + 23) & 0x20) == 0 || (v43 = (char *)sub_1649960(v42)) != 0 )
    {
      v148 = (unsigned __int64)&n.m128i_u64[1];
      sub_39CF540((__int64 *)&v148, v43, (__int64)&v43[v44]);
      v45 = dest;
      v46 = (__int8 *)dest;
      if ( (unsigned __int64 *)v148 != &n.m128i_u64[1] )
      {
        if ( dest == &v155.m128i_u64[1] )
        {
          dest = (void *)v148;
          v155 = n;
        }
        else
        {
          v47 = v155.m128i_i64[1];
          dest = (void *)v148;
          v155 = n;
          if ( v46 )
          {
            v148 = (unsigned __int64)v46;
            n.m128i_i64[1] = v47;
LABEL_70:
            n.m128i_i64[0] = 0;
            *v46 = 0;
            if ( (unsigned __int64 *)v148 != &n.m128i_u64[1] )
              j_j___libc_free_0(v148);
            v41 = *(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v33);
            goto LABEL_73;
          }
        }
        v148 = (unsigned __int64)&n.m128i_u64[1];
        v46 = &n.m128i_i8[8];
        goto LABEL_70;
      }
      v53 = n.m128i_i64[0];
      if ( n.m128i_i64[0] )
      {
        if ( n.m128i_i64[0] == 1 )
          *(_BYTE *)dest = n.m128i_i8[8];
        else
          memcpy(dest, &n.m128i_u64[1], n.m128i_u64[0]);
        v53 = n.m128i_i64[0];
        v45 = dest;
      }
    }
    else
    {
      n.m128i_i8[8] = 0;
      v45 = dest;
      v53 = 0;
      v148 = (unsigned __int64)&n.m128i_u64[1];
    }
    v155.m128i_i64[0] = v53;
    v45[v53] = 0;
    v46 = (__int8 *)v148;
    goto LABEL_70;
  }
LABEL_87:
  v54 = *(_QWORD *)(v5 + 80);
  v140 = *(_QWORD *)(v5 + 88);
  if ( v140 != v54 )
  {
    while ( 1 )
    {
      LOBYTE(v153) = 0;
      v152 = (__m128i)(unsigned __int64)&v153;
      v155 = 0u;
      sub_39CF6E0(*(_DWORD *)v54, (__int64)&v152, v133);
      v55 = *(unsigned int *)(a1 + 64);
      v56 = *(_QWORD *)(a1 + 48);
      if ( !(_DWORD)v55 )
        goto LABEL_100;
      v57 = *(_DWORD *)(v54 + 4);
      v58 = (v55 - 1) & (37 * v57);
      v59 = (int *)(v56 + 48LL * v58);
      v60 = 1;
      v61 = *v59;
      if ( v57 != *v59 )
        break;
LABEL_95:
      v62 = (unsigned int)v59[10];
      if ( *((_BYTE *)v59 + 44) )
      {
        v136 = a2[38] + (v62 << 8);
        sub_2240AE0((unsigned __int64 *)(v136 + 56), (unsigned __int64 *)&v152);
        *(__m128i *)(v136 + 88) = _mm_load_si128(&v155);
        *(_BYTE *)(a2[38] + ((unsigned __int64)(unsigned int)v59[10] << 8) + 104) = *(_BYTE *)(v54 + 8);
      }
      else
      {
        v137 = a2[41] + 320 * v62;
        sub_2240AE0((unsigned __int64 *)(v137 + 104), (unsigned __int64 *)&v152);
        *(__m128i *)(v137 + 136) = _mm_load_si128(&v155);
        *(_BYTE *)(a2[41] + 320LL * (unsigned int)v59[10] + 152) = *(_BYTE *)(v54 + 8);
      }
      if ( (__int64 *)v152.m128i_i64[0] != &v153 )
        j_j___libc_free_0(v152.m128i_u64[0]);
      v54 += 12;
      if ( v140 == v54 )
        goto LABEL_101;
    }
    while ( v61 != 0x7FFFFFFF )
    {
      v58 = (v55 - 1) & (v60 + v58);
      v59 = (int *)(v56 + 48LL * v58);
      v61 = *v59;
      if ( v57 == *v59 )
        goto LABEL_95;
      ++v60;
    }
LABEL_100:
    v59 = (int *)(v56 + 48 * v55);
    goto LABEL_95;
  }
LABEL_101:
  v63 = *(_DWORD *)(v5 + 120);
  if ( v63 )
  {
    v64 = 0;
    while ( 1 )
    {
      v71 = *(unsigned int *)(a1 + 64);
      v72 = *(_QWORD *)(a1 + 48);
      v73 = *(_QWORD *)(v5 + 112) + 16LL * v64;
      v74 = *(_QWORD *)(v73 + 8);
      if ( !(_DWORD)v71 )
        goto LABEL_108;
      v65 = *(_DWORD *)v73;
      v66 = (v71 - 1) & (37 * *(_DWORD *)v73);
      v67 = (int *)(v72 + 48LL * v66);
      v68 = *v67;
      if ( v65 != *v67 )
        break;
LABEL_104:
      v69 = a2[41] + 320LL * (unsigned int)v67[10];
      v70 = *(_BYTE *)(v69 + 168) == 0;
      *(_QWORD *)(v69 + 160) = v74;
      if ( v70 )
        *(_BYTE *)(v69 + 168) = 1;
      if ( ++v64 == v63 )
        goto LABEL_109;
    }
    v101 = 1;
    while ( v68 != 0x7FFFFFFF )
    {
      v102 = v101 + 1;
      v66 = (v71 - 1) & (v101 + v66);
      v67 = (int *)(v72 + 48LL * v66);
      v68 = *v67;
      if ( v65 == *v67 )
        goto LABEL_104;
      v101 = v102;
    }
LABEL_108:
    v67 = (int *)(v72 + 48 * v71);
    goto LABEL_104;
  }
LABEL_109:
  if ( *(_DWORD *)(v5 + 68) != -1 )
  {
    v150.m128i_i32[2] = 1;
    v152.m128i_i64[0] = (__int64)&v148;
    v150.m128i_i64[0] = 0;
    v148 = (unsigned __int64)&unk_49EFBE0;
    v155.m128i_i64[0] = (__int64)&v156;
    v151 = a2 + 18;
    n = 0u;
    v152.m128i_i64[1] = a4;
    v153 = a1 + 8;
    dest = (void *)(a1 + 40);
    v155.m128i_i64[1] = 0x800000000LL;
    sub_39D3860(v152.m128i_i64, *(_DWORD *)(v5 + 68));
    if ( (int *)v155.m128i_i64[0] != &v156 )
      _libc_free(v155.m128i_u64[0]);
    sub_16E7BC0((__int64 *)&v148);
  }
  v75 = *(_QWORD *)(a3 + 608);
  v76 = 32LL * *(unsigned int *)(a3 + 616);
  result = v75 + v76;
  v141 = v75 + v76;
  if ( v75 != v75 + v76 )
  {
    v78 = a4;
    v135 = a1;
    v79 = *(_QWORD *)(a3 + 608);
    while ( 1 )
    {
      v80 = *(unsigned int *)(v135 + 64);
      v81 = *(_QWORD *)(v135 + 48);
      if ( !(_DWORD)v80 )
        goto LABEL_125;
      v82 = *(_DWORD *)(v79 + 16);
      v83 = (v80 - 1) & (37 * v82);
      v84 = (int *)(v81 + 48LL * v83);
      v85 = *v84;
      if ( *v84 != v82 )
        break;
LABEL_114:
      v86 = (unsigned int)v84[10];
      v87 = *(unsigned __int8 **)v79;
      v88 = *(_QWORD *)(v79 + 8);
      v89 = *(_QWORD *)(v79 + 24);
      if ( *((_BYTE *)v84 + 44) )
      {
        v148 = *(_QWORD *)v79;
        v90 = 0;
        n.m128i_i64[0] = v88;
        v91 = (__m128i *)(a2[38] + (v86 << 8));
        n.m128i_i64[1] = v89;
        m128i_i64 = (__int64)v91[7].m128i_i64;
        v144 = (__int64)v91[10].m128i_i64;
        v143 = v91 + 7;
        v145.m128i_i64[0] = (__int64)v91[13].m128i_i64;
        v93 = v78;
        v94 = v79;
        v95 = v93;
        while ( 1 )
        {
          v155.m128i_i64[1] = m128i_i64;
          v90 += 8;
          v152 = (__m128i)(unsigned __int64)&unk_49EFBE0;
          v155.m128i_i32[0] = 1;
          dest = 0;
          v153 = 0;
          sub_1556260(v87, (__int64)&v152, v95, 0);
          sub_16E7BC0(v152.m128i_i64);
          if ( v90 == 24 )
            break;
          m128i_i64 = *(__int64 *)((char *)&v143 + v90);
          v87 = *(unsigned __int8 **)((char *)&v148 + v90);
        }
      }
      else
      {
        v148 = *(_QWORD *)v79;
        v96 = 0;
        n.m128i_i64[0] = v88;
        v97 = a2[41] + 320 * v86;
        n.m128i_i64[1] = v89;
        v98 = v97 + 176;
        v145.m128i_i64[0] = v97 + 272;
        v143 = (__m128i *)(v97 + 176);
        v144 = v97 + 224;
        v99 = v78;
        v94 = v79;
        v95 = v99;
        while ( 1 )
        {
          v96 += 8;
          v155.m128i_i64[1] = v98;
          v152 = (__m128i)(unsigned __int64)&unk_49EFBE0;
          v155.m128i_i32[0] = 1;
          dest = 0;
          v153 = 0;
          sub_1556260(v87, (__int64)&v152, v95, 0);
          sub_16E7BC0(v152.m128i_i64);
          if ( v96 == 24 )
            break;
          v98 = *(__int64 *)((char *)&v143 + v96);
          v87 = *(unsigned __int8 **)((char *)&v148 + v96);
        }
      }
      result = v95;
      v79 = v94 + 32;
      v78 = result;
      if ( v141 == v79 )
        return result;
    }
    v100 = 1;
    while ( v85 != 0x7FFFFFFF )
    {
      v110 = v100 + 1;
      v83 = (v80 - 1) & (v100 + v83);
      v84 = (int *)(v81 + 48LL * v83);
      v85 = *v84;
      if ( v82 == *v84 )
        goto LABEL_114;
      v100 = v110;
    }
LABEL_125:
    v84 = (int *)(v81 + 48 * v80);
    goto LABEL_114;
  }
  return result;
}
