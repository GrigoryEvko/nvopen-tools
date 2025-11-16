// Function: sub_313BDD0
// Address: 0x313bdd0
//
void __fastcall sub_313BDD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __m128i *v10; // r13
  int v11; // eax
  __int64 *v12; // r13
  void (__fastcall *v13)(__int64 *, __int8 *, __int64); // rax
  __int64 v14; // rax
  void *v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // rcx
  __int8 *v18; // r12
  unsigned int v19; // eax
  char v20; // r13
  __int64 *v21; // rax
  __int64 *v22; // r12
  __int64 *i; // r13
  __int64 v24; // rsi
  _QWORD *v25; // r12
  unsigned __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 *v29; // r13
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  _QWORD *v33; // r14
  __int64 v34; // rax
  unsigned __int64 v35; // r15
  __int64 v36; // rbx
  char v37; // al
  char v38; // dl
  unsigned __int64 v39; // r13
  __int16 v40; // dx
  unsigned __int64 *v41; // r8
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // r13
  __m128i *v46; // r15
  unsigned __int64 v47; // r14
  unsigned __int64 v48; // r12
  __int64 v49; // r13
  int v50; // ebx
  unsigned __int64 v51; // r13
  __m128i *v52; // rax
  __m128i *v53; // r12
  __m128i *v54; // r13
  __m128i v55; // xmm0
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rdx
  char **v61; // rsi
  __int64 v62; // rdi
  unsigned __int64 v63; // rdi
  void (__fastcall *v64)(__m128i *, __m128i *, __int64); // rax
  __int64 *v65; // r13
  __int64 v66; // r12
  __int64 v67; // rdi
  __int64 v68; // rax
  char v69; // dh
  __int64 v70; // rcx
  __int64 v71; // r13
  char v72; // al
  __int64 v73; // r14
  __int64 v74; // r15
  __int64 v75; // rbx
  __int64 v76; // r12
  __int64 v77; // rdi
  unsigned __int64 v78; // rbx
  unsigned __int64 v79; // r12
  unsigned __int64 v80; // rdi
  void (__fastcall *v81)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v82; // rax
  __int64 v83; // rdx
  size_t v84; // rdx
  unsigned __int64 v85; // r12
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int32 v90; // r14d
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  __int32 v95; // r14d
  __m128i *v96; // rbx
  unsigned __int64 v97; // rdi
  void (__fastcall *v98)(__m128i *, __m128i *, __int64); // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __m128i *v101; // r12
  unsigned __int64 v102; // rdi
  void (__fastcall *v103)(__m128i *, __m128i *, __int64); // rax
  unsigned __int64 v104; // r12
  unsigned __int64 v105; // rbx
  unsigned __int64 v106; // rdi
  void (__fastcall *v107)(unsigned __int64, unsigned __int64, __int64); // rax
  char *v108; // r15
  _BYTE *v109; // rcx
  _BYTE *v110; // rbx
  char *v111; // rax
  __m128i v112; // xmm0
  __int64 v113; // rdx
  __int64 v114; // rax
  __m128i v115; // xmm0
  __int64 (__fastcall *v116)(__m128i *, __int64, int); // rsi
  __int64 v117; // rdx
  __int64 v118; // rax
  char **v119; // rsi
  __int64 v120; // rdi
  _BYTE *v121; // rax
  unsigned __int64 *v122; // rax
  __int64 v123; // r9
  unsigned __int64 *v124; // r12
  _BYTE *v125; // rax
  bool v126; // zf
  unsigned __int64 v127; // rax
  bool v128; // al
  __m128i *v129; // r14
  unsigned __int64 v130; // rdi
  void (__fastcall *v131)(__m128i *, __m128i *, __int64); // rax
  __int64 v132; // rdx
  __int64 v133; // rcx
  __int64 v134; // r8
  __int64 v135; // r9
  __int32 v136; // r12d
  unsigned __int64 v137; // rdi
  __m128i *v138; // r14
  _BYTE *v139; // r12
  __m128i v140; // xmm0
  __int64 v141; // rdx
  __int64 v142; // rax
  __m128i v143; // xmm0
  __int64 v144; // rcx
  __int64 v145; // rdx
  __int64 v146; // rax
  char **v147; // rsi
  __int64 v148; // rdi
  _QWORD *v149; // [rsp+18h] [rbp-A28h]
  unsigned __int64 v151; // [rsp+30h] [rbp-A10h]
  __int64 *v152; // [rsp+30h] [rbp-A10h]
  __int64 v154; // [rsp+40h] [rbp-A00h]
  __int64 *v155; // [rsp+40h] [rbp-A00h]
  int v156; // [rsp+40h] [rbp-A00h]
  __int64 v157; // [rsp+40h] [rbp-A00h]
  __int64 v158; // [rsp+48h] [rbp-9F8h]
  __int64 v159; // [rsp+48h] [rbp-9F8h]
  __int64 j; // [rsp+48h] [rbp-9F8h]
  char *v161; // [rsp+48h] [rbp-9F8h]
  __int64 v162; // [rsp+58h] [rbp-9E8h] BYREF
  __int64 v163[2]; // [rsp+60h] [rbp-9E0h] BYREF
  _QWORD v164[2]; // [rsp+70h] [rbp-9D0h] BYREF
  unsigned __int64 v165[2]; // [rsp+80h] [rbp-9C0h] BYREF
  _BYTE *v166; // [rsp+90h] [rbp-9B0h] BYREF
  __int64 v167; // [rsp+118h] [rbp-928h]
  unsigned int v168; // [rsp+128h] [rbp-918h]
  __int64 v169; // [rsp+138h] [rbp-908h]
  unsigned int v170; // [rsp+148h] [rbp-8F8h]
  __m128i v171; // [rsp+150h] [rbp-8F0h] BYREF
  __int64 (__fastcall *v172)(__m128i *, __int64, int); // [rsp+160h] [rbp-8E0h]
  __int64 (__fastcall *v173)(__int64, int *); // [rsp+168h] [rbp-8D8h]
  __int64 v174; // [rsp+190h] [rbp-8B0h]
  unsigned int v175; // [rsp+1A0h] [rbp-8A0h]
  unsigned __int64 *v176; // [rsp+1A8h] [rbp-898h]
  char *v177; // [rsp+1B8h] [rbp-888h] BYREF
  char v178; // [rsp+1C8h] [rbp-878h] BYREF
  _QWORD *v179; // [rsp+1F8h] [rbp-848h]
  _QWORD v180[6]; // [rsp+208h] [rbp-838h] BYREF
  unsigned int v181; // [rsp+238h] [rbp-808h]
  __int64 **v182; // [rsp+240h] [rbp-800h]
  __int64 *v183; // [rsp+250h] [rbp-7F0h] BYREF
  __int64 v184; // [rsp+258h] [rbp-7E8h]
  _BYTE v185[256]; // [rsp+260h] [rbp-7E0h] BYREF
  __int64 v186; // [rsp+360h] [rbp-6E0h] BYREF
  void *s; // [rsp+368h] [rbp-6D8h]
  _BYTE v188[12]; // [rsp+370h] [rbp-6D0h]
  char v189; // [rsp+37Ch] [rbp-6C4h]
  char v190; // [rsp+380h] [rbp-6C0h] BYREF
  __m128i *v191; // [rsp+480h] [rbp-5C0h] BYREF
  __int64 v192; // [rsp+488h] [rbp-5B8h]
  _BYTE v193[56]; // [rsp+490h] [rbp-5B0h] BYREF
  _BYTE v194[1400]; // [rsp+4C8h] [rbp-578h] BYREF

  s = &v190;
  v7 = *(_QWORD *)(a1 + 904);
  v183 = (__int64 *)v185;
  v184 = 0x2000000000LL;
  v191 = (__m128i *)v193;
  v192 = 0x1000000000LL;
  v8 = *(unsigned int *)(a1 + 912);
  v186 = 0;
  v189 = 1;
  *(_QWORD *)v188 = 32;
  *(_DWORD *)&v188[8] = 0;
  v151 = v7 + 88 * v8;
  if ( v7 == v151 )
  {
    v50 = 0;
LABEL_147:
    v104 = (unsigned __int64)v191;
    *(_DWORD *)(a1 + 912) = v50;
    v105 = v104 + 88LL * (unsigned int)v192;
    while ( v104 != v105 )
    {
      v105 -= 88LL;
      v106 = *(_QWORD *)(v105 + 56);
      if ( v106 != v105 + 72 )
        _libc_free(v106);
      v107 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v105 + 16);
      if ( v107 )
        v107(v105, v105, 3);
    }
    goto LABEL_77;
  }
  do
  {
    while ( 1 )
    {
      v17 = a2;
      v18 = (__int8 *)v7;
      if ( a2 && a2 != *(_QWORD *)(*(_QWORD *)(v7 + 32) + 72LL) )
      {
        v9 = (unsigned int)v192;
        v10 = v191;
        a6 = (unsigned int)v192 + 1LL;
        v11 = v192;
        if ( a6 > HIDWORD(v192) )
        {
          if ( (unsigned __int64)v191 > v7 || (unsigned __int64)v191 + 88 * (unsigned int)v192 <= v7 )
          {
            v10 = (__m128i *)sub_C8D7D0(
                               (__int64)&v191,
                               (__int64)v193,
                               (unsigned int)v192 + 1LL,
                               0x58u,
                               (unsigned __int64 *)&v171,
                               a6);
            sub_3139CC0((__int64)&v191, v10, v91, v92, v93, v94);
            v95 = v171.m128i_i32[0];
            if ( v191 != (__m128i *)v193 )
              _libc_free((unsigned __int64)v191);
            v9 = (unsigned int)v192;
            v191 = v10;
            HIDWORD(v192) = v95;
            v11 = v192;
          }
          else
          {
            v85 = v7 - (_QWORD)v191;
            v10 = (__m128i *)sub_C8D7D0(
                               (__int64)&v191,
                               (__int64)v193,
                               (unsigned int)v192 + 1LL,
                               0x58u,
                               (unsigned __int64 *)&v171,
                               a6);
            sub_3139CC0((__int64)&v191, v10, v86, v87, v88, v89);
            v90 = v171.m128i_i32[0];
            if ( v191 != (__m128i *)v193 )
              _libc_free((unsigned __int64)v191);
            v9 = (unsigned int)v192;
            v191 = v10;
            v18 = &v10->m128i_i8[v85];
            HIDWORD(v192) = v90;
            v11 = v192;
          }
        }
        v12 = &v10->m128i_i64[11 * v9];
        if ( v12 )
        {
          v12[2] = 0;
          v13 = (void (__fastcall *)(__int64 *, __int8 *, __int64))*((_QWORD *)v18 + 2);
          if ( v13 )
          {
            v13(v12, v18, 2);
            v12[3] = *((_QWORD *)v18 + 3);
            v12[2] = *((_QWORD *)v18 + 2);
          }
          v12[4] = *((_QWORD *)v18 + 4);
          v12[5] = *((_QWORD *)v18 + 5);
          v14 = *((_QWORD *)v18 + 6);
          v12[8] = 0x200000000LL;
          v12[6] = v14;
          v15 = v12 + 9;
          v12[7] = (__int64)(v12 + 9);
          v16 = *((_DWORD *)v18 + 16);
          if ( v16 && v12 + 7 != (__int64 *)(v18 + 56) )
          {
            v84 = 8LL * v16;
            if ( v16 <= 2
              || (sub_C8D5F0((__int64)(v12 + 7), v12 + 9, v16, 8u, v16, a6),
                  v15 = (void *)v12[7],
                  (v84 = 8LL * *((unsigned int *)v18 + 16)) != 0) )
            {
              memcpy(v15, *((const void **)v18 + 7), v84);
            }
            *((_DWORD *)v12 + 16) = v16;
          }
          v11 = v192;
        }
        LODWORD(v192) = v11 + 1;
        goto LABEL_10;
      }
      ++v186;
      if ( v189 )
        goto LABEL_18;
      v19 = 4 * (*(_DWORD *)&v188[4] - *(_DWORD *)&v188[8]);
      if ( v19 < 0x20 )
        v19 = 32;
      if ( *(_DWORD *)v188 <= v19 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v188);
LABEL_18:
        *(_QWORD *)&v188[4] = 0;
        goto LABEL_19;
      }
      sub_C8C990((__int64)&v186, (__int64)&v186);
LABEL_19:
      LODWORD(v184) = 0;
      sub_3136910(v7, (__int64)&v186, (__int64)&v183, v17, a5, a6);
      v158 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 72LL);
      sub_29B4290((__int64)v165, v158);
      v163[0] = (__int64)v164;
      v20 = *(_BYTE *)(a1 + 336);
      sub_3120C40(v163, ".omp_par", (__int64)"");
      sub_29AFB10(
        (__int64)&v171,
        v183,
        (unsigned int)v184,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        *(_QWORD *)(v7 + 48),
        (__int64)v163,
        v20);
      if ( (_QWORD *)v163[0] != v164 )
        j_j___libc_free_0(v163[0]);
      v21 = *(__int64 **)(v7 + 56);
      v22 = &v21[*(unsigned int *)(v7 + 64)];
      for ( i = v21; v22 != i; ++i )
      {
        v24 = *i;
        sub_29B2CB0((__int64)&v171, v24);
      }
      v25 = (_QWORD *)sub_29B77F0((__int64)&v171, (__int64)v165);
      v162 = sub_B2D7E0(v158, "target-cpu", 0xAu);
      if ( sub_A71840((__int64)&v162) )
        sub_B2CDC0((__int64)v25, v162);
      v163[0] = sub_B2D7E0(v158, "target-features", 0xFu);
      if ( sub_A71840((__int64)v163) )
        sub_B2CDC0((__int64)v25, v163[0]);
      v26 = (unsigned __int64)(v25 + 7);
      sub_B2C2B0(v25);
      v27 = *(_QWORD *)(a1 + 504);
      v28 = v27 + 24;
      if ( v27 + 24 == (*(_QWORD *)(v27 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v29 = *(__int64 **)(v27 + 32);
        sub_BA8540(v28, (__int64)v25);
        v82 = v25[7];
        v83 = *v29;
        v25[8] = v29;
        v83 &= 0xFFFFFFFFFFFFFFF8LL;
        v25[7] = v83 | v82 & 7;
        *(_QWORD *)(v83 + 8) = v26;
      }
      else
      {
        v29 = *(__int64 **)(v158 + 64);
        sub_BA8540(v28, (__int64)v25);
        v30 = v25[7];
        v31 = *v29;
        v25[8] = v29;
        v31 &= 0xFFFFFFFFFFFFFFF8LL;
        v25[7] = v31 | v30 & 7;
        *(_QWORD *)(v31 + 8) = v26;
      }
      *v29 = v26 | *v29 & 7;
      v32 = v25[10];
      v154 = v32;
      if ( !v32 )
        BUG();
      v149 = (_QWORD *)(v32 - 24);
      v159 = v32 + 24;
      v33 = (_QWORD *)(*(_QWORD *)(v32 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)(v32 + 24) != v33 )
      {
        v34 = v6;
        v35 = v7;
        v36 = v34;
        while ( 1 )
        {
          if ( !v33 )
            BUG();
          v39 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (unsigned int)*((unsigned __int8 *)v33 - 24) - 30 > 0xA )
          {
            v41 = (unsigned __int64 *)sub_AA5190(*(_QWORD *)(v35 + 32));
            if ( v41 )
            {
              v37 = v40;
              v38 = HIBYTE(v40);
            }
            else
            {
              v38 = 0;
              v37 = 0;
            }
            LOBYTE(v36) = v37;
            BYTE1(v36) = v38;
            sub_B44560(v33 - 3, *(_QWORD *)(v35 + 32), v41, v36);
          }
          if ( v159 == v39 )
            break;
          v33 = (_QWORD *)v39;
        }
        v42 = v36;
        v7 = v35;
        v6 = v42;
      }
      sub_AA4AC0(*(_QWORD *)(v7 + 32), v154);
      sub_AA5450(v149);
      if ( *(_QWORD *)(v7 + 16) )
        (*(void (__fastcall **)(unsigned __int64, _QWORD *))(v7 + 24))(v7, v25);
      if ( v182 != &v183 )
        _libc_free((unsigned __int64)v182);
      sub_C7D6A0(v180[4], 8LL * v181, 8);
      if ( v179 != v180 )
        j_j___libc_free_0((unsigned __int64)v179);
      if ( v177 != &v178 )
        _libc_free((unsigned __int64)v177);
      if ( v176 != (unsigned __int64 *)&v177 )
        _libc_free((unsigned __int64)v176);
      sub_C7D6A0(v174, 8LL * v175, 8);
      sub_C7D6A0(v169, 8LL * v170, 8);
      v43 = v168;
      if ( v168 )
      {
        v44 = v167;
        v45 = v167 + 40LL * v168;
        do
        {
          if ( *(_QWORD *)v44 != -4096 && *(_QWORD *)v44 != -8192 )
            sub_C7D6A0(*(_QWORD *)(v44 + 16), 8LL * *(unsigned int *)(v44 + 32), 8);
          v44 += 40;
        }
        while ( v45 != v44 );
        v43 = v168;
      }
      sub_C7D6A0(v167, 40 * v43, 8);
      if ( (_BYTE **)v165[0] != &v166 )
        break;
LABEL_10:
      v7 += 88LL;
      if ( v151 == v7 )
        goto LABEL_60;
    }
    _libc_free(v165[0]);
    v7 += 88LL;
  }
  while ( v151 != v7 );
LABEL_60:
  v46 = *(__m128i **)(a1 + 904);
  v47 = *(unsigned int *)(a1 + 912);
  if ( v191 != (__m128i *)v193 )
  {
    v96 = (__m128i *)((char *)v46 + 88 * v47);
    if ( v96 != v46 )
    {
      do
      {
        v96 = (__m128i *)((char *)v96 - 88);
        v97 = v96[3].m128i_u64[1];
        if ( (unsigned __int64 *)v97 != &v96[4].m128i_u64[1] )
          _libc_free(v97);
        v98 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v96[1].m128i_i64[0];
        if ( v98 )
          v98(v96, v96, 3);
      }
      while ( v96 != v46 );
      v46 = *(__m128i **)(a1 + 904);
    }
    if ( v46 != (__m128i *)(a1 + 920) )
      _libc_free((unsigned __int64)v46);
    *(_QWORD *)(a1 + 904) = v191;
    v99 = v192;
    v192 = 0;
    *(_QWORD *)(a1 + 912) = v99;
    v191 = (__m128i *)v193;
    goto LABEL_78;
  }
  v48 = (unsigned int)v192;
  v49 = a1 + 904;
  v50 = v192;
  if ( (unsigned int)v192 <= v47 )
  {
    if ( (_DWORD)v192 )
    {
      v138 = (__m128i *)((char *)v46 + 56);
      v139 = v194;
      v157 = 11LL * (unsigned int)v192;
      do
      {
        v140 = _mm_loadu_si128((const __m128i *)(v139 - 56));
        *(__m128i *)(v139 - 56) = _mm_loadu_si128(&v171);
        v171 = v140;
        v141 = *((_QWORD *)v139 - 5);
        v142 = *((_QWORD *)v139 - 4);
        *((_QWORD *)v139 - 5) = 0;
        *((_QWORD *)v139 - 4) = v173;
        v143 = _mm_loadu_si128(&v171);
        v171 = _mm_loadu_si128((__m128i *)((char *)v138 - 56));
        v144 = v138[-3].m128i_i64[1];
        *(__m128i *)((char *)v138 - 56) = v143;
        v172 = (__int64 (__fastcall *)(__m128i *, __int64, int))v144;
        v138[-3].m128i_i64[1] = v141;
        v145 = v138[-2].m128i_i64[0];
        v173 = (__int64 (__fastcall *)(__int64, int *))v145;
        v138[-2].m128i_i64[0] = v142;
        if ( v172 )
          v172(&v171, (__int64)&v171, 3);
        v146 = *((_QWORD *)v139 - 3);
        v147 = (char **)v139;
        v148 = (__int64)v138;
        v139 += 88;
        v138 = (__m128i *)((char *)v138 + 88);
        v138[-7].m128i_i64[0] = v146;
        v138[-7].m128i_i64[1] = *((_QWORD *)v139 - 13);
        v138[-6].m128i_i64[0] = *((_QWORD *)v139 - 12);
        sub_3120CF0(v148, v147, v145, v144, a5, a6);
      }
      while ( &v46[3].m128i_u64[v157 + 1] != (unsigned __int64 *)v138 );
      v46 = (__m128i *)((char *)v46 + v157 * 8);
      v100 = *(_QWORD *)(a1 + 904);
      v47 = *(unsigned int *)(a1 + 912);
    }
    else
    {
      v100 = *(_QWORD *)(a1 + 904);
      v50 = 0;
    }
    v101 = (__m128i *)(v100 + 88 * v47);
    while ( v46 != v101 )
    {
      v101 = (__m128i *)((char *)v101 - 88);
      v102 = v101[3].m128i_u64[1];
      if ( (unsigned __int64 *)v102 != &v101[4].m128i_u64[1] )
        _libc_free(v102);
      v103 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v101[1].m128i_i64[0];
      if ( v103 )
        v103(v101, v101, 3);
    }
    goto LABEL_147;
  }
  if ( *(_DWORD *)(a1 + 916) < (unsigned int)v192 )
  {
    v129 = (__m128i *)((char *)v46 + 88 * v47);
    while ( v129 != v46 )
    {
      while ( 1 )
      {
        v129 = (__m128i *)((char *)v129 - 88);
        v130 = v129[3].m128i_u64[1];
        if ( (unsigned __int64 *)v130 != &v129[4].m128i_u64[1] )
          _libc_free(v130);
        v131 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v129[1].m128i_i64[0];
        if ( !v131 )
          break;
        v131(v129, v129, 3);
        if ( v129 == v46 )
          goto LABEL_178;
      }
    }
LABEL_178:
    *(_DWORD *)(a1 + 912) = 0;
    v46 = (__m128i *)sub_C8D7D0(v49, a1 + 920, v48, 0x58u, (unsigned __int64 *)&v171, a6);
    sub_3139CC0(v49, v46, v132, v133, v134, v135);
    v136 = v171.m128i_i32[0];
    v137 = *(_QWORD *)(a1 + 904);
    if ( a1 + 920 != v137 )
      _libc_free(v137);
    v47 = 0;
    *(_DWORD *)(a1 + 916) = v136;
    v48 = (unsigned int)v192;
    *(_QWORD *)(a1 + 904) = v46;
  }
  else if ( *(_DWORD *)(a1 + 912) )
  {
    v108 = &v46[3].m128i_i8[8];
    v156 = v192;
    v109 = v194;
    v110 = v194;
    v111 = &v108[88 * v47];
    v47 *= 88LL;
    v161 = v111;
    do
    {
      v112 = _mm_loadu_si128((const __m128i *)(v110 - 56));
      *(__m128i *)(v110 - 56) = _mm_loadu_si128(&v171);
      v171 = v112;
      v113 = *((_QWORD *)v110 - 5);
      v114 = *((_QWORD *)v110 - 4);
      *((_QWORD *)v110 - 5) = 0;
      *((_QWORD *)v110 - 4) = v173;
      v115 = _mm_loadu_si128(&v171);
      v171 = _mm_loadu_si128((const __m128i *)(v108 - 56));
      v116 = (__int64 (__fastcall *)(__m128i *, __int64, int))*((_QWORD *)v108 - 5);
      *(__m128i *)(v108 - 56) = v115;
      v172 = v116;
      *((_QWORD *)v108 - 5) = v113;
      v117 = *((_QWORD *)v108 - 4);
      v173 = (__int64 (__fastcall *)(__int64, int *))v117;
      *((_QWORD *)v108 - 4) = v114;
      if ( v172 )
        v172(&v171, (__int64)&v171, 3);
      v118 = *((_QWORD *)v110 - 3);
      v119 = (char **)v110;
      v120 = (__int64)v108;
      v110 += 88;
      v108 += 88;
      *((_QWORD *)v108 - 14) = v118;
      *((_QWORD *)v108 - 13) = *((_QWORD *)v110 - 13);
      *((_QWORD *)v108 - 12) = *((_QWORD *)v110 - 12);
      sub_3120CF0(v120, v119, v117, (__int64)v109, a5, a6);
    }
    while ( v161 != v108 );
    v50 = v156;
    v48 = (unsigned int)v192;
    v46 = (__m128i *)(v47 + *(_QWORD *)(a1 + 904));
  }
  v51 = (unsigned __int64)v191;
  v52 = (__m128i *)((char *)v191 + 88 * v48);
  v53 = (__m128i *)((char *)v191 + v47);
  if ( v52 != (__m128i *)&v191->m128i_i8[v47] )
  {
    v54 = v52;
    do
    {
      while ( 1 )
      {
        if ( v46 )
        {
          v46[1].m128i_i64[0] = 0;
          v55 = _mm_loadu_si128(v53);
          *v53 = _mm_loadu_si128(v46);
          *v46 = v55;
          v56 = v53[1].m128i_i64[0];
          v53[1].m128i_i64[0] = 0;
          v57 = v46[1].m128i_i64[1];
          v46[1].m128i_i64[0] = v56;
          v58 = v53[1].m128i_i64[1];
          v53[1].m128i_i64[1] = v57;
          v46[1].m128i_i64[1] = v58;
          v46[2].m128i_i64[0] = v53[2].m128i_i64[0];
          v46[2].m128i_i64[1] = v53[2].m128i_i64[1];
          v59 = v53[3].m128i_i64[0];
          v46[4].m128i_i32[0] = 0;
          v46[3].m128i_i64[0] = v59;
          v60 = (__int64)&v46[4].m128i_i64[1];
          v46[3].m128i_i64[1] = (__int64)&v46[4].m128i_i64[1];
          v46[4].m128i_i32[1] = 2;
          if ( v53[4].m128i_i32[0] )
            break;
        }
        v53 = (__m128i *)((char *)v53 + 88);
        v46 = (__m128i *)((char *)v46 + 88);
        if ( v54 == v53 )
          goto LABEL_70;
      }
      v61 = (char **)&v53[3].m128i_i64[1];
      v62 = (__int64)&v46[3].m128i_i64[1];
      v53 = (__m128i *)((char *)v53 + 88);
      v46 = (__m128i *)((char *)v46 + 88);
      sub_3120CF0(v62, v61, v60, v57, a5, a6);
    }
    while ( v54 != v53 );
LABEL_70:
    v51 = (unsigned __int64)v191;
    v53 = (__m128i *)((char *)v191 + 88 * (unsigned int)v192);
  }
  *(_DWORD *)(a1 + 912) = v50;
  while ( (__m128i *)v51 != v53 )
  {
    v53 = (__m128i *)((char *)v53 - 88);
    v63 = v53[3].m128i_u64[1];
    if ( (unsigned __int64 *)v63 != &v53[4].m128i_u64[1] )
      _libc_free(v63);
    v64 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v53[1].m128i_i64[0];
    if ( v64 )
      v64(v53, v53, 3);
  }
LABEL_77:
  LODWORD(v192) = 0;
LABEL_78:
  v65 = *(__int64 **)(a1 + 2328);
  v155 = v65;
  v152 = &v65[*(unsigned int *)(a1 + 2336)];
  if ( v152 != v65 )
  {
    do
    {
      v66 = *v155;
      v67 = *(_QWORD *)(*v155 + 80);
      if ( v67 )
        v67 -= 24;
      v68 = sub_AA4FF0(v67);
      v70 = 1;
      v71 = v68;
      v72 = 0;
      if ( v71 )
        v72 = v69;
      BYTE1(v70) = v72;
      v73 = v70;
      v74 = *(_QWORD *)(*(_QWORD *)(v66 + 80) + 8LL);
      for ( j = v66 + 72; j != v74; v74 = *(_QWORD *)(v74 + 8) )
      {
        if ( !v74 )
          BUG();
        v75 = *(_QWORD *)(v74 + 32);
        v76 = v74 + 24;
        while ( v76 != v75 )
        {
          while ( 1 )
          {
            v77 = v75;
            v75 = *(_QWORD *)(v75 + 8);
            if ( *(_BYTE *)(v77 - 24) == 60 && (unsigned int)**(unsigned __int8 **)(v77 - 56) - 12 <= 9 )
              break;
            if ( v76 == v75 )
              goto LABEL_91;
          }
          sub_B44500((_QWORD *)(v77 - 24), v71, v73);
        }
LABEL_91:
        ;
      }
      ++v155;
    }
    while ( v152 != v155 );
  }
  v173 = sub_3121750;
  v172 = (__int64 (__fastcall *)(__m128i *, __int64, int))sub_3120B70;
  if ( !sub_3136D60(a1 + 712) )
    sub_313B3D0(a1, (__int64)&v171);
  if ( *(_BYTE *)(a1 + 341) && *(_BYTE *)(a1 + 340) )
  {
    v121 = sub_BA8CD0(*(_QWORD *)(a1 + 504), (__int64)"__openmp_nvptx_data_transfer_temporary_storage", 0x2Eu, 0);
    v165[0] = 6;
    v166 = v121;
    v165[1] = 0;
    if ( v121 != 0 && v121 + 4096 != 0 && v121 != (_BYTE *)-8192LL )
      sub_BD73F0((__int64)v165);
    v122 = (unsigned __int64 *)sub_22077B0(0x18u);
    v124 = v122;
    if ( v122 )
    {
      *v122 = 6;
      v122[1] = 0;
      v125 = v166;
      v126 = v166 + 4096 == 0;
      v124[2] = (unsigned __int64)v166;
      if ( v125 == 0 || v126 || v125 == (_BYTE *)-8192LL )
      {
LABEL_165:
        sub_3135940(a1, (__int64)"llvm.compiler.used", 18, (__int64)v124, 1u, v123);
        v127 = v124[2];
        if ( v127 != -4096 && v127 != 0 && v127 != -8192 )
          sub_BD60C0(v124);
        j_j___libc_free_0((unsigned __int64)v124);
        goto LABEL_97;
      }
      sub_BD6050(v124, v165[0] & 0xFFFFFFFFFFFFFFF8LL);
      v128 = v166 + 0x2000 != 0 && v166 != 0 && v166 + 4096 != 0;
    }
    else
    {
      v128 = v166 != 0 && v166 + 0x2000 != 0 && v166 + 4096 != 0;
    }
    if ( v128 )
      sub_BD60C0(v165);
    goto LABEL_165;
  }
LABEL_97:
  if ( v172 )
    v172(&v171, (__int64)&v171, 3);
  v78 = (unsigned __int64)v191;
  v79 = (unsigned __int64)v191 + 88 * (unsigned int)v192;
  if ( v191 != (__m128i *)v79 )
  {
    do
    {
      v79 -= 88LL;
      v80 = *(_QWORD *)(v79 + 56);
      if ( v80 != v79 + 72 )
        _libc_free(v80);
      v81 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v79 + 16);
      if ( v81 )
        v81(v79, v79, 3);
    }
    while ( v78 != v79 );
    v79 = (unsigned __int64)v191;
  }
  if ( (_BYTE *)v79 != v193 )
    _libc_free(v79);
  if ( v183 != (__int64 *)v185 )
    _libc_free((unsigned __int64)v183);
  if ( !v189 )
    _libc_free((unsigned __int64)s);
}
