// Function: sub_1ED60E0
// Address: 0x1ed60e0
//
__int64 __fastcall sub_1ED60E0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r13
  __int64 v4; // rax
  unsigned int v5; // r12d
  unsigned int v6; // r14d
  _DWORD *v7; // rsi
  _QWORD *v8; // rax
  unsigned int v9; // r15d
  _QWORD *v10; // r14
  unsigned __int64 v11; // rcx
  int v12; // r11d
  unsigned int v13; // esi
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  char *v17; // rsi
  char *v18; // rsi
  int v19; // ecx
  volatile signed __int32 *v20; // rdx
  _DWORD *v21; // rsi
  const __m128i *v22; // rdi
  __int64 v23; // r12
  char *v24; // rsi
  char *v25; // rsi
  volatile signed __int32 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int32 v31; // eax
  unsigned int v32; // r9d
  __int64 v33; // rsi
  __int64 v34; // rdi
  char *v35; // rax
  __int64 v36; // rdx
  volatile signed __int32 *v37; // rcx
  volatile signed __int32 *v38; // rax
  int v39; // eax
  unsigned int v40; // eax
  char v41; // r12
  __int64 v42; // r13
  __int64 (*v43)(void); // rax
  __int64 v44; // rax
  __int64 v45; // rbx
  int v46; // eax
  unsigned __int64 v47; // r14
  volatile signed __int32 *v48; // r11
  unsigned int v49; // r14d
  __int64 v50; // rdx
  int v51; // r9d
  unsigned int v52; // ecx
  __int64 v53; // rdx
  int v54; // edi
  __int64 v55; // rdx
  __int64 v56; // r8
  unsigned int v57; // esi
  _WORD *v58; // r10
  unsigned __int16 v59; // ax
  __int16 *v60; // rsi
  _WORD *v61; // r10
  int v62; // edx
  unsigned __int16 *v63; // r8
  unsigned int v64; // r10d
  unsigned int j; // edi
  bool v66; // cf
  __int16 *v67; // r10
  __int16 v68; // si
  __int64 *v69; // rax
  unsigned int v70; // ebx
  __int64 *v71; // r12
  char v72; // al
  __int64 *v73; // rbx
  volatile signed __int32 *v74; // rdi
  volatile signed __int32 *v75; // r12
  char v76; // al
  __int64 *v77; // rdx
  int *v78; // r12
  char v79; // dl
  int *v80; // rax
  unsigned int v81; // r13d
  __m128i *v82; // rax
  _QWORD *v83; // rbx
  _QWORD *v84; // r12
  volatile signed __int32 *v85; // rdi
  int v87; // edi
  __int64 v88; // r12
  __int64 *v89; // rax
  volatile signed __int32 *v90; // rdx
  unsigned int v91; // r13d
  __int64 v92; // rdi
  __int64 i; // rbx
  __int64 v94; // rdi
  __int64 v95; // rax
  int v96; // esi
  int v97; // eax
  __int64 v98; // rdx
  volatile signed __int32 *v99; // rax
  __int64 *v100; // rdx
  char v101; // al
  __int64 *v102; // rdx
  int v103; // esi
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 (__fastcall *v107)(__int64, __int64); // r8
  volatile signed __int32 *v108; // rcx
  int v109; // esi
  __int64 v110; // rdx
  int v111; // esi
  int v112; // eax
  __int64 v113; // rax
  _QWORD *v114; // rdi
  _QWORD *v115; // rdx
  __int64 v116; // rcx
  unsigned int v117; // [rsp+3Ch] [rbp-1B4h]
  __int64 v118; // [rsp+40h] [rbp-1B0h]
  __int64 v119; // [rsp+50h] [rbp-1A0h]
  unsigned int v120; // [rsp+58h] [rbp-198h]
  __int64 (__fastcall *v121)(__int64, __int64); // [rsp+60h] [rbp-190h]
  int v122; // [rsp+68h] [rbp-188h]
  __int64 v123; // [rsp+70h] [rbp-180h]
  int v124; // [rsp+70h] [rbp-180h]
  __int64 v125; // [rsp+78h] [rbp-178h]
  int *v126; // [rsp+78h] [rbp-178h]
  __int64 v127; // [rsp+78h] [rbp-178h]
  int v128; // [rsp+88h] [rbp-168h] BYREF
  unsigned int v129; // [rsp+8Ch] [rbp-164h]
  __int64 (__fastcall *v130)(__int64, __int64); // [rsp+90h] [rbp-160h] BYREF
  __int64 v131; // [rsp+98h] [rbp-158h]
  __m128i v132; // [rsp+A0h] [rbp-150h] BYREF
  _QWORD *v133; // [rsp+B0h] [rbp-140h]
  __int64 *v134; // [rsp+C0h] [rbp-130h] BYREF
  void *s; // [rsp+C8h] [rbp-128h] BYREF
  __int64 v136; // [rsp+D0h] [rbp-120h]
  __int64 v137; // [rsp+E0h] [rbp-110h] BYREF
  _QWORD *v138; // [rsp+E8h] [rbp-108h]
  __int64 v139; // [rsp+F0h] [rbp-100h]
  unsigned int v140; // [rsp+F8h] [rbp-F8h]
  __int64 v141; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v142; // [rsp+108h] [rbp-E8h]
  __int64 v143; // [rsp+110h] [rbp-E0h]
  __int64 v144; // [rsp+118h] [rbp-D8h]
  __int64 v145; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v146; // [rsp+128h] [rbp-C8h]
  __int64 v147; // [rsp+130h] [rbp-C0h]
  __int64 v148; // [rsp+138h] [rbp-B8h]
  const __m128i *v149; // [rsp+140h] [rbp-B0h] BYREF
  __m128i *v150; // [rsp+148h] [rbp-A8h]
  char *v151; // [rsp+150h] [rbp-A0h]
  __int64 (__fastcall *v152)(__int64, __int64); // [rsp+158h] [rbp-98h]
  __int64 (__fastcall *v153)(__int64, __int64); // [rsp+160h] [rbp-90h] BYREF
  int v154; // [rsp+168h] [rbp-88h] BYREF
  int *v155; // [rsp+170h] [rbp-80h]
  int *v156; // [rsp+178h] [rbp-78h]
  int *v157; // [rsp+180h] [rbp-70h]
  __int64 v158; // [rsp+188h] [rbp-68h]
  __int64 *v159; // [rsp+190h] [rbp-60h] BYREF
  volatile signed __int32 *v160; // [rsp+198h] [rbp-58h]
  __int64 v161; // [rsp+1A0h] [rbp-50h]
  unsigned int v162; // [rsp+1A8h] [rbp-48h]
  __int64 v163; // [rsp+1B0h] [rbp-40h]
  __int64 v164; // [rsp+1B8h] [rbp-38h]

  v2 = a2;
  v153 = sub_1ECADE0;
  v3 = a2[1];
  v156 = &v154;
  v157 = &v154;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v154 = 0;
  v155 = 0;
  v158 = 0;
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v152 = (__int64 (__fastcall *)(__int64, __int64))sub_1ECB150;
  v130 = (__int64 (__fastcall *)(__int64, __int64))a2;
  v4 = a2[21] - a2[20];
  LODWORD(v159) = 0;
  v5 = -1171354717 * (v4 >> 3);
  if ( v5 )
  {
    while ( 1 )
    {
      v7 = (_DWORD *)v2[24];
      if ( v7 == sub_1ECAFE0((_DWORD *)v2[23], (__int64)v7, (int *)&v159) )
        break;
      v6 = (_DWORD)v159 + 1;
      LODWORD(v159) = v6;
      if ( v5 <= v6 )
        goto LABEL_6;
    }
    v6 = (unsigned int)v159;
  }
  else
  {
    v6 = 0;
  }
LABEL_6:
  v122 = sub_1ECCC00((__int64)&v130);
  if ( v6 == v122 )
    goto LABEL_21;
  v8 = v2;
  v9 = v6;
  v10 = v8;
  do
  {
    v11 = *(unsigned int *)(v3 + 408);
    v12 = *(_DWORD *)(v10[20] + 88LL * v9 + 40);
    v13 = v12 & 0x7FFFFFFF;
    v14 = v12 & 0x7FFFFFFF;
    v15 = 8 * v14;
    if ( (v12 & 0x7FFFFFFFu) >= (unsigned int)v11 || (v16 = *(_QWORD *)(*(_QWORD *)(v3 + 400) + 8LL * v13)) == 0 )
    {
      v32 = v13 + 1;
      if ( (unsigned int)v11 < v13 + 1 )
      {
        v110 = v32;
        if ( v32 >= v11 )
        {
          if ( v32 > v11 )
          {
            if ( v32 > (unsigned __int64)*(unsigned int *)(v3 + 412) )
            {
              v124 = *(_DWORD *)(v10[20] + 88LL * v9 + 40);
              v127 = v32;
              sub_16CD150(v3 + 400, (const void *)(v3 + 416), v32, 8, v13, v32);
              v11 = *(unsigned int *)(v3 + 408);
              v14 = v13;
              v32 = v13 + 1;
              v15 = 8LL * v13;
              v12 = v124;
              v110 = v127;
            }
            v33 = *(_QWORD *)(v3 + 400);
            v114 = (_QWORD *)(v33 + 8 * v110);
            v115 = (_QWORD *)(v33 + 8 * v11);
            v116 = *(_QWORD *)(v3 + 416);
            if ( v114 != v115 )
            {
              do
                *v115++ = v116;
              while ( v114 != v115 );
              v33 = *(_QWORD *)(v3 + 400);
            }
            *(_DWORD *)(v3 + 408) = v32;
            goto LABEL_35;
          }
        }
        else
        {
          *(_DWORD *)(v3 + 408) = v32;
        }
      }
      v33 = *(_QWORD *)(v3 + 400);
LABEL_35:
      v123 = v14;
      *(_QWORD *)(v33 + v15) = sub_1DBA290(v12);
      v125 = *(_QWORD *)(*(_QWORD *)(v3 + 400) + 8 * v123);
      sub_1DBB110((_QWORD *)v3, v125);
      v16 = v125;
    }
    LODWORD(v134) = v9;
    v17 = (char *)v150;
    s = 0;
    v136 = v16;
    if ( v150 == (__m128i *)v151 )
    {
      sub_1ECEA80((__int64 *)&v149, v150->m128i_i8, (__int64)&v134);
      v18 = (char *)v150;
    }
    else
    {
      if ( v150 )
      {
        v150->m128i_i32[0] = v9;
        *((_QWORD *)v17 + 1) = s;
        *((_QWORD *)v17 + 2) = v136;
        v17 = (char *)v150;
      }
      v18 = v17 + 24;
      v150 = (__m128i *)v18;
    }
    ++v9;
    v132.m128i_i64[0] = (__int64)v152;
    v19 = *((_DWORD *)v18 - 6);
    v20 = (volatile signed __int32 *)*((_QWORD *)v18 - 2);
    v161 = *((_QWORD *)v18 - 1);
    LODWORD(v159) = v19;
    v160 = v20;
    sub_1ECE190(
      (__int64)v149,
      0xAAAAAAAAAAAAAAABLL * ((v18 - (char *)v149) >> 3) - 1,
      0,
      (__int64)&v159,
      (__int64 (__fastcall **)(__int64, __int64))&v132);
    LODWORD(v159) = v9;
    if ( v5 > v9 )
    {
      while ( 1 )
      {
        v21 = (_DWORD *)v10[24];
        if ( v21 == sub_1ECAFE0((_DWORD *)v10[23], (__int64)v21, (int *)&v159) )
          break;
        v9 = (_DWORD)v159 + 1;
        LODWORD(v159) = v9;
        if ( v5 <= v9 )
          goto LABEL_19;
      }
      v9 = (unsigned int)v159;
    }
LABEL_19:
    ;
  }
  while ( v122 != v9 );
  v2 = v10;
LABEL_21:
  v22 = v149;
  if ( v149 != v150 )
  {
    while ( 2 )
    {
      v23 = (__int64)v156;
      v132 = _mm_loadu_si128(v22);
      v133 = (_QWORD *)v22[1].m128i_i64[0];
      if ( v156 == &v154 )
      {
LABEL_36:
        sub_1ECDC90((__int64)&v153, (__int64)v155);
        v155 = 0;
        v158 = 0;
        v156 = &v154;
        v157 = &v154;
      }
      else
      {
        while ( 1 )
        {
          v28 = *(_QWORD *)(v23 + 48);
          v29 = *(_QWORD *)(v23 + 40);
          v30 = *(_QWORD *)(*(_QWORD *)v28 + 24 * v29 + 8);
          if ( (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v30 >> 1) & 3) > (*(_DWORD *)((*(_QWORD *)(*v133 + 24 * v132.m128i_i64[1]) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                | (unsigned int)(*(__int64 *)(*v133 + 24 * v132.m128i_i64[1]) >> 1)
                                                                                                & 3) )
            break;
          if ( v29 != *(unsigned int *)(v28 + 8) - 1LL )
          {
            v31 = *(_DWORD *)(v23 + 32);
            v136 = *(_QWORD *)(v23 + 48);
            s = (void *)(v29 + 1);
            v24 = (char *)v150;
            LODWORD(v134) = v31;
            if ( v150 == (__m128i *)v151 )
            {
              sub_1ECEA80((__int64 *)&v149, v150->m128i_i8, (__int64)&v134);
              v25 = (char *)v150;
            }
            else
            {
              if ( v150 )
              {
                v150->m128i_i32[0] = v31;
                *((_QWORD *)v24 + 1) = s;
                *((_QWORD *)v24 + 2) = v136;
                v24 = (char *)v150;
              }
              v25 = v24 + 24;
              v150 = (__m128i *)v25;
            }
            v130 = v152;
            v26 = (volatile signed __int32 *)*((_QWORD *)v25 - 2);
            v27 = *((_QWORD *)v25 - 1);
            LODWORD(v159) = *((_DWORD *)v25 - 6);
            v160 = v26;
            v161 = v27;
            sub_1ECE190((__int64)v149, 0xAAAAAAAAAAAAAAABLL * ((v25 - (char *)v149) >> 3) - 1, 0, (__int64)&v159, &v130);
          }
          v23 = sub_220EF30(v23);
          if ( (int *)v23 == &v154 )
            goto LABEL_36;
        }
        for ( i = (__int64)v156; i != v23; --v158 )
        {
          v94 = i;
          i = sub_220EF30(i);
          v95 = sub_220F330(v94, &v154);
          j_j___libc_free_0(v95, 56);
        }
      }
      v34 = (__int64)v149;
      v133 = (_QWORD *)v149[1].m128i_i64[0];
      v132.m128i_i64[1] = v149->m128i_i64[1];
      v132.m128i_i32[0] = v149->m128i_i32[0];
      v35 = (char *)v150;
      if ( (char *)v150 - (char *)v149 > 24 )
      {
        v106 = v150[-1].m128i_i64[1];
        v107 = v152;
        v108 = (volatile signed __int32 *)v150[-1].m128i_i64[0];
        v150[-1].m128i_i64[1] = v149[1].m128i_i64[0];
        v109 = *((_DWORD *)v35 - 6);
        *((_QWORD *)v35 - 2) = *(_QWORD *)(v34 + 8);
        *((_DWORD *)v35 - 6) = *(_DWORD *)v34;
        v161 = v106;
        LODWORD(v159) = v109;
        v160 = v108;
        sub_1ECE920(v34, 0, 0xAAAAAAAAAAAAAAABLL * ((__int64)&v35[-v34 - 24] >> 3), (int *)&v159, v107);
        v35 = (char *)v150;
      }
      v150 = (__m128i *)(v35 - 24);
      v117 = v132.m128i_i32[0];
      v126 = v156;
      if ( v156 != &v154 )
      {
        v119 = 88LL * v132.m128i_u32[0];
        while ( 1 )
        {
          v36 = v2[20];
          v120 = v126[8];
          v37 = *(volatile signed __int32 **)(v36 + v119 + 48);
          v118 = 88LL * v120;
          v38 = *(volatile signed __int32 **)(v36 + v118 + 48);
          if ( v37 != v38 )
          {
            if ( v37 >= v38 )
            {
              v159 = *(__int64 **)(v36 + v118 + 48);
              v160 = v37;
            }
            else
            {
              v159 = *(__int64 **)(v36 + v119 + 48);
              v160 = v38;
            }
            if ( (unsigned __int8)sub_1ECE060((__int64)&v145, (__int64 *)&v159, &v134) )
              goto LABEL_85;
          }
          v39 = v117;
          if ( v120 <= v117 )
            v39 = v120;
          v128 = v39;
          v40 = v117;
          if ( v120 >= v117 )
            v40 = v120;
          v129 = v40;
          v41 = sub_1ECE260((__int64)&v141, &v128, (int **)&v159);
          if ( v41 )
            goto LABEL_85;
          v42 = 0;
          v43 = *(__int64 (**)(void))(**(_QWORD **)(*v2 + 16LL) + 112LL);
          if ( v43 != sub_1D00B10 )
            v42 = v43();
          v44 = v2[20];
          v45 = *(_QWORD *)(v44 + v118 + 48);
          v121 = *(__int64 (__fastcall **)(__int64, __int64))(v44 + v119 + 48);
          v130 = v121;
          v131 = v45;
          if ( (unsigned __int8)sub_1ECE370((__int64)&v137, (__int64 *)&v130, &v159) && v159 != &v138[4 * v140] )
          {
            v88 = v159[3];
            v89 = (__int64 *)v159[2];
            if ( v88 )
            {
              v90 = (volatile signed __int32 *)(v88 + 8);
              if ( &_pthread_key_create )
              {
                _InterlockedAdd(v90, 1u);
                _InterlockedAdd(v90, 1u);
              }
              else
              {
                ++*(_DWORD *)(v88 + 8);
                ++*(_DWORD *)(v88 + 8);
              }
            }
            v159 = v89;
            v160 = (volatile signed __int32 *)v88;
            HIDWORD(v161) = v117;
            v163 = -1;
            v162 = v120;
            v164 = -1;
            v91 = sub_1ECF220(v2, (__int64)&v159);
            if ( v160 )
              sub_A191D0(v160);
            v92 = v2[19];
            if ( v92 )
              sub_1ECBE90(v92, v91);
            if ( v88 )
              sub_A191D0((volatile signed __int32 *)v88);
            goto LABEL_84;
          }
          v46 = *(_DWORD *)v121;
          HIDWORD(v134) = *(_DWORD *)v45 + 1;
          v47 = (unsigned int)(++v46 * HIDWORD(v134));
          LODWORD(v134) = v46;
          sub_1ECC890(&s, v47);
          v48 = (volatile signed __int32 *)((char *)s + 4 * v47);
          if ( s != v48 )
          {
            memset(s, 0, 4 * v47);
            v48 = (volatile signed __int32 *)s;
          }
          v49 = 0;
          if ( !*(_DWORD *)v121 )
            break;
          do
          {
            v50 = v49++;
            v51 = *(_DWORD *)(*((_QWORD *)v121 + 1) + 4 * v50);
            if ( *(_DWORD *)v45 )
            {
              v52 = 0;
              do
              {
                v53 = v52++;
                v54 = *(_DWORD *)(*(_QWORD *)(v45 + 8) + 4 * v53);
                if ( v51 != v54 )
                {
                  if ( v51 < 0 || v54 < 0 )
                    continue;
                  v55 = *(_QWORD *)(v42 + 8);
                  v56 = *(_QWORD *)(v42 + 56);
                  v57 = *(_DWORD *)(v55 + 24LL * (unsigned int)v51 + 16);
                  v58 = (_WORD *)(v56 + 2LL * (v57 >> 4));
                  v59 = *v58 + v51 * (v57 & 0xF);
                  v60 = v58 + 1;
                  LODWORD(v58) = *(_DWORD *)(v55 + 24LL * (unsigned int)v54 + 16);
                  v62 = v54 * ((unsigned __int8)v58 & 0xF);
                  v61 = (_WORD *)(v56 + 2LL * ((unsigned int)v58 >> 4));
                  LOWORD(v62) = *v61 + v62;
                  v63 = v61 + 1;
                  v64 = v59;
                  for ( j = (unsigned __int16)v62; ; j = (unsigned __int16)v62 )
                  {
                    v66 = v64 < j;
                    if ( v64 == j )
                      break;
                    while ( v66 )
                    {
                      v67 = v60 + 1;
                      v68 = *v60;
                      v59 += v68;
                      if ( !v68 )
                        goto LABEL_67;
                      v60 = v67;
                      v64 = v59;
                      v66 = v59 < j;
                      if ( v59 == j )
                        goto LABEL_66;
                    }
                    v87 = *v63;
                    if ( !(_WORD)v87 )
                      goto LABEL_67;
                    v62 += v87;
                    ++v63;
                  }
                }
LABEL_66:
                v41 = 1;
                v48[v52 + (unsigned __int64)(HIDWORD(v134) * v49)] = 2139095040;
                v48 = (volatile signed __int32 *)s;
LABEL_67:
                ;
              }
              while ( v52 != *(_DWORD *)v45 );
            }
          }
          while ( v49 != *(_DWORD *)v121 );
          if ( !v41 )
            break;
          v160 = v48;
          v69 = v134;
          v134 = 0;
          s = 0;
          v159 = v69;
          v70 = sub_1ED1D50(v2, v117, v120, (__int64 *)&v159);
          if ( v160 )
            j_j___libc_free_0_0(v160);
          v71 = (__int64 *)(v2[26] + 48LL * v70);
          v72 = sub_1ECE370((__int64)&v137, (__int64 *)&v130, &v159);
          v73 = v159;
          if ( v72 )
          {
            v74 = (volatile signed __int32 *)v159[3];
            goto LABEL_74;
          }
          v111 = v140;
          ++v137;
          v112 = v139 + 1;
          if ( 4 * ((int)v139 + 1) >= 3 * v140 )
          {
            v111 = 2 * v140;
          }
          else if ( v140 - HIDWORD(v139) - v112 > v140 >> 3 )
          {
            goto LABEL_150;
          }
          sub_1ECFB30((__int64)&v137, v111);
          sub_1ECE370((__int64)&v137, (__int64 *)&v130, &v159);
          v73 = v159;
          v112 = v139 + 1;
LABEL_150:
          LODWORD(v139) = v112;
          if ( *v73 != -8 || v73[1] != -8 )
            --HIDWORD(v139);
          v74 = 0;
          *v73 = (__int64)v130;
          v113 = v131;
          v73[2] = 0;
          v73[1] = v113;
          v73[3] = 0;
LABEL_74:
          v73[2] = *v71;
          v75 = (volatile signed __int32 *)v71[1];
          if ( v75 != v74 )
          {
            if ( v75 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v75 + 2, 1u);
              else
                ++*((_DWORD *)v75 + 2);
              v74 = (volatile signed __int32 *)v73[3];
            }
            if ( v74 )
              sub_A191D0(v74);
            v73[3] = (__int64)v75;
          }
          if ( s )
            j_j___libc_free_0_0(s);
LABEL_84:
          v76 = sub_1ECE260((__int64)&v141, &v128, (int **)&v159);
          v77 = v159;
          if ( !v76 )
          {
            v96 = v144;
            ++v141;
            v97 = v143 + 1;
            if ( 4 * ((int)v143 + 1) >= (unsigned int)(3 * v144) )
            {
              v96 = 2 * v144;
            }
            else if ( (int)v144 - HIDWORD(v143) - v97 > (unsigned int)v144 >> 3 )
            {
              goto LABEL_127;
            }
            sub_1ECF9B0((__int64)&v141, v96);
            sub_1ECE260((__int64)&v141, &v128, (int **)&v159);
            v77 = v159;
            v97 = v143 + 1;
LABEL_127:
            LODWORD(v143) = v97;
            if ( *(_DWORD *)v77 != -1 || *((_DWORD *)v77 + 1) != -1 )
              --HIDWORD(v143);
            *(_DWORD *)v77 = v128;
            *((_DWORD *)v77 + 1) = v129;
          }
LABEL_85:
          v126 = (int *)sub_220EF30(v126);
          if ( v126 == &v154 )
            goto LABEL_86;
        }
        if ( v48 )
          j_j___libc_free_0_0(v48);
        v98 = v2[20];
        v99 = *(volatile signed __int32 **)(v98 + v119 + 48);
        v100 = *(__int64 **)(v98 + v118 + 48);
        if ( v99 >= (volatile signed __int32 *)v100 )
        {
          v159 = v100;
          v160 = v99;
        }
        else
        {
          v159 = (__int64 *)v99;
          v160 = (volatile signed __int32 *)v100;
        }
        v101 = sub_1ECE060((__int64)&v145, (__int64 *)&v159, &v134);
        v102 = v134;
        if ( v101 )
          goto LABEL_85;
        v103 = v148;
        ++v145;
        v104 = v147 + 1;
        if ( 4 * ((int)v147 + 1) >= (unsigned int)(3 * v148) )
        {
          v103 = 2 * v148;
        }
        else if ( (int)v148 - HIDWORD(v147) - v104 > (unsigned int)v148 >> 3 )
        {
          goto LABEL_137;
        }
        sub_1ECF820((__int64)&v145, v103);
        sub_1ECE060((__int64)&v145, (__int64 *)&v159, &v134);
        v102 = v134;
        v104 = v147 + 1;
LABEL_137:
        LODWORD(v147) = v104;
        if ( *v102 != -8 || v102[1] != -8 )
          --HIDWORD(v147);
        *v102 = (__int64)v159;
        v102[1] = (__int64)v160;
        goto LABEL_85;
      }
LABEL_86:
      v78 = v155;
      if ( v155 )
      {
        while ( 1 )
        {
          v79 = v153((__int64)&v132, (__int64)(v78 + 8));
          v80 = (int *)*((_QWORD *)v78 + 3);
          if ( v79 )
            v80 = (int *)*((_QWORD *)v78 + 2);
          if ( !v80 )
            break;
          v78 = v80;
        }
        if ( !v79 )
        {
          if ( (unsigned __int8)v153((__int64)(v78 + 8), (__int64)&v132) )
            goto LABEL_94;
LABEL_96:
          v22 = v149;
          if ( v150 == v149 )
            goto LABEL_97;
          continue;
        }
        if ( v78 == v156 )
        {
LABEL_94:
          v81 = 1;
          if ( v78 != &v154 )
            goto LABEL_144;
        }
        else
        {
LABEL_142:
          v105 = sub_220EF80(v78);
          if ( !(unsigned __int8)v153(v105 + 32, (__int64)&v132) )
            goto LABEL_96;
          v81 = 1;
          if ( v78 != &v154 )
LABEL_144:
            v81 = (unsigned __int8)v153((__int64)&v132, (__int64)(v78 + 8));
        }
      }
      else
      {
        v78 = &v154;
        if ( v156 != &v154 )
          goto LABEL_142;
        v78 = &v154;
        v81 = 1;
      }
      break;
    }
    v82 = (__m128i *)sub_22077B0(56);
    v82[2] = _mm_loadu_si128(&v132);
    v82[3].m128i_i64[0] = (__int64)v133;
    sub_220F040(v81, v82, v78, &v154);
    ++v158;
    goto LABEL_96;
  }
LABEL_97:
  if ( v22 )
    j_j___libc_free_0(v22, v151 - (char *)v22);
  sub_1ECDC90((__int64)&v153, (__int64)v155);
  j___libc_free_0(v146);
  j___libc_free_0(v142);
  if ( v140 )
  {
    v83 = v138;
    v84 = &v138[4 * v140];
    do
    {
      while ( *v83 == -8 )
      {
        if ( v83[1] != -8 )
          goto LABEL_102;
        v83 += 4;
        if ( v84 == v83 )
          return j___libc_free_0(v138);
      }
      if ( *v83 != -16 || v83[1] != -16 )
      {
LABEL_102:
        v85 = (volatile signed __int32 *)v83[3];
        if ( v85 )
          sub_A191D0(v85);
      }
      v83 += 4;
    }
    while ( v84 != v83 );
  }
  return j___libc_free_0(v138);
}
