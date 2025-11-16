// Function: sub_358C950
// Address: 0x358c950
//
void __fastcall sub_358C950(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  void **v7; // rdi
  __m128i *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  const __m128i *v11; // rcx
  __int64 v12; // rsi
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  const __m128i *v15; // rax
  unsigned __int64 v16; // r15
  __int64 v17; // rax
  const __m128i *v18; // rcx
  unsigned __int64 v19; // rsi
  __m128i *v20; // rdx
  const __m128i *v21; // rax
  __int64 v22; // r14
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 *v25; // rdx
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 *v28; // rax
  __int64 v29; // rsi
  char v30; // dl
  unsigned __int64 v31; // rdx
  void **v32; // rax
  char v33; // cl
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rbx
  const __m128i *v37; // rax
  unsigned __int64 v38; // r13
  __int64 v39; // rax
  __m128i *v40; // rcx
  const __m128i *v41; // rdx
  void **v42; // r15
  __int64 v43; // rax
  unsigned __int64 v44; // rsi
  __m128i *v45; // rdx
  const __m128i *v46; // rax
  __int64 v47; // r14
  __int64 *v48; // rax
  __int64 *v49; // rdx
  _QWORD *v50; // rdi
  __int64 v51; // r13
  __int64 *v52; // rax
  __int64 v53; // rsi
  char v54; // dl
  unsigned __int64 v55; // rdx
  void **v56; // rax
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rax
  void *v59; // r15
  unsigned __int64 v60; // rax
  __int64 v61; // r13
  signed __int64 v62; // rbx
  __int64 v63; // rax
  char *v64; // r14
  signed __int64 v65; // rdx
  __int64 v66; // rbx
  __int64 v67; // r15
  char *v68; // rax
  char *v69; // rdx
  _QWORD *v70; // rax
  __int64 v71; // rdx
  int v72; // esi
  signed __int64 v73; // r13
  int v74; // edi
  __int64 *v75; // rcx
  unsigned int v76; // edx
  __int64 *v77; // rax
  __int64 v78; // r11
  void **v79; // rsi
  void *v80; // rdi
  void **v81; // r13
  char v82; // r11
  void **v83; // rbx
  __int64 v84; // rax
  void *v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rdi
  unsigned int v88; // ecx
  void **v89; // rax
  void *v90; // r9
  _QWORD *v91; // rax
  __int64 *v92; // r14
  __int64 *v93; // rbx
  __int64 v94; // rax
  _QWORD *v95; // rax
  __int64 v96; // r15
  unsigned __int64 v97; // rdi
  __int64 *v98; // r12
  __int64 *v99; // rbx
  __int64 v100; // rdx
  __int64 v101; // rcx
  unsigned __int64 v102; // rax
  __int64 v103; // r14
  unsigned __int64 v104; // rbx
  unsigned __int64 v105; // r12
  unsigned __int64 v106; // rdi
  unsigned __int64 v107; // rdi
  int v108; // eax
  int v109; // r10d
  int v110; // edx
  __int64 v111; // rcx
  __int64 v115; // [rsp+28h] [rbp-1D8h]
  unsigned __int64 v116; // [rsp+40h] [rbp-1C0h]
  unsigned __int64 v117; // [rsp+40h] [rbp-1C0h]
  unsigned __int64 v118; // [rsp+48h] [rbp-1B8h]
  unsigned __int64 v119; // [rsp+48h] [rbp-1B8h]
  void *v120; // [rsp+48h] [rbp-1B8h]
  __int64 v121; // [rsp+50h] [rbp-1B0h] BYREF
  void **v122; // [rsp+58h] [rbp-1A8h] BYREF
  unsigned __int64 v123; // [rsp+60h] [rbp-1A0h]
  char *v124; // [rsp+68h] [rbp-198h]
  void *src; // [rsp+70h] [rbp-190h] BYREF
  void **v126; // [rsp+78h] [rbp-188h] BYREF
  void **v127; // [rsp+80h] [rbp-180h]
  char *v128; // [rsp+88h] [rbp-178h]
  __m128i v129; // [rsp+90h] [rbp-170h] BYREF
  const __m128i *v130; // [rsp+A0h] [rbp-160h]
  __int64 v131; // [rsp+A8h] [rbp-158h]
  const __m128i *v132; // [rsp+B8h] [rbp-148h]
  const __m128i *v133; // [rsp+C0h] [rbp-140h]
  __m128i *v134; // [rsp+D0h] [rbp-130h] BYREF
  const __m128i *v135; // [rsp+D8h] [rbp-128h]
  const __m128i *v136; // [rsp+E0h] [rbp-120h]
  __int64 *v137; // [rsp+E8h] [rbp-118h]
  __int64 *v138; // [rsp+F0h] [rbp-110h]
  const __m128i *v139; // [rsp+F8h] [rbp-108h]
  __int64 v140; // [rsp+100h] [rbp-100h]
  __int64 v141; // [rsp+110h] [rbp-F0h] BYREF
  char *v142; // [rsp+118h] [rbp-E8h]
  __int64 v143; // [rsp+120h] [rbp-E0h]
  int v144; // [rsp+128h] [rbp-D8h]
  char v145; // [rsp+12Ch] [rbp-D4h]
  char v146; // [rsp+130h] [rbp-D0h] BYREF
  __m128i v147; // [rsp+170h] [rbp-90h] BYREF
  __int64 v148; // [rsp+180h] [rbp-80h]
  int v149; // [rsp+188h] [rbp-78h]
  char v150; // [rsp+18Ch] [rbp-74h]
  char v151; // [rsp+190h] [rbp-70h] BYREF

  v142 = &v146;
  v6 = *a1;
  v7 = (void **)&v129;
  v147.m128i_i64[0] = v6;
  v141 = 0;
  v143 = 8;
  v144 = 0;
  v145 = 1;
  sub_358C780(v129.m128i_i64, &v147, (__int64)&v141, a4, a5, a6);
  v11 = v130;
  v122 = 0;
  v12 = v129.m128i_i64[1];
  v123 = 0;
  v121 = v129.m128i_i64[0];
  v124 = 0;
  v13 = (unsigned __int64)v130 - v129.m128i_i64[1];
  if ( v130 == (const __m128i *)v129.m128i_i64[1] )
  {
    v13 = 0;
    v7 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_197;
    v14 = sub_22077B0((unsigned __int64)v130 - v129.m128i_i64[1]);
    v11 = v130;
    v12 = v129.m128i_i64[1];
    v7 = (void **)v14;
  }
  v122 = v7;
  v123 = (unsigned __int64)v7;
  v124 = (char *)v7 + v13;
  if ( v11 == (const __m128i *)v12 )
  {
    v16 = (unsigned __int64)v7;
  }
  else
  {
    v8 = (__m128i *)v7;
    v15 = (const __m128i *)v12;
    do
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(v15);
        v9 = v15[1].m128i_i64[0];
        v8[1].m128i_i64[0] = v9;
      }
      v15 = (const __m128i *)((char *)v15 + 24);
      v8 = (__m128i *)((char *)v8 + 24);
    }
    while ( v15 != v11 );
    v16 = (unsigned __int64)&v7[(((unsigned __int64)&v15[-2].m128i_u64[1] - v12) >> 3) + 3];
  }
  v12 = (__int64)v132;
  v123 = v16;
  if ( v133 == v132 )
  {
    v116 = 0;
    goto LABEL_81;
  }
  if ( (unsigned __int64)((char *)v133 - (char *)v132) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_197:
    sub_4261EA(v7, v12, v8);
  v17 = sub_22077B0((char *)v133 - (char *)v132);
  v18 = v133;
  v19 = (unsigned __int64)v132;
  v116 = v17;
  v7 = v122;
  v16 = v123;
  if ( v133 == v132 )
  {
LABEL_81:
    v118 = 0;
    goto LABEL_18;
  }
  v20 = (__m128i *)v17;
  v21 = v132;
  do
  {
    if ( v20 )
    {
      *v20 = _mm_loadu_si128(v21);
      v9 = v21[1].m128i_i64[0];
      v20[1].m128i_i64[0] = v9;
    }
    v21 = (const __m128i *)((char *)v21 + 24);
    v20 = (__m128i *)((char *)v20 + 24);
  }
  while ( v21 != v18 );
  v118 = 8 * (((unsigned __int64)&v21[-2].m128i_u64[1] - v19) >> 3) + 24;
  while ( 1 )
  {
LABEL_18:
    if ( v16 - (_QWORD)v7 != v118 )
      goto LABEL_19;
LABEL_30:
    if ( v7 == (void **)v16 )
      break;
    v31 = v116;
    v32 = v7;
    while ( *v32 == *(void **)v31 )
    {
      v33 = *((_BYTE *)v32 + 16);
      if ( v33 != *(_BYTE *)(v31 + 16) || v33 && v32[1] != *(void **)(v31 + 8) )
        break;
      v32 += 3;
      v31 += 24LL;
      if ( v32 == (void **)v16 )
        goto LABEL_37;
    }
LABEL_19:
    while ( 2 )
    {
      v22 = *(_QWORD *)(v16 - 24);
      if ( !*(_BYTE *)(v16 - 8) )
      {
        v23 = *(__int64 **)(v22 + 112);
        *(_BYTE *)(v16 - 8) = 1;
        *(_QWORD *)(v16 - 16) = v23;
        goto LABEL_21;
      }
      while ( 1 )
      {
        v23 = *(__int64 **)(v16 - 16);
LABEL_21:
        v24 = *(unsigned int *)(v22 + 120);
        if ( v23 == (__int64 *)(*(_QWORD *)(v22 + 112) + 8 * v24) )
          break;
        v25 = v23 + 1;
        *(_QWORD *)(v16 - 16) = v23 + 1;
        v26 = (_QWORD *)v121;
        v27 = *v23;
        if ( !*(_BYTE *)(v121 + 28) )
          goto LABEL_28;
        v28 = *(__int64 **)(v121 + 8);
        v29 = *(unsigned int *)(v121 + 20);
        v25 = &v28[v29];
        if ( v28 == v25 )
        {
LABEL_76:
          if ( (unsigned int)v29 < *(_DWORD *)(v121 + 16) )
          {
            *(_DWORD *)(v121 + 20) = v29 + 1;
            *v25 = v27;
            ++*v26;
LABEL_29:
            v147.m128i_i64[0] = v27;
            LOBYTE(v148) = 0;
            sub_358C570((unsigned __int64 *)&v122, &v147);
            v16 = v123;
            v7 = v122;
            if ( v123 - (_QWORD)v122 != v118 )
              goto LABEL_19;
            goto LABEL_30;
          }
LABEL_28:
          sub_C8CC70(v121, v27, (__int64)v25, v24, v9, v10);
          if ( v30 )
            goto LABEL_29;
        }
        else
        {
          while ( v27 != *v28 )
          {
            if ( v25 == ++v28 )
              goto LABEL_76;
          }
        }
      }
      v123 -= 24LL;
      v7 = v122;
      v16 = v123;
      if ( (void **)v123 != v122 )
        continue;
      break;
    }
  }
LABEL_37:
  if ( v116 )
  {
    j_j___libc_free_0(v116);
    v7 = v122;
  }
  if ( v7 )
    j_j___libc_free_0((unsigned __int64)v7);
  if ( v132 )
    j_j___libc_free_0((unsigned __int64)v132);
  if ( v129.m128i_i64[1] )
    j_j___libc_free_0(v129.m128i_u64[1]);
  v148 = 8;
  v34 = (__int64)&v126;
  v147.m128i_i64[1] = (__int64)&v151;
  v147.m128i_i64[0] = 0;
  v35 = *a1;
  v149 = 0;
  v150 = 1;
  v36 = *(_QWORD *)(v35 + 328);
  v115 = v35 + 320;
  if ( v36 != v35 + 320 )
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v36 + 120) )
        goto LABEL_47;
      v7 = (void **)&v134;
      v129.m128i_i64[0] = v36;
      sub_358C5B0((__int64 *)&v134, &v129, (__int64)&v147, v34, v9, v10);
      v12 = (__int64)v135;
      v126 = 0;
      v127 = 0;
      src = v134;
      v37 = v136;
      v128 = 0;
      v38 = (char *)v136 - (char *)v135;
      if ( v136 == v135 )
      {
        v7 = 0;
      }
      else
      {
        if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_197;
        v39 = sub_22077B0((char *)v136 - (char *)v135);
        v12 = (__int64)v135;
        v7 = (void **)v39;
        v37 = v136;
      }
      v126 = v7;
      v127 = v7;
      v128 = (char *)v7 + v38;
      if ( v37 == (const __m128i *)v12 )
      {
        v42 = v7;
      }
      else
      {
        v40 = (__m128i *)v7;
        v41 = (const __m128i *)v12;
        do
        {
          if ( v40 )
          {
            *v40 = _mm_loadu_si128(v41);
            v9 = v41[1].m128i_i64[0];
            v40[1].m128i_i64[0] = v9;
          }
          v41 = (const __m128i *)((char *)v41 + 24);
          v40 = (__m128i *)((char *)v40 + 24);
        }
        while ( v37 != v41 );
        v42 = &v7[3
                * ((0xAAAAAAAAAAAAAABLL * (((unsigned __int64)&v37[-2].m128i_u64[1] - v12) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
                + 3];
      }
      v34 = v140;
      v12 = (__int64)v139;
      v127 = v42;
      v8 = (__m128i *)(v140 - (_QWORD)v139);
      if ( (const __m128i *)v140 == v139 )
      {
        v117 = 0;
LABEL_173:
        v119 = 0;
        goto LABEL_66;
      }
      if ( (unsigned __int64)v8 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_197;
      v43 = sub_22077B0(v140 - (_QWORD)v139);
      v34 = v140;
      v44 = (unsigned __int64)v139;
      v117 = v43;
      v7 = v126;
      v42 = v127;
      if ( v139 == (const __m128i *)v140 )
        goto LABEL_173;
      v45 = (__m128i *)v43;
      v46 = v139;
      do
      {
        if ( v45 )
        {
          *v45 = _mm_loadu_si128(v46);
          v9 = v46[1].m128i_i64[0];
          v45[1].m128i_i64[0] = v9;
        }
        v46 = (const __m128i *)((char *)v46 + 24);
        v45 = (__m128i *)((char *)v45 + 24);
      }
      while ( v46 != (const __m128i *)v34 );
      v34 = 0x1FFFFFFFFFFFFFFFLL;
      v119 = 8
           * (3
            * ((0xAAAAAAAAAAAAAABLL * (((unsigned __int64)&v46[-2].m128i_u64[1] - v44) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
            + 3);
LABEL_66:
      if ( (char *)v42 - (char *)v7 != v119 )
        goto LABEL_67;
LABEL_86:
      if ( v42 != v7 )
      {
        v55 = v117;
        v56 = v7;
        while ( *v56 == *(void **)v55 )
        {
          v34 = *((unsigned __int8 *)v56 + 16);
          if ( (_BYTE)v34 != *(_BYTE *)(v55 + 16) )
            break;
          if ( (_BYTE)v34 )
          {
            v34 = *(_QWORD *)(v55 + 8);
            if ( v56[1] != (void *)v34 )
              break;
          }
          v56 += 3;
          v55 += 24LL;
          if ( v42 == v56 )
            goto LABEL_93;
        }
LABEL_67:
        v47 = (__int64)*(v42 - 3);
        if ( !*((_BYTE *)v42 - 8) )
        {
          v48 = *(__int64 **)(v47 + 64);
          *((_BYTE *)v42 - 8) = 1;
          *(v42 - 2) = v48;
          goto LABEL_69;
        }
        while ( 1 )
        {
          v48 = (__int64 *)*(v42 - 2);
LABEL_69:
          v34 = *(unsigned int *)(v47 + 72);
          if ( v48 == (__int64 *)(*(_QWORD *)(v47 + 64) + 8 * v34) )
          {
            v127 -= 3;
            v7 = v126;
            v42 = v127;
            if ( v127 != v126 )
              goto LABEL_67;
            goto LABEL_66;
          }
          v49 = v48 + 1;
          *(v42 - 2) = v48 + 1;
          v50 = src;
          v51 = *v48;
          if ( *((_BYTE *)src + 28) )
          {
            v52 = (__int64 *)*((_QWORD *)src + 1);
            v53 = *((unsigned int *)src + 5);
            v49 = &v52[v53];
            if ( v52 != v49 )
            {
              while ( v51 != *v52 )
              {
                if ( v49 == ++v52 )
                  goto LABEL_159;
              }
              continue;
            }
LABEL_159:
            if ( (unsigned int)v53 < *((_DWORD *)src + 4) )
            {
              *((_DWORD *)src + 5) = v53 + 1;
              *v49 = v51;
              ++*v50;
LABEL_85:
              v129.m128i_i64[0] = v51;
              LOBYTE(v130) = 0;
              sub_358C570((unsigned __int64 *)&v126, &v129);
              v42 = v127;
              v7 = v126;
              if ( (char *)v127 - (char *)v126 != v119 )
                goto LABEL_67;
              goto LABEL_86;
            }
          }
          sub_C8CC70((__int64)src, v51, (__int64)v49, v34, v9, v10);
          if ( v54 )
            goto LABEL_85;
        }
      }
LABEL_93:
      if ( v117 )
      {
        j_j___libc_free_0(v117);
        v7 = v126;
      }
      if ( v7 )
        j_j___libc_free_0((unsigned __int64)v7);
      if ( v139 )
        j_j___libc_free_0((unsigned __int64)v139);
      if ( !v135 )
      {
LABEL_47:
        v36 = *(_QWORD *)(v36 + 8);
        if ( v115 == v36 )
          break;
      }
      else
      {
        j_j___libc_free_0((unsigned __int64)v135);
        v36 = *(_QWORD *)(v36 + 8);
        if ( v115 == v36 )
          break;
      }
    }
  }
  LODWORD(v131) = 0;
  v129.m128i_i64[1] = 0;
  v130 = 0;
  src = 0;
  v126 = 0;
  v127 = 0;
  if ( HIDWORD(v143) == v144 )
  {
    v129.m128i_i64[0] = 1;
  }
  else
  {
    v129.m128i_i64[0] = 1;
    v57 = (4 * (HIDWORD(v143) - v144) / 3u + 1) | ((unsigned __int64)(4 * (HIDWORD(v143) - v144) / 3u + 1) >> 1);
    v58 = (((v57 >> 2) | v57) >> 4) | (v57 >> 2) | v57;
    sub_2E3E470((__int64)&v129, ((((v58 >> 8) | v58) >> 16) | (v58 >> 8) | v58) + 1);
    v59 = src;
    v60 = (unsigned int)(HIDWORD(v143) - v144);
    if ( v60 > ((char *)v127 - (_BYTE *)src) >> 3 )
    {
      v61 = 8 * v60;
      v62 = (char *)v126 - (_BYTE *)src;
      if ( HIDWORD(v143) == v144 )
      {
        v65 = (char *)v126 - (_BYTE *)src;
        v64 = 0;
      }
      else
      {
        v63 = sub_22077B0(8 * v60);
        v59 = src;
        v64 = (char *)v63;
        v65 = (char *)v126 - (_BYTE *)src;
      }
      if ( v65 > 0 )
      {
        memmove(v64, v59, v65);
      }
      else if ( !v59 )
      {
        goto LABEL_107;
      }
      j_j___libc_free_0((unsigned __int64)v59);
LABEL_107:
      src = v64;
      v126 = (void **)&v64[v62];
      v127 = (void **)&v64[v61];
    }
  }
  v66 = *(_QWORD *)(*a1 + 328);
  v67 = *a1 + 320;
  if ( v67 != v66 )
  {
    while ( 2 )
    {
      if ( v145 )
      {
        v68 = v142;
        v69 = &v142[8 * HIDWORD(v143)];
        if ( v142 == v69 )
          goto LABEL_125;
        while ( v66 != *(_QWORD *)v68 )
        {
          v68 += 8;
          if ( v69 == v68 )
            goto LABEL_125;
        }
      }
      else if ( !sub_C8CA60((__int64)&v141, v66) )
      {
        goto LABEL_125;
      }
      if ( v150 )
      {
        v70 = (_QWORD *)v147.m128i_i64[1];
        v71 = v147.m128i_i64[1] + 8LL * HIDWORD(v148);
        if ( v147.m128i_i64[1] == v71 )
          goto LABEL_125;
        while ( v66 != *v70 )
        {
          if ( (_QWORD *)v71 == ++v70 )
            goto LABEL_125;
        }
      }
      else if ( !sub_C8CA60((__int64)&v147, v66) )
      {
        goto LABEL_125;
      }
      v72 = v131;
      v121 = v66;
      v73 = ((char *)v126 - (_BYTE *)src) >> 3;
      if ( (_DWORD)v131 )
      {
        v74 = 1;
        v75 = 0;
        v76 = (v131 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v77 = (__int64 *)(v129.m128i_i64[1] + 16LL * v76);
        v78 = *v77;
        if ( v66 == *v77 )
        {
LABEL_121:
          v77[1] = v73;
          v79 = v126;
          v134 = (__m128i *)v66;
          if ( v126 == v127 )
          {
            sub_2E3CE90((__int64)&src, v126, &v134);
          }
          else
          {
            if ( v126 )
            {
              *v126 = (void *)v66;
              v79 = v126;
            }
            v126 = v79 + 1;
          }
LABEL_125:
          v66 = *(_QWORD *)(v66 + 8);
          if ( v67 == v66 )
            goto LABEL_126;
          continue;
        }
        while ( v78 != -4096 )
        {
          if ( v78 == -8192 && !v75 )
            v75 = v77;
          v76 = (v131 - 1) & (v74 + v76);
          v77 = (__int64 *)(v129.m128i_i64[1] + 16LL * v76);
          v78 = *v77;
          if ( v66 == *v77 )
            goto LABEL_121;
          ++v74;
        }
        if ( v75 )
          v77 = v75;
        ++v129.m128i_i64[0];
        v110 = (_DWORD)v130 + 1;
        v134 = (__m128i *)v77;
        if ( 4 * ((int)v130 + 1) < (unsigned int)(3 * v131) )
        {
          v111 = v66;
          if ( (int)v131 - HIDWORD(v130) - v110 > (unsigned int)v131 >> 3 )
          {
LABEL_191:
            LODWORD(v130) = v110;
            if ( *v77 != -4096 )
              --HIDWORD(v130);
            *v77 = v111;
            v77[1] = 0;
            goto LABEL_121;
          }
LABEL_196:
          sub_2E3E470((__int64)&v129, v72);
          sub_3585550((__int64)&v129, &v121, &v134);
          v111 = v121;
          v110 = (_DWORD)v130 + 1;
          v77 = (__int64 *)v134;
          goto LABEL_191;
        }
      }
      else
      {
        ++v129.m128i_i64[0];
        v134 = 0;
      }
      break;
    }
    v72 = 2 * v131;
    goto LABEL_196;
  }
LABEL_126:
  sub_3585700(a2);
  sub_35858D0(a3);
  v80 = src;
  v81 = v126;
  v82 = 0;
  v83 = (void **)src;
  if ( v126 == src )
    goto LABEL_153;
  while ( 2 )
  {
    while ( 2 )
    {
      v84 = a1[2];
      v85 = *v83;
      v134 = (__m128i *)*v83;
      v86 = *(unsigned int *)(v84 + 24);
      v87 = *(_QWORD *)(v84 + 8);
      if ( !(_DWORD)v86 )
      {
LABEL_128:
        if ( v81 == ++v83 )
          goto LABEL_134;
        continue;
      }
      break;
    }
    v88 = (v86 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
    v89 = (void **)(v87 + 16LL * v88);
    v90 = *v89;
    if ( v85 != *v89 )
    {
      v108 = 1;
      while ( v90 != (void *)-4096LL )
      {
        v109 = v108 + 1;
        v88 = (v86 - 1) & (v108 + v88);
        v89 = (void **)(v87 + 16LL * v88);
        v90 = *v89;
        if ( v85 == *v89 )
          goto LABEL_131;
        v108 = v109;
      }
      goto LABEL_128;
    }
LABEL_131:
    if ( v89 == (void **)(v87 + 16 * v86) || !v89[1] )
      goto LABEL_128;
    v120 = v89[1];
    ++v83;
    v91 = sub_3588500(a2, (__int64 *)&v134);
    v82 = 1;
    *v91 = v120;
    if ( v81 != v83 )
      continue;
    break;
  }
LABEL_134:
  v80 = src;
  if ( (unsigned __int64)((char *)v126 - (_BYTE *)src) > 8 && v82 )
  {
    sub_358B4D0((unsigned __int64 *)&v134, (__int64)a1, (char ***)&src, (__int64)&v129);
    sub_2A60C60(&v134);
    v92 = (__int64 *)v126;
    v93 = (__int64 *)src;
    if ( v126 != src )
    {
      do
      {
        v94 = *v93++;
        v121 = v94;
        v95 = sub_3588500((__int64)&v129, &v121);
        v96 = v134[5 * *v95 + 1].m128i_i64[1];
        *sub_3588500(a2, &v121) = v96;
      }
      while ( v92 != v93 );
    }
    v97 = (unsigned __int64)v137;
    v98 = v138;
    v99 = v137;
    if ( v138 != v137 )
    {
      do
      {
        v100 = *v99;
        v101 = v99[1];
        v99 += 5;
        v102 = *((_QWORD *)src + v101);
        v121 = *((_QWORD *)src + v100);
        v122 = (void **)v102;
        v103 = *(v99 - 1);
        *sub_3589880(a3, &v121) = v103;
      }
      while ( v98 != v99 );
      v97 = (unsigned __int64)v137;
    }
    if ( v97 )
      j_j___libc_free_0(v97);
    v104 = (unsigned __int64)v135;
    v105 = (unsigned __int64)v134;
    if ( v135 != v134 )
    {
      do
      {
        v106 = *(_QWORD *)(v105 + 56);
        if ( v106 )
          j_j___libc_free_0(v106);
        v107 = *(_QWORD *)(v105 + 32);
        if ( v107 )
          j_j___libc_free_0(v107);
        v105 += 80LL;
      }
      while ( v104 != v105 );
      v105 = (unsigned __int64)v134;
    }
    if ( v105 )
      j_j___libc_free_0(v105);
    v80 = src;
  }
LABEL_153:
  if ( v80 )
    j_j___libc_free_0((unsigned __int64)v80);
  sub_C7D6A0(v129.m128i_i64[1], 16LL * (unsigned int)v131, 8);
  if ( !v150 )
    _libc_free(v147.m128i_u64[1]);
  if ( !v145 )
    _libc_free((unsigned __int64)v142);
}
