// Function: sub_26D93E0
// Address: 0x26d93e0
//
void __fastcall sub_26D93E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  void **v8; // rdi
  __m128i *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  const __m128i *v13; // rdx
  const __m128i *v14; // rax
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  void **v17; // r14
  __m128i *v18; // rax
  __m128i *v19; // rax
  const __m128i *v20; // rdx
  signed __int64 v21; // r15
  void **v22; // r15
  __int64 v23; // rbx
  __int64 v24; // r13
  unsigned __int64 v25; // rax
  int v26; // edx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  int v29; // eax
  unsigned int v30; // esi
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 *v33; // rdx
  _QWORD *v34; // rdi
  __int64 v35; // r14
  __int64 *v36; // rax
  __int64 v37; // rsi
  unsigned __int64 v38; // rdx
  void **v39; // rax
  __int64 v40; // rax
  __int64 v41; // rbx
  unsigned __int64 v42; // rax
  char v43; // dl
  __m128i *v44; // rcx
  unsigned __int64 v45; // r12
  __int64 v46; // rax
  __m128i *v47; // rdx
  const __m128i *v48; // rax
  void **v49; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // rsi
  __m128i *v52; // rdx
  const __m128i *v53; // rax
  __int64 i; // rax
  __int64 v55; // rdx
  _QWORD *v56; // rdi
  __int64 v57; // r12
  _QWORD *v58; // rax
  unsigned __int64 v59; // rdx
  void **v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  void *v63; // r15
  unsigned __int64 v64; // rax
  __int64 v65; // r13
  signed __int64 v66; // rbx
  __int64 v67; // rax
  char *v68; // r14
  signed __int64 v69; // rdx
  __int64 v70; // r14
  __int64 v71; // rbx
  char *v72; // rax
  char *v73; // rdx
  _QWORD *v74; // rax
  __int64 v75; // rdx
  int v76; // esi
  int v77; // edi
  __int64 *v78; // rcx
  unsigned int v79; // edx
  __int64 *v80; // rax
  __int64 v81; // r8
  void **v82; // rsi
  int v83; // edx
  __int64 v84; // rax
  _QWORD *v85; // rdx
  _QWORD *k; // rax
  int v87; // eax
  __int64 v88; // rsi
  _QWORD *v89; // rax
  _QWORD *m; // rdx
  void *v91; // rdi
  void **v92; // r13
  void **v93; // rbx
  char v94; // r11
  __int64 v95; // rax
  void *v96; // rsi
  __int64 v97; // rdx
  __int64 v98; // rdi
  unsigned int v99; // ecx
  void **v100; // rax
  void *v101; // r9
  __int64 *v102; // rax
  __int64 *v103; // r14
  __int64 *v104; // rbx
  __int64 v105; // rax
  __int64 *v106; // rax
  __int64 v107; // r15
  unsigned __int64 v108; // rdi
  __int64 *v109; // r12
  __int64 *v110; // rbx
  __int64 v111; // rdx
  __int64 v112; // rcx
  void **v113; // rax
  __int64 v114; // r14
  unsigned __int64 v115; // rbx
  unsigned __int64 v116; // r12
  unsigned __int64 v117; // rdi
  unsigned __int64 v118; // rdi
  char v119; // dl
  int v120; // eax
  int v121; // r10d
  unsigned int v122; // edx
  unsigned int v123; // eax
  int v124; // r13d
  unsigned int v125; // eax
  unsigned int v126; // ecx
  unsigned int v127; // edx
  int v128; // r13d
  unsigned int v129; // eax
  int v130; // edx
  __int64 v131; // rcx
  __int64 v132; // [rsp+0h] [rbp-200h]
  __int64 v134; // [rsp+20h] [rbp-1E0h]
  unsigned __int64 v135; // [rsp+38h] [rbp-1C8h]
  unsigned __int64 v136; // [rsp+38h] [rbp-1C8h]
  signed __int64 v137; // [rsp+38h] [rbp-1C8h]
  __int64 v138; // [rsp+40h] [rbp-1C0h]
  unsigned __int64 v139; // [rsp+40h] [rbp-1C0h]
  __int64 j; // [rsp+40h] [rbp-1C0h]
  __int64 v141; // [rsp+40h] [rbp-1C0h]
  __int64 v143; // [rsp+50h] [rbp-1B0h] BYREF
  void **v144; // [rsp+58h] [rbp-1A8h] BYREF
  void **v145; // [rsp+60h] [rbp-1A0h]
  char *v146; // [rsp+68h] [rbp-198h]
  void *src; // [rsp+70h] [rbp-190h] BYREF
  void **v148; // [rsp+78h] [rbp-188h] BYREF
  void **v149; // [rsp+80h] [rbp-180h]
  char *v150; // [rsp+88h] [rbp-178h]
  __m128i v151; // [rsp+90h] [rbp-170h] BYREF
  const __m128i *v152; // [rsp+A0h] [rbp-160h]
  __int64 v153; // [rsp+A8h] [rbp-158h]
  const __m128i *v154; // [rsp+B8h] [rbp-148h]
  const __m128i *v155; // [rsp+C0h] [rbp-140h]
  __m128i *v156; // [rsp+D0h] [rbp-130h] BYREF
  __m128i *v157; // [rsp+D8h] [rbp-128h]
  __m128i *v158; // [rsp+E0h] [rbp-120h]
  __int64 *v159; // [rsp+E8h] [rbp-118h]
  __int64 *v160; // [rsp+F0h] [rbp-110h]
  __m128i *v161; // [rsp+F8h] [rbp-108h]
  __int64 v162; // [rsp+100h] [rbp-100h]
  __int64 v163; // [rsp+110h] [rbp-F0h] BYREF
  char *v164; // [rsp+118h] [rbp-E8h]
  __int64 v165; // [rsp+120h] [rbp-E0h]
  int v166; // [rsp+128h] [rbp-D8h]
  char v167; // [rsp+12Ch] [rbp-D4h]
  char v168; // [rsp+130h] [rbp-D0h] BYREF
  __m128i v169; // [rsp+170h] [rbp-90h] BYREF
  __int64 v170; // [rsp+180h] [rbp-80h]
  int v171; // [rsp+188h] [rbp-78h]
  char v172; // [rsp+18Ch] [rbp-74h]
  char v173; // [rsp+190h] [rbp-70h] BYREF

  v164 = &v168;
  v7 = *a1;
  v8 = (void **)&v151;
  v169.m128i_i64[0] = v7;
  v9 = &v169;
  v163 = 0;
  v165 = 8;
  v166 = 0;
  v167 = 1;
  sub_26D8E50(v151.m128i_i64, &v169, (__int64)&v163, a4, a5, a6);
  v13 = (const __m128i *)v151.m128i_i64[1];
  v144 = 0;
  v145 = 0;
  v143 = v151.m128i_i64[0];
  v14 = v152;
  v146 = 0;
  v15 = (unsigned __int64)v152 - v151.m128i_i64[1];
  if ( v152 == (const __m128i *)v151.m128i_i64[1] )
  {
    v8 = 0;
  }
  else
  {
    if ( v15 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_254;
    v16 = sub_22077B0((unsigned __int64)v152 - v151.m128i_i64[1]);
    v13 = (const __m128i *)v151.m128i_i64[1];
    v8 = (void **)v16;
    v14 = v152;
  }
  v144 = v8;
  v145 = v8;
  v146 = (char *)v8 + v15;
  if ( v13 == v14 )
  {
    v17 = v8;
  }
  else
  {
    v17 = (void **)((char *)v8 + (char *)v14 - (char *)v13);
    v18 = (__m128i *)v8;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v13);
        v18[1] = _mm_loadu_si128(v13 + 1);
      }
      v18 += 2;
      v13 += 2;
    }
    while ( v18 != (__m128i *)v17 );
  }
  v13 = v154;
  v145 = v17;
  v9 = (__m128i *)((char *)v155 - (char *)v154);
  if ( v155 == v154 )
  {
    v135 = 0;
    goto LABEL_67;
  }
  if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_254:
    sub_4261EA(v8, v9, v13);
  v19 = (__m128i *)sub_22077B0((char *)v155 - (char *)v154);
  v20 = v154;
  v8 = v144;
  v135 = (unsigned __int64)v19;
  v17 = v145;
  if ( v155 != v154 )
  {
    v21 = (char *)v155 - (char *)v154;
    v10 = (__int64)v19->m128i_i64 + (char *)v155 - (char *)v154;
    do
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v20);
        v19[1] = _mm_loadu_si128(v20 + 1);
      }
      v19 += 2;
      v20 += 2;
    }
    while ( v19 != (__m128i *)v10 );
    v138 = v21;
    goto LABEL_17;
  }
LABEL_67:
  v138 = 0;
LABEL_17:
  v22 = v17;
  while ( v138 != (char *)v22 - (char *)v8 )
  {
LABEL_19:
    while ( 2 )
    {
      v23 = (__int64)*(v22 - 4);
      v24 = v23 + 48;
      if ( !*((_BYTE *)v22 - 8) )
      {
        v25 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v24 == v25 )
        {
          v27 = 0;
        }
        else
        {
          if ( !v25 )
            goto LABEL_59;
          v26 = *(unsigned __int8 *)(v25 - 24);
          v27 = v25 - 24;
          if ( (unsigned int)(v26 - 30) >= 0xB )
            v27 = 0;
        }
        *(v22 - 3) = (void *)v27;
        *((_DWORD *)v22 - 4) = 0;
        *((_BYTE *)v22 - 8) = 1;
      }
LABEL_25:
      v28 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v24 == v28 )
        goto LABEL_62;
LABEL_26:
      if ( !v28 )
LABEL_59:
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 <= 0xA )
      {
        v29 = sub_B46E30(v28 - 24);
        v30 = *((_DWORD *)v22 - 4);
        if ( v30 == v29 )
          goto LABEL_63;
        goto LABEL_29;
      }
LABEL_62:
      while ( 1 )
      {
        v30 = *((_DWORD *)v22 - 4);
        if ( !v30 )
          break;
LABEL_29:
        v31 = (__int64)*(v22 - 3);
        *((_DWORD *)v22 - 4) = v30 + 1;
        v32 = sub_B46EC0(v31, v30);
        v34 = (_QWORD *)v143;
        v35 = v32;
        if ( *(_BYTE *)(v143 + 28) )
        {
          v36 = *(__int64 **)(v143 + 8);
          v37 = *(unsigned int *)(v143 + 20);
          v33 = &v36[v37];
          if ( v36 != v33 )
          {
            while ( v35 != *v36 )
            {
              if ( v33 == ++v36 )
                goto LABEL_33;
            }
            goto LABEL_25;
          }
LABEL_33:
          if ( (unsigned int)v37 < *(_DWORD *)(v143 + 16) )
          {
            *(_DWORD *)(v143 + 20) = v37 + 1;
            *v33 = v35;
            ++*v34;
LABEL_35:
            v169.m128i_i64[0] = v35;
            LOBYTE(v171) = 0;
            sub_26D8E10((__int64)&v144, &v169);
            v22 = v145;
            v8 = v144;
            if ( v138 != (char *)v145 - (char *)v144 )
              goto LABEL_19;
            goto LABEL_36;
          }
        }
        sub_C8CC70(v143, v35, (__int64)v33, v10, v11, v12);
        if ( v43 )
          goto LABEL_35;
        v28 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v24 != v28 )
          goto LABEL_26;
      }
LABEL_63:
      v145 -= 4;
      v8 = v144;
      v22 = v145;
      if ( v145 != v144 )
        continue;
      break;
    }
  }
LABEL_36:
  if ( v22 != v8 )
  {
    v38 = v135;
    v39 = v8;
    while ( *v39 == *(void **)v38 )
    {
      v10 = *((unsigned __int8 *)v39 + 24);
      if ( (_BYTE)v10 != *(_BYTE *)(v38 + 24) || (_BYTE)v10 && *((_DWORD *)v39 + 4) != *(_DWORD *)(v38 + 16) )
        break;
      v39 += 4;
      v38 += 32LL;
      if ( v22 == v39 )
        goto LABEL_43;
    }
    goto LABEL_19;
  }
LABEL_43:
  if ( v135 )
  {
    j_j___libc_free_0(v135);
    v8 = v144;
  }
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
  if ( v154 )
    j_j___libc_free_0((unsigned __int64)v154);
  if ( v151.m128i_i64[1] )
    j_j___libc_free_0(v151.m128i_u64[1]);
  v170 = 8;
  v169.m128i_i64[1] = (__int64)&v173;
  v169.m128i_i64[0] = 0;
  v40 = *a1;
  v171 = 0;
  v172 = 1;
  v41 = *(_QWORD *)(v40 + 80);
  v134 = v40 + 72;
  if ( v41 != v40 + 72 )
  {
    v132 = a2;
    while ( 1 )
    {
      if ( !v41 )
        BUG();
      v42 = *(_QWORD *)(v41 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v42 != v41 + 24 )
      {
        if ( !v42 )
          goto LABEL_59;
        if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 <= 0xA && (unsigned int)sub_B46E30(v42 - 24) )
        {
LABEL_55:
          v41 = *(_QWORD *)(v41 + 8);
          if ( v134 == v41 )
            goto LABEL_115;
          continue;
        }
      }
      v8 = (void **)&v156;
      v151.m128i_i64[0] = v41 - 24;
      sub_26D9210((__int64 *)&v156, &v151, (__int64)&v169, v10, v11, v12);
      v44 = v158;
      v148 = 0;
      v9 = v157;
      v149 = 0;
      src = v156;
      v150 = 0;
      v45 = (char *)v158 - (char *)v157;
      if ( v158 == v157 )
      {
        v8 = 0;
      }
      else
      {
        if ( v45 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_254;
        v46 = sub_22077B0((char *)v158 - (char *)v157);
        v44 = v158;
        v9 = v157;
        v8 = (void **)v46;
      }
      v148 = v8;
      v149 = v8;
      v150 = (char *)v8 + v45;
      if ( v9 == v44 )
      {
        v49 = v8;
      }
      else
      {
        v47 = (__m128i *)v8;
        v48 = v9;
        do
        {
          if ( v47 )
          {
            *v47 = _mm_loadu_si128(v48);
            v11 = v48[1].m128i_i64[0];
            v47[1].m128i_i64[0] = v11;
          }
          v48 = (const __m128i *)((char *)v48 + 24);
          v47 = (__m128i *)((char *)v47 + 24);
        }
        while ( v48 != v44 );
        v49 = &v8[3
                * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&v48[-2].m128i_u64[1] - (char *)v9) >> 3))
                 & 0x1FFFFFFFFFFFFFFFLL)
                + 3];
      }
      v10 = v162;
      v9 = v161;
      v149 = v49;
      v13 = (const __m128i *)(v162 - (_QWORD)v161);
      if ( (__m128i *)v162 == v161 )
        break;
      if ( (unsigned __int64)v13 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_254;
      v50 = sub_22077B0(v162 - (_QWORD)v161);
      v10 = v162;
      v51 = (unsigned __int64)v161;
      v136 = v50;
      v8 = v148;
      v49 = v149;
      if ( (__m128i *)v162 == v161 )
        goto LABEL_227;
      v52 = (__m128i *)v50;
      v53 = v161;
      do
      {
        if ( v52 )
        {
          *v52 = _mm_loadu_si128(v53);
          v11 = v53[1].m128i_i64[0];
          v52[1].m128i_i64[0] = v11;
        }
        v53 = (const __m128i *)((char *)v53 + 24);
        v52 = (__m128i *)((char *)v52 + 24);
      }
      while ( v53 != (const __m128i *)v10 );
      v139 = 8
           * (3
            * ((0xAAAAAAAAAAAAAABLL * (((unsigned __int64)&v53[-2].m128i_u64[1] - v51) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
            + 3);
LABEL_87:
      if ( (char *)v49 - (char *)v8 == v139 )
        goto LABEL_100;
      while ( 1 )
      {
        do
        {
          if ( *((_BYTE *)v49 - 8) )
            goto LABEL_89;
          while ( 1 )
          {
            for ( i = *((_QWORD *)*(v49 - 3) + 2); i; i = *(_QWORD *)(i + 8) )
            {
              if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
                break;
            }
            *(v49 - 2) = (void *)i;
            *((_BYTE *)v49 - 8) = 1;
LABEL_90:
            if ( i )
              break;
            v149 -= 3;
            v8 = v148;
            v49 = v149;
            if ( v149 == v148 )
              goto LABEL_87;
            if ( *((_BYTE *)v149 - 8) )
              goto LABEL_89;
          }
          v55 = *(_QWORD *)(i + 8);
          for ( *(v49 - 2) = (void *)v55; v55; *(v49 - 2) = (void *)v55 )
          {
            v10 = (unsigned int)**(unsigned __int8 **)(v55 + 24) - 30;
            if ( (unsigned __int8)(**(_BYTE **)(v55 + 24) - 30) <= 0xAu )
              break;
            v55 = *(_QWORD *)(v55 + 8);
          }
          v56 = src;
          v57 = *(_QWORD *)(*(_QWORD *)(i + 24) + 40LL);
          if ( !*((_BYTE *)src + 28) )
            goto LABEL_190;
          v58 = (_QWORD *)*((_QWORD *)src + 1);
          v10 = *((unsigned int *)src + 5);
          v55 = (__int64)&v58[v10];
          if ( v58 != (_QWORD *)v55 )
          {
            while ( v57 != *v58 )
            {
              if ( (_QWORD *)v55 == ++v58 )
                goto LABEL_97;
            }
LABEL_89:
            i = (__int64)*(v49 - 2);
            goto LABEL_90;
          }
LABEL_97:
          if ( (unsigned int)v10 < *((_DWORD *)src + 4) )
          {
            *((_DWORD *)src + 5) = v10 + 1;
            *(_QWORD *)v55 = v57;
            ++*v56;
          }
          else
          {
LABEL_190:
            sub_C8CC70((__int64)src, v57, v55, v10, v11, v12);
            if ( !v119 )
              goto LABEL_89;
          }
          v151.m128i_i64[0] = v57;
          LOBYTE(v152) = 0;
          sub_26D91D0((unsigned __int64 *)&v148, &v151);
          v49 = v149;
          v8 = v148;
        }
        while ( (char *)v149 - (char *)v148 != v139 );
LABEL_100:
        if ( v8 == v49 )
          break;
        v59 = v136;
        v60 = v8;
        while ( *v60 == *(void **)v59 )
        {
          v10 = *((unsigned __int8 *)v60 + 16);
          if ( (_BYTE)v10 != *(_BYTE *)(v59 + 16) || (_BYTE)v10 && v60[1] != *(void **)(v59 + 8) )
            break;
          v60 += 3;
          v59 += 24LL;
          if ( v60 == v49 )
            goto LABEL_107;
        }
      }
LABEL_107:
      if ( v136 )
      {
        j_j___libc_free_0(v136);
        v8 = v148;
      }
      if ( v8 )
        j_j___libc_free_0((unsigned __int64)v8);
      if ( v161 )
        j_j___libc_free_0((unsigned __int64)v161);
      if ( !v157 )
        goto LABEL_55;
      j_j___libc_free_0((unsigned __int64)v157);
      v41 = *(_QWORD *)(v41 + 8);
      if ( v134 == v41 )
      {
LABEL_115:
        a2 = v132;
        goto LABEL_116;
      }
    }
    v136 = 0;
LABEL_227:
    v139 = 0;
    goto LABEL_87;
  }
LABEL_116:
  LODWORD(v153) = 0;
  v151.m128i_i64[1] = 0;
  v152 = 0;
  src = 0;
  v148 = 0;
  v149 = 0;
  if ( HIDWORD(v165) != v166 )
  {
    v151.m128i_i64[0] = 1;
    v61 = (4 * (HIDWORD(v165) - v166) / 3u + 1) | ((unsigned __int64)(4 * (HIDWORD(v165) - v166) / 3u + 1) >> 1);
    v62 = (((v61 >> 2) | v61) >> 4) | (v61 >> 2) | v61;
    sub_FE19E0((__int64)&v151, ((((v62 >> 8) | v62) >> 16) | (v62 >> 8) | v62) + 1);
    v63 = src;
    v64 = (unsigned int)(HIDWORD(v165) - v166);
    if ( v64 <= ((char *)v149 - (_BYTE *)src) >> 3 )
      goto LABEL_123;
    v65 = 8 * v64;
    v66 = (char *)v148 - (_BYTE *)src;
    if ( HIDWORD(v165) == v166 )
    {
      v69 = (char *)v148 - (_BYTE *)src;
      v68 = 0;
    }
    else
    {
      v67 = sub_22077B0(8 * v64);
      v63 = src;
      v68 = (char *)v67;
      v69 = (char *)v148 - (_BYTE *)src;
    }
    if ( v69 > 0 )
    {
      memmove(v68, v63, v69);
    }
    else if ( !v63 )
    {
LABEL_122:
      src = v68;
      v148 = (void **)&v68[v66];
      v149 = (void **)&v68[v65];
      goto LABEL_123;
    }
    j_j___libc_free_0((unsigned __int64)v63);
    goto LABEL_122;
  }
  v151.m128i_i64[0] = 1;
LABEL_123:
  v70 = *(_QWORD *)(*a1 + 80);
  for ( j = *a1 + 72; j != v70; v70 = *(_QWORD *)(v70 + 8) )
  {
    v71 = v70 - 24;
    if ( !v70 )
      v71 = 0;
    if ( v167 )
    {
      v72 = v164;
      v73 = &v164[8 * HIDWORD(v165)];
      if ( v164 == v73 )
        continue;
      while ( v71 != *(_QWORD *)v72 )
      {
        v72 += 8;
        if ( v73 == v72 )
          goto LABEL_142;
      }
    }
    else if ( !sub_C8CA60((__int64)&v163, v71) )
    {
      continue;
    }
    if ( v172 )
    {
      v74 = (_QWORD *)v169.m128i_i64[1];
      v75 = v169.m128i_i64[1] + 8LL * HIDWORD(v170);
      if ( v169.m128i_i64[1] != v75 )
      {
        while ( v71 != *v74 )
        {
          if ( (_QWORD *)v75 == ++v74 )
            goto LABEL_142;
        }
LABEL_136:
        v76 = v153;
        v143 = v71;
        v137 = ((char *)v148 - (_BYTE *)src) >> 3;
        if ( (_DWORD)v153 )
        {
          v77 = 1;
          v78 = 0;
          v79 = (v153 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
          v80 = (__int64 *)(v151.m128i_i64[1] + 16LL * v79);
          v81 = *v80;
          if ( v71 == *v80 )
          {
LABEL_138:
            v80[1] = v137;
            v82 = v148;
            v156 = (__m128i *)v71;
            if ( v148 == v149 )
            {
              sub_A413F0((__int64)&src, v148, &v156);
            }
            else
            {
              if ( v148 )
              {
                *v148 = (void *)v71;
                v82 = v148;
              }
              v148 = v82 + 1;
            }
            continue;
          }
          while ( v81 != -4096 )
          {
            if ( !v78 && v81 == -8192 )
              v78 = v80;
            v79 = (v153 - 1) & (v77 + v79);
            v80 = (__int64 *)(v151.m128i_i64[1] + 16LL * v79);
            v81 = *v80;
            if ( v71 == *v80 )
              goto LABEL_138;
            ++v77;
          }
          if ( v78 )
            v80 = v78;
          ++v151.m128i_i64[0];
          v130 = (_DWORD)v152 + 1;
          v156 = (__m128i *)v80;
          if ( 4 * ((int)v152 + 1) < (unsigned int)(3 * v153) )
          {
            v131 = v71;
            if ( (int)v153 - HIDWORD(v152) - v130 > (unsigned int)v153 >> 3 )
            {
LABEL_246:
              LODWORD(v152) = v130;
              if ( *v80 != -4096 )
                --HIDWORD(v152);
              *v80 = v131;
              v80[1] = 0;
              goto LABEL_138;
            }
LABEL_251:
            sub_FE19E0((__int64)&v151, v76);
            sub_26C35D0((__int64)&v151, &v143, &v156);
            v131 = v143;
            v130 = (_DWORD)v152 + 1;
            v80 = (__int64 *)v156;
            goto LABEL_246;
          }
        }
        else
        {
          ++v151.m128i_i64[0];
          v156 = 0;
        }
        v76 = 2 * v153;
        goto LABEL_251;
      }
    }
    else if ( sub_C8CA60((__int64)&v169, v71) )
    {
      goto LABEL_136;
    }
LABEL_142:
    ;
  }
  v83 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  if ( !v83 )
  {
    if ( !*(_DWORD *)(a2 + 20) )
      goto LABEL_149;
    v84 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v84 <= 0x40 )
      goto LABEL_146;
    sub_C7D6A0(*(_QWORD *)(a2 + 8), 16 * v84, 8);
    *(_DWORD *)(a2 + 24) = 0;
LABEL_253:
    *(_QWORD *)(a2 + 8) = 0;
LABEL_148:
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_149;
  }
  v126 = 4 * v83;
  v84 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)(4 * v83) < 0x40 )
    v126 = 64;
  if ( v126 >= (unsigned int)v84 )
  {
LABEL_146:
    v85 = *(_QWORD **)(a2 + 8);
    for ( k = &v85[2 * v84]; k != v85; v85 += 2 )
      *v85 = -4096;
    goto LABEL_148;
  }
  v127 = v83 - 1;
  if ( v127 )
  {
    _BitScanReverse(&v127, v127);
    v128 = 1 << (33 - (v127 ^ 0x1F));
    if ( v128 < 64 )
      v128 = 64;
    if ( v128 == (_DWORD)v84 )
      goto LABEL_225;
  }
  else
  {
    v128 = 64;
  }
  sub_C7D6A0(*(_QWORD *)(a2 + 8), 16 * v84, 8);
  v129 = sub_26BC060(v128);
  *(_DWORD *)(a2 + 24) = v129;
  if ( !v129 )
    goto LABEL_253;
  *(_QWORD *)(a2 + 8) = sub_C7D670(16LL * v129, 8);
LABEL_225:
  sub_26C5740(a2);
LABEL_149:
  v87 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  if ( v87 )
  {
    v122 = 4 * v87;
    v88 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)(4 * v87) < 0x40 )
      v122 = 64;
    if ( (unsigned int)v88 <= v122 )
      goto LABEL_152;
    v123 = v87 - 1;
    if ( v123 )
    {
      _BitScanReverse(&v123, v123);
      v124 = 1 << (33 - (v123 ^ 0x1F));
      if ( v124 < 64 )
        v124 = 64;
      if ( v124 == (_DWORD)v88 )
      {
        sub_26C8010(a3);
        goto LABEL_155;
      }
    }
    else
    {
      v124 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(a3 + 8), 24 * v88, 8);
    v125 = sub_26BC060(v124);
    *(_DWORD *)(a3 + 24) = v125;
    if ( !v125 )
      goto LABEL_257;
    *(_QWORD *)(a3 + 8) = sub_C7D670(24LL * v125, 8);
    sub_26C8010(a3);
  }
  else if ( *(_DWORD *)(a3 + 20) )
  {
    v88 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)v88 <= 0x40 )
    {
LABEL_152:
      v89 = *(_QWORD **)(a3 + 8);
      for ( m = &v89[3 * v88]; m != v89; *(v89 - 2) = -4096 )
      {
        *v89 = -4096;
        v89 += 3;
      }
      *(_QWORD *)(a3 + 16) = 0;
      goto LABEL_155;
    }
    sub_C7D6A0(*(_QWORD *)(a3 + 8), 24 * v88, 8);
    *(_DWORD *)(a3 + 24) = 0;
LABEL_257:
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
  }
LABEL_155:
  v91 = src;
  v92 = v148;
  if ( v148 == src )
    goto LABEL_182;
  v93 = (void **)src;
  v94 = 0;
  do
  {
    while ( 1 )
    {
      v95 = a1[2];
      v96 = *v93;
      v156 = (__m128i *)*v93;
      v97 = *(unsigned int *)(v95 + 24);
      v98 = *(_QWORD *)(v95 + 8);
      if ( (_DWORD)v97 )
      {
        v99 = (v97 - 1) & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
        v100 = (void **)(v98 + 16LL * v99);
        v101 = *v100;
        if ( v96 != *v100 )
        {
          v120 = 1;
          while ( v101 != (void *)-4096LL )
          {
            v121 = v120 + 1;
            v99 = (v97 - 1) & (v120 + v99);
            v100 = (void **)(v98 + 16LL * v99);
            v101 = *v100;
            if ( v96 == *v100 )
              goto LABEL_160;
            v120 = v121;
          }
          goto LABEL_157;
        }
LABEL_160:
        if ( v100 != (void **)(v98 + 16 * v97) && v100[1] )
          break;
      }
LABEL_157:
      if ( v92 == ++v93 )
        goto LABEL_163;
    }
    v141 = (__int64)v100[1];
    ++v93;
    v102 = sub_26CC460(a2, (__int64 *)&v156);
    v94 = 1;
    *v102 = v141;
  }
  while ( v92 != v93 );
LABEL_163:
  v91 = src;
  if ( (unsigned __int64)((char *)v148 - (_BYTE *)src) > 8 && v94 )
  {
    sub_26D7FA0((unsigned __int64 *)&v156, (__int64)a1, (__int64 **)&src, (__int64)&v151);
    sub_2A60C60(&v156);
    v103 = (__int64 *)v148;
    v104 = (__int64 *)src;
    if ( v148 != src )
    {
      do
      {
        v105 = *v104++;
        v143 = v105;
        v106 = sub_26CC460((__int64)&v151, &v143);
        v107 = v156[5 * *v106 + 1].m128i_i64[1];
        *sub_26CC460(a2, &v143) = v107;
      }
      while ( v103 != v104 );
    }
    v108 = (unsigned __int64)v159;
    v109 = v160;
    v110 = v159;
    if ( v160 != v159 )
    {
      do
      {
        v111 = *v110;
        v112 = v110[1];
        v110 += 5;
        v113 = (void **)*((_QWORD *)src + v112);
        v143 = *((_QWORD *)src + v111);
        v144 = v113;
        v114 = *(v110 - 1);
        *sub_26CC870(a3, &v143) = v114;
      }
      while ( v109 != v110 );
      v108 = (unsigned __int64)v159;
    }
    if ( v108 )
      j_j___libc_free_0(v108);
    v115 = (unsigned __int64)v157;
    v116 = (unsigned __int64)v156;
    if ( v157 != v156 )
    {
      do
      {
        v117 = *(_QWORD *)(v116 + 56);
        if ( v117 )
          j_j___libc_free_0(v117);
        v118 = *(_QWORD *)(v116 + 32);
        if ( v118 )
          j_j___libc_free_0(v118);
        v116 += 80LL;
      }
      while ( v115 != v116 );
      v116 = (unsigned __int64)v156;
    }
    if ( v116 )
      j_j___libc_free_0(v116);
    v91 = src;
  }
LABEL_182:
  if ( v91 )
    j_j___libc_free_0((unsigned __int64)v91);
  sub_C7D6A0(v151.m128i_i64[1], 16LL * (unsigned int)v153, 8);
  if ( !v172 )
    _libc_free(v169.m128i_u64[1]);
  if ( !v167 )
    _libc_free((unsigned __int64)v164);
}
