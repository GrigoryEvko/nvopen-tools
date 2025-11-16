// Function: sub_C9B590
// Address: 0xc9b590
//
int __fastcall sub_C9B590(__int64 a1, __int64 a2)
{
  __int128 *v2; // rax
  unsigned int v3; // edi
  _QWORD *v4; // r15
  __m128i *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  unsigned __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // r9
  unsigned __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  int v25; // ecx
  int v26; // edx
  _QWORD *v27; // rcx
  __int64 v28; // rax
  _QWORD *j; // r15
  _QWORD *k; // rbx
  size_t v31; // r13
  const void *v32; // r12
  __int64 v33; // r15
  int v34; // eax
  __int64 *v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // rax
  unsigned int v39; // r9d
  __int64 **v40; // rcx
  _QWORD *v41; // r10
  int v42; // eax
  __int64 *v43; // rdx
  __int64 v44; // rax
  const __m128i *v45; // r12
  const __m128i *v46; // rdi
  const __m128i *v47; // rbx
  const __m128i *v48; // r13
  __m128i *v49; // r14
  __m128i v50; // xmm1
  const __m128i *v51; // rdi
  const __m128i *v52; // rax
  __int64 v53; // rcx
  __int64 *v54; // rax
  __int64 **v55; // r14
  __int64 **v56; // rbx
  __int64 **v57; // r15
  __int64 *v58; // r14
  __int64 v59; // rdx
  _BYTE *v60; // rsi
  __m128i *v61; // rdi
  __int64 *v62; // rdx
  __int64 **v63; // rax
  __m128i *v64; // r12
  __m128i *v65; // r13
  __int64 v66; // rbx
  unsigned __int64 v67; // rax
  __m128i *v68; // rbx
  __m128i *v69; // rdi
  const __m128i *v70; // rbx
  __int64 v71; // rax
  __m128i *v72; // r12
  size_t v73; // r13
  int v74; // eax
  unsigned int v75; // r10d
  __int64 *v76; // rdx
  __int64 v77; // rax
  unsigned int v78; // r10d
  __int64 **v79; // rcx
  _QWORD *v80; // r11
  int v81; // eax
  __int64 *v82; // rax
  __int64 *v83; // rax
  const __m128i *v84; // rbx
  const __m128i *v85; // r12
  __int64 **v86; // r8
  __int64 v87; // r12
  unsigned __int64 v88; // rbx
  __int64 *v89; // rdi
  int result; // eax
  _QWORD *v91; // rdx
  _QWORD *v92; // rbx
  __int64 v93; // rax
  _QWORD *i; // r15
  size_t v95; // rbx
  __m128i *v96; // r12
  __int64 v97; // r13
  int v98; // eax
  __int64 *v99; // rax
  __int64 v100; // rdx
  _QWORD *v101; // rax
  __int64 v102; // rax
  unsigned int v103; // r9d
  __int64 **v104; // rcx
  _QWORD *v105; // r10
  int v106; // eax
  __int64 *v107; // rdx
  _QWORD *v108; // [rsp+0h] [rbp-1D0h]
  unsigned __int64 v109; // [rsp+10h] [rbp-1C0h]
  _QWORD *v110; // [rsp+10h] [rbp-1C0h]
  pthread_mutex_t *mutex; // [rsp+18h] [rbp-1B8h]
  __int64 v112; // [rsp+20h] [rbp-1B0h]
  __int64 v113; // [rsp+20h] [rbp-1B0h]
  __int64 align; // [rsp+30h] [rbp-1A0h]
  _QWORD *v116; // [rsp+30h] [rbp-1A0h]
  _QWORD *v117; // [rsp+30h] [rbp-1A0h]
  __int64 v118; // [rsp+38h] [rbp-198h]
  __int64 v119; // [rsp+38h] [rbp-198h]
  _QWORD *v120; // [rsp+38h] [rbp-198h]
  __int64 v121; // [rsp+40h] [rbp-190h]
  __m128i v122; // [rsp+40h] [rbp-190h]
  _QWORD *v123; // [rsp+40h] [rbp-190h]
  _QWORD *v124; // [rsp+40h] [rbp-190h]
  const __m128i *v125; // [rsp+40h] [rbp-190h]
  _QWORD *v126; // [rsp+40h] [rbp-190h]
  __int64 **v127; // [rsp+48h] [rbp-188h]
  __int64 **v128; // [rsp+48h] [rbp-188h]
  __int64 **v129; // [rsp+48h] [rbp-188h]
  unsigned int v130; // [rsp+50h] [rbp-180h]
  unsigned int v131; // [rsp+50h] [rbp-180h]
  unsigned int v132; // [rsp+50h] [rbp-180h]
  __m128i *v133; // [rsp+58h] [rbp-178h]
  __int64 v134; // [rsp+58h] [rbp-178h]
  __int64 v135; // [rsp+58h] [rbp-178h]
  unsigned __int64 v136; // [rsp+68h] [rbp-168h] BYREF
  __int64 v137; // [rsp+70h] [rbp-160h] BYREF
  __int64 v138; // [rsp+78h] [rbp-158h] BYREF
  __int64 **v139; // [rsp+80h] [rbp-150h] BYREF
  __int64 v140; // [rsp+88h] [rbp-148h]
  __int64 v141; // [rsp+90h] [rbp-140h]
  const __m128i *v142; // [rsp+A0h] [rbp-130h] BYREF
  __m128i *v143; // [rsp+A8h] [rbp-128h]
  const __m128i *v144; // [rsp+B0h] [rbp-120h]
  __m128i v145; // [rsp+C0h] [rbp-110h] BYREF
  __m128i v146; // [rsp+D0h] [rbp-100h] BYREF
  const __m128i **v147; // [rsp+E0h] [rbp-F0h]
  const __m128i **v148; // [rsp+E8h] [rbp-E8h]
  _QWORD v149[2]; // [rsp+F0h] [rbp-E0h] BYREF
  int v150; // [rsp+100h] [rbp-D0h] BYREF
  char v151; // [rsp+104h] [rbp-CCh]
  __int64 v152; // [rsp+180h] [rbp-50h]
  __int64 v153; // [rsp+188h] [rbp-48h]
  __int64 v154; // [rsp+190h] [rbp-40h]
  __int64 v155; // [rsp+198h] [rbp-38h]

  v2 = sub_C95C80();
  mutex = (pthread_mutex_t *)v2;
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock((pthread_mutex_t *)v2);
    if ( v3 )
      sub_4264C5(v3);
  }
  v4 = v149;
  v154 = a2;
  v149[0] = &v150;
  v149[1] = 0x1000000001LL;
  v152 = 0;
  v153 = 0;
  v155 = 0;
  v150 = 0;
  v151 = 0;
  sub_C6ACB0((__int64)v149);
  v5 = (__m128i *)"traceEvents";
  sub_C6B410((__int64)v149, "traceEvents", 0xBu);
  sub_C6AB50((__int64)v149);
  v8 = *(_QWORD *)(a1 + 144);
  v9 = (unsigned __int64)*(unsigned int *)(a1 + 152) << 7;
  v121 = v8 + v9;
  if ( v8 != v8 + v9 )
  {
    v10 = *(_QWORD *)(a1 + 144);
    do
    {
      while ( 1 )
      {
        v138 = *(_QWORD *)(a1 + 16648);
        v11 = *(_QWORD *)v10 / 1000LL;
        v139 = (__int64 **)(v11 - *(_QWORD *)(a1 + 16576) / 1000LL);
        v12 = *(_QWORD *)(v10 + 8);
        v145.m128i_i64[0] = (__int64)v149;
        v145.m128i_i64[1] = a1;
        v147 = (const __m128i **)v10;
        v146.m128i_i64[0] = (__int64)&v138;
        v146.m128i_i64[1] = (__int64)&v139;
        v142 = (const __m128i *)(v12 / 1000 - v11);
        v148 = &v142;
        sub_C6ACB0((__int64)v149);
        sub_C97560((__int64)&v145);
        sub_C6AD90((__int64)v149);
        if ( *(_DWORD *)(v10 + 120) == 2 )
          break;
        v10 += 128;
        if ( v121 == v10 )
          goto LABEL_8;
      }
      v148 = (const __m128i **)v10;
      v10 += 128;
      v145.m128i_i64[0] = (__int64)v149;
      v146.m128i_i64[0] = (__int64)&v138;
      v145.m128i_i64[1] = a1;
      v146.m128i_i64[1] = (__int64)&v139;
      v147 = &v142;
      sub_C6ACB0((__int64)v149);
      sub_C984A0((__int64)&v145);
      sub_C6AD90((__int64)v149);
    }
    while ( v121 != v10 );
LABEL_8:
    v4 = v149;
  }
  v13 = a1;
  v112 = *(&mutex[1].__align + 1);
  align = mutex[1].__align;
  if ( align == v112 )
  {
    v109 = *(_QWORD *)(a1 + 16648);
  }
  else
  {
    do
    {
      v14 = *(_QWORD *)(*(_QWORD *)align + 144LL);
      v118 = v14 + ((unsigned __int64)*(unsigned int *)(*(_QWORD *)align + 152LL) << 7);
      if ( v14 != v118 )
      {
        v15 = v13;
        v122.m128i_i64[0] = (__int64)&v138;
        v122.m128i_i64[1] = (__int64)&v139;
        v16 = v4;
        v17 = *(_QWORD *)align;
        v18 = *(_QWORD *)(*(_QWORD *)align + 144LL);
        v19 = (__int64)v16;
        do
        {
          while ( 1 )
          {
            v138 = *(_QWORD *)(v17 + 16648);
            v20 = *(_QWORD *)v18 / 1000LL;
            v139 = (__int64 **)(v20 - *(_QWORD *)(v15 + 16576) / 1000LL);
            v21 = *(_QWORD *)(v18 + 8);
            v145.m128i_i64[0] = v19;
            v145.m128i_i64[1] = v15;
            v147 = (const __m128i **)v18;
            v146 = v122;
            v142 = (const __m128i *)(v21 / 1000 - v20);
            v148 = &v142;
            sub_C6ACB0(v19);
            sub_C97560((__int64)&v145);
            sub_C6AD90(v19);
            if ( *(_DWORD *)(v18 + 120) == 2 )
              break;
            v18 += 128;
            if ( v118 == v18 )
              goto LABEL_15;
          }
          v148 = (const __m128i **)v18;
          v18 += 128;
          v145.m128i_i64[0] = v19;
          v146 = v122;
          v145.m128i_i64[1] = v15;
          v147 = &v142;
          sub_C6ACB0(v19);
          sub_C984A0((__int64)&v145);
          sub_C6AD90(v19);
        }
        while ( v118 != v18 );
LABEL_15:
        v4 = (_QWORD *)v19;
        v13 = v15;
      }
      align += 8;
    }
    while ( v112 != align );
    v5 = (__m128i *)mutex;
    v22 = *(_QWORD *)(a1 + 16648);
    v23 = *(&mutex[1].__align + 1);
    v24 = mutex[1].__align;
    v109 = v22;
    if ( v23 != v24 )
    {
      do
      {
        v5 = *(__m128i **)(*(_QWORD *)v24 + 16648LL);
        if ( v22 < (unsigned __int64)v5 )
          v22 = *(_QWORD *)(*(_QWORD *)v24 + 16648LL);
        v24 += 8;
      }
      while ( v23 != v24 );
      v109 = v22;
    }
  }
  v139 = 0;
  v141 = 0x1800000000LL;
  v140 = 0;
  v25 = *(_DWORD *)(a1 + 16552);
  if ( v25 )
  {
    v91 = *(_QWORD **)(a1 + 16544);
    v92 = v91;
    if ( *v91 == -8 || !*v91 )
    {
      do
      {
        do
        {
          v93 = v92[1];
          ++v92;
        }
        while ( !v93 );
      }
      while ( v93 == -8 );
    }
    v126 = &v91[v25];
    if ( v92 != v126 )
    {
      v117 = v4;
      for ( i = v92; ; i = v101 )
      {
        v95 = *(_QWORD *)*i;
        v96 = (__m128i *)(*i + 24LL);
        v97 = *(_QWORD *)(*i + 8LL);
        v135 = *(_QWORD *)(*i + 16LL);
        v98 = sub_C92610();
        v5 = v96;
        v7 = (unsigned int)sub_C92740((__int64)&v139, v96, v95, v98);
        v99 = v139[v7];
        if ( v99 )
        {
          if ( v99 != (__int64 *)-8LL )
            goto LABEL_128;
          LODWORD(v141) = v141 - 1;
        }
        v129 = &v139[v7];
        v132 = v7;
        v102 = sub_C7D670(v95 + 25, 8);
        v103 = v132;
        v104 = v129;
        v105 = (_QWORD *)v102;
        if ( v95 )
        {
          v120 = (_QWORD *)v102;
          memcpy((void *)(v102 + 24), v96, v95);
          v103 = v132;
          v104 = v129;
          v105 = v120;
        }
        *((_BYTE *)v105 + v95 + 24) = 0;
        v5 = (__m128i *)v103;
        *v105 = v95;
        v105[1] = 0;
        v105[2] = 0;
        *v104 = v105;
        ++HIDWORD(v140);
        v106 = sub_C929D0((__int64 *)&v139, v103);
        v107 = (__int64 *)&v139[v106];
        v99 = (__int64 *)*v107;
        if ( !*v107 || v99 == (__int64 *)-8LL )
        {
          do
          {
            do
            {
              v99 = (__int64 *)v107[1];
              ++v107;
            }
            while ( v99 == (__int64 *)-8LL );
          }
          while ( !v99 );
        }
LABEL_128:
        v99[1] += v97;
        v99[2] += v135;
        v100 = i[1];
        v101 = i + 1;
        if ( v100 != -8 )
          goto LABEL_130;
        do
        {
          do
          {
            v100 = v101[1];
            ++v101;
          }
          while ( v100 == -8 );
LABEL_130:
          ;
        }
        while ( !v100 );
        if ( v101 == v126 )
        {
          v4 = v117;
          break;
        }
      }
    }
  }
  v113 = *(&mutex[1].__align + 1);
  v119 = mutex[1].__align;
  if ( v113 == v119 )
    goto LABEL_47;
  v108 = v4;
  do
  {
    v26 = *(_DWORD *)(*(_QWORD *)v119 + 16552LL);
    if ( !v26 )
      goto LABEL_45;
    v27 = *(_QWORD **)(*(_QWORD *)v119 + 16544LL);
    v28 = *v27;
    for ( j = v27; !v28; ++j )
LABEL_27:
      v28 = j[1];
    if ( v28 == -8 )
      goto LABEL_27;
    v123 = &v27[v26];
    if ( j != v123 )
    {
      for ( k = j; ; k = v37 )
      {
        v31 = *(_QWORD *)*k;
        v32 = (const void *)(*k + 24LL);
        v33 = *(_QWORD *)(*k + 16LL);
        v133 = *(__m128i **)(*k + 8LL);
        v34 = sub_C92610();
        v7 = (unsigned int)sub_C92740((__int64)&v139, v32, v31, v34);
        v35 = v139[v7];
        if ( v35 )
        {
          if ( v35 != (__int64 *)-8LL )
            goto LABEL_33;
          LODWORD(v141) = v141 - 1;
        }
        v127 = &v139[v7];
        v130 = v7;
        v38 = sub_C7D670(v31 + 25, 8);
        v39 = v130;
        v40 = v127;
        v41 = (_QWORD *)v38;
        if ( v31 )
        {
          v116 = (_QWORD *)v38;
          memcpy((void *)(v38 + 24), v32, v31);
          v39 = v130;
          v40 = v127;
          v41 = v116;
        }
        *((_BYTE *)v41 + v31 + 24) = 0;
        *v41 = v31;
        v41[1] = 0;
        v41[2] = 0;
        *v40 = v41;
        ++HIDWORD(v140);
        v42 = sub_C929D0((__int64 *)&v139, v39);
        v43 = (__int64 *)&v139[v42];
        v35 = (__int64 *)*v43;
        if ( !*v43 )
          break;
LABEL_43:
        if ( v35 == (__int64 *)-8LL )
          goto LABEL_42;
LABEL_33:
        v35[2] += v33;
        v5 = v133;
        v35[1] += (__int64)v133;
        v36 = k[1];
        v37 = k + 1;
        if ( v36 )
          goto LABEL_35;
        do
        {
          do
          {
            v36 = v37[1];
            ++v37;
          }
          while ( !v36 );
LABEL_35:
          ;
        }
        while ( v36 == -8 );
        if ( v37 == v123 )
          goto LABEL_45;
      }
      do
      {
LABEL_42:
        v35 = (__int64 *)v43[1];
        ++v43;
      }
      while ( !v35 );
      goto LABEL_43;
    }
LABEL_45:
    v119 += 8;
  }
  while ( v113 != v119 );
  v4 = v108;
LABEL_47:
  v142 = 0;
  v143 = 0;
  v144 = 0;
  if ( HIDWORD(v140) )
  {
    v134 = 3LL * HIDWORD(v140);
    v44 = sub_22077B0(v134 * 16);
    v45 = v143;
    v46 = v142;
    v47 = (const __m128i *)v44;
    if ( v143 != v142 )
    {
      v48 = v142 + 1;
      v49 = (__m128i *)v44;
      while ( 1 )
      {
        if ( v49 )
        {
          v49->m128i_i64[0] = (__int64)v49[1].m128i_i64;
          v52 = (const __m128i *)v48[-1].m128i_i64[0];
          if ( v48 == v52 )
          {
            v49[1] = _mm_loadu_si128(v48);
          }
          else
          {
            v49->m128i_i64[0] = (__int64)v52;
            v49[1].m128i_i64[0] = v48->m128i_i64[0];
          }
          v49->m128i_i64[1] = v48[-1].m128i_i64[1];
          v50 = _mm_loadu_si128(v48 + 1);
          v48[-1].m128i_i64[0] = (__int64)v48;
          v48[-1].m128i_i64[1] = 0;
          v48->m128i_i8[0] = 0;
          v49[2] = v50;
        }
        v51 = (const __m128i *)v48[-1].m128i_i64[0];
        if ( v48 != v51 )
        {
          v5 = (__m128i *)(v48->m128i_i64[0] + 1);
          j_j___libc_free_0(v51, v5);
        }
        v49 += 3;
        if ( v45 == &v48[2] )
          break;
        v48 += 3;
      }
      v46 = v142;
    }
    if ( v46 )
    {
      v5 = (__m128i *)((char *)v144 - (char *)v46);
      j_j___libc_free_0(v46, (char *)v144 - (char *)v46);
    }
    v142 = v47;
    v143 = (__m128i *)v47;
    v144 = &v47[v134];
  }
  v53 = (unsigned int)v140;
  if ( (_DWORD)v140 )
  {
    v54 = *v139;
    v55 = v139;
    if ( *v139 != (__int64 *)-8LL )
      goto LABEL_66;
    do
    {
      do
      {
        v54 = v55[1];
        ++v55;
      }
      while ( v54 == (__int64 *)-8LL );
LABEL_66:
      ;
    }
    while ( !v54 );
    v53 = (unsigned int)v140;
    v56 = &v139[(unsigned int)v140];
    if ( v55 == v56 )
      goto LABEL_83;
    v124 = v4;
    v57 = v55;
LABEL_69:
    v58 = *v57;
    v59 = **v57;
    v60 = *v57 + 3;
    v145.m128i_i64[0] = (__int64)&v146;
    sub_C95DE0(v145.m128i_i64, v60, (__int64)&v60[v59]);
    v5 = v143;
    if ( v143 == v144 )
    {
      sub_C99F30((__int64 *)&v142, v143, &v145, (const __m128i *)(v58 + 1));
      v61 = (__m128i *)v145.m128i_i64[0];
    }
    else
    {
      v61 = (__m128i *)v145.m128i_i64[0];
      if ( v143 )
      {
        v143->m128i_i64[0] = (__int64)v143[1].m128i_i64;
        if ( (__m128i *)v145.m128i_i64[0] == &v146 )
        {
          v5[1] = _mm_load_si128(&v146);
        }
        else
        {
          v5->m128i_i64[0] = v145.m128i_i64[0];
          v5[1].m128i_i64[0] = v146.m128i_i64[0];
        }
        v61 = &v146;
        v5->m128i_i64[1] = v145.m128i_i64[1];
        v145.m128i_i64[0] = (__int64)&v146;
        v145.m128i_i64[1] = 0;
        v146.m128i_i8[0] = 0;
        v5[2] = _mm_loadu_si128((const __m128i *)(v58 + 1));
        v5 = v143;
      }
      v5 += 3;
      v143 = v5;
    }
    if ( v61 != &v146 )
    {
      v5 = (__m128i *)(v146.m128i_i64[0] + 1);
      j_j___libc_free_0(v61, v146.m128i_i64[0] + 1);
    }
    v62 = v57[1];
    v63 = v57 + 1;
    if ( v62 != (__int64 *)-8LL )
      goto LABEL_79;
    while ( 1 )
    {
      do
      {
        v62 = v63[1];
        ++v63;
      }
      while ( v62 == (__int64 *)-8LL );
LABEL_79:
      if ( v62 )
      {
        if ( v63 == v56 )
        {
          v4 = v124;
          break;
        }
        v57 = v63;
        goto LABEL_69;
      }
    }
  }
LABEL_83:
  v64 = v143;
  v65 = (__m128i *)v142;
  if ( v143 == v142 )
  {
LABEL_87:
    if ( *(_BYTE *)(a1 + 16656) )
      goto LABEL_88;
  }
  else
  {
    v66 = (char *)v143 - (char *)v142;
    _BitScanReverse64(&v67, 0xAAAAAAAAAAAAAAABLL * (v143 - v142));
    sub_C9B170((__int64)v142, (__int64)v143, (char *)(2LL * (int)(63 - (v67 ^ 0x3F))), v53, v6, v7, (char)v108);
    if ( v66 > 768 )
    {
      v68 = v65 + 48;
      v5 = v65 + 48;
      sub_C96260((__int64)v65, v65 + 48);
      if ( v64 != &v65[48] )
      {
        do
        {
          v69 = v68;
          v68 += 3;
          sub_C96010(v69);
        }
        while ( v64 != v68 );
      }
      goto LABEL_87;
    }
    v5 = v64;
    sub_C96260((__int64)v65, v64);
    if ( *(_BYTE *)(a1 + 16656) )
    {
LABEL_88:
      v136 = v109 + 1;
      v125 = v143;
      if ( v143 != v142 )
      {
        v70 = v142;
        while ( 1 )
        {
          v137 = v70[2].m128i_i64[1] / 1000;
          v72 = (__m128i *)v70->m128i_i64[0];
          v73 = v70->m128i_u64[1];
          v74 = sub_C92610();
          v5 = v72;
          v75 = sub_C92740((__int64)&v139, v72, v73, v74);
          v76 = v139[v75];
          if ( !v76 )
            goto LABEL_94;
          if ( v76 == (__int64 *)-8LL )
            break;
LABEL_90:
          v71 = v76[1];
          v147 = (const __m128i **)v70;
          v70 += 3;
          v145.m128i_i64[0] = (__int64)v4;
          v138 = v71;
          v145.m128i_i64[1] = a1;
          v146.m128i_i64[0] = (__int64)&v136;
          v146.m128i_i64[1] = (__int64)&v137;
          v148 = (const __m128i **)&v138;
          sub_C6ACB0((__int64)v4);
          sub_C99820((__int64)&v145);
          sub_C6AD90((__int64)v4);
          ++v136;
          if ( v125 == v70 )
            goto LABEL_102;
        }
        LODWORD(v141) = v141 - 1;
LABEL_94:
        v128 = &v139[v75];
        v131 = v75;
        v77 = sub_C7D670(v73 + 25, 8);
        v78 = v131;
        v79 = v128;
        v80 = (_QWORD *)v77;
        if ( v73 )
        {
          v110 = (_QWORD *)v77;
          memcpy((void *)(v77 + 24), v72, v73);
          v78 = v131;
          v79 = v128;
          v80 = v110;
        }
        *((_BYTE *)v80 + v73 + 24) = 0;
        v5 = (__m128i *)v78;
        *v80 = v73;
        v80[1] = 0;
        v80[2] = 0;
        *v79 = v80;
        ++HIDWORD(v140);
        v81 = sub_C929D0((__int64 *)&v139, v78);
        v82 = (__int64 *)&v139[v81];
        v76 = (__int64 *)*v82;
        if ( !*v82 || v76 == (__int64 *)-8LL )
        {
          v83 = v82 + 1;
          do
          {
            do
              v76 = (__int64 *)*v83++;
            while ( v76 == (__int64 *)-8LL );
          }
          while ( !v76 );
        }
        goto LABEL_90;
      }
    }
  }
LABEL_102:
  sub_C6AC30((__int64)v4);
  sub_C6AE10((__int64)v4);
  sub_C6AD90((__int64)v4);
  v84 = v143;
  v85 = v142;
  if ( v143 != v142 )
  {
    do
    {
      if ( (const __m128i *)v85->m128i_i64[0] != &v85[1] )
      {
        v5 = (__m128i *)(v85[1].m128i_i64[0] + 1);
        j_j___libc_free_0(v85->m128i_i64[0], v5);
      }
      v85 += 3;
    }
    while ( v84 != v85 );
    v85 = v142;
  }
  if ( v85 )
  {
    v5 = (__m128i *)((char *)v144 - (char *)v85);
    j_j___libc_free_0(v85, (char *)v144 - (char *)v85);
  }
  v86 = v139;
  if ( HIDWORD(v140) && (_DWORD)v140 )
  {
    v87 = 8LL * (unsigned int)v140;
    v88 = 0;
    do
    {
      v89 = v86[v88 / 8];
      if ( v89 != (_QWORD *)-8LL && v89 )
      {
        v5 = (__m128i *)(*v89 + 25LL);
        sub_C7D6A0((__int64)v89, (__int64)v5, 8);
        v86 = v139;
      }
      v88 += 8LL;
    }
    while ( v87 != v88 );
  }
  result = _libc_free(v86, v5);
  if ( (int *)v149[0] != &v150 )
    result = _libc_free(v149[0], v5);
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(mutex);
  return result;
}
