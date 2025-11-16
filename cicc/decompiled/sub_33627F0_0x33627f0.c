// Function: sub_33627F0
// Address: 0x33627f0
//
__int64 __fastcall sub_33627F0(__int64 a1, __int64 **a2)
{
  __int64 v2; // rbx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 *v6; // rax
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rsi
  int v14; // eax
  unsigned int *v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  int v18; // edx
  unsigned int *v19; // rdx
  __int64 *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r8
  __int64 v27; // r8
  int v28; // eax
  bool v29; // al
  bool v30; // zf
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // rsi
  unsigned int v40; // ecx
  __int64 *v41; // rax
  __int64 v42; // r13
  int v43; // eax
  __int64 v44; // r12
  __int64 v45; // r14
  unsigned __int64 v46; // rax
  unsigned __int64 *v47; // rbx
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rdi
  int v53; // eax
  int v54; // r9d
  int v55; // eax
  int v56; // r10d
  _QWORD *v57; // r12
  __int64 v58; // rax
  _QWORD *v59; // r13
  __int64 v60; // rdx
  unsigned __int64 v61; // rsi
  __int64 v62; // rax
  __m128i *v63; // r13
  _QWORD *v64; // r12
  __int64 v65; // rax
  unsigned int *v66; // r9
  __int64 v67; // rbx
  unsigned int *v68; // r14
  __int64 v69; // r15
  __int64 v70; // rax
  int *v71; // r8
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rax
  __m128i v75; // xmm0
  __int64 v76; // rax
  unsigned __int64 v77; // r8
  __int64 v78; // rax
  char *v79; // r13
  __int64 v80; // rax
  char *v81; // r9
  __int64 v82; // rbx
  char *v83; // r14
  char *v84; // rax
  char *v85; // rcx
  unsigned __int64 v86; // r8
  unsigned __int64 v87; // rcx
  __int64 v88; // rax
  _QWORD *v89; // r13
  __int64 v90; // rsi
  _QWORD *v91; // r15
  unsigned int v92; // eax
  unsigned int v93; // r14d
  unsigned int v94; // ebx
  __m128i *v95; // rdx
  unsigned __int64 *v96; // r13
  __int64 v97; // rdx
  unsigned __int64 v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rdx
  unsigned __int64 v101; // rsi
  _QWORD *v102; // r15
  _QWORD *v103; // rbx
  __int64 v104; // rax
  __int64 v105; // r8
  __int64 v106; // rdx
  unsigned __int64 v107; // r9
  __int64 v108; // r13
  unsigned __int64 v109; // rdx
  unsigned __int64 v110; // r8
  __int64 *v111; // r13
  __int64 *v112; // r12
  __int64 *v113; // rbx
  __int64 *v114; // r14
  __int64 v115; // r15
  __int64 v116; // rdx
  __int64 v117; // rax
  __int64 v118; // rcx
  __int64 v119; // rax
  _QWORD *v120; // r13
  unsigned int v121; // r14d
  unsigned __int64 *v122; // rbx
  __int64 v123; // rcx
  unsigned __int64 v124; // rsi
  unsigned int v125; // eax
  __int64 v126; // rax
  __int64 v127; // rcx
  unsigned __int64 v128; // rsi
  __int64 v129; // [rsp+20h] [rbp-630h]
  __int64 *v130; // [rsp+28h] [rbp-628h]
  int v132; // [rsp+38h] [rbp-618h]
  __m128i *v133; // [rsp+38h] [rbp-618h]
  __int64 v134; // [rsp+40h] [rbp-610h]
  __int64 v135; // [rsp+40h] [rbp-610h]
  __int64 *v136; // [rsp+50h] [rbp-600h]
  unsigned __int64 v137; // [rsp+50h] [rbp-600h]
  unsigned __int64 v138; // [rsp+50h] [rbp-600h]
  unsigned __int64 v139; // [rsp+50h] [rbp-600h]
  unsigned __int64 v140; // [rsp+50h] [rbp-600h]
  int v141; // [rsp+58h] [rbp-5F8h]
  unsigned __int64 v142; // [rsp+58h] [rbp-5F8h]
  __int64 v143; // [rsp+58h] [rbp-5F8h]
  char v144; // [rsp+60h] [rbp-5F0h]
  unsigned __int64 v145; // [rsp+60h] [rbp-5F0h]
  __int64 v146; // [rsp+60h] [rbp-5F0h]
  __int64 v147; // [rsp+60h] [rbp-5F0h]
  __int64 v148; // [rsp+60h] [rbp-5F0h]
  __int64 v149; // [rsp+60h] [rbp-5F0h]
  _QWORD *i; // [rsp+68h] [rbp-5E8h]
  int *v151; // [rsp+68h] [rbp-5E8h]
  char *v152; // [rsp+68h] [rbp-5E8h]
  _QWORD *v153; // [rsp+68h] [rbp-5E8h]
  __int64 v154; // [rsp+68h] [rbp-5E8h]
  _QWORD *v155; // [rsp+68h] [rbp-5E8h]
  unsigned int v156; // [rsp+68h] [rbp-5E8h]
  __int64 v157; // [rsp+70h] [rbp-5E0h] BYREF
  _BYTE *v158; // [rsp+78h] [rbp-5D8h]
  _BYTE v159[40]; // [rsp+80h] [rbp-5D0h] BYREF
  __int64 v160; // [rsp+A8h] [rbp-5A8h]
  __int64 *v161; // [rsp+B0h] [rbp-5A0h]
  __int64 *v162; // [rsp+C0h] [rbp-590h] BYREF
  __int64 v163; // [rsp+C8h] [rbp-588h]
  _BYTE v164[64]; // [rsp+D0h] [rbp-580h] BYREF
  unsigned __int64 v165[2]; // [rsp+110h] [rbp-540h] BYREF
  _BYTE v166[40]; // [rsp+120h] [rbp-530h] BYREF
  int v167; // [rsp+148h] [rbp-508h] BYREF
  unsigned __int64 v168; // [rsp+150h] [rbp-500h]
  int *v169; // [rsp+158h] [rbp-4F8h]
  int *v170; // [rsp+160h] [rbp-4F0h]
  __int64 v171; // [rsp+168h] [rbp-4E8h]
  __int64 v172; // [rsp+170h] [rbp-4E0h] BYREF
  __int64 v173; // [rsp+178h] [rbp-4D8h]
  __int64 v174; // [rsp+180h] [rbp-4D0h] BYREF
  unsigned int v175; // [rsp+188h] [rbp-4C8h]
  __int64 v176; // [rsp+280h] [rbp-3D0h] BYREF
  __int64 v177; // [rsp+288h] [rbp-3C8h]
  __int64 v178; // [rsp+290h] [rbp-3C0h] BYREF
  unsigned int v179; // [rsp+298h] [rbp-3B8h]
  __m128i *v180; // [rsp+410h] [rbp-240h] BYREF
  __int64 v181; // [rsp+418h] [rbp-238h]
  _BYTE v182[560]; // [rsp+420h] [rbp-230h] BYREF

  v2 = a1;
  sub_3754A80(v159, **(_QWORD **)(a1 + 592), *(_QWORD *)(a1 + 584), *a2);
  v176 = 0;
  v6 = (unsigned __int64 *)&v178;
  v177 = 1;
  do
  {
    *v6 = 0;
    v6 += 3;
    *((_DWORD *)v6 - 4) = -1;
  }
  while ( v6 != (unsigned __int64 *)&v180 );
  v7 = &v174;
  v172 = 0;
  v173 = 1;
  do
  {
    *v7 = -4096;
    v7 += 2;
  }
  while ( v7 != &v176 );
  v167 = 0;
  v180 = (__m128i *)v182;
  v181 = 0x2000000000LL;
  v165[0] = (unsigned __int64)v166;
  v165[1] = 0x800000000LL;
  v169 = &v167;
  v170 = &v167;
  v8 = *(_QWORD *)(a1 + 592);
  v168 = 0;
  v9 = *(_QWORD *)(v8 + 720);
  v171 = 0;
  if ( *(_DWORD *)(v9 + 104) || *(_DWORD *)(v9 + 376) )
  {
    v157 = a1;
    v158 = v159;
    v10 = *(_QWORD *)(a1 + 584);
    if ( v10 == *(_QWORD *)(*(_QWORD *)(v10 + 32) + 328LL) )
    {
      v57 = *(_QWORD **)(v9 + 368);
      for ( i = &v57[*(unsigned int *)(v9 + 376)]; v57 != i; ++v57 )
      {
        v58 = sub_37547E0(v159, *v57, &v176, v3);
        if ( v58 )
        {
          v145 = v58;
          v59 = *a2;
          sub_2E31040((__int64 *)(*(_QWORD *)(v2 + 584) + 40LL), v58);
          v60 = *(_QWORD *)v145;
          v61 = *v59 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v145 + 8) = v59;
          *(_QWORD *)v145 = v61 | v60 & 7;
          *(_QWORD *)(v61 + 8) = v145;
          *v59 = *v59 & 7LL | v145;
          *(_BYTE *)(*v57 + 63LL) = 0;
        }
      }
LABEL_92:
      v11 = *(__int64 **)(v2 + 608);
      v144 = 1;
      v130 = *(__int64 **)(v2 + 616);
      if ( v11 == v130 )
        goto LABEL_93;
      goto LABEL_8;
    }
LABEL_7:
    v11 = *(__int64 **)(v2 + 608);
    v144 = 1;
    v130 = *(__int64 **)(v2 + 616);
    if ( v130 == v11 )
      goto LABEL_94;
    goto LABEL_8;
  }
  v5 = *(unsigned int *)(v9 + 648);
  if ( (_DWORD)v5 )
  {
    v10 = *(_QWORD *)(a1 + 584);
    v157 = v2;
    v158 = v159;
    if ( v10 == *(_QWORD *)(*(_QWORD *)(v10 + 32) + 328LL) )
      goto LABEL_92;
    goto LABEL_7;
  }
  v20 = *(__int64 **)(a1 + 616);
  v157 = v2;
  v144 = 0;
  v158 = v159;
  v11 = *(__int64 **)(v2 + 608);
  v130 = v20;
  if ( v20 == v11 )
    goto LABEL_52;
LABEL_8:
  v136 = v11;
  do
  {
    v12 = (__int64 *)*v136;
    if ( !*v136 )
    {
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *))(**(_QWORD **)(v2 + 16) + 888LL))(
        *(_QWORD *)(v2 + 16),
        v160,
        *a2);
      goto LABEL_50;
    }
    if ( !*v12 )
    {
      sub_3361C20((_QWORD *)v2, *v136, (__int64)&v172, *a2, v4, v5);
      goto LABEL_50;
    }
    v162 = (__int64 *)v164;
    v163 = 0x400000000LL;
    v13 = *v12;
    v14 = *(_DWORD *)(*v12 + 64);
    if ( v14 )
    {
      v15 = (unsigned int *)(*(_QWORD *)(v13 + 40) + 40LL * (unsigned int)(v14 - 1));
      v16 = *(_QWORD *)v15;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v15 + 48LL) + 16LL * v15[2]) == 262 )
      {
        v17 = 0;
        while ( 1 )
        {
          v162[v17] = v16;
          v17 = (unsigned int)(v163 + 1);
          LODWORD(v163) = v163 + 1;
          v18 = *(_DWORD *)(v16 + 64);
          if ( !v18 )
            break;
          v19 = (unsigned int *)(*(_QWORD *)(v16 + 40) + 40LL * (unsigned int)(v18 - 1));
          v16 = *(_QWORD *)v19;
          if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v19 + 48LL) + 16LL * v19[2]) != 262 )
            break;
          if ( v17 + 1 > (unsigned __int64)HIDWORD(v163) )
          {
            sub_C8D5F0((__int64)&v162, v164, v17 + 1, 8u, v17 + 1, v5);
            v17 = (unsigned int)v163;
          }
        }
        if ( (_DWORD)v17 )
        {
          while ( 1 )
          {
            v31 = v162[v17 - 1];
            v32 = sub_3360970(&v157, v31, v12[1] != (_QWORD)v12, (*((_BYTE *)v12 + 249) & 0x20) != 0, (__int64)&v176);
            v33 = v32;
            if ( v144 )
              sub_3361650(v31, *(_QWORD *)(v2 + 592), (__int64)v159, (__int64)&v176, (__int64)&v180, (__int64)v165, v32);
            v21 = *(_QWORD *)(v2 + 592);
            v22 = *(unsigned int *)(v21 + 752);
            v23 = *(_QWORD *)(v21 + 736);
            if ( (_DWORD)v22 )
            {
              v24 = (v22 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
              v25 = (__int64 *)(v23 + 80LL * v24);
              v26 = *v25;
              if ( v31 != *v25 )
              {
                v53 = 1;
                while ( v26 != -4096 )
                {
                  v54 = v53 + 1;
                  v24 = (v22 - 1) & (v53 + v24);
                  v25 = (__int64 *)(v23 + 80LL * v24);
                  v26 = *v25;
                  if ( v31 == *v25 )
                    goto LABEL_26;
                  v53 = v54;
                }
                goto LABEL_32;
              }
LABEL_26:
              if ( v25 != (__int64 *)(80 * v22 + v23) )
              {
                v27 = v25[4];
                if ( v33 )
                {
                  if ( v27 )
                  {
                    v28 = *(_DWORD *)(v33 + 44);
                    if ( (v28 & 4) != 0 || (v28 & 8) == 0 )
                    {
                      if ( (*(_QWORD *)(*(_QWORD *)(v33 + 16) + 24LL) & 0x80u) != 0LL )
LABEL_36:
                        sub_2E880E0(v33, *(_QWORD *)(v2 + 32), v27);
                    }
                    else
                    {
                      v129 = v27;
                      v29 = sub_2E88A90(v33, 128, 1);
                      v27 = v129;
                      if ( v29 )
                        goto LABEL_36;
                    }
                  }
                }
              }
            }
LABEL_32:
            v30 = (_DWORD)v163 == 1;
            v17 = (unsigned int)(v163 - 1);
            LODWORD(v163) = v163 - 1;
            if ( v30 )
            {
              v13 = *v12;
              goto LABEL_38;
            }
          }
        }
        v13 = *v12;
      }
    }
LABEL_38:
    v34 = sub_3360970(&v157, v13, v12[1] != (_QWORD)v12, (*((_BYTE *)v12 + 249) & 0x20) != 0, (__int64)&v176);
    v35 = v34;
    if ( v144 )
      sub_3361650(*v12, *(_QWORD *)(v2 + 592), (__int64)v159, (__int64)&v176, (__int64)&v180, (__int64)v165, v34);
    v36 = *(_QWORD *)(v2 + 592);
    v37 = *v12;
    v38 = *(unsigned int *)(v36 + 752);
    v39 = *(_QWORD *)(v36 + 736);
    if ( (_DWORD)v38 )
    {
      v4 = (unsigned int)(v38 - 1);
      v40 = v4 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v41 = (__int64 *)(v39 + 80LL * v40);
      v5 = *v41;
      if ( v37 != *v41 )
      {
        v55 = 1;
        while ( v5 != -4096 )
        {
          v56 = v55 + 1;
          v40 = v4 & (v55 + v40);
          v41 = (__int64 *)(v39 + 80LL * v40);
          v5 = *v41;
          if ( v37 == *v41 )
            goto LABEL_42;
          v55 = v56;
        }
        goto LABEL_48;
      }
LABEL_42:
      if ( v41 != (__int64 *)(80 * v38 + v39) )
      {
        v42 = v41[4];
        if ( v42 )
        {
          if ( v35 )
          {
            v43 = *(_DWORD *)(v35 + 44);
            if ( (v43 & 4) != 0 || (v43 & 8) == 0 )
            {
              if ( (*(_QWORD *)(*(_QWORD *)(v35 + 16) + 24LL) & 0x80u) == 0LL )
                goto LABEL_48;
            }
            else if ( !sub_2E88A90(v35, 128, 1) )
            {
              goto LABEL_48;
            }
            sub_2E880E0(v35, *(_QWORD *)(v2 + 32), v42);
          }
        }
      }
    }
LABEL_48:
    if ( v162 != (__int64 *)v164 )
      _libc_free((unsigned __int64)v162);
LABEL_50:
    ++v136;
  }
  while ( v130 != v136 );
  if ( v144 )
  {
LABEL_93:
    v10 = *(_QWORD *)(v2 + 584);
LABEL_94:
    v62 = sub_2E311E0(v10);
    v63 = v180;
    v64 = (_QWORD *)v62;
    v65 = (unsigned int)v181;
    v66 = (unsigned int *)&v180[v65];
    if ( !(v65 * 16) )
      goto LABEL_159;
    v146 = v2;
    v67 = (v65 * 16) >> 4;
    v68 = (unsigned int *)&v180[v65];
    do
    {
      v69 = 4 * v67;
      v70 = sub_2207800(16 * v67);
      v71 = (int *)v70;
      if ( v70 )
      {
        v72 = v70 + v69 * 4;
        v73 = v67;
        v74 = v70 + 16;
        v2 = v146;
        *(__m128i *)(v74 - 16) = _mm_loadu_si128(v63);
        if ( v72 == v74 )
        {
          v76 = (__int64)v71;
        }
        else
        {
          do
          {
            v75 = _mm_loadu_si128((const __m128i *)(v74 - 16));
            v74 += 16;
            *(__m128i *)(v74 - 16) = v75;
          }
          while ( v72 != v74 );
          v76 = (__int64)&v71[v69 - 4];
        }
        v151 = v71;
        v63->m128i_i32[0] = *(_DWORD *)v76;
        v63->m128i_i64[1] = *(_QWORD *)(v76 + 8);
        sub_3362720(v63->m128i_i32, v68, v71, v73);
        v77 = (unsigned __int64)v151;
        goto LABEL_101;
      }
      v67 >>= 1;
    }
    while ( v67 );
    v66 = v68;
    v2 = v146;
LABEL_159:
    sub_335D700(v63->m128i_i8, v66);
    v77 = 0;
LABEL_101:
    j_j___libc_free_0(v77);
    v78 = *(_QWORD *)(*(_QWORD *)(v2 + 592) + 720LL);
    v79 = *(char **)(v78 + 96);
    v80 = 8LL * *(unsigned int *)(v78 + 104);
    v81 = &v79[v80];
    if ( !v80 )
      goto LABEL_162;
    v147 = v2;
    v82 = v80 >> 3;
    v83 = &v79[v80];
    do
    {
      v84 = (char *)sub_2207800(8 * v82);
      if ( v84 )
      {
        v85 = (char *)v82;
        v2 = v147;
        v152 = v84;
        sub_335DBF0(v79, v83, v84, v85);
        v86 = (unsigned __int64)v152;
        goto LABEL_105;
      }
      v82 >>= 1;
    }
    while ( v82 );
    v81 = v83;
    v2 = v147;
LABEL_162:
    sub_335D520(v79, v81);
    v86 = 0;
LABEL_105:
    j_j___libc_free_0(v86);
    v88 = *(_QWORD *)(*(_QWORD *)(v2 + 592) + 720LL);
    v89 = *(_QWORD **)(v88 + 96);
    v153 = &v89[*(unsigned int *)(v88 + 104)];
    v132 = v181;
    if ( !(_DWORD)v181 || &v89[*(unsigned int *)(v88 + 104)] == v89 )
      goto LABEL_127;
    v90 = *v89;
    v134 = v2;
    v91 = *(_QWORD **)(v88 + 96);
    v141 = 0;
    v92 = *(_DWORD *)(*v89 + 56LL);
    v93 = 0;
LABEL_108:
    v94 = v93;
    v95 = &v180[v141];
    v93 = v95->m128i_i32[0];
    v96 = (unsigned __int64 *)v95->m128i_i64[1];
    while ( 1 )
    {
      if ( v92 < v94 || v92 >= v93 )
      {
        if ( ++v141 == v132 || v153 == v91 )
        {
          v2 = v134;
          v89 = v91;
LABEL_127:
          v102 = v153;
          v162 = (__int64 *)v164;
          v163 = 0x800000000LL;
          if ( v153 != v89 )
          {
            v154 = v2;
            v103 = v89;
            do
            {
              if ( !*(_BYTE *)(*v103 + 63LL) )
              {
                v104 = sub_37547E0(v159, *v103, &v176, v87);
                if ( v104 )
                {
                  v106 = (unsigned int)v163;
                  v107 = (unsigned int)v163 + 1LL;
                  if ( v107 > HIDWORD(v163) )
                  {
                    v143 = v104;
                    sub_C8D5F0((__int64)&v162, v164, (unsigned int)v163 + 1LL, 8u, v105, v107);
                    v106 = (unsigned int)v163;
                    v104 = v143;
                  }
                  v87 = (unsigned __int64)v162;
                  v162[v106] = v104;
                  LODWORD(v163) = v163 + 1;
                }
              }
              ++v103;
            }
            while ( v103 != v102 );
            v2 = v154;
          }
LABEL_136:
          v108 = v160;
          v110 = sub_2E313E0(v160);
          if ( v162 != &v162[(unsigned int)v163] )
          {
            v155 = v64;
            v111 = (__int64 *)(v108 + 40);
            v112 = (__int64 *)v110;
            v148 = v2;
            v113 = v162;
            v114 = &v162[(unsigned int)v163];
            do
            {
              v115 = *v113++;
              sub_2E31040(v111, v115);
              v116 = *v112;
              v117 = *(_QWORD *)v115;
              *(_QWORD *)(v115 + 8) = v112;
              v109 = v116 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)v115 = v109 | v117 & 7;
              *(_QWORD *)(v109 + 8) = v115;
              *v112 = *v112 & 7 | v115;
            }
            while ( v114 != v113 );
            v64 = v155;
            v2 = v148;
          }
          v118 = (__int64)v180;
          v119 = *(_QWORD *)(*(_QWORD *)(v2 + 592) + 720LL);
          v149 = *(_QWORD *)(v119 + 640) + 8LL * *(unsigned int *)(v119 + 648);
          v133 = &v180[(unsigned int)v181];
          if ( v133 == v180 )
            goto LABEL_152;
          v142 = (unsigned __int64)v180;
          v120 = *(_QWORD **)(v119 + 640);
          v121 = 0;
          v135 = v2;
LABEL_142:
          v122 = *(unsigned __int64 **)(v142 + 8);
          if ( !v122 )
            goto LABEL_155;
          if ( (_QWORD *)v149 == v120 )
          {
LABEL_152:
            if ( v162 != (__int64 *)v164 )
              _libc_free((unsigned __int64)v162);
            goto LABEL_52;
          }
          v156 = *(_DWORD *)v142;
          while ( 1 )
          {
LABEL_147:
            v125 = *(_DWORD *)(*v120 + 16LL);
            if ( v125 < v121 || v156 <= v125 )
            {
              v121 = v156;
LABEL_155:
              v142 += 16LL;
              if ( v133 == (__m128i *)v142 )
                goto LABEL_152;
              goto LABEL_142;
            }
            v126 = sub_37548B0(v159, *v120, v109, v118, v110);
            if ( !v126 )
              goto LABEL_146;
            if ( !v121 )
              break;
            v140 = v126;
            ++v120;
            sub_2E31040((__int64 *)(v122[3] + 40), v126);
            v127 = *(_QWORD *)v140;
            v128 = *v122 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v140 + 8) = v122;
            *(_QWORD *)v140 = v128 | v127 & 7;
            *(_QWORD *)(v128 + 8) = v140;
            v118 = *v122 & 7;
            *v122 = v118 | v140;
            if ( v120 == (_QWORD *)v149 )
              goto LABEL_152;
          }
          v139 = v126;
          sub_2E31040((__int64 *)(*(_QWORD *)(v135 + 584) + 40LL), v126);
          v123 = *(_QWORD *)v139;
          v124 = *v64 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v139 + 8) = v64;
          *(_QWORD *)v139 = v124 | v123 & 7;
          *(_QWORD *)(v124 + 8) = v139;
          v118 = *v64 & 7LL;
          *v64 = v118 | v139;
LABEL_146:
          if ( ++v120 == (_QWORD *)v149 )
            goto LABEL_152;
          goto LABEL_147;
        }
        goto LABEL_108;
      }
      if ( *(_BYTE *)(v90 + 63) )
        goto LABEL_110;
      v99 = sub_37547E0(v159, v90, &v176, v87);
      if ( !v99 )
        goto LABEL_110;
      if ( !v94 )
        break;
      v138 = v99;
      ++v91;
      sub_2E31040((__int64 *)(v96[3] + 40), v99);
      v100 = *(_QWORD *)v138;
      v101 = *v96 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v138 + 8) = v96;
      *(_QWORD *)v138 = v101 | v100 & 7;
      *(_QWORD *)(v101 + 8) = v138;
      *v96 = *v96 & 7 | v138;
      if ( v91 == v153 )
      {
LABEL_118:
        v2 = v134;
        v162 = (__int64 *)v164;
        v163 = 0x800000000LL;
        goto LABEL_136;
      }
LABEL_111:
      v90 = *v91;
      v92 = *(_DWORD *)(*v91 + 56LL);
    }
    v137 = v99;
    sub_2E31040((__int64 *)(*(_QWORD *)(v134 + 584) + 40LL), v99);
    v97 = *(_QWORD *)v137;
    v98 = *v64 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v137 + 8) = v64;
    *(_QWORD *)v137 = v98 | v97 & 7;
    *(_QWORD *)(v98 + 8) = v137;
    *v64 = *v64 & 7LL | v137;
LABEL_110:
    if ( ++v91 == v153 )
      goto LABEL_118;
    goto LABEL_111;
  }
LABEL_52:
  v44 = v160;
  *a2 = v161;
  v45 = v44 + 48;
  v46 = sub_2E313E0(v44);
  v47 = (unsigned __int64 *)v46;
  if ( v46 != v44 + 48 )
  {
    if ( !v46 )
      BUG();
    if ( (*(_BYTE *)v46 & 4) == 0 && (*(_BYTE *)(v46 + 44) & 8) != 0 )
    {
      do
        v46 = *(_QWORD *)(v46 + 8);
      while ( (*(_BYTE *)(v46 + 44) & 8) != 0 );
    }
    v48 = *(_QWORD *)(v46 + 8);
LABEL_56:
    if ( v45 != v48 )
    {
      do
      {
        if ( !v48 )
          BUG();
        v49 = v48;
        if ( (*(_BYTE *)v48 & 4) == 0 && (*(_BYTE *)(v48 + 44) & 8) != 0 )
        {
          do
            v49 = *(_QWORD *)(v49 + 8);
          while ( (*(_BYTE *)(v49 + 44) & 8) != 0 );
        }
        v50 = *(_QWORD *)(v49 + 8);
        if ( *a2 == (__int64 *)v48 )
          break;
        if ( (unsigned __int16)(*(_WORD *)(v48 + 68) - 14) > 1u )
        {
          v48 = *(_QWORD *)(v49 + 8);
          goto LABEL_56;
        }
        sub_2EAB560(*(char **)(v48 + 32), 0, 0, 0, 0, 0, 0, 0);
        v51 = v48;
        v48 = v50;
        sub_2E86600(v51, v47);
      }
      while ( v45 != v50 );
    }
  }
  sub_335D050(v168);
  if ( (_BYTE *)v165[0] != v166 )
    _libc_free(v165[0]);
  if ( v180 != (__m128i *)v182 )
    _libc_free((unsigned __int64)v180);
  if ( (v173 & 1) == 0 )
    sub_C7D6A0(v174, 16LL * v175, 8);
  if ( (v177 & 1) == 0 )
    sub_C7D6A0(v178, 24LL * v179, 8);
  return v44;
}
