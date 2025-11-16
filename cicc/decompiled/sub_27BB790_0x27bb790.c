// Function: sub_27BB790
// Address: 0x27bb790
//
unsigned __int64 __fastcall sub_27BB790(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8,
        char a9)
{
  __int64 v11; // rdx
  __int16 v12; // kr00_2
  __int64 *v13; // rax
  __int64 *v14; // rbx
  __int64 *v15; // r13
  const __m128i *v16; // r8
  unsigned __int64 v17; // r9
  const char *v18; // r12
  _BYTE *v20; // rdi
  __int64 v21; // rbx
  _BYTE *v22; // r13
  int v23; // esi
  _BYTE *v24; // rdi
  __int64 *v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // r12
  __int64 v28; // r15
  const __m128i *v29; // r12
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // r14
  unsigned __int64 *v32; // r15
  __int64 v33; // rax
  const __m128i *v34; // r13
  char *v35; // rdx
  __m128i *v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rax
  const __m128i *v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // rax
  __m128i *v42; // rdi
  unsigned __int64 *v43; // rax
  const __m128i *v44; // rax
  __m128i v45; // xmm0
  __m128i *v46; // r13
  int v47; // ebx
  size_t v48; // r14
  unsigned __int64 v49; // rax
  __m128i *v50; // r12
  const __m128i *v51; // rbx
  const __m128i *v52; // rdi
  __int64 v53; // rbx
  unsigned int v54; // r12d
  __int64 *v55; // r15
  __int64 v56; // r14
  unsigned int v57; // eax
  int v58; // eax
  __int64 v60; // rax
  char *v61; // r15
  __int64 v62; // rcx
  __int64 v63; // rax
  __int64 v64; // rcx
  unsigned int v65; // edx
  int v66; // eax
  __int64 *v67; // r8
  unsigned int v68; // edx
  int v69; // eax
  __int64 *v70; // r8
  unsigned int v71; // edx
  int v72; // eax
  __int64 *v73; // r8
  unsigned int v74; // edx
  int v75; // eax
  __int64 *v76; // r8
  __int64 v77; // r8
  const char *v78; // rcx
  unsigned __int64 v79; // rdi
  const __m128i *v80; // rsi
  int v81; // eax
  __m128i *v82; // rdx
  unsigned int v83; // eax
  __int64 v84; // rsi
  __m128i *v85; // rdx
  unsigned __int64 v86; // rdx
  unsigned __int8 *v87; // r14
  __int64 v88; // r15
  __int64 v89; // rbx
  const char *v90; // r14
  __int64 v91; // rax
  unsigned __int64 v92; // rdx
  signed __int64 v93; // r13
  __int64 *v94; // r8
  unsigned int v95; // edx
  int v96; // ebx
  __m128i *v97; // rbx
  __m128i *v98; // rbx
  __m128i v99; // xmm5
  __m128i *v100; // rcx
  __m128i *v101; // rbx
  __m128i *v102; // rcx
  __int64 *v103; // r8
  unsigned int v104; // edx
  int v105; // eax
  __int64 *v106; // r8
  unsigned int v107; // edx
  int v108; // eax
  __int64 *v109; // rax
  __int8 *v110; // rbx
  char *v111; // rbx
  _BYTE *v112; // r14
  _QWORD *v113; // r12
  __int64 v114; // rdi
  __int64 *v115; // rax
  __int64 v116; // r13
  __int64 v117; // rax
  __int64 v118; // r15
  __int64 v119; // rax
  int v120; // edi
  bool v121; // zf
  int v122; // esi
  _QWORD *v123; // rdi
  __int64 *v124; // rax
  __int64 v125; // rsi
  __int64 v126; // [rsp+0h] [rbp-2A0h]
  int v127; // [rsp+Ch] [rbp-294h]
  unsigned __int64 v129; // [rsp+30h] [rbp-270h]
  char *v130; // [rsp+38h] [rbp-268h]
  unsigned int v131; // [rsp+54h] [rbp-24Ch]
  char *v132; // [rsp+60h] [rbp-240h]
  int v133; // [rsp+60h] [rbp-240h]
  const void **v134; // [rsp+68h] [rbp-238h]
  unsigned __int64 v135; // [rsp+70h] [rbp-230h]
  unsigned __int64 v136; // [rsp+70h] [rbp-230h]
  unsigned __int64 v137; // [rsp+70h] [rbp-230h]
  unsigned __int64 v138; // [rsp+70h] [rbp-230h]
  unsigned __int64 v139; // [rsp+70h] [rbp-230h]
  unsigned __int64 v140; // [rsp+70h] [rbp-230h]
  unsigned __int64 v141; // [rsp+70h] [rbp-230h]
  unsigned __int64 v142; // [rsp+70h] [rbp-230h]
  unsigned int v143; // [rsp+78h] [rbp-228h]
  int v144; // [rsp+78h] [rbp-228h]
  int v145; // [rsp+78h] [rbp-228h]
  unsigned int v146; // [rsp+78h] [rbp-228h]
  int v147; // [rsp+78h] [rbp-228h]
  unsigned int v148; // [rsp+78h] [rbp-228h]
  int v149; // [rsp+78h] [rbp-228h]
  unsigned int v150; // [rsp+78h] [rbp-228h]
  int v151; // [rsp+78h] [rbp-228h]
  unsigned int v152; // [rsp+78h] [rbp-228h]
  int v153; // [rsp+78h] [rbp-228h]
  __int64 *v154; // [rsp+78h] [rbp-228h]
  __int64 *v155; // [rsp+78h] [rbp-228h]
  __int64 *v156; // [rsp+78h] [rbp-228h]
  __int64 *v157; // [rsp+78h] [rbp-228h]
  const char *v158; // [rsp+78h] [rbp-228h]
  unsigned int v159; // [rsp+78h] [rbp-228h]
  unsigned int v160; // [rsp+78h] [rbp-228h]
  int v161; // [rsp+78h] [rbp-228h]
  unsigned int v162; // [rsp+78h] [rbp-228h]
  int v163; // [rsp+78h] [rbp-228h]
  __int64 *v164; // [rsp+78h] [rbp-228h]
  __int64 *v165; // [rsp+78h] [rbp-228h]
  __int64 *v166; // [rsp+78h] [rbp-228h]
  int v167; // [rsp+78h] [rbp-228h]
  __int16 v168; // [rsp+78h] [rbp-228h]
  int v169; // [rsp+94h] [rbp-20Ch] BYREF
  __int64 v170; // [rsp+98h] [rbp-208h]
  _QWORD *v171; // [rsp+A0h] [rbp-200h] BYREF
  unsigned int v172; // [rsp+A8h] [rbp-1F8h]
  _QWORD *v173; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v174; // [rsp+B8h] [rbp-1E8h]
  __m128i v175; // [rsp+C0h] [rbp-1E0h] BYREF
  __m128i v176; // [rsp+D0h] [rbp-1D0h]
  void *src; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v178; // [rsp+E8h] [rbp-1B8h]
  unsigned __int64 v179; // [rsp+F0h] [rbp-1B0h] BYREF
  unsigned int v180; // [rsp+F8h] [rbp-1A8h]
  __int16 v181; // [rsp+100h] [rbp-1A0h]
  unsigned __int64 *v182; // [rsp+150h] [rbp-150h] BYREF
  __int64 v183; // [rsp+158h] [rbp-148h]
  unsigned __int64 v184; // [rsp+160h] [rbp-140h] BYREF
  unsigned int v185; // [rsp+168h] [rbp-138h]
  char v186; // [rsp+170h] [rbp-130h]
  const char *v187; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v188; // [rsp+1E8h] [rbp-B8h]
  _BYTE v189[176]; // [rsp+1F0h] [rbp-B0h] BYREF

  v11 = *(unsigned int *)(a3 + 8);
  v12 = a8;
  v13 = *(__int64 **)a3;
  if ( (_DWORD)v11 == 1 && *(_DWORD *)(a2 + 8) == 1 )
  {
    v20 = (_BYTE *)*v13;
    if ( *(_BYTE *)*v13 == 82 )
    {
      v21 = *((_QWORD *)v20 - 8);
      if ( v21 )
      {
        v22 = (_BYTE *)*((_QWORD *)v20 - 4);
        if ( *v22 == 17 )
        {
          v23 = sub_B53900((__int64)v20);
          v24 = **(_BYTE ***)a2;
          if ( *v24 == 82 && v21 == *((_QWORD *)v24 - 8) )
          {
            v112 = (_BYTE *)*((_QWORD *)v24 - 4);
            if ( *v112 == 17 )
            {
              v167 = sub_B53900((__int64)v24);
              sub_AB1A50((__int64)&v175, v23, (__int64)(v22 + 24));
              sub_AB1A50((__int64)&src, v167, (__int64)(v112 + 24));
              sub_ABB730((__int64)&v182, (__int64)&v175, (__int64)&src);
              if ( v186 )
              {
                v172 = 1;
                v171 = 0;
                if ( (unsigned __int8)sub_AAFB30((__int64)&v182, &v169, (__int64)&v171) )
                {
                  v113 = 0;
                  if ( a9 )
                  {
                    v114 = a7 ? (__int64)(a7 - 3) : 0LL;
                    v115 = (__int64 *)sub_BD5C60(v114);
                    v116 = sub_ACCFD0(v115, (__int64)&v171);
                    sub_27B9800(a1, v21, a7, a8);
                    v189[17] = 1;
                    v187 = "wide.chk";
                    v189[16] = 3;
                    v113 = sub_BD2C40(72, unk_3F10FD0);
                    if ( v113 )
                    {
                      v168 = v169;
                      v117 = (unsigned __int8)a8;
                      BYTE1(v117) = BYTE1(a8);
                      v118 = v117;
                      v119 = *(_QWORD *)(v21 + 8);
                      v120 = *(unsigned __int8 *)(v119 + 8);
                      if ( (unsigned int)(v120 - 17) > 1 )
                      {
                        v125 = sub_BCB2A0(*(_QWORD **)v119);
                      }
                      else
                      {
                        v121 = (_BYTE)v120 == 18;
                        v122 = *(_DWORD *)(v119 + 32);
                        v123 = *(_QWORD **)v119;
                        BYTE4(v170) = v121;
                        LODWORD(v170) = v122;
                        v124 = (__int64 *)sub_BCB2A0(v123);
                        v125 = sub_BCE1B0(v124, v170);
                      }
                      sub_B523C0((__int64)v113, v125, 53, v168, v21, v116, (__int64)&v187, (__int64)a7, v118, 0);
                    }
                  }
                  v173 = v113;
                  LOBYTE(v174) = 1;
                  if ( v172 > 0x40 && v171 )
                    j_j___libc_free_0_0((unsigned __int64)v171);
                  if ( v186 )
                  {
                    v186 = 0;
                    if ( v185 > 0x40 && v184 )
                      j_j___libc_free_0_0(v184);
                    if ( (unsigned int)v183 > 0x40 && v182 )
                      j_j___libc_free_0_0((unsigned __int64)v182);
                  }
                  if ( v180 > 0x40 && v179 )
                    j_j___libc_free_0_0(v179);
                  if ( (unsigned int)v178 > 0x40 && src )
                    j_j___libc_free_0_0((unsigned __int64)src);
                  if ( v176.m128i_i32[2] > 0x40u && v176.m128i_i64[0] )
                    j_j___libc_free_0_0(v176.m128i_u64[0]);
                  if ( v175.m128i_i32[2] > 0x40u && v175.m128i_i64[0] )
                    j_j___libc_free_0_0(v175.m128i_u64[0]);
                  return (unsigned __int64)v173;
                }
                if ( v172 > 0x40 && v171 )
                  j_j___libc_free_0_0((unsigned __int64)v171);
                if ( v186 )
                {
                  v186 = 0;
                  if ( v185 > 0x40 && v184 )
                    j_j___libc_free_0_0(v184);
                  if ( (unsigned int)v183 > 0x40 && v182 )
                    j_j___libc_free_0_0((unsigned __int64)v182);
                }
              }
              if ( v180 > 0x40 && v179 )
                j_j___libc_free_0_0(v179);
              if ( (unsigned int)v178 > 0x40 && src )
                j_j___libc_free_0_0((unsigned __int64)src);
              if ( v176.m128i_i32[2] > 0x40u && v176.m128i_i64[0] )
                j_j___libc_free_0_0(v176.m128i_u64[0]);
              if ( v175.m128i_i32[2] > 0x40u && v175.m128i_i64[0] )
                j_j___libc_free_0_0(v175.m128i_u64[0]);
            }
          }
          v13 = *(__int64 **)a3;
          v11 = *(unsigned int *)(a3 + 8);
        }
      }
    }
  }
  v14 = &v13[v11];
  v182 = &v184;
  v183 = 0x400000000LL;
  v187 = v189;
  v188 = 0x400000000LL;
  if ( v14 == v13 )
  {
    v25 = *(__int64 **)a2;
    v26 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 == v26 )
      goto LABEL_9;
    goto LABEL_20;
  }
  v15 = v13;
  do
  {
    if ( !(unsigned __int8)sub_27B9EC0(*v15, (__int64)&v182) )
      goto LABEL_6;
    ++v15;
  }
  while ( v14 != v15 );
  v25 = *(__int64 **)a2;
  v26 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v26 )
  {
LABEL_20:
    v27 = v25;
    do
    {
      if ( !(unsigned __int8)sub_27B9EC0(*v27, (__int64)&v182) )
        goto LABEL_6;
      ++v27;
    }
    while ( (__int64 *)v26 != v27 );
  }
  v127 = v183;
  if ( !(_DWORD)v183 )
  {
LABEL_129:
    v18 = v187;
    if ( (_DWORD)v188 == v127 )
      goto LABEL_7;
    v86 = 0;
    if ( a9 )
    {
      v87 = 0;
      if ( &v187[32 * (unsigned int)v188] != v187 )
      {
        v158 = &v187[32 * (unsigned int)v188];
        v88 = 0;
        v89 = v126;
        v90 = v187;
        do
        {
          LOWORD(a8) = v12;
          sub_27B9800(a1, *((_QWORD *)v90 + 3), a7, a8);
          if ( v88 )
          {
            v181 = 257;
            LOWORD(v89) = v12;
            v88 = sub_B504D0(28, *((_QWORD *)v90 + 3), v88, (__int64)&src, (__int64)a7, v89);
          }
          else
          {
            v88 = *((_QWORD *)v90 + 3);
          }
          v90 += 32;
        }
        while ( v90 != v158 );
        v87 = (unsigned __int8 *)v88;
      }
      src = "wide.chk";
      v181 = 259;
      sub_BD6B50(v87, (const char **)&src);
      LOWORD(a8) = v12;
      v109 = sub_27BA8D0(a1, v87, a7, a8);
      v18 = v187;
      v86 = (unsigned __int64)v109;
    }
    v173 = (_QWORD *)v86;
    LOBYTE(v174) = 1;
    if ( v18 != v189 )
      _libc_free((unsigned __int64)v18);
    if ( v182 != &v184 )
      _libc_free((unsigned __int64)v182);
    return (unsigned __int64)v173;
  }
  v28 = (unsigned int)v183;
  while ( 1 )
  {
    v29 = (const __m128i *)v182;
    v30 = *v182;
    v31 = v182[2];
    src = &v179;
    v32 = &v182[4 * v28];
    v178 = 0x300000000LL;
    do
    {
      while ( v30 != v29->m128i_i64[0] || v31 != v29[1].m128i_i64[0] )
      {
        v29 += 2;
        if ( v32 == (unsigned __int64 *)v29 )
          goto LABEL_32;
      }
      v33 = (unsigned int)v178;
      v34 = v29;
      v35 = (char *)src;
      v16 = (const __m128i *)((unsigned int)v178 + 1LL);
      if ( (unsigned __int64)v16 > HIDWORD(v178) )
      {
        if ( src > v29 || (char *)src + 32 * (unsigned int)v178 <= (char *)v29 )
        {
          v34 = v29;
          sub_C8D5F0((__int64)&src, &v179, (unsigned int)v178 + 1LL, 0x20u, (__int64)v16, v17);
          v35 = (char *)src;
          v33 = (unsigned int)v178;
        }
        else
        {
          v93 = (char *)v29 - (_BYTE *)src;
          sub_C8D5F0((__int64)&src, &v179, (unsigned int)v178 + 1LL, 0x20u, (__int64)v16, v17);
          v35 = (char *)src;
          v33 = (unsigned int)v178;
          v34 = (const __m128i *)((char *)src + v93);
        }
      }
      v29 += 2;
      v36 = (__m128i *)&v35[32 * v33];
      *v36 = _mm_loadu_si128(v34);
      v36[1] = _mm_loadu_si128(v34 + 1);
      LODWORD(v178) = v178 + 1;
    }
    while ( v32 != (unsigned __int64 *)v29 );
LABEL_32:
    v37 = (unsigned __int64)v182;
    v38 = 32LL * (unsigned int)v183;
    v39 = (const __m128i *)&v182[(unsigned __int64)v38 / 8];
    v40 = v38 >> 5;
    v41 = v38 >> 7;
    if ( v41 )
    {
      v42 = (__m128i *)v182;
      v43 = &v182[16 * v41];
      while ( v30 != v42->m128i_i64[0] || v31 != v42[1].m128i_i64[0] )
      {
        if ( v30 == v42[2].m128i_i64[0] && v31 == v42[3].m128i_i64[0] )
        {
          v42 += 2;
          goto LABEL_40;
        }
        if ( v30 == v42[4].m128i_i64[0] && v31 == v42[5].m128i_i64[0] )
        {
          v42 += 4;
          goto LABEL_40;
        }
        if ( v30 == v42[6].m128i_i64[0] && v31 == v42[7].m128i_i64[0] )
        {
          v42 += 6;
          goto LABEL_40;
        }
        v42 += 8;
        if ( v43 == (unsigned __int64 *)v42 )
        {
          v40 = ((char *)v39 - (char *)v42) >> 5;
          goto LABEL_153;
        }
      }
      goto LABEL_40;
    }
    v42 = (__m128i *)v182;
LABEL_153:
    if ( v40 == 2 )
      goto LABEL_175;
    if ( v40 == 3 )
    {
      if ( v30 == v42->m128i_i64[0] && v31 == v42[1].m128i_i64[0] )
        goto LABEL_40;
      v42 += 2;
LABEL_175:
      if ( v30 == v42->m128i_i64[0] && v31 == v42[1].m128i_i64[0] )
        goto LABEL_40;
      v42 += 2;
      goto LABEL_177;
    }
    if ( v40 != 1 )
      goto LABEL_156;
LABEL_177:
    if ( v30 != v42->m128i_i64[0] || v31 != v42[1].m128i_i64[0] )
    {
LABEL_156:
      v97 = (__m128i *)v39;
      goto LABEL_49;
    }
LABEL_40:
    if ( v39 == v42 )
      goto LABEL_156;
    v44 = v42 + 2;
    if ( v39 == &v42[2] )
    {
      v97 = v42;
    }
    else
    {
      do
      {
        while ( v30 != v44->m128i_i64[0] || v31 != v44[1].m128i_i64[0] )
        {
          v45 = _mm_loadu_si128(v44);
          v44 += 2;
          v42 += 2;
          v42[-2] = v45;
          v42[-1] = _mm_loadu_si128(v44 - 1);
          if ( v39 == v44 )
            goto LABEL_47;
        }
        v44 += 2;
      }
      while ( v39 != v44 );
LABEL_47:
      v37 = (unsigned __int64)v182;
      v16 = (const __m128i *)((char *)&v182[4 * (unsigned int)v183] - (char *)v39);
      v97 = (__m128i *)((char *)v16 + (_QWORD)v42);
      if ( &v182[4 * (unsigned int)v183] != (unsigned __int64 *)v39 )
      {
        memmove(v42, v39, (char *)&v182[4 * (unsigned int)v183] - (char *)v39);
        v37 = (unsigned __int64)v182;
      }
    }
LABEL_49:
    v46 = (__m128i *)src;
    LODWORD(v183) = (__int64)((__int64)v97->m128i_i64 - v37) >> 5;
    v47 = v178;
    v48 = 32LL * (unsigned int)v178;
    if ( (unsigned int)v178 <= 2 )
    {
      v91 = (unsigned int)v188;
      v92 = (unsigned int)v188 + (unsigned __int64)(unsigned int)v178;
      if ( v92 > HIDWORD(v188) )
      {
        sub_C8D5F0((__int64)&v187, v189, v92, 0x20u, (__int64)v16, v17);
        v91 = (unsigned int)v188;
      }
      if ( v48 )
      {
        memcpy((void *)&v187[32 * v91], v46, v48);
        LODWORD(v91) = v188;
      }
      LODWORD(v188) = v47 + v91;
      goto LABEL_126;
    }
    _BitScanReverse64(&v49, (unsigned int)v178);
    v50 = (__m128i *)((char *)src + v48);
    sub_27B9290((__m128i *)src, (__m128i *)((char *)src + v48), 2LL * (int)(63 - (v49 ^ 0x3F)), v40, (__int64)v16, v17);
    if ( v48 <= 0x200 )
    {
      sub_27B96C0(v46, &v46[v48 / 0x10]);
    }
    else
    {
      v51 = v46 + 32;
      sub_27B96C0(v46, v46 + 32);
      if ( &v46[32] != v50 )
      {
        do
        {
          v52 = v51;
          v51 += 2;
          sub_27B90E0(v52);
        }
        while ( v51 != v50 );
      }
    }
    v53 = *((_QWORD *)src + 4 * (unsigned int)v178 - 3);
    v54 = *(_DWORD *)(v53 + 32);
    v55 = (__int64 *)(*((_QWORD *)src + 1) + 24LL);
    v134 = (const void **)(v53 + 24);
    LODWORD(v174) = v54;
    v56 = 1LL << ((unsigned __int8)v54 - 1);
    if ( v54 <= 0x40 )
    {
      v173 = *(_QWORD **)(v53 + 24);
      sub_C46B40((__int64)&v173, v55);
      v57 = v174;
      v175.m128i_i64[0] = 0;
      LODWORD(v174) = 0;
      v143 = v57;
      v172 = v57;
      v175.m128i_i32[2] = v54;
      v135 = (unsigned __int64)v173;
      v171 = v173;
LABEL_55:
      v175.m128i_i64[0] |= v56;
      v58 = sub_C49970((__int64)&v171, (unsigned __int64 *)&v175);
      goto LABEL_56;
    }
    sub_C43780((__int64)&v173, v134);
    sub_C46B40((__int64)&v173, v55);
    v175.m128i_i32[2] = v54;
    v143 = v174;
    v172 = v174;
    LODWORD(v174) = 0;
    v135 = (unsigned __int64)v173;
    v171 = v173;
    sub_C43690((__int64)&v175, 0, 0);
    if ( v175.m128i_i32[2] <= 0x40u )
      goto LABEL_55;
    *(_QWORD *)(v175.m128i_i64[0] + 8LL * ((v54 - 1) >> 6)) |= v56;
    v58 = sub_C49970((__int64)&v171, (unsigned __int64 *)&v175);
    if ( v175.m128i_i32[2] > 0x40u && v175.m128i_i64[0] )
    {
      v133 = v58;
      j_j___libc_free_0_0(v175.m128i_u64[0]);
      v58 = v133;
    }
LABEL_56:
    if ( v143 > 0x40 && v135 )
    {
      v144 = v58;
      j_j___libc_free_0_0(v135);
      v58 = v144;
    }
    if ( (unsigned int)v174 > 0x40 && v173 )
    {
      v145 = v58;
      j_j___libc_free_0_0((unsigned __int64)v173);
      v58 = v145;
    }
    if ( v58 > 0 )
      goto LABEL_205;
    v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
    if ( v175.m128i_i32[2] > 0x40u )
      sub_C43780((__int64)&v175, v134);
    else
      v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
    sub_C46B40((__int64)&v175, v55);
    v131 = v175.m128i_u32[2];
    v172 = v175.m128i_u32[2];
    v129 = v175.m128i_i64[0];
    v171 = (_QWORD *)v175.m128i_i64[0];
    if ( v175.m128i_i32[2] <= 0x40u ? v175.m128i_i64[0] == 0 : v131 == (unsigned int)sub_C444A0((__int64)&v171) )
      break;
    v60 = 32LL * (unsigned int)v178;
    v61 = (char *)src + 32;
    v130 = (char *)src + v60;
    v62 = v60 - 32;
    v63 = (v60 - 32) >> 7;
    v64 = v62 >> 5;
    if ( v63 > 0 )
    {
      v132 = (char *)src + 128 * v63 + 32;
      while ( 1 )
      {
        v76 = (__int64 *)(*((_QWORD *)v61 + 1) + 24LL);
        v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
        if ( v175.m128i_i32[2] <= 0x40u )
        {
          v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
        }
        else
        {
          v154 = v76;
          sub_C43780((__int64)&v175, v134);
          v76 = v154;
        }
        sub_C46B40((__int64)&v175, v76);
        v65 = v175.m128i_u32[2];
        v175.m128i_i32[2] = 0;
        LODWORD(v174) = v65;
        v146 = v65;
        v173 = (_QWORD *)v175.m128i_i64[0];
        v136 = v175.m128i_i64[0];
        v66 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
        if ( v146 > 0x40 )
        {
          if ( v136 )
          {
            v147 = v66;
            j_j___libc_free_0_0(v136);
            v66 = v147;
            if ( v175.m128i_i32[2] > 0x40u )
            {
              if ( v175.m128i_i64[0] )
              {
                j_j___libc_free_0_0(v175.m128i_u64[0]);
                v66 = v147;
              }
            }
          }
        }
        if ( v66 >= 0 )
          break;
        v67 = (__int64 *)(*((_QWORD *)v61 + 5) + 24LL);
        v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
        if ( v175.m128i_i32[2] > 0x40u )
        {
          v155 = v67;
          sub_C43780((__int64)&v175, v134);
          v67 = v155;
        }
        else
        {
          v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
        }
        sub_C46B40((__int64)&v175, v67);
        v68 = v175.m128i_u32[2];
        v175.m128i_i32[2] = 0;
        LODWORD(v174) = v68;
        v148 = v68;
        v173 = (_QWORD *)v175.m128i_i64[0];
        v137 = v175.m128i_i64[0];
        v69 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
        if ( v148 > 0x40 )
        {
          if ( v137 )
          {
            v149 = v69;
            j_j___libc_free_0_0(v137);
            v69 = v149;
            if ( v175.m128i_i32[2] > 0x40u )
            {
              if ( v175.m128i_i64[0] )
              {
                j_j___libc_free_0_0(v175.m128i_u64[0]);
                v69 = v149;
              }
            }
          }
        }
        if ( v69 >= 0 )
        {
          v61 += 32;
          break;
        }
        v70 = (__int64 *)(*((_QWORD *)v61 + 9) + 24LL);
        v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
        if ( v175.m128i_i32[2] > 0x40u )
        {
          v156 = v70;
          sub_C43780((__int64)&v175, v134);
          v70 = v156;
        }
        else
        {
          v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
        }
        sub_C46B40((__int64)&v175, v70);
        v71 = v175.m128i_u32[2];
        v175.m128i_i32[2] = 0;
        LODWORD(v174) = v71;
        v150 = v71;
        v173 = (_QWORD *)v175.m128i_i64[0];
        v138 = v175.m128i_i64[0];
        v72 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
        if ( v150 > 0x40 )
        {
          if ( v138 )
          {
            v151 = v72;
            j_j___libc_free_0_0(v138);
            v72 = v151;
            if ( v175.m128i_i32[2] > 0x40u )
            {
              if ( v175.m128i_i64[0] )
              {
                j_j___libc_free_0_0(v175.m128i_u64[0]);
                v72 = v151;
              }
            }
          }
        }
        if ( v72 >= 0 )
        {
          v61 += 64;
          break;
        }
        v73 = (__int64 *)(*((_QWORD *)v61 + 13) + 24LL);
        v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
        if ( v175.m128i_i32[2] > 0x40u )
        {
          v157 = v73;
          sub_C43780((__int64)&v175, v134);
          v73 = v157;
        }
        else
        {
          v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
        }
        sub_C46B40((__int64)&v175, v73);
        v74 = v175.m128i_u32[2];
        v175.m128i_i32[2] = 0;
        LODWORD(v174) = v74;
        v152 = v74;
        v173 = (_QWORD *)v175.m128i_i64[0];
        v139 = v175.m128i_i64[0];
        v75 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
        if ( v152 > 0x40 )
        {
          if ( v139 )
          {
            v153 = v75;
            j_j___libc_free_0_0(v139);
            v75 = v153;
            if ( v175.m128i_i32[2] > 0x40u )
            {
              if ( v175.m128i_i64[0] )
              {
                j_j___libc_free_0_0(v175.m128i_u64[0]);
                v75 = v153;
              }
            }
          }
        }
        if ( v75 >= 0 )
        {
          v61 += 96;
          break;
        }
        v61 += 128;
        if ( v132 == v61 )
        {
          v64 = (v130 - v61) >> 5;
          goto LABEL_159;
        }
      }
LABEL_114:
      if ( v130 != v61 )
        break;
      goto LABEL_115;
    }
LABEL_159:
    if ( v64 != 2 )
    {
      if ( v64 != 3 )
      {
        if ( v64 != 1 )
          goto LABEL_115;
        goto LABEL_162;
      }
      v103 = (__int64 *)(*((_QWORD *)v61 + 1) + 24LL);
      v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
      if ( v175.m128i_i32[2] > 0x40u )
      {
        v166 = v103;
        sub_C43780((__int64)&v175, v134);
        v103 = v166;
      }
      else
      {
        v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
      }
      sub_C46B40((__int64)&v175, v103);
      v104 = v175.m128i_u32[2];
      v175.m128i_i32[2] = 0;
      LODWORD(v174) = v104;
      v160 = v104;
      v173 = (_QWORD *)v175.m128i_i64[0];
      v141 = v175.m128i_i64[0];
      v105 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
      if ( v160 > 0x40 )
      {
        if ( v141 )
        {
          v161 = v105;
          j_j___libc_free_0_0(v141);
          v105 = v161;
          if ( v175.m128i_i32[2] > 0x40u )
          {
            if ( v175.m128i_i64[0] )
            {
              j_j___libc_free_0_0(v175.m128i_u64[0]);
              v105 = v161;
            }
          }
        }
      }
      if ( v105 >= 0 )
        goto LABEL_114;
      v61 += 32;
    }
    v106 = (__int64 *)(*((_QWORD *)v61 + 1) + 24LL);
    v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
    if ( v175.m128i_i32[2] > 0x40u )
    {
      v164 = v106;
      sub_C43780((__int64)&v175, v134);
      v106 = v164;
    }
    else
    {
      v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
    }
    sub_C46B40((__int64)&v175, v106);
    v107 = v175.m128i_u32[2];
    v175.m128i_i32[2] = 0;
    LODWORD(v174) = v107;
    v162 = v107;
    v173 = (_QWORD *)v175.m128i_i64[0];
    v142 = v175.m128i_i64[0];
    v108 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
    if ( v162 > 0x40 )
    {
      if ( v142 )
      {
        v163 = v108;
        j_j___libc_free_0_0(v142);
        v108 = v163;
        if ( v175.m128i_i32[2] > 0x40u )
        {
          if ( v175.m128i_i64[0] )
          {
            j_j___libc_free_0_0(v175.m128i_u64[0]);
            v108 = v163;
          }
        }
      }
    }
    if ( v108 >= 0 )
      goto LABEL_114;
    v61 += 32;
LABEL_162:
    v94 = (__int64 *)(*((_QWORD *)v61 + 1) + 24LL);
    v175.m128i_i32[2] = *(_DWORD *)(v53 + 32);
    if ( v175.m128i_i32[2] > 0x40u )
    {
      v165 = v94;
      sub_C43780((__int64)&v175, v134);
      v94 = v165;
    }
    else
    {
      v175.m128i_i64[0] = *(_QWORD *)(v53 + 24);
    }
    sub_C46B40((__int64)&v175, v94);
    v95 = v175.m128i_u32[2];
    v175.m128i_i32[2] = 0;
    LODWORD(v174) = v95;
    v159 = v95;
    v173 = (_QWORD *)v175.m128i_i64[0];
    v140 = v175.m128i_i64[0];
    v96 = sub_C49970((__int64)&v173, (unsigned __int64 *)&v171);
    if ( v159 > 0x40 )
    {
      if ( v140 )
      {
        j_j___libc_free_0_0(v140);
        if ( v175.m128i_i32[2] > 0x40u )
        {
          if ( v175.m128i_i64[0] )
            j_j___libc_free_0_0(v175.m128i_u64[0]);
        }
      }
    }
    if ( v96 >= 0 )
      goto LABEL_114;
LABEL_115:
    v77 = (unsigned int)v188;
    v78 = v187;
    v79 = HIDWORD(v188);
    v80 = (const __m128i *)src;
    v81 = v188;
    v82 = (__m128i *)&v187[32 * (unsigned int)v188];
    if ( (unsigned int)v188 >= (unsigned __int64)HIDWORD(v188) )
    {
      v17 = (unsigned int)v188 + 1LL;
      v101 = &v175;
      v175 = _mm_loadu_si128((const __m128i *)src);
      v176 = _mm_loadu_si128((const __m128i *)src + 1);
      if ( HIDWORD(v188) < v17 )
      {
        if ( v187 > (const char *)&v175 || v82 <= &v175 )
        {
          v101 = &v175;
          sub_C8D5F0((__int64)&v187, v189, (unsigned int)v188 + 1LL, 0x20u, (unsigned int)v188, v17);
          v78 = v187;
          v77 = (unsigned int)v188;
        }
        else
        {
          v111 = (char *)((char *)&v175 - v187);
          sub_C8D5F0((__int64)&v187, v189, (unsigned int)v188 + 1LL, 0x20u, (unsigned int)v188, v17);
          v78 = v187;
          v77 = (unsigned int)v188;
          v101 = (__m128i *)&v111[(_QWORD)v187];
        }
      }
      v102 = (__m128i *)&v78[32 * v77];
      *v102 = _mm_loadu_si128(v101);
      v102[1] = _mm_loadu_si128(v101 + 1);
      v80 = (const __m128i *)src;
      v79 = HIDWORD(v188);
      v78 = v187;
      v83 = v188 + 1;
      LODWORD(v188) = v188 + 1;
    }
    else
    {
      if ( v82 )
      {
        *v82 = _mm_loadu_si128((const __m128i *)src);
        v82[1] = _mm_loadu_si128(v80 + 1);
        v81 = v188;
        v80 = (const __m128i *)src;
        v79 = HIDWORD(v188);
        v78 = v187;
      }
      v83 = v81 + 1;
      LODWORD(v188) = v83;
    }
    v16 = &v80[2 * (unsigned int)v178 - 2];
    v84 = v83;
    v85 = (__m128i *)&v78[32 * v83];
    if ( v79 <= v83 )
    {
      v98 = &v175;
      v175 = _mm_loadu_si128(v16);
      v99 = _mm_loadu_si128(v16 + 1);
      v16 = (const __m128i *)(v83 + 1LL);
      v176 = v99;
      if ( v79 < (unsigned __int64)v16 )
      {
        if ( v78 > (const char *)&v175 || v85 <= &v175 )
        {
          v98 = &v175;
          sub_C8D5F0((__int64)&v187, v189, (unsigned __int64)v16, 0x20u, (__int64)v16, v17);
          v78 = v187;
          v84 = (unsigned int)v188;
        }
        else
        {
          v110 = (__int8 *)((char *)&v175 - v78);
          sub_C8D5F0((__int64)&v187, v189, (unsigned __int64)v16, 0x20u, (__int64)v16, v17);
          v78 = v187;
          v84 = (unsigned int)v188;
          v98 = (__m128i *)&v110[(_QWORD)v187];
        }
      }
      v100 = (__m128i *)&v78[32 * v84];
      *v100 = _mm_loadu_si128(v98);
      v100[1] = _mm_loadu_si128(v98 + 1);
      LODWORD(v188) = v188 + 1;
    }
    else
    {
      if ( v85 )
      {
        *v85 = _mm_loadu_si128(v16);
        v85[1] = _mm_loadu_si128(v16 + 1);
        v83 = v188;
      }
      LODWORD(v188) = v83 + 1;
    }
    if ( v131 > 0x40 && v129 )
      j_j___libc_free_0_0(v129);
LABEL_126:
    if ( src != &v179 )
      _libc_free((unsigned __int64)src);
    v28 = (unsigned int)v183;
    if ( !(_DWORD)v183 )
      goto LABEL_129;
  }
  if ( v131 > 0x40 && v129 )
    j_j___libc_free_0_0(v129);
LABEL_205:
  if ( src != &v179 )
    _libc_free((unsigned __int64)src);
LABEL_6:
  v18 = v187;
LABEL_7:
  if ( v18 != v189 )
    _libc_free((unsigned __int64)v18);
LABEL_9:
  if ( v182 != &v184 )
    _libc_free((unsigned __int64)v182);
  LOBYTE(v174) = 0;
  return (unsigned __int64)v173;
}
