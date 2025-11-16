// Function: sub_9FF220
// Address: 0x9ff220
//
__int64 __fastcall sub_9FF220(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        __int64 a7)
{
  _QWORD *v7; // r13
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  size_t v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r8
  _BYTE *v13; // r13
  _BYTE *v14; // r14
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rbx
  volatile signed __int32 *v18; // r12
  signed __int32 v19; // eax
  void (*v20)(); // rax
  signed __int32 v21; // eax
  __int64 (__fastcall *v22)(__int64); // rcx
  char *v23; // rbx
  char *v24; // r12
  volatile signed __int32 *v25; // r13
  signed __int32 v26; // eax
  void (*v27)(); // rax
  signed __int32 v28; // eax
  __int64 (__fastcall *v29)(__int64); // rdx
  size_t v31; // rdi
  int v32; // edx
  __m128i *v33; // rsi
  __m128i v34; // xmm1
  __m128i v35; // xmm0
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // r15
  __m128i v40; // xmm1
  __m128i v41; // xmm0
  char *v42; // rsi
  void (__fastcall *v43)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 (__fastcall *v44)(__int64 *, int *, unsigned int *); // rdx
  int v45; // edi
  int v46; // eax
  __int64 v47; // r8
  __m128i v48; // xmm1
  __m128i v49; // xmm0
  __int64 v50; // rsi
  _QWORD *v51; // r15
  char *v52; // r15
  char *v53; // r12
  volatile signed __int32 *v54; // rdi
  signed __int32 v55; // eax
  void (*v56)(void); // rax
  signed __int32 v57; // eax
  void (*v58)(void); // rax
  __m128i *p_src; // r8
  size_t v60; // rdx
  __int64 v61; // rsi
  char v62; // al
  unsigned __int64 v63; // rax
  void (__fastcall *v64)(size_t *, size_t *, __int64); // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // rbx
  __int64 v68; // rax
  __m128i *v69; // rdi
  __int64 v70; // rdx
  __int64 v71; // r8
  __int64 v72; // r8
  _BYTE *v73; // r15
  _BYTE *v74; // r12
  _BYTE *v75; // r15
  __int64 v76; // rdi
  __int64 v77; // r13
  __int64 v78; // rbx
  volatile signed __int32 *v79; // r14
  signed __int32 v80; // eax
  void (*v81)(); // rax
  signed __int32 v82; // eax
  __int64 (__fastcall *v83)(__int64); // rdx
  char *v84; // r15
  char *v85; // r12
  volatile signed __int32 *v86; // rdi
  signed __int32 v87; // eax
  void (*v88)(void); // rax
  signed __int32 v89; // eax
  void (*v90)(void); // rax
  __int64 v91; // r15
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 v94; // r13
  char v95; // r15
  __m128i v96; // xmm0
  void (__fastcall *v97)(_QWORD *, _QWORD *, int); // rax
  __int64 v98; // rdi
  unsigned __int64 v99; // rax
  char v100; // al
  __int64 v101; // r8
  __int64 v102; // r8
  _QWORD *v103; // rax
  _QWORD *v104; // r12
  __int64 v105; // rdi
  __int64 v106; // r13
  __int64 v107; // rbx
  volatile signed __int32 *v108; // r14
  signed __int32 v109; // eax
  void (*v110)(); // rax
  signed __int32 v111; // eax
  __int64 (__fastcall *v112)(__int64); // rdx
  void (__fastcall *v113)(__m128i *, __int64, __int64); // rax
  void (__fastcall *v114)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v115)(char **, __int64, __int64); // rax
  size_t v116; // rdx
  _QWORD *v117; // [rsp+10h] [rbp-590h]
  __int64 v118; // [rsp+18h] [rbp-588h]
  __m128i *v119; // [rsp+28h] [rbp-578h]
  size_t v120; // [rsp+30h] [rbp-570h]
  _QWORD *v121; // [rsp+30h] [rbp-570h]
  __m128i *v122; // [rsp+38h] [rbp-568h]
  __int64 v123; // [rsp+38h] [rbp-568h]
  char v124; // [rsp+4Ch] [rbp-554h]
  _BYTE *v129; // [rsp+78h] [rbp-528h]
  __int64 v130; // [rsp+78h] [rbp-528h]
  _QWORD *v131; // [rsp+78h] [rbp-528h]
  __int64 v132; // [rsp+B8h] [rbp-4E8h] BYREF
  void *dest; // [rsp+C0h] [rbp-4E0h]
  size_t v134; // [rsp+C8h] [rbp-4D8h]
  _QWORD v135[2]; // [rsp+D0h] [rbp-4D0h] BYREF
  __m128i v136; // [rsp+E0h] [rbp-4C0h] BYREF
  __m128i v137; // [rsp+F0h] [rbp-4B0h]
  __m128i v138; // [rsp+100h] [rbp-4A0h] BYREF
  void (__fastcall *v139)(__int64 *, __m128i *, __int64); // [rsp+110h] [rbp-490h]
  __int64 (__fastcall *v140)(__int64 *, int *, unsigned int *); // [rsp+118h] [rbp-488h]
  _BYTE v141[16]; // [rsp+120h] [rbp-480h] BYREF
  void (__fastcall *v142)(_QWORD **, _BYTE *, __int64); // [rsp+130h] [rbp-470h]
  __int64 v143; // [rsp+138h] [rbp-468h]
  char v144; // [rsp+140h] [rbp-460h]
  __m128i v145; // [rsp+150h] [rbp-450h] BYREF
  __m128i v146; // [rsp+160h] [rbp-440h] BYREF
  __int64 v147; // [rsp+170h] [rbp-430h]
  char *v148; // [rsp+178h] [rbp-428h]
  char *v149; // [rsp+180h] [rbp-420h]
  __int64 (__fastcall *v150)(__int64 *, int *, unsigned int *); // [rsp+188h] [rbp-418h]
  _BYTE *v151; // [rsp+190h] [rbp-410h] BYREF
  __int64 v152; // [rsp+198h] [rbp-408h]
  _BYTE v153[256]; // [rsp+1A0h] [rbp-400h] BYREF
  __int64 v154; // [rsp+2A0h] [rbp-300h]
  __m128i v155; // [rsp+2B0h] [rbp-2F0h] BYREF
  __m128i v156; // [rsp+2C0h] [rbp-2E0h] BYREF
  __int64 v157; // [rsp+2D0h] [rbp-2D0h]
  char *v158; // [rsp+2D8h] [rbp-2C8h] BYREF
  char *v159; // [rsp+2E0h] [rbp-2C0h]
  __int64 (__fastcall *v160)(__int64 *, int *, unsigned int *); // [rsp+2E8h] [rbp-2B8h]
  _BYTE *v161; // [rsp+2F0h] [rbp-2B0h] BYREF
  __int64 v162; // [rsp+2F8h] [rbp-2A8h]
  _BYTE v163[16]; // [rsp+300h] [rbp-2A0h] BYREF
  void (__fastcall *v164)(_QWORD **, _BYTE *, __int64); // [rsp+310h] [rbp-290h]
  __int64 v165; // [rsp+318h] [rbp-288h]
  char v166; // [rsp+320h] [rbp-280h]
  __int64 v167; // [rsp+400h] [rbp-1A0h]
  size_t n[2]; // [rsp+410h] [rbp-190h] BYREF
  __m128i src; // [rsp+420h] [rbp-180h] BYREF
  __int64 v170; // [rsp+430h] [rbp-170h] BYREF
  char *v171; // [rsp+438h] [rbp-168h] BYREF
  void (__fastcall *v172)(__int64 *, __m128i *, __int64); // [rsp+440h] [rbp-160h]
  __int64 (__fastcall *v173)(__int64 *, int *, unsigned int *); // [rsp+448h] [rbp-158h]
  _QWORD *v174; // [rsp+450h] [rbp-150h] BYREF
  __int64 v175; // [rsp+458h] [rbp-148h]
  _QWORD v176[2]; // [rsp+460h] [rbp-140h] BYREF
  void (__fastcall *v177)(_QWORD *, _QWORD *, __int64); // [rsp+470h] [rbp-130h]
  __int64 v178; // [rsp+478h] [rbp-128h]
  char v179; // [rsp+480h] [rbp-120h]
  __int64 v180; // [rsp+560h] [rbp-40h]

  v7 = a2;
  v8 = *a2;
  v145.m128i_i64[1] = a2[1];
  v147 = 0x200000000LL;
  v145.m128i_i64[0] = v8;
  v9 = a2[6];
  v151 = v153;
  v152 = 0x800000000LL;
  v124 = a4;
  v146 = 0u;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v154 = 0;
  dest = v135;
  v134 = 0;
  LOBYTE(v135[0]) = 0;
  if ( v9 == -1 )
    goto LABEL_54;
  sub_9CDFE0((__int64 *)n, (__int64)&v145, v9, a4);
  v10 = n[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_3:
    v11 = a1;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v10;
    goto LABEL_4;
  }
  sub_9CF510((__m128i *)n, (__int64)&v145);
  v31 = n[0];
  v32 = v170 & 1;
  a4 = (unsigned int)(2 * v32);
  LOBYTE(v170) = (2 * v32) | v170 & 0xFD;
  if ( (_BYTE)v32 )
    goto LABEL_52;
  p_src = (__m128i *)dest;
  v60 = n[1];
  if ( (__m128i *)n[0] == &src )
  {
    if ( n[1] )
    {
      if ( n[1] == 1 )
        *(_BYTE *)dest = src.m128i_i8[0];
      else
        memcpy(dest, &src, n[1]);
      v60 = n[1];
      p_src = (__m128i *)dest;
    }
    v134 = v60;
    p_src->m128i_i8[v60] = 0;
    p_src = (__m128i *)n[0];
  }
  else
  {
    if ( dest == v135 )
    {
      dest = (void *)n[0];
      v134 = n[1];
      v135[0] = src.m128i_i64[0];
    }
    else
    {
      v61 = v135[0];
      dest = (void *)n[0];
      v134 = n[1];
      v135[0] = src.m128i_i64[0];
      if ( p_src )
      {
        n[0] = (size_t)p_src;
        src.m128i_i64[0] = v61;
        goto LABEL_83;
      }
    }
    n[0] = (size_t)&src;
    p_src = &src;
  }
LABEL_83:
  n[1] = 0;
  p_src->m128i_i8[0] = 0;
  v62 = v170;
  v31 = n[0];
  LOBYTE(v170) = v170 & 0xFD;
  if ( (v62 & 1) != 0 )
  {
LABEL_52:
    v155.m128i_i64[0] = v31 | 1;
    goto LABEL_53;
  }
  v155.m128i_i64[0] = 1;
  if ( (__m128i *)n[0] != &src )
    j_j___libc_free_0(n[0], src.m128i_i64[0] + 1);
LABEL_53:
  v10 = v155.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v155.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_3;
LABEL_54:
  v33 = &v145;
  sub_9CDFE0((__int64 *)n, (__int64)&v145, v7[7], a4);
  v10 = n[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_3;
  v34 = _mm_loadu_si128(&v145);
  v35 = _mm_loadu_si128(&v146);
  v157 = v147;
  v158 = v148;
  v148 = 0;
  v159 = v149;
  v149 = 0;
  v160 = v150;
  v161 = v163;
  v162 = 0x800000000LL;
  v150 = 0;
  v155 = v34;
  v156 = v35;
  if ( (_DWORD)v152 )
  {
    v33 = (__m128i *)&v151;
    sub_9D06B0((__int64)&v161, (__int64)&v151);
  }
  v36 = v7[4];
  v167 = v154;
  v130 = v7[5];
  v122 = (__m128i *)dest;
  v120 = v134;
  v37 = sub_22077B0(2032);
  v38 = v37;
  if ( !v37 )
    goto LABEL_100;
  v39 = v37 + 8;
  v40 = _mm_loadu_si128(&v155);
  v41 = _mm_loadu_si128(&v156);
  v42 = v158;
  v158 = 0;
  v43 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v159;
  v170 = v157;
  v44 = v160;
  v45 = HIDWORD(v157);
  v174 = v176;
  v175 = 0x800000000LL;
  v46 = v162;
  v171 = v42;
  v172 = v159;
  v173 = v160;
  v160 = 0;
  v159 = 0;
  *(__m128i *)n = v40;
  src = v41;
  if ( (_DWORD)v162 )
  {
    sub_9D06B0((__int64)&v174, (__int64)&v161);
    v45 = HIDWORD(v170);
    v42 = v171;
    v43 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v172;
    v44 = v173;
    v46 = v175;
  }
  v47 = v167;
  v48 = _mm_loadu_si128((const __m128i *)n);
  *(_QWORD *)(v38 + 8) = 0;
  v49 = _mm_loadu_si128(&src);
  *(_QWORD *)(v38 + 16) = 0;
  v180 = v47;
  LODWORD(v47) = v170;
  *(_QWORD *)(v38 + 24) = 0;
  *(_QWORD *)(v38 + 72) = v42;
  v50 = 0x800000000LL;
  *(_QWORD *)(v38 + 88) = v44;
  *(_DWORD *)(v38 + 64) = v47;
  *(_DWORD *)(v38 + 68) = v45;
  *(_QWORD *)(v38 + 80) = v43;
  v173 = 0;
  v172 = 0;
  v171 = 0;
  *(_QWORD *)(v38 + 96) = v38 + 112;
  *(_QWORD *)(v38 + 104) = 0x800000000LL;
  *(__m128i *)(v38 + 32) = v48;
  *(__m128i *)(v38 + 48) = v49;
  if ( !v46 )
  {
    *(_QWORD *)(v38 + 376) = v36;
    *(_BYTE *)(v38 + 392) = 0;
    *(_QWORD *)(v38 + 384) = v130;
    v119 = (__m128i *)(v38 + 416);
    *(_QWORD *)(v38 + 400) = v38 + 416;
    *(_QWORD *)(v38 + 408) = 0;
    *(_BYTE *)(v38 + 416) = 0;
    *(_QWORD *)(v38 + 368) = v39;
    v51 = v174;
    goto LABEL_62;
  }
  v50 = (__int64)&v174;
  sub_9D06B0(v38 + 96, (__int64)&v174);
  v101 = (unsigned int)v175;
  *(_QWORD *)(v38 + 368) = v39;
  *(_QWORD *)(v38 + 376) = v36;
  *(_QWORD *)(v38 + 384) = v130;
  v102 = 4 * v101;
  v119 = (__m128i *)(v38 + 416);
  *(_QWORD *)(v38 + 400) = v38 + 416;
  v103 = v174;
  *(_BYTE *)(v38 + 392) = 0;
  v51 = &v103[v102];
  *(_BYTE *)(v38 + 416) = 0;
  *(_QWORD *)(v38 + 408) = 0;
  v131 = v103;
  if ( v103 != &v103[v102] )
  {
    v118 = v38;
    v117 = v7;
    v104 = &v103[v102];
    while ( 1 )
    {
      v105 = *(v104 - 3);
      v106 = *(v104 - 2);
      v104 -= 4;
      v107 = v105;
      if ( v106 == v105 )
        goto LABEL_196;
      do
      {
        while ( 1 )
        {
          v108 = *(volatile signed __int32 **)(v107 + 8);
          if ( !v108 )
            goto LABEL_183;
          if ( &_pthread_key_create )
          {
            v109 = _InterlockedExchangeAdd(v108 + 2, 0xFFFFFFFF);
          }
          else
          {
            v109 = *((_DWORD *)v108 + 2);
            *((_DWORD *)v108 + 2) = v109 - 1;
          }
          if ( v109 != 1 )
            goto LABEL_183;
          v110 = *(void (**)())(*(_QWORD *)v108 + 16LL);
          if ( v110 != nullsub_25 )
            ((void (__fastcall *)(volatile signed __int32 *))v110)(v108);
          if ( &_pthread_key_create )
          {
            v111 = _InterlockedExchangeAdd(v108 + 3, 0xFFFFFFFF);
          }
          else
          {
            v111 = *((_DWORD *)v108 + 3);
            *((_DWORD *)v108 + 3) = v111 - 1;
          }
          if ( v111 != 1 )
            goto LABEL_183;
          v112 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v108 + 24LL);
          if ( v112 == sub_9C26E0 )
            break;
          v112((__int64)v108);
LABEL_183:
          v107 += 16;
          if ( v106 == v107 )
            goto LABEL_195;
        }
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v108 + 8LL))(v108);
        v107 += 16;
      }
      while ( v106 != v107 );
LABEL_195:
      v105 = v104[1];
LABEL_196:
      if ( v105 )
      {
        v50 = v104[3] - v105;
        j_j___libc_free_0(v105, v50);
      }
      if ( v131 == v104 )
      {
        v38 = v118;
        v7 = v117;
        v51 = v174;
        break;
      }
    }
  }
LABEL_62:
  if ( v51 != v176 )
    _libc_free(v51, v50);
  v52 = v171;
  if ( v172 != v171 )
  {
    v53 = v172;
    do
    {
      v54 = (volatile signed __int32 *)*((_QWORD *)v52 + 1);
      if ( v54 )
      {
        if ( &_pthread_key_create )
        {
          v55 = _InterlockedExchangeAdd(v54 + 2, 0xFFFFFFFF);
        }
        else
        {
          v55 = *((_DWORD *)v54 + 2);
          *((_DWORD *)v54 + 2) = v55 - 1;
        }
        if ( v55 == 1 )
        {
          v56 = *(void (**)(void))(*(_QWORD *)v54 + 16LL);
          if ( v56 != nullsub_25 )
            v56();
          if ( &_pthread_key_create )
          {
            v57 = _InterlockedExchangeAdd(v54 + 3, 0xFFFFFFFF);
          }
          else
          {
            v57 = *((_DWORD *)v54 + 3);
            *((_DWORD *)v54 + 3) = v57 - 1;
          }
          if ( v57 == 1 )
          {
            v58 = *(void (**)(void))(*(_QWORD *)v54 + 24LL);
            if ( (char *)v58 == (char *)sub_9C26E0 )
              (*(void (**)(void))(*(_QWORD *)v54 + 8LL))();
            else
              v58();
          }
        }
      }
      v52 += 16;
    }
    while ( v53 != v52 );
    v52 = v171;
  }
  if ( v52 )
    j_j___libc_free_0(v52, (char *)v173 - (char *)v52);
  *(_QWORD *)(v38 + 440) = 0;
  *(_QWORD *)v38 = off_49793E0;
  *(_QWORD *)(v38 + 448) = 0;
  *(_QWORD *)(v38 + 432) = a3;
  *(_QWORD *)(v38 + 664) = v38 + 680;
  *(_QWORD *)(v38 + 456) = 0;
  *(_BYTE *)(v38 + 464) = 0;
  *(_QWORD *)(v38 + 472) = 0;
  *(_QWORD *)(v38 + 480) = 0;
  *(_QWORD *)(v38 + 488) = 0;
  *(_QWORD *)(v38 + 496) = 0;
  *(_QWORD *)(v38 + 504) = 0;
  *(_QWORD *)(v38 + 512) = 0;
  *(_QWORD *)(v38 + 520) = 0;
  *(_QWORD *)(v38 + 528) = 0;
  *(_QWORD *)(v38 + 536) = 0;
  *(_QWORD *)(v38 + 544) = 0;
  *(_QWORD *)(v38 + 552) = 0;
  *(_QWORD *)(v38 + 560) = 0;
  *(_QWORD *)(v38 + 568) = 0;
  *(_DWORD *)(v38 + 576) = 0;
  *(_QWORD *)(v38 + 584) = 0;
  *(_QWORD *)(v38 + 592) = 0;
  *(_QWORD *)(v38 + 600) = 0;
  *(_DWORD *)(v38 + 608) = 0;
  *(_QWORD *)(v38 + 616) = 0;
  *(_QWORD *)(v38 + 624) = 0;
  *(_QWORD *)(v38 + 632) = 0;
  *(_DWORD *)(v38 + 640) = 0;
  *(_QWORD *)(v38 + 648) = 0;
  *(_QWORD *)(v38 + 656) = 0;
  *(_QWORD *)(v38 + 672) = 0x400000000LL;
  *(_QWORD *)(v38 + 712) = v38 + 728;
  src.m128i_i64[1] = (__int64)sub_9DDA20;
  src.m128i_i64[0] = (__int64)sub_9C2A00;
  v63 = *(_QWORD *)(v38 + 40);
  *(_QWORD *)(v38 + 720) = 0;
  *(_QWORD *)(v38 + 728) = 0;
  if ( v63 > 0xFFFFFFFE )
    LODWORD(v63) = -1;
  *(_QWORD *)(v38 + 736) = 1;
  *(_QWORD *)(v38 + 744) = 0;
  *(_QWORD *)(v38 + 752) = 0;
  *(_QWORD *)(v38 + 760) = 0;
  *(_DWORD *)(v38 + 768) = v63;
  *(_QWORD *)(v38 + 792) = 0;
  n[0] = v38;
  sub_9C2A00((_QWORD *)(v38 + 776), n, 2);
  *(_QWORD *)(v38 + 800) = src.m128i_i64[1];
  v64 = (void (__fastcall *)(size_t *, size_t *, __int64))src.m128i_i64[0];
  *(_QWORD *)(v38 + 792) = src.m128i_i64[0];
  if ( v64 )
    v64(n, n, 3);
  *(_BYTE *)(v38 + 816) = 0;
  *(_QWORD *)(v38 + 880) = v38 + 896;
  *(_QWORD *)(v38 + 888) = 0x4000000000LL;
  *(_QWORD *)(v38 + 1528) = v38 + 1512;
  *(_QWORD *)(v38 + 1536) = v38 + 1512;
  *(_QWORD *)(v38 + 824) = 0;
  *(_QWORD *)(v38 + 832) = 0;
  *(_QWORD *)(v38 + 840) = 0;
  *(_QWORD *)(v38 + 848) = 0;
  *(_QWORD *)(v38 + 856) = 0;
  *(_QWORD *)(v38 + 864) = 0;
  *(_DWORD *)(v38 + 872) = 0;
  *(_QWORD *)(v38 + 1408) = 0;
  *(_QWORD *)(v38 + 1416) = 0;
  *(_QWORD *)(v38 + 1424) = 0;
  *(_QWORD *)(v38 + 1432) = 0;
  *(_QWORD *)(v38 + 1440) = 0;
  *(_QWORD *)(v38 + 1448) = 0;
  *(_QWORD *)(v38 + 1456) = 0;
  *(_QWORD *)(v38 + 1464) = 0;
  *(_QWORD *)(v38 + 1472) = 0;
  *(_QWORD *)(v38 + 1480) = 0;
  *(_QWORD *)(v38 + 1488) = 0;
  *(_QWORD *)(v38 + 1496) = 0;
  *(_DWORD *)(v38 + 1512) = 0;
  *(_QWORD *)(v38 + 1520) = 0;
  *(_QWORD *)(v38 + 1544) = 0;
  *(_QWORD *)(v38 + 1552) = 0;
  *(_QWORD *)(v38 + 1560) = 0;
  *(_QWORD *)(v38 + 1568) = 0;
  *(_QWORD *)(v38 + 1576) = 0;
  *(_QWORD *)(v38 + 1584) = 0;
  *(_QWORD *)(v38 + 1592) = 0;
  *(_QWORD *)(v38 + 1600) = 0;
  *(_QWORD *)(v38 + 1608) = 0;
  *(_QWORD *)(v38 + 1616) = 0;
  *(_DWORD *)(v38 + 1624) = 0;
  *(_BYTE *)(v38 + 1632) = 0;
  *(_QWORD *)(v38 + 1640) = 0;
  *(_QWORD *)(v38 + 1648) = 0;
  *(_QWORD *)(v38 + 1656) = 0;
  *(_DWORD *)(v38 + 1664) = 0;
  *(_QWORD *)(v38 + 1672) = 0;
  *(_QWORD *)(v38 + 1680) = 0;
  *(_QWORD *)(v38 + 1688) = 0;
  *(_QWORD *)(v38 + 1696) = 0;
  *(_QWORD *)(v38 + 1704) = 0;
  *(_QWORD *)(v38 + 1712) = 0;
  *(_DWORD *)(v38 + 1720) = 0;
  *(_QWORD *)(v38 + 1728) = 0;
  *(_QWORD *)(v38 + 1744) = 0;
  *(_QWORD *)(v38 + 1752) = 0;
  *(_QWORD *)(v38 + 1760) = 0;
  *(_QWORD *)(v38 + 1768) = 0;
  *(_QWORD *)(v38 + 1776) = 0;
  *(_QWORD *)(v38 + 1784) = 0;
  *(_QWORD *)(v38 + 1792) = 0;
  *(_QWORD *)(v38 + 1800) = 0;
  *(_QWORD *)(v38 + 1736) = 8;
  v65 = sub_22077B0(64);
  v66 = *(_QWORD *)(v38 + 1736);
  *(_QWORD *)(v38 + 1728) = v65;
  v67 = (__int64 *)(v65 + ((4 * v66 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v68 = sub_22077B0(512);
  *(_QWORD *)(v38 + 1768) = v67;
  *v67 = v68;
  *(_QWORD *)(v38 + 1752) = v68;
  *(_QWORD *)(v38 + 1784) = v68;
  *(_QWORD *)(v38 + 1744) = v68;
  *(_QWORD *)(v38 + 1776) = v68;
  *(_QWORD *)(v38 + 1936) = v38 + 1960;
  *(_QWORD *)(v38 + 1760) = v68 + 512;
  *(_QWORD *)(v38 + 1792) = v68 + 512;
  *(_QWORD *)(v38 + 1800) = v67;
  *(_QWORD *)(v38 + 1808) = 0;
  *(_QWORD *)(v38 + 1816) = 0;
  *(_QWORD *)(v38 + 1824) = 0;
  *(_DWORD *)(v38 + 1832) = 0;
  *(_BYTE *)(v38 + 1836) = 0;
  *(_QWORD *)(v38 + 1840) = 0;
  *(_QWORD *)(v38 + 1848) = 0;
  *(_QWORD *)(v38 + 1856) = 0;
  *(_QWORD *)(v38 + 1864) = 0;
  *(_DWORD *)(v38 + 1872) = 0;
  *(_QWORD *)(v38 + 1880) = 0;
  *(_QWORD *)(v38 + 1888) = 0;
  *(_QWORD *)(v38 + 1896) = 0;
  *(_DWORD *)(v38 + 1904) = 0;
  *(_QWORD *)(v38 + 1912) = 0;
  *(_QWORD *)(v38 + 1920) = 0;
  *(_QWORD *)(v38 + 1928) = 0;
  *(_QWORD *)(v38 + 1944) = 0;
  *(_QWORD *)(v38 + 1952) = 8;
  *(_BYTE *)(v38 + 2000) = 0;
  *(_QWORD *)(v38 + 2008) = 0;
  *(_QWORD *)(v38 + 2016) = 0;
  *(_QWORD *)(v38 + 2024) = 0;
  v33 = v122;
  n[0] = (size_t)&src;
  sub_9C2D70((__int64 *)n, v122, (__int64)v122->m128i_i64 + v120);
  v69 = *(__m128i **)(v38 + 400);
  if ( (__m128i *)n[0] == &src )
  {
    v116 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
      {
        v69->m128i_i8[0] = src.m128i_i8[0];
      }
      else
      {
        v33 = &src;
        memcpy(v69, &src, n[1]);
      }
      v116 = n[1];
      v69 = *(__m128i **)(v38 + 400);
    }
    *(_QWORD *)(v38 + 408) = v116;
    v69->m128i_i8[v116] = 0;
    v69 = (__m128i *)n[0];
    goto LABEL_98;
  }
  v33 = (__m128i *)n[1];
  v70 = src.m128i_i64[0];
  if ( v69 == v119 )
  {
    *(_QWORD *)(v38 + 400) = n[0];
    *(_QWORD *)(v38 + 408) = v33;
    *(_QWORD *)(v38 + 416) = v70;
  }
  else
  {
    v71 = *(_QWORD *)(v38 + 416);
    *(_QWORD *)(v38 + 400) = n[0];
    *(_QWORD *)(v38 + 408) = v33;
    *(_QWORD *)(v38 + 416) = v70;
    if ( v69 )
    {
      n[0] = (size_t)v69;
      src.m128i_i64[0] = v71;
      goto LABEL_98;
    }
  }
  n[0] = (size_t)&src;
  v69 = &src;
LABEL_98:
  n[1] = 0;
  v69->m128i_i8[0] = 0;
  if ( (__m128i *)n[0] != &src )
  {
    v33 = (__m128i *)(src.m128i_i64[0] + 1);
    j_j___libc_free_0(n[0], src.m128i_i64[0] + 1);
  }
LABEL_100:
  v72 = 32LL * (unsigned int)v162;
  v73 = &v161[v72];
  if ( v161 == &v161[v72] )
    goto LABEL_121;
  v123 = v38;
  v121 = v7;
  v74 = &v161[v72];
  v75 = v161;
  do
  {
    v76 = *((_QWORD *)v74 - 3);
    v77 = *((_QWORD *)v74 - 2);
    v74 -= 32;
    v78 = v76;
    if ( v77 == v76 )
      goto LABEL_117;
    do
    {
      while ( 1 )
      {
        v79 = *(volatile signed __int32 **)(v78 + 8);
        if ( !v79 )
          goto LABEL_104;
        if ( &_pthread_key_create )
        {
          v80 = _InterlockedExchangeAdd(v79 + 2, 0xFFFFFFFF);
        }
        else
        {
          v80 = *((_DWORD *)v79 + 2);
          *((_DWORD *)v79 + 2) = v80 - 1;
        }
        if ( v80 != 1 )
          goto LABEL_104;
        v81 = *(void (**)())(*(_QWORD *)v79 + 16LL);
        if ( v81 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v81)(v79);
        if ( &_pthread_key_create )
        {
          v82 = _InterlockedExchangeAdd(v79 + 3, 0xFFFFFFFF);
        }
        else
        {
          v82 = *((_DWORD *)v79 + 3);
          *((_DWORD *)v79 + 3) = v82 - 1;
        }
        if ( v82 != 1 )
          goto LABEL_104;
        v83 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v79 + 24LL);
        if ( v83 == sub_9C26E0 )
          break;
        v83((__int64)v79);
LABEL_104:
        v78 += 16;
        if ( v77 == v78 )
          goto LABEL_116;
      }
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v79 + 8LL))(v79);
      v78 += 16;
    }
    while ( v77 != v78 );
LABEL_116:
    v76 = *((_QWORD *)v74 + 1);
LABEL_117:
    if ( v76 )
    {
      v33 = (__m128i *)(*((_QWORD *)v74 + 3) - v76);
      j_j___libc_free_0(v76, v33);
    }
  }
  while ( v75 != v74 );
  v38 = v123;
  v7 = v121;
  v73 = v161;
LABEL_121:
  if ( v73 != v163 )
    _libc_free(v73, v33);
  v84 = v158;
  if ( v159 != v158 )
  {
    v85 = v159;
    do
    {
      v86 = (volatile signed __int32 *)*((_QWORD *)v84 + 1);
      if ( v86 )
      {
        if ( &_pthread_key_create )
        {
          v87 = _InterlockedExchangeAdd(v86 + 2, 0xFFFFFFFF);
        }
        else
        {
          v87 = *((_DWORD *)v86 + 2);
          *((_DWORD *)v86 + 2) = v87 - 1;
        }
        if ( v87 == 1 )
        {
          v88 = *(void (**)(void))(*(_QWORD *)v86 + 16LL);
          if ( v88 != nullsub_25 )
            v88();
          if ( &_pthread_key_create )
          {
            v89 = _InterlockedExchangeAdd(v86 + 3, 0xFFFFFFFF);
          }
          else
          {
            v89 = *((_DWORD *)v86 + 3);
            *((_DWORD *)v86 + 3) = v89 - 1;
          }
          if ( v89 == 1 )
          {
            v90 = *(void (**)(void))(*(_QWORD *)v86 + 24LL);
            if ( (char *)v90 == (char *)sub_9C26E0 )
              (*(void (**)(void))(*(_QWORD *)v86 + 8LL))();
            else
              v90();
          }
        }
      }
      v84 += 16;
    }
    while ( v85 != v84 );
    v84 = v158;
  }
  if ( v84 )
    j_j___libc_free_0(v84, (char *)v160 - (char *)v84);
  v91 = v7[3];
  v92 = v7[2];
  v93 = sub_22077B0(880);
  v94 = v93;
  if ( v93 )
    sub_BA8740(v93, v92, v91, a3);
  sub_BA9760(v94, v38);
  LOBYTE(v157) = 0;
  if ( *(_BYTE *)(a7 + 32) )
  {
    v156.m128i_i64[0] = 0;
    v113 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a7 + 16);
    if ( v113 )
    {
      v113(&v155, a7, 2);
      v156 = *(__m128i *)(a7 + 16);
    }
    LOBYTE(v157) = 1;
  }
  LOBYTE(v162) = 0;
  if ( *(_BYTE *)(a7 + 72) )
  {
    v160 = 0;
    v115 = *(void (__fastcall **)(char **, __int64, __int64))(a7 + 56);
    if ( v115 )
    {
      v115(&v158, a7 + 40, 2);
      v161 = *(_BYTE **)(a7 + 64);
      v160 = *(__int64 (__fastcall **)(__int64 *, int *, unsigned int *))(a7 + 56);
    }
    LOBYTE(v162) = 1;
  }
  v166 = 0;
  v95 = *(_BYTE *)(a7 + 112);
  if ( v95 )
  {
    v164 = 0;
    v114 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a7 + 96);
    if ( v114 )
    {
      v114(v163, a7 + 80, 2);
      v165 = *(_QWORD *)(a7 + 104);
      v164 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a7 + 96);
    }
    v166 = 1;
  }
  n[0] = v38;
  v137.m128i_i64[1] = (__int64)sub_9CB930;
  v96 = _mm_loadu_si128((const __m128i *)n);
  v97 = (void (__fastcall *)(_QWORD *, _QWORD *, int))sub_9C2A30;
  v139 = (void (__fastcall *)(__int64 *, __m128i *, __int64))sub_9C2A60;
  *(_QWORD *)(v38 + 440) = v94;
  v144 = 0;
  v137.m128i_i64[0] = (__int64)sub_9C2A30;
  v140 = sub_9C2B40;
  v136 = v96;
  v138 = v96;
  if ( !v95 )
  {
LABEL_149:
    src.m128i_i64[0] = 0;
    goto LABEL_150;
  }
  v142 = 0;
  if ( !v164 )
  {
    v144 = 1;
    goto LABEL_149;
  }
  v164(v141, v163, 2);
  v144 = 1;
  src.m128i_i64[0] = 0;
  v143 = v165;
  v142 = (void (__fastcall *)(_QWORD **, _BYTE *, __int64))v164;
  v97 = (void (__fastcall *)(_QWORD *, _QWORD *, int))v137.m128i_i64[0];
  if ( v137.m128i_i64[0] )
  {
LABEL_150:
    v97(n, &v136, 2);
    src = v137;
  }
  v172 = 0;
  if ( v139 )
  {
    v139(&v170, &v138, 2);
    v173 = v140;
    v172 = (char *)v139;
  }
  LOBYTE(v177) = 0;
  if ( v144 )
  {
    v176[0] = 0;
    if ( v142 )
    {
      v142(&v174, v141, 2);
      v176[1] = v143;
      v176[0] = v142;
    }
    LOBYTE(v177) = 1;
  }
  sub_A03720(&v132, v38 + 32, v94, v38 + 744, a6, n);
  v98 = v38 + 808;
  if ( *(_BYTE *)(v38 + 816) )
  {
    sub_A04970(v98, &v132);
  }
  else
  {
    sub_A03710(v98, &v132);
    *(_BYTE *)(v38 + 816) = 1;
  }
  sub_A049B0(&v132);
  if ( (_BYTE)v177 )
  {
    LOBYTE(v177) = 0;
    if ( v176[0] )
      ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v176[0])(&v174, &v174, 3);
  }
  if ( v172 )
    ((void (__fastcall *)(__int64 *, __int64 *, __int64))v172)(&v170, &v170, 3);
  if ( src.m128i_i64[0] )
    ((void (__fastcall *)(size_t *, size_t *, __int64))src.m128i_i64[0])(n, n, 3);
  LOBYTE(v170) = 0;
  if ( (_BYTE)v157 )
  {
    src.m128i_i64[0] = 0;
    if ( v156.m128i_i64[0] )
    {
      ((void (__fastcall *)(size_t *, __m128i *, __int64))v156.m128i_i64[0])(n, &v155, 2);
      src = v156;
    }
    LOBYTE(v170) = 1;
  }
  LOBYTE(v175) = 0;
  if ( (_BYTE)v162 )
  {
    v173 = 0;
    if ( v160 )
    {
      v160((__int64 *)&v171, (int *)&v158, (unsigned int *)2);
      v174 = v161;
      v173 = v160;
    }
    LOBYTE(v175) = 1;
  }
  v179 = 0;
  if ( v166 )
  {
    v177 = 0;
    if ( v164 )
    {
      v164(v176, v163, 2);
      v178 = v165;
      v177 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v164;
    }
    v179 = 1;
  }
  sub_9E7B10(&v132, (const __m128i *)v38, 0, a5, (__int64)n);
  if ( v179 )
  {
    v179 = 0;
    if ( v177 )
      v177(v176, v176, 3);
  }
  if ( (_BYTE)v175 )
  {
    LOBYTE(v175) = 0;
    if ( v173 )
      v173((__int64 *)&v171, (int *)&v171, (unsigned int *)3);
  }
  if ( (_BYTE)v170 )
  {
    LOBYTE(v170) = 0;
    if ( src.m128i_i64[0] )
      ((void (__fastcall *)(size_t *, size_t *, __int64))src.m128i_i64[0])(n, n, 3);
  }
  if ( v144 )
  {
    v144 = 0;
    if ( v142 )
      v142((_QWORD **)v141, v141, 3);
  }
  if ( v139 )
    v139(v138.m128i_i64, &v138, 3);
  if ( v137.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v137.m128i_i64[0])(&v136, &v136, 3);
  if ( v166 )
  {
    v166 = 0;
    if ( v164 )
      v164(v163, v163, 3);
  }
  if ( (_BYTE)v162 )
  {
    LOBYTE(v162) = 0;
    if ( v160 )
      v160((__int64 *)&v158, (int *)&v158, (unsigned int *)3);
  }
  if ( (_BYTE)v157 )
  {
    LOBYTE(v157) = 0;
    if ( v156.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v156.m128i_i64[0])(&v155, &v155, 3);
  }
  v99 = v132 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v132 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_205;
  if ( v124 )
  {
    sub_BA97D0(n, v94);
    v99 = n[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_205;
LABEL_178:
    v11 = a1;
    v100 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = v94;
    *(_BYTE *)(a1 + 8) = v100 & 0xFC | 2;
  }
  else
  {
    sub_9FF010((__int64 *)n, v38);
    v99 = n[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      goto LABEL_178;
LABEL_205:
    v11 = a1;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v99;
    if ( v94 )
    {
      sub_BA9C10(v94);
      v11 = 880;
      j_j___libc_free_0(v94, 880);
    }
  }
LABEL_4:
  if ( dest != v135 )
  {
    v11 = v135[0] + 1LL;
    j_j___libc_free_0(dest, v135[0] + 1LL);
  }
  v12 = 32LL * (unsigned int)v152;
  v129 = v151;
  v13 = &v151[v12];
  if ( v151 != &v151[v12] )
  {
    v14 = &v151[v12];
    while ( 1 )
    {
      v15 = *((_QWORD *)v14 - 3);
      v16 = *((_QWORD *)v14 - 2);
      v14 -= 32;
      v17 = v15;
      if ( v16 != v15 )
        break;
LABEL_23:
      if ( v15 )
      {
        v11 = *((_QWORD *)v14 + 3) - v15;
        j_j___libc_free_0(v15, v11);
      }
      if ( v129 == v14 )
      {
        v13 = v151;
        goto LABEL_27;
      }
    }
    while ( 1 )
    {
LABEL_11:
      v18 = *(volatile signed __int32 **)(v17 + 8);
      if ( !v18 )
        goto LABEL_10;
      if ( &_pthread_key_create )
      {
        v19 = _InterlockedExchangeAdd(v18 + 2, 0xFFFFFFFF);
      }
      else
      {
        v19 = *((_DWORD *)v18 + 2);
        *((_DWORD *)v18 + 2) = v19 - 1;
      }
      if ( v19 != 1 )
        goto LABEL_10;
      v20 = *(void (**)())(*(_QWORD *)v18 + 16LL);
      if ( v20 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v20)(v18);
      if ( &_pthread_key_create )
      {
        v21 = _InterlockedExchangeAdd(v18 + 3, 0xFFFFFFFF);
      }
      else
      {
        v21 = *((_DWORD *)v18 + 3);
        *((_DWORD *)v18 + 3) = v21 - 1;
      }
      if ( v21 != 1 )
        goto LABEL_10;
      v22 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v18 + 24LL);
      if ( v22 != sub_9C26E0 )
        break;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 8LL))(v18);
      v17 += 16;
      if ( v16 == v17 )
      {
LABEL_22:
        v15 = *((_QWORD *)v14 + 1);
        goto LABEL_23;
      }
    }
    v22((__int64)v18);
LABEL_10:
    v17 += 16;
    if ( v16 == v17 )
      goto LABEL_22;
    goto LABEL_11;
  }
LABEL_27:
  if ( v13 != v153 )
    _libc_free(v13, v11);
  v23 = v149;
  v24 = v148;
  if ( v149 != v148 )
  {
    while ( 1 )
    {
      v25 = (volatile signed __int32 *)*((_QWORD *)v24 + 1);
      if ( !v25 )
        goto LABEL_31;
      if ( &_pthread_key_create )
      {
        v26 = _InterlockedExchangeAdd(v25 + 2, 0xFFFFFFFF);
      }
      else
      {
        v26 = *((_DWORD *)v25 + 2);
        *((_DWORD *)v25 + 2) = v26 - 1;
      }
      if ( v26 != 1 )
        goto LABEL_31;
      v27 = *(void (**)())(*(_QWORD *)v25 + 16LL);
      if ( v27 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v27)(v25);
      if ( &_pthread_key_create )
      {
        v28 = _InterlockedExchangeAdd(v25 + 3, 0xFFFFFFFF);
      }
      else
      {
        v28 = *((_DWORD *)v25 + 3);
        *((_DWORD *)v25 + 3) = v28 - 1;
      }
      if ( v28 != 1 )
        goto LABEL_31;
      v29 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 24LL);
      if ( v29 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v25 + 8LL))(v25);
        v24 += 16;
        if ( v23 == v24 )
        {
LABEL_43:
          v24 = v148;
          break;
        }
      }
      else
      {
        v29((__int64)v25);
LABEL_31:
        v24 += 16;
        if ( v23 == v24 )
          goto LABEL_43;
      }
    }
  }
  if ( v24 )
    j_j___libc_free_0(v24, (char *)v150 - (char *)v24);
  return a1;
}
