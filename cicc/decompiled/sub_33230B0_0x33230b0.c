// Function: sub_33230B0
// Address: 0x33230b0
//
void __fastcall sub_33230B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int32 a4)
{
  int v5; // ebx
  __int64 *v6; // rax
  __int64 v7; // r8
  __int64 (*v8)(); // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  char v13; // al
  __int64 (*v14)(); // rdx
  _QWORD *v15; // rcx
  __int64 v16; // rax
  const __m128i *v17; // rax
  bool v18; // zf
  bool v19; // sf
  __int64 v20; // rdx
  const __m128i *v21; // r14
  __int64 v22; // rbx
  __int64 (__fastcall ***v23)(); // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rax
  __m128i v28; // xmm0
  __int64 v29; // rax
  unsigned int v30; // r8d
  __int64 v31; // rsi
  unsigned int v32; // eax
  __int64 *v33; // rcx
  __int64 v34; // rdi
  int v35; // ecx
  int v36; // r10d
  __int64 v37; // rbx
  __int64 *v38; // r12
  __int64 v39; // r13
  int v40; // eax
  __int64 (__fastcall ***v41)(); // rdx
  size_t v42; // r15
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rcx
  __int64 v45; // r8
  unsigned __int64 v46; // r9
  __int64 *v47; // rdi
  __int64 v48; // rax
  __int64 *v49; // rbx
  __int64 *i; // r12
  __int64 v51; // r13
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rdx
  unsigned int v58; // eax
  __int64 v59; // rbx
  int v60; // eax
  __int64 v61; // r13
  __int64 v62; // r14
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  unsigned __int64 v66; // rcx
  unsigned __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r14
  __int64 v73; // r15
  __int64 *v74; // rsi
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rbx
  int v78; // r11d
  _QWORD *v79; // r10
  unsigned int v80; // edx
  _QWORD *v81; // rdi
  __int64 v82; // rcx
  int v83; // eax
  __int64 v84; // rcx
  __int64 v85; // rax
  __int64 v86; // r12
  unsigned __int64 v87; // rdx
  __int64 v88; // rax
  unsigned __int64 v89; // rcx
  unsigned __int64 v90; // rdx
  __int64 v91; // rax
  _QWORD *v92; // r12
  _QWORD *v93; // r13
  unsigned int v94; // eax
  _QWORD *v95; // rdi
  __int64 v96; // rcx
  unsigned int v97; // edx
  _QWORD *v98; // r10
  __int64 v99; // rdi
  int v100; // eax
  int v101; // r11d
  _QWORD *v102; // r11
  unsigned int v103; // edx
  __int64 v104; // rdi
  __int64 v105; // rax
  unsigned __int64 v106; // rcx
  unsigned __int64 v107; // rdx
  unsigned int v108; // edx
  __int64 v109; // rdi
  int v110; // r11d
  int v111; // r11d
  unsigned int v112; // edx
  __int64 v113; // rdi
  __int64 v114; // rax
  unsigned __int64 v115; // rcx
  unsigned __int64 v116; // rdx
  size_t v117; // [rsp+10h] [rbp-630h]
  char v118; // [rsp+27h] [rbp-619h]
  _BYTE *v119; // [rsp+50h] [rbp-5F0h]
  __m128i v120; // [rsp+60h] [rbp-5E0h] BYREF
  __int64 v121; // [rsp+70h] [rbp-5D0h]
  __int64 v122; // [rsp+78h] [rbp-5C8h]
  __int64 v123; // [rsp+80h] [rbp-5C0h]
  __int64 v124; // [rsp+88h] [rbp-5B8h]
  __m128i v125; // [rsp+90h] [rbp-5B0h]
  __int64 v126; // [rsp+A8h] [rbp-598h] BYREF
  __int64 (__fastcall **v127)(); // [rsp+B0h] [rbp-590h] BYREF
  __int64 v128; // [rsp+B8h] [rbp-588h]
  __int64 *v129; // [rsp+C0h] [rbp-580h]
  const __m128i **v130; // [rsp+C8h] [rbp-578h]
  __int64 (__fastcall **v131)(); // [rsp+D0h] [rbp-570h] BYREF
  __int64 v132; // [rsp+D8h] [rbp-568h]
  __int64 *v133; // [rsp+E0h] [rbp-560h]
  const __m128i **v134; // [rsp+E8h] [rbp-558h]
  _QWORD v135[8]; // [rsp+F0h] [rbp-550h] BYREF
  __int64 v136; // [rsp+130h] [rbp-510h]
  int v137; // [rsp+138h] [rbp-508h]
  __int64 v138; // [rsp+140h] [rbp-500h]
  __int64 v139; // [rsp+148h] [rbp-4F8h]
  __int64 v140; // [rsp+150h] [rbp-4F0h] BYREF
  __int64 v141; // [rsp+158h] [rbp-4E8h]
  _QWORD *v142; // [rsp+160h] [rbp-4E0h]
  __int64 v143; // [rsp+168h] [rbp-4D8h]
  __int64 v144; // [rsp+170h] [rbp-4D0h] BYREF
  __int64 v145; // [rsp+180h] [rbp-4C0h] BYREF
  __int64 v146; // [rsp+188h] [rbp-4B8h]
  __int64 v147; // [rsp+190h] [rbp-4B0h]
  __int64 v148; // [rsp+198h] [rbp-4A8h]
  _BYTE *v149; // [rsp+1A0h] [rbp-4A0h]
  __int64 v150; // [rsp+1A8h] [rbp-498h]
  _BYTE v151[128]; // [rsp+1B0h] [rbp-490h] BYREF
  const __m128i *v152; // [rsp+230h] [rbp-410h] BYREF
  __int64 *v153; // [rsp+238h] [rbp-408h]
  __int64 v154; // [rsp+240h] [rbp-400h]
  int v155; // [rsp+248h] [rbp-3F8h]
  unsigned __int32 v156; // [rsp+24Ch] [rbp-3F4h]
  __int16 v157; // [rsp+250h] [rbp-3F0h]
  bool v158; // [rsp+252h] [rbp-3EEh]
  char v159; // [rsp+253h] [rbp-3EDh]
  char v160; // [rsp+254h] [rbp-3ECh]
  __int64 (__fastcall ***v161)(); // [rsp+258h] [rbp-3E8h] BYREF
  __int64 v162; // [rsp+260h] [rbp-3E0h]
  _BYTE v163[512]; // [rsp+268h] [rbp-3D8h] BYREF
  __int64 v164; // [rsp+468h] [rbp-1D8h] BYREF
  __int64 v165; // [rsp+470h] [rbp-1D0h]
  __int64 v166; // [rsp+478h] [rbp-1C8h]
  __int64 v167; // [rsp+480h] [rbp-1C0h]
  _QWORD *v168; // [rsp+488h] [rbp-1B8h] BYREF
  __int64 v169; // [rsp+490h] [rbp-1B0h]
  _BYTE v170[256]; // [rsp+498h] [rbp-1A8h] BYREF
  __int64 v171; // [rsp+598h] [rbp-A8h]
  __int64 v172; // [rsp+5A0h] [rbp-A0h]
  __int64 v173; // [rsp+5A8h] [rbp-98h]
  unsigned int v174; // [rsp+5B0h] [rbp-90h]
  __int64 v175; // [rsp+5B8h] [rbp-88h]
  __int64 v176; // [rsp+5C0h] [rbp-80h] BYREF
  void *s; // [rsp+5C8h] [rbp-78h]
  _BYTE v178[12]; // [rsp+5D0h] [rbp-70h]
  char v179; // [rsp+5DCh] [rbp-64h]
  char v180; // [rsp+5E0h] [rbp-60h] BYREF
  unsigned int v181; // [rsp+600h] [rbp-40h]

  v5 = a2;
  v6 = *(__int64 **)(a1 + 16);
  v152 = (const __m128i *)a1;
  v153 = v6;
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL);
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 152LL);
  v9 = 0;
  if ( v8 != sub_30593D0 )
  {
    v120.m128i_i32[0] = a4;
    v9 = ((__int64 (__fastcall *)(__int64))v8)(v7);
    a4 = v120.m128i_i32[0];
  }
  v154 = v9;
  v161 = (__int64 (__fastcall ***)())v163;
  v162 = 0x4000000000LL;
  v168 = v170;
  v169 = 0x2000000000LL;
  v155 = 0;
  v156 = a4;
  v157 = 0;
  v158 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v175 = a3;
  v176 = 0;
  s = &v180;
  *(_QWORD *)v178 = 4;
  *(_DWORD *)&v178[8] = 0;
  v179 = 1;
  v10 = v9;
  v159 = sub_33CC5C0();
  v13 = 0;
  if ( v10 )
  {
    v14 = *(__int64 (**)())(*(_QWORD *)v154 + 120LL);
    if ( v14 != sub_325D4B0 )
    {
      a2 = v156;
      v13 = ((__int64 (__fastcall *)(__int64, _QWORD))v14)(v10, v156);
    }
  }
  v160 = v13;
  v15 = byte_444C4A0;
  v16 = 1;
  v181 = 0;
  do
  {
    while ( 1 )
    {
      if ( v16 != 1 )
      {
        a2 = (unsigned __int64)v153;
        if ( v153[(int)v16 + 14] )
        {
          a2 = v181;
          if ( *v15 >= (unsigned __int64)v181 )
            break;
        }
      }
      ++v16;
      v15 += 2;
      if ( v16 == 274 )
        goto LABEL_12;
    }
    ++v16;
    v181 = *v15;
    v15 += 2;
  }
  while ( v16 != 274 );
LABEL_12:
  if ( (_BYTE)qword_5037B48 )
  {
    v12 = v156;
    if ( !v156 )
      goto LABEL_42;
  }
  v17 = v152;
  v155 = v5;
  LOBYTE(v157) = v5 > 2;
  HIBYTE(v157) = v5 > 1;
  v18 = v5 == 0;
  v19 = v5 < 0;
  v20 = v152[48].m128i_i64[0];
  v127 = off_4A360F0;
  v21 = v152 + 25;
  v130 = &v152;
  v22 = v152[25].m128i_i64[1];
  v128 = v20;
  v23 = &v127;
  v129 = (__int64 *)v152;
  v152[48].m128i_i64[0] = (__int64)&v127;
  v158 = !v19 && !v18;
  if ( v21 == (const __m128i *)v22 )
    goto LABEL_26;
  do
  {
    while ( 1 )
    {
      if ( !v22 )
        BUG();
      if ( *(_DWORD *)(v22 + 16) == 328 )
        goto LABEL_17;
      if ( !*(_QWORD *)(v22 + 48) )
        break;
      v11 = *(unsigned int *)(v22 + 80);
      if ( (int)v11 < 0 )
        goto LABEL_22;
LABEL_17:
      v22 = *(_QWORD *)(v22 + 8);
      if ( v21 == (const __m128i *)v22 )
        goto LABEL_25;
    }
    a2 = (unsigned __int64)&v145;
    v145 = v22 - 8;
    sub_32B3B20((__int64)&v164, &v145);
    v11 = *(unsigned int *)(v22 + 80);
    if ( (int)v11 >= 0 )
      goto LABEL_17;
LABEL_22:
    v24 = (unsigned int)v162;
    v15 = (_QWORD *)HIDWORD(v162);
    v25 = (unsigned int)v162 + 1LL;
    *(_DWORD *)(v22 + 80) = v162;
    if ( v25 > (unsigned __int64)v15 )
    {
      a2 = (unsigned __int64)v163;
      sub_C8D5F0((__int64)&v161, v163, v25, 8u, v11, v12);
      v24 = (unsigned int)v162;
    }
    v23 = v161;
    v161[v24] = (__int64 (__fastcall **)())(v22 - 8);
    LODWORD(v162) = v162 + 1;
    v22 = *(_QWORD *)(v22 + 8);
  }
  while ( v21 != (const __m128i *)v22 );
LABEL_25:
  v17 = v152;
LABEL_26:
  v26 = v17[24].m128i_i64[0];
  v120 = _mm_loadu_si128(v17 + 24);
  v27 = sub_33ECD10(1, a2, v23, v15, v11, v12);
  v28 = _mm_load_si128(&v120);
  v144 = 0;
  v135[6] = v27;
  v136 = 0x100000000LL;
  v139 = 0xFFFFFFFFLL;
  v142 = v135;
  v125 = v28;
  v135[7] = 0;
  v137 = 0;
  v138 = 0;
  v143 = 0;
  LODWORD(v141) = v28.m128i_i32[2];
  v140 = v28.m128i_i64[0];
  v29 = *(_QWORD *)(v26 + 56);
  memset(v135, 0, 24);
  v135[3] = 328;
  v135[4] = -65536;
  v144 = v29;
  if ( v29 )
    *(_QWORD *)(v29 + 24) = &v144;
  v143 = v26 + 56;
  v135[5] = &v140;
  *(_QWORD *)(v26 + 56) = &v140;
  LODWORD(v136) = 1;
LABEL_29:
  v30 = v169;
  while ( v30 )
  {
    v31 = v168[v30 - 1];
    if ( (_DWORD)v167 )
    {
      v32 = (v167 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v33 = (__int64 *)(v165 + 8LL * v32);
      v34 = *v33;
      if ( v31 == *v33 )
      {
LABEL_30:
        *v33 = -8192;
        v30 = v169;
        LODWORD(v166) = v166 - 1;
        ++HIDWORD(v166);
      }
      else
      {
        v35 = 1;
        while ( v34 != -4096 )
        {
          v36 = v35 + 1;
          v32 = (v167 - 1) & (v35 + v32);
          v33 = (__int64 *)(v165 + 8LL * v32);
          v34 = *v33;
          if ( v31 == *v33 )
            goto LABEL_30;
          v35 = v36;
        }
      }
    }
    LODWORD(v169) = --v30;
    if ( !*(_QWORD *)(v31 + 56) )
    {
      sub_32CF870((__int64)&v152, v31);
      goto LABEL_29;
    }
  }
  v40 = v162;
  v41 = &v161[(unsigned int)v162];
  while ( v40 )
  {
    v42 = (size_t)*(v41 - 1);
    --v40;
    --v41;
    LODWORD(v162) = v40;
    if ( v42 )
    {
      *(_DWORD *)(v42 + 88) = -2;
      if ( (unsigned __int8)sub_32CF870((__int64)&v152, v42) )
        goto LABEL_29;
      v47 = (__int64 *)v152;
      v18 = (_BYTE)v157 == 0;
      v48 = v152[48].m128i_i64[0];
      v133 = (__int64 *)v152;
      v132 = v48;
      v152[48].m128i_i64[0] = (__int64)&v131;
      v131 = off_4A360B8;
      v134 = &v152;
      if ( v18 )
        goto LABEL_54;
      v145 = 0;
      v149 = v151;
      v146 = 0;
      v147 = 0;
      v148 = 0;
      v150 = 0x1000000000LL;
      v118 = sub_334C6A0(v47, v42, &v145);
      v119 = &v149[8 * (unsigned int)v150];
      if ( v149 == v119 )
        goto LABEL_91;
      v120.m128i_i64[0] = (__int64)v149;
      v117 = v42;
      while ( 2 )
      {
        v72 = *(_QWORD *)v120.m128i_i64[0];
        v73 = *(_QWORD *)(*(_QWORD *)v120.m128i_i64[0] + 56LL);
        if ( !v73 )
        {
LABEL_87:
          if ( *(_DWORD *)(v72 + 24) != 328 )
          {
            v126 = v72;
            sub_32B3B20((__int64)&v164, &v126);
            if ( *(int *)(v72 + 88) < 0 )
            {
              v105 = (unsigned int)v162;
              v106 = HIDWORD(v162);
              v107 = (unsigned int)v162 + 1LL;
              *(_DWORD *)(v72 + 88) = v162;
              if ( v107 > v106 )
              {
                sub_C8D5F0((__int64)&v161, v163, v107, 8u, v70, v71);
                v105 = (unsigned int)v162;
              }
              v161[v105] = (__int64 (__fastcall **)())v72;
              LODWORD(v162) = v162 + 1;
            }
          }
          v120.m128i_i64[0] += 8;
          if ( v119 != (_BYTE *)v120.m128i_i64[0] )
            continue;
          v42 = v117;
          v119 = v149;
LABEL_91:
          if ( !v118 )
          {
            if ( v119 != v151 )
              _libc_free((unsigned __int64)v119);
            sub_C7D6A0(v146, 8LL * (unsigned int)v148, 8);
            goto LABEL_95;
          }
          if ( v119 != v151 )
            _libc_free((unsigned __int64)v119);
          sub_C7D6A0(v146, 8LL * (unsigned int)v148, 8);
LABEL_54:
          v49 = *(__int64 **)(v42 + 40);
          for ( i = &v49[5 * *(unsigned int *)(v42 + 64)]; i != v49; LODWORD(v162) = v162 + 1 )
          {
            while ( 1 )
            {
              v51 = *v49;
              if ( *(_DWORD *)(*v49 + 24) != 328 && *(_DWORD *)(v51 + 88) != -2 )
              {
                v145 = *v49;
                sub_32B3B20((__int64)&v164, &v145);
                v44 = *(unsigned int *)(v51 + 88);
                if ( (v44 & 0x80000000) != 0LL )
                  break;
              }
              v49 += 5;
              if ( i == v49 )
                goto LABEL_63;
            }
            v52 = (unsigned int)v162;
            v44 = HIDWORD(v162);
            v53 = (unsigned int)v162 + 1LL;
            *(_DWORD *)(v51 + 88) = v162;
            if ( v53 > v44 )
            {
              sub_C8D5F0((__int64)&v161, v163, v53, 8u, v45, v46);
              v52 = (unsigned int)v162;
            }
            v43 = (unsigned __int64)v161;
            v49 += 5;
            v161[v52] = (__int64 (__fastcall **)())v51;
          }
LABEL_63:
          v54 = sub_3322160((__int64 *)&v152, v42, v43, v44, v45, v46);
          v56 = v55;
          v145 = v54;
          v57 = v54;
          v146 = v56;
          if ( v54 )
          {
            ++v176;
            if ( v179 )
              goto LABEL_69;
            v58 = 4 * (*(_DWORD *)&v178[4] - *(_DWORD *)&v178[8]);
            if ( v58 < 0x20 )
              v58 = 32;
            if ( *(_DWORD *)v178 > v58 )
            {
              sub_C8C990((__int64)&v176, v42);
              v57 = v145;
            }
            else
            {
              memset(s, -1, 8LL * *(unsigned int *)v178);
              v57 = v145;
LABEL_69:
              *(_QWORD *)&v178[4] = 0;
            }
            if ( v42 == v57 )
              goto LABEL_95;
            if ( *(_DWORD *)(v42 + 68) == *(_DWORD *)(v57 + 68) )
              sub_34158F0(v152, v42, v57);
            else
              sub_3415F70(v152, v42, &v145);
            v59 = v145;
            v60 = *(_DWORD *)(v145 + 24);
            if ( v60 != 1 )
            {
              v61 = *(_QWORD *)(v145 + 56);
              if ( v61 )
              {
                do
                {
                  while ( 1 )
                  {
                    v62 = *(_QWORD *)(v61 + 16);
                    if ( *(_DWORD *)(v62 + 24) != 328 )
                    {
                      v126 = *(_QWORD *)(v61 + 16);
                      sub_32B3B20((__int64)&v164, &v126);
                      if ( *(int *)(v62 + 88) < 0 )
                        break;
                    }
                    v61 = *(_QWORD *)(v61 + 32);
                    if ( !v61 )
                      goto LABEL_82;
                  }
                  v65 = (unsigned int)v162;
                  v66 = HIDWORD(v162);
                  v67 = (unsigned int)v162 + 1LL;
                  *(_DWORD *)(v62 + 88) = v162;
                  if ( v67 > v66 )
                  {
                    sub_C8D5F0((__int64)&v161, v163, v67, 8u, v63, v64);
                    v65 = (unsigned int)v162;
                  }
                  v161[v65] = (__int64 (__fastcall **)())v62;
                  LODWORD(v162) = v162 + 1;
                  v61 = *(_QWORD *)(v61 + 32);
                }
                while ( v61 );
LABEL_82:
                v60 = *(_DWORD *)(v59 + 24);
              }
              if ( v60 != 328 )
              {
                v126 = v59;
                sub_32B3B20((__int64)&v164, &v126);
                if ( *(int *)(v59 + 88) < 0 )
                {
                  v114 = (unsigned int)v162;
                  v115 = HIDWORD(v162);
                  v116 = (unsigned int)v162 + 1LL;
                  *(_DWORD *)(v59 + 88) = v162;
                  if ( v116 > v115 )
                  {
                    sub_C8D5F0((__int64)&v161, v163, v116, 8u, v68, v69);
                    v114 = (unsigned int)v162;
                  }
                  v161[v114] = (__int64 (__fastcall **)())v59;
                  LODWORD(v162) = v162 + 1;
                }
              }
            }
            sub_32CF870((__int64)&v152, v42);
            v133[96] = v132;
          }
          else
          {
LABEL_95:
            v133[96] = v132;
          }
          goto LABEL_29;
        }
        break;
      }
      while ( 2 )
      {
        v77 = *(_QWORD *)(v73 + 16);
        if ( *(_DWORD *)(v77 + 24) == 328 )
          goto LABEL_102;
        v126 = *(_QWORD *)(v73 + 16);
        if ( (_DWORD)v166 )
        {
          if ( !(_DWORD)v167 )
          {
            ++v164;
            goto LABEL_149;
          }
          v76 = (unsigned int)(v167 - 1);
          v75 = v165;
          v78 = 1;
          v79 = 0;
          v80 = v76 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v81 = (_QWORD *)(v165 + 8LL * v80);
          v82 = *v81;
          if ( v77 != *v81 )
          {
            while ( v82 != -4096 )
            {
              if ( v79 || v82 != -8192 )
                v81 = v79;
              v80 = v76 & (v78 + v80);
              v82 = *(_QWORD *)(v165 + 8LL * v80);
              if ( v77 == v82 )
                goto LABEL_101;
              ++v78;
              v79 = v81;
              v81 = (_QWORD *)(v165 + 8LL * v80);
            }
            if ( !v79 )
              v79 = v81;
            v83 = v166 + 1;
            ++v164;
            if ( 4 * ((int)v166 + 1) < (unsigned int)(3 * v167) )
            {
              v84 = v77;
              if ( (int)v167 - HIDWORD(v166) - v83 <= (unsigned int)v167 >> 3 )
              {
                sub_32B3220((__int64)&v164, v167);
                if ( !(_DWORD)v167 )
                {
LABEL_200:
                  LODWORD(v166) = v166 + 1;
                  BUG();
                }
                v84 = v126;
                v75 = v165;
                v76 = 0;
                v111 = 1;
                v112 = (v167 - 1) & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
                v79 = (_QWORD *)(v165 + 8LL * v112);
                v113 = *v79;
                v83 = v166 + 1;
                if ( *v79 != v126 )
                {
                  while ( v113 != -4096 )
                  {
                    if ( !v76 && v113 == -8192 )
                      v76 = (__int64)v79;
                    v112 = (v167 - 1) & (v111 + v112);
                    v79 = (_QWORD *)(v165 + 8LL * v112);
                    v113 = *v79;
                    if ( v126 == *v79 )
                      goto LABEL_112;
                    ++v111;
                  }
                  goto LABEL_153;
                }
              }
              goto LABEL_112;
            }
LABEL_149:
            sub_32B3220((__int64)&v164, 2 * v167);
            if ( !(_DWORD)v167 )
              goto LABEL_200;
            v84 = v126;
            v75 = v165;
            v108 = (v167 - 1) & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
            v79 = (_QWORD *)(v165 + 8LL * v108);
            v109 = *v79;
            v83 = v166 + 1;
            if ( *v79 != v126 )
            {
              v110 = 1;
              v76 = 0;
              while ( v109 != -4096 )
              {
                if ( v109 == -8192 && !v76 )
                  v76 = (__int64)v79;
                v108 = (v167 - 1) & (v110 + v108);
                v79 = (_QWORD *)(v165 + 8LL * v108);
                v109 = *v79;
                if ( v126 == *v79 )
                  goto LABEL_112;
                ++v110;
              }
LABEL_153:
              if ( v76 )
                v79 = (_QWORD *)v76;
            }
LABEL_112:
            LODWORD(v166) = v83;
            if ( *v79 != -4096 )
              --HIDWORD(v166);
            *v79 = v84;
            v85 = (unsigned int)v169;
            v86 = v126;
            v87 = (unsigned int)v169 + 1LL;
            if ( v87 > HIDWORD(v169) )
            {
              sub_C8D5F0((__int64)&v168, v170, v87, 8u, v75, v76);
              v85 = (unsigned int)v169;
            }
            v168[v85] = v86;
            LODWORD(v169) = v169 + 1;
            if ( *(int *)(v77 + 88) < 0 )
            {
LABEL_117:
              v88 = (unsigned int)v162;
              v89 = HIDWORD(v162);
              v90 = (unsigned int)v162 + 1LL;
              *(_DWORD *)(v77 + 88) = v162;
              if ( v90 > v89 )
              {
                sub_C8D5F0((__int64)&v161, v163, v90, 8u, v75, v76);
                v88 = (unsigned int)v162;
              }
              v161[v88] = (__int64 (__fastcall **)())v77;
              LODWORD(v162) = v162 + 1;
            }
LABEL_102:
            v73 = *(_QWORD *)(v73 + 32);
            if ( !v73 )
              goto LABEL_87;
            continue;
          }
LABEL_101:
          if ( *(int *)(v77 + 88) < 0 )
            goto LABEL_117;
          goto LABEL_102;
        }
        break;
      }
      v74 = &v168[(unsigned int)v169];
      if ( v74 != sub_325EB50(v168, (__int64)v74, &v126) )
        goto LABEL_101;
      if ( v75 + 1 > (unsigned __int64)HIDWORD(v169) )
      {
        sub_C8D5F0((__int64)&v168, v170, v75 + 1, 8u, v75, v76);
        v74 = &v168[(unsigned int)v169];
      }
      *v74 = v77;
      v91 = (unsigned int)(v169 + 1);
      LODWORD(v169) = v91;
      if ( (unsigned int)v91 <= 0x20 )
        goto LABEL_101;
      v92 = v168;
      v93 = &v168[v91];
      while ( 2 )
      {
        if ( !(_DWORD)v167 )
        {
          ++v164;
          goto LABEL_128;
        }
        v76 = (unsigned int)(v167 - 1);
        v75 = v165;
        v94 = v76 & (((unsigned int)*v92 >> 9) ^ ((unsigned int)*v92 >> 4));
        v95 = (_QWORD *)(v165 + 8LL * v94);
        v96 = *v95;
        if ( *v92 == *v95 )
        {
LABEL_125:
          if ( v93 == ++v92 )
            goto LABEL_101;
          continue;
        }
        break;
      }
      v101 = 1;
      v98 = 0;
      while ( v96 != -4096 )
      {
        if ( v98 || v96 != -8192 )
          v95 = v98;
        v94 = v76 & (v101 + v94);
        v96 = *(_QWORD *)(v165 + 8LL * v94);
        if ( *v92 == v96 )
          goto LABEL_125;
        ++v101;
        v98 = v95;
        v95 = (_QWORD *)(v165 + 8LL * v94);
      }
      if ( !v98 )
        v98 = v95;
      ++v164;
      v100 = v166 + 1;
      if ( 4 * ((int)v166 + 1) >= (unsigned int)(3 * v167) )
      {
LABEL_128:
        sub_32B3220((__int64)&v164, 2 * v167);
        if ( !(_DWORD)v167 )
          goto LABEL_201;
        v75 = v165;
        v97 = (v167 - 1) & (((unsigned int)*v92 >> 9) ^ ((unsigned int)*v92 >> 4));
        v98 = (_QWORD *)(v165 + 8LL * v97);
        v99 = *v98;
        v100 = v166 + 1;
        if ( *v98 != *v92 )
        {
          v76 = 1;
          v102 = 0;
          while ( v99 != -4096 )
          {
            if ( v99 == -8192 && !v102 )
              v102 = v98;
            v97 = (v167 - 1) & (v76 + v97);
            v98 = (_QWORD *)(v165 + 8LL * v97);
            v99 = *v98;
            if ( *v92 == *v98 )
              goto LABEL_130;
            v76 = (unsigned int)(v76 + 1);
          }
          goto LABEL_142;
        }
      }
      else if ( (int)v167 - HIDWORD(v166) - v100 <= (unsigned int)v167 >> 3 )
      {
        sub_32B3220((__int64)&v164, v167);
        if ( !(_DWORD)v167 )
        {
LABEL_201:
          LODWORD(v166) = v166 + 1;
          BUG();
        }
        v76 = 1;
        v102 = 0;
        v75 = v165;
        v103 = (v167 - 1) & (((unsigned int)*v92 >> 9) ^ ((unsigned int)*v92 >> 4));
        v98 = (_QWORD *)(v165 + 8LL * v103);
        v104 = *v98;
        v100 = v166 + 1;
        if ( *v98 != *v92 )
        {
          while ( v104 != -4096 )
          {
            if ( !v102 && v104 == -8192 )
              v102 = v98;
            v103 = (v167 - 1) & (v76 + v103);
            v98 = (_QWORD *)(v165 + 8LL * v103);
            v104 = *v98;
            if ( *v92 == *v98 )
              goto LABEL_130;
            v76 = (unsigned int)(v76 + 1);
          }
LABEL_142:
          if ( v102 )
            v98 = v102;
        }
      }
LABEL_130:
      LODWORD(v166) = v100;
      if ( *v98 != -4096 )
        --HIDWORD(v166);
      *v98 = *v92;
      goto LABEL_125;
    }
  }
  v37 = v140;
  v38 = (__int64 *)v152;
  v39 = v141;
  if ( v140 )
  {
    nullsub_1875(v140, v152, 0);
    v124 = v39;
    v123 = v37;
    v38[48] = v37;
    *((_DWORD *)v38 + 98) = v124;
    sub_33E2B60(v38, 0);
    v38 = (__int64 *)v152;
  }
  else
  {
    v122 = v141;
    v121 = 0;
    v152[24].m128i_i64[0] = 0;
    *((_DWORD *)v38 + 98) = v122;
  }
  sub_33F7860(v38);
  sub_33CF710(v135);
  v129[96] = v128;
LABEL_42:
  if ( !v179 )
    _libc_free((unsigned __int64)s);
  sub_C7D6A0(v172, 24LL * v174, 8);
  if ( v168 != (_QWORD *)v170 )
    _libc_free((unsigned __int64)v168);
  sub_C7D6A0(v165, 8LL * (unsigned int)v167, 8);
  if ( v161 != (__int64 (__fastcall ***)())v163 )
    _libc_free((unsigned __int64)v161);
}
