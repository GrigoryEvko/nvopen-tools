// Function: sub_205D140
// Address: 0x205d140
//
__int64 __fastcall sub_205D140(_QWORD *a1, __int64 *a2, unsigned int a3, unsigned int a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rbx
  unsigned int v8; // r13d
  _QWORD *v9; // rax
  bool v10; // dl
  int v11; // ebx
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int32 v16; // edx
  unsigned int v17; // r13d
  unsigned __int64 v19; // r15
  unsigned int v20; // eax
  __int64 v21; // r13
  __int64 v22; // rbx
  bool v23; // al
  unsigned __int32 v24; // edx
  __int64 v25; // rsi
  char *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rdi
  bool v29; // zf
  char v30; // al
  unsigned __int32 v31; // r14d
  __int64 v32; // rbx
  __int64 v33; // r9
  __int32 v34; // ebx
  __int64 v35; // r13
  char *v36; // r14
  __m128i *v37; // r8
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdi
  unsigned __int64 v41; // r9
  __int64 v42; // rbx
  __int64 v43; // r10
  __int64 v44; // rax
  unsigned int v45; // ecx
  char *v46; // r13
  __int64 v47; // rsi
  unsigned __int32 v48; // eax
  __int64 v49; // r14
  __int64 v50; // rsi
  unsigned __int32 v51; // eax
  __int32 v52; // r8d
  int v53; // eax
  __int64 v54; // rcx
  int v55; // eax
  int v56; // eax
  __int64 v57; // rcx
  unsigned int v58; // eax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // rbx
  unsigned __int64 v62; // rax
  __m128i *v63; // rbx
  __m128i *v64; // r12
  _QWORD *v65; // rax
  int v66; // r8d
  __int64 v67; // rdx
  __int64 v68; // r10
  __int64 v69; // rax
  __m128i *v70; // rax
  __int64 *v71; // r8
  __int64 v72; // r8
  __int64 v73; // rbx
  __int64 v74; // rdx
  unsigned int v75; // eax
  unsigned __int64 v76; // r13
  unsigned __int64 v77; // r14
  int v78; // r12d
  __int64 v79; // rax
  char *v80; // rdi
  bool v81; // al
  int v82; // eax
  __int64 v83; // r8
  __int64 v84; // rsi
  __int64 *v85; // rbx
  __int64 *v86; // rdi
  __int64 *v87; // r10
  unsigned __int32 v88; // r13d
  __int64 v89; // r15
  int v90; // eax
  __int64 *v91; // rsi
  __int64 v92; // r13
  __int64 v93; // rsi
  unsigned __int64 v94; // rdi
  unsigned __int64 v95; // rax
  bool v96; // cf
  unsigned __int64 v97; // rax
  __int64 v98; // r14
  __int64 v99; // rax
  __int64 v100; // r14
  __m128i *v101; // rcx
  unsigned int v102; // r11d
  __int64 v103; // rdx
  __int64 v104; // rax
  unsigned __int64 v105; // r9
  unsigned __int64 v106; // r10
  __int32 v107; // esi
  __m128i *v108; // rdi
  __int64 v109; // r14
  __int64 v110; // r12
  unsigned int v111; // eax
  __int64 v112; // rax
  unsigned int v113; // eax
  unsigned int v114; // eax
  __int64 v115; // r12
  unsigned __int64 v116; // rdi
  __int64 v117; // rdi
  __int64 v118; // rax
  char v119; // [rsp+Fh] [rbp-1F1h]
  __int64 v120; // [rsp+10h] [rbp-1F0h]
  __int64 v121; // [rsp+18h] [rbp-1E8h]
  char v123; // [rsp+28h] [rbp-1D8h]
  unsigned __int64 v125; // [rsp+30h] [rbp-1D0h]
  unsigned int v126; // [rsp+38h] [rbp-1C8h]
  unsigned __int64 v128; // [rsp+40h] [rbp-1C0h]
  unsigned __int64 v129; // [rsp+40h] [rbp-1C0h]
  unsigned __int64 *v132; // [rsp+58h] [rbp-1A8h]
  unsigned __int64 v133; // [rsp+58h] [rbp-1A8h]
  unsigned int v134; // [rsp+58h] [rbp-1A8h]
  unsigned int v135; // [rsp+58h] [rbp-1A8h]
  void *s; // [rsp+60h] [rbp-1A0h]
  __int64 v137; // [rsp+68h] [rbp-198h]
  unsigned __int64 v138; // [rsp+68h] [rbp-198h]
  unsigned __int64 v139; // [rsp+68h] [rbp-198h]
  __int64 v141; // [rsp+70h] [rbp-190h]
  unsigned int v142; // [rsp+70h] [rbp-190h]
  unsigned int v143; // [rsp+70h] [rbp-190h]
  unsigned int v144; // [rsp+70h] [rbp-190h]
  __int64 v145; // [rsp+70h] [rbp-190h]
  __int64 v146; // [rsp+70h] [rbp-190h]
  unsigned int v147; // [rsp+78h] [rbp-188h]
  unsigned int v148; // [rsp+78h] [rbp-188h]
  __int64 v149; // [rsp+78h] [rbp-188h]
  unsigned int v150; // [rsp+78h] [rbp-188h]
  unsigned int v151; // [rsp+80h] [rbp-180h]
  __int64 v152; // [rsp+80h] [rbp-180h]
  char *v153; // [rsp+80h] [rbp-180h]
  __int64 v154; // [rsp+80h] [rbp-180h]
  int v155; // [rsp+80h] [rbp-180h]
  int v156; // [rsp+80h] [rbp-180h]
  __int64 v157; // [rsp+80h] [rbp-180h]
  __int64 v158; // [rsp+80h] [rbp-180h]
  __int64 v159; // [rsp+80h] [rbp-180h]
  __int64 v160; // [rsp+88h] [rbp-178h]
  unsigned __int64 v161; // [rsp+88h] [rbp-178h]
  unsigned __int32 v162; // [rsp+88h] [rbp-178h]
  bool v163; // [rsp+88h] [rbp-178h]
  char v164; // [rsp+88h] [rbp-178h]
  unsigned int v165; // [rsp+88h] [rbp-178h]
  __int64 v166; // [rsp+90h] [rbp-170h] BYREF
  unsigned int v167; // [rsp+98h] [rbp-168h]
  unsigned __int64 v168; // [rsp+A0h] [rbp-160h] BYREF
  unsigned __int32 v169; // [rsp+A8h] [rbp-158h]
  unsigned __int64 v170; // [rsp+B0h] [rbp-150h] BYREF
  unsigned __int32 v171; // [rsp+B8h] [rbp-148h]
  unsigned __int64 v172; // [rsp+C0h] [rbp-140h] BYREF
  unsigned int v173; // [rsp+C8h] [rbp-138h]
  void *src; // [rsp+D0h] [rbp-130h] BYREF
  __m128i *v175; // [rsp+D8h] [rbp-128h]
  __m128i *v176; // [rsp+E0h] [rbp-120h]
  char *v177; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v178; // [rsp+F8h] [rbp-108h]
  _BYTE v179[96]; // [rsp+100h] [rbp-100h] BYREF
  __m128i v180; // [rsp+160h] [rbp-A0h] BYREF
  __m128i v181[9]; // [rsp+170h] [rbp-90h] BYREF

  v6 = ((unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1[89] + 8LL) + 104LL)
                               - *(_QWORD *)(*(_QWORD *)(a1[89] + 8LL) + 96LL)) >> 3)
      + 63) >> 6;
  s = (void *)malloc(8LL * v6);
  if ( !s )
  {
    if ( 8LL * v6 || (v118 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed", 1u);
    else
      s = (void *)v118;
  }
  v160 = *a2;
  v7 = a3;
  if ( v6 )
  {
    memset(s, 0, 8LL * v6);
    if ( a3 > (__int64)a4 )
    {
      v8 = 0;
LABEL_7:
      v151 = v8;
      v11 = 0;
      v12 = 0;
      do
      {
        v13 = *((_QWORD *)s + v12++);
        v11 += sub_39FAC40(v13);
      }
      while ( v6 > (unsigned int)v12 );
      v8 = v151;
      goto LABEL_10;
    }
  }
  else if ( a3 > (__int64)a4 )
  {
    v8 = 0;
    goto LABEL_229;
  }
  v8 = 0;
  v9 = (_QWORD *)(v160 + 40LL * a3 + 8);
  do
  {
    *((_QWORD *)s + (*(_DWORD *)(v9[2] + 48LL) >> 6)) |= 1LL << *(_DWORD *)(v9[2] + 48LL);
    v10 = *v9 != v9[1];
    ++v7;
    v9 += 5;
    v8 += v10 + 1;
  }
  while ( v7 <= a4 );
  if ( v6 )
    goto LABEL_7;
LABEL_229:
  v11 = 0;
LABEL_10:
  v121 = 40LL * a3;
  v14 = *(_QWORD *)(v160 + v121 + 8);
  v167 = *(_DWORD *)(v14 + 32);
  if ( v167 > 0x40 )
  {
    sub_16A4FD0((__int64)&v166, (const void **)(v14 + 24));
    v160 = *a2;
  }
  else
  {
    v166 = *(_QWORD *)(v14 + 24);
  }
  v120 = 40LL * a4;
  v15 = *(_QWORD *)(v160 + v120 + 16);
  v169 = *(_DWORD *)(v15 + 32);
  if ( v169 > 0x40 )
    sub_16A4FD0((__int64)&v168, (const void **)(v15 + 24));
  else
    v168 = *(_QWORD *)(v15 + 24);
  v152 = sub_1E0A0C0(*(_QWORD *)(a1[69] + 32LL));
  v161 = 8 * (unsigned int)sub_15A95A0(v152, 0);
  v180.m128i_i32[2] = v169;
  if ( v169 > 0x40 )
    sub_16A4FD0((__int64)&v180, (const void **)&v168);
  else
    v180.m128i_i64[0] = v168;
  sub_16A7590((__int64)&v180, &v166);
  v16 = v180.m128i_u32[2];
  v180.m128i_i32[2] = 0;
  LODWORD(v178) = v16;
  v177 = (char *)v180.m128i_i64[0];
  if ( v16 > 0x40 )
  {
    v132 = (unsigned __int64 *)v180.m128i_i64[0];
    v16 -= sub_16A57B0((__int64)&v177);
    if ( v16 > 0x40 )
    {
      v19 = -1;
    }
    else
    {
      v19 = *v132;
      if ( *v132 == -1 )
      {
        j_j___libc_free_0_0(v132);
        if ( v180.m128i_i32[2] <= 0x40u )
          goto LABEL_18;
LABEL_31:
        if ( v180.m128i_i64[0] )
          j_j___libc_free_0_0(v180.m128i_i64[0]);
        goto LABEL_33;
      }
      ++v19;
    }
    if ( !v132 )
      goto LABEL_33;
    j_j___libc_free_0_0(v132);
    if ( v180.m128i_i32[2] <= 0x40u )
      goto LABEL_33;
    goto LABEL_31;
  }
  if ( v180.m128i_i64[0] == -1 )
  {
LABEL_18:
    v17 = 0;
    goto LABEL_19;
  }
  v19 = v180.m128i_i64[0] + 1;
LABEL_33:
  if ( v161 < v19 )
    goto LABEL_18;
  if ( (v11 != 1 || v8 <= 2) && (v11 != 2 || v8 <= 4) )
  {
    LOBYTE(v16) = v8 > 5 && v11 == 3;
    v17 = v16;
    if ( !(_BYTE)v16 )
      goto LABEL_19;
  }
  v171 = 1;
  v170 = 0;
  v173 = 1;
  v172 = 0;
  v20 = 8 * sub_15A9520(v152, 0);
  if ( v20 == 32 )
  {
    v123 = 5;
  }
  else if ( v20 > 0x20 )
  {
    v123 = 6;
    if ( v20 != 64 )
    {
      v29 = v20 == 128;
      v30 = 7;
      if ( !v29 )
        v30 = 0;
      v123 = v30;
    }
  }
  else
  {
    v123 = 3;
    if ( v20 != 8 )
      v123 = 4 * (v20 == 16);
  }
  v21 = a3 + 1;
  v147 = a3 + 1;
  if ( v21 > a4 )
  {
LABEL_69:
    v119 = 1;
  }
  else
  {
    v22 = 8 * (5 * v21 - 5);
    while ( 1 )
    {
      v25 = *(_QWORD *)(*a2 + v22 + 16);
      LODWORD(v178) = *(_DWORD *)(v25 + 32);
      if ( (unsigned int)v178 > 0x40 )
        sub_16A4FD0((__int64)&v177, (const void **)(v25 + 24));
      else
        v177 = *(char **)(v25 + 24);
      v22 += 40;
      sub_16A7490((__int64)&v177, 1);
      v24 = v178;
      v26 = v177;
      LODWORD(v178) = 0;
      v27 = *a2;
      v180.m128i_i32[2] = v24;
      v180.m128i_i64[0] = (__int64)v177;
      v28 = *(_QWORD *)(v27 + v22 + 8);
      if ( *(_DWORD *)(v28 + 32) > 0x40u )
      {
        v162 = v24;
        v23 = sub_16A5220(v28 + 24, (const void **)&v180);
        v24 = v162;
      }
      else
      {
        v23 = *(_QWORD *)(v28 + 24) == (_QWORD)v177;
      }
      if ( v24 > 0x40 )
      {
        if ( v26 )
        {
          v163 = v23;
          j_j___libc_free_0_0(v26);
          v23 = v163;
          if ( (unsigned int)v178 > 0x40 )
          {
            if ( v177 )
            {
              j_j___libc_free_0_0(v177);
              v23 = v163;
            }
          }
        }
      }
      if ( !v23 )
        break;
      if ( ++v21 > a4 )
        goto LABEL_69;
    }
    v119 = 0;
  }
  v31 = v167;
  v32 = v166;
  v164 = (v167 - 1) & 0x3F;
  if ( v167 > 0x40 )
  {
    if ( (*(_QWORD *)(v166 + 8LL * ((v167 - 1) >> 6)) & (1LL << ((unsigned __int8)v167 - 1))) != 0 )
      goto LABEL_76;
    v81 = v31 == (unsigned int)sub_16A57B0((__int64)&v166);
  }
  else
  {
    if ( ((1LL << ((unsigned __int8)v167 - 1)) & v166) != 0 )
    {
LABEL_72:
      if ( v171 <= 0x40 && v31 <= 0x40 )
      {
        v171 = v31;
        v170 = (0xFFFFFFFFFFFFFFFFLL >> (63 - v164)) & v32;
        goto LABEL_77;
      }
LABEL_76:
      sub_16A51C0((__int64)&v170, (__int64)&v166);
LABEL_77:
      v180.m128i_i32[2] = v169;
      if ( v169 > 0x40 )
        sub_16A4FD0((__int64)&v180, (const void **)&v168);
      else
        v180.m128i_i64[0] = v168;
      sub_16A7590((__int64)&v180, &v166);
      v34 = v180.m128i_i32[2];
      v180.m128i_i32[2] = 0;
      v35 = v180.m128i_i64[0];
      if ( v173 > 0x40 && v172 )
      {
        j_j___libc_free_0_0(v172);
        v172 = v35;
        v173 = v34;
        if ( v180.m128i_i32[2] > 0x40u && v180.m128i_i64[0] )
          j_j___libc_free_0_0(v180.m128i_i64[0]);
      }
      else
      {
        v172 = v180.m128i_i64[0];
        v173 = v34;
      }
      goto LABEL_81;
    }
    v81 = v166 == 0;
  }
  if ( v81 )
    goto LABEL_72;
  v82 = sub_2045180(v123);
  v83 = v82;
  if ( v169 > 0x40 )
  {
    v88 = v169 + 1;
    v133 = v168;
    v157 = v82;
    v89 = *(_QWORD *)(v168 + 8LL * ((v169 - 1) >> 6)) & (1LL << ((unsigned __int8)v169 - 1));
    if ( v89 )
    {
      v90 = sub_16A5810((__int64)&v168);
      v83 = v157;
      v91 = (__int64 *)v133;
    }
    else
    {
      v90 = sub_16A57B0((__int64)&v168);
      v91 = (__int64 *)v133;
      v83 = v157;
    }
    if ( v88 - v90 > 0x40 )
    {
      if ( !v89 )
        goto LABEL_72;
      goto LABEL_156;
    }
    v84 = *v91;
  }
  else
  {
    v84 = (__int64)(v168 << (64 - (unsigned __int8)v169)) >> (64 - (unsigned __int8)v169);
  }
  if ( v83 <= v84 )
    goto LABEL_72;
LABEL_156:
  v180.m128i_i32[2] = v31;
  if ( v31 > 0x40 )
    sub_16A4EF0((__int64)&v180, 0, 0);
  else
    v180.m128i_i64[0] = 0;
  if ( v171 > 0x40 && v170 )
    j_j___libc_free_0_0(v170);
  v170 = v180.m128i_i64[0];
  v171 = v180.m128i_u32[2];
  if ( v173 <= 0x40 && v169 <= 0x40 )
  {
    v173 = v169;
    v119 = 0;
    v172 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v169) & v168;
  }
  else
  {
    sub_16A51C0((__int64)&v172, (__int64)&v168);
    v119 = 0;
  }
LABEL_81:
  src = 0;
  v175 = 0;
  v176 = 0;
  if ( a3 > a4 )
  {
    v165 = 0;
    v177 = v179;
    v178 = 0x300000000LL;
  }
  else
  {
    v36 = 0;
    v37 = 0;
    v38 = *a2;
    v165 = 0;
    v39 = a3;
    v40 = *a2;
    while ( 1 )
    {
      v41 = 0xAAAAAAAAAAAAAAABLL * (((char *)v37 - v36) >> 3);
      v42 = 40 * v39;
      if ( v41 )
      {
        v43 = v38 + v42;
        v44 = 0;
        v45 = 0;
        while ( 1 )
        {
          v46 = &v36[24 * v44];
          if ( *((_QWORD *)v46 + 1) == *(_QWORD *)(v38 + v42 + 24) )
            break;
          v44 = ++v45;
          if ( v45 >= v41 )
          {
            v59 = 24LL * v45;
            if ( v45 == v41 )
              goto LABEL_110;
            v46 = &v36[24 * v45];
            break;
          }
        }
      }
      else
      {
        v59 = 0;
LABEL_110:
        v60 = *(_QWORD *)(v38 + v42 + 24);
        v180.m128i_i64[0] = 0;
        v181[0].m128i_i64[0] = 0;
        v180.m128i_i64[1] = v60;
        if ( v37 == v176 )
        {
          sub_205CF90((const __m128i **)&src, v37, &v180);
          v36 = (char *)src;
          v40 = *a2;
        }
        else
        {
          if ( v37 )
          {
            *v37 = _mm_loadu_si128(&v180);
            v36 = (char *)src;
            v37[1].m128i_i64[0] = v181[0].m128i_i64[0];
            v37 = v175;
            v40 = *a2;
          }
          v175 = (__m128i *)((char *)v37 + 24);
        }
        v46 = &v36[v59];
        v43 = v40 + v42;
      }
      v47 = *(_QWORD *)(v43 + 8);
      v180.m128i_i32[2] = *(_DWORD *)(v47 + 32);
      if ( v180.m128i_i32[2] > 0x40u )
        sub_16A4FD0((__int64)&v180, (const void **)(v47 + 24));
      else
        v180.m128i_i64[0] = *(_QWORD *)(v47 + 24);
      sub_16A7590((__int64)&v180, (__int64 *)&v170);
      v48 = v180.m128i_u32[2];
      LODWORD(v49) = v180.m128i_i32[0];
      v180.m128i_i32[2] = 0;
      if ( v48 > 0x40 )
      {
        v49 = *(_QWORD *)v180.m128i_i64[0];
        j_j___libc_free_0_0(v180.m128i_i64[0]);
        if ( v180.m128i_i32[2] > 0x40u )
        {
          if ( v180.m128i_i64[0] )
            j_j___libc_free_0_0(v180.m128i_i64[0]);
        }
      }
      v50 = *(_QWORD *)(*a2 + v42 + 16);
      v180.m128i_i32[2] = *(_DWORD *)(v50 + 32);
      if ( v180.m128i_i32[2] > 0x40u )
        sub_16A4FD0((__int64)&v180, (const void **)(v50 + 24));
      else
        v180.m128i_i64[0] = *(_QWORD *)(v50 + 24);
      sub_16A7590((__int64)&v180, (__int64 *)&v170);
      v51 = v180.m128i_u32[2];
      v52 = v180.m128i_i32[0];
      v180.m128i_i32[2] = 0;
      if ( v51 > 0x40 )
      {
        v137 = *(_QWORD *)v180.m128i_i64[0];
        j_j___libc_free_0_0(v180.m128i_i64[0]);
        v52 = v137;
        if ( v180.m128i_i32[2] > 0x40u )
        {
          if ( v180.m128i_i64[0] )
          {
            j_j___libc_free_0_0(v180.m128i_i64[0]);
            v52 = v137;
          }
        }
      }
      v53 = *((_DWORD *)v46 + 4) + 1 - v49;
      *(_QWORD *)v46 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v49 + 63 - v52) << v49;
      v54 = *((unsigned int *)v46 + 5);
      v55 = v52 + v53;
      v36 = (char *)src;
      v37 = v175;
      *((_DWORD *)v46 + 4) = v55;
      v56 = *((_DWORD *)v46 + 5) + *(_DWORD *)(*a2 + v42 + 32);
      if ( (unsigned __int64)*(unsigned int *)(*a2 + v42 + 32) + v54 > 0x80000000 )
        v56 = 0x80000000;
      *((_DWORD *)v46 + 5) = v56;
      v38 = *a2;
      v57 = *(unsigned int *)(*a2 + v42 + 32);
      v40 = *a2;
      v58 = v165 + v57;
      if ( v57 + (unsigned __int64)v165 > 0x80000000 )
        v58 = 0x80000000;
      v165 = v58;
      if ( a4 < v147 )
        break;
      v39 = v147++;
    }
    v177 = v179;
    v178 = 0x300000000LL;
    if ( v37 != (__m128i *)v36 )
    {
      v153 = (char *)v37;
      v61 = (char *)v37 - v36;
      _BitScanReverse64(&v62, 0xAAAAAAAAAAAAAAABLL * (((char *)v37 - v36) >> 3));
      sub_20461E0((__int64)v36, v37, 2LL * (int)(63 - (v62 ^ 0x3F)), v57, (__int64)v37, v33);
      if ( v61 > 384 )
      {
        v85 = (__int64 *)(v36 + 384);
        sub_2045B30(v36, v36 + 384);
        if ( v153 != v36 + 384 )
        {
          do
          {
            v86 = v85;
            v85 += 3;
            sub_2045380(v86);
          }
          while ( v87 != v85 );
        }
      }
      else
      {
        sub_2045B30(v36, v153);
      }
      v63 = v175;
      if ( src != v175 )
      {
        v64 = (__m128i *)src;
        do
        {
          v65 = sub_1E0B6F0(*(_QWORD *)(a1[89] + 8LL), *(_QWORD *)(a5 + 40));
          v67 = v64->m128i_i64[1];
          v68 = (__int64)v65;
          LODWORD(v65) = v64[1].m128i_i32[1];
          v180.m128i_i64[0] = v64->m128i_i64[0];
          v181[0].m128i_i32[2] = (int)v65;
          v69 = (unsigned int)v178;
          v180.m128i_i64[1] = v68;
          v181[0].m128i_i64[0] = v67;
          if ( (unsigned int)v178 >= HIDWORD(v178) )
          {
            sub_16CD150((__int64)&v177, v179, 0, 32, v66, v33);
            v69 = (unsigned int)v178;
          }
          v70 = (__m128i *)&v177[32 * v69];
          v64 = (__m128i *)((char *)v64 + 24);
          *v70 = _mm_loadu_si128(&v180);
          v70[1] = _mm_loadu_si128(v181);
          LODWORD(v178) = v178 + 1;
        }
        while ( v63 != v64 );
      }
    }
  }
  if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
    v71 = *(__int64 **)(a5 - 8);
  else
    v71 = (__int64 *)(a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
  v72 = *v71;
  v73 = a1[80];
  if ( v73 != a1[81] )
  {
    v74 = v171;
    v75 = v173;
    v180.m128i_i64[0] = (__int64)v181;
    v180.m128i_i64[1] = 0x300000000LL;
    v76 = v170;
    v171 = 0;
    v77 = v172;
    v173 = 0;
    if ( (_DWORD)v178 )
    {
      v142 = v75;
      v150 = v74;
      v159 = v72;
      sub_2044B00((__int64)&v180, &v177, v74, (__int64)v181, v72, v33);
      v75 = v142;
      v74 = v150;
      v72 = v159;
    }
    if ( v73 )
    {
      *(_DWORD *)(v73 + 24) = v75;
      *(_QWORD *)(v73 + 32) = v72;
      *(_BYTE *)(v73 + 46) = v119;
      *(_QWORD *)(v73 + 64) = v73 + 80;
      *(_DWORD *)(v73 + 8) = v74;
      *(_QWORD *)v73 = v76;
      *(_QWORD *)(v73 + 16) = v77;
      *(_DWORD *)(v73 + 40) = -1;
      *(_WORD *)(v73 + 44) = 1;
      *(_QWORD *)(v73 + 48) = 0;
      *(_QWORD *)(v73 + 56) = 0;
      *(_QWORD *)(v73 + 72) = 0x300000000LL;
      if ( v180.m128i_i32[2] )
        sub_2044B00(v73 + 64, (char **)&v180, v74, (__int64)v181, 1, v180.m128i_i32[2]);
      *(_DWORD *)(v73 + 180) = -1;
      *(_DWORD *)(v73 + 176) = v165;
      if ( (__m128i *)v180.m128i_i64[0] != v181 )
        _libc_free(v180.m128i_u64[0]);
    }
    else
    {
      if ( (__m128i *)v180.m128i_i64[0] != v181 )
      {
        v148 = v75;
        v155 = v74;
        _libc_free(v180.m128i_u64[0]);
        v75 = v148;
        LODWORD(v74) = v155;
      }
      if ( v75 > 0x40 && v77 )
      {
        v156 = v74;
        j_j___libc_free_0_0(v77);
        LODWORD(v74) = v156;
      }
      if ( (unsigned int)v74 > 0x40 && v76 )
        j_j___libc_free_0_0(v76);
    }
    v154 = a1[80];
    a1[80] = v154 + 184;
    v78 = -373475417 * ((v154 + 184 - a1[79]) >> 3) - 1;
    goto LABEL_134;
  }
  v92 = a1[79];
  v93 = v73 - v92;
  v94 = 0xD37A6F4DE9BD37A7LL * ((v73 - v92) >> 3);
  if ( v94 == 0xB21642C8590B21LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v95 = 1;
  if ( v94 )
    v95 = 0xD37A6F4DE9BD37A7LL * ((v73 - v92) >> 3);
  v96 = __CFADD__(v94, v95);
  v97 = v94 + v95;
  if ( v96 )
  {
    v98 = 0x7FFFFFFFFFFFFFB8LL;
  }
  else
  {
    if ( !v97 )
    {
      v149 = 0;
      v100 = 184;
      v158 = 0;
      goto LABEL_191;
    }
    if ( v97 > 0xB21642C8590B21LL )
      v97 = 0xB21642C8590B21LL;
    v98 = 184 * v97;
  }
  v141 = v72;
  v99 = sub_22077B0(v98);
  v72 = v141;
  v93 = v73 - v92;
  v158 = v99;
  v149 = v99 + v98;
  v100 = v99 + 184;
LABEL_191:
  v101 = v181;
  v102 = v171;
  v180.m128i_i64[1] = 0x300000000LL;
  v103 = v173;
  v180.m128i_i64[0] = (__int64)v181;
  v104 = v93 + v158;
  v105 = v170;
  v106 = v172;
  v171 = 0;
  v173 = 0;
  if ( (_DWORD)v178 )
  {
    v125 = v172;
    v126 = v103;
    v129 = v170;
    v135 = v102;
    v145 = v72;
    sub_2044B00((__int64)&v180, &v177, v103, (__int64)v181, v72, v170);
    v101 = v181;
    v106 = v125;
    v103 = v126;
    v105 = v129;
    v102 = v135;
    v104 = v93 + v158;
    v72 = v145;
  }
  if ( v104 )
  {
    *(_DWORD *)(v104 + 24) = v103;
    v103 = 1;
    *(_QWORD *)(v104 + 64) = v104 + 80;
    v107 = v180.m128i_i32[2];
    *(_BYTE *)(v104 + 46) = v119;
    *(_DWORD *)(v104 + 8) = v102;
    *(_QWORD *)v104 = v105;
    *(_QWORD *)(v104 + 16) = v106;
    *(_QWORD *)(v104 + 32) = v72;
    *(_DWORD *)(v104 + 40) = -1;
    *(_WORD *)(v104 + 44) = 1;
    *(_QWORD *)(v104 + 48) = 0;
    *(_QWORD *)(v104 + 56) = 0;
    *(_QWORD *)(v104 + 72) = 0x300000000LL;
    if ( v107 )
    {
      v146 = v104;
      sub_2044B00(v104 + 64, (char **)&v180, 1, (__int64)v181, v72, v105);
      v101 = v181;
      v104 = v146;
    }
    v108 = (__m128i *)v180.m128i_i64[0];
    *(_DWORD *)(v104 + 180) = -1;
    *(_DWORD *)(v104 + 176) = v165;
    if ( v108 != v181 )
      _libc_free((unsigned __int64)v108);
  }
  else
  {
    if ( (__m128i *)v180.m128i_i64[0] != v181 )
    {
      v128 = v106;
      v134 = v103;
      v138 = v105;
      v143 = v102;
      _libc_free(v180.m128i_u64[0]);
      v106 = v128;
      v103 = v134;
      v105 = v138;
      v102 = v143;
    }
    if ( (unsigned int)v103 > 0x40 && v106 )
    {
      v139 = v105;
      v144 = v102;
      j_j___libc_free_0_0(v106);
      v105 = v139;
      v102 = v144;
    }
    if ( v102 > 0x40 && v105 )
      j_j___libc_free_0_0(v105);
  }
  if ( v73 == v92 )
  {
    v78 = 0;
    goto LABEL_223;
  }
  v109 = v158;
  v110 = v92;
  while ( 1 )
  {
    if ( !v109 )
      goto LABEL_205;
    v113 = *(_DWORD *)(v110 + 8);
    *(_DWORD *)(v109 + 8) = v113;
    if ( v113 <= 0x40 )
    {
      *(_QWORD *)v109 = *(_QWORD *)v110;
      v111 = *(_DWORD *)(v110 + 24);
      *(_DWORD *)(v109 + 24) = v111;
      if ( v111 > 0x40 )
        goto LABEL_210;
    }
    else
    {
      sub_16A4FD0(v109, (const void **)v110);
      v114 = *(_DWORD *)(v110 + 24);
      *(_DWORD *)(v109 + 24) = v114;
      if ( v114 > 0x40 )
      {
LABEL_210:
        sub_16A4FD0(v109 + 16, (const void **)(v110 + 16));
        goto LABEL_202;
      }
    }
    *(_QWORD *)(v109 + 16) = *(_QWORD *)(v110 + 16);
LABEL_202:
    *(_QWORD *)(v109 + 32) = *(_QWORD *)(v110 + 32);
    *(_DWORD *)(v109 + 40) = *(_DWORD *)(v110 + 40);
    *(_BYTE *)(v109 + 44) = *(_BYTE *)(v110 + 44);
    *(_BYTE *)(v109 + 45) = *(_BYTE *)(v110 + 45);
    *(_BYTE *)(v109 + 46) = *(_BYTE *)(v110 + 46);
    *(_QWORD *)(v109 + 48) = *(_QWORD *)(v110 + 48);
    v112 = *(_QWORD *)(v110 + 56);
    *(_DWORD *)(v109 + 72) = 0;
    *(_QWORD *)(v109 + 56) = v112;
    *(_QWORD *)(v109 + 64) = v109 + 80;
    *(_DWORD *)(v109 + 76) = 3;
    if ( *(_DWORD *)(v110 + 72) )
      sub_2044700(v109 + 64, v110 + 64, v103, (__int64)v101, v72, v105);
    *(_DWORD *)(v109 + 176) = *(_DWORD *)(v110 + 176);
    *(_DWORD *)(v109 + 180) = *(_DWORD *)(v110 + 180);
LABEL_205:
    v110 += 184;
    if ( v73 == v110 )
      break;
    v109 += 184;
  }
  v100 = v109 + 368;
  v115 = v92;
  do
  {
    v116 = *(_QWORD *)(v115 + 64);
    if ( v116 != v115 + 80 )
      _libc_free(v116);
    if ( *(_DWORD *)(v115 + 24) > 0x40u )
    {
      v117 = *(_QWORD *)(v115 + 16);
      if ( v117 )
        j_j___libc_free_0_0(v117);
    }
    if ( *(_DWORD *)(v115 + 8) > 0x40u && *(_QWORD *)v115 )
      j_j___libc_free_0_0(*(_QWORD *)v115);
    v115 += 184;
  }
  while ( v73 != v115 );
  v78 = -373475417 * ((v100 - v158) >> 3) - 1;
LABEL_223:
  if ( v92 )
    j_j___libc_free_0(v92, a1[81] - v92);
  a1[79] = v158;
  a1[80] = v100;
  a1[81] = v149;
LABEL_134:
  v79 = *(_QWORD *)(*a2 + v121 + 8);
  v80 = v177;
  *(_QWORD *)(a6 + 16) = *(_QWORD *)(*a2 + v120 + 16);
  *(_DWORD *)a6 = 2;
  *(_QWORD *)(a6 + 8) = v79;
  *(_DWORD *)(a6 + 24) = v78;
  *(_DWORD *)(a6 + 32) = v165;
  if ( v80 != v179 )
    _libc_free((unsigned __int64)v80);
  if ( src )
    j_j___libc_free_0(src, (char *)v176 - (_BYTE *)src);
  if ( v173 > 0x40 && v172 )
    j_j___libc_free_0_0(v172);
  if ( v171 > 0x40 && v170 )
    j_j___libc_free_0_0(v170);
  v17 = 1;
LABEL_19:
  if ( v169 > 0x40 && v168 )
    j_j___libc_free_0_0(v168);
  if ( v167 > 0x40 && v166 )
    j_j___libc_free_0_0(v166);
  _libc_free((unsigned __int64)s);
  return v17;
}
