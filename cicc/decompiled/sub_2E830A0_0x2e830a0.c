// Function: sub_2E830A0
// Address: 0x2e830a0
//
__int64 __fastcall sub_2E830A0(__int64 *a1, __int64 a2)
{
  unsigned int i; // r14d
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rax
  const char *v10; // rsi
  char v11; // bl
  __int64 v12; // rax
  int v13; // edx
  bool v14; // al
  unsigned int v15; // r15d
  __int64 (__fastcall *v16)(__int64); // rax
  __m128i *v17; // rsi
  unsigned int j; // ebx
  __m128i *v19; // rax
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rax
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rdx
  char *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // r14
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r14
  unsigned __int64 v34; // rdi
  int v35; // eax
  const char *v36; // rbx
  __int64 v38; // rdx
  __int64 (__fastcall *v39)(__int64); // rax
  __m128i *v40; // r8
  __int64 v41; // rax
  void *v42; // r14
  __m128i v43; // rax
  __m128i v44; // rax
  char v45; // al
  _QWORD *v46; // rdx
  char v47; // al
  __m128i *v48; // rcx
  char v49; // dl
  _QWORD *v50; // rsi
  char v51; // al
  __m128i *v52; // rsi
  __m128i *v53; // rcx
  char v54; // dl
  __m128i *v55; // rsi
  __m128i *v56; // rcx
  char v57; // al
  __m128i *v58; // rsi
  __m128i *v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rax
  void *v62; // rax
  _QWORD *v63; // rax
  __m128i *v64; // rdx
  __int64 v65; // r14
  __m128i si128; // xmm0
  unsigned __int8 *v67; // rax
  size_t v68; // rdx
  void *v69; // rdi
  _QWORD *v70; // rax
  _DWORD *v71; // rdx
  __int64 v72; // r13
  __m128i v73; // rax
  char v74; // al
  __m128i *v75; // rdx
  char *v76; // rax
  __int64 v77; // rdx
  _QWORD *v78; // rax
  _WORD *v79; // rdx
  __int64 v80; // r13
  void *v81; // rdi
  _BYTE *v82; // rax
  unsigned int v83; // eax
  const char *v84; // rcx
  __int64 v85; // rbx
  const char *v86; // rdi
  size_t v87; // r13
  void *v88; // r14
  __int64 v89; // r9
  __int64 v90; // rax
  __m128i v91; // xmm5
  __m128i v92; // xmm2
  __m128i v93; // xmm3
  __m128i v94; // xmm5
  __m128i v95; // xmm7
  __m128i v96; // xmm4
  __m128i v97; // xmm0
  __m128i v98; // xmm6
  __m128i v99; // xmm3
  __m128i v100; // xmm2
  __int64 v101; // [rsp+8h] [rbp-558h]
  __int64 v102; // [rsp+10h] [rbp-550h]
  __int64 v103; // [rsp+18h] [rbp-548h]
  __int64 v104; // [rsp+20h] [rbp-540h]
  __int64 v105; // [rsp+28h] [rbp-538h]
  __int64 v106; // [rsp+30h] [rbp-530h]
  __int64 v107; // [rsp+38h] [rbp-528h]
  __int64 v108; // [rsp+40h] [rbp-520h]
  __int64 v109; // [rsp+48h] [rbp-518h]
  __int64 v110; // [rsp+50h] [rbp-510h]
  __int64 v111; // [rsp+58h] [rbp-508h]
  __int64 v112; // [rsp+60h] [rbp-500h]
  __int64 v113; // [rsp+60h] [rbp-500h]
  __int64 v114; // [rsp+68h] [rbp-4F8h]
  __int64 v115; // [rsp+70h] [rbp-4F0h]
  __int64 v117; // [rsp+78h] [rbp-4E8h]
  bool v118; // [rsp+86h] [rbp-4DAh]
  bool v119; // [rsp+87h] [rbp-4D9h]
  bool v120; // [rsp+98h] [rbp-4C8h]
  size_t v121; // [rsp+98h] [rbp-4C8h]
  unsigned __int8 *src; // [rsp+A0h] [rbp-4C0h]
  const char *srca; // [rsp+A0h] [rbp-4C0h]
  size_t v124; // [rsp+B0h] [rbp-4B0h]
  void *s1; // [rsp+C0h] [rbp-4A0h] BYREF
  size_t n; // [rsp+C8h] [rbp-498h]
  __int64 v127; // [rsp+D0h] [rbp-490h]
  char v128; // [rsp+D8h] [rbp-488h] BYREF
  void *s2; // [rsp+E0h] [rbp-480h] BYREF
  size_t v130; // [rsp+E8h] [rbp-478h]
  __int64 v131; // [rsp+F0h] [rbp-470h]
  char v132; // [rsp+F8h] [rbp-468h] BYREF
  __m128i v133; // [rsp+100h] [rbp-460h] BYREF
  __m128i v134; // [rsp+110h] [rbp-450h] BYREF
  __int64 *v135; // [rsp+120h] [rbp-440h]
  _QWORD v136[4]; // [rsp+130h] [rbp-430h] BYREF
  char v137; // [rsp+150h] [rbp-410h]
  char v138; // [rsp+151h] [rbp-40Fh]
  __m128i v139; // [rsp+160h] [rbp-400h] BYREF
  __m128i v140; // [rsp+170h] [rbp-3F0h] BYREF
  __int64 *v141; // [rsp+180h] [rbp-3E0h]
  unsigned __int8 *v142; // [rsp+190h] [rbp-3D0h] BYREF
  size_t v143; // [rsp+198h] [rbp-3C8h]
  __int16 v144; // [rsp+1B0h] [rbp-3B0h]
  __m128i v145; // [rsp+1C0h] [rbp-3A0h] BYREF
  __m128i v146; // [rsp+1D0h] [rbp-390h] BYREF
  __int64 *v147; // [rsp+1E0h] [rbp-380h]
  __m128i v148; // [rsp+1F0h] [rbp-370h] BYREF
  __m128i v149; // [rsp+200h] [rbp-360h] BYREF
  __int64 *v150; // [rsp+210h] [rbp-350h]
  __int64 v151; // [rsp+220h] [rbp-340h] BYREF
  __m128i v152; // [rsp+240h] [rbp-320h] BYREF
  __m128i v153; // [rsp+250h] [rbp-310h] BYREF
  __int64 *v154; // [rsp+260h] [rbp-300h]
  __int64 v155; // [rsp+270h] [rbp-2F0h] BYREF
  __m128i v156; // [rsp+290h] [rbp-2D0h] BYREF
  __m128i v157; // [rsp+2A0h] [rbp-2C0h] BYREF
  __int64 *v158; // [rsp+2B0h] [rbp-2B0h]
  __int64 v159; // [rsp+2C0h] [rbp-2A0h] BYREF
  __m128i v160; // [rsp+2E0h] [rbp-280h] BYREF
  __m128i v161; // [rsp+2F0h] [rbp-270h] BYREF
  __int64 *v162; // [rsp+300h] [rbp-260h]
  __int64 v163; // [rsp+310h] [rbp-250h] BYREF
  __m128i v164; // [rsp+330h] [rbp-230h] BYREF
  __m128i v165; // [rsp+340h] [rbp-220h] BYREF
  __int64 *v166; // [rsp+350h] [rbp-210h]
  __int64 v167; // [rsp+360h] [rbp-200h] BYREF
  __m128i v168; // [rsp+380h] [rbp-1E0h] BYREF
  _BYTE v169[24]; // [rsp+390h] [rbp-1D0h] BYREF
  __int64 v170; // [rsp+3A8h] [rbp-1B8h]
  void **p_s1; // [rsp+3B0h] [rbp-1B0h]
  __int64 v172; // [rsp+3B8h] [rbp-1A8h]
  char v173; // [rsp+3C8h] [rbp-198h]
  unsigned __int64 *v174; // [rsp+3D0h] [rbp-190h]
  __int64 v175; // [rsp+3D8h] [rbp-188h]
  _BYTE v176[324]; // [rsp+3E0h] [rbp-180h] BYREF
  int v177; // [rsp+524h] [rbp-3Ch]
  __int64 v178; // [rsp+528h] [rbp-38h]

  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_192:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_50208C0 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_192;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_50208C0);
  v8 = (__int64 *)sub_2EAA930(v7 + 176, a2);
  v9 = sub_B6F970(**(_QWORD **)(a2 + 40));
  v10 = "size-info";
  v11 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v9 + 24LL))(v9, "size-info", 9);
  if ( v11 )
  {
    v10 = (const char *)v8[41];
    for ( i = 0; v8 + 40 != (__int64 *)v10; v10 = (const char *)*((_QWORD *)v10 + 1) )
    {
      v12 = *((_QWORD *)v10 + 7);
      if ( v10 + 48 != (const char *)v12 )
      {
        v13 = 0;
        do
        {
          v12 = *(_QWORD *)(v12 + 8);
          ++v13;
        }
        while ( v10 + 48 != (const char *)v12 );
        i += v13;
      }
    }
  }
  n = 0;
  s1 = &v128;
  s2 = &v132;
  v127 = 0;
  v130 = 0;
  v131 = 0;
  if ( dword_4F82DA8[0] && (v41 = sub_BB9590(a1[2], (__int64)v10)) != 0 )
  {
    src = *(unsigned __int8 **)(v41 + 16);
    v124 = *(_QWORD *)(v41 + 24);
    v14 = sub_BC6270(src, v124);
  }
  else
  {
    v124 = 0;
    src = 0;
    v14 = sub_BC6270(0, 0);
  }
  v118 = v14;
  v119 = v14 && dword_4F82DA8[0] != 0;
  if ( v119 )
  {
    v76 = (char *)sub_2E791E0(v8);
    v120 = 0;
    v119 = sub_BC63A0(v76, v77);
    if ( v119 )
    {
      v170 = 0x100000000LL;
      v168.m128i_i64[1] = 2;
      memset(v169, 0, sizeof(v169));
      v168.m128i_i64[0] = (__int64)&unk_49DD288;
      p_s1 = &s1;
      sub_CB5980((__int64)&v168, 0, 0, 0);
      sub_2E823F0((__int64)v8, (__int64)&v168, 0);
      v168.m128i_i64[0] = (__int64)&unk_49DD388;
      sub_CB5840((__int64)&v168);
      v120 = v119;
    }
  }
  else
  {
    v120 = !v14;
  }
  v15 = (unsigned __int8)qword_50200A8;
  v8[43] = (_DWORD)v8[43] & ~*((_DWORD *)a1 + 48) & 0xFFF;
  if ( (_BYTE)v15 )
  {
    sub_3140B20(&v168, 0);
    v168.m128i_i64[0] = (__int64)&unk_4A381E8;
    v113 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
    v114 = v38;
    sub_34E2490(&v168, v113, v38, v8);
    v39 = *(__int64 (__fastcall **)(__int64))(*a1 + 152);
    if ( v39 == sub_2E82CA0 )
    {
      v15 = 0;
      sub_300BC60(a1 + 25);
      v40 = (__m128i *)v113;
    }
    else
    {
      v83 = ((__int64 (__fastcall *)(__int64 *, __int64 *))v39)(a1, v8);
      v40 = (__m128i *)v113;
      v15 = v83;
    }
    v17 = v40;
    sub_34E2510(&v168, v40, v114, v8);
    v168.m128i_i64[0] = (__int64)&unk_4A381E8;
    sub_2272350((__int64)&v168);
  }
  else
  {
    v16 = *(__int64 (__fastcall **)(__int64))(*a1 + 152);
    v17 = (__m128i *)v8;
    if ( v16 == sub_2E82CA0 )
      sub_300BC60(a1 + 25);
    else
      v15 = ((__int64 (__fastcall *)(__int64 *, __int64 *))v16)(a1, v8);
  }
  if ( v11 )
  {
    v17 = (__m128i *)v8[41];
    for ( j = 0; v8 + 40 != (__int64 *)v17; v17 = (__m128i *)v17->m128i_i64[1] )
    {
      v19 = (__m128i *)v17[3].m128i_i64[1];
      if ( &v17[3] != v19 )
      {
        v20 = 0;
        do
        {
          v19 = (__m128i *)v19->m128i_i64[1];
          ++v20;
        }
        while ( &v17[3] != v19 );
        j += v20;
      }
    }
    if ( i != j )
    {
      v145 = (__m128i)(unsigned __int64)v8;
      v21 = sub_B2BE50(*v8);
      if ( sub_B6EA50(v21)
        || (v60 = sub_B2BE50(*(_QWORD *)v145.m128i_i64[0]),
            v61 = sub_B6F970(v60),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v61 + 48LL))(v61)) )
      {
        v111 = j - (unsigned __int64)i;
        v115 = v8[41];
        v22 = sub_B92180(*v8);
        sub_B15890(&v164, v22);
        v23 = _mm_loadu_si128(&v164);
        v24 = **(_QWORD **)(v115 + 32);
        v174 = (unsigned __int64 *)v176;
        *(__m128i *)&v169[8] = v23;
        *(_QWORD *)v169 = v24;
        v170 = (__int64)"size-info";
        p_s1 = (void **)"FunctionMISizeChange";
        v175 = 0x400000000LL;
        v176[320] = 0;
        v168.m128i_i64[0] = (__int64)&unk_4A28EB8;
        v25 = *a1;
        v168.m128i_i32[2] = 21;
        v168.m128i_i8[12] = 2;
        v172 = 20;
        v173 = 0;
        v177 = -1;
        v178 = v115;
        v26 = (_BYTE *)(*(__int64 (__fastcall **)(__int64 *))(v25 + 16))(a1);
        sub_B16430((__int64)&v164, "Pass", 4u, v26, v27);
        v112 = sub_2E82FF0((__int64)&v168, (__int64)&v164);
        sub_B18290(v112, ": Function: ", 0xCu);
        v28 = (char *)sub_BD5D20(a2);
        sub_B16430((__int64)&v160, "Function", 8u, v28, v29);
        v117 = sub_2E82FF0(v112, (__int64)&v160);
        sub_B18290(v117, ": ", 2u);
        sub_B18290(v117, "MI Instruction count changed from ", 0x22u);
        sub_B169E0(v156.m128i_i64, "MIInstrsBefore", 14, i);
        v30 = sub_2E82FF0(v117, (__int64)&v156);
        sub_B18290(v30, " to ", 4u);
        sub_B169E0(v152.m128i_i64, "MIInstrsAfter", 13, j);
        v31 = sub_2E82FF0(v30, (__int64)&v152);
        sub_B18290(v31, "; Delta: ", 9u);
        sub_B167F0(v148.m128i_i64, "Delta", 5, v111);
        sub_2E82FF0(v31, (__int64)&v148);
        if ( v150 != &v151 )
          j_j___libc_free_0((unsigned __int64)v150);
        if ( (__m128i *)v148.m128i_i64[0] != &v149 )
          j_j___libc_free_0(v148.m128i_u64[0]);
        if ( v154 != &v155 )
          j_j___libc_free_0((unsigned __int64)v154);
        if ( (__m128i *)v152.m128i_i64[0] != &v153 )
          j_j___libc_free_0(v152.m128i_u64[0]);
        if ( v158 != &v159 )
          j_j___libc_free_0((unsigned __int64)v158);
        if ( (__m128i *)v156.m128i_i64[0] != &v157 )
          j_j___libc_free_0(v156.m128i_u64[0]);
        if ( v162 != &v163 )
          j_j___libc_free_0((unsigned __int64)v162);
        if ( (__m128i *)v160.m128i_i64[0] != &v161 )
          j_j___libc_free_0(v160.m128i_u64[0]);
        if ( v166 != &v167 )
          j_j___libc_free_0((unsigned __int64)v166);
        if ( (__m128i *)v164.m128i_i64[0] != &v165 )
          j_j___libc_free_0(v164.m128i_u64[0]);
        v17 = &v168;
        sub_2EAFC50(&v145, &v168);
        v32 = v174;
        v168.m128i_i64[0] = (__int64)&unk_49D9D40;
        v33 = &v174[10 * (unsigned int)v175];
        if ( v174 != v33 )
        {
          do
          {
            v33 -= 10;
            v34 = v33[4];
            if ( (unsigned __int64 *)v34 != v33 + 6 )
            {
              v17 = (__m128i *)(v33[6] + 1);
              j_j___libc_free_0(v34);
            }
            if ( (unsigned __int64 *)*v33 != v33 + 2 )
            {
              v17 = (__m128i *)(v33[2] + 1);
              j_j___libc_free_0(*v33);
            }
          }
          while ( v32 != v33 );
          v33 = v174;
        }
        if ( v33 != (unsigned __int64 *)v176 )
          _libc_free((unsigned __int64)v33);
      }
    }
  }
  v8[43] |= a1[23];
  if ( !v120 )
    goto LABEL_70;
  if ( v119 )
  {
    v170 = 0x100000000LL;
    v168.m128i_i64[1] = 2;
    memset(v169, 0, sizeof(v169));
    v168.m128i_i64[0] = (__int64)&unk_49DD288;
    p_s1 = &s2;
    sub_CB5980((__int64)&v168, 0, 0, 0);
    v17 = &v168;
    sub_2E823F0((__int64)v8, (__int64)&v168, 0);
    v168.m128i_i64[0] = (__int64)&unk_49DD388;
    sub_CB5840((__int64)&v168);
  }
  if ( !v118 )
  {
    v35 = dword_4F82DA8[0];
    if ( dword_4F82DA8[0] == 1 || dword_4F82DA8[0] == 3 )
    {
      v36 = " filtered out";
      goto LABEL_126;
    }
LABEL_69:
    if ( v35 != 5 )
      goto LABEL_70;
    v36 = " omitted because no change";
    if ( !v118 )
      v36 = " filtered out";
    goto LABEL_126;
  }
  if ( n == v130 )
  {
    if ( !n || (v17 = (__m128i *)s2, !memcmp(s1, s2, n)) )
    {
      v35 = dword_4F82DA8[0];
      if ( dword_4F82DA8[0] == 1 || dword_4F82DA8[0] == 3 )
      {
        v36 = " omitted because no change";
LABEL_126:
        v63 = sub_CB72A0();
        v64 = (__m128i *)v63[4];
        v65 = (__int64)v63;
        if ( v63[3] - (_QWORD)v64 <= 0x11u )
        {
          v17 = (__m128i *)"*** IR Dump After ";
          v65 = sub_CB6200((__int64)v63, "*** IR Dump After ", 0x12u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4450750);
          v64[1].m128i_i16[0] = 8306;
          *v64 = si128;
          v63[4] += 18LL;
        }
        v67 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64 *, __m128i *))(*a1 + 16))(a1, v17);
        v69 = *(void **)(v65 + 32);
        if ( v68 > *(_QWORD *)(v65 + 24) - (_QWORD)v69 )
        {
          sub_CB6200(v65, v67, v68);
        }
        else if ( v68 )
        {
          v121 = v68;
          memcpy(v69, v67, v68);
          *(_QWORD *)(v65 + 32) += v121;
        }
        if ( v124 )
        {
          v78 = sub_CB72A0();
          v79 = (_WORD *)v78[4];
          v80 = (__int64)v78;
          if ( v78[3] - (_QWORD)v79 <= 1u )
          {
            v90 = sub_CB6200((__int64)v78, (unsigned __int8 *)" (", 2u);
            v81 = *(void **)(v90 + 32);
            v80 = v90;
          }
          else
          {
            *v79 = 10272;
            v81 = (void *)(v78[4] + 2LL);
            v78[4] = v81;
          }
          if ( v124 > *(_QWORD *)(v80 + 24) - (_QWORD)v81 )
          {
            v80 = sub_CB6200(v80, src, v124);
            v82 = *(_BYTE **)(v80 + 32);
          }
          else
          {
            memcpy(v81, src, v124);
            v82 = (_BYTE *)(v124 + *(_QWORD *)(v80 + 32));
            *(_QWORD *)(v80 + 32) = v82;
          }
          if ( *(_BYTE **)(v80 + 24) == v82 )
          {
            sub_CB6200(v80, (unsigned __int8 *)")", 1u);
          }
          else
          {
            *v82 = 41;
            ++*(_QWORD *)(v80 + 32);
          }
        }
        v70 = sub_CB72A0();
        v71 = (_DWORD *)v70[4];
        v72 = (__int64)v70;
        if ( v70[3] - (_QWORD)v71 <= 3u )
        {
          v72 = sub_CB6200((__int64)v70, (unsigned __int8 *)" on ", 4u);
        }
        else
        {
          *v71 = 544108320;
          v70[4] += 4LL;
        }
        v164.m128i_i64[0] = (__int64)" ***\n";
        LOWORD(v166) = 259;
        v73.m128i_i64[0] = (__int64)sub_2E791E0(v8);
        v161.m128i_i64[0] = (__int64)v36;
        v160 = v73;
        v74 = (char)v166;
        LOWORD(v162) = 773;
        if ( (_BYTE)v166 )
        {
          if ( (_BYTE)v166 == 1 )
          {
            v91 = _mm_loadu_si128(&v161);
            v168 = _mm_loadu_si128(&v160);
            *(_QWORD *)&v169[16] = v162;
            *(__m128i *)v169 = v91;
          }
          else
          {
            if ( BYTE1(v166) == 1 )
            {
              v101 = v164.m128i_i64[1];
              v75 = (__m128i *)v164.m128i_i64[0];
            }
            else
            {
              v75 = &v164;
              v74 = 2;
            }
            *(_QWORD *)v169 = v75;
            v168.m128i_i64[0] = (__int64)&v160;
            *(_QWORD *)&v169[8] = v101;
            v169[16] = 2;
            v169[17] = v74;
          }
        }
        else
        {
          *(_WORD *)&v169[16] = 256;
        }
        sub_CA0E80((__int64)&v168, v72);
        goto LABEL_70;
      }
      goto LABEL_69;
    }
  }
  v42 = sub_CB72A0();
  LOWORD(v166) = 259;
  v164.m128i_i64[0] = (__int64)" ***\n";
  v43.m128i_i64[0] = (__int64)sub_2E791E0(v8);
  LOWORD(v158) = 261;
  v156 = v43;
  v148.m128i_i64[0] = (__int64)") on ";
  v144 = 261;
  v142 = src;
  v143 = v124;
  v136[0] = " (";
  v43.m128i_i64[0] = *a1;
  LOWORD(v150) = 259;
  v138 = 1;
  v137 = 3;
  v44.m128i_i64[0] = (*(__int64 (__fastcall **)(__int64 *, __m128i *))(v43.m128i_i64[0] + 16))(a1, v17);
  v133.m128i_i64[0] = (__int64)"*** IR Dump After ";
  v134 = v44;
  v45 = v137;
  LOWORD(v135) = 1283;
  if ( !v137 )
  {
    LOWORD(v141) = 256;
    goto LABEL_115;
  }
  if ( v137 != 1 )
  {
    if ( v138 == 1 )
    {
      v46 = (_QWORD *)v136[0];
      v110 = v136[1];
    }
    else
    {
      v46 = v136;
      v45 = 2;
    }
    BYTE1(v141) = v45;
    v47 = v144;
    v139.m128i_i64[0] = (__int64)&v133;
    v140.m128i_i64[0] = (__int64)v46;
    v140.m128i_i64[1] = v110;
    LOBYTE(v141) = 2;
    if ( (_BYTE)v144 )
    {
      if ( (_BYTE)v144 != 1 )
        goto LABEL_86;
      goto LABEL_168;
    }
LABEL_115:
    LOWORD(v147) = 256;
    goto LABEL_116;
  }
  v92 = _mm_loadu_si128(&v133);
  v93 = _mm_loadu_si128(&v134);
  v141 = v135;
  v47 = v144;
  v139 = v92;
  v140 = v93;
  if ( !(_BYTE)v144 )
    goto LABEL_115;
  if ( (_BYTE)v144 != 1 )
  {
    if ( BYTE1(v141) == 1 )
    {
      v48 = (__m128i *)v139.m128i_i64[0];
      v49 = 3;
      v109 = v139.m128i_i64[1];
LABEL_87:
      if ( HIBYTE(v144) == 1 )
      {
        v50 = v142;
        v108 = v143;
      }
      else
      {
        v50 = &v142;
        v47 = 2;
      }
      BYTE1(v147) = v47;
      v51 = (char)v150;
      v145.m128i_i64[0] = (__int64)v48;
      v145.m128i_i64[1] = v109;
      v146.m128i_i64[0] = (__int64)v50;
      v146.m128i_i64[1] = v108;
      LOBYTE(v147) = v49;
      if ( (_BYTE)v150 )
        goto LABEL_90;
LABEL_116:
      LOWORD(v154) = 256;
      goto LABEL_117;
    }
LABEL_86:
    v48 = &v139;
    v49 = 2;
    goto LABEL_87;
  }
LABEL_168:
  v94 = _mm_loadu_si128(&v140);
  v49 = (char)v141;
  v145 = _mm_loadu_si128(&v139);
  v147 = v141;
  v146 = v94;
  if ( !(_BYTE)v141 )
    goto LABEL_116;
  v51 = (char)v150;
  if ( !(_BYTE)v150 )
    goto LABEL_116;
  if ( (_BYTE)v141 == 1 )
  {
    v95 = _mm_loadu_si128(&v149);
    v152 = _mm_loadu_si128(&v148);
    v154 = v150;
    v153 = v95;
    goto LABEL_96;
  }
LABEL_90:
  if ( v51 == 1 )
  {
    v100 = _mm_loadu_si128(&v146);
    v51 = (char)v147;
    v152 = _mm_loadu_si128(&v145);
    v154 = v147;
    v153 = v100;
    if ( !(_BYTE)v147 )
      goto LABEL_117;
  }
  else
  {
    if ( BYTE1(v147) == 1 )
    {
      v107 = v145.m128i_i64[1];
      v52 = (__m128i *)v145.m128i_i64[0];
    }
    else
    {
      v52 = &v145;
      v49 = 2;
    }
    if ( BYTE1(v150) == 1 )
    {
      v106 = v148.m128i_i64[1];
      v53 = (__m128i *)v148.m128i_i64[0];
    }
    else
    {
      v53 = &v148;
      v51 = 2;
    }
    v152.m128i_i64[0] = (__int64)v52;
    v153.m128i_i64[0] = (__int64)v53;
    v152.m128i_i64[1] = v107;
    LOBYTE(v154) = v49;
    v153.m128i_i64[1] = v106;
    BYTE1(v154) = v51;
    v51 = v49;
  }
LABEL_96:
  v54 = (char)v158;
  if ( !(_BYTE)v158 )
  {
LABEL_117:
    LOWORD(v162) = 256;
    goto LABEL_118;
  }
  if ( v51 == 1 )
  {
    v96 = _mm_loadu_si128(&v157);
    v160 = _mm_loadu_si128(&v156);
    v162 = v158;
    v161 = v96;
  }
  else if ( (_BYTE)v158 == 1 )
  {
    v98 = _mm_loadu_si128(&v153);
    v54 = (char)v154;
    v160 = _mm_loadu_si128(&v152);
    v162 = v154;
    v161 = v98;
    if ( !(_BYTE)v154 )
      goto LABEL_118;
  }
  else
  {
    if ( BYTE1(v154) == 1 )
    {
      v105 = v152.m128i_i64[1];
      v55 = (__m128i *)v152.m128i_i64[0];
    }
    else
    {
      v55 = &v152;
      v51 = 2;
    }
    if ( BYTE1(v158) == 1 )
    {
      v104 = v156.m128i_i64[1];
      v56 = (__m128i *)v156.m128i_i64[0];
    }
    else
    {
      v56 = &v156;
      v54 = 2;
    }
    v160.m128i_i64[0] = (__int64)v55;
    v161.m128i_i64[0] = (__int64)v56;
    v160.m128i_i64[1] = v105;
    LOBYTE(v162) = v51;
    v161.m128i_i64[1] = v104;
    BYTE1(v162) = v54;
    v54 = v51;
  }
  v57 = (char)v166;
  if ( (_BYTE)v166 )
  {
    if ( v54 == 1 )
    {
      v97 = _mm_loadu_si128(&v165);
      v168 = _mm_loadu_si128(&v164);
      *(_QWORD *)&v169[16] = v166;
      *(__m128i *)v169 = v97;
    }
    else if ( (_BYTE)v166 == 1 )
    {
      v99 = _mm_loadu_si128(&v161);
      v168 = _mm_loadu_si128(&v160);
      *(_QWORD *)&v169[16] = v162;
      *(__m128i *)v169 = v99;
    }
    else
    {
      if ( BYTE1(v162) == 1 )
      {
        v103 = v160.m128i_i64[1];
        v58 = (__m128i *)v160.m128i_i64[0];
      }
      else
      {
        v58 = &v160;
        v54 = 2;
      }
      if ( BYTE1(v166) == 1 )
      {
        v102 = v164.m128i_i64[1];
        v59 = (__m128i *)v164.m128i_i64[0];
      }
      else
      {
        v59 = &v164;
        v57 = 2;
      }
      v168.m128i_i64[0] = (__int64)v58;
      *(_QWORD *)v169 = v59;
      v168.m128i_i64[1] = v103;
      v169[16] = v54;
      *(_QWORD *)&v169[8] = v102;
      v169[17] = v57;
    }
    goto LABEL_119;
  }
LABEL_118:
  *(_WORD *)&v169[16] = 256;
LABEL_119:
  sub_CA0E80((__int64)&v168, (__int64)v42);
  if ( dword_4F82DA8[0] > 6 )
  {
    if ( (unsigned int)(dword_4F82DA8[0] - 7) > 1 )
      goto LABEL_70;
LABEL_123:
    v62 = sub_CB72A0();
    sub_CB6200((__int64)v62, (unsigned __int8 *)s2, v130);
    goto LABEL_70;
  }
  if ( dword_4F82DA8[0] > 2 )
  {
    if ( (unsigned int)(dword_4F82DA8[0] - 5) <= 1 )
    {
      v84 = "\x1B[31m-%l\x1B[0m\n";
      v85 = 13;
      v86 = "\x1B[32m+%l\x1B[0m\n";
    }
    else
    {
      v84 = "-%l\n";
      v85 = 4;
      v86 = "+%l\n";
    }
    srca = v84;
    v87 = strlen(v86);
    v88 = sub_CB72A0();
    sub_BC7A80(&v168, (__int64)s1, n, (__int64)s2, v130, v89, srca, v85, v86, v87, " %l\n", 4);
    sub_CB6200((__int64)v88, (unsigned __int8 *)v168.m128i_i64[0], v168.m128i_u64[1]);
    if ( (_BYTE *)v168.m128i_i64[0] != v169 )
      j_j___libc_free_0(v168.m128i_u64[0]);
  }
  else
  {
    if ( !dword_4F82DA8[0] )
      BUG();
    if ( (unsigned int)(dword_4F82DA8[0] - 1) <= 1 )
      goto LABEL_123;
  }
LABEL_70:
  if ( s2 != &v132 )
    _libc_free((unsigned __int64)s2);
  if ( s1 != &v128 )
    _libc_free((unsigned __int64)s1);
  return v15;
}
