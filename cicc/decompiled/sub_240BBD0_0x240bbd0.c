// Function: sub_240BBD0
// Address: 0x240bbd0
//
__int64 __fastcall sub_240BBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  int v11; // ebx
  unsigned int i; // eax
  __int64 v13; // r9
  unsigned int v14; // eax
  __int64 v16; // r13
  __int64 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rax
  const void *v23; // r14
  size_t v24; // r13
  __int64 v25; // rbx
  int v26; // eax
  int v27; // eax
  __int64 v28; // rax
  const char *v29; // r14
  size_t v30; // rdx
  size_t v31; // r13
  __int64 v32; // rbx
  int v33; // eax
  int v34; // eax
  __int64 v35; // rax
  unsigned __int64 *v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 *v39; // rsi
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 **v42; // rbx
  __int64 *v43; // r15
  __int64 **v44; // r14
  __int64 **v45; // rbx
  __int64 *v46; // rsi
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 **v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  _QWORD ***v53; // r15
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // r9
  __m128i v57; // xmm1
  __m128i v58; // xmm2
  __m128i v59; // xmm3
  __int64 *v60; // r14
  __int64 *v61; // rbx
  __int64 v62; // r15
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  unsigned int v67; // r15d
  __int64 ***v68; // r14
  __int64 *v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rax
  __int64 ***v72; // rbx
  __int64 v73; // rcx
  __int64 v74; // r12
  char *v75; // rax
  unsigned __int64 v76; // r9
  char v77; // bl
  unsigned __int64 *v78; // rbx
  unsigned __int64 *v79; // r14
  __int64 v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // r15
  unsigned __int64 v83; // r12
  unsigned __int64 v84; // rdi
  unsigned __int64 v85; // rdi
  unsigned __int64 v86; // r15
  unsigned __int64 v87; // r12
  unsigned __int64 v88; // rdi
  unsigned __int64 v89; // r13
  __int64 v90; // rax
  __int64 v91; // r12
  __int64 v92; // r15
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 *v95; // r15
  __int64 *v96; // r14
  __int64 v97; // rsi
  __int64 *v98; // r14
  __int64 v99; // r13
  __int64 v100; // r15
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __m128i v104; // xmm5
  __m128i v105; // xmm7
  __int64 v106; // rdx
  __int64 *v107; // [rsp+40h] [rbp-650h]
  char v108; // [rsp+4Fh] [rbp-641h]
  __int64 v109; // [rsp+50h] [rbp-640h]
  __int64 v110; // [rsp+50h] [rbp-640h]
  char *v111; // [rsp+50h] [rbp-640h]
  __int64 **v112; // [rsp+58h] [rbp-638h]
  __int64 **v114; // [rsp+70h] [rbp-620h] BYREF
  __int64 v115; // [rsp+78h] [rbp-618h]
  _BYTE v116[64]; // [rsp+80h] [rbp-610h] BYREF
  __int64 **v117; // [rsp+C0h] [rbp-5D0h] BYREF
  __int64 v118; // [rsp+C8h] [rbp-5C8h]
  _QWORD v119[2]; // [rsp+D0h] [rbp-5C0h] BYREF
  __int64 *v120; // [rsp+E0h] [rbp-5B0h]
  __int64 v121; // [rsp+F0h] [rbp-5A0h] BYREF
  __int64 *v122; // [rsp+110h] [rbp-580h] BYREF
  __int64 v123; // [rsp+118h] [rbp-578h]
  _QWORD v124[2]; // [rsp+120h] [rbp-570h] BYREF
  __int64 *v125; // [rsp+130h] [rbp-560h]
  __int64 v126; // [rsp+140h] [rbp-550h] BYREF
  __int64 *v127; // [rsp+160h] [rbp-530h] BYREF
  __int64 v128; // [rsp+168h] [rbp-528h]
  __int64 v129; // [rsp+170h] [rbp-520h] BYREF
  __int64 v130; // [rsp+178h] [rbp-518h]
  _BYTE *v131; // [rsp+180h] [rbp-510h]
  __int64 v132; // [rsp+188h] [rbp-508h]
  _QWORD v133[2]; // [rsp+190h] [rbp-500h] BYREF
  __m128i v134; // [rsp+1A0h] [rbp-4F0h] BYREF
  unsigned __int8 *v135[4]; // [rsp+1B0h] [rbp-4E0h] BYREF
  __int64 v136; // [rsp+1D0h] [rbp-4C0h]
  __int64 *v137; // [rsp+1D8h] [rbp-4B8h]
  __int64 v138; // [rsp+1E0h] [rbp-4B0h]
  unsigned __int64 v139; // [rsp+1E8h] [rbp-4A8h]
  unsigned __int64 v140; // [rsp+1F0h] [rbp-4A0h]
  __int64 v141; // [rsp+1F8h] [rbp-498h]
  __int64 v142; // [rsp+200h] [rbp-490h]
  __int64 v143; // [rsp+208h] [rbp-488h]
  __int64 v144; // [rsp+210h] [rbp-480h]
  __int64 v145; // [rsp+218h] [rbp-478h]
  __int64 v146; // [rsp+220h] [rbp-470h]
  __int64 v147; // [rsp+228h] [rbp-468h]
  __int64 v148; // [rsp+230h] [rbp-460h]
  __int64 v149; // [rsp+238h] [rbp-458h]
  __int64 v150; // [rsp+240h] [rbp-450h]
  __int64 v151; // [rsp+248h] [rbp-448h]
  __int64 v152; // [rsp+250h] [rbp-440h]
  __int64 v153; // [rsp+258h] [rbp-438h]
  __int64 v154; // [rsp+260h] [rbp-430h]
  __int64 v155; // [rsp+268h] [rbp-428h]
  __int64 v156; // [rsp+270h] [rbp-420h]
  __int64 v157; // [rsp+278h] [rbp-418h]
  __int64 v158; // [rsp+280h] [rbp-410h]
  __int64 v159; // [rsp+288h] [rbp-408h]
  unsigned int v160; // [rsp+290h] [rbp-400h]
  __int64 v161; // [rsp+298h] [rbp-3F8h]
  __int64 v162; // [rsp+2A0h] [rbp-3F0h]
  __int64 v163; // [rsp+2A8h] [rbp-3E8h]
  unsigned int v164; // [rsp+2B0h] [rbp-3E0h]
  __int64 v165; // [rsp+2B8h] [rbp-3D8h]
  unsigned __int64 *v166; // [rsp+2C0h] [rbp-3D0h]
  __int64 v167; // [rsp+2C8h] [rbp-3C8h]
  __int64 v168; // [rsp+2D0h] [rbp-3C0h]
  __int64 v169; // [rsp+2D8h] [rbp-3B8h]
  __int64 v170; // [rsp+2E0h] [rbp-3B0h]
  __int64 v171; // [rsp+2E8h] [rbp-3A8h]
  unsigned int v172; // [rsp+2F0h] [rbp-3A0h]
  void *src; // [rsp+300h] [rbp-390h] BYREF
  __int64 v174; // [rsp+308h] [rbp-388h]
  __int64 v175; // [rsp+310h] [rbp-380h] BYREF
  __m128i v176; // [rsp+318h] [rbp-378h] BYREF
  __int64 v177; // [rsp+328h] [rbp-368h]
  __m128i v178; // [rsp+330h] [rbp-360h] BYREF
  __m128i v179; // [rsp+340h] [rbp-350h]
  _BYTE *v180; // [rsp+350h] [rbp-340h] BYREF
  __int64 v181; // [rsp+358h] [rbp-338h]
  _BYTE v182[320]; // [rsp+360h] [rbp-330h] BYREF
  char v183; // [rsp+4A0h] [rbp-1F0h]
  int v184; // [rsp+4A4h] [rbp-1ECh]
  __int64 v185; // [rsp+4A8h] [rbp-1E8h]
  void *v186; // [rsp+4B0h] [rbp-1E0h] BYREF
  __int64 v187; // [rsp+4B8h] [rbp-1D8h]
  __int64 v188; // [rsp+4C0h] [rbp-1D0h] BYREF
  __m128i v189; // [rsp+4C8h] [rbp-1C8h] BYREF
  __int64 v190; // [rsp+4D8h] [rbp-1B8h]
  __m128i v191; // [rsp+4E0h] [rbp-1B0h] BYREF
  __m128i v192; // [rsp+4F0h] [rbp-1A0h] BYREF
  char v193[8]; // [rsp+500h] [rbp-190h] BYREF
  unsigned int v194; // [rsp+508h] [rbp-188h]
  _BYTE v195[276]; // [rsp+540h] [rbp-150h] BYREF
  int v196; // [rsp+654h] [rbp-3Ch]
  __int64 v197; // [rsp+658h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v7 = *(_QWORD *)(a3 + 40);
  v8 = *(_QWORD *)(v6 + 8);
  v9 = *(unsigned int *)(v8 + 88);
  v10 = *(_QWORD *)(v8 + 72);
  if ( !(_DWORD)v9 )
    goto LABEL_7;
  v11 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))); ; i = (v9 - 1) & v14 )
  {
    v13 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F87C68 && v7 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_7;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v10 + 24 * v9 )
    goto LABEL_7;
  v16 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
  if ( !v16 )
    goto LABEL_7;
  v17 = &v188;
  v187 = 1;
  do
  {
    *v17 = -4096;
    v17 += 2;
  }
  while ( v17 != (__int64 *)v195 );
  if ( (v187 & 1) == 0 )
    sub_C7D6A0(v188, 16LL * v189.m128i_u32[0], 8);
  if ( !*(_QWORD *)(v16 + 16) )
  {
LABEL_7:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v18 = sub_BC1CD0(a4, &unk_4F8D9A8, a3);
  v19 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v109 = sub_BC1CD0(a4, &unk_4FDBD00, a3) + 8;
  v20 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v21 = v109;
  v135[0] = (unsigned __int8 *)a3;
  v135[1] = (unsigned __int8 *)(v18 + 8);
  v137 = (__int64 *)(v20 + 8);
  v135[2] = (unsigned __int8 *)(v19 + 8);
  v135[3] = (unsigned __int8 *)(v16 + 8);
  v136 = v109;
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
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v108 = qword_4FE2DE8;
  if ( (_BYTE)qword_4FE2DE8 )
  {
    v108 = 0;
    goto LABEL_89;
  }
  if ( !(_BYTE)qword_4FE2D08 )
  {
    if ( qword_4FE2950 | qword_4FE2A50 )
    {
      v22 = *(_QWORD *)(a3 + 40);
      v23 = *(const void **)(v22 + 168);
      v24 = *(_QWORD *)(v22 + 176);
      v25 = qword_4FE27B0 + 8LL * (unsigned int)qword_4FE27B8;
      v26 = sub_C92610();
      v27 = sub_C92860(&qword_4FE27B0, v23, v24, v26);
      if ( v27 == -1 )
        v28 = qword_4FE27B0 + 8LL * (unsigned int)qword_4FE27B8;
      else
        v28 = qword_4FE27B0 + 8LL * v27;
      if ( v25 == v28 )
      {
        v29 = sub_BD5D20(a3);
        v31 = v30;
        v32 = qword_4FE2790 + 8LL * (unsigned int)qword_4FE2798;
        v33 = sub_C92610();
        v34 = sub_C92860(&qword_4FE2790, v29, v31, v33);
        v35 = v34 == -1 ? qword_4FE2790 + 8LL * (unsigned int)qword_4FE2798 : qword_4FE2790 + 8LL * v34;
        if ( v32 == v35 )
          goto LABEL_84;
      }
    }
    else
    {
      if ( !*(_QWORD *)(v16 + 16) )
        goto LABEL_89;
      sub_B2EE70((__int64)&v186, a3, 0);
      if ( !(_BYTE)v188 || !sub_D84440(v16 + 8, (unsigned __int64)v186) )
      {
        v108 = 0;
        goto LABEL_84;
      }
    }
    v21 = v136;
  }
  v36 = *(unsigned __int64 **)(v21 + 32);
  v114 = (__int64 **)v116;
  v115 = 0x800000000LL;
  v39 = sub_2406750(v135, v36, (__int64)&v114);
  if ( v39 )
    sub_23FABC0((__int64)&v114, (__int64)v39, v37, v38, v40, v41);
  v42 = v114;
  v117 = (__int64 **)v119;
  v118 = 0x800000000LL;
  if ( v114 == &v114[(unsigned int)v115] )
    goto LABEL_129;
  v112 = &v114[(unsigned int)v115];
  do
  {
    v43 = *v42;
    src = 0;
    v174 = 0;
    v175 = 0;
    v176.m128i_i64[0] = 0;
    sub_23FC070(v43, (__int64)&src);
    sub_2401D40((__int64)&v186, (__int64)v135, (__int64)v43, 0, 0, 0, (__int64)&v117, (__int64)&src);
    if ( v186 != &v188 )
      _libc_free((unsigned __int64)v186);
    ++v42;
    sub_C7D6A0(v174, 8LL * v176.m128i_u32[0], 8);
  }
  while ( v112 != v42 );
  v44 = v117;
  v45 = &v117[(unsigned int)v118];
  if ( v117 == v45 )
  {
LABEL_129:
    v122 = v124;
    v123 = 0x800000000LL;
LABEL_130:
    src = &v175;
    v174 = 0x800000000LL;
LABEL_131:
    v68 = (__int64 ***)&v188;
    v72 = (__int64 ***)&v188;
    v186 = &v188;
    v187 = 0x800000000LL;
LABEL_132:
    sub_23FCE80(v68, v72, (__int64)sub_23FAAB0);
    v76 = 0;
    goto LABEL_72;
  }
  do
  {
    v46 = *v44++;
    sub_2407140((__int64)v135, v46, (__int64)v46);
  }
  while ( v45 != v44 );
  v48 = (__int64)v117;
  v49 = &v117[(unsigned int)v118];
  v122 = v124;
  v123 = 0x800000000LL;
  if ( v117 == v49 )
    goto LABEL_130;
  v50 = (__int64)v117;
  do
  {
    while ( 1 )
    {
      v53 = *(_QWORD ****)v50;
      if ( *(_DWORD *)(*(_QWORD *)v50 + 1768LL)
         + *(_DWORD *)(*(_QWORD *)v50 + 1736LL)
         + *(_DWORD *)(*(_QWORD *)v50 + 888LL)
         + *(_DWORD *)(*(_QWORD *)v50 + 920LL) >= (unsigned int)qword_4FE2B48 )
      {
        v51 = (unsigned int)v123;
        v52 = (unsigned int)v123 + 1LL;
        if ( v52 > HIDWORD(v123) )
        {
          sub_C8D5F0((__int64)&v122, v124, v52, 8u, v47, v48);
          v51 = (unsigned int)v123;
        }
        v122[v51] = (__int64)v53;
        LODWORD(v123) = v123 + 1;
        goto LABEL_41;
      }
      v107 = v137;
      v110 = *v137;
      v54 = sub_B2BE50(*v137);
      if ( sub_B6EA50(v54) )
        break;
      v93 = sub_B2BE50(v110);
      v94 = sub_B6F970(v93);
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v94 + 48LL))(v94) )
        break;
LABEL_41:
      v50 += 8;
      if ( v49 == (__int64 **)v50 )
        goto LABEL_55;
    }
    v55 = sub_986580(***v53 & 0xFFFFFFFFFFFFFFF8LL);
    sub_B176B0((__int64)&v186, (__int64)"chr", (__int64)"DropScopeWithOneBranchOrSelect", 30, v55);
    sub_B18290((__int64)&v186, "Drop scope with < ", 0x12u);
    sub_B169E0((__int64 *)&v127, "CHRMergeThreshold", 17, qword_4FE2B48);
    src = &v175;
    sub_23FAF10((__int64 *)&src, v127, (__int64)v127 + v128);
    v176.m128i_i64[1] = (__int64)&v178;
    sub_23FAF10(&v176.m128i_i64[1], v131, (__int64)&v131[v132]);
    v179 = _mm_loadu_si128(&v134);
    sub_B180C0((__int64)&v186, (unsigned __int64)&src);
    if ( (__m128i *)v176.m128i_i64[1] != &v178 )
      j_j___libc_free_0(v176.m128i_u64[1]);
    if ( src != &v175 )
      j_j___libc_free_0((unsigned __int64)src);
    sub_B18290((__int64)&v186, " biased branch(es) or select(s)", 0x1Fu);
    v57 = _mm_loadu_si128(&v189);
    v181 = 0x400000000LL;
    v58 = _mm_loadu_si128(&v191);
    LODWORD(v174) = v187;
    v59 = _mm_loadu_si128(&v192);
    v176 = v57;
    BYTE4(v174) = BYTE4(v187);
    v178 = v58;
    v175 = v188;
    v179 = v59;
    src = &unk_49D9D40;
    v177 = v190;
    v180 = v182;
    if ( v194 )
      sub_23FE010((__int64)&v180, (__int64)v193, (__int64)v182, v194, (__int64)v193, v56);
    v183 = v195[272];
    v184 = v196;
    v185 = v197;
    src = &unk_49D9DB0;
    if ( v131 != (_BYTE *)v133 )
      j_j___libc_free_0((unsigned __int64)v131);
    if ( v127 != &v129 )
      j_j___libc_free_0((unsigned __int64)v127);
    v50 += 8;
    v186 = &unk_49D9D40;
    sub_23FD590((__int64)v193);
    sub_1049740(v107, (__int64)&src);
    src = &unk_49D9D40;
    sub_23FD590((__int64)&v180);
  }
  while ( v49 != (__int64 **)v50 );
LABEL_55:
  v60 = v122;
  v61 = &v122[(unsigned int)v123];
  src = &v175;
  v174 = 0x800000000LL;
  if ( v61 == v122 )
    goto LABEL_131;
  do
  {
    v62 = *v60;
    sub_2400A70((__int64)v135, *v60, *v60);
    v65 = (unsigned int)v174;
    v66 = (unsigned int)v174 + 1LL;
    if ( v66 > HIDWORD(v174) )
    {
      sub_C8D5F0((__int64)&src, &v175, v66, 8u, v63, v64);
      v65 = (unsigned int)v174;
    }
    ++v60;
    *((_QWORD *)src + v65) = v62;
    v67 = v174 + 1;
    LODWORD(v174) = v174 + 1;
  }
  while ( v61 != v60 );
  v186 = &v188;
  v187 = 0x800000000LL;
  if ( !v67 )
  {
    v72 = (__int64 ***)&v188;
    v68 = (__int64 ***)&v188;
    goto LABEL_132;
  }
  v68 = (__int64 ***)&v188;
  v69 = &v188;
  if ( v67 > 8uLL )
  {
    sub_C8D5F0((__int64)&v186, &v188, v67, 8u, v63, v64);
    v68 = (__int64 ***)v186;
    v69 = (__int64 *)((char *)v186 + 8 * (unsigned int)v187);
  }
  v70 = v67;
  if ( &v68[v70] != (__int64 ***)v69 )
  {
    do
    {
      if ( v69 )
        *v69 = 0;
      ++v69;
    }
    while ( &v68[v70] != (__int64 ***)v69 );
    v68 = (__int64 ***)v186;
  }
  LODWORD(v187) = v67;
  if ( !(8LL * (unsigned int)v174) )
  {
    v72 = &v68[v70];
    v73 = (v70 * 8) >> 3;
    goto LABEL_69;
  }
  memmove(v68, src, 8LL * (unsigned int)v174);
  v68 = (__int64 ***)v186;
  v71 = 8LL * (unsigned int)v187;
  v72 = (__int64 ***)((char *)v186 + v71);
  v73 = v71 >> 3;
  if ( !v71 )
    goto LABEL_132;
LABEL_69:
  v74 = v73;
  while ( 1 )
  {
    v75 = (char *)sub_2207800(8 * v74);
    if ( v75 )
      break;
    v74 >>= 1;
    if ( !v74 )
      goto LABEL_132;
  }
  v111 = v75;
  sub_23FCBD0(v68, v72, v75, v74, (__int64)sub_23FAAB0);
  v76 = (unsigned __int64)v111;
LABEL_72:
  v77 = 0;
  j_j___libc_free_0(v76);
  if ( (_DWORD)v187 )
  {
    v95 = (__int64 *)v186;
    v127 = 0;
    v128 = 0;
    v129 = 0;
    v96 = (__int64 *)((char *)v186 + 8 * (unsigned int)v187);
    v130 = 0;
    do
    {
      v97 = *v95++;
      sub_2408DA0((char *)v135, v97, (__int64)&v127);
    }
    while ( v96 != v95 );
    v77 = 1;
    sub_C7D6A0(v128, 8LL * (unsigned int)v130, 8);
  }
  if ( v186 != &v188 )
    _libc_free((unsigned __int64)v186);
  if ( src != &v175 )
    _libc_free((unsigned __int64)src);
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
  if ( v117 != v119 )
    _libc_free((unsigned __int64)v117);
  if ( v114 != (__int64 **)v116 )
    _libc_free((unsigned __int64)v114);
  if ( v77 )
  {
    v98 = v137;
    if ( (unsigned __int8)sub_23FAEB0(*v137) )
    {
      sub_B17560((__int64)&v186, (__int64)"chr", (__int64)"Stats", 5, (__int64)v135[0]);
      sub_B16080((__int64)&v127, "Function", 8, v135[0]);
      src = &v175;
      sub_23FAF10((__int64 *)&src, v127, (__int64)v127 + v128);
      v176.m128i_i64[1] = (__int64)&v178;
      sub_23FAF10(&v176.m128i_i64[1], v131, (__int64)&v131[v132]);
      v179 = _mm_loadu_si128(&v134);
      sub_B180C0((__int64)&v186, (unsigned __int64)&src);
      if ( (__m128i *)v176.m128i_i64[1] != &v178 )
        j_j___libc_free_0(v176.m128i_u64[1]);
      if ( src != &v175 )
        j_j___libc_free_0((unsigned __int64)src);
      sub_B18290((__int64)&v186, " ", 1u);
      sub_B18290((__int64)&v186, "Reduced the number of branches in hot paths by ", 0x2Fu);
      sub_B16B10((__int64 *)&v122, "NumBranchesDelta", 16, v139);
      v99 = sub_23FD640((__int64)&v186, (__int64)&v122);
      sub_B18290(v99, " (static) and ", 0xEu);
      sub_B16B10((__int64 *)&v117, "WeightedNumBranchesDelta", 24, v140);
      v100 = sub_23FD640(v99, (__int64)&v117);
      sub_B18290(v100, " (weighted by PGO count)", 0x18u);
      LODWORD(v174) = *(_DWORD *)(v100 + 8);
      BYTE4(v174) = *(_BYTE *)(v100 + 12);
      v175 = *(_QWORD *)(v100 + 16);
      v104 = _mm_loadu_si128((const __m128i *)(v100 + 24));
      src = &unk_49D9D40;
      v176 = v104;
      v177 = *(_QWORD *)(v100 + 40);
      v178 = _mm_loadu_si128((const __m128i *)(v100 + 48));
      v105 = _mm_loadu_si128((const __m128i *)(v100 + 64));
      v180 = v182;
      v181 = 0x400000000LL;
      v179 = v105;
      v106 = *(unsigned int *)(v100 + 88);
      if ( (_DWORD)v106 )
        sub_23FE010((__int64)&v180, v100 + 80, v106, v101, v102, v103);
      v183 = *(_BYTE *)(v100 + 416);
      v184 = *(_DWORD *)(v100 + 420);
      v185 = *(_QWORD *)(v100 + 424);
      src = &unk_49D9D78;
      if ( v120 != &v121 )
        j_j___libc_free_0((unsigned __int64)v120);
      if ( v117 != v119 )
        j_j___libc_free_0((unsigned __int64)v117);
      if ( v125 != &v126 )
        j_j___libc_free_0((unsigned __int64)v125);
      if ( v122 != v124 )
        j_j___libc_free_0((unsigned __int64)v122);
      if ( v131 != (_BYTE *)v133 )
        j_j___libc_free_0((unsigned __int64)v131);
      if ( v127 != &v129 )
        j_j___libc_free_0((unsigned __int64)v127);
      v186 = &unk_49D9D40;
      sub_23FD590((__int64)v193);
      sub_1049740(v98, (__int64)&src);
      src = &unk_49D9D40;
      sub_23FD590((__int64)&v180);
    }
    v108 = v77;
  }
LABEL_84:
  v78 = v166;
  v79 = &v166[(unsigned int)v168];
  if ( (_DWORD)v167 && v166 != v79 )
  {
    while ( *v78 == -8192 || *v78 == -4096 )
    {
      if ( ++v78 == v79 )
        goto LABEL_89;
    }
LABEL_111:
    if ( v78 != v79 )
    {
      v89 = *v78;
      if ( *v78 )
      {
        v90 = *(unsigned int *)(v89 + 1808);
        if ( (_DWORD)v90 )
        {
          v91 = *(_QWORD *)(v89 + 1792);
          v92 = v91 + 40 * v90;
          do
          {
            if ( *(_QWORD *)v91 != -8192 && *(_QWORD *)v91 != -4096 )
              sub_C7D6A0(*(_QWORD *)(v91 + 16), 8LL * *(unsigned int *)(v91 + 32), 8);
            v91 += 40;
          }
          while ( v92 != v91 );
          v90 = *(unsigned int *)(v89 + 1808);
        }
        sub_C7D6A0(*(_QWORD *)(v89 + 1792), 40 * v90, 8);
        sub_C7D6A0(*(_QWORD *)(v89 + 1760), 8LL * *(unsigned int *)(v89 + 1776), 8);
        sub_C7D6A0(*(_QWORD *)(v89 + 1728), 8LL * *(unsigned int *)(v89 + 1744), 8);
        v82 = *(_QWORD *)(v89 + 936);
        v83 = v82 + 96LL * *(unsigned int *)(v89 + 944);
        if ( v82 != v83 )
        {
          do
          {
            v83 -= 96LL;
            v84 = *(_QWORD *)(v83 + 16);
            if ( v84 != v83 + 32 )
              _libc_free(v84);
          }
          while ( v82 != v83 );
          v83 = *(_QWORD *)(v89 + 936);
        }
        if ( v83 != v89 + 952 )
          _libc_free(v83);
        sub_C7D6A0(*(_QWORD *)(v89 + 912), 8LL * *(unsigned int *)(v89 + 928), 8);
        sub_C7D6A0(*(_QWORD *)(v89 + 880), 8LL * *(unsigned int *)(v89 + 896), 8);
        v85 = *(_QWORD *)(v89 + 784);
        if ( v85 != v89 + 800 )
          _libc_free(v85);
        v86 = *(_QWORD *)v89;
        v87 = *(_QWORD *)v89 + 96LL * *(unsigned int *)(v89 + 8);
        if ( *(_QWORD *)v89 != v87 )
        {
          do
          {
            v87 -= 96LL;
            v88 = *(_QWORD *)(v87 + 16);
            if ( v88 != v87 + 32 )
              _libc_free(v88);
          }
          while ( v86 != v87 );
          v87 = *(_QWORD *)v89;
        }
        if ( v87 != v89 + 16 )
          _libc_free(v87);
        j_j___libc_free_0(v89);
      }
      while ( ++v78 != v79 )
      {
        if ( *v78 != -8192 && *v78 != -4096 )
          goto LABEL_111;
      }
    }
  }
LABEL_89:
  sub_C7D6A0(v170, 16LL * v172, 8);
  sub_C7D6A0((__int64)v166, 8LL * (unsigned int)v168, 8);
  sub_C7D6A0(v162, 16LL * v164, 8);
  sub_C7D6A0(v158, 16LL * v160, 8);
  sub_C7D6A0(v154, 8LL * (unsigned int)v156, 8);
  sub_C7D6A0(v150, 8LL * (unsigned int)v152, 8);
  sub_C7D6A0(v146, 8LL * (unsigned int)v148, 8);
  sub_C7D6A0(v142, 8LL * (unsigned int)v144, 8);
  v80 = a1 + 32;
  v81 = a1 + 80;
  if ( v108 )
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v80;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v81;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v81;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
  }
  return a1;
}
