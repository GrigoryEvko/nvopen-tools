// Function: sub_2DE4000
// Address: 0x2de4000
//
__int64 __fastcall sub_2DE4000(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rbx
  unsigned int v26; // eax
  _QWORD **v27; // r15
  _QWORD **i; // rbx
  __int64 v29; // rax
  _QWORD *v30; // r12
  unsigned __int64 v31; // r13
  __int64 v32; // rdi
  unsigned int v33; // eax
  _QWORD *v34; // rbx
  _QWORD *v35; // r12
  __int64 v36; // rdi
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r12
  bool v42; // al
  bool v43; // r15
  _QWORD *v44; // rbx
  _QWORD *v45; // r13
  unsigned __int64 v46; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // rdx
  _QWORD *v55; // rbx
  _QWORD *v56; // r13
  unsigned __int64 v57; // rsi
  _QWORD *v58; // rax
  _QWORD *v59; // rdi
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // rax
  _QWORD *v63; // rdi
  __int64 v64; // rcx
  __int64 v65; // rdx
  _QWORD *v66; // rbx
  _QWORD *v67; // r13
  unsigned __int64 v68; // rsi
  _QWORD *v69; // rax
  _QWORD *v70; // rdi
  __int64 v71; // rcx
  __int64 v72; // rdx
  __int64 v73; // rax
  _QWORD *v74; // rdi
  __int64 v75; // rcx
  __int64 v76; // rdx
  _QWORD *v77; // rbx
  _QWORD *v78; // r13
  unsigned __int64 v79; // rsi
  _QWORD *v80; // rax
  _QWORD *v81; // rdi
  __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // rax
  _QWORD *v85; // rdi
  __int64 v86; // rcx
  __int64 v87; // rdx
  _QWORD *v88; // rbx
  _QWORD *v89; // r13
  unsigned __int64 v90; // rsi
  _QWORD *v91; // rax
  _QWORD *v92; // rdi
  __int64 v93; // rcx
  __int64 v94; // rdx
  __int64 v95; // rax
  _QWORD *v96; // rdi
  __int64 v97; // rcx
  __int64 v98; // rdx
  _QWORD *v99; // rbx
  _QWORD *v100; // r13
  unsigned __int64 v101; // rsi
  _QWORD *v102; // rax
  _QWORD *v103; // rdi
  __int64 v104; // rcx
  __int64 v105; // rdx
  __int64 v106; // rax
  _QWORD *v107; // rdi
  __int64 v108; // rcx
  __int64 v109; // rdx
  _QWORD *v110; // r13
  _QWORD **v111; // r12
  _QWORD **j; // rbx
  _QWORD *v113; // rsi
  __m128i v115; // xmm3
  __int64 v117; // [rsp+8h] [rbp-118h]
  __int64 v118; // [rsp+10h] [rbp-110h]
  __int64 v119; // [rsp+18h] [rbp-108h]
  __int64 v120; // [rsp+20h] [rbp-100h]
  __int64 v121; // [rsp+28h] [rbp-F8h]
  __int64 v122; // [rsp+30h] [rbp-F0h]
  __int64 v123; // [rsp+38h] [rbp-E8h]
  __m128i v124; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v125; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v126; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v127; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v128; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v129; // [rsp+90h] [rbp-90h] BYREF
  _QWORD *v130; // [rsp+98h] [rbp-88h]
  bool v131; // [rsp+A0h] [rbp-80h]
  __int64 v132; // [rsp+A8h] [rbp-78h]
  __int64 v133; // [rsp+B0h] [rbp-70h]
  __int64 v134; // [rsp+B8h] [rbp-68h]
  __int64 v135; // [rsp+C0h] [rbp-60h]
  __int64 v136; // [rsp+C8h] [rbp-58h]
  __int64 v137; // [rsp+D0h] [rbp-50h]
  __m128i *v138; // [rsp+D8h] [rbp-48h]
  unsigned __int8 v139; // [rsp+E0h] [rbp-40h]
  unsigned int v140; // [rsp+E8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_180:
    BUG();
  v6 = a1;
  while ( *(_UNKNOWN **)v3 != &unk_4F875EC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_180;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F875EC);
  v8 = *(__int64 **)(a1 + 8);
  v117 = v7 + 176;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_175:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F881C8 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_175;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F881C8);
  v12 = *(__int64 **)(a1 + 8);
  v122 = *(_QWORD *)(v11 + 176);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_176:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F8144C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_176;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F8144C);
  v16 = *(__int64 **)(a1 + 8);
  v121 = v15 + 176;
  v17 = *v16;
  v18 = v16[1];
  if ( v17 == v18 )
LABEL_177:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F89C28 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_177;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F89C28);
  v120 = sub_DFED00(v19, a2);
  v20 = sub_B2BEC0(a2);
  v21 = *(__int64 **)(a1 + 8);
  v119 = v20;
  v22 = *v21;
  v23 = v21[1];
  if ( v22 == v23 )
LABEL_178:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F8FAE4 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_178;
  }
  v118 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(
                       *(_QWORD *)(v22 + 8),
                       &unk_4F8FAE4)
                   + 176);
  v24 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F6D3F0);
  if ( v24 && (v25 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v24 + 104LL))(v24, &unk_4F6D3F0)) != 0 )
  {
    sub_BBB200((__int64)&v129);
    sub_983BD0((__int64)&v124, v25 + 176, a2);
    v123 = v25 + 408;
    if ( *(_BYTE *)(v25 + 488) )
    {
      *(__m128i *)(v25 + 408) = _mm_loadu_si128(&v124);
      *(__m128i *)(v25 + 424) = _mm_loadu_si128(&v125);
      *(__m128i *)(v25 + 440) = _mm_loadu_si128(&v126);
      *(__m128i *)(v25 + 456) = _mm_loadu_si128(&v127);
      *(__m128i *)(v25 + 472) = _mm_loadu_si128(&v128);
    }
    else
    {
      *(__m128i *)(v25 + 408) = _mm_loadu_si128(&v124);
      *(__m128i *)(v25 + 424) = _mm_loadu_si128(&v125);
      *(__m128i *)(v25 + 440) = _mm_loadu_si128(&v126);
      *(__m128i *)(v25 + 456) = _mm_loadu_si128(&v127);
      v115 = _mm_loadu_si128(&v128);
      *(_BYTE *)(v25 + 488) = 1;
      *(__m128i *)(v25 + 472) = v115;
    }
    sub_C7D6A0((__int64)v138, 24LL * v140, 8);
    v26 = v136;
    if ( (_DWORD)v136 )
    {
      v27 = (_QWORD **)(v134 + 32LL * (unsigned int)v136);
      for ( i = (_QWORD **)(v134 + 8); ; i += 4 )
      {
        v29 = (__int64)*(i - 1);
        if ( v29 != -4096 && v29 != -8192 )
        {
          v30 = *i;
          while ( v30 != i )
          {
            v31 = (unsigned __int64)v30;
            v30 = (_QWORD *)*v30;
            v32 = *(_QWORD *)(v31 + 24);
            if ( v32 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
            j_j___libc_free_0(v31);
          }
        }
        if ( v27 == i + 3 )
          break;
      }
      v6 = a1;
      v26 = v136;
    }
    sub_C7D6A0(v134, 32LL * v26, 8);
    v33 = v132;
    if ( (_DWORD)v132 )
    {
      v34 = v130;
      v35 = &v130[2 * (unsigned int)v132];
      do
      {
        if ( *v34 != -8192 && *v34 != -4096 )
        {
          v36 = v34[1];
          if ( v36 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
        }
        v34 += 2;
      }
      while ( v35 != v34 );
      v33 = v132;
    }
    sub_C7D6A0((__int64)v130, 16LL * v33, 8);
  }
  else
  {
    v123 = 0;
  }
  v37 = *(__int64 **)(v6 + 8);
  v38 = *v37;
  v39 = v37[1];
  if ( v38 == v39 )
LABEL_179:
    BUG();
  while ( *(_UNKNOWN **)v38 != &unk_4F8662C )
  {
    v38 += 16;
    if ( v39 == v38 )
      goto LABEL_179;
  }
  v40 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(*(_QWORD *)(v38 + 8), &unk_4F8662C);
  v41 = sub_CFFAC0(v40, a2);
  v42 = sub_BB9560(v6, (__int64)&unk_4F90E2C);
  v125.m128i_i64[0] = 0;
  v124 = 0;
  v43 = v42;
  v44 = sub_C52410();
  v45 = v44 + 1;
  v46 = sub_C959E0();
  v47 = (_QWORD *)v44[2];
  if ( v47 )
  {
    v48 = v44 + 1;
    do
    {
      while ( 1 )
      {
        v49 = v47[2];
        v50 = v47[3];
        if ( v46 <= v47[4] )
          break;
        v47 = (_QWORD *)v47[3];
        if ( !v50 )
          goto LABEL_54;
      }
      v48 = v47;
      v47 = (_QWORD *)v47[2];
    }
    while ( v49 );
LABEL_54:
    if ( v45 != v48 && v46 >= v48[4] )
      v45 = v48;
  }
  if ( v45 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v51 = v45[7];
    if ( v51 )
    {
      v52 = v45 + 6;
      do
      {
        while ( 1 )
        {
          v53 = *(_QWORD *)(v51 + 16);
          v54 = *(_QWORD *)(v51 + 24);
          if ( *(_DWORD *)(v51 + 32) >= dword_501E688 )
            break;
          v51 = *(_QWORD *)(v51 + 24);
          if ( !v54 )
            goto LABEL_63;
        }
        v52 = (_QWORD *)v51;
        v51 = *(_QWORD *)(v51 + 16);
      }
      while ( v53 );
LABEL_63:
      if ( v45 + 6 != v52 && dword_501E688 >= *((_DWORD *)v52 + 8) && *((_DWORD *)v52 + 9) )
      {
        v125.m128i_i8[1] = 1;
        v125.m128i_i8[0] = qword_501E708;
      }
    }
  }
  v55 = sub_C52410();
  v56 = v55 + 1;
  v57 = sub_C959E0();
  v58 = (_QWORD *)v55[2];
  if ( v58 )
  {
    v59 = v55 + 1;
    do
    {
      while ( 1 )
      {
        v60 = v58[2];
        v61 = v58[3];
        if ( v57 <= v58[4] )
          break;
        v58 = (_QWORD *)v58[3];
        if ( !v61 )
          goto LABEL_72;
      }
      v59 = v58;
      v58 = (_QWORD *)v58[2];
    }
    while ( v60 );
LABEL_72:
    if ( v56 != v59 && v57 >= v59[4] )
      v56 = v59;
  }
  if ( v56 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v62 = v56[7];
    if ( v62 )
    {
      v63 = v56 + 6;
      do
      {
        while ( 1 )
        {
          v64 = *(_QWORD *)(v62 + 16);
          v65 = *(_QWORD *)(v62 + 24);
          if ( *(_DWORD *)(v62 + 32) >= dword_501E5A8 )
            break;
          v62 = *(_QWORD *)(v62 + 24);
          if ( !v65 )
            goto LABEL_81;
        }
        v63 = (_QWORD *)v62;
        v62 = *(_QWORD *)(v62 + 16);
      }
      while ( v64 );
LABEL_81:
      if ( v56 + 6 != v63 && dword_501E5A8 >= *((_DWORD *)v63 + 8) && *((_DWORD *)v63 + 9) )
      {
        v125.m128i_i8[3] = 1;
        v125.m128i_i8[2] = qword_501E628;
      }
    }
  }
  v66 = sub_C52410();
  v67 = v66 + 1;
  v68 = sub_C959E0();
  v69 = (_QWORD *)v66[2];
  if ( v69 )
  {
    v70 = v66 + 1;
    do
    {
      while ( 1 )
      {
        v71 = v69[2];
        v72 = v69[3];
        if ( v68 <= v69[4] )
          break;
        v69 = (_QWORD *)v69[3];
        if ( !v72 )
          goto LABEL_90;
      }
      v70 = v69;
      v69 = (_QWORD *)v69[2];
    }
    while ( v71 );
LABEL_90:
    if ( v67 != v70 && v68 >= v70[4] )
      v67 = v70;
  }
  if ( v67 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v73 = v67[7];
    if ( v73 )
    {
      v74 = v67 + 6;
      do
      {
        while ( 1 )
        {
          v75 = *(_QWORD *)(v73 + 16);
          v76 = *(_QWORD *)(v73 + 24);
          if ( *(_DWORD *)(v73 + 32) >= dword_501E4C8 )
            break;
          v73 = *(_QWORD *)(v73 + 24);
          if ( !v76 )
            goto LABEL_99;
        }
        v74 = (_QWORD *)v73;
        v73 = *(_QWORD *)(v73 + 16);
      }
      while ( v75 );
LABEL_99:
      if ( v67 + 6 != v74 && dword_501E4C8 >= *((_DWORD *)v74 + 8) && *((_DWORD *)v74 + 9) )
      {
        v125.m128i_i8[5] = 1;
        v125.m128i_i8[4] = qword_501E548;
      }
    }
  }
  v77 = sub_C52410();
  v78 = v77 + 1;
  v79 = sub_C959E0();
  v80 = (_QWORD *)v77[2];
  if ( v80 )
  {
    v81 = v77 + 1;
    do
    {
      while ( 1 )
      {
        v82 = v80[2];
        v83 = v80[3];
        if ( v79 <= v80[4] )
          break;
        v80 = (_QWORD *)v80[3];
        if ( !v83 )
          goto LABEL_108;
      }
      v81 = v80;
      v80 = (_QWORD *)v80[2];
    }
    while ( v82 );
LABEL_108:
    if ( v78 != v81 && v79 >= v81[4] )
      v78 = v81;
  }
  if ( v78 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v84 = v78[7];
    if ( v84 )
    {
      v85 = v78 + 6;
      do
      {
        while ( 1 )
        {
          v86 = *(_QWORD *)(v84 + 16);
          v87 = *(_QWORD *)(v84 + 24);
          if ( *(_DWORD *)(v84 + 32) >= dword_501E228 )
            break;
          v84 = *(_QWORD *)(v84 + 24);
          if ( !v87 )
            goto LABEL_117;
        }
        v85 = (_QWORD *)v84;
        v84 = *(_QWORD *)(v84 + 16);
      }
      while ( v86 );
LABEL_117:
      if ( v78 + 6 != v85 && dword_501E228 >= *((_DWORD *)v85 + 8) && *((_DWORD *)v85 + 9) )
      {
        v125.m128i_i8[7] = 1;
        v125.m128i_i8[6] = qword_501E2A8;
      }
    }
  }
  v88 = sub_C52410();
  v89 = v88 + 1;
  v90 = sub_C959E0();
  v91 = (_QWORD *)v88[2];
  if ( v91 )
  {
    v92 = v88 + 1;
    do
    {
      while ( 1 )
      {
        v93 = v91[2];
        v94 = v91[3];
        if ( v90 <= v91[4] )
          break;
        v91 = (_QWORD *)v91[3];
        if ( !v94 )
          goto LABEL_126;
      }
      v92 = v91;
      v91 = (_QWORD *)v91[2];
    }
    while ( v93 );
LABEL_126:
    if ( v89 != v92 && v90 >= v92[4] )
      v89 = v92;
  }
  if ( v89 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v95 = v89[7];
    if ( v95 )
    {
      v96 = v89 + 6;
      do
      {
        while ( 1 )
        {
          v97 = *(_QWORD *)(v95 + 16);
          v98 = *(_QWORD *)(v95 + 24);
          if ( *(_DWORD *)(v95 + 32) >= dword_501E3E8 )
            break;
          v95 = *(_QWORD *)(v95 + 24);
          if ( !v98 )
            goto LABEL_135;
        }
        v96 = (_QWORD *)v95;
        v95 = *(_QWORD *)(v95 + 16);
      }
      while ( v97 );
LABEL_135:
      if ( v89 + 6 != v96 && dword_501E3E8 >= *((_DWORD *)v96 + 8) && *((_DWORD *)v96 + 9) )
      {
        v124.m128i_i8[4] = 1;
        v124.m128i_i32[0] = qword_501E468;
      }
    }
  }
  v99 = sub_C52410();
  v100 = v99 + 1;
  v101 = sub_C959E0();
  v102 = (_QWORD *)v99[2];
  if ( v102 )
  {
    v103 = v99 + 1;
    do
    {
      while ( 1 )
      {
        v104 = v102[2];
        v105 = v102[3];
        if ( v101 <= v102[4] )
          break;
        v102 = (_QWORD *)v102[3];
        if ( !v105 )
          goto LABEL_144;
      }
      v103 = v102;
      v102 = (_QWORD *)v102[2];
    }
    while ( v104 );
LABEL_144:
    if ( v100 != v103 && v101 >= v103[4] )
      v100 = v103;
  }
  if ( v100 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v106 = v100[7];
    if ( v106 )
    {
      v107 = v100 + 6;
      do
      {
        while ( 1 )
        {
          v108 = *(_QWORD *)(v106 + 16);
          v109 = *(_QWORD *)(v106 + 24);
          if ( *(_DWORD *)(v106 + 32) >= dword_501E308 )
            break;
          v106 = *(_QWORD *)(v106 + 24);
          if ( !v109 )
            goto LABEL_153;
        }
        v107 = (_QWORD *)v106;
        v106 = *(_QWORD *)(v106 + 16);
      }
      while ( v108 );
LABEL_153:
      if ( v100 + 6 != v107 && dword_501E308 >= *((_DWORD *)v107 + 8) && *((_DWORD *)v107 + 9) )
      {
        v124.m128i_i8[12] = 1;
        v124.m128i_i32[2] = qword_501E388;
      }
    }
  }
  v136 = v41;
  v131 = v43;
  v129 = v122;
  v139 = 0;
  v130 = (_QWORD *)v117;
  v132 = v121;
  v133 = v119;
  v134 = v120;
  v135 = v123;
  v137 = v118;
  v138 = &v124;
  v110 = (_QWORD *)sub_B2BE50(a2);
  v111 = (_QWORD **)v130[5];
  for ( j = (_QWORD **)v130[4]; v111 != j; ++j )
  {
    while ( 1 )
    {
      v113 = *j;
      if ( !**j )
        break;
      if ( v111 == ++j )
        return v139;
    }
    sub_2DE22E0((__int64)&v129, (__int64)v113, v110);
  }
  return v139;
}
