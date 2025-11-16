// Function: sub_2F89F20
// Address: 0x2f89f20
//
__int64 __fastcall sub_2F89F20(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned int v16; // eax
  _QWORD **v17; // rbx
  __int64 v18; // rax
  _QWORD *v19; // r15
  unsigned __int64 v20; // r14
  __int64 v21; // rdi
  unsigned int v22; // eax
  _QWORD *v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // rdi
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 *v33; // r8
  char v34; // bl
  unsigned __int64 *v35; // rsi
  __m128i *v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rsi
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  unsigned __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  _QWORD *v52; // r12
  _QWORD *v53; // rbx
  void (__fastcall *v54)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v55; // rax
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // r12
  __int64 v58; // r14
  __int64 *v59; // r15
  __int64 *v60; // rbx
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rax
  unsigned int v64; // eax
  __int64 v65; // rdx
  char v66; // al
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // rdi
  __int64 *v69; // rbx
  __int64 *v70; // r12
  __int64 v71; // rsi
  __int64 v72; // rdi
  __int64 *v73; // rax
  __int64 *v74; // rbx
  __int64 *v75; // r13
  __int64 v76; // rdi
  unsigned int v77; // ecx
  __int64 v78; // rsi
  __int64 *v79; // rbx
  __int64 *v80; // r12
  __int64 v81; // rsi
  __int64 v82; // rdi
  __m128i v84; // xmm1
  unsigned __int64 v85; // rbx
  unsigned __int64 *v86; // r12
  unsigned __int64 v87; // r13
  unsigned __int64 v88; // rdi
  int v89; // eax
  unsigned __int64 v90; // r13
  unsigned __int64 *v91; // r8
  unsigned __int64 v92; // r9
  unsigned __int64 v93; // rdi
  __int64 v94; // [rsp+8h] [rbp-AD8h]
  __int64 v95; // [rsp+20h] [rbp-AC0h]
  unsigned __int64 v96; // [rsp+28h] [rbp-AB8h]
  __int64 v97; // [rsp+30h] [rbp-AB0h]
  _QWORD **v98; // [rsp+38h] [rbp-AA8h]
  _QWORD *v99; // [rsp+38h] [rbp-AA8h]
  __int64 v100; // [rsp+38h] [rbp-AA8h]
  unsigned __int8 v101; // [rsp+38h] [rbp-AA8h]
  unsigned __int64 *v102; // [rsp+38h] [rbp-AA8h]
  unsigned __int64 *v103; // [rsp+38h] [rbp-AA8h]
  __int64 v104[10]; // [rsp+40h] [rbp-AA0h] BYREF
  unsigned __int64 v105[18]; // [rsp+90h] [rbp-A50h] BYREF
  char v106[8]; // [rsp+120h] [rbp-9C0h] BYREF
  __int64 v107; // [rsp+128h] [rbp-9B8h]
  unsigned int v108; // [rsp+138h] [rbp-9A8h]
  unsigned __int64 v109; // [rsp+140h] [rbp-9A0h]
  unsigned __int64 v110; // [rsp+148h] [rbp-998h]
  __int64 v111; // [rsp+158h] [rbp-988h]
  __int64 i; // [rsp+160h] [rbp-980h]
  __int64 *v113; // [rsp+168h] [rbp-978h]
  unsigned int v114; // [rsp+170h] [rbp-970h]
  char v115; // [rsp+178h] [rbp-968h] BYREF
  __int64 *v116; // [rsp+198h] [rbp-948h]
  unsigned int v117; // [rsp+1A0h] [rbp-940h]
  __int64 v118; // [rsp+1A8h] [rbp-938h] BYREF
  __m128i v119; // [rsp+1C0h] [rbp-920h] BYREF
  __m128i v120; // [rsp+1D0h] [rbp-910h] BYREF
  __m128i v121; // [rsp+1E0h] [rbp-900h] BYREF
  __m128i v122; // [rsp+1F0h] [rbp-8F0h] BYREF
  __m128i v123[29]; // [rsp+200h] [rbp-8E0h] BYREF
  __int64 v124; // [rsp+3D0h] [rbp-710h]
  __int64 v125; // [rsp+3D8h] [rbp-708h]
  __int64 v126; // [rsp+3E0h] [rbp-700h]
  __int64 v127; // [rsp+3E8h] [rbp-6F8h]
  char v128; // [rsp+3F0h] [rbp-6F0h]
  __int64 v129; // [rsp+3F8h] [rbp-6E8h]
  char *v130; // [rsp+400h] [rbp-6E0h]
  __int64 v131; // [rsp+408h] [rbp-6D8h]
  int v132; // [rsp+410h] [rbp-6D0h]
  char v133; // [rsp+414h] [rbp-6CCh]
  char v134; // [rsp+418h] [rbp-6C8h] BYREF
  __int16 v135; // [rsp+458h] [rbp-688h]
  _QWORD *v136; // [rsp+460h] [rbp-680h]
  _QWORD *v137; // [rsp+468h] [rbp-678h]
  __int64 v138; // [rsp+470h] [rbp-670h]
  char v139[8]; // [rsp+480h] [rbp-660h] BYREF
  _QWORD *v140; // [rsp+488h] [rbp-658h]
  unsigned int v141; // [rsp+498h] [rbp-648h]
  __int64 v142; // [rsp+4A8h] [rbp-638h]
  unsigned int v143; // [rsp+4B8h] [rbp-628h]
  __int64 v144; // [rsp+4C8h] [rbp-618h]
  unsigned int v145; // [rsp+4D8h] [rbp-608h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
    goto LABEL_124;
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_124;
  }
  v7 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                     *(_QWORD *)(v3 + 8),
                     &unk_5027190)
                 + 256);
  *(_QWORD *)(a1 + 176) = v7;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 16LL);
  if ( v8 == sub_23CE270 )
    goto LABEL_124;
  v9 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v7, a2);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
  if ( v10 == sub_2C8F680 || (v94 = ((__int64 (__fastcall *)(__int64))v10)(v9)) == 0 )
    sub_C64ED0("TargetLowering instance is required", 1u);
  v11 = sub_B2BEC0(a2);
  v12 = *(__int64 **)(a1 + 8);
  v97 = v11;
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
    goto LABEL_124;
  while ( *(_UNKNOWN **)v13 != &unk_4F6D3F0 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_124;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v139);
  sub_983BD0((__int64)&v119, v15 + 176, a2);
  v95 = v15 + 408;
  if ( *(_BYTE *)(v15 + 488) )
  {
    *(__m128i *)(v15 + 408) = _mm_loadu_si128(&v119);
    *(__m128i *)(v15 + 424) = _mm_loadu_si128(&v120);
    *(__m128i *)(v15 + 440) = _mm_loadu_si128(&v121);
    *(__m128i *)(v15 + 456) = _mm_loadu_si128(&v122);
    *(__m128i *)(v15 + 472) = _mm_loadu_si128(v123);
  }
  else
  {
    *(__m128i *)(v15 + 408) = _mm_loadu_si128(&v119);
    *(__m128i *)(v15 + 424) = _mm_loadu_si128(&v120);
    *(__m128i *)(v15 + 440) = _mm_loadu_si128(&v121);
    *(__m128i *)(v15 + 456) = _mm_loadu_si128(&v122);
    v84 = _mm_loadu_si128(v123);
    *(_BYTE *)(v15 + 488) = 1;
    *(__m128i *)(v15 + 472) = v84;
  }
  sub_C7D6A0(v144, 24LL * v145, 8);
  v16 = v143;
  if ( v143 )
  {
    v17 = (_QWORD **)(v142 + 8);
    v98 = (_QWORD **)(v142 + 32LL * v143);
    while ( 1 )
    {
      v18 = (__int64)*(v17 - 1);
      if ( v18 != -4096 && v18 != -8192 )
      {
        v19 = *v17;
        while ( v19 != v17 )
        {
          v20 = (unsigned __int64)v19;
          v19 = (_QWORD *)*v19;
          v21 = *(_QWORD *)(v20 + 24);
          if ( v21 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
          j_j___libc_free_0(v20);
        }
      }
      if ( v98 == v17 + 3 )
        break;
      v17 += 4;
    }
    v16 = v143;
  }
  sub_C7D6A0(v142, 32LL * v16, 8);
  v22 = v141;
  if ( v141 )
  {
    v23 = v140;
    v24 = &v140[2 * v141];
    do
    {
      if ( *v23 != -4096 && *v23 != -8192 )
      {
        v25 = v23[1];
        if ( v25 )
        {
          v99 = v24;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
          v24 = v99;
        }
      }
      v23 += 2;
    }
    while ( v24 != v23 );
    v22 = v141;
  }
  sub_C7D6A0((__int64)v140, 16LL * v22, 8);
  v26 = *(__int64 **)(a1 + 8);
  v27 = *v26;
  v28 = v26[1];
  if ( v27 == v28 )
LABEL_124:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F8662C )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_124;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F8662C);
  v30 = sub_CFFAC0(v29, a2);
  memset(v105, 0, 0x88u);
  v31 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v31 && (v32 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v31 + 104LL))(v31, &unk_4F8144C)) != 0 )
  {
    v33 = (unsigned __int64 *)(v32 + 176);
    v34 = 1;
    v35 = (unsigned __int64 *)(v32 + 176);
  }
  else
  {
    if ( LOBYTE(v105[16]) )
    {
      v90 = v105[3];
      LOBYTE(v105[16]) = 0;
      v91 = (unsigned __int64 *)(v105[3] + 8LL * LODWORD(v105[4]));
      if ( (unsigned __int64 *)v105[3] != v91 )
      {
        do
        {
          v92 = *--v91;
          if ( v92 )
          {
            v93 = *(_QWORD *)(v92 + 24);
            if ( v93 != v92 + 40 )
            {
              v96 = v92;
              v102 = v91;
              _libc_free(v93);
              v92 = v96;
              v91 = v102;
            }
            v103 = v91;
            j_j___libc_free_0(v92);
            v91 = v103;
          }
        }
        while ( (unsigned __int64 *)v90 != v91 );
        v91 = (unsigned __int64 *)v105[3];
      }
      if ( v91 != &v105[5] )
        _libc_free((unsigned __int64)v91);
      if ( (unsigned __int64 *)v105[0] != &v105[2] )
        _libc_free(v105[0]);
    }
    v105[0] = (unsigned __int64)&v105[2];
    v105[1] = 0x100000000LL;
    v105[4] = 0x600000000LL;
    v89 = *(_DWORD *)(a2 + 92);
    v105[3] = (unsigned __int64)&v105[5];
    v105[12] = 0;
    LOBYTE(v105[14]) = 0;
    HIDWORD(v105[14]) = 0;
    v105[13] = a2;
    LODWORD(v105[15]) = v89;
    sub_B1F440((__int64)v105);
    v35 = v105;
    LOBYTE(v105[16]) = 1;
    v34 = 0;
    v33 = v105;
  }
  v100 = (__int64)v33;
  sub_D51D90((__int64)v106, (__int64)v35);
  v119.m128i_i64[0] = (__int64)&v120;
  v130 = &v134;
  v119.m128i_i64[1] = 0x1000000000LL;
  v126 = v100;
  v135 = 0;
  v124 = 0;
  v125 = 0;
  v127 = 0;
  v128 = 1;
  v129 = 0;
  v131 = 8;
  v132 = 0;
  v133 = 1;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  sub_D98CB0((__int64)v139, a2, v95, v30, v100, (__int64)v106);
  v36 = 0;
  v104[0] = a2;
  v104[4] = (__int64)v139;
  if ( v34 )
    v36 = &v119;
  v104[1] = v94;
  v104[2] = v97;
  v104[3] = (__int64)v36;
  v37 = (__int64 *)sub_B2BE50(a2);
  v104[5] = sub_BCE3C0(v37, *(_DWORD *)(v97 + 4));
  v38 = sub_B2BE50(a2);
  v104[6] = sub_AE4420(v97, v38, 0);
  v39 = (_QWORD *)sub_B2BE50(a2);
  v104[8] = 0;
  v104[7] = sub_BCB2D0(v39);
  v101 = sub_2F86AD0(v104, v38, v40, v41, v42, v43);
  sub_DA11D0((__int64)v139, v38);
  sub_FFCE90((__int64)&v119, v38, v44, v45, v46, v47);
  sub_FFD870((__int64)&v119, v38, v48, v49, v50, v51);
  sub_FFBC40((__int64)&v119, v38);
  v52 = v137;
  v53 = v136;
  if ( v137 != v136 )
  {
    do
    {
      v54 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v53[7];
      *v53 = &unk_49E5048;
      if ( v54 )
      {
        v38 = (__int64)(v53 + 5);
        v54(v53 + 5, v53 + 5, 3);
      }
      *v53 = &unk_49DB368;
      v55 = v53[3];
      if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
        sub_BD60C0(v53 + 1);
      v53 += 9;
    }
    while ( v52 != v53 );
    v53 = v136;
  }
  if ( v53 )
  {
    v38 = v138 - (_QWORD)v53;
    j_j___libc_free_0((unsigned __int64)v53);
  }
  if ( !v133 )
    _libc_free((unsigned __int64)v130);
  if ( (__m128i *)v119.m128i_i64[0] != &v120 )
    _libc_free(v119.m128i_u64[0]);
  sub_D786F0((__int64)v106);
  v56 = v110;
  v57 = v109;
  if ( v109 != v110 )
  {
    do
    {
      v58 = *(_QWORD *)v57;
      v59 = *(__int64 **)(*(_QWORD *)v57 + 8LL);
      v60 = *(__int64 **)(*(_QWORD *)v57 + 16LL);
      if ( v59 == v60 )
      {
        *(_BYTE *)(v58 + 152) = 1;
      }
      else
      {
        do
        {
          v61 = *v59++;
          sub_D47BB0(v61, v38);
        }
        while ( v60 != v59 );
        *(_BYTE *)(v58 + 152) = 1;
        v62 = *(_QWORD *)(v58 + 8);
        if ( *(_QWORD *)(v58 + 16) != v62 )
          *(_QWORD *)(v58 + 16) = v62;
      }
      v63 = *(_QWORD *)(v58 + 32);
      if ( v63 != *(_QWORD *)(v58 + 40) )
        *(_QWORD *)(v58 + 40) = v63;
      ++*(_QWORD *)(v58 + 56);
      if ( *(_BYTE *)(v58 + 84) )
      {
        *(_QWORD *)v58 = 0;
      }
      else
      {
        v64 = 4 * (*(_DWORD *)(v58 + 76) - *(_DWORD *)(v58 + 80));
        v65 = *(unsigned int *)(v58 + 72);
        if ( v64 < 0x20 )
          v64 = 32;
        if ( (unsigned int)v65 > v64 )
        {
          sub_C8C990(v58 + 56, v38);
        }
        else
        {
          v38 = 0xFFFFFFFFLL;
          memset(*(void **)(v58 + 64), -1, 8 * v65);
        }
        v66 = *(_BYTE *)(v58 + 84);
        *(_QWORD *)v58 = 0;
        if ( !v66 )
          _libc_free(*(_QWORD *)(v58 + 64));
      }
      v67 = *(_QWORD *)(v58 + 32);
      if ( v67 )
      {
        v38 = *(_QWORD *)(v58 + 48) - v67;
        j_j___libc_free_0(v67);
      }
      v68 = *(_QWORD *)(v58 + 8);
      if ( v68 )
      {
        v38 = *(_QWORD *)(v58 + 24) - v68;
        j_j___libc_free_0(v68);
      }
      v57 += 8LL;
    }
    while ( v56 != v57 );
    if ( v109 != v110 )
      v110 = v109;
  }
  v69 = v116;
  v70 = &v116[2 * v117];
  if ( v116 != v70 )
  {
    do
    {
      v71 = v69[1];
      v72 = *v69;
      v69 += 2;
      sub_C7D6A0(v72, v71, 16);
    }
    while ( v70 != v69 );
  }
  v117 = 0;
  if ( !v114 )
    goto LABEL_87;
  v73 = v113;
  v118 = 0;
  v74 = &v113[v114];
  v75 = v113 + 1;
  v111 = *v113;
  for ( i = v111 + 4096; v74 != v75; v73 = v113 )
  {
    v76 = *v75;
    v77 = (unsigned int)(v75 - v73) >> 7;
    v78 = 4096LL << v77;
    if ( v77 >= 0x1E )
      v78 = 0x40000000000LL;
    ++v75;
    sub_C7D6A0(v76, v78, 16);
  }
  v114 = 1;
  sub_C7D6A0(*v73, 4096, 16);
  v79 = v116;
  v80 = &v116[2 * v117];
  if ( v116 != v80 )
  {
    do
    {
      v81 = v79[1];
      v82 = *v79;
      v79 += 2;
      sub_C7D6A0(v82, v81, 16);
    }
    while ( v80 != v79 );
LABEL_87:
    v80 = v116;
  }
  if ( v80 != &v118 )
    _libc_free((unsigned __int64)v80);
  if ( v113 != (__int64 *)&v115 )
    _libc_free((unsigned __int64)v113);
  if ( v109 )
    j_j___libc_free_0(v109);
  sub_C7D6A0(v107, 16LL * v108, 8);
  if ( LOBYTE(v105[16]) )
  {
    v85 = v105[3];
    LOBYTE(v105[16]) = 0;
    v86 = (unsigned __int64 *)(v105[3] + 8LL * LODWORD(v105[4]));
    if ( (unsigned __int64 *)v105[3] != v86 )
    {
      do
      {
        v87 = *--v86;
        if ( v87 )
        {
          v88 = *(_QWORD *)(v87 + 24);
          if ( v88 != v87 + 40 )
            _libc_free(v88);
          j_j___libc_free_0(v87);
        }
      }
      while ( (unsigned __int64 *)v85 != v86 );
      v86 = (unsigned __int64 *)v105[3];
    }
    if ( v86 != &v105[5] )
      _libc_free((unsigned __int64)v86);
    if ( (unsigned __int64 *)v105[0] != &v105[2] )
      _libc_free(v105[0]);
  }
  return v101;
}
