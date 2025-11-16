// Function: sub_2684FD0
// Address: 0x2684fd0
//
_BOOL8 __fastcall sub_2684FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  signed __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  const char **v11; // rax
  const char **v12; // rdi
  __int64 v13; // rdx
  int v14; // r14d
  char v15; // r12
  __int64 v16; // rax
  unsigned __int8 *v17; // rbx
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 *v24; // rax
  __int64 v25; // r14
  __int64 v26; // rax
  __int8 *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __m128i v32; // xmm0
  __m128i v33; // xmm1
  __m128i v34; // xmm2
  __int64 v35; // rcx
  __m128i v36; // xmm3
  __m128i v37; // xmm4
  __m128i v38; // xmm5
  __int64 v40; // rsi
  _QWORD **v41; // r14
  __int64 **v42; // r13
  const char *v43; // rax
  __int64 v44; // rdx
  _QWORD *v45; // rax
  unsigned __int64 v46; // rbx
  _BYTE *v47; // r14
  _BYTE *v48; // r12
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rax
  bool v57; // r13
  unsigned __int64 v58; // rdx
  __int64 *v59; // rax
  __int64 v60; // r14
  __int64 *v61; // r13
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __m128i v67; // xmm6
  __m128i v68; // xmm7
  __m128i v69; // xmm6
  __int64 v70; // rcx
  __int64 v71; // r9
  __m128i v72; // xmm7
  __m128i v73; // xmm7
  __m128i v74; // xmm7
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  bool v79; // [rsp+Fh] [rbp-5B1h]
  __int64 v80; // [rsp+10h] [rbp-5B0h]
  __int64 v82; // [rsp+20h] [rbp-5A0h]
  __int64 v83; // [rsp+28h] [rbp-598h]
  __int64 v84; // [rsp+28h] [rbp-598h]
  __int64 v85; // [rsp+38h] [rbp-588h]
  bool v86; // [rsp+48h] [rbp-578h]
  __int64 v87; // [rsp+48h] [rbp-578h]
  int v88; // [rsp+50h] [rbp-570h]
  __int64 *v89; // [rsp+50h] [rbp-570h]
  __int64 v90; // [rsp+50h] [rbp-570h]
  __int64 *v91; // [rsp+58h] [rbp-568h]
  _BYTE *v92; // [rsp+60h] [rbp-560h] BYREF
  __int64 v93; // [rsp+68h] [rbp-558h]
  _BYTE v94[16]; // [rsp+70h] [rbp-550h] BYREF
  void *v95; // [rsp+80h] [rbp-540h] BYREF
  int v96; // [rsp+88h] [rbp-538h]
  char v97; // [rsp+8Ch] [rbp-534h]
  const char *v98; // [rsp+90h] [rbp-530h]
  __m128i v99; // [rsp+98h] [rbp-528h]
  __int64 v100; // [rsp+A8h] [rbp-518h]
  __m128i v101; // [rsp+B0h] [rbp-510h]
  __m128i v102; // [rsp+C0h] [rbp-500h]
  _BYTE *v103; // [rsp+D0h] [rbp-4F0h] BYREF
  __int64 v104; // [rsp+D8h] [rbp-4E8h]
  _BYTE v105[320]; // [rsp+E0h] [rbp-4E0h] BYREF
  char v106; // [rsp+220h] [rbp-3A0h]
  int v107; // [rsp+224h] [rbp-39Ch]
  __int64 v108; // [rsp+228h] [rbp-398h]
  void *v109; // [rsp+230h] [rbp-390h] BYREF
  int v110; // [rsp+238h] [rbp-388h]
  char v111; // [rsp+23Ch] [rbp-384h]
  const char *v112; // [rsp+240h] [rbp-380h]
  __m128i v113; // [rsp+248h] [rbp-378h] BYREF
  __int64 v114; // [rsp+258h] [rbp-368h]
  __m128i v115; // [rsp+260h] [rbp-360h] BYREF
  __m128i v116; // [rsp+270h] [rbp-350h] BYREF
  char v117[8]; // [rsp+280h] [rbp-340h] BYREF
  int v118; // [rsp+288h] [rbp-338h]
  char v119; // [rsp+3D0h] [rbp-1F0h]
  int v120; // [rsp+3D4h] [rbp-1ECh]
  __int64 v121; // [rsp+3D8h] [rbp-1E8h]
  const char **v122; // [rsp+3E0h] [rbp-1E0h] BYREF
  __int64 v123; // [rsp+3E8h] [rbp-1D8h]
  const char *v124; // [rsp+3F0h] [rbp-1D0h] BYREF
  __m128i v125; // [rsp+3F8h] [rbp-1C8h] BYREF
  __int64 v126; // [rsp+408h] [rbp-1B8h]
  __m128i v127; // [rsp+410h] [rbp-1B0h] BYREF
  __m128i v128; // [rsp+420h] [rbp-1A0h] BYREF
  _BYTE *v129; // [rsp+430h] [rbp-190h] BYREF
  __int64 v130; // [rsp+438h] [rbp-188h]
  _BYTE v131[320]; // [rsp+440h] [rbp-180h] BYREF
  char v132; // [rsp+580h] [rbp-40h]
  int v133; // [rsp+584h] [rbp-3Ch]
  __int64 v134; // [rsp+588h] [rbp-38h]

  v79 = 0;
  v80 = *(_QWORD *)(a1 + 72);
  if ( !*(_QWORD *)(v80 + 28912) )
    return v79;
  if ( (_BYTE)qword_4FF4C08 )
    return v79;
  v6 = *(_QWORD *)(a1 + 40);
  v82 = *(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8);
  if ( *(_QWORD *)v6 == v82 )
    return v79;
  v91 = *(__int64 **)v6;
  do
  {
    v7 = 0;
    v8 = *v91;
    v9 = *(_QWORD *)(*v91 + 16);
    v92 = v94;
    v93 = 0x200000000LL;
    v123 = 0x800000000LL;
    v10 = v9;
    v122 = &v124;
    if ( !v9 )
      goto LABEL_33;
    do
    {
      v10 = *(_QWORD *)(v10 + 8);
      ++v7;
    }
    while ( v10 );
    v11 = &v124;
    if ( v7 > 8 )
    {
      sub_C8D5F0((__int64)&v122, &v124, v7, 8u, a5, a6);
      v11 = &v122[(unsigned int)v123];
    }
    do
    {
      *v11 = (const char *)v9;
      v9 = *(_QWORD *)(v9 + 8);
      ++v11;
    }
    while ( v9 );
    v12 = v122;
    LODWORD(v123) = v123 + v7;
    if ( !(_DWORD)v123 )
    {
      if ( v122 != &v124 )
        _libc_free((unsigned __int64)v122);
      goto LABEL_33;
    }
    v85 = v8;
    v13 = 0;
    v14 = 0;
    v86 = 0;
    v15 = 0;
    v88 = 0;
    do
    {
      while ( 1 )
      {
        a6 = (__int64)v12[v13];
        v17 = *(unsigned __int8 **)(a6 + 24);
        v18 = *v17;
        if ( (_BYTE)v18 == 5 )
        {
          v54 = *((_QWORD *)v17 + 2);
          v16 = (unsigned int)v123;
          if ( v54 )
          {
            do
            {
              if ( v16 + 1 > (unsigned __int64)HIDWORD(v123) )
              {
                sub_C8D5F0((__int64)&v122, &v124, v16 + 1, 8u, a5, a6);
                v16 = (unsigned int)v123;
              }
              v122[v16] = (const char *)v54;
              v16 = (unsigned int)(v123 + 1);
              LODWORD(v123) = v123 + 1;
              v54 = *(_QWORD *)(v54 + 8);
            }
            while ( v54 );
            v12 = v122;
          }
          goto LABEL_15;
        }
        if ( (unsigned __int8)v18 > 0x1Cu )
        {
          if ( (unsigned __int8)(v18 - 34) > 0x33u
            || (v40 = 0x8000000000041LL, !_bittest64(&v40, (unsigned int)(v18 - 34))) )
          {
            if ( (_BYTE)v18 == 82 )
              goto LABEL_62;
            goto LABEL_14;
          }
          if ( (unsigned __int8 *)a6 == v17 - 32 )
            break;
        }
        if ( (_BYTE)v18 == 82 )
        {
LABEL_62:
          v16 = (unsigned int)v93;
          v53 = (unsigned int)v93 + 1LL;
          if ( v53 > HIDWORD(v93) )
          {
            v84 = a6;
            sub_C8D5F0((__int64)&v92, v94, v53, 8u, a5, a6);
            v16 = (unsigned int)v93;
            a6 = v84;
          }
          *(_QWORD *)&v92[8 * v16] = a6;
          LODWORD(v16) = v123;
          LODWORD(v93) = v93 + 1;
          v12 = v122;
          goto LABEL_15;
        }
        if ( (_BYTE)v18 == 85 )
        {
          if ( (v17[7] & 0x80u) == 0
            || ((v83 = (__int64)v12[v13],
                 v19 = sub_BD2BC0(*(_QWORD *)(a6 + 24)),
                 a6 = v83,
                 v21 = v19 + v20,
                 (v17[7] & 0x80u) == 0)
              ? (v23 = v21 >> 4)
              : (v22 = sub_BD2BC0((__int64)v17), a6 = v83, v23 = (v21 - v22) >> 4),
                !(_DWORD)v23) )
          {
            v55 = *(_QWORD *)(v80 + 28912);
            if ( v55 )
            {
              v56 = *((_QWORD *)v17 - 4);
              if ( v56 )
              {
                if ( !*(_BYTE *)v56 )
                {
                  v57 = v55 == v56 && *(_QWORD *)(v56 + 24) == *((_QWORD *)v17 + 10);
                  if ( v57
                    && !v86
                    && (unsigned int)((a6 - (__int64)&v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)]) >> 5) == 6 )
                  {
                    v16 = (unsigned int)v93;
                    v58 = (unsigned int)v93 + 1LL;
                    if ( v58 > HIDWORD(v93) )
                    {
                      v87 = a6;
                      sub_C8D5F0((__int64)&v92, v94, v58, 8u, a5, a6);
                      v16 = (unsigned int)v93;
                      a6 = v87;
                    }
                    v86 = v57;
                    *(_QWORD *)&v92[8 * v16] = a6;
                    LODWORD(v16) = v123;
                    LODWORD(v93) = v93 + 1;
                    v12 = v122;
                    goto LABEL_15;
                  }
                }
              }
            }
          }
          v12 = v122;
        }
LABEL_14:
        LODWORD(v16) = v123;
        v15 = 1;
LABEL_15:
        v13 = (unsigned int)(v14 + 1);
        v14 = v13;
        if ( (unsigned int)v13 >= (unsigned int)v16 )
          goto LABEL_40;
      }
      v13 = (unsigned int)(v14 + 1);
      ++v88;
      v14 = v13;
    }
    while ( (unsigned int)v13 < (unsigned int)v123 );
LABEL_40:
    if ( v12 != &v124 )
      _libc_free((unsigned __int64)v12);
    if ( v86 )
    {
      if ( v15 || v88 != 1 || (unsigned int)v93 > 2 )
      {
        v24 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 56))(*(_QWORD *)(a1 + 64), v85);
        v25 = *v24;
        v89 = v24;
        v26 = sub_B2BE50(*v24);
        if ( sub_B6EA50(v26)
          || (v75 = sub_B2BE50(v25),
              v76 = sub_B6F970(v75),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v76 + 48LL))(v76)) )
        {
          sub_B179F0((__int64)&v109, (__int64)"openmp-opt", (__int64)"OMP101", 6, v85);
          sub_B18290((__int64)&v109, "Parallel region is used in ", 0x1Bu);
          v27 = "unknown";
          if ( !v15 )
            v27 = "unexpected";
          sub_B18290((__int64)&v109, v27, v15 == 0 ? 10LL : 7LL);
          sub_B18290((__int64)&v109, " ways. Will not attempt to rewrite the state machine.", 0x35u);
          v32 = _mm_loadu_si128(&v113);
          v33 = _mm_loadu_si128(&v115);
          LODWORD(v123) = v110;
          v34 = _mm_loadu_si128(&v116);
          v125 = v32;
          BYTE4(v123) = v111;
          v127 = v33;
          v124 = v112;
          v122 = (const char **)&unk_49D9D40;
          v128 = v34;
          v126 = v114;
          v129 = v131;
          v130 = 0x400000000LL;
          if ( v118 )
            sub_26781A0((__int64)&v129, (__int64)v117, v28, v29, v30, v31);
          v132 = v119;
          v133 = v120;
          v134 = v121;
          v122 = (const char **)&unk_49D9DE8;
          sub_B18290((__int64)&v122, " [", 2u);
          sub_B18290((__int64)&v122, "OMP101", 6u);
          sub_B18290((__int64)&v122, "]", 1u);
          v36 = _mm_loadu_si128(&v125);
          v37 = _mm_loadu_si128(&v127);
          v104 = 0x400000000LL;
          v96 = v123;
          v38 = _mm_loadu_si128(&v128);
          v99 = v36;
          v97 = BYTE4(v123);
          v101 = v37;
          v98 = v124;
          v95 = &unk_49D9D40;
          v102 = v38;
          v100 = v126;
          v103 = v105;
          if ( (_DWORD)v130 )
            sub_26781A0((__int64)&v103, (__int64)&v129, (__int64)v105, v35, (__int64)&v103, (unsigned int)v130);
          v106 = v132;
          v95 = &unk_49D9DE8;
          v107 = v133;
          v122 = (const char **)&unk_49D9D40;
          v108 = v134;
          sub_23FD590((__int64)&v129);
          v109 = &unk_49D9D40;
          sub_23FD590((__int64)v117);
          sub_1049740(v89, (__int64)&v95);
          v95 = &unk_49D9D40;
          sub_23FD590((__int64)&v103);
        }
      }
      else
      {
        if ( sub_26847B0(a1, v85) )
        {
          v41 = *(_QWORD ***)(v85 + 40);
          v42 = (__int64 **)sub_BCB2B0(*v41);
          v90 = sub_ACA8A0(v42);
          v43 = sub_BD5D20(v85);
          BYTE4(v109) = 0;
          v122 = (const char **)v43;
          v124 = ".ID";
          v125.m128i_i16[4] = 773;
          v123 = v44;
          v45 = sub_BD2C40(88, unk_3F0FAE8);
          v46 = (unsigned __int64)v45;
          if ( v45 )
            sub_B30000((__int64)v45, (__int64)v41, v42, 1, 8, v90, (__int64)&v122, 0, 0, (__int64)v109, 0);
          v47 = v92;
          v48 = &v92[8 * (unsigned int)v93];
          if ( v48 != v92 )
          {
            do
            {
              v49 = *(_QWORD *)v47;
              v50 = sub_ADB060(v46, *(_QWORD *)(**(_QWORD **)v47 + 8LL));
              if ( *(_QWORD *)v49 )
              {
                v51 = *(_QWORD *)(v49 + 8);
                **(_QWORD **)(v49 + 16) = v51;
                if ( v51 )
                  *(_QWORD *)(v51 + 16) = *(_QWORD *)(v49 + 16);
              }
              *(_QWORD *)v49 = v50;
              if ( v50 )
              {
                v52 = *(_QWORD *)(v50 + 16);
                *(_QWORD *)(v49 + 8) = v52;
                if ( v52 )
                  *(_QWORD *)(v52 + 16) = v49 + 8;
                *(_QWORD *)(v49 + 16) = v50 + 16;
                *(_QWORD *)(v50 + 16) = v49;
              }
              v47 += 8;
            }
            while ( v48 != v47 );
            v47 = v92;
          }
          if ( v47 != v94 )
            _libc_free((unsigned __int64)v47);
          v79 = v86;
          goto LABEL_35;
        }
        v59 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 56))(*(_QWORD *)(a1 + 64), v85);
        v60 = *v59;
        v61 = v59;
        v62 = sub_B2BE50(*v59);
        if ( sub_B6EA50(v62)
          || (v77 = sub_B2BE50(v60),
              v78 = sub_B6F970(v77),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v78 + 48LL))(v78)) )
        {
          sub_B179F0((__int64)&v109, (__int64)"openmp-opt", (__int64)"OMP102", 6, v85);
          sub_B18290(
            (__int64)&v109,
            "Parallel region is not called from a unique kernel. Will not attempt to rewrite the state machine.",
            0x62u);
          v67 = _mm_loadu_si128(&v113);
          v68 = _mm_loadu_si128(&v115);
          LODWORD(v123) = v110;
          v125 = v67;
          v69 = _mm_loadu_si128(&v116);
          BYTE4(v123) = v111;
          v127 = v68;
          v124 = v112;
          v122 = (const char **)&unk_49D9D40;
          v128 = v69;
          v126 = v114;
          v129 = v131;
          v130 = 0x400000000LL;
          if ( v118 )
            sub_26781A0((__int64)&v129, (__int64)v117, v63, v64, v65, v66);
          v132 = v119;
          v133 = v120;
          v134 = v121;
          v122 = (const char **)&unk_49D9DE8;
          sub_B18290((__int64)&v122, " [", 2u);
          sub_B18290((__int64)&v122, "OMP102", 6u);
          sub_B18290((__int64)&v122, "]", 1u);
          v72 = _mm_loadu_si128(&v125);
          v104 = 0x400000000LL;
          v96 = v123;
          v99 = v72;
          v73 = _mm_loadu_si128(&v127);
          v97 = BYTE4(v123);
          v101 = v73;
          v74 = _mm_loadu_si128(&v128);
          v98 = v124;
          v95 = &unk_49D9D40;
          v102 = v74;
          v100 = v126;
          v103 = v105;
          if ( (_DWORD)v130 )
            sub_26781A0((__int64)&v103, (__int64)&v129, (__int64)v105, v70, (__int64)&v103, v71);
          v106 = v132;
          v95 = &unk_49D9DE8;
          v107 = v133;
          v122 = (const char **)&unk_49D9D40;
          v108 = v134;
          sub_23FD590((__int64)&v129);
          v109 = &unk_49D9D40;
          sub_23FD590((__int64)v117);
          sub_1049740(v61, (__int64)&v95);
          v95 = &unk_49D9D40;
          sub_23FD590((__int64)&v103);
        }
      }
    }
LABEL_33:
    if ( v92 != v94 )
      _libc_free((unsigned __int64)v92);
LABEL_35:
    ++v91;
  }
  while ( (__int64 *)v82 != v91 );
  return v79;
}
