// Function: sub_1DFF720
// Address: 0x1dff720
//
__int64 __fastcall sub_1DFF720(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // r14d
  _QWORD *v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r8
  unsigned int v18; // esi
  __int64 v19; // rcx
  __int64 v20; // r9
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r12
  unsigned int v25; // esi
  __int64 v26; // rcx
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // r8
  __int64 v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // r8
  _DWORD *v33; // rdx
  __int64 v34; // r12
  _BOOL4 v35; // esi
  __int64 v36; // rcx
  __int64 v37; // r13
  unsigned __int64 v38; // rdx
  __int64 v39; // r8
  unsigned __int64 v40; // rdi
  int v41; // r10d
  int v42; // r10d
  unsigned int v43; // ecx
  int v44; // r10d
  char *v45; // r9
  __int64 i; // rdi
  unsigned __int64 v47; // rdi
  int v48; // eax
  unsigned int v49; // esi
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // rdi
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // r10
  unsigned __int64 v55; // rcx
  __int64 v56; // rax
  char *v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rdi
  _BYTE *v60; // rax
  __int64 v61; // r13
  void *v62; // rax
  __m128i *v63; // rdx
  __int64 v64; // r8
  __int64 v65; // r12
  _QWORD *v66; // rax
  __m128i *v67; // rdx
  __int64 v68; // r8
  __m128i si128; // xmm0
  _QWORD *v70; // rbx
  _QWORD *v71; // r12
  __int64 v72; // r13
  unsigned int v73; // edi
  unsigned int v74; // r10d
  unsigned int v75; // edx
  __int64 v76; // rcx
  __int64 v77; // rax
  unsigned int v78; // edi
  unsigned int v79; // r10d
  unsigned int v80; // edx
  __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rcx
  int v84; // edx
  int v85; // r9d
  int v86; // r11d
  __int64 *v87; // rdi
  int v88; // eax
  int v89; // edx
  __int64 *v90; // rdi
  int v91; // edx
  int v92; // r11d
  __int64 *v93; // r13
  __int64 *v94; // r12
  _BOOL8 v95; // [rsp+8h] [rbp-118h]
  unsigned int v96; // [rsp+8h] [rbp-118h]
  int v97; // [rsp+10h] [rbp-110h]
  _BOOL8 v98; // [rsp+10h] [rbp-110h]
  int v99; // [rsp+10h] [rbp-110h]
  _BOOL8 v100; // [rsp+10h] [rbp-110h]
  __int64 v101; // [rsp+18h] [rbp-108h]
  unsigned int v103; // [rsp+28h] [rbp-F8h]
  unsigned int v104; // [rsp+28h] [rbp-F8h]
  _BOOL8 v105; // [rsp+28h] [rbp-F8h]
  char v106; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v107; // [rsp+30h] [rbp-F0h]
  __int64 v108; // [rsp+38h] [rbp-E8h]
  __int64 v109; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v110[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v111; // [rsp+60h] [rbp-C0h]
  __int64 v112; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v113; // [rsp+78h] [rbp-A8h]
  __int64 v114; // [rsp+80h] [rbp-A0h]
  unsigned int v115; // [rsp+88h] [rbp-98h]
  __m128i *v116; // [rsp+90h] [rbp-90h]
  size_t v117; // [rsp+98h] [rbp-88h]
  __m128i v118; // [rsp+A0h] [rbp-80h] BYREF
  const char *v119; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v120; // [rsp+B8h] [rbp-68h]
  _QWORD v121[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 *v122; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v123; // [rsp+D8h] [rbp-48h]
  _QWORD v124[8]; // [rsp+E0h] [rbp-40h] BYREF

  v3 = *(_QWORD *)a1;
  v110[0] = 0;
  v4 = *(_QWORD *)(v3 + 328);
  v110[1] = 0;
  v111 = 0;
  sub_1DFC3F0((__int64)v110, v4);
  v5 = sub_1DF88C0(v110, *(_QWORD *)(a1 + 8));
  if ( (_BYTE)v5 )
    goto LABEL_2;
  v7 = *(_QWORD **)(a1 + 120);
  ++*(_QWORD *)(a1 + 112);
  v101 = a1 + 112;
  v8 = *(_DWORD *)(a1 + 136);
  v113 = v7;
  v9 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  v115 = v8;
  v112 = 1;
  v114 = v9;
  sub_1DFEBA0((_QWORD *)a1);
  v107 = 1;
  v10 = *(_QWORD *)(*(_QWORD *)a1 + 328LL);
  v108 = *(_QWORD *)a1 + 320LL;
  if ( v10 == v108 )
    goto LABEL_58;
  while ( 1 )
  {
    v11 = *(_QWORD *)(a1 + 16);
    v109 = v10;
    sub_1E06620(v11);
    v12 = *(_QWORD *)(v11 + 1312);
    v13 = *(unsigned int *)(v12 + 48);
    if ( !(_DWORD)v13 )
      goto LABEL_57;
    v14 = *(_QWORD *)(v12 + 32);
    v15 = (v13 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v10 != *v16 )
    {
      v84 = 1;
      while ( v17 != -8 )
      {
        v85 = v84 + 1;
        v15 = (v13 - 1) & (v84 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( *v16 == v10 )
          goto LABEL_8;
        v84 = v85;
      }
      goto LABEL_57;
    }
LABEL_8:
    if ( v16 == (__int64 *)(v14 + 16 * v13) || !v16[1] )
      goto LABEL_57;
    v18 = *(_DWORD *)(a1 + 136);
    if ( !v18 )
    {
      ++*(_QWORD *)(a1 + 112);
      goto LABEL_115;
    }
    v19 = v109;
    v20 = *(_QWORD *)(a1 + 120);
    v21 = (v18 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
    v22 = (__int64 *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( *v22 != v109 )
    {
      v86 = 1;
      v87 = 0;
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v87 )
          v22 = v87;
        v21 = (v18 - 1) & (v86 + v21);
        v94 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v94;
        if ( v109 == *v94 )
        {
          v24 = v94[1];
          goto LABEL_13;
        }
        ++v86;
        v87 = v22;
        v22 = (__int64 *)(v20 + 16LL * v21);
      }
      if ( !v87 )
        v87 = v22;
      v88 = *(_DWORD *)(a1 + 128);
      ++*(_QWORD *)(a1 + 112);
      v89 = v88 + 1;
      if ( 4 * (v88 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a1 + 132) - v89 <= v18 >> 3 )
          goto LABEL_116;
        goto LABEL_108;
      }
LABEL_115:
      v18 *= 2;
LABEL_116:
      sub_1DFD130(v101, v18);
      sub_1DF9290(v101, &v109, &v122);
      v87 = v122;
      v19 = v109;
      v89 = *(_DWORD *)(a1 + 128) + 1;
LABEL_108:
      *(_DWORD *)(a1 + 128) = v89;
      if ( *v87 != -8 )
        --*(_DWORD *)(a1 + 132);
      *v87 = v19;
      v24 = 0;
      v87[1] = 0;
      v25 = v115;
      if ( !v115 )
        goto LABEL_111;
      goto LABEL_14;
    }
    v24 = v22[1];
LABEL_13:
    v25 = v115;
    if ( !v115 )
    {
LABEL_111:
      ++v112;
LABEL_112:
      v25 *= 2;
LABEL_113:
      sub_1DFD130((__int64)&v112, v25);
      sub_1DF9290((__int64)&v112, &v109, &v122);
      v90 = v122;
      v26 = v109;
      v91 = v114 + 1;
      goto LABEL_123;
    }
LABEL_14:
    v26 = v109;
    v27 = (v25 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
    v28 = &v113[2 * v27];
    v29 = *v28;
    if ( *v28 == v109 )
    {
      v30 = v28[1];
      goto LABEL_16;
    }
    v92 = 1;
    v90 = 0;
    while ( v29 != -8 )
    {
      if ( v29 != -16 || v90 )
        v28 = v90;
      v27 = (v25 - 1) & (v92 + v27);
      v93 = &v113[2 * v27];
      v29 = *v93;
      if ( v109 == *v93 )
      {
        v30 = v93[1];
        goto LABEL_16;
      }
      ++v92;
      v90 = v28;
      v28 = &v113[2 * v27];
    }
    if ( !v90 )
      v90 = v28;
    ++v112;
    v91 = v114 + 1;
    if ( 4 * ((int)v114 + 1) >= 3 * v25 )
      goto LABEL_112;
    if ( v25 - HIDWORD(v114) - v91 <= v25 >> 3 )
      goto LABEL_113;
LABEL_123:
    LODWORD(v114) = v91;
    if ( *v90 != -8 )
      --HIDWORD(v114);
    *v90 = v26;
    v30 = 0;
    v90[1] = 0;
LABEL_16:
    if ( *(_DWORD *)(v30 + 8) != *(_DWORD *)(v24 + 8) || *(_DWORD *)(v30 + 12) != *(_DWORD *)(v24 + 12) )
      goto LABEL_17;
    v73 = (unsigned int)(*(_DWORD *)(v30 + 40) + 63) >> 6;
    v74 = (unsigned int)(*(_DWORD *)(v24 + 40) + 63) >> 6;
    v75 = v74;
    if ( v73 <= v74 )
      v75 = (unsigned int)(*(_DWORD *)(v30 + 40) + 63) >> 6;
    if ( v75 )
      break;
LABEL_73:
    if ( v75 != v73 )
    {
      while ( !*(_QWORD *)(*(_QWORD *)(v30 + 24) + 8LL * v75) )
      {
        if ( v73 == ++v75 )
          goto LABEL_76;
      }
      goto LABEL_17;
    }
    if ( v73 != v74 )
    {
      while ( !*(_QWORD *)(*(_QWORD *)(v24 + 24) + 8LL * v73) )
      {
        if ( v74 == ++v73 )
          goto LABEL_76;
      }
      goto LABEL_17;
    }
LABEL_76:
    v78 = (unsigned int)(*(_DWORD *)(v30 + 64) + 63) >> 6;
    v79 = (unsigned int)(*(_DWORD *)(v24 + 64) + 63) >> 6;
    v80 = v79;
    if ( v78 <= v79 )
      v80 = (unsigned int)(*(_DWORD *)(v30 + 64) + 63) >> 6;
    if ( v80 )
    {
      v81 = v80 + 1;
      v82 = 1;
      while ( *(_QWORD *)(*(_QWORD *)(v30 + 48) + 8 * v82 - 8) == *(_QWORD *)(*(_QWORD *)(v24 + 48) + 8 * v82 - 8) )
      {
        v80 = v82++;
        if ( v81 == v82 )
          goto LABEL_82;
      }
      goto LABEL_17;
    }
LABEL_82:
    if ( v80 != v78 )
    {
      while ( !*(_QWORD *)(*(_QWORD *)(v30 + 48) + 8LL * v80) )
      {
        if ( v78 == ++v80 )
          goto LABEL_57;
      }
      goto LABEL_17;
    }
    if ( v78 != v79 )
    {
      while ( !*(_QWORD *)(*(_QWORD *)(v24 + 48) + 8LL * v78) )
      {
        if ( v79 == ++v78 )
          goto LABEL_57;
      }
      goto LABEL_17;
    }
LABEL_57:
    v10 = *(_QWORD *)(v10 + 8);
    if ( v108 == v10 )
      goto LABEL_58;
  }
  v76 = v75 + 1;
  v77 = 1;
  while ( *(_QWORD *)(*(_QWORD *)(v30 + 24) + 8 * v77 - 8) == *(_QWORD *)(*(_QWORD *)(v24 + 24) + 8 * v77 - 8) )
  {
    v75 = v77++;
    if ( v76 == v77 )
      goto LABEL_73;
  }
LABEL_17:
  if ( a2 )
  {
    v31 = sub_16E8CB0();
    v33 = (_DWORD *)v31[3];
    v34 = (__int64)v31;
    if ( v31[2] - (_QWORD)v33 <= 3u )
    {
      v34 = sub_16E7EE0((__int64)v31, "BB: ", 4u);
    }
    else
    {
      *v33 = 540688962;
      v31[3] += 4LL;
    }
    v35 = *(int *)(v109 + 48) < 0;
    v36 = abs32(*(_DWORD *)(v109 + 48));
    if ( (unsigned int)v36 <= 9 )
    {
      v100 = *(int *)(v109 + 48) < 0;
      v106 = v36;
      v119 = (const char *)v121;
      sub_2240A50(&v119, (unsigned int)(v35 + 1), 45, v36, v32);
      v45 = (char *)&v119[v100];
      LOBYTE(v43) = v106;
      goto LABEL_35;
    }
    if ( (unsigned int)v36 <= 0x63 )
    {
      v98 = *(int *)(v109 + 48) < 0;
      v104 = v36;
      v119 = (const char *)v121;
      sub_2240A50(&v119, (unsigned int)(v35 + 2), 45, v36, v32);
      v43 = v104;
      v45 = (char *)&v119[v98];
    }
    else
    {
      if ( (unsigned int)v36 <= 0x3E7 )
      {
        v42 = 2;
        v39 = 3;
        v37 = (unsigned int)v36;
      }
      else
      {
        v37 = (unsigned int)v36;
        v38 = (unsigned int)v36;
        if ( (unsigned int)v36 <= 0x270F )
        {
          v42 = 3;
          v39 = 4;
        }
        else
        {
          LODWORD(v39) = 1;
          do
          {
            v40 = v38;
            v41 = v39;
            v39 = (unsigned int)(v39 + 4);
            v38 /= 0x2710u;
            if ( v40 <= 0x1869F )
            {
              v42 = v41 + 3;
              goto LABEL_30;
            }
            if ( (unsigned int)v38 <= 0x63 )
            {
              v96 = v36;
              v99 = v39;
              v105 = *(int *)(v109 + 48) < 0;
              v119 = (const char *)v121;
              sub_2240A50(&v119, (unsigned int)(v41 + v35 + 5), 45, v36, v39);
              v43 = v96;
              v45 = (char *)&v119[v105];
              v44 = v99;
              goto LABEL_31;
            }
            if ( (unsigned int)v38 <= 0x3E7 )
            {
              v39 = (unsigned int)(v41 + 6);
              v42 = v41 + 5;
              goto LABEL_30;
            }
          }
          while ( (unsigned int)v38 > 0x270F );
          v39 = (unsigned int)(v41 + 7);
          v42 = v41 + 6;
        }
      }
LABEL_30:
      v95 = *(int *)(v109 + 48) < 0;
      v97 = v42;
      v103 = v36;
      v119 = (const char *)v121;
      sub_2240A50(&v119, (unsigned int)(v39 + v35), 45, v36, v39);
      v43 = v103;
      v44 = v97;
      v45 = (char *)&v119[v95];
LABEL_31:
      for ( i = v37; ; i = v43 )
      {
        v47 = (unsigned __int64)(1374389535 * i) >> 37;
        v48 = v43 - 100 * v47;
        v49 = v43;
        v43 = v47;
        v50 = (unsigned int)(2 * v48);
        v51 = (unsigned int)(v50 + 1);
        LOBYTE(v50) = a00010203040506[v50];
        v45[v44] = a00010203040506[v51];
        v52 = (unsigned int)(v44 - 1);
        v44 -= 2;
        v45[v52] = v50;
        if ( v49 <= 0x270F )
          break;
      }
      if ( v49 <= 0x3E7 )
      {
LABEL_35:
        *v45 = v43 + 48;
        goto LABEL_36;
      }
    }
    v83 = 2 * v43;
    v45[1] = a00010203040506[(unsigned int)(v83 + 1)];
    *v45 = a00010203040506[v83];
LABEL_36:
    v122 = v124;
    sub_1DF72E0((__int64 *)&v122, "bb.", (__int64)"");
    v53 = 15;
    v54 = 15;
    if ( v122 != v124 )
      v54 = v124[0];
    v55 = v123 + v120;
    if ( v123 + v120 <= v54 )
      goto LABEL_42;
    if ( v119 != (const char *)v121 )
      v53 = v121[0];
    if ( v55 <= v53 )
    {
      v56 = sub_2241130(&v119, 0, 0, v122, v123);
      v58 = v56 + 16;
      v116 = &v118;
      v57 = *(char **)v56;
      if ( *(_QWORD *)v56 == v56 + 16 )
        goto LABEL_100;
LABEL_43:
      v116 = (__m128i *)v57;
      v118.m128i_i64[0] = *(_QWORD *)(v56 + 16);
    }
    else
    {
LABEL_42:
      v56 = sub_2241490(&v122, v119, v120, v55, v123);
      v116 = &v118;
      v57 = *(char **)v56;
      v58 = v56 + 16;
      if ( *(_QWORD *)v56 != v56 + 16 )
        goto LABEL_43;
LABEL_100:
      v118 = _mm_loadu_si128((const __m128i *)(v56 + 16));
    }
    v117 = *(_QWORD *)(v56 + 8);
    *(_QWORD *)v56 = v58;
    *(_QWORD *)(v56 + 8) = 0;
    *(_BYTE *)(v56 + 16) = 0;
    if ( v122 != v124 )
      j_j___libc_free_0(v122, v124[0] + 1LL);
    if ( v119 != (const char *)v121 )
      j_j___libc_free_0(v119, v121[0] + 1LL);
    v59 = sub_16E7EE0(v34, v116->m128i_i8, v117);
    v60 = *(_BYTE **)(v59 + 24);
    if ( *(_BYTE **)(v59 + 16) == v60 )
    {
      sub_16E7EE0(v59, "\n", 1u);
    }
    else
    {
      *v60 = 10;
      ++*(_QWORD *)(v59 + 24);
    }
    if ( v116 != &v118 )
      j_j___libc_free_0(v116, v118.m128i_i64[0] + 1);
    v61 = sub_1DFD350(v101, &v109)[1];
    v62 = sub_16E8CB0();
    v63 = (__m128i *)*((_QWORD *)v62 + 3);
    v64 = (__int64)v62;
    if ( *((_QWORD *)v62 + 2) - (_QWORD)v63 <= 0xFu )
    {
      v64 = sub_16E7EE0((__int64)v62, "Correct RP Info\n", 0x10u);
    }
    else
    {
      *v63 = _mm_load_si128((const __m128i *)&xmmword_42EAB10);
      *((_QWORD *)v62 + 3) += 16LL;
    }
    sub_1DF8C50(a1, v64, v61);
    v65 = sub_1DFD350((__int64)&v112, &v109)[1];
    v66 = sub_16E8CB0();
    v67 = (__m128i *)v66[3];
    v68 = (__int64)v66;
    if ( v66[2] - (_QWORD)v67 <= 0x11u )
    {
      v68 = sub_16E7EE0((__int64)v66, "Incorrect RP Info\n", 0x12u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EAB20);
      v67[1].m128i_i16[0] = 2671;
      *v67 = si128;
      v66[3] += 18LL;
    }
    sub_1DF8C50(a1, v68, v65);
    v107 = 0;
    goto LABEL_57;
  }
  v107 = 0;
LABEL_58:
  if ( v115 )
  {
    v70 = v113;
    v71 = &v113[2 * v115];
    do
    {
      if ( *v70 != -16 && *v70 != -8 )
      {
        v72 = v70[1];
        if ( v72 )
        {
          _libc_free(*(_QWORD *)(v72 + 48));
          _libc_free(*(_QWORD *)(v72 + 24));
          j_j___libc_free_0(v72, 72);
        }
      }
      v70 += 2;
    }
    while ( v71 != v70 );
  }
  j___libc_free_0(v113);
  v5 = v107;
LABEL_2:
  if ( v110[0] )
    j_j___libc_free_0(v110[0], v111 - v110[0]);
  return v5;
}
