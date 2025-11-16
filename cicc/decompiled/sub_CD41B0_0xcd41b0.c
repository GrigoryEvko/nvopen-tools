// Function: sub_CD41B0
// Address: 0xcd41b0
//
_QWORD *__fastcall sub_CD41B0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 *v7; // r15
  __int64 *v8; // rbx
  __int64 i; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 *v12; // rbx
  __int64 *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdi
  unsigned __int8 v16; // r10
  char v17; // al
  __m128i *v18; // rdx
  unsigned __int8 v19; // r10
  __m128i v20; // xmm0
  void **v21; // rdi
  __m128i *v22; // rdx
  __m128i v23; // xmm0
  __int64 v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // rdi
  __m128i v27; // xmm0
  void *v28; // rdx
  __int64 v29; // rax
  _WORD *v30; // rdx
  __m128i *v31; // rdx
  unsigned __int8 v32; // r10
  __m128i v33; // xmm0
  void **v34; // rdi
  __m128i *v35; // rdx
  __m128i v36; // xmm0
  __int64 v37; // rax
  __m128i *v38; // rdx
  __int64 v39; // rdi
  __m128i v40; // xmm0
  void *v41; // rdx
  __int64 v42; // rax
  _WORD *v43; // rdx
  unsigned int v44; // r10d
  __m128i *v45; // rdx
  unsigned int v46; // r10d
  __m128i v47; // xmm0
  void **v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rax
  __m128i *v51; // rdx
  __int64 v52; // rdi
  __m128i v53; // xmm0
  __int64 v54; // rax
  __int64 v55; // rax
  __m128i *v56; // rdx
  unsigned __int8 v57; // r10
  __m128i si128; // xmm0
  void **v59; // rdi
  __m128i *v60; // rdx
  __m128i v61; // xmm0
  __int64 v62; // rax
  __m128i *v63; // rdx
  __int64 v64; // rdi
  __m128i v65; // xmm0
  void *v66; // rdx
  __int64 v67; // rax
  _WORD *v68; // rdx
  unsigned __int8 v69; // r10
  size_t v70; // r14
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r8
  unsigned __int64 v74; // rax
  __int64 v75; // rcx
  unsigned __int8 v76; // r10
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  void *v105; // r15
  __int64 v106; // rbx
  __int64 v107; // r8
  __int64 v108; // r9
  unsigned __int8 v109; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v110; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v111; // [rsp+Ch] [rbp-A4h]
  unsigned int v112; // [rsp+Ch] [rbp-A4h]
  unsigned int v113; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v114; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v115; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v116; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v117; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v118; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v119; // [rsp+Ch] [rbp-A4h]
  __int64 v120; // [rsp+18h] [rbp-98h]
  _QWORD *v121; // [rsp+20h] [rbp-90h] BYREF
  __int64 v122; // [rsp+28h] [rbp-88h]
  _QWORD v123[2]; // [rsp+30h] [rbp-80h] BYREF
  void *v124; // [rsp+40h] [rbp-70h] BYREF
  __int64 v125; // [rsp+48h] [rbp-68h]
  __int64 v126; // [rsp+50h] [rbp-60h]
  __int64 v127; // [rsp+58h] [rbp-58h]
  __int64 v128; // [rsp+60h] [rbp-50h]
  __int64 v129; // [rsp+68h] [rbp-48h]
  _QWORD *v130; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v3) <= 3 )
  {
    *a1 = 0;
    return a1;
  }
  v5 = a2;
  v6 = sub_22077B0(96);
  if ( v6 )
  {
    memset((void *)v6, 0, 0x60u);
    *(_QWORD *)(v6 + 88) = 1;
    *(_QWORD *)(v6 + 16) = v6 + 32;
    *(_QWORD *)(v6 + 24) = 0x400000000LL;
    *(_QWORD *)(v6 + 64) = v6 + 80;
  }
  if ( *(_DWORD *)v3 != 2135835629 )
    goto LABEL_7;
  v109 = *(_BYTE *)(v3 + 4);
  if ( v109 == 1 )
  {
    v16 = *(_BYTE *)(v3 + 5);
    v17 = 1;
    if ( v16 <= 0x41u )
      goto LABEL_25;
    v121 = v123;
    v129 = 0x100000000LL;
    v130 = &v121;
    v116 = v16;
    v124 = &unk_49DD210;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v89 = sub_904010((__int64)&v124, "Linked container's ");
    v90 = sub_904010(v89, "minor NvvmContainer version (");
    v91 = sub_CB59D0(v90, v116);
    v92 = sub_904010(v91, ") newer than tool ");
    v93 = sub_904010(v92, "(should be ");
    v94 = sub_CB59F0(v93, 65);
    a2 = (__int64)")\n";
    sub_904010(v94, ")\n");
  }
  else
  {
    v121 = v123;
    v129 = 0x100000000LL;
    v130 = &v121;
    v124 = &unk_49DD210;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v56 = (__m128i *)v128;
    v57 = v109;
    if ( (unsigned __int64)(v127 - v128) <= 0x12 )
    {
      v103 = sub_CB6200((__int64)&v124, "Linked container's ", 0x13u);
      v57 = v109;
      v60 = *(__m128i **)(v103 + 32);
      v59 = (void **)v103;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F6E100);
      *(_BYTE *)(v128 + 18) = 32;
      v59 = &v124;
      v56[1].m128i_i16[0] = 29479;
      *v56 = si128;
      v60 = (__m128i *)(v128 + 19);
      v128 += 19;
    }
    if ( (unsigned __int64)((_BYTE *)v59[3] - (_BYTE *)v60) <= 0x1C )
    {
      v119 = v57;
      v102 = sub_CB6200((__int64)v59, "NvvmContainer major version (", 0x1Du);
      v57 = v119;
      v59 = (void **)v102;
    }
    else
    {
      v61 = _mm_load_si128((const __m128i *)&xmmword_3F6E110);
      qmemcpy(&v60[1], "jor version (", 13);
      *v60 = v61;
      v59[4] = (char *)v59[4] + 29;
    }
    v62 = sub_CB59D0((__int64)v59, v57);
    v63 = *(__m128i **)(v62 + 32);
    v64 = v62;
    if ( *(_QWORD *)(v62 + 24) - (_QWORD)v63 <= 0x10u )
    {
      v101 = sub_CB6200(v62, ") not compatible ", 0x11u);
      v66 = *(void **)(v101 + 32);
      v64 = v101;
    }
    else
    {
      v65 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v63[1].m128i_i8[0] = 32;
      *v63 = v65;
      v66 = (void *)(*(_QWORD *)(v62 + 32) + 17LL);
      *(_QWORD *)(v62 + 32) = v66;
    }
    if ( *(_QWORD *)(v64 + 24) - (_QWORD)v66 <= 0xAu )
    {
      v64 = sub_CB6200(v64, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v66, "(should be ", 11);
      *(_QWORD *)(v64 + 32) += 11LL;
    }
    a2 = 1;
    v67 = sub_CB59F0(v64, 1);
    v68 = *(_WORD **)(v67 + 32);
    if ( *(_QWORD *)(v67 + 24) - (_QWORD)v68 <= 1u )
    {
      a2 = (__int64)")\n";
      sub_CB6200(v67, (unsigned __int8 *)")\n", 2u);
    }
    else
    {
      *v68 = 2601;
      *(_QWORD *)(v67 + 32) += 2LL;
    }
  }
  if ( v128 != v126 )
    sub_CB5AE0((__int64 *)&v124);
  v124 = &unk_49DD210;
  sub_CB5840((__int64)&v124);
  if ( v121 != v123 )
  {
    a2 = v123[0] + 1LL;
    j_j___libc_free_0(v121, v123[0] + 1LL);
  }
  v17 = 0;
LABEL_25:
  v110 = *(_BYTE *)(v3 + 6);
  if ( v110 == 2 )
  {
    v76 = *(_BYTE *)(v3 + 7);
    if ( v76 <= 0x62u )
      goto LABEL_41;
    v121 = v123;
    v129 = 0x100000000LL;
    v130 = &v121;
    v114 = v76;
    v124 = &unk_49DD210;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v77 = sub_904010((__int64)&v124, "Linked container's ");
    v78 = sub_904010(v77, "minor NvvmIR version (");
    v79 = sub_CB59D0(v78, v114);
    v80 = sub_904010(v79, ") newer than tool ");
    v81 = sub_904010(v80, "(should be ");
    v82 = sub_CB59F0(v81, 98);
    a2 = (__int64)")\n";
    sub_904010(v82, ")\n");
  }
  else
  {
    v121 = v123;
    v129 = 0x100000000LL;
    v130 = &v121;
    v124 = &unk_49DD210;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v18 = (__m128i *)v128;
    v19 = v110;
    if ( (unsigned __int64)(v127 - v128) <= 0x12 )
    {
      v95 = sub_CB6200((__int64)&v124, "Linked container's ", 0x13u);
      v19 = v110;
      v22 = *(__m128i **)(v95 + 32);
      v21 = (void **)v95;
    }
    else
    {
      v20 = _mm_load_si128((const __m128i *)&xmmword_3F6E100);
      *(_BYTE *)(v128 + 18) = 32;
      v21 = &v124;
      v18[1].m128i_i16[0] = 29479;
      *v18 = v20;
      v22 = (__m128i *)(v128 + 19);
      v128 += 19;
    }
    if ( (unsigned __int64)((_BYTE *)v21[3] - (_BYTE *)v22) <= 0x15 )
    {
      v118 = v19;
      v100 = sub_CB6200((__int64)v21, "NvvmIR major version (", 0x16u);
      v19 = v118;
      v21 = (void **)v100;
    }
    else
    {
      v23 = _mm_load_si128((const __m128i *)&xmmword_3F6E120);
      v22[1].m128i_i32[0] = 1852795251;
      v22[1].m128i_i16[2] = 10272;
      *v22 = v23;
      v21[4] = (char *)v21[4] + 22;
    }
    v24 = sub_CB59D0((__int64)v21, v19);
    v25 = *(__m128i **)(v24 + 32);
    v26 = v24;
    if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0x10u )
    {
      v99 = sub_CB6200(v24, ") not compatible ", 0x11u);
      v28 = *(void **)(v99 + 32);
      v26 = v99;
    }
    else
    {
      v27 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v25[1].m128i_i8[0] = 32;
      *v25 = v27;
      v28 = (void *)(*(_QWORD *)(v24 + 32) + 17LL);
      *(_QWORD *)(v24 + 32) = v28;
    }
    if ( *(_QWORD *)(v26 + 24) - (_QWORD)v28 <= 0xAu )
    {
      v26 = sub_CB6200(v26, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v28, "(should be ", 11);
      *(_QWORD *)(v26 + 32) += 11LL;
    }
    a2 = 2;
    v29 = sub_CB59F0(v26, 2);
    v30 = *(_WORD **)(v29 + 32);
    if ( *(_QWORD *)(v29 + 24) - (_QWORD)v30 <= 1u )
    {
      a2 = (__int64)")\n";
      sub_CB6200(v29, (unsigned __int8 *)")\n", 2u);
    }
    else
    {
      *v30 = 2601;
      *(_QWORD *)(v29 + 32) += 2LL;
    }
  }
  if ( v128 != v126 )
    sub_CB5AE0((__int64 *)&v124);
  v124 = &unk_49DD210;
  sub_CB5840((__int64)&v124);
  if ( v121 != v123 )
  {
    a2 = v123[0] + 1LL;
    j_j___libc_free_0(v121, v123[0] + 1LL);
  }
  v17 = 0;
LABEL_41:
  v111 = *(_BYTE *)(v3 + 8);
  if ( v111 != 3 )
  {
    v121 = v123;
    v129 = 0x100000000LL;
    v130 = &v121;
    v124 = &unk_49DD210;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v31 = (__m128i *)v128;
    v32 = v111;
    if ( (unsigned __int64)(v127 - v128) <= 0x12 )
    {
      v96 = sub_CB6200((__int64)&v124, "Linked container's ", 0x13u);
      v32 = v111;
      v35 = *(__m128i **)(v96 + 32);
      v34 = (void **)v96;
    }
    else
    {
      v33 = _mm_load_si128((const __m128i *)&xmmword_3F6E100);
      *(_BYTE *)(v128 + 18) = 32;
      v34 = &v124;
      v31[1].m128i_i16[0] = 29479;
      *v31 = v33;
      v35 = (__m128i *)(v128 + 19);
      v128 += 19;
    }
    if ( (unsigned __int64)((_BYTE *)v34[3] - (_BYTE *)v35) <= 0x18 )
    {
      v117 = v32;
      v98 = sub_CB6200((__int64)v34, "NvvmDebug major version (", 0x19u);
      v32 = v117;
      v34 = (void **)v98;
    }
    else
    {
      v36 = _mm_load_si128((const __m128i *)&xmmword_3F6E130);
      v35[1].m128i_i8[8] = 40;
      v35[1].m128i_i64[0] = 0x206E6F6973726576LL;
      *v35 = v36;
      v34[4] = (char *)v34[4] + 25;
    }
    v37 = sub_CB59D0((__int64)v34, v32);
    v38 = *(__m128i **)(v37 + 32);
    v39 = v37;
    if ( *(_QWORD *)(v37 + 24) - (_QWORD)v38 <= 0x10u )
    {
      v97 = sub_CB6200(v37, ") not compatible ", 0x11u);
      v41 = *(void **)(v97 + 32);
      v39 = v97;
    }
    else
    {
      v40 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v38[1].m128i_i8[0] = 32;
      *v38 = v40;
      v41 = (void *)(*(_QWORD *)(v37 + 32) + 17LL);
      *(_QWORD *)(v37 + 32) = v41;
    }
    if ( *(_QWORD *)(v39 + 24) - (_QWORD)v41 <= 0xAu )
    {
      v39 = sub_CB6200(v39, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v41, "(should be ", 11);
      *(_QWORD *)(v39 + 32) += 11LL;
    }
    v42 = sub_CB59F0(v39, 3);
    v43 = *(_WORD **)(v42 + 32);
    if ( *(_QWORD *)(v42 + 24) - (_QWORD)v43 <= 1u )
    {
      a2 = (__int64)")\n";
      sub_CB6200(v42, (unsigned __int8 *)")\n", 2u);
    }
    else
    {
      a2 = 2601;
      *v43 = 2601;
      *(_QWORD *)(v42 + 32) += 2LL;
    }
LABEL_52:
    if ( v128 != v126 )
      sub_CB5AE0((__int64 *)&v124);
    v124 = &unk_49DD210;
    sub_CB5840((__int64)&v124);
    if ( v121 != v123 )
    {
      a2 = v123[0] + 1LL;
      j_j___libc_free_0(v121, v123[0] + 1LL);
    }
    v44 = *(unsigned __int8 *)(v3 + 11) + 100 * *(unsigned __int8 *)(v3 + 10);
    if ( v44 <= 0x7D0 )
      goto LABEL_7;
    goto LABEL_57;
  }
  v69 = *(_BYTE *)(v3 + 9);
  if ( v69 > 2u )
  {
    v121 = v123;
    v129 = 0x100000000LL;
    v130 = &v121;
    v115 = v69;
    v124 = &unk_49DD210;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v83 = sub_904010((__int64)&v124, "Linked container's ");
    v84 = sub_904010(v83, "minor NvvmDebug version (");
    v85 = sub_CB59D0(v84, v115);
    v86 = sub_904010(v85, ") newer than tool ");
    v87 = sub_904010(v86, "(should be ");
    v88 = sub_CB59F0(v87, 2);
    a2 = (__int64)")\n";
    sub_904010(v88, ")\n");
    goto LABEL_52;
  }
  v44 = *(unsigned __int8 *)(v3 + 11) + 100 * *(unsigned __int8 *)(v3 + 10);
  if ( v44 > 0x7D0 )
  {
LABEL_57:
    v112 = v44;
    v121 = v123;
    v129 = 0x100000000LL;
    v122 = 0;
    LOBYTE(v123[0]) = 0;
    v124 = &unk_49DD210;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    v130 = &v121;
    sub_CB5980((__int64)&v124, 0, 0, 0);
    v45 = (__m128i *)v128;
    v46 = v112;
    if ( (unsigned __int64)(v127 - v128) <= 0x12 )
    {
      v104 = sub_CB6200((__int64)&v124, "Linked container's ", 0x13u);
      v46 = v112;
      v48 = (void **)v104;
    }
    else
    {
      v47 = _mm_load_si128((const __m128i *)&xmmword_3F6E100);
      *(_BYTE *)(v128 + 18) = 32;
      v48 = &v124;
      v45[1].m128i_i16[0] = 29479;
      *v45 = v47;
      v128 += 19;
    }
    v113 = v46;
    v49 = sub_904010((__int64)v48, "LLVM version (");
    v50 = sub_CB59D0(v49, v113);
    v51 = *(__m128i **)(v50 + 32);
    v52 = v50;
    if ( *(_QWORD *)(v50 + 24) - (_QWORD)v51 <= 0x10u )
    {
      v52 = sub_CB6200(v50, ") not compatible ", 0x11u);
    }
    else
    {
      v53 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v51[1].m128i_i8[0] = 32;
      *v51 = v53;
      *(_QWORD *)(v50 + 32) += 17LL;
    }
    v54 = sub_904010(v52, "(should be ");
    v55 = sub_CB59D0(v54, 0x7D0u);
    a2 = (__int64)")\n";
    sub_904010(v55, ")\n");
    if ( v128 != v126 )
      sub_CB5AE0((__int64 *)&v124);
    v124 = &unk_49DD210;
    sub_CB5840((__int64)&v124);
    if ( v121 != v123 )
    {
      a2 = v123[0] + 1LL;
      j_j___libc_free_0(v121, v123[0] + 1LL);
    }
    goto LABEL_7;
  }
  if ( v17 )
  {
    v70 = 0;
    v71 = sub_CD1D80(v5, (__int64 *)v6);
    a2 = *(_QWORD *)(v5 + 16);
    v72 = *(_QWORD *)(v5 + 8);
    v73 = v71;
    v74 = *(unsigned int *)(v3 + 20);
    v75 = a2 - v72;
    if ( v74 <= a2 - v72 )
    {
      a2 = v72 + v74;
      v70 = v75 - v74;
    }
    if ( v73 && *(_DWORD *)(v73 + 228) )
    {
      v120 = v73;
      v105 = (void *)sub_2207820(v70);
      memcpy(v105, (const void *)a2, v70);
      v106 = sub_16886D0(*(unsigned int *)(v120 + 228));
      sub_16887A0(v106, v105, (unsigned int)v70);
      sub_1688720(v106);
      a2 = (__int64)v105;
      LOWORD(v128) = 257;
      sub_C7DE20(&v121, v105, v70, (__int64)&v124, v107, v108);
      j_j___libc_free_0_0(v105);
      *a1 = v121;
    }
    else
    {
      sub_C7DA90(&v124, a2, v70, byte_3F871B3, 0, 0);
      *a1 = v124;
    }
    goto LABEL_8;
  }
LABEL_7:
  *a1 = 0;
LABEL_8:
  if ( v6 )
  {
    v7 = *(__int64 **)(v6 + 16);
    v8 = &v7[*(unsigned int *)(v6 + 24)];
    if ( v7 != v8 )
    {
      for ( i = *(_QWORD *)(v6 + 16); ; i = *(_QWORD *)(v6 + 16) )
      {
        v10 = *v7;
        v11 = (unsigned int)(((__int64)v7 - i) >> 3) >> 7;
        a2 = 4096LL << v11;
        if ( v11 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v7;
        sub_C7D6A0(v10, a2, 16);
        if ( v8 == v7 )
          break;
      }
    }
    v12 = *(__int64 **)(v6 + 64);
    v13 = &v12[2 * *(unsigned int *)(v6 + 72)];
    if ( v12 != v13 )
    {
      do
      {
        a2 = v12[1];
        v14 = *v12;
        v12 += 2;
        sub_C7D6A0(v14, a2, 16);
      }
      while ( v13 != v12 );
      v13 = *(__int64 **)(v6 + 64);
    }
    if ( v13 != (__int64 *)(v6 + 80) )
      _libc_free(v13, a2);
    v15 = *(_QWORD *)(v6 + 16);
    if ( v15 != v6 + 32 )
      _libc_free(v15, a2);
    j_j___libc_free_0(v6, 96);
  }
  return a1;
}
