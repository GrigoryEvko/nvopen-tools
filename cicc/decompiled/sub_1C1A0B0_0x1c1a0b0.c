// Function: sub_1C1A0B0
// Address: 0x1c1a0b0
//
_QWORD *__fastcall sub_1C1A0B0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v6; // r12
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r14
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  char v14; // al
  __int64 v15; // rax
  unsigned __int8 v16; // cl
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __m128i v19; // xmm0
  __int64 v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // rdi
  __m128i v23; // xmm0
  void *v24; // rdx
  __int64 v25; // rax
  _WORD *v26; // rdx
  __int64 v27; // rax
  unsigned __int8 v28; // cl
  __m128i *v29; // rdx
  __int64 v30; // rdi
  __m128i v31; // xmm0
  __int64 v32; // rax
  __m128i *v33; // rdx
  __int64 v34; // rdi
  __m128i v35; // xmm0
  void *v36; // rdx
  __int64 v37; // rax
  _WORD *v38; // rdx
  unsigned int v39; // ecx
  __int64 v40; // rax
  unsigned int v41; // ecx
  void *v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rax
  __m128i *v45; // rdx
  __int64 v46; // rdi
  __m128i v47; // xmm0
  void *v48; // rdx
  __int64 v49; // rax
  _WORD *v50; // rdx
  __int64 v51; // rax
  unsigned __int8 v52; // cl
  __m128i *v53; // rdx
  __int64 v54; // rdi
  __m128i si128; // xmm0
  __int64 v56; // rax
  __m128i *v57; // rdx
  __int64 v58; // rdi
  __m128i v59; // xmm0
  void *v60; // rdx
  __int64 v61; // rax
  _WORD *v62; // rdx
  size_t v63; // r14
  __int64 v64; // rax
  const void *v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // r8
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rcx
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
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
  _BYTE *v96; // r15
  __int64 v97; // rdx
  _QWORD *v98; // rbx
  unsigned __int8 src; // [rsp+10h] [rbp-90h]
  unsigned __int8 srca; // [rsp+10h] [rbp-90h]
  unsigned __int8 srcb; // [rsp+10h] [rbp-90h]
  unsigned int srcc; // [rsp+10h] [rbp-90h]
  unsigned __int8 srcd; // [rsp+10h] [rbp-90h]
  unsigned __int8 srce; // [rsp+10h] [rbp-90h]
  unsigned __int8 srcf; // [rsp+10h] [rbp-90h]
  __int64 v106; // [rsp+18h] [rbp-88h]
  _QWORD *v107; // [rsp+20h] [rbp-80h] BYREF
  __int64 v108; // [rsp+28h] [rbp-78h]
  _QWORD v109[2]; // [rsp+30h] [rbp-70h] BYREF
  void *v110; // [rsp+40h] [rbp-60h] BYREF
  __int64 v111; // [rsp+48h] [rbp-58h]
  __int64 v112; // [rsp+50h] [rbp-50h]
  __int64 v113; // [rsp+58h] [rbp-48h]
  int v114; // [rsp+60h] [rbp-40h]
  _QWORD *v115; // [rsp+68h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v3) <= 3 )
  {
    *a1 = 0;
    return a1;
  }
  v6 = sub_22077B0(104);
  if ( v6 )
  {
    memset((void *)v6, 0, 0x68u);
    *(_QWORD *)(v6 + 88) = 1;
    *(_QWORD *)(v6 + 16) = v6 + 32;
    *(_QWORD *)(v6 + 24) = 0x400000000LL;
    *(_QWORD *)(v6 + 64) = v6 + 80;
  }
  if ( *(_DWORD *)v3 != 2135835629 )
    goto LABEL_7;
  src = *(_BYTE *)(v3 + 4);
  if ( src == 1 )
  {
    v14 = 1;
    if ( *(_BYTE *)(v3 + 5) <= 0x41u )
      goto LABEL_21;
    srcf = *(_BYTE *)(v3 + 5);
    v107 = v109;
    v115 = &v107;
    v108 = 0;
    LOBYTE(v109[0]) = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v110 = &unk_49EFBE0;
    v82 = sub_1263B40((__int64)&v110, "Linked container's ");
    v83 = sub_1263B40(v82, "minor NvvmContainer version (");
    v84 = sub_16E7A90(v83, srcf);
    v85 = sub_1263B40(v84, ") newer than tool ");
    v86 = sub_1263B40(v85, "(should be ");
    v87 = sub_16E7AB0(v86, 65);
    sub_1263B40(v87, ")\n");
  }
  else
  {
    v108 = 0;
    v107 = v109;
    v115 = &v107;
    LOBYTE(v109[0]) = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v110 = &unk_49EFBE0;
    v51 = sub_16E7EE0((__int64)&v110, "Linked container's ", 0x13u);
    v52 = src;
    v53 = *(__m128i **)(v51 + 24);
    v54 = v51;
    if ( *(_QWORD *)(v51 + 16) - (_QWORD)v53 <= 0x1Cu )
    {
      v89 = sub_16E7EE0(v51, "NvvmContainer major version (", 0x1Du);
      v52 = src;
      v54 = v89;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F6E110);
      qmemcpy(&v53[1], "jor version (", 13);
      *v53 = si128;
      *(_QWORD *)(v51 + 24) += 29LL;
    }
    v56 = sub_16E7A90(v54, v52);
    v57 = *(__m128i **)(v56 + 24);
    v58 = v56;
    if ( *(_QWORD *)(v56 + 16) - (_QWORD)v57 <= 0x10u )
    {
      v92 = sub_16E7EE0(v56, ") not compatible ", 0x11u);
      v60 = *(void **)(v92 + 24);
      v58 = v92;
    }
    else
    {
      v59 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v57[1].m128i_i8[0] = 32;
      *v57 = v59;
      v60 = (void *)(*(_QWORD *)(v56 + 24) + 17LL);
      *(_QWORD *)(v56 + 24) = v60;
    }
    if ( *(_QWORD *)(v58 + 16) - (_QWORD)v60 <= 0xAu )
    {
      v58 = sub_16E7EE0(v58, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v60, "(should be ", 11);
      *(_QWORD *)(v58 + 24) += 11LL;
    }
    v61 = sub_16E7AB0(v58, 1);
    v62 = *(_WORD **)(v61 + 24);
    if ( *(_QWORD *)(v61 + 16) - (_QWORD)v62 <= 1u )
    {
      sub_16E7EE0(v61, ")\n", 2u);
    }
    else
    {
      *v62 = 2601;
      *(_QWORD *)(v61 + 24) += 2LL;
    }
  }
  if ( v113 != v111 )
    sub_16E7BA0((__int64 *)&v110);
  sub_16E7BC0((__int64 *)&v110);
  if ( v107 != v109 )
    j_j___libc_free_0(v107, v109[0] + 1LL);
  v14 = 0;
LABEL_21:
  srca = *(_BYTE *)(v3 + 6);
  if ( srca == 2 )
  {
    if ( *(_BYTE *)(v3 + 7) <= 0x62u )
      goto LABEL_35;
    srcd = *(_BYTE *)(v3 + 7);
    v107 = v109;
    v115 = &v107;
    v108 = 0;
    LOBYTE(v109[0]) = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v110 = &unk_49EFBE0;
    v70 = sub_1263B40((__int64)&v110, "Linked container's ");
    v71 = sub_1263B40(v70, "minor NvvmIR version (");
    v72 = sub_16E7A90(v71, srcd);
    v73 = sub_1263B40(v72, ") newer than tool ");
    v74 = sub_1263B40(v73, "(should be ");
    v75 = sub_16E7AB0(v74, 98);
    sub_1263B40(v75, ")\n");
  }
  else
  {
    LOBYTE(v109[0]) = 0;
    v107 = v109;
    v115 = &v107;
    v108 = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v110 = &unk_49EFBE0;
    v15 = sub_16E7EE0((__int64)&v110, "Linked container's ", 0x13u);
    v16 = srca;
    v17 = *(__m128i **)(v15 + 24);
    v18 = v15;
    if ( *(_QWORD *)(v15 + 16) - (_QWORD)v17 <= 0x15u )
    {
      v88 = sub_16E7EE0(v15, "NvvmIR major version (", 0x16u);
      v16 = srca;
      v18 = v88;
    }
    else
    {
      v19 = _mm_load_si128((const __m128i *)&xmmword_3F6E120);
      v17[1].m128i_i32[0] = 1852795251;
      v17[1].m128i_i16[2] = 10272;
      *v17 = v19;
      *(_QWORD *)(v15 + 24) += 22LL;
    }
    v20 = sub_16E7A90(v18, v16);
    v21 = *(__m128i **)(v20 + 24);
    v22 = v20;
    if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 0x10u )
    {
      v90 = sub_16E7EE0(v20, ") not compatible ", 0x11u);
      v24 = *(void **)(v90 + 24);
      v22 = v90;
    }
    else
    {
      v23 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v21[1].m128i_i8[0] = 32;
      *v21 = v23;
      v24 = (void *)(*(_QWORD *)(v20 + 24) + 17LL);
      *(_QWORD *)(v20 + 24) = v24;
    }
    if ( *(_QWORD *)(v22 + 16) - (_QWORD)v24 <= 0xAu )
    {
      v22 = sub_16E7EE0(v22, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v24, "(should be ", 11);
      *(_QWORD *)(v22 + 24) += 11LL;
    }
    v25 = sub_16E7AB0(v22, 2);
    v26 = *(_WORD **)(v25 + 24);
    if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 1u )
    {
      sub_16E7EE0(v25, ")\n", 2u);
    }
    else
    {
      *v26 = 2601;
      *(_QWORD *)(v25 + 24) += 2LL;
    }
  }
  if ( v113 != v111 )
    sub_16E7BA0((__int64 *)&v110);
  sub_16E7BC0((__int64 *)&v110);
  if ( v107 != v109 )
    j_j___libc_free_0(v107, v109[0] + 1LL);
  v14 = 0;
LABEL_35:
  srcb = *(_BYTE *)(v3 + 8);
  if ( srcb != 3 )
  {
    v108 = 0;
    v107 = v109;
    v115 = &v107;
    LOBYTE(v109[0]) = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v110 = &unk_49EFBE0;
    v27 = sub_16E7EE0((__int64)&v110, "Linked container's ", 0x13u);
    v28 = srcb;
    v29 = *(__m128i **)(v27 + 24);
    v30 = v27;
    if ( *(_QWORD *)(v27 + 16) - (_QWORD)v29 <= 0x18u )
    {
      v91 = sub_16E7EE0(v27, "NvvmDebug major version (", 0x19u);
      v28 = srcb;
      v30 = v91;
    }
    else
    {
      v31 = _mm_load_si128((const __m128i *)&xmmword_3F6E130);
      v29[1].m128i_i8[8] = 40;
      v29[1].m128i_i64[0] = 0x206E6F6973726576LL;
      *v29 = v31;
      *(_QWORD *)(v27 + 24) += 25LL;
    }
    v32 = sub_16E7A90(v30, v28);
    v33 = *(__m128i **)(v32 + 24);
    v34 = v32;
    if ( *(_QWORD *)(v32 + 16) - (_QWORD)v33 <= 0x10u )
    {
      v93 = sub_16E7EE0(v32, ") not compatible ", 0x11u);
      v36 = *(void **)(v93 + 24);
      v34 = v93;
    }
    else
    {
      v35 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v33[1].m128i_i8[0] = 32;
      *v33 = v35;
      v36 = (void *)(*(_QWORD *)(v32 + 24) + 17LL);
      *(_QWORD *)(v32 + 24) = v36;
    }
    if ( *(_QWORD *)(v34 + 16) - (_QWORD)v36 <= 0xAu )
    {
      v34 = sub_16E7EE0(v34, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v36, "(should be ", 11);
      *(_QWORD *)(v34 + 24) += 11LL;
    }
    v37 = sub_16E7AB0(v34, 3);
    v38 = *(_WORD **)(v37 + 24);
    if ( *(_QWORD *)(v37 + 16) - (_QWORD)v38 <= 1u )
    {
      sub_16E7EE0(v37, ")\n", 2u);
    }
    else
    {
      *v38 = 2601;
      *(_QWORD *)(v37 + 24) += 2LL;
    }
LABEL_44:
    if ( v113 != v111 )
      sub_16E7BA0((__int64 *)&v110);
    sub_16E7BC0((__int64 *)&v110);
    if ( v107 != v109 )
      j_j___libc_free_0(v107, v109[0] + 1LL);
    v39 = *(unsigned __int8 *)(v3 + 11) + 100 * *(unsigned __int8 *)(v3 + 10);
    if ( v39 <= 0x2BC )
      goto LABEL_7;
    goto LABEL_49;
  }
  if ( *(_BYTE *)(v3 + 9) > 2u )
  {
    srce = *(_BYTE *)(v3 + 9);
    v107 = v109;
    v115 = &v107;
    v108 = 0;
    LOBYTE(v109[0]) = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v110 = &unk_49EFBE0;
    v76 = sub_1263B40((__int64)&v110, "Linked container's ");
    v77 = sub_1263B40(v76, "minor NvvmDebug version (");
    v78 = sub_16E7A90(v77, srce);
    v79 = sub_1263B40(v78, ") newer than tool ");
    v80 = sub_1263B40(v79, "(should be ");
    v81 = sub_16E7AB0(v80, 2);
    sub_1263B40(v81, ")\n");
    goto LABEL_44;
  }
  v39 = *(unsigned __int8 *)(v3 + 11) + 100 * *(unsigned __int8 *)(v3 + 10);
  if ( v39 > 0x2BC )
  {
LABEL_49:
    srcc = v39;
    v107 = v109;
    v108 = 0;
    v110 = &unk_49EFBE0;
    LOBYTE(v109[0]) = 0;
    v114 = 1;
    v113 = 0;
    v112 = 0;
    v111 = 0;
    v115 = &v107;
    v40 = sub_16E7EE0((__int64)&v110, "Linked container's ", 0x13u);
    v41 = srcc;
    v42 = *(void **)(v40 + 24);
    v43 = v40;
    if ( *(_QWORD *)(v40 + 16) - (_QWORD)v42 <= 0xDu )
    {
      v94 = sub_16E7EE0(v40, "LLVM version (", 0xEu);
      v41 = srcc;
      v43 = v94;
    }
    else
    {
      qmemcpy(v42, "LLVM version (", 14);
      *(_QWORD *)(v40 + 24) += 14LL;
    }
    v44 = sub_16E7A90(v43, v41);
    v45 = *(__m128i **)(v44 + 24);
    v46 = v44;
    if ( *(_QWORD *)(v44 + 16) - (_QWORD)v45 <= 0x10u )
    {
      v95 = sub_16E7EE0(v44, ") not compatible ", 0x11u);
      v48 = *(void **)(v95 + 24);
      v46 = v95;
    }
    else
    {
      v47 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v45[1].m128i_i8[0] = 32;
      *v45 = v47;
      v48 = (void *)(*(_QWORD *)(v44 + 24) + 17LL);
      *(_QWORD *)(v44 + 24) = v48;
    }
    if ( *(_QWORD *)(v46 + 16) - (_QWORD)v48 <= 0xAu )
    {
      v46 = sub_16E7EE0(v46, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v48, "(should be ", 11);
      *(_QWORD *)(v46 + 24) += 11LL;
    }
    v49 = sub_16E7A90(v46, 700);
    v50 = *(_WORD **)(v49 + 24);
    if ( *(_QWORD *)(v49 + 16) - (_QWORD)v50 <= 1u )
    {
      sub_16E7EE0(v49, ")\n", 2u);
    }
    else
    {
      *v50 = 2601;
      *(_QWORD *)(v49 + 24) += 2LL;
    }
    if ( v113 != v111 )
      sub_16E7BA0((__int64 *)&v110);
    sub_16E7BC0((__int64 *)&v110);
    if ( v107 != v109 )
      j_j___libc_free_0(v107, v109[0] + 1LL);
    goto LABEL_7;
  }
  if ( v14 )
  {
    v63 = 0;
    v64 = sub_1C17CB0(a2, (__int64 *)v6);
    v65 = *(const void **)(a2 + 16);
    v66 = *(_QWORD *)(a2 + 8);
    v67 = v64;
    v68 = *(unsigned int *)(v3 + 20);
    v69 = *(_QWORD *)(a2 + 16) - v66;
    if ( v68 <= v69 )
    {
      v65 = (const void *)(v66 + v68);
      v63 = v69 - v68;
    }
    if ( v67 && *(_DWORD *)(v67 + 228) )
    {
      v106 = v67;
      v96 = (_BYTE *)sub_2207820(v63);
      memcpy(v96, v65, v63);
      v98 = sub_16886D0(*(unsigned int *)(v106 + 228), (__int64)v65, v97);
      sub_16887A0((__int64)v98, v96, v63);
      sub_1688720(v98);
      LOWORD(v112) = 257;
      sub_16C28C0(&v107, v96, v63, (__int64)&v110);
      j_j___libc_free_0_0(v96);
      *a1 = v107;
    }
    else
    {
      sub_16C2450(&v110, (__int64)v65, v63, (__int64)byte_3F871B3, 0);
      *a1 = v110;
    }
    goto LABEL_8;
  }
LABEL_7:
  *a1 = 0;
LABEL_8:
  if ( v6 )
  {
    v7 = *(unsigned __int64 **)(v6 + 16);
    v8 = &v7[*(unsigned int *)(v6 + 24)];
    while ( v8 != v7 )
    {
      v9 = *v7++;
      _libc_free(v9);
    }
    v10 = *(unsigned __int64 **)(v6 + 64);
    v11 = (unsigned __int64)&v10[2 * *(unsigned int *)(v6 + 72)];
    if ( v10 != (unsigned __int64 *)v11 )
    {
      do
      {
        v12 = *v10;
        v10 += 2;
        _libc_free(v12);
      }
      while ( v10 != (unsigned __int64 *)v11 );
      v11 = *(_QWORD *)(v6 + 64);
    }
    if ( v11 != v6 + 80 )
      _libc_free(v11);
    v13 = *(_QWORD *)(v6 + 16);
    if ( v13 != v6 + 32 )
      _libc_free(v13);
    j_j___libc_free_0(v6, 104);
  }
  return a1;
}
