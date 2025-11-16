// Function: sub_22E9250
// Address: 0x22e9250
//
void __fastcall sub_22E9250(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, char a5)
{
  char *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // rax
  void *v13; // rcx
  unsigned __int64 *v14; // rdx
  size_t v15; // rsi
  __int64 v16; // r12
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r10
  __int64 v21; // r9
  _QWORD *v22; // rbx
  _QWORD *v23; // r12
  void *v24; // rdi
  size_t v25; // r11
  unsigned __int64 v26; // rcx
  int v27; // eax
  unsigned __int64 v28; // r14
  __int64 *v29; // rax
  __int64 *v30; // rbx
  _BYTE *v31; // rsi
  size_t v32; // rdx
  __int64 v33; // rsi
  char v34; // al
  unsigned __int64 v35; // rdx
  __int64 *v36; // r13
  unsigned __int64 v37; // r12
  __int64 *v38; // rax
  _BYTE *p_src; // rdi
  __int64 v40; // rsi
  _QWORD *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rax
  _DWORD *v45; // rdx
  char *v46; // rsi
  __m128i *v47; // r15
  const char *v48; // rax
  __int64 v49; // rdx
  char v50; // al
  _QWORD *v51; // rdx
  char v52; // al
  char v53; // dl
  _QWORD *v54; // rcx
  size_t v55; // r15
  _QWORD *v56; // rsi
  unsigned __int64 v57; // rdi
  _QWORD *v58; // rcx
  unsigned __int64 v59; // rdx
  __int64 *v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // rdi
  _BYTE *v63; // rax
  _QWORD *v64; // rax
  __m128i *v65; // rdx
  __m128i si128; // xmm0
  size_t v67; // rdx
  __m128i v68; // xmm5
  __int64 v69; // [rsp+8h] [rbp-248h]
  __int64 v70; // [rsp+10h] [rbp-240h]
  __int64 v71; // [rsp+18h] [rbp-238h]
  size_t v74; // [rsp+38h] [rbp-218h]
  __int64 v75; // [rsp+48h] [rbp-208h]
  size_t v76; // [rsp+50h] [rbp-200h]
  __int64 v77; // [rsp+58h] [rbp-1F8h]
  unsigned __int64 v78; // [rsp+60h] [rbp-1F0h]
  __int64 v79; // [rsp+68h] [rbp-1E8h] BYREF
  int v80; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 (__fastcall **v81)(); // [rsp+78h] [rbp-1D8h]
  void *s1; // [rsp+80h] [rbp-1D0h] BYREF
  size_t n; // [rsp+88h] [rbp-1C8h]
  __m128i v84; // [rsp+90h] [rbp-1C0h] BYREF
  _QWORD *v85; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-1A8h]
  _BYTE v87[16]; // [rsp+B0h] [rbp-1A0h] BYREF
  _BYTE *v88; // [rsp+C0h] [rbp-190h]
  __int64 v89; // [rsp+C8h] [rbp-188h]
  _QWORD v90[2]; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v91[2]; // [rsp+E0h] [rbp-170h] BYREF
  _QWORD v92[2]; // [rsp+F0h] [rbp-160h] BYREF
  const char *v93; // [rsp+100h] [rbp-150h] BYREF
  __int64 v94; // [rsp+108h] [rbp-148h]
  __int16 v95; // [rsp+120h] [rbp-130h]
  __m128i v96; // [rsp+130h] [rbp-120h] BYREF
  __m128i v97; // [rsp+140h] [rbp-110h] BYREF
  __int64 v98; // [rsp+150h] [rbp-100h]
  _QWORD v99[4]; // [rsp+160h] [rbp-F0h] BYREF
  char v100; // [rsp+180h] [rbp-D0h]
  char v101; // [rsp+181h] [rbp-CFh]
  size_t v102[2]; // [rsp+190h] [rbp-C0h] BYREF
  __m128i src; // [rsp+1A0h] [rbp-B0h] BYREF
  __int64 v104; // [rsp+1B0h] [rbp-A0h]
  _BYTE *v105; // [rsp+1C0h] [rbp-90h] BYREF
  size_t v106; // [rsp+1C8h] [rbp-88h]
  _OWORD v107[8]; // [rsp+1D0h] [rbp-80h] BYREF

  v79 = a2;
  v7 = (char *)sub_BD5D20(a1);
  if ( v7 )
  {
    v96.m128i_i64[0] = (__int64)&v97;
    sub_22E4AB0(v96.m128i_i64, v7, (__int64)&v7[v8]);
    if ( a3 )
      goto LABEL_3;
LABEL_101:
    v87[0] = 0;
    v85 = v87;
    v86 = 0;
    goto LABEL_4;
  }
  v97.m128i_i8[0] = 0;
  v96 = (__m128i)(unsigned __int64)&v97;
  if ( !a3 )
    goto LABEL_101;
LABEL_3:
  v85 = v87;
  sub_22E4AB0((__int64 *)&v85, a3, (__int64)&a3[a4]);
  if ( v86 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_121;
LABEL_4:
  v9 = sub_2241490((unsigned __int64 *)&v85, ".", 1u);
  v105 = v107;
  if ( (unsigned __int64 *)*v9 == v9 + 2 )
  {
    v107[0] = _mm_loadu_si128((const __m128i *)v9 + 1);
  }
  else
  {
    v105 = (_BYTE *)*v9;
    *(_QWORD *)&v107[0] = v9[2];
  }
  v106 = v9[1];
  *v9 = (unsigned __int64)(v9 + 2);
  v9[1] = 0;
  *((_BYTE *)v9 + 16) = 0;
  v10 = 15;
  v11 = 15;
  if ( v105 != (_BYTE *)v107 )
    v11 = *(_QWORD *)&v107[0];
  if ( v106 + v96.m128i_i64[1] <= v11 )
    goto LABEL_12;
  if ( (__m128i *)v96.m128i_i64[0] != &v97 )
    v10 = v97.m128i_i64[0];
  if ( v106 + v96.m128i_i64[1] <= v10 )
  {
    v12 = sub_2241130((unsigned __int64 *)&v96, 0, 0, v105, v106);
    s1 = &v84;
    v13 = (void *)*v12;
    v14 = v12 + 2;
    if ( (unsigned __int64 *)*v12 != v12 + 2 )
      goto LABEL_13;
  }
  else
  {
LABEL_12:
    v12 = sub_2241490((unsigned __int64 *)&v105, (char *)v96.m128i_i64[0], v96.m128i_u64[1]);
    s1 = &v84;
    v13 = (void *)*v12;
    v14 = v12 + 2;
    if ( (unsigned __int64 *)*v12 != v12 + 2 )
    {
LABEL_13:
      s1 = v13;
      v84.m128i_i64[0] = v12[2];
      goto LABEL_14;
    }
  }
  v84 = _mm_loadu_si128((const __m128i *)v12 + 1);
LABEL_14:
  n = v12[1];
  *v12 = (unsigned __int64)v14;
  v12[1] = 0;
  *((_BYTE *)v12 + 16) = 0;
  if ( v105 != (_BYTE *)v107 )
    j_j___libc_free_0((unsigned __int64)v105);
  if ( v85 != (_QWORD *)v87 )
    j_j___libc_free_0((unsigned __int64)v85);
  if ( (__m128i *)v96.m128i_i64[0] != &v97 )
    j_j___libc_free_0(v96.m128i_u64[0]);
  v76 = n;
  if ( n > 0xFA )
  {
    sub_22410F0((unsigned __int64 *)&s1, 0xFAu, 0);
    v76 = n;
  }
  if ( !v76 )
    goto LABEL_43;
  v15 = v76;
  v16 = 0;
  while ( 1 )
  {
    v17 = sub_22076E0((__int64 *)s1, v15, 3339675911LL);
    v18 = qword_4FDC168;
    v19 = v17;
    v20 = v17 % qword_4FDC168;
    v75 = v17 % qword_4FDC168;
    v21 = *(_QWORD *)(qword_4FDC160 + v75 * 8);
    if ( !v21 )
      break;
    v22 = *(_QWORD **)v21;
    v77 = v16;
    v23 = *(_QWORD **)(qword_4FDC160 + 8 * (v17 % qword_4FDC168));
    v24 = s1;
    v25 = n;
    v26 = *(_QWORD *)(*(_QWORD *)v21 + 40LL);
    while ( 1 )
    {
      if ( v19 == v26 && v25 == v22[2] )
      {
        v78 = v20;
        if ( !v25 )
          break;
        v74 = v25;
        v27 = memcmp(v24, (const void *)v22[1], v25);
        v25 = v74;
        v20 = v78;
        if ( !v27 )
          break;
      }
      if ( !*v22 )
        goto LABEL_36;
      v26 = *(_QWORD *)(*v22 + 40LL);
      v23 = v22;
      if ( v20 != v26 % v18 )
        goto LABEL_36;
      v22 = (_QWORD *)*v22;
    }
    if ( !*v23 )
      break;
    v16 = v77 + 1;
    sub_22410F0((unsigned __int64 *)&s1, (unsigned __int8)(-7 - v77), 0);
    if ( v76 == v77 + 1 )
      goto LABEL_42;
    v15 = n;
  }
LABEL_36:
  v28 = v19;
  v29 = (__int64 *)sub_22077B0(0x30u);
  v30 = v29;
  if ( v29 )
    *v29 = 0;
  v31 = s1;
  v32 = n;
  v29[1] = (__int64)(v29 + 3);
  sub_22E4B60(v29 + 1, v31, (__int64)&v31[v32]);
  v33 = qword_4FDC168;
  v34 = sub_222DA10((__int64)&dword_4FDC180, qword_4FDC168, qword_4FDC178, 1);
  v36 = (__int64 *)qword_4FDC160;
  v37 = v35;
  if ( v34 )
  {
    if ( v35 == 1 )
    {
      qword_4FDC190 = 0;
      v36 = &qword_4FDC190;
    }
    else
    {
      if ( v35 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(&dword_4FDC180, v33, v35);
      v55 = 8 * v35;
      v36 = (__int64 *)sub_22077B0(8 * v35);
      memset(v36, 0, v55);
    }
    v56 = (_QWORD *)qword_4FDC170;
    qword_4FDC170 = 0;
    if ( !v56 )
    {
LABEL_76:
      if ( (__int64 *)qword_4FDC160 != &qword_4FDC190 )
        j_j___libc_free_0(qword_4FDC160);
      qword_4FDC168 = v37;
      qword_4FDC160 = (__int64)v36;
      v75 = v28 % v37;
      goto LABEL_39;
    }
    v57 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v58 = v56;
        v56 = (_QWORD *)*v56;
        v59 = v58[5] % v37;
        v60 = &v36[v59];
        if ( !*v60 )
          break;
        *v58 = *(_QWORD *)*v60;
        *(_QWORD *)*v60 = v58;
LABEL_72:
        if ( !v56 )
          goto LABEL_76;
      }
      *v58 = qword_4FDC170;
      qword_4FDC170 = (__int64)v58;
      *v60 = (__int64)&qword_4FDC170;
      if ( !*v58 )
      {
        v57 = v59;
        goto LABEL_72;
      }
      v36[v57] = (__int64)v58;
      v57 = v59;
      if ( !v56 )
        goto LABEL_76;
    }
  }
LABEL_39:
  v30[5] = v28;
  v38 = &v36[v75];
  if ( v36[v75] )
  {
    *v30 = *(_QWORD *)*v38;
    *(_QWORD *)*v38 = v30;
  }
  else
  {
    v61 = qword_4FDC170;
    qword_4FDC170 = (__int64)v30;
    *v30 = v61;
    if ( v61 )
    {
      v36[*(_QWORD *)(v61 + 40) % (unsigned __int64)qword_4FDC168] = (__int64)v30;
      v38 = (__int64 *)(qword_4FDC160 + v75 * 8);
    }
    *v38 = (__int64)&qword_4FDC170;
  }
  ++qword_4FDC178;
LABEL_42:
  v76 = n;
LABEL_43:
  v102[0] = (size_t)&src;
  sub_22E4B60((__int64 *)v102, s1, (__int64)s1 + v76);
  if ( 0x3FFFFFFFFFFFFFFFLL - v102[1] <= 3 )
    goto LABEL_121;
  sub_2241490(v102, ".dot", 4u);
  p_src = s1;
  if ( (__m128i *)v102[0] == &src )
  {
    v67 = v102[1];
    if ( v102[1] )
    {
      if ( v102[1] == 1 )
        *(_BYTE *)s1 = src.m128i_i8[0];
      else
        memcpy(s1, &src, v102[1]);
      v67 = v102[1];
      p_src = s1;
    }
    n = v67;
    p_src[v67] = 0;
    p_src = (_BYTE *)v102[0];
  }
  else
  {
    if ( s1 == &v84 )
    {
      s1 = (void *)v102[0];
      n = v102[1];
      v84.m128i_i64[0] = src.m128i_i64[0];
    }
    else
    {
      v40 = v84.m128i_i64[0];
      s1 = (void *)v102[0];
      n = v102[1];
      v84.m128i_i64[0] = src.m128i_i64[0];
      if ( p_src )
      {
        v102[0] = (size_t)p_src;
        src.m128i_i64[0] = v40;
        goto LABEL_48;
      }
    }
    v102[0] = (size_t)&src;
    p_src = &src;
  }
LABEL_48:
  v102[1] = 0;
  *p_src = 0;
  if ( (__m128i *)v102[0] != &src )
    j_j___libc_free_0(v102[0]);
  v80 = 0;
  v81 = sub_2241E40();
  v41 = sub_CB72A0();
  v42 = v41[4];
  v43 = (__int64)v41;
  if ( (unsigned __int64)(v41[3] - v42) <= 8 )
  {
    v43 = sub_CB6200((__int64)v41, "Writing '", 9u);
  }
  else
  {
    *(_BYTE *)(v42 + 8) = 39;
    *(_QWORD *)v42 = 0x20676E6974697257LL;
    v41[4] += 9LL;
  }
  v44 = sub_CB6200(v43, (unsigned __int8 *)s1, n);
  v45 = *(_DWORD **)(v44 + 32);
  if ( *(_QWORD *)(v44 + 24) - (_QWORD)v45 <= 3u )
  {
    sub_CB6200(v44, "'...", 4u);
  }
  else
  {
    *v45 = 774778407;
    *(_QWORD *)(v44 + 32) += 4LL;
  }
  v46 = (char *)s1;
  sub_CB7060((__int64)&v105, s1, n, (__int64)&v80, 3u);
  v88 = v90;
  strcpy((char *)v90, "Region Graph");
  v89 = 12;
  if ( v80 )
  {
    v64 = sub_CB72A0();
    v65 = (__m128i *)v64[4];
    if ( v64[3] - (_QWORD)v65 <= 0x20u )
    {
      v46 = "  error opening file for writing!";
      sub_CB6200((__int64)v64, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v65[2].m128i_i8[0] = 33;
      *v65 = si128;
      v65[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v64[4] += 33LL;
    }
    goto LABEL_91;
  }
  v101 = 1;
  v47 = (__m128i *)v91;
  v99[0] = "' function";
  v100 = 3;
  v48 = sub_BD5D20(a1);
  v91[0] = (__int64)v92;
  v95 = 261;
  v94 = v49;
  v93 = v48;
  sub_22E4B60(v91, v88, (__int64)&v88[v89]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v91[1]) <= 5 )
LABEL_121:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v91, " for '", 6u);
  v50 = v95;
  if ( !(_BYTE)v95 )
  {
    LOWORD(v98) = 256;
    goto LABEL_88;
  }
  if ( (_BYTE)v95 == 1 )
  {
    v52 = v100;
    v96.m128i_i64[0] = (__int64)v91;
    LOWORD(v98) = 260;
    if ( v100 )
    {
      if ( v100 != 1 )
      {
        v53 = 4;
        v70 = v96.m128i_i64[1];
        goto LABEL_63;
      }
LABEL_114:
      v68 = _mm_load_si128(&v97);
      *(__m128i *)v102 = _mm_load_si128(&v96);
      v104 = v98;
      src = v68;
      goto LABEL_89;
    }
LABEL_88:
    LOWORD(v104) = 256;
    goto LABEL_89;
  }
  if ( HIBYTE(v95) == 1 )
  {
    v51 = v93;
    v71 = v94;
  }
  else
  {
    v51 = &v93;
    v50 = 2;
  }
  BYTE1(v98) = v50;
  v52 = v100;
  v96.m128i_i64[0] = (__int64)v91;
  v97.m128i_i64[0] = (__int64)v51;
  v97.m128i_i64[1] = v71;
  LOBYTE(v98) = 4;
  if ( !v100 )
    goto LABEL_88;
  if ( v100 == 1 )
    goto LABEL_114;
  v47 = &v96;
  v53 = 2;
LABEL_63:
  if ( v101 == 1 )
  {
    v54 = (_QWORD *)v99[0];
    v69 = v99[1];
  }
  else
  {
    v54 = v99;
    v52 = 2;
  }
  v102[0] = (size_t)v47;
  src.m128i_i64[0] = (__int64)v54;
  v102[1] = v70;
  LOBYTE(v104) = v53;
  src.m128i_i64[1] = v69;
  BYTE1(v104) = v52;
LABEL_89:
  v46 = (char *)&v79;
  sub_22E9130((__int64)&v105, &v79, a5, (void **)v102);
  if ( (_QWORD *)v91[0] != v92 )
  {
    v46 = (char *)(v92[0] + 1LL);
    j_j___libc_free_0(v91[0]);
  }
LABEL_91:
  v62 = sub_CB72A0();
  v63 = (_BYTE *)v62[4];
  if ( (_BYTE *)v62[3] == v63 )
  {
    v46 = "\n";
    sub_CB6200((__int64)v62, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v63 = 10;
    ++v62[4];
  }
  if ( v88 != (_BYTE *)v90 )
  {
    v46 = (char *)(v90[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v88);
  }
  sub_CB5B00((int *)&v105, (__int64)v46);
  if ( s1 != &v84 )
    j_j___libc_free_0((unsigned __int64)s1);
}
