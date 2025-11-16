// Function: sub_2A2B740
// Address: 0x2a2b740
//
__int64 __fastcall sub_2A2B740(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rsi
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  const __m128i *v10; // rax
  char *v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  const __m128i *v19; // rsi
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  const __m128i *v23; // rdi
  unsigned __int64 v24; // rdx
  signed __int64 v25; // r14
  __int64 v26; // rax
  unsigned __int64 v27; // rsi
  __m128i *v28; // rdx
  const __m128i *v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rcx
  signed __int64 v33; // r14
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  __m128i *v36; // rdx
  const __m128i *v37; // rax
  unsigned __int64 v38; // r14
  unsigned __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r15
  __int64 *v42; // rax
  __int64 *v43; // rdx
  __int64 v44; // r13
  _QWORD *v45; // rax
  char v46; // dl
  unsigned __int64 v47; // rdx
  __int64 *v48; // rdi
  __int64 *v49; // r12
  unsigned int v50; // r15d
  __int64 v51; // rbx
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  __int64 v54; // r14
  unsigned int v55; // r13d
  __int64 v56; // rsi
  _QWORD *v57; // rax
  _QWORD *v58; // rdx
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  __int64 v62; // rax
  __int64 *v63; // r14
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rax
  _QWORD *v71; // r14
  _QWORD *v72; // rbx
  __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rax
  _QWORD *v78; // r14
  _QWORD *v79; // rbx
  __int64 v80; // rsi
  __int64 *v81; // [rsp+30h] [rbp-430h]
  int v82; // [rsp+50h] [rbp-410h]
  __int64 *v87; // [rsp+88h] [rbp-3D8h]
  __int64 *v88; // [rsp+88h] [rbp-3D8h]
  unsigned __int8 v89; // [rsp+98h] [rbp-3C8h]
  __int64 *v90; // [rsp+A0h] [rbp-3C0h] BYREF
  __int64 v91; // [rsp+A8h] [rbp-3B8h]
  _BYTE v92[64]; // [rsp+B0h] [rbp-3B0h] BYREF
  unsigned __int64 v93[16]; // [rsp+F0h] [rbp-370h] BYREF
  __m128i v94; // [rsp+170h] [rbp-2F0h] BYREF
  __int64 v95; // [rsp+180h] [rbp-2E0h]
  int v96; // [rsp+188h] [rbp-2D8h]
  char v97; // [rsp+18Ch] [rbp-2D4h]
  _QWORD v98[8]; // [rsp+190h] [rbp-2D0h] BYREF
  unsigned __int64 v99; // [rsp+1D0h] [rbp-290h] BYREF
  unsigned __int64 v100; // [rsp+1D8h] [rbp-288h]
  unsigned __int64 v101; // [rsp+1E0h] [rbp-280h]
  char *v102; // [rsp+1F0h] [rbp-270h] BYREF
  unsigned __int64 v103; // [rsp+1F8h] [rbp-268h] BYREF
  __int64 v104; // [rsp+200h] [rbp-260h]
  __int64 v105; // [rsp+208h] [rbp-258h]
  _QWORD v106[8]; // [rsp+210h] [rbp-250h] BYREF
  const __m128i *v107; // [rsp+250h] [rbp-210h] BYREF
  unsigned __int64 v108; // [rsp+258h] [rbp-208h]
  unsigned __int64 v109; // [rsp+260h] [rbp-200h]
  __int64 *v110; // [rsp+270h] [rbp-1F0h] BYREF
  unsigned __int64 v111; // [rsp+278h] [rbp-1E8h] BYREF
  __int64 v112; // [rsp+280h] [rbp-1E0h] BYREF
  __int64 v113; // [rsp+288h] [rbp-1D8h]
  _QWORD v114[8]; // [rsp+290h] [rbp-1D0h] BYREF
  unsigned __int64 v115; // [rsp+2D0h] [rbp-190h]
  unsigned __int64 v116; // [rsp+2D8h] [rbp-188h]
  unsigned __int64 v117; // [rsp+2E0h] [rbp-180h]
  __m128i v118; // [rsp+2F0h] [rbp-170h] BYREF
  char v119; // [rsp+300h] [rbp-160h]
  _QWORD *v120; // [rsp+308h] [rbp-158h]
  char v121[8]; // [rsp+310h] [rbp-150h] BYREF
  unsigned int v122; // [rsp+318h] [rbp-148h]
  _QWORD *v123; // [rsp+328h] [rbp-138h]
  unsigned int v124; // [rsp+338h] [rbp-128h]
  char v125; // [rsp+340h] [rbp-120h]
  const __m128i *v126; // [rsp+350h] [rbp-110h]
  char *v127; // [rsp+358h] [rbp-108h]
  unsigned __int64 v128; // [rsp+360h] [rbp-100h] BYREF
  char v129[8]; // [rsp+368h] [rbp-F8h] BYREF
  unsigned __int64 v130; // [rsp+370h] [rbp-F0h]
  char v131; // [rsp+384h] [rbp-DCh]
  char v132[40]; // [rsp+388h] [rbp-D8h] BYREF
  __int64 v133; // [rsp+3B0h] [rbp-B0h]
  unsigned int v134; // [rsp+3C0h] [rbp-A0h]
  const __m128i *v135; // [rsp+3C8h] [rbp-98h]
  const __m128i *v136; // [rsp+3D0h] [rbp-90h]
  unsigned __int64 v137; // [rsp+3D8h] [rbp-88h]
  unsigned int v138; // [rsp+3E0h] [rbp-80h]
  __int64 v139; // [rsp+3F0h] [rbp-70h]
  unsigned int v140; // [rsp+400h] [rbp-60h]

  v90 = (__int64 *)v92;
  v4 = *(__int64 **)(a1 + 40);
  v91 = 0x800000000LL;
  v81 = v4;
  if ( *(__int64 **)(a1 + 32) == v4 )
    return 0;
  v87 = *(__int64 **)(a1 + 32);
  do
  {
    v5 = *v87;
    v99 = 0;
    memset(v93, 0, 0x78u);
    v93[1] = (unsigned __int64)&v93[4];
    v95 = 0x100000008LL;
    v98[0] = v5;
    v118.m128i_i64[0] = v5;
    LODWORD(v93[2]) = 8;
    BYTE4(v93[3]) = 1;
    v94.m128i_i64[1] = (__int64)v98;
    v100 = 0;
    v101 = 0;
    v96 = 0;
    v97 = 1;
    v94.m128i_i64[0] = 1;
    v119 = 0;
    sub_2A2B700((__int64)&v99, &v118);
    sub_C8CF70((__int64)&v110, v114, 8, (__int64)&v93[4], (__int64)v93);
    v6 = v93[12];
    memset(&v93[12], 0, 24);
    v115 = v6;
    v116 = v93[13];
    v117 = v93[14];
    sub_C8CF70((__int64)&v102, v106, 8, (__int64)v98, (__int64)&v94);
    v7 = v99;
    v99 = 0;
    v107 = (const __m128i *)v7;
    v8 = v100;
    v100 = 0;
    v108 = v8;
    v9 = v101;
    v101 = 0;
    v109 = v9;
    sub_C8CF70((__int64)&v118, v121, 8, (__int64)v106, (__int64)&v102);
    v10 = v107;
    v107 = 0;
    v126 = v10;
    v11 = (char *)v108;
    v108 = 0;
    v127 = v11;
    v12 = v109;
    v109 = 0;
    v128 = v12;
    sub_C8CF70((__int64)v129, v132, 8, (__int64)v114, (__int64)&v110);
    v16 = v115;
    v115 = 0;
    v135 = (const __m128i *)v16;
    v17 = v116;
    v116 = 0;
    v136 = (const __m128i *)v17;
    v18 = v117;
    v117 = 0;
    v137 = v18;
    if ( v107 )
      j_j___libc_free_0((unsigned __int64)v107);
    if ( !BYTE4(v105) )
      _libc_free(v103);
    if ( v115 )
      j_j___libc_free_0(v115);
    if ( !BYTE4(v113) )
      _libc_free(v111);
    if ( v99 )
      j_j___libc_free_0(v99);
    if ( !v97 )
      _libc_free(v94.m128i_u64[1]);
    if ( v93[12] )
      j_j___libc_free_0(v93[12]);
    if ( !BYTE4(v93[3]) )
      _libc_free(v93[1]);
    v19 = (const __m128i *)v106;
    sub_C8CD80((__int64)&v102, (__int64)v106, (__int64)&v118, v13, v14, v15);
    v22 = (__int64)v127;
    v23 = v126;
    v107 = 0;
    v108 = 0;
    v109 = 0;
    v24 = v127 - (char *)v126;
    if ( v127 == (char *)v126 )
    {
      v25 = 0;
      v27 = 0;
    }
    else
    {
      v25 = v127 - (char *)v126;
      if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_149;
      v26 = sub_22077B0(v127 - (char *)v126);
      v22 = (__int64)v127;
      v23 = v126;
      v27 = v26;
    }
    v107 = (const __m128i *)v27;
    v108 = v27;
    v109 = v27 + v25;
    if ( (const __m128i *)v22 != v23 )
    {
      v28 = (__m128i *)v27;
      v29 = v23;
      do
      {
        if ( v28 )
        {
          *v28 = _mm_loadu_si128(v29);
          v20 = v29[1].m128i_i64[0];
          v28[1].m128i_i64[0] = v20;
        }
        v29 = (const __m128i *)((char *)v29 + 24);
        v28 = (__m128i *)((char *)v28 + 24);
      }
      while ( (const __m128i *)v22 != v29 );
      v27 += 8 * ((unsigned __int64)(v22 - 24 - (_QWORD)v23) >> 3) + 24;
    }
    v108 = v27;
    v23 = (const __m128i *)&v110;
    sub_C8CD80((__int64)&v110, (__int64)v114, (__int64)v129, v22, v20, v21);
    v32 = (__int64)v136;
    v19 = v135;
    v115 = 0;
    v116 = 0;
    v117 = 0;
    v24 = (char *)v136 - (char *)v135;
    if ( v136 == v135 )
    {
      v33 = 0;
      v35 = 0;
    }
    else
    {
      v33 = (char *)v136 - (char *)v135;
      if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_149:
        sub_4261EA(v23, v19, v24);
      v34 = sub_22077B0((char *)v136 - (char *)v135);
      v32 = (__int64)v136;
      v19 = v135;
      v35 = v34;
    }
    v115 = v35;
    v117 = v35 + v33;
    v36 = (__m128i *)v35;
    v116 = v35;
    if ( v19 != (const __m128i *)v32 )
    {
      v37 = v19;
      do
      {
        if ( v36 )
        {
          *v36 = _mm_loadu_si128(v37);
          v30 = v37[1].m128i_i64[0];
          v36[1].m128i_i64[0] = v30;
        }
        v37 = (const __m128i *)((char *)v37 + 24);
        v36 = (__m128i *)((char *)v36 + 24);
      }
      while ( (const __m128i *)v32 != v37 );
      v36 = (__m128i *)(v35 + 8 * ((unsigned __int64)(v32 - 24 - (_QWORD)v19) >> 3) + 24);
    }
    v38 = v108;
    v39 = (unsigned __int64)v107;
    v116 = (unsigned __int64)v36;
    v40 = v108 - (_QWORD)v107;
    if ( (__m128i *)(v108 - (_QWORD)v107) != (__m128i *)((char *)v36 - v35) )
    {
LABEL_38:
      v41 = *(_QWORD *)(v38 - 24);
      if ( *(_QWORD *)(v41 + 16) == *(_QWORD *)(v41 + 8) )
      {
        v60 = (unsigned int)v91;
        v40 = HIDWORD(v91);
        v61 = (unsigned int)v91 + 1LL;
        if ( v61 > HIDWORD(v91) )
        {
          sub_C8D5F0((__int64)&v90, v92, v61, 8u, v30, v31);
          v60 = (unsigned int)v91;
        }
        v90[v60] = v41;
        v38 = v108;
        LODWORD(v91) = v91 + 1;
LABEL_48:
        v41 = *(_QWORD *)(v38 - 24);
      }
      if ( !*(_BYTE *)(v38 - 8) )
      {
        v42 = *(__int64 **)(v41 + 8);
        *(_BYTE *)(v38 - 8) = 1;
        *(_QWORD *)(v38 - 16) = v42;
        if ( v42 == *(__int64 **)(v41 + 16) )
          goto LABEL_47;
        goto LABEL_41;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v42 = *(__int64 **)(v38 - 16);
          if ( v42 == *(__int64 **)(v41 + 16) )
          {
LABEL_47:
            v108 -= 24LL;
            v39 = (unsigned __int64)v107;
            v38 = v108;
            if ( (const __m128i *)v108 != v107 )
              goto LABEL_48;
LABEL_51:
            v35 = v115;
            v40 = v38 - v39;
            if ( v38 - v39 == v116 - v115 )
              goto LABEL_52;
            goto LABEL_38;
          }
LABEL_41:
          v43 = v42 + 1;
          *(_QWORD *)(v38 - 16) = v42 + 1;
          v44 = *v42;
          if ( BYTE4(v105) )
            break;
LABEL_49:
          sub_C8CC70((__int64)&v102, v44, (__int64)v43, v40, v30, v31);
          if ( v46 )
            goto LABEL_50;
        }
        v45 = (_QWORD *)v103;
        v40 = HIDWORD(v104);
        v43 = (__int64 *)(v103 + 8LL * HIDWORD(v104));
        if ( (__int64 *)v103 == v43 )
        {
LABEL_96:
          if ( HIDWORD(v104) < (unsigned int)v104 )
          {
            ++HIDWORD(v104);
            *v43 = v44;
            ++v102;
LABEL_50:
            v94.m128i_i64[0] = v44;
            LOBYTE(v95) = 0;
            sub_2A2B700((__int64)&v107, &v94);
            v39 = (unsigned __int64)v107;
            v38 = v108;
            goto LABEL_51;
          }
          goto LABEL_49;
        }
        while ( v44 != *v45 )
        {
          if ( v43 == ++v45 )
            goto LABEL_96;
        }
      }
    }
LABEL_52:
    if ( v39 != v38 )
    {
      v47 = v35;
      while ( *(_QWORD *)v39 == *(_QWORD *)v47 )
      {
        v40 = *(unsigned __int8 *)(v39 + 16);
        if ( (_BYTE)v40 != *(_BYTE *)(v47 + 16) || (_BYTE)v40 && *(_QWORD *)(v39 + 8) != *(_QWORD *)(v47 + 8) )
          break;
        v39 += 24LL;
        v47 += 24LL;
        if ( v39 == v38 )
          goto LABEL_59;
      }
      goto LABEL_38;
    }
LABEL_59:
    if ( v35 )
      j_j___libc_free_0(v35);
    if ( !BYTE4(v113) )
      _libc_free(v111);
    if ( v107 )
      j_j___libc_free_0((unsigned __int64)v107);
    if ( !BYTE4(v105) )
      _libc_free(v103);
    if ( v135 )
      j_j___libc_free_0((unsigned __int64)v135);
    if ( !v131 )
      _libc_free(v130);
    if ( v126 )
      j_j___libc_free_0((unsigned __int64)v126);
    if ( !BYTE4(v120) )
      _libc_free(v118.m128i_u64[1]);
    ++v87;
  }
  while ( v81 != v87 );
  v48 = v90;
  v88 = &v90[(unsigned int)v91];
  if ( v88 == v90 )
  {
    v50 = 0;
    goto LABEL_93;
  }
  v49 = v90;
  v50 = 0;
  while ( 2 )
  {
    while ( 2 )
    {
      v51 = *v49;
      v89 = sub_D4B3D0(*v49);
      if ( !v89 )
        goto LABEL_91;
      v52 = sub_D47930(v51);
      if ( !v52 )
        goto LABEL_91;
      v53 = *(_QWORD *)(v52 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v53 == v52 + 48 )
        goto LABEL_91;
      if ( !v53 )
        BUG();
      v54 = v53 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v53 - 24) - 30 > 0xA )
        goto LABEL_91;
      v82 = sub_B46E30(v53 - 24);
      if ( !v82 )
        goto LABEL_91;
      v55 = 0;
      while ( 1 )
      {
        v56 = sub_B46EC0(v54, v55);
        if ( !*(_BYTE *)(v51 + 84) )
          break;
        v57 = *(_QWORD **)(v51 + 64);
        v58 = &v57[*(unsigned int *)(v51 + 76)];
        if ( v57 == v58 )
          goto LABEL_104;
        while ( v56 != *v57 )
        {
          if ( v58 == ++v57 )
            goto LABEL_104;
        }
LABEL_90:
        if ( ++v55 == v82 )
          goto LABEL_91;
      }
      if ( sub_C8CA60(v51 + 56, v56) )
        goto LABEL_90;
LABEL_104:
      if ( !sub_D46F00(v51)
        || (v62 = sub_D440B0(a2, v51), v63 = (__int64 *)v62, *(_BYTE *)(v62 + 41))
        || !*(_DWORD *)(*(_QWORD *)(v62 + 8) + 304LL)
        && (v76 = sub_D9B120(*(_QWORD *)v62), (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v76 + 8LL))(v76)) )
      {
LABEL_91:
        if ( v88 == ++v49 )
          goto LABEL_92;
        continue;
      }
      break;
    }
    if ( !sub_D48C30(v51, a3, 1) )
      sub_11D2180(v51, a3, a1, a4, v64, v65);
    sub_2A28870((__int64)&v118, v63, *(const void **)(v63[1] + 296), *(unsigned int *)(v63[1] + 304), v51, a1, a3, a4);
    sub_F6D5D0((__int64)&v110, v118.m128i_i64[0], v66, v67, v68, v69);
    sub_2A28FB0(v118.m128i_i64, (__int64)&v110);
    if ( v110 != &v112 )
      _libc_free((unsigned __int64)v110);
    sub_2A2B690((__int64)&v118);
    sub_D37540(a2);
    sub_C7D6A0(v139, 16LL * v140, 8);
    sub_C7D6A0((__int64)v136, 16LL * v138, 8);
    sub_C7D6A0(v133, 16LL * v134, 8);
    if ( v126 != (const __m128i *)&v128 )
      _libc_free((unsigned __int64)v126);
    if ( v125 )
    {
      v77 = v124;
      v125 = 0;
      if ( v124 )
      {
        v78 = v123;
        v79 = &v123[2 * v124];
        do
        {
          if ( *v78 != -8192 && *v78 != -4096 )
          {
            v80 = v78[1];
            if ( v80 )
              sub_B91220((__int64)(v78 + 1), v80);
          }
          v78 += 2;
        }
        while ( v79 != v78 );
        v77 = v124;
      }
      sub_C7D6A0((__int64)v123, 16 * v77, 8);
      v70 = v122;
      if ( v122 )
        goto LABEL_117;
    }
    else
    {
      v70 = v122;
      if ( !v122 )
        goto LABEL_115;
LABEL_117:
      v71 = v120;
      v103 = 2;
      v104 = 0;
      v72 = &v120[8 * (unsigned __int64)(unsigned int)v70];
      v105 = -4096;
      v102 = (char *)&unk_49DD7B0;
      v110 = (__int64 *)&unk_49DD7B0;
      v73 = -4096;
      v106[0] = 0;
      v111 = 2;
      v112 = 0;
      v113 = -8192;
      v114[0] = 0;
      while ( 1 )
      {
        v74 = v71[3];
        if ( v73 != v74 && v74 != v113 )
        {
          v75 = v71[7];
          if ( v75 != 0 && v75 != -4096 && v75 != -8192 )
          {
            sub_BD60C0(v71 + 5);
            v74 = v71[3];
          }
        }
        *v71 = &unk_49DB368;
        if ( v74 != 0 && v74 != -4096 && v74 != -8192 )
          sub_BD60C0(v71 + 1);
        v71 += 8;
        if ( v72 == v71 )
          break;
        v73 = v105;
      }
      v110 = (__int64 *)&unk_49DB368;
      if ( v113 != 0 && v113 != -4096 && v113 != -8192 )
        sub_BD60C0(&v111);
      v102 = (char *)&unk_49DB368;
      if ( v105 != 0 && v105 != -4096 && v105 != -8192 )
        sub_BD60C0(&v103);
      v70 = v122;
    }
LABEL_115:
    ++v49;
    sub_C7D6A0((__int64)v120, v70 << 6, 8);
    v50 = v89;
    if ( v88 != v49 )
      continue;
    break;
  }
LABEL_92:
  v48 = v90;
LABEL_93:
  if ( v48 != (__int64 *)v92 )
    _libc_free((unsigned __int64)v48);
  return v50;
}
