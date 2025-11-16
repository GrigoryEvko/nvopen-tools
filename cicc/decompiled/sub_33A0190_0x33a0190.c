// Function: sub_33A0190
// Address: 0x33a0190
//
void __fastcall sub_33A0190(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  int v3; // r15d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  __int64 (*v10)(); // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdi
  int v14; // edx
  int v15; // eax
  unsigned int v16; // ebx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  __int64 v29; // r13
  __int64 v30; // r12
  unsigned __int64 v31; // r14
  __int64 v32; // r15
  int v33; // eax
  int v34; // edx
  int v35; // r9d
  __int64 v36; // rbx
  int v37; // edx
  _QWORD *v38; // rax
  unsigned int v39; // edx
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // r8
  __int64 v44; // r9
  _BYTE *v45; // rax
  unsigned __int64 v46; // rdx
  _BYTE *v47; // rcx
  __int64 v48; // r12
  _BYTE *i; // rcx
  __int64 v50; // r8
  _OWORD *v51; // rdx
  _OWORD *v52; // rax
  _OWORD *j; // r12
  unsigned int v54; // r11d
  unsigned __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // r14
  int v60; // r13d
  _BYTE *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // r8
  int v64; // eax
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // r12
  bool v68; // al
  __int64 v69; // r9
  __int64 v70; // r8
  unsigned int v71; // r11d
  _OWORD *v72; // rax
  unsigned __int64 v73; // rcx
  __int64 v74; // rcx
  unsigned int *v75; // rax
  __int64 v76; // rax
  int v77; // edx
  __int64 v78; // rax
  int v79; // edx
  __int64 v80; // rdx
  unsigned __int64 v81; // rcx
  unsigned int v82; // edx
  unsigned __int64 v83; // rax
  int v84; // eax
  unsigned int v85; // edx
  char v86; // al
  unsigned __int64 v87; // rdi
  char v88; // al
  char v89; // r10
  unsigned __int64 v90; // rcx
  unsigned __int64 v91; // rdi
  unsigned __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // rbx
  __int64 v98; // r12
  __int64 v99; // r14
  unsigned int v100; // edx
  __int64 v101; // r9
  unsigned int v102; // ecx
  __int64 v103; // rax
  __int64 v104; // rsi
  __int64 (__fastcall *v105)(__int64, __int64); // rax
  unsigned int v106; // edx
  __int64 v107; // rax
  __int64 *v108; // rax
  __int128 v109; // [rsp-10h] [rbp-390h]
  __int128 v110; // [rsp-10h] [rbp-390h]
  __int128 v111; // [rsp-10h] [rbp-390h]
  char v112; // [rsp+4Ah] [rbp-336h]
  bool v113; // [rsp+4Bh] [rbp-335h]
  unsigned int v114; // [rsp+4Ch] [rbp-334h]
  int v115; // [rsp+50h] [rbp-330h]
  char v116; // [rsp+5Fh] [rbp-321h]
  __int64 v117; // [rsp+60h] [rbp-320h]
  __int64 v118; // [rsp+70h] [rbp-310h]
  __int64 v119; // [rsp+80h] [rbp-300h]
  unsigned int v120; // [rsp+80h] [rbp-300h]
  __int64 v121; // [rsp+88h] [rbp-2F8h]
  int v122; // [rsp+90h] [rbp-2F0h]
  __int64 v123; // [rsp+98h] [rbp-2E8h]
  unsigned __int16 v124; // [rsp+A0h] [rbp-2E0h]
  __int64 v125; // [rsp+A0h] [rbp-2E0h]
  int v126; // [rsp+A8h] [rbp-2D8h]
  unsigned int v127; // [rsp+A8h] [rbp-2D8h]
  __int64 v128; // [rsp+B0h] [rbp-2D0h]
  int v129; // [rsp+B0h] [rbp-2D0h]
  __int64 v130; // [rsp+B0h] [rbp-2D0h]
  __int64 v131; // [rsp+B0h] [rbp-2D0h]
  __int64 v132; // [rsp+B8h] [rbp-2C8h]
  unsigned __int16 v133; // [rsp+B8h] [rbp-2C8h]
  __int64 v134; // [rsp+C0h] [rbp-2C0h]
  unsigned int v135; // [rsp+C0h] [rbp-2C0h]
  unsigned __int64 v136; // [rsp+C0h] [rbp-2C0h]
  int v138; // [rsp+130h] [rbp-250h]
  __int64 v139; // [rsp+150h] [rbp-230h] BYREF
  int v140; // [rsp+158h] [rbp-228h]
  __int64 v141; // [rsp+160h] [rbp-220h]
  unsigned __int64 v142; // [rsp+168h] [rbp-218h]
  __m128i v143; // [rsp+170h] [rbp-210h]
  __int128 v144; // [rsp+180h] [rbp-200h] BYREF
  __int64 v145; // [rsp+190h] [rbp-1F0h]
  __m128i v146; // [rsp+1A0h] [rbp-1E0h] BYREF
  __m128i v147; // [rsp+1B0h] [rbp-1D0h] BYREF
  _BYTE *v148; // [rsp+1C0h] [rbp-1C0h] BYREF
  __int64 v149; // [rsp+1C8h] [rbp-1B8h]
  _BYTE v150[64]; // [rsp+1D0h] [rbp-1B0h] BYREF
  unsigned __int64 v151[2]; // [rsp+210h] [rbp-170h] BYREF
  _BYTE v152[64]; // [rsp+220h] [rbp-160h] BYREF
  unsigned __int64 v153[2]; // [rsp+260h] [rbp-120h] BYREF
  _BYTE v154[64]; // [rsp+270h] [rbp-110h] BYREF
  _BYTE *v155; // [rsp+2B0h] [rbp-D0h] BYREF
  __int64 v156; // [rsp+2B8h] [rbp-C8h]
  _BYTE v157[64]; // [rsp+2C0h] [rbp-C0h] BYREF
  _OWORD *v158; // [rsp+300h] [rbp-80h] BYREF
  __int64 v159; // [rsp+308h] [rbp-78h]
  _OWORD v160[7]; // [rsp+310h] [rbp-70h] BYREF

  v113 = sub_B46500((unsigned __int8 *)a2);
  if ( v113 )
  {
    sub_339FBA0(a1, a2, v5, v6, v7, v8);
    return;
  }
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v123 = *(_QWORD *)(a2 - 32);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 2216LL);
  if ( v10 != sub_302E1B0 && ((unsigned __int8 (__fastcall *)(__int64))v10)(v9) )
  {
    v86 = *(_BYTE *)v123;
    if ( *(_BYTE *)v123 == 22 )
    {
      if ( (unsigned __int8)sub_B2D650(v123) )
      {
LABEL_81:
        sub_337E530(a1, a2);
        return;
      }
      v86 = *(_BYTE *)v123;
    }
    if ( v86 == 60 && *(char *)(v123 + 2) < 0 )
      goto LABEL_81;
  }
  v141 = 0;
  v126 = sub_338B750(a1, v123);
  v11 = *(_QWORD *)(a2 + 8);
  v151[0] = (unsigned __int64)v152;
  v134 = v11;
  v149 = 0x400000000LL;
  v151[1] = 0x400000000LL;
  v153[1] = 0x400000000LL;
  v12 = *(_QWORD *)(a1 + 864);
  v153[0] = (unsigned __int64)v154;
  v13 = *(__int64 **)(v12 + 40);
  v122 = v14;
  v148 = v150;
  LOBYTE(v142) = 0;
  v15 = sub_2E79000(v13);
  sub_34B8C80(v9, v15, v134, (unsigned int)&v148, (unsigned int)v151, (unsigned int)v153, __PAIR128__(v142, 0));
  v16 = v149;
  if ( !(_DWORD)v149 )
    goto LABEL_22;
  v124 = *(_WORD *)(a2 + 2);
  sub_B91FC0(v146.m128i_i64, a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 29) && (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    v118 = sub_B91C10(a2, 4);
  else
    v118 = 0;
  v128 = *(_QWORD *)(a1 + 888);
  v112 = *(_WORD *)(a2 + 2) & 1;
  v132 = *(_QWORD *)(a1 + 880);
  v17 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v133 = sub_2FEC4A0(v9, a2, v17, v132, v128);
  if ( v112 )
  {
    v139 = 0;
    v101 = sub_33738B0(a1, a2, v18, v19, v20, v21);
    v102 = v100;
    v2 = v100;
    v103 = *(_QWORD *)a1;
    v135 = v100;
    LODWORD(v121) = v101;
    v140 = *(_DWORD *)(a1 + 848);
    if ( v103 )
    {
      if ( &v139 != (__int64 *)(v103 + 48) )
      {
        v104 = *(_QWORD *)(v103 + 48);
        v139 = v104;
        if ( v104 )
        {
          v120 = v100;
          v131 = v101;
          sub_B96E90((__int64)&v139, v104, 1);
          v101 = v131;
          v102 = v120;
        }
      }
    }
    v105 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 2400LL);
    if ( v105 != sub_302E280 )
    {
      v2 = v102;
      LODWORD(v121) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64 *, _QWORD))v105)(
                        v9,
                        v101,
                        v102,
                        &v139,
                        *(_QWORD *)(a1 + 864));
      v135 = v106;
    }
  }
  else
  {
    if ( v16 > 0x40 )
    {
      v113 = 0;
      v138 = sub_33738A0(a1);
      v2 = v39;
      LODWORD(v121) = v138;
      v135 = v39;
    }
    else
    {
      v22 = *(__int64 **)(a1 + 872);
      if ( !v22 )
        goto LABEL_63;
      v23 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
      v24 = sub_9208B0(v23, v134);
      v159 = v25;
      v26 = (unsigned __int64)(v24 + 7) >> 3;
      if ( (_BYTE)v25 )
        v26 |= 0x4000000000000000uLL;
      v159 = v26;
      v27 = _mm_load_si128(&v146);
      v28 = _mm_load_si128(&v147);
      v158 = (_OWORD *)v123;
      v160[0] = v27;
      v160[1] = v28;
      if ( (unsigned __int8)sub_CF4FA0(*v22, (__int64)&v158, (__int64)(v22 + 1), 0) )
      {
LABEL_63:
        v78 = *(_QWORD *)(a1 + 864);
        v113 = 0;
        v2 = *(unsigned int *)(v78 + 392);
        v121 = *(_QWORD *)(v78 + 384);
        v135 = *(_DWORD *)(v78 + 392);
      }
      else
      {
        v133 |= 0x20u;
        v135 = 0;
        v113 = 1;
        v121 = *(_QWORD *)(a1 + 864) + 288LL;
      }
    }
    v40 = *(_DWORD *)(a1 + 848);
    v41 = *(_QWORD *)a1;
    v139 = 0;
    v140 = v40;
    if ( v41 )
    {
      if ( &v139 != (__int64 *)(v41 + 48) )
      {
        v42 = *(_QWORD *)(v41 + 48);
        v139 = v42;
        if ( v42 )
          sub_B96E90((__int64)&v139, v42, 1);
      }
    }
  }
  v114 = -1;
  if ( sub_CE8520(a2) )
    v114 = sub_CE8560(a2);
  v45 = v157;
  v46 = v16;
  v47 = v157;
  v155 = v157;
  v156 = 0x400000000LL;
  if ( v16 > 4 )
  {
    sub_C8D5F0((__int64)&v155, v157, v16, 0x10u, v43, v44);
    v47 = v155;
    v46 = v16;
    v45 = &v155[16 * (unsigned int)v156];
  }
  v48 = v46;
  for ( i = &v47[16 * v46]; i != v45; v45 += 16 )
  {
    if ( v45 )
    {
      *(_QWORD *)v45 = 0;
      *((_DWORD *)v45 + 2) = 0;
    }
  }
  LODWORD(v156) = v16;
  if ( v16 <= 0x3F )
  {
    v52 = v160;
    v50 = v16;
    v158 = v160;
    v159 = 0x400000000LL;
    if ( v16 <= 4 )
    {
      v51 = v160;
      goto LABEL_47;
    }
  }
  else
  {
    v48 = 64;
    v46 = 64;
    v50 = 64;
    v158 = v160;
    v159 = 0x400000000LL;
  }
  v129 = v50;
  sub_C8D5F0((__int64)&v158, v160, v46, 0x10u, v50, v44);
  v51 = v158;
  LODWORD(v50) = v129;
  v52 = &v158[(unsigned int)v159];
LABEL_47:
  for ( j = &v51[v48]; j != v52; ++v52 )
  {
    if ( v52 )
    {
      *(_QWORD *)v52 = 0;
      *((_DWORD *)v52 + 2) = 0;
    }
  }
  LODWORD(v159) = v50;
  v54 = 0;
  _BitScanReverse64(&v55, 1LL << (v124 >> 1));
  v116 = 63 - (v55 ^ 0x3F);
  v56 = v16 - 1;
  v57 = 0;
  v117 = 16 * v56;
  v58 = v135;
  v136 = v2;
  v59 = a1;
  v60 = v3;
  v119 = v58;
  v115 = v126;
  while ( 1 )
  {
    if ( *(_BYTE *)(v57 + v153[0] + 8) && *(_QWORD *)(v57 + v153[0]) )
    {
      v144 = 0u;
      LODWORD(v145) = 0;
      BYTE4(v145) = 0;
    }
    else
    {
      *((_QWORD *)&v144 + 1) = *(_QWORD *)(v57 + v153[0]);
      v79 = 0;
      BYTE4(v145) = 0;
      *(_QWORD *)&v144 = v123 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v123 )
      {
        v80 = *(_QWORD *)(v123 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v80 + 8) - 17 <= 1 )
          v80 = **(_QWORD **)(v80 + 16);
        v79 = *(_DWORD *)(v80 + 8) >> 8;
      }
      LODWORD(v145) = v79;
    }
    v62 = *(_QWORD *)(v59 + 864);
    v143 = _mm_loadu_si128((const __m128i *)(v57 + v153[0]));
    v127 = v54;
    LOBYTE(v60) = v116;
    v63 = sub_3409320(v62, v115, v122, v143.m128i_i32[0], v143.m128i_i32[2], (unsigned int)&v139, 1);
    v64 = v60;
    BYTE1(v64) = 1;
    v60 = v64;
    v136 = v119 | v136 & 0xFFFFFFFF00000000LL;
    v130 = sub_33F1F00(
             *(_QWORD *)(v59 + 864),
             *(_DWORD *)(v57 + v151[0]),
             *(_QWORD *)(v57 + v151[0] + 8),
             (unsigned int)&v139,
             v121,
             v136,
             v63,
             v65,
             v144,
             v145,
             v64,
             v133,
             (__int64)&v146,
             v118);
    v67 = v66;
    v125 = *(_QWORD *)(v130 + 112);
    v68 = sub_CE8520(a2);
    v70 = v130;
    v71 = v127;
    if ( v68 )
    {
      v81 = *(_QWORD *)(v125 + 24);
      v82 = v114 & 0x80000000;
      LOBYTE(v83) = -1;
      if ( (v81 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
      {
LABEL_70:
        v114 >>= v83;
        *(_DWORD *)(v125 + 80) = v82;
        goto LABEL_61;
      }
      v87 = v81 >> 3;
      v88 = *(_BYTE *)(v125 + 24);
      LODWORD(v69) = v88 & 6;
      v89 = v88 & 2;
      if ( (_BYTE)v69 == 2 || (v88 & 1) != 0 )
      {
        if ( v89 )
        {
          v82 = v114 & ~(-1 << ((unsigned int)(HIWORD(v81) + 7) >> 3));
          if ( (_BYTE)v69 == 2 )
            goto LABEL_93;
        }
        else
        {
          v82 = v114 & ~(-1 << ((HIDWORD(v81) + 7) >> 3));
          if ( (_BYTE)v69 == 2 )
          {
LABEL_89:
            v91 = v87 >> 29;
LABEL_90:
            v83 = (v91 + 7) >> 3;
            goto LABEL_70;
          }
        }
      }
      else
      {
        v90 = HIWORD(v81);
        v69 = HIDWORD(*(_QWORD *)(v125 + 24));
        if ( !v89 )
          LODWORD(v90) = HIDWORD(*(_QWORD *)(v125 + 24));
        v82 = v114
            & ~(-1 << (((unsigned __int64)((unsigned int)v90
                                         * (unsigned __int16)((unsigned int)*(_QWORD *)(v125 + 24) >> 8))
                      + 7) >> 3));
      }
      if ( (v88 & 1) == 0 )
      {
        v69 = v87 >> 29;
        v92 = v87 >> 45;
        if ( !v89 )
          LODWORD(v92) = v87 >> 29;
        v83 = ((unsigned __int64)((unsigned int)v92 * (unsigned __int16)(v87 >> 5)) + 7) >> 3;
        goto LABEL_70;
      }
      if ( !v89 )
        goto LABEL_89;
LABEL_93:
      v91 = v87 >> 45;
      goto LABEL_90;
    }
LABEL_61:
    v72 = &v158[v127];
    *(_QWORD *)v72 = v130;
    v73 = v151[0];
    *((_DWORD *)v72 + 2) = 1;
    v74 = v57 + v73;
    v75 = (unsigned int *)&v148[v57];
    if ( *(_WORD *)&v148[v57] != *(_WORD *)v74 || !*(_WORD *)v74 && *((_QWORD *)v75 + 1) != *(_QWORD *)(v74 + 8) )
    {
      v76 = sub_33FB4C0(*(_QWORD *)(v59 + 864), v130, v67, &v139, *v75, *((_QWORD *)v75 + 1));
      v71 = v127;
      v70 = v76;
      LODWORD(v67) = v77;
    }
    v54 = v71 + 1;
    v61 = &v155[v57];
    *(_QWORD *)v61 = v70;
    *((_DWORD *)v61 + 2) = v67;
    if ( v57 == v117 )
      break;
    if ( v54 == 64 )
    {
      *((_QWORD *)&v110 + 1) = 64;
      *(_QWORD *)&v110 = v158;
      v84 = sub_33FC220(*(_QWORD *)(v59 + 864), 2, (unsigned int)&v139, 1, 0, v69, v110);
      v54 = 0;
      LODWORD(v121) = v84;
      v136 = v85 | v136 & 0xFFFFFFFF00000000LL;
      v119 = v85;
    }
    v57 += 16;
  }
  v29 = v59;
  if ( !v113 )
  {
    *((_QWORD *)&v111 + 1) = v54;
    *(_QWORD *)&v111 = v158;
    v93 = sub_33FC220(*(_QWORD *)(v59 + 864), 2, (unsigned int)&v139, 1, 0, v69, v111);
    v97 = v93;
    v98 = v94;
    if ( v112 )
    {
      v99 = *(_QWORD *)(v59 + 864);
      if ( v93 )
      {
        nullsub_1875(v93, v99, 0);
        *(_QWORD *)(v99 + 384) = v97;
        *(_DWORD *)(v99 + 392) = v98;
        sub_33E2B60(v99, 0);
      }
      else
      {
        *(_QWORD *)(v99 + 384) = 0;
        *(_DWORD *)(v99 + 392) = v94;
      }
    }
    else
    {
      v107 = *(unsigned int *)(v59 + 136);
      if ( v107 + 1 > (unsigned __int64)*(unsigned int *)(v59 + 140) )
      {
        sub_C8D5F0(v59 + 128, (const void *)(v59 + 144), v107 + 1, 0x10u, v95, v96);
        v107 = *(unsigned int *)(v59 + 136);
      }
      v108 = (__int64 *)(*(_QWORD *)(v59 + 128) + 16 * v107);
      *v108 = v97;
      v108[1] = v98;
      ++*(_DWORD *)(v59 + 136);
    }
  }
  v30 = *(_QWORD *)(v29 + 864);
  v31 = (unsigned __int64)v155;
  v32 = (unsigned int)v156;
  v33 = sub_33E5830(v30, v148);
  *((_QWORD *)&v109 + 1) = v32;
  *(_QWORD *)&v109 = v31;
  v36 = sub_3411630(v30, 55, (unsigned int)&v139, v33, v34, v35, v109);
  LODWORD(v30) = v37;
  *(_QWORD *)&v144 = a2;
  v38 = sub_337DC20(v29 + 8, (__int64 *)&v144);
  *v38 = v36;
  *((_DWORD *)v38 + 2) = v30;
  if ( v158 != v160 )
    _libc_free((unsigned __int64)v158);
  if ( v155 != v157 )
    _libc_free((unsigned __int64)v155);
  if ( v139 )
    sub_B91220((__int64)&v139, v139);
LABEL_22:
  if ( (_BYTE *)v153[0] != v154 )
    _libc_free(v153[0]);
  if ( (_BYTE *)v151[0] != v152 )
    _libc_free(v151[0]);
  if ( v148 != v150 )
    _libc_free((unsigned __int64)v148);
}
