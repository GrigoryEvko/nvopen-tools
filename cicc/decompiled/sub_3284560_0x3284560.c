// Function: sub_3284560
// Address: 0x3284560
//
__int64 __fastcall sub_3284560(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v8; // eax
  __int16 v9; // dx
  __int64 *v10; // rdx
  _QWORD *v11; // r10
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // r11
  unsigned int v15; // eax
  __int16 v16; // dx
  __int16 v17; // dx
  __int64 v18; // rdx
  unsigned int v19; // r15d
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // r9
  unsigned int v23; // r11d
  __int64 v24; // rdx
  __int64 v25; // r8
  _BYTE *v26; // rax
  __int64 v27; // rcx
  unsigned __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // rbx
  __int64 v31; // rax
  _DWORD *v32; // rax
  __int64 v33; // r12
  unsigned int v34; // r11d
  __int64 v35; // rdx
  int v36; // eax
  unsigned __int64 v37; // r15
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rax
  _BYTE *v41; // rdi
  __int64 v42; // rdx
  unsigned __int64 v43; // r8
  __int64 *v44; // rcx
  __int64 v45; // rax
  unsigned __int64 v46; // r15
  __int64 v47; // rax
  __int64 *v48; // rdx
  __m128i v49; // xmm1
  __m128i v50; // xmm2
  __m128i v51; // xmm3
  int *v52; // rax
  int **v53; // rdx
  char v54; // r13
  __int64 v55; // rax
  __int64 v56; // rdi
  char v57; // al
  char v58; // al
  int v59; // r11d
  unsigned __int64 v60; // rax
  unsigned int v61; // r11d
  __int64 v62; // rsi
  unsigned __int64 v63; // r12
  _QWORD *v64; // rax
  _QWORD *v65; // rdx
  int v66; // eax
  int v67; // r9d
  __int64 v68; // rax
  __int64 v69; // r11
  char v70; // al
  __int64 v71; // r14
  char v72; // al
  __int128 v73; // rax
  int v74; // eax
  int v75; // edx
  int v76; // eax
  int v77; // edx
  __int64 v78; // rax
  unsigned int v79; // edx
  __int128 v80; // [rsp-30h] [rbp-2A0h]
  __int128 v81; // [rsp-10h] [rbp-280h]
  __int128 v82; // [rsp-10h] [rbp-280h]
  unsigned int v83; // [rsp+8h] [rbp-268h]
  unsigned int v85; // [rsp+1Ch] [rbp-254h]
  int v86; // [rsp+20h] [rbp-250h]
  unsigned __int64 v87; // [rsp+28h] [rbp-248h]
  __int64 v88; // [rsp+38h] [rbp-238h]
  unsigned int v89; // [rsp+44h] [rbp-22Ch]
  unsigned int v90; // [rsp+48h] [rbp-228h]
  _QWORD *v91; // [rsp+48h] [rbp-228h]
  _QWORD *v92; // [rsp+50h] [rbp-220h]
  _QWORD *v93; // [rsp+50h] [rbp-220h]
  unsigned __int8 v94; // [rsp+50h] [rbp-220h]
  _QWORD *v95; // [rsp+50h] [rbp-220h]
  _QWORD *v96; // [rsp+50h] [rbp-220h]
  _QWORD *v97; // [rsp+50h] [rbp-220h]
  __int64 v98; // [rsp+58h] [rbp-218h]
  _QWORD *v99; // [rsp+60h] [rbp-210h]
  _QWORD *v100; // [rsp+70h] [rbp-200h]
  unsigned int v101; // [rsp+70h] [rbp-200h]
  unsigned int v102; // [rsp+70h] [rbp-200h]
  int v103; // [rsp+70h] [rbp-200h]
  _QWORD *v104; // [rsp+70h] [rbp-200h]
  _QWORD *v105; // [rsp+70h] [rbp-200h]
  unsigned __int8 v106; // [rsp+70h] [rbp-200h]
  unsigned int v107; // [rsp+78h] [rbp-1F8h]
  _QWORD *v108; // [rsp+78h] [rbp-1F8h]
  unsigned __int8 *v109; // [rsp+78h] [rbp-1F8h]
  __int64 v110; // [rsp+78h] [rbp-1F8h]
  unsigned __int8 v111; // [rsp+78h] [rbp-1F8h]
  __int64 v112; // [rsp+78h] [rbp-1F8h]
  _QWORD *v113; // [rsp+80h] [rbp-1F0h]
  unsigned __int64 v114; // [rsp+88h] [rbp-1E8h]
  unsigned __int64 v115; // [rsp+90h] [rbp-1E0h]
  __int64 v116; // [rsp+90h] [rbp-1E0h]
  _QWORD *v117; // [rsp+90h] [rbp-1E0h]
  int v118; // [rsp+98h] [rbp-1D8h]
  __int64 v119; // [rsp+98h] [rbp-1D8h]
  __int64 v120; // [rsp+98h] [rbp-1D8h]
  int v121; // [rsp+BCh] [rbp-1B4h] BYREF
  int v122; // [rsp+C0h] [rbp-1B0h] BYREF
  int v123; // [rsp+C4h] [rbp-1ACh] BYREF
  __int64 v124; // [rsp+C8h] [rbp-1A8h] BYREF
  __int16 v125; // [rsp+D0h] [rbp-1A0h]
  __int64 v126; // [rsp+D8h] [rbp-198h]
  unsigned int v127; // [rsp+E0h] [rbp-190h] BYREF
  __int64 v128; // [rsp+E8h] [rbp-188h]
  __int64 v129; // [rsp+F0h] [rbp-180h] BYREF
  int v130; // [rsp+F8h] [rbp-178h]
  int *v131[4]; // [rsp+100h] [rbp-170h] BYREF
  __m128i v132; // [rsp+120h] [rbp-150h] BYREF
  __m128i v133; // [rsp+130h] [rbp-140h] BYREF
  __m128i v134; // [rsp+140h] [rbp-130h] BYREF
  __int64 v135; // [rsp+150h] [rbp-120h]
  __m128i v136; // [rsp+160h] [rbp-110h] BYREF
  __m128i v137; // [rsp+170h] [rbp-100h]
  __m128i v138; // [rsp+180h] [rbp-F0h]
  __int128 v139; // [rsp+190h] [rbp-E0h]
  _QWORD *v140; // [rsp+1A0h] [rbp-D0h] BYREF
  __int64 v141; // [rsp+1A8h] [rbp-C8h]
  _QWORD v142[8]; // [rsp+1B0h] [rbp-C0h] BYREF
  _BYTE *v143; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 v144; // [rsp+1F8h] [rbp-78h]
  _BYTE v145[112]; // [rsp+200h] [rbp-70h] BYREF

  if ( *((_BYTE *)a1 + 33) )
    return 0;
  if ( !*((_DWORD *)a1 + 7) )
    return 0;
  v8 = *(unsigned __int16 *)(a2 + 96);
  v126 = *(_QWORD *)(a2 + 104);
  v125 = v8;
  if ( (unsigned __int16)(v8 - 5) > 2u )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) != 0 )
    return 0;
  v9 = *(_WORD *)(a2 + 32);
  if ( (v9 & 8) != 0 || (v9 & 0x380) != 0 )
    return 0;
  v10 = *(__int64 **)(a2 + 40);
  v11 = v142;
  v12 = v10[1];
  v13 = *v10;
  v140 = v142;
  v142[0] = a2;
  v118 = v12;
  v14 = *(_QWORD *)&byte_444C4A0[16 * v8 - 16];
  v141 = 0x800000001LL;
  v15 = 0x40 / (unsigned int)v14;
  v121 = v14;
  while ( *(_DWORD *)(v13 + 24) == 299 )
  {
    v16 = *(_WORD *)(v13 + 96);
    if ( v125 == v16
      && (v126 == *(_QWORD *)(v13 + 104) || v16)
      && (*(_BYTE *)(*(_QWORD *)(v13 + 112) + 37LL) & 0xF) == 0 )
    {
      v17 = *(_WORD *)(v13 + 32);
      if ( (v17 & 8) == 0 && (v17 & 0x380) == 0 )
      {
        v18 = *(_QWORD *)(v13 + 56);
        if ( v18 )
        {
          if ( !*(_QWORD *)(v18 + 32) )
          {
            v42 = (unsigned int)v141;
            v43 = (unsigned int)v141 + 1LL;
            if ( v43 > HIDWORD(v141) )
            {
              v102 = v15;
              v110 = v14;
              v117 = v11;
              sub_C8D5F0((__int64)&v140, v11, (unsigned int)v141 + 1LL, 8u, v43, a6);
              v42 = (unsigned int)v141;
              v15 = v102;
              v14 = v110;
              v11 = v117;
            }
            v140[v42] = v13;
            v44 = *(__int64 **)(v13 + 40);
            LODWORD(v141) = v141 + 1;
            v13 = *v44;
            v118 = *((_DWORD *)v44 + 2);
            if ( v15 >= (unsigned int)v141 )
              continue;
          }
        }
      }
    }
    goto LABEL_18;
  }
  v115 = (unsigned int)v141;
  v19 = v141;
  if ( (unsigned int)v141 <= 1 )
    goto LABEL_18;
  v100 = v11;
  v107 = v14;
  v20 = *(_QWORD **)(*a1 + 64LL);
  v122 = v141;
  v88 = (__int64)v20;
  v89 = v14 * v141;
  v21 = sub_327FC40(v20, (int)v14 * (int)v141);
  v23 = v107;
  v11 = v100;
  v127 = v21;
  v128 = v24;
  if ( (unsigned __int16)(v21 - 6) > 2u )
  {
LABEL_18:
    result = 0;
    goto LABEL_19;
  }
  v143 = v145;
  v144 = 0x800000000LL;
  v25 = 8LL * v19;
  if ( v19 > 8uLL )
  {
    sub_C8D5F0((__int64)&v143, v145, v19, 8u, v25, v22);
    v64 = v143;
    v23 = v107;
    v11 = v100;
    v65 = &v143[8 * v19];
    do
      *v64++ = 0x7FFFFFFFFFFFFFFFLL;
    while ( v65 != v64 );
  }
  else
  {
    v26 = v145;
    do
    {
      *(_QWORD *)v26 = 0x7FFFFFFFFFFFFFFFLL;
      v26 += 8;
    }
    while ( &v145[v25] != v26 );
  }
  LODWORD(v144) = v19;
  v124 = 0x7FFFFFFFFFFFFFFFLL;
  v136 = 0;
  v137 = 0;
  v92 = &v140[(unsigned int)v141];
  v138 = 0;
  v139 = 0;
  if ( v92 == v140 )
  {
    sub_2E79000(*(__int64 **)(*a1 + 40LL));
    v123 = 0;
    BUG();
  }
  v101 = 0;
  v27 = 0;
  v98 = 0;
  v85 = v19;
  v90 = v23;
  v99 = a1;
  v108 = v11;
  v28 = (unsigned __int64)v140;
  v86 = v13;
  v29 = 0;
  do
  {
    v30 = *(_QWORD *)v28;
    v31 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v28 + 40LL) + 40LL);
    if ( *(_DWORD *)(v31 + 24) != 216 )
    {
LABEL_40:
      v11 = v108;
      v41 = v143;
LABEL_41:
      result = 0;
      goto LABEL_42;
    }
    v32 = *(_DWORD **)(v31 + 40);
    v33 = *(_QWORD *)v32;
    v34 = v32[2];
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v32 + 24LL) - 191) > 1 )
    {
      v37 = 0;
    }
    else
    {
      v35 = *(_QWORD *)(*(_QWORD *)(v33 + 40) + 40LL);
      v36 = *(_DWORD *)(v35 + 24);
      if ( v36 == 35 || (v37 = 0, v36 == 11) )
      {
        v45 = *(_QWORD *)(v35 + 96);
        v46 = *(_QWORD *)(v45 + 24);
        if ( *(_DWORD *)(v45 + 32) > 0x40u )
          v46 = **(_QWORD **)(v45 + 24);
        if ( v46 % v90 || sub_3263630(v33, v34) - (unsigned __int64)v90 < v46 )
          goto LABEL_40;
        v47 = *(_QWORD *)(v33 + 40);
        v37 = v46 / v90;
        v33 = *(_QWORD *)v47;
        v34 = *(_DWORD *)(v47 + 8);
      }
    }
    if ( !v29 )
    {
      v101 = v34;
      v29 = v33;
      goto LABEL_59;
    }
    if ( v29 != v33 )
    {
      if ( (unsigned int)(*(_DWORD *)(v29 + 24) - 213) <= 3 )
      {
LABEL_57:
        v48 = *(__int64 **)(v29 + 40);
        v39 = *v48;
        v38 = *((_DWORD *)v48 + 2);
      }
      else
      {
        v38 = 0;
        v39 = 0;
      }
      if ( v39 == v33 && v38 == v34 )
        goto LABEL_75;
      goto LABEL_37;
    }
    if ( v34 == v101 )
      goto LABEL_59;
    if ( (unsigned int)(*(_DWORD *)(v29 + 24) - 213) <= 3 )
      goto LABEL_57;
LABEL_37:
    if ( (unsigned int)(*(_DWORD *)(v33 + 24) - 213) > 3 )
      goto LABEL_40;
    v40 = *(_QWORD *)(v33 + 40);
    if ( *(_QWORD *)v40 != v29 || *(_DWORD *)(v40 + 8) != v101 )
      goto LABEL_40;
LABEL_75:
    v83 = v34;
    v87 = sub_3263630(v33, v34);
    v60 = sub_3263630(v29, v101);
    v61 = v83;
    if ( v87 <= v60 )
      v61 = v101;
    else
      v29 = v33;
    v62 = v61;
    v101 = v61;
    v63 = sub_3263630(v29, v61);
    if ( v63 < sub_32844A0((unsigned __int16 *)&v127, v62) )
      goto LABEL_40;
LABEL_59:
    sub_33644B0(&v132, v30, *v99, v27);
    v131[0] = 0;
    if ( BYTE8(v139) )
    {
      if ( !(unsigned __int8)sub_3364290(&v136, &v132, *v99, v131) )
        goto LABEL_40;
      v52 = v131[0];
    }
    else
    {
      v49 = _mm_loadu_si128(&v132);
      BYTE8(v139) = 1;
      v50 = _mm_loadu_si128(&v133);
      v51 = _mm_loadu_si128(&v134);
      *(_QWORD *)&v139 = v135;
      v52 = 0;
      v136 = v49;
      v137 = v50;
      v138 = v51;
    }
    if ( v124 > (__int64)v52 )
    {
      v124 = (__int64)v52;
      v98 = v30;
    }
    v41 = v143;
    if ( v115 <= v37 || (v27 = 0x7FFFFFFFFFFFFFFFLL, v53 = (int **)&v143[8 * v37], *v53 != (int *)0x7FFFFFFFFFFFFFFFLL) )
    {
      v11 = v108;
      goto LABEL_41;
    }
    *v53 = v52;
    v28 += 8LL;
  }
  while ( v92 != (_QWORD *)v28 );
  v116 = v29;
  v54 = 0;
  v93 = v108;
  v55 = sub_2E79000(*(__int64 **)(*v99 + 40LL));
  v56 = v99[1];
  v123 = 0;
  v109 = (unsigned __int8 *)v55;
  v57 = sub_2FEBB30(v56, v88, v55, v127, v128, *(_QWORD *)(v98 + 112), &v123);
  v11 = v93;
  if ( !v57 || !v123 )
    goto LABEL_71;
  v91 = v93;
  v131[0] = &v122;
  v131[1] = (int *)&v143;
  v131[2] = &v121;
  v131[3] = (int *)&v124;
  v94 = *v109;
  v58 = sub_325DDD0(v131, *v109 ^ 1u);
  v11 = v91;
  if ( v58 )
    goto LABEL_95;
  if ( v59 == 8 )
  {
    v72 = sub_325DDD0(v131, v94);
    v11 = v91;
    v54 = v72;
    if ( !v72 )
      goto LABEL_71;
LABEL_95:
    v67 = 0;
LABEL_86:
    v129 = *(_QWORD *)(a2 + 80);
    if ( v129 )
    {
      v95 = v11;
      v111 = v67;
      sub_325F5D0(&v129);
      v11 = v95;
      v67 = v111;
    }
    v130 = *(_DWORD *)(a2 + 72);
    v112 = v101;
    v68 = *(_QWORD *)(v29 + 48) + 16LL * v101;
    if ( (_WORD)v127 != *(_WORD *)v68 || !(_WORD)v127 && (v68 = *(_QWORD *)(v68 + 8), v128 != v68) )
    {
      v97 = v11;
      v106 = v67;
      *((_QWORD *)&v82 + 1) = v112;
      *(_QWORD *)&v82 = v29;
      v78 = sub_33FAF80(*v99, 216, (unsigned int)&v129, v127, v128, v67, v82);
      v11 = v97;
      v67 = v106;
      v116 = v78;
      v68 = v79;
      v114 = v79 | v112 & 0xFFFFFFFF00000000LL;
      v112 = v79;
    }
    v69 = *v99;
    if ( v54 )
    {
      v105 = v11;
      *((_QWORD *)&v81 + 1) = v112 | v114 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v81 = v116;
      v76 = sub_33FAF80(*v99, 197, (unsigned int)&v129, v127, v128, v67, v81);
      v69 = *v99;
      v11 = v105;
      LODWORD(v116) = v76;
      LODWORD(v112) = v77;
    }
    else if ( (_BYTE)v67 )
    {
      v104 = v11;
      *(_QWORD *)&v73 = sub_3400BD0(v69, v89 >> 1, (unsigned int)&v129, v127, v128, 0, 0, v68);
      *((_QWORD *)&v80 + 1) = v112 | v114 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v80 = v116;
      v74 = sub_3406EB0(*v99, 194, (unsigned int)&v129, v127, v128, DWORD2(v73), v80, v73);
      v69 = *v99;
      v11 = v104;
      LODWORD(v116) = v74;
      LODWORD(v112) = v75;
    }
    v132 = 0u;
    v133 = 0u;
    v96 = v11;
    v103 = v69;
    v70 = sub_2EAC4F0(*(_QWORD *)(v98 + 112));
    v71 = sub_33F4560(
            v103,
            v86,
            v118,
            (unsigned int)&v129,
            v116,
            v112,
            *(_QWORD *)(*(_QWORD *)(v98 + 40) + 80LL),
            *(_QWORD *)(*(_QWORD *)(v98 + 40) + 88LL),
            *(_OWORD *)*(_QWORD *)(v98 + 112),
            *(_QWORD *)(*(_QWORD *)(v98 + 112) + 16LL),
            v70,
            0,
            (__int64)&v132);
    sub_34158F0(*v99, a2, v71);
    sub_9C6650(&v129);
    v41 = v143;
    v11 = v96;
    result = v71;
    goto LABEL_42;
  }
  if ( v85 == 2 )
  {
    v66 = sub_325DDD0(v131, v94);
    v11 = v91;
    v67 = v66;
    if ( (_BYTE)v66 )
    {
      v54 = 0;
      goto LABEL_86;
    }
  }
LABEL_71:
  v41 = v143;
  result = 0;
LABEL_42:
  if ( v41 != v145 )
  {
    v113 = v11;
    v120 = result;
    _libc_free((unsigned __int64)v41);
    v11 = v113;
    result = v120;
  }
LABEL_19:
  if ( v140 != v11 )
  {
    v119 = result;
    _libc_free((unsigned __int64)v140);
    return v119;
  }
  return result;
}
