// Function: sub_348B620
// Address: 0x348b620
//
__int64 __fastcall sub_348B620(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        int a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  unsigned __int16 *v12; // r8
  int v13; // r14d
  __int64 v14; // r15
  _QWORD *v15; // r13
  __int64 *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // eax
  __int16 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i v25; // xmm1
  __m128i v26; // xmm0
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // r13
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int128 v36; // rdi
  int v37; // eax
  __int64 v38; // rbx
  unsigned int v39; // r10d
  __int64 v40; // r9
  char v41; // al
  unsigned int v42; // edx
  unsigned __int8 *v43; // rax
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rdx
  unsigned int v47; // r10d
  __int64 v48; // r9
  char v49; // al
  unsigned int v50; // edx
  __int64 v51; // r8
  __int64 v52; // r9
  __int128 v53; // rax
  __int128 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // rdx
  unsigned int v62; // edx
  unsigned int v63; // edx
  __int64 v64; // rax
  unsigned int v65; // edx
  unsigned int v66; // edx
  __int64 v67; // r9
  unsigned int v68; // edx
  __int64 v69; // r9
  unsigned __int8 *v70; // rax
  unsigned int v71; // edx
  __int128 v72; // rax
  __int64 v73; // r9
  __int128 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r9
  __int64 v79; // r9
  unsigned __int8 *v80; // rax
  __int128 v81; // rdi
  __int32 v82; // edx
  __int128 v83; // rax
  __int64 v84; // r9
  __int128 v85; // [rsp-20h] [rbp-5C0h]
  __int128 v86; // [rsp-20h] [rbp-5C0h]
  __int128 v87; // [rsp-10h] [rbp-5B0h]
  __int128 v88; // [rsp-10h] [rbp-5B0h]
  __int64 v89; // [rsp+8h] [rbp-598h]
  __int64 v90; // [rsp+8h] [rbp-598h]
  unsigned int v91; // [rsp+8h] [rbp-598h]
  __int128 v92; // [rsp+10h] [rbp-590h]
  __int64 v93; // [rsp+10h] [rbp-590h]
  unsigned __int8 *v94; // [rsp+10h] [rbp-590h]
  unsigned int v95; // [rsp+20h] [rbp-580h]
  __int128 v96; // [rsp+30h] [rbp-570h]
  unsigned __int8 *v97; // [rsp+30h] [rbp-570h]
  char v98; // [rsp+40h] [rbp-560h]
  __int128 v99; // [rsp+40h] [rbp-560h]
  __int128 v100; // [rsp+40h] [rbp-560h]
  __int128 v101; // [rsp+40h] [rbp-560h]
  unsigned int v102; // [rsp+50h] [rbp-550h]
  __int128 v104; // [rsp+60h] [rbp-540h]
  __int128 v105; // [rsp+70h] [rbp-530h]
  char v106; // [rsp+C9h] [rbp-4D7h] BYREF
  char v107; // [rsp+CAh] [rbp-4D6h] BYREF
  char v108; // [rsp+CBh] [rbp-4D5h] BYREF
  char v109; // [rsp+CCh] [rbp-4D4h] BYREF
  char v110; // [rsp+CDh] [rbp-4D3h] BYREF
  char v111; // [rsp+CEh] [rbp-4D2h] BYREF
  char v112; // [rsp+CFh] [rbp-4D1h] BYREF
  __int64 v113; // [rsp+D0h] [rbp-4D0h] BYREF
  __int64 v114; // [rsp+D8h] [rbp-4C8h]
  __int16 v115; // [rsp+E0h] [rbp-4C0h] BYREF
  __int64 v116; // [rsp+E8h] [rbp-4B8h]
  __int64 v117; // [rsp+F0h] [rbp-4B0h] BYREF
  __int64 v118; // [rsp+F8h] [rbp-4A8h]
  unsigned int v119; // [rsp+100h] [rbp-4A0h] BYREF
  __int64 v120; // [rsp+108h] [rbp-498h]
  __m128i v121; // [rsp+110h] [rbp-490h] BYREF
  void (__fastcall *v122)(__m128i *, __m128i *, __int64, __int64); // [rsp+120h] [rbp-480h]
  void *v123; // [rsp+128h] [rbp-478h]
  __int64 *v124; // [rsp+130h] [rbp-470h] BYREF
  __int64 v125; // [rsp+138h] [rbp-468h]
  _BYTE v126[256]; // [rsp+140h] [rbp-460h] BYREF
  __int64 *v127; // [rsp+240h] [rbp-360h] BYREF
  __int64 v128; // [rsp+248h] [rbp-358h]
  _BYTE v129[256]; // [rsp+250h] [rbp-350h] BYREF
  __int64 *v130; // [rsp+350h] [rbp-250h] BYREF
  __int64 v131; // [rsp+358h] [rbp-248h]
  _BYTE v132[256]; // [rsp+360h] [rbp-240h] BYREF
  char *v133; // [rsp+460h] [rbp-140h]
  __int64 v134; // [rsp+468h] [rbp-138h]
  char v135; // [rsp+470h] [rbp-130h] BYREF

  v12 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * a5);
  *(_QWORD *)&v105 = a2;
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  *((_QWORD *)&v105 + 1) = a3;
  v15 = *(_QWORD **)(a8 + 16);
  LOWORD(v113) = v13;
  v114 = v14;
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v14 = 0;
      LOWORD(v13) = word_4456580[v13 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v113) )
  {
    LOWORD(v13) = sub_3009970((__int64)&v113, a2, v58, v59, v60);
    v14 = v61;
  }
  v16 = (__int64 *)v15[5];
  v115 = v13;
  v116 = v14;
  v17 = sub_2E79000(v16);
  v18 = (unsigned int)v113;
  v19 = sub_2FE6750(a1, (unsigned int)v113, v114, v17);
  LODWORD(v117) = v19;
  v20 = v19;
  v118 = v21;
  if ( (_WORD)v19 )
  {
    if ( (unsigned __int16)(v19 - 17) <= 0xD3u )
    {
      v22 = 0;
      v20 = word_4456580[(unsigned __int16)v19 - 1];
      goto LABEL_7;
    }
  }
  else if ( sub_30070B0((__int64)&v117) )
  {
    v20 = sub_3009970((__int64)&v117, v18, v33, v34, v35);
    goto LABEL_7;
  }
  v22 = v118;
LABEL_7:
  LOWORD(v119) = v20;
  v120 = v22;
  if ( *(int *)(a8 + 8) > 1 )
  {
    v23 = 1;
    if ( (_WORD)v113 != 1 )
    {
      if ( !(_WORD)v113 )
        return 0;
      v23 = (unsigned __int16)v113;
      if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v113 + 112) )
        return 0;
    }
    if ( (*(_BYTE *)(a1 + 500 * v23 + 6472) & 0xFB) != 0 )
      return 0;
  }
  v106 = 1;
  v124 = (__int64 *)v126;
  v125 = 0x1000000000LL;
  v128 = 0x1000000000LL;
  v131 = 0x1000000000LL;
  v134 = 0x1000000000LL;
  v24 = *(_QWORD *)(a4 + 40);
  v107 = 1;
  v25 = _mm_loadu_si128((const __m128i *)(v24 + 40));
  v108 = 0;
  v109 = 1;
  v110 = 0;
  v111 = 1;
  v112 = 0;
  v127 = (__int64 *)v129;
  v130 = (__int64 *)v132;
  v26 = _mm_loadu_si128((const __m128i *)v24);
  v27 = *(_QWORD *)(v24 + 40);
  v133 = &v135;
  v89 = v27;
  v122 = 0;
  v92 = (__int128)v26;
  v28 = (_QWORD *)sub_22077B0(0x70u);
  if ( v28 )
  {
    v28[8] = v15;
    *v28 = &v106;
    v28[1] = &v112;
    v28[2] = &v108;
    v28[3] = &v109;
    v28[4] = &v107;
    v28[5] = &v110;
    v28[7] = &v124;
    v28[11] = &v127;
    v28[6] = &v111;
    v28[9] = a9;
    v28[10] = &v115;
    v28[12] = &v119;
    v28[13] = &v130;
  }
  v121.m128i_i64[0] = (__int64)v28;
  v123 = sub_34428D0;
  v122 = (void (__fastcall *)(__m128i *, __m128i *, __int64, __int64))sub_343FA30;
  v29 = (unsigned int)sub_33CACD0(v25.m128i_i64[0], v25.m128i_i64[1], a7, DWORD2(a7), (__int64)&v121, 0, 0);
  if ( v122 )
  {
    v98 = v29;
    v122(&v121, &v121, 3, v29);
    LOBYTE(v29) = v98;
  }
  if ( !(_BYTE)v29 || v109 || v111 )
    goto LABEL_15;
  *(_QWORD *)&v36 = v124;
  v37 = *(_DWORD *)(v89 + 24);
  if ( v37 == 156 )
  {
    if ( v108 )
    {
      *((_QWORD *)&v36 + 1) = (unsigned int)v125;
      v121.m128i_i64[0] = (__int64)sub_33CF170;
      v123 = sub_343F120;
      v122 = (void (__fastcall *)(__m128i *, __m128i *, __int64, __int64))sub_343F130;
      sub_344D9B0(v36, &v121, 0, 0);
      if ( v122 )
        *(double *)v26.m128i_i64 = ((double (__fastcall *)(__m128i *, __m128i *, __int64))v122)(&v121, &v121, 3);
      v80 = sub_3400BD0((__int64)v15, 0, a9, v119, v120, 0, v26, 0);
      *((_QWORD *)&v81 + 1) = (unsigned int)v128;
      *(_QWORD *)&v81 = v127;
      v121.m128i_i64[0] = (__int64)sub_33CF460;
      v123 = sub_343F120;
      v122 = (void (__fastcall *)(__m128i *, __m128i *, __int64, __int64))sub_343F130;
      sub_344D9B0(v81, &v121, (__int64)v80, v82);
      if ( v122 )
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v122)(&v121, &v121, 3);
    }
    *((_QWORD *)&v87 + 1) = (unsigned int)v125;
    *(_QWORD *)&v87 = v124;
    *(_QWORD *)&v99 = sub_33FC220(v15, 156, a9, v113, v114, v30, v87);
    *((_QWORD *)&v99 + 1) = v66;
    *((_QWORD *)&v86 + 1) = (unsigned int)v128;
    *(_QWORD *)&v86 = v127;
    *(_QWORD *)&v96 = sub_33FC220(v15, 156, a9, v117, v118, v67, v86);
    *((_QWORD *)&v96 + 1) = v68;
    *((_QWORD *)&v88 + 1) = (unsigned int)v131;
    *(_QWORD *)&v88 = v130;
    v70 = sub_33FC220(v15, 156, a9, v113, v114, v69, v88);
    v95 = v71;
    v38 = (__int64)v70;
  }
  else if ( v37 == 168 )
  {
    *(_QWORD *)&v99 = sub_3288900((__int64)v15, v113, v114, a9, *v124, v124[1]);
    *((_QWORD *)&v99 + 1) = v62;
    *(_QWORD *)&v96 = sub_3288900((__int64)v15, v117, v118, a9, *v127, v127[1]);
    *((_QWORD *)&v96 + 1) = v63;
    v64 = sub_3288900((__int64)v15, v113, v114, a9, *v130, v130[1]);
    v95 = v65;
    v38 = v64;
  }
  else
  {
    *(_QWORD *)&v99 = *v124;
    *((_QWORD *)&v99 + 1) = *((unsigned int *)v124 + 2);
    *(_QWORD *)&v96 = *v127;
    v38 = *v130;
    *((_QWORD *)&v96 + 1) = *((unsigned int *)v127 + 2);
    v95 = *((_DWORD *)v130 + 2);
  }
  if ( !v106 && !v107 )
  {
    v39 = v113;
    v40 = v114;
    if ( *(int *)(a8 + 8) > 1 )
    {
      v102 = v113;
      v90 = v114;
      v41 = sub_328A020(a1, 0x39u, v113, v114, 0);
      v40 = v90;
      v39 = v102;
      if ( !v41 )
        goto LABEL_15;
    }
    *(_QWORD *)&v92 = sub_3406EB0(v15, 0x39u, a9, v39, v40, v40, v92, a7);
    *((_QWORD *)&v92 + 1) = v42 | *((_QWORD *)&v92 + 1) & 0xFFFFFFFF00000000LL;
  }
  v43 = sub_3406EB0(v15, 0x3Au, a9, (unsigned int)v113, v114, v30, v92, v99);
  *((_QWORD *)&v100 + 1) = v45;
  v46 = *(unsigned int *)(a10 + 8);
  *(_QWORD *)&v100 = v43;
  if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(a10 + 12) )
  {
    v94 = v43;
    sub_C8D5F0(a10, (const void *)(a10 + 16), v46 + 1, 8u, v46 + 1, v44);
    v46 = *(unsigned int *)(a10 + 8);
    v43 = v94;
  }
  *(_QWORD *)(*(_QWORD *)a10 + 8 * v46) = v43;
  ++*(_DWORD *)(a10 + 8);
  if ( v110 )
  {
    v47 = v113;
    v48 = v114;
    if ( *(int *)(a8 + 8) > 1 )
    {
      v91 = v113;
      v93 = v114;
      v49 = sub_328A020(a1, 0xC2u, v113, v114, 0);
      v48 = v93;
      v47 = v91;
      if ( !v49 )
        goto LABEL_15;
    }
    v97 = sub_3406EB0(v15, 0xC2u, a9, v47, v48, v48, v100, v96);
    *((_QWORD *)&v100 + 1) = v50 | *((_QWORD *)&v100 + 1) & 0xFFFFFFFF00000000LL;
    sub_3489D20(a10, (__int64)v97, v50, *((__int64 *)&v100 + 1), v51, v52);
    v43 = v97;
  }
  *(_QWORD *)&v100 = v43;
  *(_QWORD *)&v53 = sub_33ED040(v15, 3 * (unsigned int)(a6 == 17) + 10);
  *((_QWORD *)&v85 + 1) = v95;
  *(_QWORD *)&v85 = v38;
  *(_QWORD *)&v54 = sub_340F900(v15, 0xD0u, a9, v105, *((__int64 *)&v105 + 1), v95, v100, v85, v53);
  if ( !v112 )
  {
    v31 = v54;
    goto LABEL_16;
  }
  v101 = v54;
  sub_3489D20(a10, v54, *((__int64 *)&v54 + 1), v55, v56, v57);
  *(_QWORD *)&v72 = sub_33ED040(v15, 0xDu);
  *(_QWORD *)&v74 = sub_340F900(v15, 0xD0u, a9, v105, *((__int64 *)&v105 + 1), v73, *(_OWORD *)&v25, a7, v72);
  v104 = v74;
  sub_3489D20(a10, v74, *((__int64 *)&v74 + 1), v75, v76, v77);
  if ( (unsigned __int8)sub_328A020(a1, 0xCEu, v105, *((__int64 *)&v105 + 1), 0) )
  {
    *(_QWORD *)&v83 = sub_3401740((__int64)v15, a6 != 17, a9, (unsigned int)v105, *((__int64 *)&v105 + 1), v78, v105);
    v31 = sub_340F900(v15, 0xCEu, a9, v105, *((__int64 *)&v105 + 1), v84, v104, v83, v101);
    goto LABEL_16;
  }
  if ( (unsigned __int8)sub_328A020(a1, 0xBCu, v105, *((__int64 *)&v105 + 1), 0) )
  {
    v31 = (__int64)sub_3406EB0(v15, 0xBCu, a9, (unsigned int)v105, *((__int64 *)&v105 + 1), v79, v101, v104);
    goto LABEL_16;
  }
LABEL_15:
  v31 = 0;
LABEL_16:
  if ( v130 != (__int64 *)v132 )
    _libc_free((unsigned __int64)v130);
  if ( v127 != (__int64 *)v129 )
    _libc_free((unsigned __int64)v127);
  if ( v124 != (__int64 *)v126 )
    _libc_free((unsigned __int64)v124);
  return v31;
}
