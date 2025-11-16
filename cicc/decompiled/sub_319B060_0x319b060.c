// Function: sub_319B060
// Address: 0x319b060
//
void __fastcall sub_319B060(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        char a7,
        char a8,
        char a9,
        _QWORD *a10,
        __int64 a11)
{
  unsigned int v11; // ebx
  _QWORD *v12; // r12
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // eax
  unsigned int v19; // ecx
  unsigned int v20; // edx
  unsigned __int64 v21; // rax
  _QWORD *v22; // rbx
  char *v23; // rbx
  char v24; // dh
  __int64 v25; // r12
  char v26; // al
  char v27; // dl
  __int64 v28; // r14
  __int16 v29; // cx
  __int16 v30; // r13
  __int64 v31; // rax
  unsigned __int8 v32; // r14
  __int64 *v33; // rdi
  char v34; // bl
  __int64 v35; // r12
  unsigned __int64 v36; // rax
  char v37; // r13
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r15
  __int16 v42; // ax
  _QWORD *v43; // rbx
  __int64 v44; // r9
  unsigned int *v45; // r15
  unsigned int *v46; // r12
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // r15
  _QWORD *v50; // rax
  __int64 v51; // r9
  __int64 v52; // r12
  unsigned int *v53; // r15
  unsigned int *v54; // r13
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int16 v57; // ax
  __int16 v58; // ax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rbx
  unsigned __int64 v63; // rax
  int v64; // edx
  unsigned __int8 *v65; // rdi
  unsigned __int8 *v66; // rax
  unsigned __int64 v67; // rax
  int v68; // edx
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned __int64 v72; // rax
  char v73; // r15
  unsigned __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rbx
  __int64 v77; // r8
  int v78; // eax
  int v79; // eax
  unsigned int v80; // edx
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // r12
  _QWORD *v85; // rax
  unsigned int *v86; // r13
  unsigned int *v87; // r12
  __int64 v88; // rdx
  unsigned int v89; // esi
  __int64 v90; // r13
  __int64 v91; // r9
  _QWORD *v92; // r12
  unsigned int *v93; // r15
  unsigned int *v94; // r13
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int16 v97; // ax
  __int16 v98; // ax
  _BYTE *v99; // rax
  __int64 v100; // r12
  int v101; // eax
  int v102; // eax
  unsigned int v103; // edx
  __int64 v104; // rax
  __int64 v105; // rdx
  __int64 v106; // rdx
  _BYTE *v107; // rax
  __int64 v108; // rax
  __int64 v109; // r15
  _QWORD *v110; // rax
  __int64 v111; // r9
  __int64 v112; // rbx
  unsigned int *v113; // r13
  unsigned int *v114; // r12
  __int64 v115; // rdx
  unsigned int v116; // esi
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // [rsp-10h] [rbp-2C0h]
  char v120; // [rsp+18h] [rbp-298h]
  unsigned int v121; // [rsp+1Ch] [rbp-294h]
  __int64 v122; // [rsp+38h] [rbp-278h]
  __int64 v123; // [rsp+40h] [rbp-270h]
  __int64 v124; // [rsp+50h] [rbp-260h]
  __int64 v125; // [rsp+58h] [rbp-258h]
  __int64 v126; // [rsp+68h] [rbp-248h]
  unsigned int v127; // [rsp+70h] [rbp-240h]
  __int64 v128; // [rsp+78h] [rbp-238h]
  __int64 v130; // [rsp+80h] [rbp-230h]
  __int64 *v132; // [rsp+88h] [rbp-228h]
  __int64 *v133; // [rsp+90h] [rbp-220h]
  __int64 v134; // [rsp+98h] [rbp-218h]
  unsigned int v138; // [rsp+B0h] [rbp-200h]
  __int64 v139; // [rsp+B8h] [rbp-1F8h]
  __int64 v140; // [rsp+C0h] [rbp-1F0h]
  __int16 v141; // [rsp+D6h] [rbp-1DAh]
  __int64 v142; // [rsp+D8h] [rbp-1D8h]
  __int64 *v144; // [rsp+E0h] [rbp-1D0h]
  __int64 v145; // [rsp+E8h] [rbp-1C8h]
  __int64 *v146; // [rsp+F8h] [rbp-1B8h] BYREF
  _BYTE *v147[4]; // [rsp+100h] [rbp-1B0h] BYREF
  __int16 v148; // [rsp+120h] [rbp-190h]
  __int64 v149[4]; // [rsp+130h] [rbp-180h] BYREF
  __int16 v150; // [rsp+150h] [rbp-160h]
  __int64 *v151; // [rsp+160h] [rbp-150h] BYREF
  __int64 v152; // [rsp+168h] [rbp-148h]
  _BYTE v153[112]; // [rsp+170h] [rbp-140h] BYREF
  void *v154; // [rsp+1E0h] [rbp-D0h]
  char *v155; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v156; // [rsp+1F8h] [rbp-B8h]
  _BYTE v157[16]; // [rsp+200h] [rbp-B0h] BYREF
  char v158; // [rsp+210h] [rbp-A0h]
  char v159; // [rsp+211h] [rbp-9Fh]
  __int64 v160; // [rsp+220h] [rbp-90h]
  __int64 v161; // [rsp+228h] [rbp-88h]
  __int64 v162; // [rsp+230h] [rbp-80h]
  __int64 v163; // [rsp+238h] [rbp-78h]
  void **v164; // [rsp+240h] [rbp-70h]
  _QWORD *v165; // [rsp+248h] [rbp-68h]
  __int64 v166; // [rsp+250h] [rbp-60h]
  int v167; // [rsp+258h] [rbp-58h]
  __int16 v168; // [rsp+25Ch] [rbp-54h]
  char v169; // [rsp+25Eh] [rbp-52h]
  __int64 v170; // [rsp+260h] [rbp-50h]
  __int64 v171; // [rsp+268h] [rbp-48h]
  void *v172; // [rsp+270h] [rbp-40h] BYREF
  _QWORD v173[7]; // [rsp+278h] [rbp-38h] BYREF

  v11 = *(_DWORD *)(a4 + 32);
  if ( v11 <= 0x40 )
  {
    if ( !*(_QWORD *)(a4 + 24) )
      return;
  }
  else if ( v11 == (unsigned int)sub_C444A0(a4 + 24) )
  {
    return;
  }
  v12 = *(_QWORD **)(a1 + 40);
  v13 = v12[9];
  v133 = (__int64 *)sub_AA48A0((__int64)v12);
  v134 = sub_B2BEC0(v13);
  v146 = v133;
  v14 = sub_B8CD90(&v146, (__int64)"MemCopyDomain", 13, 0);
  v124 = sub_B8CD90(&v146, (__int64)"MemCopyAliasScope", 17, v14);
  v140 = *(_QWORD *)(a4 + 8);
  v127 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8;
  v121 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8;
  v15 = sub_DFDDE0(a10, v133, a4, v127, v121, a5, a6, a11);
  v139 = sub_BCB2B0(v133);
  v16 = sub_9208B0(v134, v15);
  v156 = v17;
  v155 = (char *)((unsigned __int64)(v16 + 7) >> 3);
  v18 = sub_CA1930(&v155);
  v19 = *(_DWORD *)(a4 + 32);
  v20 = v18;
  v21 = *(_QWORD *)(a4 + 24);
  if ( v19 > 0x40 )
    v21 = *(_QWORD *)v21;
  v126 = 0;
  v125 = v20;
  v145 = v20 * (v21 / v20);
  if ( v145 )
  {
    v155 = "memcpy-split";
    v159 = 1;
    v158 = 3;
    v61 = sub_AA8550(v12, (__int64 *)(a1 + 24), 0, (__int64)&v155, 0);
    v159 = 1;
    v126 = v61;
    v62 = v61;
    v155 = "load-store-loop";
    v158 = 3;
    v123 = sub_22077B0(0x50u);
    if ( v123 )
      sub_AA4D50(v123, (__int64)v133, (__int64)&v155, v13, v62);
    v63 = v12[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12 + 6 == (_QWORD *)v63 )
    {
      v65 = 0;
    }
    else
    {
      if ( !v63 )
        BUG();
      v64 = *(unsigned __int8 *)(v63 - 24);
      v65 = 0;
      v66 = (unsigned __int8 *)(v63 - 24);
      if ( (unsigned int)(v64 - 30) < 0xB )
        v65 = v66;
    }
    sub_B46F90(v65, 0, v123);
    v67 = v12[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v67 == v12 + 6 )
    {
      v69 = 0;
    }
    else
    {
      if ( !v67 )
        BUG();
      v68 = *(unsigned __int8 *)(v67 - 24);
      v69 = 0;
      v70 = v67 - 24;
      if ( (unsigned int)(v68 - 30) < 0xB )
        v69 = v70;
    }
    sub_23D0AB0((__int64)&v151, v69, 0, 0, 0);
    v120 = -1;
    v71 = v125 | (1LL << a6);
    if ( (v71 & -v71) != 0 )
    {
      _BitScanReverse64(&v72, v71 & -(v125 | (1LL << a6)));
      v120 = 63 - (v72 ^ 0x3F);
    }
    v73 = -1;
    v74 = (v125 | (1LL << a5)) & -(v125 | (1LL << a5));
    if ( v74 )
    {
      _BitScanReverse64(&v74, v74);
      v73 = 63 - (v74 ^ 0x3F);
    }
    v75 = sub_AA48A0(v123);
    v169 = 7;
    v163 = v75;
    v164 = &v172;
    v165 = v173;
    v168 = 512;
    v172 = &unk_49DA100;
    v155 = v157;
    LOWORD(v162) = 0;
    v173[0] = &unk_49DA0B0;
    v156 = 0x200000000LL;
    v161 = v123 + 48;
    v160 = v123;
    v166 = 0;
    v167 = 0;
    v170 = 0;
    v171 = 0;
    v149[0] = (__int64)"loop-index";
    v150 = 259;
    v76 = sub_D5C860((__int64 *)&v155, v140, 2, (__int64)v149);
    v77 = sub_AD64C0(v140, 0, 0);
    v78 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
    if ( v78 == *(_DWORD *)(v76 + 72) )
    {
      v142 = v77;
      sub_B48D90(v76);
      v77 = v142;
      v78 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
    }
    v79 = (v78 + 1) & 0x7FFFFFF;
    v80 = v79 | *(_DWORD *)(v76 + 4) & 0xF8000000;
    v81 = *(_QWORD *)(v76 - 8) + 32LL * (unsigned int)(v79 - 1);
    *(_DWORD *)(v76 + 4) = v80;
    if ( *(_QWORD *)v81 )
    {
      v82 = *(_QWORD *)(v81 + 8);
      **(_QWORD **)(v81 + 16) = v82;
      if ( v82 )
        *(_QWORD *)(v82 + 16) = *(_QWORD *)(v81 + 16);
    }
    *(_QWORD *)v81 = v77;
    if ( v77 )
    {
      v83 = *(_QWORD *)(v77 + 16);
      *(_QWORD *)(v81 + 8) = v83;
      if ( v83 )
        *(_QWORD *)(v83 + 16) = v81 + 8;
      *(_QWORD *)(v81 + 16) = v77 + 16;
      *(_QWORD *)(v77 + 16) = v81;
    }
    *(_QWORD *)(*(_QWORD *)(v76 - 8)
              + 32LL * *(unsigned int *)(v76 + 72)
              + 8LL * ((*(_DWORD *)(v76 + 4) & 0x7FFFFFFu) - 1)) = v12;
    v150 = 257;
    v147[0] = (_BYTE *)v76;
    v84 = sub_921130((unsigned int **)&v155, v139, a2, v147, 1, (__int64)v149, 3u);
    v148 = 257;
    v150 = 257;
    v85 = sub_BD2C40(80, 1u);
    v122 = (__int64)v85;
    if ( v85 )
      sub_B4D190((__int64)v85, v15, v84, (__int64)v149, a7, v73, 0, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, _BYTE **, __int64, __int64))(*v165 + 16LL))(v165, v122, v147, v161, v162);
    v86 = (unsigned int *)v155;
    v87 = (unsigned int *)&v155[16 * (unsigned int)v156];
    if ( v155 != (char *)v87 )
    {
      do
      {
        v88 = *((_QWORD *)v86 + 1);
        v89 = *v86;
        v86 += 4;
        sub_B99FD0(v122, v89, v88);
      }
      while ( v87 != v86 );
    }
    if ( !a9 )
    {
      v149[0] = v124;
      v118 = sub_B9C770(v133, v149, (__int64 *)1, 0, 1);
      sub_B99FD0(v122, 7u, v118);
    }
    v150 = 257;
    v147[0] = (_BYTE *)v76;
    v90 = sub_921130((unsigned int **)&v155, v139, a3, v147, 1, (__int64)v149, 3u);
    v150 = 257;
    v92 = sub_BD2C40(80, unk_3F10A10);
    if ( v92 )
      sub_B4D3C0((__int64)v92, v122, v90, a8, v120, v91, 0, 0);
    (*(void (__fastcall **)(_QWORD *, _QWORD *, __int64 *, __int64, __int64))(*v165 + 16LL))(
      v165,
      v92,
      v149,
      v161,
      v162);
    v93 = (unsigned int *)&v155[16 * (unsigned int)v156];
    v94 = (unsigned int *)v155;
    if ( v155 != (char *)v93 )
    {
      do
      {
        v95 = *((_QWORD *)v94 + 1);
        v96 = *v94;
        v94 += 4;
        sub_B99FD0((__int64)v92, v96, v95);
      }
      while ( v93 != v94 );
    }
    if ( !a9 )
    {
      v149[0] = v124;
      v117 = sub_B9C770(v133, v149, (__int64 *)1, 0, 1);
      sub_B99FD0((__int64)v92, 8u, v117);
    }
    if ( BYTE4(a11) )
    {
      v97 = *(_WORD *)(v122 + 2);
      *(_BYTE *)(v122 + 72) = 1;
      v97 &= 0xFC7Fu;
      LOBYTE(v97) = v97 | 0x80;
      *(_WORD *)(v122 + 2) = v97;
      v98 = *((_WORD *)v92 + 1);
      *((_BYTE *)v92 + 72) = 1;
      v98 &= 0xFC7Fu;
      LOBYTE(v98) = v98 | 0x80;
      *((_WORD *)v92 + 1) = v98;
    }
    v150 = 257;
    v99 = (_BYTE *)sub_AD64C0(v140, v125, 0);
    v100 = sub_929C50((unsigned int **)&v155, (_BYTE *)v76, v99, (__int64)v149, 0, 0);
    v101 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
    if ( v101 == *(_DWORD *)(v76 + 72) )
    {
      sub_B48D90(v76);
      v101 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
    }
    v102 = (v101 + 1) & 0x7FFFFFF;
    v103 = v102 | *(_DWORD *)(v76 + 4) & 0xF8000000;
    v104 = *(_QWORD *)(v76 - 8) + 32LL * (unsigned int)(v102 - 1);
    *(_DWORD *)(v76 + 4) = v103;
    if ( *(_QWORD *)v104 )
    {
      v105 = *(_QWORD *)(v104 + 8);
      **(_QWORD **)(v104 + 16) = v105;
      if ( v105 )
        *(_QWORD *)(v105 + 16) = *(_QWORD *)(v104 + 16);
    }
    *(_QWORD *)v104 = v100;
    if ( v100 )
    {
      v106 = *(_QWORD *)(v100 + 16);
      *(_QWORD *)(v104 + 8) = v106;
      if ( v106 )
        *(_QWORD *)(v106 + 16) = v104 + 8;
      *(_QWORD *)(v104 + 16) = v100 + 16;
      *(_QWORD *)(v100 + 16) = v104;
    }
    *(_QWORD *)(*(_QWORD *)(v76 - 8)
              + 32LL * *(unsigned int *)(v76 + 72)
              + 8LL * ((*(_DWORD *)(v76 + 4) & 0x7FFFFFFu) - 1)) = v123;
    v107 = (_BYTE *)sub_AD64C0(v140, v145, 0);
    v148 = 257;
    v108 = sub_92B530((unsigned int **)&v155, 0x24u, v100, v107, (__int64)v147);
    v150 = 257;
    v109 = v108;
    v110 = sub_BD2C40(72, 3u);
    v112 = (__int64)v110;
    if ( v110 )
      sub_B4C9A0((__int64)v110, v123, v126, v109, 3u, v111, 0, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v165 + 16LL))(
      v165,
      v112,
      v149,
      v161,
      v162);
    v113 = (unsigned int *)v155;
    v114 = (unsigned int *)&v155[16 * (unsigned int)v156];
    if ( v155 != (char *)v114 )
    {
      do
      {
        v115 = *((_QWORD *)v113 + 1);
        v116 = *v113;
        v113 += 4;
        sub_B99FD0(v112, v116, v115);
      }
      while ( v114 != v113 );
    }
    nullsub_61();
    v172 = &unk_49DA100;
    nullsub_63();
    if ( v155 != v157 )
      _libc_free((unsigned __int64)v155);
    nullsub_61();
    v154 = &unk_49DA100;
    nullsub_63();
    if ( v151 != (__int64 *)v153 )
      _libc_free((unsigned __int64)v151);
    v19 = *(_DWORD *)(a4 + 32);
  }
  v22 = *(_QWORD **)(a4 + 24);
  if ( v19 > 0x40 )
    v22 = (_QWORD *)*v22;
  v23 = (char *)v22 - v145;
  if ( v23 )
  {
    if ( v126 )
    {
      v25 = sub_AA4FF0(v126);
      if ( !v25 )
        BUG();
      v26 = v24;
      v27 = 1;
    }
    else
    {
      v26 = 0;
      v27 = 0;
      v25 = a1 + 24;
    }
    v28 = *(_QWORD *)(v25 + 16);
    LOBYTE(v29) = v27;
    HIBYTE(v29) = v26;
    v30 = v29;
    v31 = sub_AA48A0(v28);
    v169 = 7;
    v163 = v31;
    v164 = &v172;
    v165 = v173;
    v155 = v157;
    v172 = &unk_49DA100;
    v156 = 0x200000000LL;
    v168 = 512;
    LOWORD(v162) = 0;
    v173[0] = &unk_49DA0B0;
    v166 = 0;
    v167 = 0;
    v170 = 0;
    v171 = 0;
    v160 = 0;
    v161 = 0;
    sub_A88F30((__int64)&v155, v28, v25, v30);
    v32 = a6;
    v151 = (__int64 *)v153;
    v152 = 0x500000000LL;
    sub_DFDE40(a10, (__int64)&v151, v133, (int)v23, v127, v121, a5, a6, a11);
    v33 = v151;
    v132 = &v151[(unsigned int)v152];
    if ( v132 != v151 )
    {
      v144 = v151;
      v128 = 1LL << a5;
      v130 = 1LL << v32;
      do
      {
        v34 = -1;
        v35 = *v144;
        if ( ((v145 | v128) & -(v145 | v128)) != 0 )
        {
          _BitScanReverse64(&v36, (v145 | v128) & -(v145 | v128));
          v34 = 63 - (v36 ^ 0x3F);
        }
        v37 = -1;
        if ( ((v145 | v130) & -(v145 | v130)) != 0 )
        {
          _BitScanReverse64(&v38, (v145 | v130) & -(v145 | v130));
          v37 = 63 - (v38 ^ 0x3F);
        }
        v39 = sub_9208B0(v134, *v144);
        v149[1] = v40;
        v149[0] = (unsigned __int64)(v39 + 7) >> 3;
        v138 = sub_CA1930(v149);
        v150 = 257;
        v147[0] = (_BYTE *)sub_AD64C0(v140, v145, 0);
        v41 = sub_921130((unsigned int **)&v155, v139, a2, v147, 1, (__int64)v149, 3u);
        v148 = 257;
        HIBYTE(v42) = HIBYTE(v141);
        LOBYTE(v42) = v34;
        v141 = v42;
        v150 = 257;
        v43 = sub_BD2C40(80, 1u);
        if ( v43 )
        {
          sub_B4D190((__int64)v43, v35, v41, (__int64)v149, a7, v141, 0, 0);
          v44 = v119;
        }
        (*(void (__fastcall **)(_QWORD *, _QWORD *, _BYTE **, __int64, __int64, __int64))(*v165 + 16LL))(
          v165,
          v43,
          v147,
          v161,
          v162,
          v44);
        v45 = (unsigned int *)v155;
        v46 = (unsigned int *)&v155[16 * (unsigned int)v156];
        if ( v155 != (char *)v46 )
        {
          do
          {
            v47 = *((_QWORD *)v45 + 1);
            v48 = *v45;
            v45 += 4;
            sub_B99FD0((__int64)v43, v48, v47);
          }
          while ( v46 != v45 );
        }
        if ( !a9 )
        {
          v149[0] = v124;
          v60 = sub_B9C770(v133, v149, (__int64 *)1, 0, 1);
          sub_B99FD0((__int64)v43, 7u, v60);
        }
        v150 = 257;
        v147[0] = (_BYTE *)sub_AD64C0(v140, v145, 0);
        v49 = sub_921130((unsigned int **)&v155, v139, a3, v147, 1, (__int64)v149, 3u);
        v150 = 257;
        v50 = sub_BD2C40(80, unk_3F10A10);
        v52 = (__int64)v50;
        if ( v50 )
          sub_B4D3C0((__int64)v50, (__int64)v43, v49, a8, v37, v51, 0, 0);
        (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v165 + 16LL))(
          v165,
          v52,
          v149,
          v161,
          v162);
        v53 = (unsigned int *)v155;
        v54 = (unsigned int *)&v155[16 * (unsigned int)v156];
        if ( v155 != (char *)v54 )
        {
          do
          {
            v55 = *((_QWORD *)v53 + 1);
            v56 = *v53;
            v53 += 4;
            sub_B99FD0(v52, v56, v55);
          }
          while ( v54 != v53 );
        }
        if ( !a9 )
        {
          v149[0] = v124;
          v59 = sub_B9C770(v133, v149, (__int64 *)1, 0, 1);
          sub_B99FD0(v52, 8u, v59);
        }
        if ( BYTE4(a11) )
        {
          v57 = *((_WORD *)v43 + 1);
          *((_BYTE *)v43 + 72) = 1;
          v57 &= 0xFC7Fu;
          LOBYTE(v57) = v57 | 0x80;
          *((_WORD *)v43 + 1) = v57;
          v58 = *(_WORD *)(v52 + 2);
          *(_BYTE *)(v52 + 72) = 1;
          v58 &= 0xFC7Fu;
          LOBYTE(v58) = v58 | 0x80;
          *(_WORD *)(v52 + 2) = v58;
        }
        ++v144;
        v145 += v138;
      }
      while ( v132 != v144 );
      v33 = v151;
    }
    if ( v33 != (__int64 *)v153 )
      _libc_free((unsigned __int64)v33);
    nullsub_61();
    v172 = &unk_49DA100;
    nullsub_63();
    if ( v155 != v157 )
      _libc_free((unsigned __int64)v155);
  }
}
