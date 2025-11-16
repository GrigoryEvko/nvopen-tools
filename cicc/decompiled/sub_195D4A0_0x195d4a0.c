// Function: sub_195D4A0
// Address: 0x195d4a0
//
__int64 __fastcall sub_195D4A0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  int v26; // edx
  __int64 v27; // r13
  __int64 v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 *v31; // r13
  unsigned int v32; // r12d
  __int64 v33; // r14
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  _QWORD *v36; // r13
  _QWORD *v37; // rsi
  __int64 v38; // rax
  __int64 v40; // rax
  __m128i *v41; // rdx
  __int64 v42; // r13
  __m128i si128; // xmm0
  const char *v44; // rax
  size_t v45; // rdx
  _BYTE *v46; // rdi
  char *v47; // rsi
  unsigned __int64 v48; // rax
  __int64 v49; // rax
  _QWORD *v50; // r12
  _QWORD *v51; // r13
  __int64 v52; // r14
  __int64 v53; // rdi
  _QWORD *v54; // rax
  _QWORD *v55; // rax
  __int64 *v56; // rax
  __int64 *v57; // r15
  __int64 *v58; // r12
  __int64 v59; // r13
  __int64 *v60; // rbx
  __int64 *v61; // r14
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // rax
  void *v65; // rdi
  unsigned int v66; // eax
  __int64 v67; // rdx
  unsigned __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rdi
  unsigned __int64 *v72; // r12
  unsigned __int64 *v73; // rbx
  unsigned __int64 v74; // rdi
  unsigned __int64 *v75; // r12
  unsigned __int64 *v76; // rdx
  unsigned __int64 *v77; // rbx
  unsigned __int64 *v78; // r12
  unsigned __int64 v79; // rdi
  unsigned __int64 *v80; // rbx
  unsigned __int64 v81; // rdi
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // [rsp+8h] [rbp-308h]
  __int64 v85; // [rsp+10h] [rbp-300h]
  char v86; // [rsp+27h] [rbp-2E9h]
  __int64 v87; // [rsp+28h] [rbp-2E8h]
  __int64 *v88; // [rsp+30h] [rbp-2E0h]
  _QWORD *v89; // [rsp+38h] [rbp-2D8h]
  __int64 *v90; // [rsp+40h] [rbp-2D0h]
  size_t v92; // [rsp+50h] [rbp-2C0h]
  __int64 v93; // [rsp+58h] [rbp-2B8h]
  __int64 v94; // [rsp+68h] [rbp-2A8h] BYREF
  __int64 *v95; // [rsp+70h] [rbp-2A0h] BYREF
  __int64 v96; // [rsp+78h] [rbp-298h] BYREF
  __int64 v97; // [rsp+80h] [rbp-290h] BYREF
  __int64 v98; // [rsp+88h] [rbp-288h]
  _QWORD *v99; // [rsp+90h] [rbp-280h]
  __int64 v100; // [rsp+98h] [rbp-278h]
  unsigned int v101; // [rsp+A0h] [rbp-270h]
  __int64 v102; // [rsp+B0h] [rbp-260h]
  char v103; // [rsp+B8h] [rbp-258h]
  int v104; // [rsp+BCh] [rbp-254h]
  __int64 *v105; // [rsp+C0h] [rbp-250h] BYREF
  _QWORD v106[2]; // [rsp+C8h] [rbp-248h] BYREF
  __int64 v107; // [rsp+D8h] [rbp-238h]
  __int64 *v108; // [rsp+E0h] [rbp-230h]
  __int64 *v109; // [rsp+E8h] [rbp-228h]
  __int64 v110; // [rsp+F0h] [rbp-220h]
  unsigned __int64 v111; // [rsp+F8h] [rbp-218h]
  unsigned __int64 v112; // [rsp+100h] [rbp-210h]
  unsigned __int64 *v113; // [rsp+108h] [rbp-208h]
  unsigned int v114; // [rsp+110h] [rbp-200h]
  char v115; // [rsp+118h] [rbp-1F8h] BYREF
  unsigned __int64 *v116; // [rsp+138h] [rbp-1D8h]
  unsigned int v117; // [rsp+140h] [rbp-1D0h]
  __int64 v118; // [rsp+148h] [rbp-1C8h] BYREF
  __int64 v119; // [rsp+160h] [rbp-1B0h] BYREF
  _BYTE *v120; // [rsp+168h] [rbp-1A8h]
  __int64 v121; // [rsp+170h] [rbp-1A0h]
  _BYTE v122[256]; // [rsp+178h] [rbp-198h] BYREF
  __int64 v123; // [rsp+278h] [rbp-98h]
  _BYTE *v124; // [rsp+280h] [rbp-90h]
  _BYTE *v125; // [rsp+288h] [rbp-88h]
  __int64 v126; // [rsp+290h] [rbp-80h]
  int v127; // [rsp+298h] [rbp-78h]
  _BYTE v128[112]; // [rsp+2A0h] [rbp-70h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v93 = a2;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_120:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_120;
  }
  v89 = (_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
                     *(_QWORD *)(v11 + 8),
                     &unk_4F9B6E8)
                 + 360);
  v13 = *(__int64 **)(a1 + 8);
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_123:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9E06C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_123;
  }
  v85 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9E06C)
      + 160;
  v16 = *(__int64 **)(a1 + 8);
  v17 = *v16;
  v18 = v16[1];
  if ( v17 == v18 )
LABEL_122:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F99130 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_122;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F99130);
  v90 = (__int64 *)sub_13EB2D0(v19);
  v20 = *(__int64 **)(a1 + 8);
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
LABEL_121:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4F96DB4 )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_121;
  }
  v23 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
                      *(_QWORD *)(v21 + 8),
                      &unk_4F96DB4)
                  + 160);
  v126 = 8;
  v123 = 0;
  v87 = v23;
  v127 = 0;
  v119 = v85;
  v120 = v122;
  v121 = 0x1000000000LL;
  v124 = v128;
  v125 = v128;
  sub_15E44B0(a2);
  v86 = v26 != 0;
  if ( !v26 )
  {
    v27 = 0;
    v88 = 0;
    goto LABEL_19;
  }
  v98 = 0;
  v96 = 0x100000000LL;
  v95 = &v97;
  v102 = a2;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v103 = 0;
  v104 = 0;
  sub_15D3930((__int64)&v95);
  sub_14019E0((__int64)&v105, (__int64)&v95);
  if ( v101 )
  {
    v50 = v99;
    v51 = &v99[2 * v101];
    do
    {
      if ( *v50 != -16 && *v50 != -8 )
      {
        v52 = v50[1];
        if ( v52 )
        {
          v53 = *(_QWORD *)(v52 + 24);
          if ( v53 )
            j_j___libc_free_0(v53, *(_QWORD *)(v52 + 40) - v53);
          j_j___libc_free_0(v52, 56);
        }
      }
      v50 += 2;
    }
    while ( v51 != v50 );
  }
  j___libc_free_0(v99);
  if ( v95 != &v97 )
    _libc_free((unsigned __int64)v95);
  v54 = (_QWORD *)sub_22077B0(408);
  v27 = (__int64)v54;
  if ( v54 )
  {
    *v54 = 0;
    v55 = v54 + 14;
    *(v55 - 13) = 0;
    *(v55 - 12) = 0;
    *((_DWORD *)v55 - 22) = 0;
    *(v55 - 10) = 0;
    *(v55 - 9) = 0;
    *(v55 - 8) = 0;
    *((_DWORD *)v55 - 14) = 0;
    *(_QWORD *)(v27 + 80) = v55;
    *(_QWORD *)(v27 + 88) = v55;
    *(_QWORD *)(v27 + 72) = 0;
    *(_QWORD *)(v27 + 96) = 16;
    *(_DWORD *)(v27 + 104) = 0;
    *(_QWORD *)(v27 + 240) = 0;
    *(_QWORD *)(v27 + 248) = v27 + 280;
    *(_QWORD *)(v27 + 256) = v27 + 280;
    *(_QWORD *)(v27 + 264) = 16;
    *(_DWORD *)(v27 + 272) = 0;
    sub_137CAE0(v27, (__int64 *)a2, (__int64)&v105, v89);
  }
  v56 = (__int64 *)sub_22077B0(8);
  v88 = v56;
  if ( v56 )
    sub_13702A0(v56, (const void *)a2, v27, (__int64)&v105);
  sub_142D890((__int64)&v105);
  v57 = v109;
  v58 = v108;
  if ( v108 != v109 )
  {
    v84 = v27;
    do
    {
      v59 = *v58;
      v60 = *(__int64 **)(*v58 + 8);
      v61 = *(__int64 **)(*v58 + 16);
      if ( v60 == v61 )
      {
        *(_BYTE *)(v59 + 160) = 1;
      }
      else
      {
        do
        {
          v62 = *v60++;
          sub_13FACC0(v62);
        }
        while ( v61 != v60 );
        *(_BYTE *)(v59 + 160) = 1;
        v63 = *(_QWORD *)(v59 + 8);
        if ( *(_QWORD *)(v59 + 16) != v63 )
          *(_QWORD *)(v59 + 16) = v63;
      }
      v64 = *(_QWORD *)(v59 + 32);
      if ( v64 != *(_QWORD *)(v59 + 40) )
        *(_QWORD *)(v59 + 40) = v64;
      ++*(_QWORD *)(v59 + 56);
      v65 = *(void **)(v59 + 72);
      if ( v65 == *(void **)(v59 + 64) )
      {
        *(_QWORD *)v59 = 0;
      }
      else
      {
        v66 = 4 * (*(_DWORD *)(v59 + 84) - *(_DWORD *)(v59 + 88));
        v67 = *(unsigned int *)(v59 + 80);
        if ( v66 < 0x20 )
          v66 = 32;
        if ( (unsigned int)v67 > v66 )
          sub_16CC920(v59 + 56);
        else
          memset(v65, -1, 8 * v67);
        v68 = *(_QWORD *)(v59 + 72);
        v69 = *(_QWORD *)(v59 + 64);
        *(_QWORD *)v59 = 0;
        if ( v68 != v69 )
          _libc_free(v68);
      }
      v70 = *(_QWORD *)(v59 + 32);
      if ( v70 )
        j_j___libc_free_0(v70, *(_QWORD *)(v59 + 48) - v70);
      v71 = *(_QWORD *)(v59 + 8);
      if ( v71 )
        j_j___libc_free_0(v71, *(_QWORD *)(v59 + 24) - v71);
      ++v58;
    }
    while ( v57 != v58 );
    v27 = v84;
    if ( v108 != v109 )
      v109 = v108;
  }
  v72 = v116;
  v73 = &v116[2 * v117];
  if ( v116 != v73 )
  {
    do
    {
      v74 = *v72;
      v72 += 2;
      _libc_free(v74);
    }
    while ( v73 != v72 );
  }
  v117 = 0;
  if ( v114 )
  {
    v76 = v113;
    v118 = 0;
    v77 = &v113[v114];
    v78 = v113 + 1;
    v111 = *v113;
    v112 = v111 + 4096;
    if ( v77 != v113 + 1 )
    {
      do
      {
        v79 = *v78++;
        _libc_free(v79);
      }
      while ( v77 != v78 );
      v76 = v113;
    }
    v114 = 1;
    _libc_free(*v76);
    v80 = v116;
    v75 = &v116[2 * v117];
    if ( v116 == v75 )
      goto LABEL_94;
    do
    {
      v81 = *v80;
      v80 += 2;
      _libc_free(v81);
    }
    while ( v75 != v80 );
  }
  v75 = v116;
LABEL_94:
  if ( v75 != (unsigned __int64 *)&v118 )
    _libc_free((unsigned __int64)v75);
  if ( v113 != (unsigned __int64 *)&v115 )
    _libc_free((unsigned __int64)v113);
  if ( v108 )
    j_j___libc_free_0(v108, v110 - (_QWORD)v108);
  j___libc_free_0(v106[0]);
LABEL_19:
  v94 = v27;
  v105 = v88;
  v28 = a1 + 160;
  v29 = sub_195CAE0(
          a1 + 160,
          a2,
          (__int64)v89,
          (__int64)v90,
          v87,
          (__int64)&v119,
          a3,
          a4,
          a5,
          a6,
          v24,
          v25,
          a9,
          a10,
          v86,
          (__int64 *)&v105,
          &v94);
  v31 = v105;
  v32 = v29;
  if ( v105 )
  {
    sub_1368A00(v105);
    a2 = 8;
    v28 = (__int64)v31;
    j_j___libc_free_0(v31, 8);
  }
  v33 = v94;
  if ( v94 )
  {
    v34 = *(_QWORD *)(v94 + 256);
    if ( v34 != *(_QWORD *)(v94 + 248) )
      _libc_free(v34);
    v35 = *(_QWORD *)(v33 + 88);
    if ( v35 != *(_QWORD *)(v33 + 80) )
      _libc_free(v35);
    j___libc_free_0(*(_QWORD *)(v33 + 40));
    if ( *(_DWORD *)(v33 + 24) )
    {
      v96 = 2;
      v97 = 0;
      v98 = -8;
      v95 = (__int64 *)&unk_49E8A80;
      v99 = 0;
      v106[0] = 2;
      v106[1] = 0;
      v107 = -16;
      v105 = (__int64 *)&unk_49E8A80;
      v108 = 0;
      v36 = *(_QWORD **)(v33 + 8);
      v37 = &v36[5 * *(unsigned int *)(v33 + 24)];
      if ( v36 != v37 )
      {
        do
        {
          v38 = v36[3];
          *v36 = &unk_49EE2B0;
          if ( v38 != -8 && v38 != 0 && v38 != -16 )
            sub_1649B30(v36 + 1);
          v36 += 5;
        }
        while ( v37 != v36 );
        v32 = (unsigned __int8)v32;
        v105 = (__int64 *)&unk_49EE2B0;
        if ( v107 != -16 && v107 != -8 && v107 )
          sub_1649B30(v106);
      }
      v95 = (__int64 *)&unk_49EE2B0;
      if ( v98 != 0 && v98 != -8 && v98 != -16 )
        sub_1649B30(&v96);
    }
    j___libc_free_0(*(_QWORD *)(v33 + 8));
    a2 = 408;
    v28 = v33;
    j_j___libc_free_0(v33, 408);
  }
  if ( byte_4FB00E0 )
  {
    v40 = sub_16BA580(v28, a2, v30);
    v41 = *(__m128i **)(v40 + 24);
    v42 = v40;
    if ( *(_QWORD *)(v40 + 16) - (_QWORD)v41 <= 0x11u )
    {
      v42 = sub_16E7EE0(v40, "LVI for function '", 0x12u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_428A440);
      v41[1].m128i_i16[0] = 10016;
      *v41 = si128;
      *(_QWORD *)(v40 + 24) += 18LL;
    }
    v44 = sub_1649960(v93);
    v46 = *(_BYTE **)(v42 + 24);
    v47 = (char *)v44;
    v48 = *(_QWORD *)(v42 + 16) - (_QWORD)v46;
    if ( v45 > v48 )
    {
      v83 = sub_16E7EE0(v42, v47, v45);
      v46 = *(_BYTE **)(v83 + 24);
      v42 = v83;
      v48 = *(_QWORD *)(v83 + 16) - (_QWORD)v46;
    }
    else if ( v45 )
    {
      v92 = v45;
      memcpy(v46, v47, v45);
      v82 = *(_QWORD *)(v42 + 16);
      v45 = *(_QWORD *)(v42 + 24) + v92;
      *(_QWORD *)(v42 + 24) = v45;
      v46 = (_BYTE *)v45;
      v48 = v82 - v45;
    }
    if ( v48 <= 2 )
    {
      v47 = "':\n";
      v46 = (_BYTE *)v42;
      sub_16E7EE0(v42, "':\n", 3u);
    }
    else
    {
      v46[2] = 10;
      *(_WORD *)v46 = 14887;
      *(_QWORD *)(v42 + 24) += 3LL;
    }
    v49 = sub_16BA580((__int64)v46, (__int64)v47, v45);
    sub_13EB990(v90, v93, v85, v49);
  }
  if ( v125 != v124 )
    _libc_free((unsigned __int64)v125);
  if ( v120 != v122 )
    _libc_free((unsigned __int64)v120);
  return v32;
}
