// Function: sub_30462A0
// Address: 0x30462a0
//
__int64 __fastcall sub_30462A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rax
  int v12; // esi
  bool v13; // cc
  _QWORD *v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // r12
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned int v25; // edx
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rsi
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r9
  unsigned __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __m128i v38; // xmm0
  const __m128i *v39; // rdx
  __int64 v40; // rax
  __m128i v41; // xmm0
  const __m128i *v42; // rdx
  __int64 v43; // rax
  __m128i v44; // xmm0
  __int64 v45; // rcx
  int v46; // r8d
  __int64 v47; // rax
  unsigned int v48; // edx
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // rax
  _QWORD *v52; // rsi
  __int64 v53; // r8
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r9
  unsigned __int64 v57; // rdx
  __int64 *v58; // rax
  const __m128i *v59; // rdx
  __int64 v60; // rax
  __m128i v61; // xmm0
  __int64 v62; // rcx
  int v63; // r8d
  unsigned int v64; // edx
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // rsi
  __int64 v69; // r8
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // r9
  unsigned __int64 v73; // rdx
  __int64 *v74; // rax
  const __m128i *v75; // rdx
  __int64 v76; // rax
  __m128i v77; // xmm0
  __int64 v78; // rcx
  int v79; // r8d
  unsigned int v80; // edx
  __int64 v81; // r8
  __int64 v82; // rdx
  __int64 v83; // rcx
  _QWORD *v84; // rax
  __int32 v85; // r10d
  __int64 v86; // rax
  _QWORD *v87; // rsi
  __int64 v88; // r8
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r9
  __int32 v92; // r10d
  unsigned __int64 v93; // rdx
  __int64 *v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rax
  __m128i v97; // xmm0
  const __m128i *v98; // rdx
  __int64 v99; // rax
  __m128i v100; // xmm0
  const __m128i *v101; // rdx
  __int64 v102; // rax
  __m128i v103; // xmm0
  __int64 v104; // rcx
  int v105; // r8d
  __int64 v106; // rdi
  unsigned int v107; // eax
  unsigned int v108; // edx
  __int64 v109; // r8
  __int64 v110; // rax
  __int64 v111; // rax
  _QWORD *v112; // rsi
  __int64 v113; // r8
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // r9
  unsigned __int64 v117; // rdx
  __int64 *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rax
  __m128i v121; // xmm0
  const __m128i *v122; // rdx
  __int64 v123; // rax
  __m128i v124; // xmm0
  const __m128i *v125; // rdx
  __int64 v126; // rax
  __m128i v127; // xmm0
  __int64 v128; // rcx
  int v129; // r8d
  __int64 *v130; // rdi
  __int64 v131; // r8
  unsigned int v132; // eax
  __int64 v133; // rdi
  unsigned int v134; // eax
  unsigned int v135; // edx
  __int64 v136; // r8
  __int64 v137; // rax
  __int64 v138; // rdx
  _QWORD *v139; // rax
  unsigned int v140; // r14d
  unsigned int v141; // ebx
  __int64 v142; // rax
  int v143; // ecx
  __int64 v144; // rax
  int v145; // r11d
  __int64 v146; // rax
  _QWORD *v147; // rsi
  __int64 v148; // r8
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // r9
  __int32 v152; // r11d
  unsigned __int64 v153; // rdx
  __int64 *v154; // rax
  __int64 v155; // rdx
  __int64 v156; // rax
  __m128i v157; // xmm0
  __int64 v158; // rax
  __m128i v159; // xmm0
  __int64 v160; // rcx
  int v161; // r8d
  unsigned int v162; // edx
  __int64 v163; // r8
  __int64 v164; // rax
  __int64 v165; // rax
  _QWORD *v166; // rsi
  __int64 v167; // r8
  __int64 v168; // rax
  __int64 v169; // rdx
  __int64 v170; // r9
  unsigned __int64 v171; // rdx
  __int64 *v172; // rax
  const __m128i *v173; // rdx
  __int64 v174; // rax
  __m128i v175; // xmm0
  __int64 v176; // rcx
  int v177; // r8d
  __m128i v178; // xmm0
  __int64 v179; // rax
  _DWORD *v180; // rax
  __int32 v181; // [rsp+0h] [rbp-D0h]
  __int64 v182; // [rsp+0h] [rbp-D0h]
  __int64 v183; // [rsp+8h] [rbp-C8h]
  __m128i v184; // [rsp+10h] [rbp-C0h] BYREF
  __m128i v185; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v186; // [rsp+30h] [rbp-A0h] BYREF
  int v187; // [rsp+38h] [rbp-98h]
  __int64 v188; // [rsp+40h] [rbp-90h] BYREF
  int v189; // [rsp+48h] [rbp-88h]
  _BYTE *v190; // [rsp+50h] [rbp-80h] BYREF
  __int64 v191; // [rsp+58h] [rbp-78h]
  _BYTE v192[112]; // [rsp+60h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *(_QWORD *)(v8 + 40);
  v186 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v186, v9, 1);
  v11 = *(_QWORD *)(v10 + 96);
  v12 = *(_DWORD *)(a2 + 72);
  v13 = *(_DWORD *)(v11 + 32) <= 0x40u;
  v14 = *(_QWORD **)(v11 + 24);
  v187 = v12;
  if ( !v13 )
    v14 = (_QWORD *)*v14;
  if ( (_DWORD)v14 == 10143 )
  {
    v133 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
    v134 = *(_DWORD *)(v133 + 340);
    if ( v134 > 0x408 )
    {
      if ( v134 - 1101 > 1 )
        goto LABEL_125;
    }
    else if ( v134 <= 0x3E8 || ((1LL << ((unsigned __int8)v134 + 23)) & 0xC0000C03) == 0 )
    {
      goto LABEL_125;
    }
    v162 = *(_DWORD *)(v133 + 336);
    if ( __ROR4__(-858993459 * v134 + 1717986918, 1) <= 0x19999999u && v162 > 0x57 || v162 > 0x55 )
    {
      v163 = *(_QWORD *)(a2 + 80);
      v188 = v163;
      if ( v163 )
      {
        sub_B96E90((__int64)&v188, v163, 1);
        v12 = *(_DWORD *)(a2 + 72);
      }
      v164 = *(_QWORD *)(a2 + 40);
      v189 = v12;
      v165 = *(_QWORD *)(*(_QWORD *)(v164 + 80) + 96LL);
      v166 = *(_QWORD **)(v165 + 24);
      if ( *(_DWORD *)(v165 + 32) > 0x40u )
        v166 = (_QWORD *)*v166;
      v190 = v192;
      v191 = 0x200000000LL;
      v167 = sub_3400BD0(a4, (_DWORD)v166, (unsigned int)&v188, 7, 0, 1, 0);
      v168 = (unsigned int)v191;
      v170 = v169;
      v171 = (unsigned int)v191 + 1LL;
      if ( v171 > HIDWORD(v191) )
      {
        v185.m128i_i64[0] = v167;
        v185.m128i_i64[1] = v170;
        sub_C8D5F0((__int64)&v190, v192, v171, 0x10u, v167, v170);
        v168 = (unsigned int)v191;
        v170 = v185.m128i_i64[1];
        v167 = v185.m128i_i64[0];
      }
      v172 = (__int64 *)&v190[16 * v168];
      *v172 = v167;
      v172[1] = v170;
      v173 = *(const __m128i **)(a2 + 40);
      LODWORD(v191) = v191 + 1;
      v174 = (unsigned int)v191;
      v175 = _mm_loadu_si128(v173);
      if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
      {
        v185 = v175;
        sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v167, v170);
        v174 = (unsigned int)v191;
        v175 = _mm_load_si128(&v185);
      }
      *(__m128i *)&v190[16 * v174] = v175;
      v176 = *(_QWORD *)(a2 + 48);
      v177 = *(_DWORD *)(a2 + 68);
      LODWORD(v191) = v191 + 1;
      v47 = sub_33E66D0(a4, 4830, (unsigned int)&v188, v176, v177, v170, (__int64)v190, (unsigned int)v191);
      goto LABEL_164;
    }
LABEL_125:
    sub_C64ED0(
      "tcgen05.fence supported only on arch-conditional or family-conditional variants from SM100 onwards.",
      1u);
  }
  if ( (unsigned int)v14 > 0x279F )
  {
    if ( (unsigned int)v14 <= 0x286E )
    {
      if ( (unsigned int)v14 > 0x2849 )
        goto LABEL_35;
      if ( (unsigned int)v14 <= 0x27EB )
      {
        if ( (unsigned int)v14 <= 0x27C6 )
          goto LABEL_18;
LABEL_35:
        v17 = sub_3030470(a2, a4, (__int64)v14, a4, a5, a6);
        goto LABEL_19;
      }
      if ( (_DWORD)v14 != 10311 )
        goto LABEL_18;
      v19 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
      v20 = *(_DWORD *)(v19 + 340);
      if ( v20 > 0x408 )
      {
        if ( v20 - 1101 > 1 )
          goto LABEL_25;
      }
      else if ( v20 <= 0x3E8 || ((1LL << ((unsigned __int8)v20 + 23)) & 0xC0000C03) == 0 )
      {
        goto LABEL_25;
      }
      v48 = *(_DWORD *)(v19 + 336);
      if ( __ROR4__(-858993459 * v20 + 1717986918, 1) <= 0x19999999u && v48 > 0x57 || v48 > 0x55 )
      {
        v49 = *(_QWORD *)(a2 + 80);
        v188 = v49;
        if ( v49 )
        {
          sub_B96E90((__int64)&v188, v49, 1);
          v12 = *(_DWORD *)(a2 + 72);
        }
        v50 = *(_QWORD *)(a2 + 40);
        v189 = v12;
        v51 = *(_QWORD *)(*(_QWORD *)(v50 + 80) + 96LL);
        v52 = *(_QWORD **)(v51 + 24);
        if ( *(_DWORD *)(v51 + 32) > 0x40u )
          v52 = (_QWORD *)*v52;
        v190 = v192;
        v191 = 0x200000000LL;
        v53 = sub_3400BD0(a4, (_DWORD)v52, (unsigned int)&v188, 7, 0, 1, 0);
        v54 = (unsigned int)v191;
        v56 = v55;
        v57 = (unsigned int)v191 + 1LL;
        if ( v57 > HIDWORD(v191) )
        {
          v185.m128i_i64[0] = v53;
          v185.m128i_i64[1] = v56;
          sub_C8D5F0((__int64)&v190, v192, v57, 0x10u, v53, v56);
          v54 = (unsigned int)v191;
          v56 = v185.m128i_i64[1];
          v53 = v185.m128i_i64[0];
        }
        v58 = (__int64 *)&v190[16 * v54];
        *v58 = v53;
        v58[1] = v56;
        v59 = *(const __m128i **)(a2 + 40);
        LODWORD(v191) = v191 + 1;
        v60 = (unsigned int)v191;
        v61 = _mm_loadu_si128(v59);
        if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
        {
          v185 = v61;
          sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v53, v56);
          v60 = (unsigned int)v191;
          v61 = _mm_load_si128(&v185);
        }
        *(__m128i *)&v190[16 * v60] = v61;
        v62 = *(_QWORD *)(a2 + 48);
        v63 = *(_DWORD *)(a2 + 68);
        LODWORD(v191) = v191 + 1;
        v47 = sub_33E66D0(a4, 4941, (unsigned int)&v188, v62, v63, v56, (__int64)v190, (unsigned int)v191);
        goto LABEL_164;
      }
LABEL_25:
      sub_C64ED0(
        "tcgen05.relinquish.alloc supported only on arch-conditional or family-conditional variants from SM100 onwards.",
        1u);
    }
    if ( (_DWORD)v14 != 10351 )
      goto LABEL_18;
    v21 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
    v22 = *(_DWORD *)(v21 + 340);
    if ( v22 > 0x408 )
    {
      if ( v22 - 1101 > 1 )
        goto LABEL_29;
    }
    else if ( v22 <= 0x3E8 || ((1LL << ((unsigned __int8)v22 + 23)) & 0xC0000C03) == 0 )
    {
      goto LABEL_29;
    }
    v64 = *(_DWORD *)(v21 + 336);
    if ( __ROR4__(-858993459 * v22 + 1717986918, 1) <= 0x19999999u && v64 > 0x57 || v64 > 0x55 )
    {
      v65 = *(_QWORD *)(a2 + 80);
      v188 = v65;
      if ( v65 )
      {
        sub_B96E90((__int64)&v188, v65, 1);
        v12 = *(_DWORD *)(a2 + 72);
      }
      v66 = *(_QWORD *)(a2 + 40);
      v189 = v12;
      v67 = *(_QWORD *)(*(_QWORD *)(v66 + 80) + 96LL);
      v68 = *(_QWORD **)(v67 + 24);
      if ( *(_DWORD *)(v67 + 32) > 0x40u )
        v68 = (_QWORD *)*v68;
      v190 = v192;
      v191 = 0x200000000LL;
      v69 = sub_3400BD0(a4, (_DWORD)v68, (unsigned int)&v188, 7, 0, 1, 0);
      v70 = (unsigned int)v191;
      v72 = v71;
      v73 = (unsigned int)v191 + 1LL;
      if ( v73 > HIDWORD(v191) )
      {
        v185.m128i_i64[0] = v69;
        v185.m128i_i64[1] = v72;
        sub_C8D5F0((__int64)&v190, v192, v73, 0x10u, v69, v72);
        v70 = (unsigned int)v191;
        v72 = v185.m128i_i64[1];
        v69 = v185.m128i_i64[0];
      }
      v74 = (__int64 *)&v190[16 * v70];
      *v74 = v69;
      v74[1] = v72;
      v75 = *(const __m128i **)(a2 + 40);
      LODWORD(v191) = v191 + 1;
      v76 = (unsigned int)v191;
      v77 = _mm_loadu_si128(v75);
      if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
      {
        v185 = v77;
        sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v69, v72);
        v76 = (unsigned int)v191;
        v77 = _mm_load_si128(&v185);
      }
      *(__m128i *)&v190[16 * v76] = v77;
      v78 = *(_QWORD *)(a2 + 48);
      v79 = *(_DWORD *)(a2 + 68);
      LODWORD(v191) = v191 + 1;
      v47 = sub_33E66D0(a4, 5020, (unsigned int)&v188, v78, v79, v72, (__int64)v190, (unsigned int)v191);
      goto LABEL_164;
    }
LABEL_29:
    sub_C64ED0("tcgen05.wait supported only on arch-conditional or family-conditional variants from SM100 onwards.", 1u);
  }
  if ( (unsigned int)v14 > 0x276C )
  {
    if ( (_DWORD)v14 != 10101 )
    {
      if ( (_DWORD)v14 == 10140 )
      {
        v23 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
        v24 = *(_DWORD *)(v23 + 340);
        if ( v24 > 0x408 )
        {
          if ( v24 - 1101 > 1 )
            goto LABEL_34;
        }
        else if ( v24 <= 0x3E8 || ((1LL << ((unsigned __int8)v24 + 23)) & 0xC0000C03) == 0 )
        {
          goto LABEL_34;
        }
        v25 = *(_DWORD *)(v23 + 336);
        if ( __ROR4__(-858993459 * v24 + 1717986918, 1) <= 0x19999999u && v25 > 0x57 || v25 > 0x55 )
        {
          v26 = *(_QWORD *)(a2 + 80);
          v188 = v26;
          if ( v26 )
          {
            sub_B96E90((__int64)&v188, v26, 1);
            v12 = *(_DWORD *)(a2 + 72);
          }
          v27 = *(_QWORD *)(a2 + 40);
          v189 = v12;
          v28 = *(_QWORD *)(*(_QWORD *)(v27 + 80) + 96LL);
          v29 = *(_QWORD **)(v28 + 24);
          if ( *(_DWORD *)(v28 + 32) > 0x40u )
            v29 = (_QWORD *)*v29;
          v190 = v192;
          v191 = 0x400000000LL;
          v30 = sub_3400BD0(a4, (_DWORD)v29, (unsigned int)&v188, 7, 0, 1, 0);
          v31 = (unsigned int)v191;
          v33 = v32;
          v34 = (unsigned int)v191 + 1LL;
          if ( v34 > HIDWORD(v191) )
          {
            v185.m128i_i64[0] = v30;
            v185.m128i_i64[1] = v33;
            sub_C8D5F0((__int64)&v190, v192, v34, 0x10u, v30, v33);
            v31 = (unsigned int)v191;
            v33 = v185.m128i_i64[1];
            v30 = v185.m128i_i64[0];
          }
          v35 = (__int64 *)&v190[16 * v31];
          *v35 = v30;
          v35[1] = v33;
          v36 = *(_QWORD *)(a2 + 40);
          LODWORD(v191) = v191 + 1;
          v37 = (unsigned int)v191;
          v38 = _mm_loadu_si128((const __m128i *)(v36 + 120));
          if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
          {
            v185 = v38;
            sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v30, v33);
            v37 = (unsigned int)v191;
            v38 = _mm_load_si128(&v185);
          }
          *(__m128i *)&v190[16 * v37] = v38;
          v39 = *(const __m128i **)(a2 + 40);
          LODWORD(v191) = v191 + 1;
          v40 = (unsigned int)v191;
          v41 = _mm_loadu_si128(v39 + 10);
          if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
          {
            v185 = v41;
            sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v30, v33);
            v40 = (unsigned int)v191;
            v41 = _mm_load_si128(&v185);
          }
          *(__m128i *)&v190[16 * v40] = v41;
          v42 = *(const __m128i **)(a2 + 40);
          LODWORD(v191) = v191 + 1;
          v43 = (unsigned int)v191;
          v44 = _mm_loadu_si128(v42);
          if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
          {
            v185 = v44;
            sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v30, v33);
            v43 = (unsigned int)v191;
            v44 = _mm_load_si128(&v185);
          }
          *(__m128i *)&v190[16 * v43] = v44;
          v45 = *(_QWORD *)(a2 + 48);
          v46 = *(_DWORD *)(a2 + 68);
          LODWORD(v191) = v191 + 1;
          v47 = sub_33E66D0(a4, 4827, (unsigned int)&v188, v45, v46, v33, (__int64)v190, (unsigned int)v191);
          goto LABEL_164;
        }
LABEL_34:
        sub_C64ED0(
          "tcgen05.dealloc supported only on arch-conditional or family-conditional variants from SM100 onwards.",
          1u);
      }
LABEL_18:
      v17 = 0;
      goto LABEL_19;
    }
    v106 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
    v107 = *(_DWORD *)(v106 + 340);
    if ( v107 > 0x408 )
    {
      if ( v107 - 1101 > 1 )
        goto LABEL_99;
    }
    else if ( v107 <= 0x3E8 || ((1LL << ((unsigned __int8)v107 + 23)) & 0xC0000C03) == 0 )
    {
      goto LABEL_99;
    }
    v108 = *(_DWORD *)(v106 + 336);
    if ( __ROR4__(-858993459 * v107 + 1717986918, 1) <= 0x19999999u && v108 > 0x57 || v108 > 0x55 )
    {
      v109 = *(_QWORD *)(a2 + 80);
      v188 = v109;
      if ( v109 )
      {
        sub_B96E90((__int64)&v188, v109, 1);
        v12 = *(_DWORD *)(a2 + 72);
      }
      v110 = *(_QWORD *)(a2 + 40);
      v189 = v12;
      v111 = *(_QWORD *)(*(_QWORD *)(v110 + 80) + 96LL);
      v112 = *(_QWORD **)(v111 + 24);
      if ( *(_DWORD *)(v111 + 32) > 0x40u )
        v112 = (_QWORD *)*v112;
      v190 = v192;
      v191 = 0x400000000LL;
      v113 = sub_3400BD0(a4, (_DWORD)v112, (unsigned int)&v188, 7, 0, 1, 0);
      v114 = (unsigned int)v191;
      v116 = v115;
      v117 = (unsigned int)v191 + 1LL;
      if ( v117 > HIDWORD(v191) )
      {
        v185.m128i_i64[0] = v113;
        v185.m128i_i64[1] = v116;
        sub_C8D5F0((__int64)&v190, v192, v117, 0x10u, v113, v116);
        v114 = (unsigned int)v191;
        v116 = v185.m128i_i64[1];
        v113 = v185.m128i_i64[0];
      }
      v118 = (__int64 *)&v190[16 * v114];
      *v118 = v113;
      v118[1] = v116;
      v119 = *(_QWORD *)(a2 + 40);
      LODWORD(v191) = v191 + 1;
      v120 = (unsigned int)v191;
      v121 = _mm_loadu_si128((const __m128i *)(v119 + 120));
      if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
      {
        v185 = v121;
        sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v113, v116);
        v120 = (unsigned int)v191;
        v121 = _mm_load_si128(&v185);
      }
      *(__m128i *)&v190[16 * v120] = v121;
      v122 = *(const __m128i **)(a2 + 40);
      LODWORD(v191) = v191 + 1;
      v123 = (unsigned int)v191;
      v124 = _mm_loadu_si128(v122 + 10);
      if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
      {
        v185 = v124;
        sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v113, v116);
        v123 = (unsigned int)v191;
        v124 = _mm_load_si128(&v185);
      }
      *(__m128i *)&v190[16 * v123] = v124;
      v125 = *(const __m128i **)(a2 + 40);
      LODWORD(v191) = v191 + 1;
      v126 = (unsigned int)v191;
      v127 = _mm_loadu_si128(v125);
      if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
      {
        v185 = v127;
        sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v113, v116);
        v126 = (unsigned int)v191;
        v127 = _mm_load_si128(&v185);
      }
      *(__m128i *)&v190[16 * v126] = v127;
      v128 = *(_QWORD *)(a2 + 48);
      v129 = *(_DWORD *)(a2 + 68);
      LODWORD(v191) = v191 + 1;
      v47 = sub_33E66D0(a4, 4790, (unsigned int)&v188, v128, v129, v116, (__int64)v190, (unsigned int)v191);
      goto LABEL_164;
    }
LABEL_99:
    sub_C64ED0("tcgen05.cp.* supported only on arch-conditional or family-conditional variants from SM100 onwards.", 1u);
  }
  if ( (unsigned int)v14 > 0x2768 )
  {
    v130 = *(__int64 **)(a4 + 40);
    v131 = v130[2];
    v132 = *(_DWORD *)(v131 + 340);
    if ( v132 > 0x408 )
    {
      if ( v132 - 1101 > 1 )
        goto LABEL_122;
    }
    else if ( v132 <= 0x3E8 || ((1LL << ((unsigned __int8)v132 + 23)) & 0xC0000C03) == 0 )
    {
      goto LABEL_122;
    }
    v135 = *(_DWORD *)(v131 + 336);
    if ( __ROR4__(-858993459 * v132 + 1717986918, 1) <= 0x19999999u && v135 > 0x57 || v135 > 0x55 )
    {
      v136 = *(_QWORD *)(a2 + 80);
      v188 = v136;
      if ( v136 )
      {
        sub_B96E90((__int64)&v188, v136, 1);
        v12 = *(_DWORD *)(a2 + 72);
        v130 = *(__int64 **)(a4 + 40);
      }
      v137 = *(_QWORD *)(a2 + 40);
      v189 = v12;
      v138 = *(_QWORD *)(*(_QWORD *)(v137 + 40) + 96LL);
      v139 = *(_QWORD **)(v138 + 24);
      if ( *(_DWORD *)(v138 + 32) > 0x40u )
        v139 = (_QWORD *)*v139;
      v140 = (_DWORD)v139 - 10090;
      v141 = (_DWORD)v139 - 10091;
      v185.m128i_i32[0] = (_DWORD)v139 - 10090;
      v142 = sub_2E79000(v130);
      v143 = sub_AE2980(v142, 3u)[1];
      v144 = *(_QWORD *)(a2 + 40);
      if ( v140 <= 1 )
      {
        if ( (unsigned __int16)(*(_WORD *)(*(_QWORD *)(*(_QWORD *)(v144 + 160) + 48LL)
                                         + 16LL * *(unsigned int *)(v144 + 168))
                              - 6) > 1u )
          sub_C64ED0("tcgen05.commit.* supports only 16-bit and 32-bit multicast mask size.", 1u);
        v145 = 4773;
        if ( v141 <= 1 )
          v145 = (v143 != 32) + 4774;
      }
      else
      {
        v145 = 4772;
        if ( v141 <= 1 )
          v145 = (v143 != 32) + 4776;
      }
      v146 = *(_QWORD *)(*(_QWORD *)(v144 + 80) + 96LL);
      v147 = *(_QWORD **)(v146 + 24);
      if ( *(_DWORD *)(v146 + 32) > 0x40u )
        v147 = (_QWORD *)*v147;
      v190 = v192;
      v184.m128i_i32[0] = v145;
      v191 = 0x400000000LL;
      v148 = sub_3400BD0(a4, (_DWORD)v147, (unsigned int)&v188, 7, 0, 1, 0);
      v149 = (unsigned int)v191;
      v151 = v150;
      v152 = v184.m128i_i32[0];
      v153 = (unsigned int)v191 + 1LL;
      if ( v153 > HIDWORD(v191) )
      {
        v182 = v148;
        v183 = v151;
        sub_C8D5F0((__int64)&v190, v192, v153, 0x10u, v148, v151);
        v149 = (unsigned int)v191;
        v148 = v182;
        v151 = v183;
        v152 = v184.m128i_i32[0];
      }
      v154 = (__int64 *)&v190[16 * v149];
      *v154 = v148;
      v154[1] = v151;
      v155 = *(_QWORD *)(a2 + 40);
      LODWORD(v191) = v191 + 1;
      v156 = (unsigned int)v191;
      v157 = _mm_loadu_si128((const __m128i *)(v155 + 120));
      if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
      {
        v181 = v152;
        v184 = v157;
        sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v148, v151);
        v156 = (unsigned int)v191;
        v152 = v181;
        v157 = _mm_load_si128(&v184);
      }
      *(__m128i *)&v190[16 * v156] = v157;
      v158 = (unsigned int)(v191 + 1);
      LODWORD(v191) = v191 + 1;
      if ( v185.m128i_i32[0] <= 1u )
      {
        v178 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 160LL));
        if ( v158 + 1 > (unsigned __int64)HIDWORD(v191) )
        {
          v185.m128i_i32[0] = v152;
          v184 = v178;
          sub_C8D5F0((__int64)&v190, v192, v158 + 1, 0x10u, v148, v151);
          v158 = (unsigned int)v191;
          v178 = _mm_load_si128(&v184);
          v152 = v185.m128i_i32[0];
        }
        *(__m128i *)&v190[16 * v158] = v178;
        v158 = (unsigned int)(v191 + 1);
        LODWORD(v191) = v191 + 1;
      }
      v159 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      if ( v158 + 1 > (unsigned __int64)HIDWORD(v191) )
      {
        v184.m128i_i32[0] = v152;
        v185 = v159;
        sub_C8D5F0((__int64)&v190, v192, v158 + 1, 0x10u, v148, v151);
        v158 = (unsigned int)v191;
        v152 = v184.m128i_i32[0];
        v159 = _mm_load_si128(&v185);
      }
      *(__m128i *)&v190[16 * v158] = v159;
      v160 = *(_QWORD *)(a2 + 48);
      v161 = *(_DWORD *)(a2 + 68);
      LODWORD(v191) = v191 + 1;
      v47 = sub_33E66D0(a4, v152, (unsigned int)&v188, v160, v161, v151, (__int64)v190, (unsigned int)v191);
      goto LABEL_164;
    }
LABEL_122:
    sub_C64ED0(
      "tcgen05.commit.* supported only on arch-conditional or family-conditional variants from SM100 onwards.",
      1u);
  }
  if ( (_DWORD)v14 != 10080 && (_DWORD)v14 != 10083 )
    goto LABEL_18;
  v15 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 16LL);
  v16 = *(_DWORD *)(v15 + 340);
  if ( v16 > 0x408 )
  {
    if ( v16 - 1101 > 1 )
      goto LABEL_13;
  }
  else if ( v16 <= 0x3E8 || ((1LL << ((unsigned __int8)v16 + 23)) & 0xC0000C03) == 0 )
  {
    goto LABEL_13;
  }
  v80 = *(_DWORD *)(v15 + 336);
  if ( (__ROR4__(-858993459 * v16 + 1717986918, 1) > 0x19999999u || v80 <= 0x57) && v80 <= 0x55 )
LABEL_13:
    sub_C64ED0(
      "tcgen05.alloc supported only on arch-conditional or family-conditional variants from SM100 onwards.",
      1u);
  v81 = *(_QWORD *)(a2 + 80);
  v188 = v81;
  if ( v81 )
  {
    sub_B96E90((__int64)&v188, v81, 1);
    v12 = *(_DWORD *)(a2 + 72);
  }
  v82 = *(_QWORD *)(a2 + 40);
  v189 = v12;
  v83 = *(_QWORD *)(*(_QWORD *)(v82 + 40) + 96LL);
  v84 = *(_QWORD **)(v83 + 24);
  if ( *(_DWORD *)(v83 + 32) > 0x40u )
    v84 = (_QWORD *)*v84;
  v85 = 4765;
  if ( (_DWORD)v84 == 10083 )
  {
    v179 = sub_2E79000(*(__int64 **)(a4 + 40));
    v180 = sub_AE2980(v179, 3u);
    v82 = *(_QWORD *)(a2 + 40);
    v85 = 4770;
    if ( v180[1] != 32 )
      v85 = 4771;
  }
  v86 = *(_QWORD *)(*(_QWORD *)(v82 + 80) + 96LL);
  v87 = *(_QWORD **)(v86 + 24);
  if ( *(_DWORD *)(v86 + 32) > 0x40u )
    v87 = (_QWORD *)*v87;
  v185.m128i_i32[0] = v85;
  v190 = v192;
  v191 = 0x400000000LL;
  v88 = sub_3400BD0(a4, (_DWORD)v87, (unsigned int)&v188, 7, 0, 1, 0);
  v89 = (unsigned int)v191;
  v91 = v90;
  v92 = v185.m128i_i32[0];
  v93 = (unsigned int)v191 + 1LL;
  if ( v93 > HIDWORD(v191) )
  {
    v184.m128i_i64[0] = v88;
    v184.m128i_i64[1] = v91;
    sub_C8D5F0((__int64)&v190, v192, v93, 0x10u, v88, v91);
    v89 = (unsigned int)v191;
    v91 = v184.m128i_i64[1];
    v88 = v184.m128i_i64[0];
    v92 = v185.m128i_i32[0];
  }
  v94 = (__int64 *)&v190[16 * v89];
  *v94 = v88;
  v94[1] = v91;
  v95 = *(_QWORD *)(a2 + 40);
  LODWORD(v191) = v191 + 1;
  v96 = (unsigned int)v191;
  v97 = _mm_loadu_si128((const __m128i *)(v95 + 120));
  if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
  {
    v185.m128i_i32[0] = v92;
    v184 = v97;
    sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v88, v91);
    v96 = (unsigned int)v191;
    v97 = _mm_load_si128(&v184);
    v92 = v185.m128i_i32[0];
  }
  *(__m128i *)&v190[16 * v96] = v97;
  v98 = *(const __m128i **)(a2 + 40);
  LODWORD(v191) = v191 + 1;
  v99 = (unsigned int)v191;
  v100 = _mm_loadu_si128(v98 + 10);
  if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
  {
    v185.m128i_i32[0] = v92;
    v184 = v100;
    sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v88, v91);
    v99 = (unsigned int)v191;
    v100 = _mm_load_si128(&v184);
    v92 = v185.m128i_i32[0];
  }
  *(__m128i *)&v190[16 * v99] = v100;
  v101 = *(const __m128i **)(a2 + 40);
  LODWORD(v191) = v191 + 1;
  v102 = (unsigned int)v191;
  v103 = _mm_loadu_si128(v101);
  if ( (unsigned __int64)(unsigned int)v191 + 1 > HIDWORD(v191) )
  {
    v185.m128i_i32[0] = v92;
    v184 = v103;
    sub_C8D5F0((__int64)&v190, v192, (unsigned int)v191 + 1LL, 0x10u, v88, v91);
    v102 = (unsigned int)v191;
    v103 = _mm_load_si128(&v184);
    v92 = v185.m128i_i32[0];
  }
  *(__m128i *)&v190[16 * v102] = v103;
  v104 = *(_QWORD *)(a2 + 48);
  v105 = *(_DWORD *)(a2 + 68);
  LODWORD(v191) = v191 + 1;
  v47 = sub_33E66D0(a4, v92, (unsigned int)&v188, v104, v105, v91, (__int64)v190, (unsigned int)v191);
LABEL_164:
  v17 = v47;
  if ( v190 != v192 )
    _libc_free((unsigned __int64)v190);
  if ( v188 )
    sub_B91220((__int64)&v188, v188);
LABEL_19:
  if ( v186 )
    sub_B91220((__int64)&v186, v186);
  return v17;
}
