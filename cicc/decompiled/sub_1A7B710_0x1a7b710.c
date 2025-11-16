// Function: sub_1A7B710
// Address: 0x1a7b710
//
__int64 __fastcall sub_1A7B710(__int64 a1, _BYTE *a2, __int64 *a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v12; // r15
  __int64 i; // rbx
  int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // rbx
  _BYTE *v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r14
  unsigned int v27; // r12d
  __int64 *v28; // r9
  __int64 *v29; // r8
  unsigned int v30; // r15d
  unsigned int v31; // edi
  __int64 *v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rbx
  int v36; // ecx
  __int64 v37; // rax
  __int64 *v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rax
  _BYTE *v41; // rbx
  char v42; // r11
  _BYTE *v43; // r15
  unsigned int v44; // ecx
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  __int64 v48; // r12
  int v49; // edx
  __int64 v50; // rcx
  __int64 v51; // r11
  int v52; // edi
  __int64 *v53; // rsi
  __int64 v54; // rax
  int v55; // r8d
  int v56; // r9d
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r14
  int v60; // r14d
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r14
  __int64 v64; // r14
  _QWORD *v65; // rdi
  unsigned __int8 v66; // al
  _QWORD *v67; // rax
  _BYTE *v68; // r14
  __int64 v69; // rax
  __int64 v70; // rax
  __m128i v71; // xmm0
  __m128i v72; // xmm1
  char *v73; // r14
  _QWORD *v74; // r14
  _QWORD *v75; // rdi
  int v76; // r10d
  unsigned int v77; // r15d
  __int64 v78; // rsi
  unsigned int v79; // eax
  __int64 v80; // rcx
  __int64 v81; // rdi
  __int64 *v82; // rdx
  __int64 v83; // r10
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r14
  __int64 v87; // rdx
  __int64 v88; // rax
  _BYTE *v89; // rdx
  int v90; // r13d
  __int64 *v91; // rcx
  int v92; // edx
  int v93; // r10d
  __int64 *v94; // r9
  unsigned int v95; // eax
  __int64 v96; // r8
  int v97; // edi
  __int64 *v98; // rsi
  int v99; // esi
  unsigned int v100; // ebx
  __int64 *v101; // rax
  __int64 v102; // rdi
  int v103; // esi
  __int64 v104; // r14
  __int64 *v105; // rcx
  __int64 v106; // rdi
  int v107; // r10d
  _BYTE *v108; // rdx
  char *v109; // rdi
  int v110; // r10d
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // [rsp+8h] [rbp-9D8h]
  unsigned __int8 v115; // [rsp+38h] [rbp-9A8h]
  _QWORD *v116; // [rsp+38h] [rbp-9A8h]
  char *v117; // [rsp+38h] [rbp-9A8h]
  __int64 v119; // [rsp+48h] [rbp-998h]
  int v120; // [rsp+48h] [rbp-998h]
  __int64 v121; // [rsp+50h] [rbp-990h] BYREF
  __int64 v122; // [rsp+58h] [rbp-988h]
  __int64 v123; // [rsp+60h] [rbp-980h]
  unsigned int v124; // [rsp+68h] [rbp-978h]
  _BYTE *v125; // [rsp+70h] [rbp-970h] BYREF
  __int64 v126; // [rsp+78h] [rbp-968h]
  _BYTE v127[256]; // [rsp+80h] [rbp-960h] BYREF
  _BYTE *v128; // [rsp+180h] [rbp-860h] BYREF
  __int64 v129; // [rsp+188h] [rbp-858h]
  _BYTE v130[256]; // [rsp+190h] [rbp-850h] BYREF
  _BYTE *v131; // [rsp+290h] [rbp-750h] BYREF
  __int64 v132; // [rsp+298h] [rbp-748h]
  _BYTE v133[256]; // [rsp+2A0h] [rbp-740h] BYREF
  void *v134; // [rsp+3A0h] [rbp-640h] BYREF
  int v135; // [rsp+3A8h] [rbp-638h]
  char v136; // [rsp+3ACh] [rbp-634h]
  __int64 v137; // [rsp+3B0h] [rbp-630h]
  __m128i v138; // [rsp+3B8h] [rbp-628h]
  __int64 v139; // [rsp+3C8h] [rbp-618h]
  __int64 v140; // [rsp+3D0h] [rbp-610h]
  __m128i v141; // [rsp+3D8h] [rbp-608h]
  __int64 v142; // [rsp+3E8h] [rbp-5F8h]
  char v143; // [rsp+3F0h] [rbp-5F0h]
  _BYTE *v144; // [rsp+3F8h] [rbp-5E8h] BYREF
  __int64 v145; // [rsp+400h] [rbp-5E0h]
  _BYTE v146[352]; // [rsp+408h] [rbp-5D8h] BYREF
  char v147; // [rsp+568h] [rbp-478h]
  int v148; // [rsp+56Ch] [rbp-474h]
  __int64 v149; // [rsp+570h] [rbp-470h]
  void *v150; // [rsp+580h] [rbp-460h] BYREF
  int v151; // [rsp+588h] [rbp-458h]
  char v152; // [rsp+58Ch] [rbp-454h]
  __int64 v153; // [rsp+590h] [rbp-450h]
  __m128i v154; // [rsp+598h] [rbp-448h] BYREF
  __int64 v155; // [rsp+5A8h] [rbp-438h]
  __int64 v156; // [rsp+5B0h] [rbp-430h]
  __m128i v157; // [rsp+5B8h] [rbp-428h] BYREF
  __int64 v158; // [rsp+5C8h] [rbp-418h]
  char v159; // [rsp+5D0h] [rbp-410h]
  char *v160; // [rsp+5D8h] [rbp-408h] BYREF
  unsigned int v161; // [rsp+5E0h] [rbp-400h]
  char v162; // [rsp+5E8h] [rbp-3F8h] BYREF
  char v163; // [rsp+748h] [rbp-298h]
  int v164; // [rsp+74Ch] [rbp-294h]
  __int64 v165; // [rsp+750h] [rbp-290h]
  __int64 v166; // [rsp+760h] [rbp-280h] BYREF
  _BYTE *v167; // [rsp+768h] [rbp-278h]
  _BYTE *v168; // [rsp+770h] [rbp-270h]
  __int64 v169; // [rsp+778h] [rbp-268h]
  int v170; // [rsp+780h] [rbp-260h]
  _BYTE v171[256]; // [rsp+788h] [rbp-258h] BYREF
  __int64 v172; // [rsp+888h] [rbp-158h] BYREF
  _BYTE *v173; // [rsp+890h] [rbp-150h]
  _BYTE *v174; // [rsp+898h] [rbp-148h]
  __int64 v175; // [rsp+8A0h] [rbp-140h]
  int v176; // [rsp+8A8h] [rbp-138h]
  _BYTE v177[304]; // [rsp+8B0h] [rbp-130h] BYREF

  v115 = sub_15E3780(a1);
  if ( v115 )
    return 0;
  *a2 = 1;
  v167 = v171;
  v168 = v171;
  v166 = 0;
  v169 = 32;
  v170 = 0;
  v172 = 0;
  v173 = v177;
  v174 = v177;
  v175 = 32;
  v176 = 0;
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, (__int64)a2);
    v8 = *(_QWORD *)(a1 + 88);
    v9 = v8 + 40LL * *(_QWORD *)(a1 + 96);
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(a1, (__int64)a2);
      v8 = *(_QWORD *)(a1 + 88);
    }
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 88);
    v9 = v8 + 40LL * *(_QWORD *)(a1 + 96);
  }
  while ( v9 != v8 )
  {
    while ( !(unsigned __int8)sub_15E0450(v8) )
    {
      v8 += 40;
      if ( v9 == v8 )
        goto LABEL_11;
    }
    v10 = v8;
    v8 += 40;
    sub_1A7AAD0((__int64)&v166, v10, v3, v4, v5, v6);
  }
LABEL_11:
  v11 = *(_QWORD *)(a1 + 80);
  v12 = a1 + 72;
  if ( v11 != a1 + 72 )
  {
    do
    {
      if ( !v11 )
        BUG();
      for ( i = *(_QWORD *)(v11 + 24); v11 + 16 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        if ( *(_BYTE *)(i - 8) == 53 )
          sub_1A7AAD0((__int64)&v166, i - 24, v3, v4, v5, v6);
      }
      v11 = *(_QWORD *)(v11 + 8);
    }
    while ( v12 != v11 );
    v12 = *(_QWORD *)(a1 + 80);
  }
  v121 = 0;
  v14 = 1;
  v125 = v127;
  v126 = 0x2000000000LL;
  v129 = 0x2000000000LL;
  v132 = 0x2000000000LL;
  if ( v12 )
    v12 -= 24;
  v128 = v130;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v131 = v133;
  do
  {
    v15 = *(_QWORD *)(v12 + 48);
    v16 = v12 + 40;
    if ( v15 != v12 + 40 )
    {
      v119 = v12;
      while ( 1 )
      {
        v18 = v173;
        v24 = 0;
        if ( v15 )
          v24 = v15 - 24;
        if ( v174 == v173 )
        {
          v17 = &v173[8 * HIDWORD(v175)];
          if ( v173 == v17 )
          {
            v89 = v173;
          }
          else
          {
            do
            {
              if ( v24 == *v18 )
                break;
              ++v18;
            }
            while ( v17 != (_BYTE *)v18 );
            v89 = &v173[8 * HIDWORD(v175)];
          }
        }
        else
        {
          v17 = &v174[8 * (unsigned int)v175];
          v18 = sub_16CC9F0((__int64)&v172, v24);
          if ( v24 == *v18 )
          {
            if ( v174 == v173 )
              v89 = &v174[8 * HIDWORD(v175)];
            else
              v89 = &v174[8 * (unsigned int)v175];
          }
          else
          {
            if ( v174 != v173 )
            {
              v18 = &v174[8 * (unsigned int)v175];
              goto LABEL_28;
            }
            v18 = &v174[8 * HIDWORD(v175)];
            v89 = v18;
          }
        }
        while ( v89 != (_BYTE *)v18 && *v18 >= 0xFFFFFFFFFFFFFFFELL )
          ++v18;
LABEL_28:
        if ( v18 != (_QWORD *)v17 )
          v14 = 2;
        if ( *(_BYTE *)(v24 + 16) != 78 )
          goto LABEL_41;
        if ( (*(_WORD *)(v24 + 18) & 3u) - 1 <= 1 )
          goto LABEL_41;
        v19 = *(_QWORD *)(v24 - 24);
        if ( !*(_BYTE *)(v19 + 16)
          && (*(_BYTE *)(v19 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v19 + 36) - 35) <= 3 )
        {
          goto LABEL_41;
        }
        if ( (*(_WORD *)(v24 + 18) & 3) == 3 )
          goto LABEL_40;
        if ( *(char *)(v24 + 23) < 0 )
        {
          v20 = sub_1648A40(v24);
          v22 = v20 + v21;
          v23 = 0;
          if ( *(char *)(v24 + 23) < 0 )
            v23 = sub_1648A40(v24);
          if ( (unsigned int)((v22 - v23) >> 4) )
            goto LABEL_40;
        }
        if ( (unsigned __int8)sub_1560260((_QWORD *)(v24 + 56), -1, 36) )
          goto LABEL_96;
        if ( *(char *)(v24 + 23) >= 0 )
          goto LABEL_295;
        v84 = sub_1648A40(v24);
        v86 = v84 + v85;
        v87 = 0;
        if ( *(char *)(v24 + 23) < 0 )
          v87 = sub_1648A40(v24);
        if ( !(unsigned int)((v86 - v87) >> 4) )
        {
LABEL_295:
          v88 = *(_QWORD *)(v24 - 24);
          if ( !*(_BYTE *)(v88 + 16) )
          {
            v150 = *(void **)(v88 + 112);
            if ( (unsigned __int8)sub_1560260(&v150, -1, 36) )
            {
LABEL_96:
              if ( *(char *)(v24 + 23) < 0 )
              {
                v57 = sub_1648A40(v24);
                v59 = v57 + v58;
                if ( *(char *)(v24 + 23) >= 0 )
                {
                  if ( (unsigned int)(v59 >> 4) )
LABEL_286:
                    BUG();
                }
                else if ( (unsigned int)((v59 - sub_1648A40(v24)) >> 4) )
                {
                  if ( *(char *)(v24 + 23) >= 0 )
                    goto LABEL_286;
                  v60 = *(_DWORD *)(sub_1648A40(v24) + 8);
                  if ( *(char *)(v24 + 23) >= 0 )
                    BUG();
                  v61 = sub_1648A40(v24);
                  v63 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v61 + v62 - 4) - v60);
                  goto LABEL_104;
                }
              }
              v63 = -24;
LABEL_104:
              v113 = v24 + v63;
              v64 = v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF);
              if ( v113 == v64 )
              {
LABEL_118:
                v70 = sub_15E0530(*a3);
                if ( sub_1602790(v70)
                  || (v111 = sub_15E0530(*a3),
                      v112 = sub_16033E0(v111),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v112 + 48LL))(v112)) )
                {
                  sub_15CA3B0((__int64)&v150, (__int64)"tailcallelim", (__int64)"tailcall-readnone", 17, v24);
                  sub_15CAB20((__int64)&v150, "marked as tail call candidate (readnone)", 0x28u);
                  v71 = _mm_loadu_si128(&v154);
                  v72 = _mm_loadu_si128(&v157);
                  v135 = v151;
                  v138 = v71;
                  v136 = v152;
                  v141 = v72;
                  v137 = v153;
                  v139 = v155;
                  v134 = &unk_49ECF68;
                  v140 = v156;
                  v143 = v159;
                  if ( v159 )
                    v142 = v158;
                  v144 = v146;
                  v145 = 0x400000000LL;
                  if ( v161 )
                  {
                    sub_1A7B480((__int64)&v144, (__int64)&v160);
                    v147 = v163;
                    v117 = v160;
                    v148 = v164;
                    v149 = v165;
                    v134 = &unk_49ECF98;
                    v150 = &unk_49ECF68;
                    v73 = &v160[88 * v161];
                    if ( v160 != v73 )
                    {
                      do
                      {
                        v73 -= 88;
                        v109 = (char *)*((_QWORD *)v73 + 4);
                        if ( v109 != v73 + 48 )
                          j_j___libc_free_0(v109, *((_QWORD *)v73 + 6) + 1LL);
                        if ( *(char **)v73 != v73 + 16 )
                          j_j___libc_free_0(*(_QWORD *)v73, *((_QWORD *)v73 + 2) + 1LL);
                      }
                      while ( v117 != v73 );
                      v73 = v160;
                    }
                  }
                  else
                  {
                    v73 = v160;
                    v147 = v163;
                    v148 = v164;
                    v149 = v165;
                    v134 = &unk_49ECF98;
                  }
                  if ( v73 != &v162 )
                    _libc_free((unsigned __int64)v73);
                  sub_143AA50(a3, (__int64)&v134);
                  v116 = v144;
                  v134 = &unk_49ECF68;
                  v74 = &v144[88 * (unsigned int)v145];
                  if ( v144 != (_BYTE *)v74 )
                  {
                    do
                    {
                      v74 -= 11;
                      v75 = (_QWORD *)v74[4];
                      if ( v75 != v74 + 6 )
                        j_j___libc_free_0(v75, v74[6] + 1LL);
                      if ( (_QWORD *)*v74 != v74 + 2 )
                        j_j___libc_free_0(*v74, v74[2] + 1LL);
                    }
                    while ( v116 != v74 );
                    v74 = v144;
                  }
                  if ( v74 != (_QWORD *)v146 )
                    _libc_free((unsigned __int64)v74);
                }
                v115 = 1;
                *(_WORD *)(v24 + 18) = *(_WORD *)(v24 + 18) & 0xFFFC | 1;
                goto LABEL_41;
              }
              while ( 1 )
              {
                v65 = sub_1648700(v64);
                v66 = *((_BYTE *)v65 + 16);
                if ( v66 > 0x10u && (v66 != 17 || (unsigned __int8)sub_15E0450((__int64)v65)) )
                  break;
                v64 += 24;
                if ( v113 == v64 )
                  goto LABEL_118;
              }
            }
          }
        }
        if ( v14 != 1 )
          goto LABEL_40;
        v67 = v167;
        if ( v168 == v167 )
        {
          v68 = &v167[8 * HIDWORD(v169)];
          if ( v167 == v68 )
          {
            v108 = v167;
          }
          else
          {
            do
            {
              if ( v24 == *v67 )
                break;
              ++v67;
            }
            while ( v68 != (_BYTE *)v67 );
            v108 = &v167[8 * HIDWORD(v169)];
          }
        }
        else
        {
          v68 = &v168[8 * (unsigned int)v169];
          v67 = sub_16CC9F0((__int64)&v166, v24);
          if ( v24 == *v67 )
          {
            if ( v168 == v167 )
              v108 = &v168[8 * HIDWORD(v169)];
            else
              v108 = &v168[8 * (unsigned int)v169];
          }
          else
          {
            if ( v168 != v167 )
            {
              v67 = &v168[8 * (unsigned int)v169];
              goto LABEL_113;
            }
            v67 = &v168[8 * HIDWORD(v169)];
            v108 = v67;
          }
        }
        while ( v108 != (_BYTE *)v67 && *v67 >= 0xFFFFFFFFFFFFFFFELL )
          ++v67;
LABEL_113:
        if ( v68 == (_BYTE *)v67 )
        {
          v69 = (unsigned int)v132;
          if ( (unsigned int)v132 >= HIDWORD(v132) )
          {
            sub_16CD150((__int64)&v131, v133, 0, 8, v55, v56);
            v69 = (unsigned int)v132;
          }
          *(_QWORD *)&v131[8 * v69] = v24;
          LODWORD(v132) = v132 + 1;
          goto LABEL_41;
        }
LABEL_40:
        *a2 = 0;
LABEL_41:
        v15 = *(_QWORD *)(v15 + 8);
        if ( v16 == v15 )
        {
          v12 = v119;
          break;
        }
      }
    }
    v25 = sub_157EBA0(v12);
    if ( v25 )
    {
      v120 = sub_15F4D60(v25);
      v26 = sub_157EBA0(v12);
      if ( v120 )
      {
        v27 = 0;
        while ( 1 )
        {
          v34 = sub_15F4DF0(v26, v27);
          v35 = v34;
          if ( !v124 )
            break;
          LODWORD(v28) = v124 - 1;
          LODWORD(v29) = v122;
          v30 = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
          v31 = (v124 - 1) & v30;
          v32 = (__int64 *)(v122 + 16LL * v31);
          v33 = *v32;
          if ( v35 == *v32 )
          {
LABEL_62:
            if ( *((_DWORD *)v32 + 2) < v14 )
            {
              v38 = v32;
              goto LABEL_71;
            }
LABEL_63:
            if ( v120 == ++v27 )
              goto LABEL_75;
          }
          else
          {
            v76 = 1;
            v38 = 0;
            while ( v33 != -8 )
            {
              if ( !v38 && v33 == -16 )
                v38 = v32;
              v31 = (unsigned int)v28 & (v76 + v31);
              v32 = (__int64 *)(v122 + 16LL * v31);
              v33 = *v32;
              if ( v35 == *v32 )
                goto LABEL_62;
              ++v76;
            }
            if ( !v38 )
              v38 = v32;
            ++v121;
            v36 = v123 + 1;
            if ( 4 * ((int)v123 + 1) < 3 * v124 )
            {
              if ( v124 - HIDWORD(v123) - v36 <= v124 >> 3 )
              {
                sub_1A7A670((__int64)&v121, v124);
                if ( !v124 )
                {
LABEL_292:
                  LODWORD(v123) = v123 + 1;
                  BUG();
                }
                v29 = 0;
                v77 = (v124 - 1) & v30;
                LODWORD(v28) = 1;
                v36 = v123 + 1;
                v38 = (__int64 *)(v122 + 16LL * v77);
                v78 = *v38;
                if ( v35 != *v38 )
                {
                  while ( v78 != -8 )
                  {
                    if ( v78 == -16 && !v29 )
                      v29 = v38;
                    v110 = (_DWORD)v28 + 1;
                    LODWORD(v28) = (v124 - 1) & (v77 + (_DWORD)v28);
                    v77 = (unsigned int)v28;
                    v38 = (__int64 *)(v122 + 16LL * (unsigned int)v28);
                    v78 = *v38;
                    if ( v35 == *v38 )
                      goto LABEL_68;
                    LODWORD(v28) = v110;
                  }
                  if ( v29 )
                    v38 = v29;
                }
              }
              goto LABEL_68;
            }
LABEL_66:
            sub_1A7A670((__int64)&v121, 2 * v124);
            if ( !v124 )
              goto LABEL_292;
            LODWORD(v29) = v122;
            v36 = v123 + 1;
            LODWORD(v37) = (v124 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
            v38 = (__int64 *)(v122 + 16LL * (unsigned int)v37);
            v39 = *v38;
            if ( v35 != *v38 )
            {
              v107 = 1;
              v28 = 0;
              while ( v39 != -8 )
              {
                if ( !v28 && v39 == -16 )
                  v28 = v38;
                v37 = (v124 - 1) & ((_DWORD)v37 + v107);
                v38 = (__int64 *)(v122 + 16 * v37);
                v39 = *v38;
                if ( v35 == *v38 )
                  goto LABEL_68;
                ++v107;
              }
              if ( v28 )
                v38 = v28;
            }
LABEL_68:
            LODWORD(v123) = v36;
            if ( *v38 != -8 )
              --HIDWORD(v123);
            *v38 = v35;
            *((_DWORD *)v38 + 2) = 0;
LABEL_71:
            *((_DWORD *)v38 + 2) = v14;
            if ( v14 == 2 )
            {
              v54 = (unsigned int)v129;
              if ( (unsigned int)v129 >= HIDWORD(v129) )
              {
                sub_16CD150((__int64)&v128, v130, 0, 8, (int)v29, (int)v28);
                v54 = (unsigned int)v129;
              }
              *(_QWORD *)&v128[8 * v54] = v35;
              LODWORD(v129) = v129 + 1;
              goto LABEL_63;
            }
            v40 = (unsigned int)v126;
            if ( (unsigned int)v126 >= HIDWORD(v126) )
            {
              sub_16CD150((__int64)&v125, v127, 0, 8, (int)v29, (int)v28);
              v40 = (unsigned int)v126;
            }
            ++v27;
            *(_QWORD *)&v125[8 * v40] = v35;
            LODWORD(v126) = v126 + 1;
            if ( v120 == v27 )
              goto LABEL_75;
          }
        }
        ++v121;
        goto LABEL_66;
      }
    }
LABEL_75:
    if ( !(_DWORD)v129 )
    {
      v79 = v126;
      while ( 1 )
      {
        if ( !v79 )
          goto LABEL_78;
        v80 = v79--;
        v12 = *(_QWORD *)&v125[8 * v80 - 8];
        LODWORD(v126) = v79;
        if ( !v124 )
          break;
        LODWORD(v81) = (v124 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v82 = (__int64 *)(v122 + 16LL * (unsigned int)v81);
        v83 = *v82;
        if ( v12 == *v82 )
        {
LABEL_162:
          v14 = *((_DWORD *)v82 + 2);
          if ( v14 == 1 )
            goto LABEL_77;
        }
        else
        {
          v90 = 1;
          v91 = 0;
          while ( v83 != -8 )
          {
            if ( !v91 && v83 == -16 )
              v91 = v82;
            v81 = (v124 - 1) & ((_DWORD)v81 + v90);
            v82 = (__int64 *)(v122 + 16 * v81);
            v83 = *v82;
            if ( v12 == *v82 )
              goto LABEL_162;
            ++v90;
          }
          if ( !v91 )
            v91 = v82;
          ++v121;
          v92 = v123 + 1;
          if ( 4 * ((int)v123 + 1) >= 3 * v124 )
            goto LABEL_203;
          if ( v124 - HIDWORD(v123) - v92 <= v124 >> 3 )
          {
            sub_1A7A670((__int64)&v121, v124);
            if ( !v124 )
            {
LABEL_287:
              LODWORD(v123) = v123 + 1;
              BUG();
            }
            v99 = 1;
            v100 = (v124 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v92 = v123 + 1;
            v101 = 0;
            v91 = (__int64 *)(v122 + 16LL * v100);
            v102 = *v91;
            if ( v12 != *v91 )
            {
              while ( v102 != -8 )
              {
                if ( !v101 && v102 == -16 )
                  v101 = v91;
                v100 = (v124 - 1) & (v99 + v100);
                v91 = (__int64 *)(v122 + 16LL * v100);
                v102 = *v91;
                if ( v12 == *v91 )
                  goto LABEL_177;
                ++v99;
              }
              if ( v101 )
                v91 = v101;
            }
          }
LABEL_177:
          LODWORD(v123) = v92;
          if ( *v91 != -8 )
            --HIDWORD(v123);
          *v91 = v12;
          *((_DWORD *)v91 + 2) = 0;
          v79 = v126;
        }
      }
      ++v121;
LABEL_203:
      sub_1A7A670((__int64)&v121, 2 * v124);
      if ( !v124 )
        goto LABEL_287;
      v92 = v123 + 1;
      v95 = (v124 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v91 = (__int64 *)(v122 + 16LL * v95);
      v96 = *v91;
      if ( v12 != *v91 )
      {
        v97 = 1;
        v98 = 0;
        while ( v96 != -8 )
        {
          if ( v96 == -16 && !v98 )
            v98 = v91;
          v95 = (v124 - 1) & (v97 + v95);
          v91 = (__int64 *)(v122 + 16LL * v95);
          v96 = *v91;
          if ( v12 == *v91 )
            goto LABEL_177;
          ++v97;
        }
        if ( v98 )
          v91 = v98;
      }
      goto LABEL_177;
    }
    v14 = 2;
    v12 = *(_QWORD *)&v128[8 * (unsigned int)v129 - 8];
    LODWORD(v129) = v129 - 1;
LABEL_77:
    ;
  }
  while ( v12 );
LABEL_78:
  v41 = v131;
  v42 = v115;
  v43 = &v131[8 * (unsigned int)v132];
  if ( v131 != v43 )
  {
    while ( 1 )
    {
      v47 = *(_QWORD *)v41;
      v48 = *(_QWORD *)(*(_QWORD *)v41 + 40LL);
      if ( !v124 )
        break;
      v44 = (v124 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v45 = (__int64 *)(v122 + 16LL * v44);
      v46 = *v45;
      if ( v48 != *v45 )
      {
        v93 = 1;
        v94 = 0;
        while ( v46 != -8 )
        {
          if ( v46 == -16 && !v94 )
            v94 = v45;
          v44 = (v124 - 1) & (v93 + v44);
          v45 = (__int64 *)(v122 + 16LL * v44);
          v46 = *v45;
          if ( v48 == *v45 )
            goto LABEL_81;
          ++v93;
        }
        if ( v94 )
          v45 = v94;
        ++v121;
        v49 = v123 + 1;
        if ( 4 * ((int)v123 + 1) < 3 * v124 )
        {
          if ( v124 - HIDWORD(v123) - v49 <= v124 >> 3 )
          {
            sub_1A7A670((__int64)&v121, v124);
            if ( !v124 )
            {
LABEL_289:
              LODWORD(v123) = v123 + 1;
              BUG();
            }
            v103 = 1;
            LODWORD(v104) = (v124 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v49 = v123 + 1;
            v105 = 0;
            v45 = (__int64 *)(v122 + 16LL * (unsigned int)v104);
            v106 = *v45;
            if ( v48 != *v45 )
            {
              while ( v106 != -8 )
              {
                if ( v106 == -16 && !v105 )
                  v105 = v45;
                v104 = (v124 - 1) & ((_DWORD)v104 + v103);
                v45 = (__int64 *)(v122 + 16 * v104);
                v106 = *v45;
                if ( v48 == *v45 )
                  goto LABEL_187;
                ++v103;
              }
              if ( v105 )
                v45 = v105;
            }
          }
          goto LABEL_187;
        }
LABEL_85:
        sub_1A7A670((__int64)&v121, 2 * v124);
        if ( !v124 )
          goto LABEL_289;
        v49 = v123 + 1;
        LODWORD(v50) = (v124 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v45 = (__int64 *)(v122 + 16LL * (unsigned int)v50);
        v51 = *v45;
        if ( v48 != *v45 )
        {
          v52 = 1;
          v53 = 0;
          while ( v51 != -8 )
          {
            if ( v51 == -16 && !v53 )
              v53 = v45;
            v50 = (v124 - 1) & ((_DWORD)v50 + v52);
            v45 = (__int64 *)(v122 + 16 * v50);
            v51 = *v45;
            if ( v48 == *v45 )
              goto LABEL_187;
            ++v52;
          }
          if ( v53 )
            v45 = v53;
        }
LABEL_187:
        LODWORD(v123) = v49;
        if ( *v45 != -8 )
          --HIDWORD(v123);
        *v45 = v48;
        *((_DWORD *)v45 + 2) = 0;
        goto LABEL_190;
      }
LABEL_81:
      if ( *((_DWORD *)v45 + 2) == 2 )
      {
        v41 += 8;
        *a2 = 0;
        if ( v43 == v41 )
          goto LABEL_191;
      }
      else
      {
LABEL_190:
        v41 += 8;
        v42 = 1;
        *(_WORD *)(v47 + 18) = *(_WORD *)(v47 + 18) & 0xFFFC | 1;
        if ( v43 == v41 )
        {
LABEL_191:
          v115 = v42;
          v43 = v131;
          goto LABEL_192;
        }
      }
    }
    ++v121;
    goto LABEL_85;
  }
LABEL_192:
  if ( v43 != v133 )
    _libc_free((unsigned __int64)v43);
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
  j___libc_free_0(v122);
  if ( v174 != v173 )
    _libc_free((unsigned __int64)v174);
  if ( v168 != v167 )
    _libc_free((unsigned __int64)v168);
  return v115;
}
