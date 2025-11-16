// Function: sub_1BF6690
// Address: 0x1bf6690
//
__int64 __fastcall sub_1BF6690(__int64 *a1, __m128i a2, __m128i a3)
{
  __int64 *v3; // r12
  _DWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r8
  bool v7; // dl
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r15
  char v11; // al
  __int64 v12; // r14
  _QWORD *v13; // rax
  unsigned int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rbx
  _QWORD *v18; // r13
  char *v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  _QWORD *v22; // rdi
  __int64 *v24; // r12
  char *v25; // rax
  __int64 v26; // rdx
  int v27; // eax
  int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rbx
  _QWORD *v31; // r13
  char *v32; // rax
  _QWORD *v33; // rbx
  _QWORD *v34; // rdi
  __int64 v35; // rbx
  _QWORD *v36; // r13
  char *v37; // rax
  _QWORD *v38; // rbx
  _QWORD *v39; // rdi
  __int64 v40; // r9
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 *v46; // rax
  unsigned int v47; // esi
  __int64 v48; // rdi
  __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // r9
  __int64 v52; // r12
  __int64 v53; // r8
  __int64 v54; // rcx
  __int64 v55; // rax
  unsigned __int64 *v56; // r14
  __int64 v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // r8
  int v60; // r9d
  __int64 v61; // r8
  int v62; // r9d
  __int64 v63; // rbx
  char *v64; // rax
  _QWORD *v65; // rbx
  _QWORD *v66; // r12
  _QWORD *v67; // rdi
  __int64 v68; // rbx
  _QWORD *v69; // r13
  char *v70; // rax
  _QWORD *v71; // rbx
  _QWORD *v72; // rdi
  __int64 v73; // rdi
  __int64 v74; // r13
  __int64 v75; // r12
  __int64 *v76; // rax
  unsigned int v77; // eax
  __int64 v78; // rbx
  char *v79; // rax
  _QWORD *v80; // rbx
  _QWORD *v81; // r12
  _QWORD *v82; // rdi
  __int64 *v83; // rdi
  unsigned int v84; // r8d
  __int64 *v85; // rcx
  int v86; // r9d
  __int64 v87; // rdi
  int v88; // eax
  __int64 v89; // rcx
  int v90; // r10d
  __int64 v91; // rsi
  char v92; // al
  __int64 *v93; // rax
  __int64 v94; // rax
  __int64 *v95; // rsi
  unsigned int v96; // edi
  __int64 *v97; // rcx
  int v98; // r11d
  int v99; // eax
  int v100; // r9d
  __int64 v101; // rdi
  int v102; // r10d
  __int64 v103; // rcx
  __int64 v104; // r13
  _QWORD *v105; // rbx
  char *v106; // rax
  _QWORD *v107; // rbx
  _QWORD *v108; // rdi
  __int64 v109; // rbx
  _QWORD *v110; // r13
  char *v111; // rax
  _QWORD *v112; // rbx
  _QWORD *v113; // rdi
  __int64 v114; // rbx
  char *v115; // rax
  _QWORD *v116; // rbx
  _QWORD *v117; // rdi
  __int64 v118; // rbx
  _QWORD *v119; // r13
  char *v120; // rax
  _QWORD *v121; // rbx
  _QWORD *v122; // rdi
  __int64 *v123; // r11
  __int64 *v124; // [rsp+8h] [rbp-348h]
  __int64 *v125; // [rsp+20h] [rbp-330h]
  __int64 v126; // [rsp+28h] [rbp-328h]
  __int64 v127; // [rsp+30h] [rbp-320h]
  __int64 v128; // [rsp+38h] [rbp-318h]
  _QWORD *v129; // [rsp+38h] [rbp-318h]
  _QWORD *v130; // [rsp+38h] [rbp-318h]
  _QWORD *v131; // [rsp+38h] [rbp-318h]
  _QWORD v132[2]; // [rsp+40h] [rbp-310h] BYREF
  __int64 v133; // [rsp+50h] [rbp-300h]
  int v134; // [rsp+58h] [rbp-2F8h]
  __int64 v135; // [rsp+60h] [rbp-2F0h]
  __int64 v136; // [rsp+68h] [rbp-2E8h]
  _BYTE *v137; // [rsp+70h] [rbp-2E0h]
  __int64 v138; // [rsp+78h] [rbp-2D8h]
  _BYTE v139[16]; // [rsp+80h] [rbp-2D0h] BYREF
  _QWORD v140[2]; // [rsp+90h] [rbp-2C0h] BYREF
  __int64 v141; // [rsp+A0h] [rbp-2B0h]
  __int64 v142; // [rsp+A8h] [rbp-2A8h]
  __int64 v143; // [rsp+B0h] [rbp-2A0h]
  __int64 v144; // [rsp+B8h] [rbp-298h]
  __int64 v145; // [rsp+C0h] [rbp-290h]
  char v146; // [rsp+C8h] [rbp-288h]
  __int64 v147; // [rsp+D0h] [rbp-280h] BYREF
  _BYTE *v148; // [rsp+D8h] [rbp-278h]
  _BYTE *v149; // [rsp+E0h] [rbp-270h]
  __int64 v150; // [rsp+E8h] [rbp-268h]
  int v151; // [rsp+F0h] [rbp-260h]
  _BYTE v152[72]; // [rsp+F8h] [rbp-258h] BYREF
  __int64 v153[11]; // [rsp+140h] [rbp-210h] BYREF
  _QWORD *v154; // [rsp+198h] [rbp-1B8h]
  unsigned int v155; // [rsp+1A0h] [rbp-1B0h]
  _BYTE v156[424]; // [rsp+1A8h] [rbp-1A8h] BYREF

  v3 = a1;
  v126 = **(_QWORD **)(*a1 + 32);
  v153[0] = sub_1560340((_QWORD *)(*(_QWORD *)(v126 + 56) + 112LL), -1, "no-nans-fp-math", 0xFu);
  v4 = (_DWORD *)sub_155D8B0(v153);
  v6 = v5;
  v7 = 0;
  if ( v6 == 4 )
    v7 = *v4 == 1702195828;
  v8 = *a1;
  *((_BYTE *)a1 + 448) = v7;
  v124 = *(__int64 **)(v8 + 40);
  if ( *(__int64 **)(v8 + 32) == v124 )
    goto LABEL_16;
  v125 = *(__int64 **)(v8 + 32);
LABEL_5:
  v9 = *(_QWORD *)(*v125 + 48);
  v128 = *v125 + 40;
  if ( v128 == v9 )
    goto LABEL_15;
  v127 = *v125;
  v10 = (__int64)v3;
  while ( 1 )
  {
    if ( !v9 )
      BUG();
    v11 = *(_BYTE *)(v9 - 8);
    v12 = v9 - 24;
    if ( v11 != 77 )
      break;
    if ( (*(_BYTE *)(*(_QWORD *)(v9 - 24) + 8LL) & 0xFB) != 0xB
      && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v9 - 24) + 8LL) - 1) > 5u )
    {
      v109 = *(_QWORD *)v10;
      v110 = *(_QWORD **)(v10 + 56);
      v111 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
      sub_1BF1750((__int64)v153, (__int64)v111, (__int64)"CFGNotUnderstood", 16, v109, v12);
      sub_15CAB20((__int64)v153, "loop control flow is not understood by vectorizer", 0x31u);
      sub_143AA50(v110, (__int64)v153);
      v112 = v154;
      v153[0] = (__int64)&unk_49ECF68;
      v21 = &v154[11 * v155];
      if ( v154 == v21 )
        goto LABEL_30;
      do
      {
        v21 -= 11;
        v113 = (_QWORD *)v21[4];
        if ( v113 != v21 + 6 )
          j_j___libc_free_0(v113, v21[6] + 1LL);
        if ( (_QWORD *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21, v21[2] + 1LL);
      }
      while ( v112 != v21 );
      goto LABEL_29;
    }
    if ( v127 != v126 )
    {
      if ( !(unsigned __int8)sub_1BF1270(*(_QWORD *)v10, v9 - 24, v10 + 376) )
        goto LABEL_13;
      v68 = *(_QWORD *)v10;
      v69 = *(_QWORD **)(v10 + 56);
      v70 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
      sub_1BF1750((__int64)v153, (__int64)v70, (__int64)"NeitherInductionNorReduction", 28, v68, v12);
      sub_15CAB20((__int64)v153, "value could not be identified as an induction or reduction variable", 0x43u);
      sub_143AA50(v69, (__int64)v153);
      v71 = v154;
      v153[0] = (__int64)&unk_49ECF68;
      v21 = &v154[11 * v155];
      if ( v154 == v21 )
        goto LABEL_30;
      do
      {
        v21 -= 11;
        v72 = (_QWORD *)v21[4];
        if ( v72 != v21 + 6 )
          j_j___libc_free_0(v72, v21[6] + 1LL);
        if ( (_QWORD *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21, v21[2] + 1LL);
      }
      while ( v71 != v21 );
      goto LABEL_29;
    }
    if ( (*(_DWORD *)(v9 - 4) & 0xFFFFFFF) != 2 )
    {
      v104 = *(_QWORD *)v10;
      v105 = *(_QWORD **)(v10 + 56);
      v106 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
      sub_1BF1750((__int64)v153, (__int64)v106, (__int64)"CFGNotUnderstood", 16, v104, v12);
      sub_15CAB20((__int64)v153, "control flow not understood by vectorizer", 0x29u);
      sub_143AA50(v105, (__int64)v153);
      v107 = v154;
      v153[0] = (__int64)&unk_49ECF68;
      v21 = &v154[11 * v155];
      if ( v154 == v21 )
        goto LABEL_30;
      do
      {
        v21 -= 11;
        v108 = (_QWORD *)v21[4];
        if ( v108 != v21 + 6 )
          j_j___libc_free_0(v108, v21[6] + 1LL);
        if ( (_QWORD *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21, v21[2] + 1LL);
      }
      while ( v107 != v21 );
      goto LABEL_29;
    }
    v40 = *(_QWORD *)(v10 + 32);
    v41 = *(_QWORD *)(v10 + 472);
    v42 = *(_QWORD *)(v10 + 480);
    v43 = *(_QWORD *)v10;
    v140[0] = 6;
    v140[1] = 0;
    v141 = 0;
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v145 = 0;
    v146 = 0;
    v147 = 0;
    v148 = v152;
    v149 = v152;
    v150 = 8;
    v151 = 0;
    if ( !(unsigned __int8)sub_1B1CF40(v9 - 24, v43, (__int64)v140, v41, v42, v40) )
    {
      v57 = *(_QWORD *)v10;
      v58 = *(_QWORD *)(v10 + 16);
      v137 = v139;
      v132[0] = 6;
      v132[1] = 0;
      v133 = 0;
      v134 = 0;
      v135 = 0;
      v136 = 0;
      v138 = 0x200000000LL;
      if ( (unsigned __int8)sub_1B1D140(v9 - 24, v57, v58, (__int64)v132, 0, a2, a3) )
      {
        sub_1BF60E0(v10, (_QWORD *)(v9 - 24), (__int64)v132, v10 + 376, v59, v60);
        if ( v136 )
        {
          v92 = *(_BYTE *)(v136 + 17) >> 1;
          if ( ((v92 & 0x3F) != 0x3F || (v92 & 0x40) == 0) && !*(_BYTE *)(v10 + 448) )
          {
            v94 = *(_QWORD *)(v10 + 456);
            if ( !*(_QWORD *)(v94 + 8) )
              *(_QWORD *)(v94 + 8) = v136;
          }
        }
        goto LABEL_148;
      }
      if ( (unsigned __int8)sub_1B1D750(v9 - 24, *(_QWORD *)v10, v10 + 336, *(_QWORD *)(v10 + 32)) )
      {
        v93 = *(__int64 **)(v10 + 240);
        if ( *(__int64 **)(v10 + 248) == v93 )
        {
          v95 = &v93[*(unsigned int *)(v10 + 260)];
          v96 = *(_DWORD *)(v10 + 260);
          if ( v93 != v95 )
          {
            v97 = 0;
            do
            {
              if ( v12 == *v93 )
                goto LABEL_148;
              if ( *v93 == -2 )
                v97 = v93;
              ++v93;
            }
            while ( v95 != v93 );
            if ( v97 )
            {
              *v97 = v12;
              --*(_DWORD *)(v10 + 264);
              ++*(_QWORD *)(v10 + 232);
              goto LABEL_148;
            }
          }
          if ( v96 < *(_DWORD *)(v10 + 256) )
          {
            *(_DWORD *)(v10 + 260) = v96 + 1;
            *v95 = v12;
            ++*(_QWORD *)(v10 + 232);
            goto LABEL_148;
          }
        }
        sub_16CCBA0(v10 + 232, v9 - 24);
      }
      else
      {
        if ( !(unsigned __int8)sub_1B1D140(v9 - 24, *(_QWORD *)v10, *(_QWORD *)(v10 + 16), (__int64)v132, 1, a2, a3) )
        {
          v63 = *(_QWORD *)v10;
          v129 = *(_QWORD **)(v10 + 56);
          v64 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
          sub_1BF1750((__int64)v153, (__int64)v64, (__int64)"NonReductionValueUsedOutsideLoop", 32, v63, v12);
          sub_15CAB20((__int64)v153, "value that could not be identified as reduction is used outside the loop", 0x48u);
          sub_143AA50(v129, (__int64)v153);
          v65 = v154;
          v153[0] = (__int64)&unk_49ECF68;
          v66 = &v154[11 * v155];
          if ( v154 != v66 )
          {
            do
            {
              v66 -= 11;
              v67 = (_QWORD *)v66[4];
              if ( v67 != v66 + 6 )
                j_j___libc_free_0(v67, v66[6] + 1LL);
              if ( (_QWORD *)*v66 != v66 + 2 )
                j_j___libc_free_0(*v66, v66[2] + 1LL);
            }
            while ( v65 != v66 );
            v66 = v154;
          }
          if ( v66 != (_QWORD *)v156 )
            _libc_free((unsigned __int64)v66);
          if ( v137 != v139 )
            _libc_free((unsigned __int64)v137);
          if ( v133 != 0 && v133 != -8 && v133 != -16 )
            sub_1649B30(v132);
          if ( v149 != v148 )
            _libc_free((unsigned __int64)v149);
          if ( v141 != 0 && v141 != -8 && v141 != -16 )
            sub_1649B30(v140);
          return 0;
        }
        sub_1BF60E0(v10, (_QWORD *)(v9 - 24), (__int64)v132, v10 + 376, v61, v62);
      }
LABEL_148:
      if ( v137 != v139 )
        _libc_free((unsigned __int64)v137);
      if ( v133 != 0 && v133 != -8 && v133 != -16 )
        sub_1649B30(v132);
      goto LABEL_80;
    }
    if ( v144 )
    {
      v44 = *(_QWORD *)(v10 + 456);
      if ( !*(_QWORD *)(v44 + 8) )
        *(_QWORD *)(v44 + 8) = v144;
    }
    v45 = v142;
    v46 = *(__int64 **)(v10 + 384);
    if ( *(__int64 **)(v10 + 392) != v46 )
      goto LABEL_67;
    v83 = &v46[*(unsigned int *)(v10 + 404)];
    v84 = *(_DWORD *)(v10 + 404);
    if ( v46 != v83 )
    {
      v85 = 0;
      do
      {
        if ( v142 == *v46 )
          goto LABEL_68;
        if ( *v46 == -2 )
          v85 = v46;
        ++v46;
      }
      while ( v83 != v46 );
      if ( v85 )
      {
        *v85 = v142;
        v47 = *(_DWORD *)(v10 + 96);
        v48 = v10 + 72;
        --*(_DWORD *)(v10 + 408);
        ++*(_QWORD *)(v10 + 376);
        if ( !v47 )
        {
LABEL_137:
          ++*(_QWORD *)(v10 + 72);
          goto LABEL_138;
        }
        goto LABEL_69;
      }
    }
    if ( v84 < *(_DWORD *)(v10 + 400) )
    {
      *(_DWORD *)(v10 + 404) = v84 + 1;
      *v83 = v45;
      ++*(_QWORD *)(v10 + 376);
    }
    else
    {
LABEL_67:
      sub_16CCBA0(v10 + 376, v142);
    }
LABEL_68:
    v47 = *(_DWORD *)(v10 + 96);
    v48 = v10 + 72;
    if ( !v47 )
      goto LABEL_137;
LABEL_69:
    v49 = *(_QWORD *)(v10 + 80);
    v50 = (v47 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    LODWORD(v51) = 5 * v50;
    v52 = v49 + 176LL * v50;
    v53 = *(_QWORD *)v52;
    if ( v12 == *(_QWORD *)v52 )
    {
      v54 = *(_QWORD *)(v52 + 24);
      goto LABEL_71;
    }
    v98 = 1;
    v51 = 0;
    while ( 1 )
    {
      if ( v53 == -8 )
      {
        v99 = *(_DWORD *)(v10 + 88);
        if ( v51 )
          v52 = v51;
        ++*(_QWORD *)(v10 + 72);
        v88 = v99 + 1;
        if ( 4 * v88 < 3 * v47 )
        {
          v49 = v47 - *(_DWORD *)(v10 + 92) - v88;
          if ( (unsigned int)v49 > v47 >> 3 )
            goto LABEL_177;
          sub_1BA42A0(v48, v47);
          v100 = *(_DWORD *)(v10 + 96);
          if ( v100 )
          {
            LODWORD(v51) = v100 - 1;
            v101 = *(_QWORD *)(v10 + 80);
            v91 = 0;
            v102 = 1;
            v49 = (unsigned int)v51 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v52 = v101 + 176 * v49;
            v88 = *(_DWORD *)(v10 + 88) + 1;
            v103 = *(_QWORD *)v52;
            if ( v12 == *(_QWORD *)v52 )
              goto LABEL_177;
            while ( v103 != -8 )
            {
              if ( !v91 && v103 == -16 )
                v91 = v52;
              LODWORD(v53) = v102 + 1;
              v49 = (unsigned int)v51 & ((_DWORD)v49 + v102);
              v52 = v101 + 176 * v49;
              v103 = *(_QWORD *)v52;
              if ( v12 == *(_QWORD *)v52 )
                goto LABEL_177;
              ++v102;
            }
LABEL_142:
            if ( v91 )
              v52 = v91;
LABEL_177:
            *(_DWORD *)(v10 + 88) = v88;
            if ( *(_QWORD *)v52 != -8 )
              --*(_DWORD *)(v10 + 92);
            *(_QWORD *)v52 = v12;
            v56 = (unsigned __int64 *)(v52 + 8);
            memset((void *)(v52 + 8), 0, 0xA8u);
            LODWORD(v54) = 0;
            *(_QWORD *)(v52 + 80) = v52 + 112;
            *(_QWORD *)(v52 + 8) = 6;
            v55 = v141;
            *(_QWORD *)(v52 + 88) = v52 + 112;
            *(_QWORD *)(v52 + 96) = 8;
            if ( v55 )
            {
LABEL_75:
              *(_QWORD *)(v52 + 24) = v55;
              LOBYTE(v54) = v55 != 0;
              if ( v55 != 0 && v55 != -8 && v55 != -16 )
                sub_1649AC0(v56, v140[0] & 0xFFFFFFFFFFFFFFF8LL);
            }
            goto LABEL_78;
          }
          goto LABEL_233;
        }
LABEL_138:
        sub_1BA42A0(v48, 2 * v47);
        v86 = *(_DWORD *)(v10 + 96);
        if ( v86 )
        {
          LODWORD(v51) = v86 - 1;
          v87 = *(_QWORD *)(v10 + 80);
          v49 = (unsigned int)v51 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v52 = v87 + 176 * v49;
          v88 = *(_DWORD *)(v10 + 88) + 1;
          v89 = *(_QWORD *)v52;
          if ( v12 == *(_QWORD *)v52 )
            goto LABEL_177;
          v90 = 1;
          v91 = 0;
          while ( v89 != -8 )
          {
            if ( !v91 && v89 == -16 )
              v91 = v52;
            LODWORD(v53) = v90 + 1;
            v49 = (unsigned int)v51 & ((_DWORD)v49 + v90);
            v52 = v87 + 176 * v49;
            v89 = *(_QWORD *)v52;
            if ( v12 == *(_QWORD *)v52 )
              goto LABEL_177;
            ++v90;
          }
          goto LABEL_142;
        }
LABEL_233:
        ++*(_DWORD *)(v10 + 88);
        BUG();
      }
      if ( v53 != -16 || v51 )
        v52 = v51;
      LODWORD(v51) = v98 + 1;
      v50 = (v47 - 1) & (v98 + v50);
      v123 = (__int64 *)(v49 + 176LL * v50);
      v53 = *v123;
      if ( v12 == *v123 )
        break;
      v98 = v51;
      v51 = v52;
      v52 = v49 + 176LL * v50;
    }
    v54 = v123[3];
    v52 = v49 + 176LL * v50;
LABEL_71:
    v55 = v141;
    if ( v54 != v141 )
    {
      v56 = (unsigned __int64 *)(v52 + 8);
      LOBYTE(v49) = v54 != 0;
      if ( v54 != 0 && v54 != -8 && v54 != -16 )
      {
        sub_1649B30((_QWORD *)(v52 + 8));
        v55 = v141;
      }
      goto LABEL_75;
    }
LABEL_78:
    *(_QWORD *)(v52 + 32) = v142;
    *(_QWORD *)(v52 + 40) = v143;
    *(_QWORD *)(v52 + 48) = v144;
    *(_QWORD *)(v52 + 56) = v145;
    *(_BYTE *)(v52 + 64) = v146;
    if ( (__int64 *)(v52 + 72) != &v147 )
      sub_16CCD50(v52 + 72, (__int64)&v147, v49, v54, v53, v51);
LABEL_80:
    if ( v149 != v148 )
      _libc_free((unsigned __int64)v149);
    if ( v141 != -8 && v141 != 0 && v141 != -16 )
      sub_1649B30(v140);
LABEL_13:
    v9 = *(_QWORD *)(v9 + 8);
    if ( v128 == v9 )
    {
      v3 = (__int64 *)v10;
LABEL_15:
      if ( v124 == ++v125 )
      {
LABEL_16:
        v13 = (_QWORD *)v3[8];
        if ( v13 )
        {
          v14 = 1;
          if ( v3[46] != *v13 )
            v3[8] = 0;
          return v14;
        }
        v14 = 1;
        if ( v3[18] != v3[17] )
          return v14;
        v118 = *v3;
        v119 = (_QWORD *)v3[7];
        v120 = sub_1BF18B0(v3[58]);
        sub_1BF1750((__int64)v153, (__int64)v120, (__int64)"NoInductionVariable", 19, v118, 0);
        sub_15CAB20((__int64)v153, "loop induction variable could not be identified", 0x2Fu);
        sub_143AA50(v119, (__int64)v153);
        v21 = v154;
        v153[0] = (__int64)&unk_49ECF68;
        v121 = &v154[11 * v155];
        if ( v154 != v121 )
        {
          do
          {
            v121 -= 11;
            v122 = (_QWORD *)v121[4];
            if ( v122 != v121 + 6 )
              j_j___libc_free_0(v122, v121[6] + 1LL);
            if ( (_QWORD *)*v121 != v121 + 2 )
              j_j___libc_free_0(*v121, v121[2] + 1LL);
          }
          while ( v21 != v121 );
          goto LABEL_29;
        }
LABEL_30:
        if ( v21 != (_QWORD *)v156 )
          _libc_free((unsigned __int64)v21);
        return 0;
      }
      goto LABEL_5;
    }
  }
  v15 = 0;
  if ( v11 != 78 )
    goto LABEL_41;
  if ( (unsigned int)sub_14C3B40(v9 - 24, *(__int64 **)(v10 + 24)) )
    goto LABEL_39;
  v16 = *(_QWORD *)(v9 - 48);
  if ( *(_BYTE *)(v9 - 8) == 78 )
  {
    if ( !*(_BYTE *)(v16 + 16) )
    {
      if ( (*(_BYTE *)(v16 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v16 + 36) - 35) > 3 )
        goto LABEL_37;
      goto LABEL_39;
    }
    goto LABEL_23;
  }
  if ( *(_BYTE *)(v16 + 16) )
    goto LABEL_23;
LABEL_37:
  v24 = *(__int64 **)(v10 + 24);
  if ( !v24 || (v25 = (char *)sub_1649960(v16), !(unsigned __int8)sub_149D110(*v24, v25, v26)) )
  {
LABEL_23:
    v17 = *(_QWORD *)v10;
    v18 = *(_QWORD **)(v10 + 56);
    v19 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
    sub_1BF1750((__int64)v153, (__int64)v19, (__int64)"CantVectorizeCall", 17, v17, v12);
    sub_15CAB20((__int64)v153, "call instruction cannot be vectorized", 0x25u);
    sub_143AA50(v18, (__int64)v153);
    v20 = v154;
    v153[0] = (__int64)&unk_49ECF68;
    v21 = &v154[11 * v155];
    if ( v154 == v21 )
      goto LABEL_30;
    do
    {
      v21 -= 11;
      v22 = (_QWORD *)v21[4];
      if ( v22 != v21 + 6 )
        j_j___libc_free_0(v22, v21[6] + 1LL);
      if ( (_QWORD *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21, v21[2] + 1LL);
    }
    while ( v20 != v21 );
    goto LABEL_29;
  }
LABEL_39:
  v27 = sub_14C3B40(v9 - 24, *(__int64 **)(v10 + 24));
  if ( sub_14C3B20(v27, 1) )
  {
    v73 = *(_QWORD *)(v10 + 16);
    v74 = *(_QWORD *)v10;
    v75 = *(_QWORD *)(v73 + 112);
    v76 = sub_1494E70(v73, *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 - 4) & 0xFFFFFFF)) - 24), a2, a3);
    LOBYTE(v77) = sub_146CEE0(v75, (__int64)v76, v74);
    if ( !(_BYTE)v77 )
    {
      v14 = v77;
      v78 = *(_QWORD *)v10;
      v130 = *(_QWORD **)(v10 + 56);
      v79 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
      sub_1BF1750((__int64)v153, (__int64)v79, (__int64)"CantVectorizeIntrinsic", 22, v78, v12);
      sub_15CAB20((__int64)v153, "intrinsic instruction cannot be vectorized", 0x2Au);
      sub_143AA50(v130, (__int64)v153);
      v80 = v154;
      v153[0] = (__int64)&unk_49ECF68;
      v81 = &v154[11 * v155];
      if ( v154 != v81 )
      {
        do
        {
          v81 -= 11;
          v82 = (_QWORD *)v81[4];
          if ( v82 != v81 + 6 )
            j_j___libc_free_0(v82, v81[6] + 1LL);
          if ( (_QWORD *)*v81 != v81 + 2 )
            j_j___libc_free_0(*v81, v81[2] + 1LL);
        }
        while ( v80 != v81 );
        goto LABEL_126;
      }
      goto LABEL_127;
    }
  }
  v15 = v9 - 24;
LABEL_41:
  if ( !(unsigned __int8)sub_1643F10(*(_QWORD *)(v9 - 24)) && *(_BYTE *)(*(_QWORD *)(v9 - 24) + 8LL)
    || (v28 = *(unsigned __int8 *)(v9 - 8), (_BYTE)v28 == 83) )
  {
    v35 = *(_QWORD *)v10;
    v36 = *(_QWORD **)(v10 + 56);
    v37 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
    sub_1BF1750((__int64)v153, (__int64)v37, (__int64)"CantVectorizeInstructionReturnType", 34, v35, v12);
    sub_15CAB20((__int64)v153, "instruction return type cannot be vectorized", 0x2Cu);
    sub_143AA50(v36, (__int64)v153);
    v38 = v154;
    v153[0] = (__int64)&unk_49ECF68;
    v21 = &v154[11 * v155];
    if ( v154 == v21 )
      goto LABEL_30;
    do
    {
      v21 -= 11;
      v39 = (_QWORD *)v21[4];
      if ( v39 != v21 + 6 )
        j_j___libc_free_0(v39, v21[6] + 1LL);
      if ( (_QWORD *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21, v21[2] + 1LL);
    }
    while ( v38 != v21 );
    goto LABEL_29;
  }
  if ( (_BYTE)v28 != 55 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v9 - 24) + 8LL) - 1) <= 5u
      && (v15 || (unsigned int)(v28 - 35) <= 0x11)
      && !sub_15F2480(v9 - 24) )
    {
      *(_BYTE *)(*(_QWORD *)(v10 + 464) + 64LL) = 1;
    }
LABEL_45:
    if ( !(unsigned __int8)sub_1BF1270(*(_QWORD *)v10, v9 - 24, v10 + 376) )
      goto LABEL_13;
    v30 = *(_QWORD *)v10;
    v31 = *(_QWORD **)(v10 + 56);
    v32 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
    sub_1BF1750((__int64)v153, (__int64)v32, (__int64)"ValueUsedOutsideLoop", 20, v30, v12);
    sub_15CAB20((__int64)v153, "value cannot be used outside the loop", 0x25u);
    sub_143AA50(v31, (__int64)v153);
    v33 = v154;
    v153[0] = (__int64)&unk_49ECF68;
    v21 = &v154[11 * v155];
    if ( v154 == v21 )
      goto LABEL_30;
    do
    {
      v21 -= 11;
      v34 = (_QWORD *)v21[4];
      if ( v34 != v21 + 6 )
        j_j___libc_free_0(v34, v21[6] + 1LL);
      if ( (_QWORD *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21, v21[2] + 1LL);
    }
    while ( v33 != v21 );
LABEL_29:
    v21 = v154;
    goto LABEL_30;
  }
  v29 = sub_1643F10(**(_QWORD **)(v9 - 72));
  if ( (_BYTE)v29 )
    goto LABEL_45;
  v14 = v29;
  v114 = *(_QWORD *)v10;
  v131 = *(_QWORD **)(v10 + 56);
  v115 = sub_1BF18B0(*(_QWORD *)(v10 + 464));
  sub_1BF1750((__int64)v153, (__int64)v115, (__int64)"CantVectorizeStore", 18, v114, v12);
  sub_15CAB20((__int64)v153, "store instruction cannot be vectorized", 0x26u);
  sub_143AA50(v131, (__int64)v153);
  v116 = v154;
  v153[0] = (__int64)&unk_49ECF68;
  v81 = &v154[11 * v155];
  if ( v154 != v81 )
  {
    do
    {
      v81 -= 11;
      v117 = (_QWORD *)v81[4];
      if ( v117 != v81 + 6 )
        j_j___libc_free_0(v117, v81[6] + 1LL);
      if ( (_QWORD *)*v81 != v81 + 2 )
        j_j___libc_free_0(*v81, v81[2] + 1LL);
    }
    while ( v116 != v81 );
LABEL_126:
    v81 = v154;
  }
LABEL_127:
  if ( v81 != (_QWORD *)v156 )
    _libc_free((unsigned __int64)v81);
  return v14;
}
