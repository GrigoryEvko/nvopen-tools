// Function: sub_26A1DB0
// Address: 0x26a1db0
//
__int64 __fastcall sub_26A1DB0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v4; // r13
  __int64 v5; // rbx
  __int64 j; // r14
  __int64 result; // rax
  int v8; // eax
  int v9; // eax
  unsigned __int64 v10; // rsi
  __m128i v11; // xmm0
  unsigned __int64 v12; // rsi
  int v13; // eax
  unsigned __int64 v14; // rsi
  __int128 v15; // kr00_16
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rcx
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  __m128i v28; // xmm2
  __int64 v29; // rsi
  _QWORD *v30; // rax
  _BYTE *v31; // rax
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rdi
  int v35; // edx
  void *v36; // rcx
  int v37; // eax
  __int64 v38; // rax
  unsigned __int8 v39; // cl
  unsigned __int64 v40; // rdx
  __int64 v41; // r11
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // r8
  int v49; // eax
  void (*v50)(); // rdx
  int v51; // edx
  __m128i v52; // xmm4
  __int64 v53; // rsi
  _QWORD *v54; // rax
  __int64 v55; // rax
  int v56; // edx
  __int64 v57; // rdi
  int v58; // edx
  unsigned int v59; // eax
  void *v60; // rcx
  __int64 v61; // rax
  unsigned __int8 v62; // cl
  unsigned __int64 v63; // rdx
  __int64 v64; // r10
  __int64 v65; // r13
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r14
  int v71; // eax
  void (*v72)(); // rdx
  int v73; // r14d
  __int64 v74; // rax
  __m128i v75; // rax
  __m128i v76; // xmm7
  __int64 v77; // rsi
  _QWORD *v78; // rax
  __int64 v79; // rax
  int v80; // edx
  int v81; // edx
  unsigned int v82; // edi
  void *v83; // rcx
  int v84; // eax
  __int64 v85; // rax
  unsigned __int8 v86; // cl
  unsigned __int64 v87; // rdx
  __int64 *v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 v93; // rax
  __int64 v94; // r8
  int v95; // eax
  void (*v96)(); // rdx
  int v97; // edx
  char v98; // al
  __int64 (__fastcall *v99)(__int64); // rax
  __int64 v100; // rdi
  __int64 (__fastcall *v101)(__int64); // rax
  __int64 v102; // rdi
  char v103; // al
  __int64 (__fastcall *v104)(__int64); // rax
  _BYTE *v105; // rdi
  void (*v106)(void); // rax
  __int64 v107; // rax
  unsigned __int64 v108; // rdi
  unsigned __int8 v109; // al
  __int64 v110; // rax
  int v111; // edx
  int v112; // edx
  int v113; // ecx
  unsigned int i; // eax
  __int64 v115; // r8
  unsigned int v116; // eax
  unsigned __int64 v117; // rdi
  unsigned __int8 v118; // al
  __int64 v119; // rax
  int v120; // ecx
  int v121; // ecx
  int v122; // esi
  unsigned int k; // eax
  __int64 v124; // r8
  unsigned int v125; // eax
  __int64 v126; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v127; // [rsp+10h] [rbp-A0h]
  __int64 v128; // [rsp+18h] [rbp-98h]
  bool v129; // [rsp+18h] [rbp-98h]
  __int64 v130; // [rsp+18h] [rbp-98h]
  __int64 v131; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v132; // [rsp+20h] [rbp-90h]
  char v133; // [rsp+20h] [rbp-90h]
  __int64 v134; // [rsp+20h] [rbp-90h]
  char v135; // [rsp+2Bh] [rbp-85h]
  unsigned int v136; // [rsp+2Ch] [rbp-84h]
  unsigned __int8 *v137; // [rsp+30h] [rbp-80h]
  __int64 v138; // [rsp+38h] [rbp-78h]
  __int64 v139; // [rsp+38h] [rbp-78h]
  __int64 v140; // [rsp+38h] [rbp-78h]
  int v141; // [rsp+38h] [rbp-78h]
  __int64 v142; // [rsp+38h] [rbp-78h]
  char v143; // [rsp+38h] [rbp-78h]
  __int64 v144; // [rsp+38h] [rbp-78h]
  __int64 v145; // [rsp+38h] [rbp-78h]
  __int64 v146; // [rsp+38h] [rbp-78h]
  int v147; // [rsp+38h] [rbp-78h]
  __m128i v148; // [rsp+50h] [rbp-60h] BYREF
  _BYTE v149[72]; // [rsp+60h] [rbp-50h] BYREF

  if ( !byte_4FF4EA8 )
  {
    sub_250D230((unsigned __int64 *)v149, a2, 4, 0);
    sub_269D460(a1, *(__int64 *)v149, *(__int64 *)&v149[8], 0, 2);
  }
  *(_QWORD *)&v149[8] = 0;
  *(_QWORD *)v149 = a2 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  sub_269E5E0(a1, *(__int64 *)v149, *(__int64 *)&v149[8], 0, 2, 0, 1);
  if ( !byte_4FF4EA8 )
  {
    sub_250D230((unsigned __int64 *)v149, a2, 4, 0);
    sub_269DF00(a1, *(__int64 *)v149, *(__int64 *)&v149[8], 0, 2, 0, 1);
  }
  if ( (unsigned __int8)sub_B2D610(a2, 6) )
  {
    sub_250D230((unsigned __int64 *)v149, a2, 4, 0);
    v148 = _mm_loadu_si128((const __m128i *)v149);
    if ( !(unsigned __int8)sub_250E300(a1, &v148) )
      v148.m128i_i64[1] = 0;
    v52 = _mm_loadu_si128(&v148);
    v53 = (__int64)v149;
    *(_QWORD *)v149 = &unk_438A675;
    *(__m128i *)&v149[8] = v52;
    v54 = sub_25134D0(a1 + 136, (__int64 *)v149);
    if ( !v54 || !v54[3] )
    {
      v55 = *(_QWORD *)(a1 + 4376);
      if ( !v55 )
        goto LABEL_88;
      v56 = *(_DWORD *)(v55 + 24);
      v57 = *(_QWORD *)(v55 + 8);
      if ( !v56 )
        goto LABEL_6;
      v58 = v56 - 1;
      v53 = 1;
      v59 = v58 & (((unsigned int)&unk_438A675 >> 9) ^ ((unsigned int)&unk_438A675 >> 4));
      v60 = *(void **)(v57 + 8LL * v59);
      if ( v60 == &unk_438A675 )
      {
LABEL_88:
        v61 = sub_25096F0(&v148);
        if ( v61 )
        {
          v142 = v61;
          if ( (unsigned __int8)sub_B2D610(v61, 20) )
            goto LABEL_6;
          v53 = 48;
          if ( (unsigned __int8)sub_B2D610(v142, 48) )
            goto LABEL_6;
        }
        if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
          goto LABEL_6;
        if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
        {
          v137 = sub_250CBE0(v148.m128i_i64, v53);
          v62 = sub_2509800(&v148);
          if ( v62 > 7u || ((1LL << v62) & 0xA8) == 0 )
            goto LABEL_98;
          v63 = v148.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
          if ( (v148.m128i_i8[0] & 3) == 3 )
            v63 = *(_QWORD *)(v63 + 24);
          if ( **(_BYTE **)(v63 - 32) != 25 )
          {
LABEL_98:
            v143 = sub_250CC70(a1, v148.m128i_i64);
            if ( v143 )
            {
              if ( !v137 || *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_266EE70(*(_QWORD *)(a1 + 200), (__int64)v137) )
                goto LABEL_102;
              v108 = v148.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
              if ( (v148.m128i_i8[0] & 3) == 3 )
                v108 = *(_QWORD *)(v108 + 24);
              v109 = *(_BYTE *)v108;
              if ( *(_BYTE *)v108 )
              {
                if ( v109 == 22 )
                {
                  v108 = *(_QWORD *)(v108 + 24);
                }
                else if ( v109 <= 0x1Cu )
                {
                  v108 = 0;
                }
                else
                {
                  v110 = sub_B43CB0(v108);
                  v64 = *(_QWORD *)(a1 + 200);
                  v108 = v110;
                }
              }
              if ( !*(_DWORD *)(v64 + 40) )
                goto LABEL_102;
              v111 = *(_DWORD *)(v64 + 24);
              if ( v111 )
              {
                v112 = v111 - 1;
                v113 = 1;
                for ( i = v112 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4)); ; i = v112 & v116 )
                {
                  v115 = *(_QWORD *)(*(_QWORD *)(v64 + 8) + 8LL * i);
                  if ( v108 == v115 )
                    break;
                  if ( v115 == -4096 )
                    goto LABEL_156;
                  v116 = v113 + i;
                  ++v113;
                }
                goto LABEL_102;
              }
            }
          }
        }
LABEL_156:
        v143 = 0;
LABEL_102:
        v65 = sub_25663F0(&v148, a1);
        *(_QWORD *)v149 = &unk_438A675;
        *(__m128i *)&v149[8] = _mm_loadu_si128((const __m128i *)(v65 + 72));
        *sub_2519B70(a1 + 136, (__int64)v149) = v65;
        if ( *(_DWORD *)(a1 + 3552) <= 1u )
        {
          *(_QWORD *)v149 = v65 & 0xFFFFFFFFFFFFFFFBLL;
          sub_269CF50(a1 + 224, (unsigned __int64 *)v149, v66, v67, v68, v69);
          if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v65) )
            goto LABEL_153;
        }
        *(_QWORD *)v149 = v65;
        v70 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250AFB0, (__int64)v149);
        v71 = *(_DWORD *)(a1 + 3556);
        *(_DWORD *)(a1 + 3556) = v71 + 1;
        v72 = *(void (**)())(*(_QWORD *)v65 + 24LL);
        if ( v72 != nullsub_1516 )
        {
          ((void (__fastcall *)(__int64, __int64))v72)(v65, a1);
          v71 = *(_DWORD *)(a1 + 3556) - 1;
        }
        *(_DWORD *)(a1 + 3556) = v71;
        if ( v70 )
          sub_C9AF60(v70);
        if ( v143 )
        {
          v73 = *(_DWORD *)(a1 + 3552);
          *(_DWORD *)(a1 + 3552) = 1;
          sub_251C580(a1, v65);
          *(_DWORD *)(a1 + 3552) = v73;
        }
        else
        {
LABEL_153:
          v101 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v65 + 40LL);
          if ( v101 == sub_2505F20 )
            v102 = v65 + 88;
          else
            v102 = v101(v65);
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v102 + 40LL))(v102);
        }
        goto LABEL_6;
      }
      while ( v60 != (void *)-4096LL )
      {
        v59 = v58 & (v53 + v59);
        v60 = *(void **)(v57 + 8LL * v59);
        if ( v60 == &unk_438A675 )
          goto LABEL_88;
        v53 = (unsigned int)(v53 + 1);
      }
    }
  }
LABEL_6:
  v4 = a2 + 72;
  v5 = *(_QWORD *)(a2 + 80);
  if ( v4 == v5 )
  {
    j = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    while ( 1 )
    {
      j = *(_QWORD *)(v5 + 32);
      if ( j != v5 + 24 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        break;
      if ( !v5 )
        BUG();
    }
  }
  result = (unsigned int)&unk_438A664 >> 9;
  v136 = result ^ ((unsigned int)&unk_438A664 >> 4);
  if ( v4 != v5 )
  {
    while ( 1 )
    {
      if ( !j )
        BUG();
      v8 = *(unsigned __int8 *)(j - 24);
      if ( (_BYTE)v8 == 61 )
      {
        v148.m128i_i8[0] = 0;
        v9 = *(unsigned __int8 *)(j - 24);
        v10 = j - 24;
        if ( (_BYTE)v9 == 22 )
        {
          *(_OWORD *)v149 = v10 & 0xFFFFFFFFFFFFFFFCLL;
          nullsub_1518();
          v11 = _mm_loadu_si128((const __m128i *)v149);
        }
        else if ( (unsigned __int8)v9 > 0x1Cu
               && (v26 = (unsigned int)(v9 - 34), (unsigned __int8)v26 <= 0x33u)
               && (v27 = 0x8000000000041LL, _bittest64(&v27, v26)) )
        {
          *(_OWORD *)v149 = v10 & 0xFFFFFFFFFFFFFFFCLL | 1;
          nullsub_1518();
          v11 = _mm_loadu_si128((const __m128i *)v149);
        }
        else
        {
          sub_250D230((unsigned __int64 *)v149, v10, 1, 0);
          v11 = _mm_loadu_si128((const __m128i *)v149);
        }
        *(__m128i *)v149 = v11;
        sub_2527850(a1, (__m128i *)v149, 0, &v148, 2u);
        v12 = *(_QWORD *)(j - 56);
        v13 = *(unsigned __int8 *)v12;
        if ( (_BYTE)v13 == 22 )
        {
          *(_QWORD *)&v149[8] = 0;
          v14 = v12 & 0xFFFFFFFFFFFFFFFCLL;
          goto LABEL_19;
        }
        if ( (unsigned __int8)v13 > 0x1Cu
          && (v24 = (unsigned int)(v13 - 34), (unsigned __int8)v24 <= 0x33u)
          && (v25 = 0x8000000000041LL, _bittest64(&v25, v24)) )
        {
          *(_QWORD *)&v149[8] = 0;
          v14 = v12 & 0xFFFFFFFFFFFFFFFCLL | 1;
LABEL_19:
          *(_QWORD *)v149 = v14;
          nullsub_1518();
          v15 = *(_OWORD *)v149;
        }
        else
        {
          sub_250D230((unsigned __int64 *)v149, v12, 1, 0);
          v15 = *(_OWORD *)v149;
        }
        sub_269D9A0(a1, v15, *((__int64 *)&v15 + 1), 0, 2, 0, 1);
        goto LABEL_21;
      }
      if ( (unsigned __int8)(v8 - 34) > 0x33u )
        goto LABEL_21;
      v17 = 0x8000000000041LL;
      if ( _bittest64(&v17, (unsigned int)(v8 - 34)) )
        break;
LABEL_34:
      switch ( (_BYTE)v8 )
      {
        case '>':
          v18 = sub_250D2C0(j - 24, 0);
          sub_251BBC0(a1, v18, v19, 0, 2, 0, 1);
          v20 = sub_250D2C0(*(_QWORD *)(j - 56), 0);
          sub_269D9A0(a1, v20, v21, 0, 2, 0, 1);
          break;
        case '@':
          v22 = sub_250D2C0(j - 24, 0);
          sub_251BBC0(a1, v22, v23, 0, 2, 0, 1);
          break;
        case 'U':
          v74 = *(_QWORD *)(j - 56);
          if ( v74 )
          {
            if ( !*(_BYTE *)v74
              && *(_QWORD *)(v74 + 24) == *(_QWORD *)(j + 56)
              && (*(_BYTE *)(v74 + 33) & 0x20) != 0
              && *(_DWORD *)(v74 + 36) == 11 )
            {
              v75.m128i_i64[0] = sub_250D2C0(*(_QWORD *)(j - 32LL * (*(_DWORD *)(j - 20) & 0x7FFFFFF) - 24), 0);
              v148 = v75;
              if ( !(unsigned __int8)sub_250E300(a1, &v148) )
                v148.m128i_i64[1] = 0;
              v76 = _mm_loadu_si128(&v148);
              v77 = (__int64)v149;
              *(_QWORD *)v149 = &unk_438A664;
              *(__m128i *)&v149[8] = v76;
              v78 = sub_25134D0(a1 + 136, (__int64 *)v149);
              if ( !v78 || !v78[3] )
              {
                v79 = *(_QWORD *)(a1 + 4376);
                if ( v79 )
                {
                  v80 = *(_DWORD *)(v79 + 24);
                  v77 = *(_QWORD *)(v79 + 8);
                  if ( !v80 )
                    break;
                  v81 = v80 - 1;
                  v82 = v81 & v136;
                  v83 = *(void **)(v77 + 8LL * (v81 & v136));
                  v84 = 1;
                  if ( v83 != &unk_438A664 )
                  {
                    while ( v83 != (void *)-4096LL )
                    {
                      v82 = v81 & (v84 + v82);
                      v83 = *(void **)(v77 + 8LL * v82);
                      if ( v83 == &unk_438A664 )
                        goto LABEL_123;
                      ++v84;
                    }
                    break;
                  }
                }
LABEL_123:
                v85 = sub_25096F0(&v148);
                if ( !v85
                  || (v131 = v85, !(unsigned __int8)sub_B2D610(v85, 20))
                  && (v77 = 48, !(unsigned __int8)sub_B2D610(v131, 48)) )
                {
                  if ( *(_DWORD *)(a1 + 3556) <= dword_4FEEF68[0] )
                  {
                    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
                    {
                      v132 = sub_250CBE0(v148.m128i_i64, v77);
                      v86 = sub_2509800(&v148);
                      if ( v86 > 7u || ((1LL << v86) & 0xA8) == 0 )
                        goto LABEL_133;
                      v87 = v148.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
                      if ( (v148.m128i_i8[0] & 3) == 3 )
                        v87 = *(_QWORD *)(v87 + 24);
                      if ( **(_BYTE **)(v87 - 32) != 25 )
                      {
LABEL_133:
                        if ( (v86 & 0xFD) == 4 )
                        {
                          if ( (v132[32] & 0xFu) - 7 <= 1 && (unsigned __int8)sub_250CC70(a1, v148.m128i_i64) )
                          {
LABEL_136:
                            if ( *(_BYTE *)(a1 + 4296)
                              || (unsigned __int8)sub_266EE70(*(_QWORD *)(a1 + 200), (__int64)v132) )
                            {
                              goto LABEL_138;
                            }
                            v107 = sub_25096F0(&v148);
                            v133 = sub_266EE70(*(_QWORD *)(a1 + 200), v107);
                            goto LABEL_139;
                          }
                        }
                        else if ( (unsigned __int8)sub_250CC70(a1, v148.m128i_i64) )
                        {
                          if ( v132 )
                            goto LABEL_136;
LABEL_138:
                          v133 = 1;
LABEL_139:
                          v144 = sub_2563EA0(&v148, a1);
                          *(_QWORD *)v149 = &unk_438A664;
                          *(__m128i *)&v149[8] = _mm_loadu_si128((const __m128i *)(v144 + 72));
                          v88 = sub_2519B70(a1 + 136, (__int64)v149);
                          v92 = v144;
                          *v88 = v144;
                          if ( *(_DWORD *)(a1 + 3552) <= 1u )
                          {
                            *(_QWORD *)v149 = v144 & 0xFFFFFFFFFFFFFFFBLL;
                            sub_269CF50(a1 + 224, (unsigned __int64 *)v149, v89, v90, v91, v144);
                            v92 = v144;
                            if ( !*(_DWORD *)(a1 + 3552) )
                            {
                              v103 = sub_250E880(a1, v144);
                              v92 = v144;
                              if ( !v103 )
                                goto LABEL_164;
                            }
                          }
                          *(_QWORD *)v149 = v92;
                          v145 = v92;
                          v93 = sub_C99770(
                                  "initialize",
                                  10,
                                  (void (__fastcall *)(__m128i **, __int64))sub_250A3B0,
                                  (__int64)v149);
                          v92 = v145;
                          v94 = v93;
                          v95 = *(_DWORD *)(a1 + 3556);
                          *(_DWORD *)(a1 + 3556) = v95 + 1;
                          v96 = *(void (**)())(*(_QWORD *)v145 + 24LL);
                          if ( v96 != nullsub_1516 )
                          {
                            v130 = v94;
                            ((void (__fastcall *)(__int64, __int64))v96)(v145, a1);
                            v94 = v130;
                            v92 = v145;
                            v95 = *(_DWORD *)(a1 + 3556) - 1;
                          }
                          *(_DWORD *)(a1 + 3556) = v95;
                          if ( v94 )
                          {
                            v146 = v92;
                            sub_C9AF60(v94);
                            v92 = v146;
                          }
                          if ( v133 )
                          {
                            v97 = *(_DWORD *)(a1 + 3552);
                            *(_DWORD *)(a1 + 3552) = 1;
                            v147 = v97;
                            sub_251C580(a1, v92);
                            *(_DWORD *)(a1 + 3552) = v147;
                          }
                          else
                          {
LABEL_164:
                            v104 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v92 + 40LL);
                            if ( v104 == sub_2505DE0 )
                              v105 = (_BYTE *)(v92 + 88);
                            else
                              v105 = (_BYTE *)v104(v92);
                            v106 = *(void (**)(void))(*(_QWORD *)v105 + 40LL);
                            if ( (char *)v106 == (char *)sub_2505EB0 )
                              v105[17] = v105[16];
                            else
                              v106();
                          }
                          break;
                        }
                      }
                    }
                    v133 = 0;
                    goto LABEL_139;
                  }
                }
              }
            }
          }
          break;
      }
LABEL_21:
      for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v5 + 32) )
      {
        v16 = v5 - 24;
        if ( !v5 )
          v16 = 0;
        result = v16 + 48;
        if ( j != result )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v4 == v5 )
          return result;
        if ( !v5 )
          BUG();
      }
      if ( v5 == v4 )
        return result;
    }
    if ( sub_B491E0(j - 24) )
    {
      sub_250D230((unsigned __int64 *)v149, j - 24, 5, 0);
      v148 = _mm_loadu_si128((const __m128i *)v149);
      if ( !(unsigned __int8)sub_250E300(a1, &v148) )
        v148.m128i_i64[1] = 0;
      v28 = _mm_loadu_si128(&v148);
      v29 = (__int64)v149;
      *(_QWORD *)v149 = &unk_438A65A;
      *(__m128i *)&v149[8] = v28;
      v30 = sub_25134D0(a1 + 136, (__int64 *)v149);
      if ( (!v30 || !v30[3]) && (unsigned __int8)sub_2509800(&v148) == 5 )
      {
        v31 = (_BYTE *)sub_2509740(&v148);
        if ( *v31 == 85 )
        {
          v128 = (__int64)v31;
          if ( sub_B491E0((__int64)v31) )
          {
            v129 = sub_B49200(v128);
            if ( !v129 )
            {
              v32 = *(_QWORD *)(a1 + 4376);
              if ( v32 )
              {
                v33 = *(_DWORD *)(v32 + 24);
                v34 = *(_QWORD *)(v32 + 8);
                if ( !v33 )
                  goto LABEL_33;
                v35 = v33 - 1;
                v29 = v35 & (((unsigned int)&unk_438A65A >> 9) ^ ((unsigned int)&unk_438A65A >> 4));
                v36 = *(void **)(v34 + 8 * v29);
                v37 = 1;
                if ( v36 != &unk_438A65A )
                {
                  while ( v36 != (void *)-4096LL )
                  {
                    v29 = v35 & (unsigned int)(v37 + v29);
                    v36 = *(void **)(v34 + 8LL * (unsigned int)v29);
                    if ( v36 == &unk_438A65A )
                      goto LABEL_59;
                    ++v37;
                  }
                  goto LABEL_33;
                }
              }
LABEL_59:
              v38 = sub_25096F0(&v148);
              if ( !v38
                || (v126 = v38, !(unsigned __int8)sub_B2D610(v38, 20))
                && (v29 = 48, !(unsigned __int8)sub_B2D610(v126, 48)) )
              {
                if ( *(_DWORD *)(a1 + 3556) <= dword_4FEEF68[0] )
                {
                  if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
                    goto LABEL_74;
                  v127 = sub_250CBE0(v148.m128i_i64, v29);
                  v39 = sub_2509800(&v148);
                  if ( v39 <= 7u && ((1LL << v39) & 0xA8) != 0 )
                  {
                    v40 = v148.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
                    if ( (v148.m128i_i8[0] & 3) == 3 )
                      v40 = *(_QWORD *)(v40 + 24);
                    if ( **(_BYTE **)(v40 - 32) == 25 )
                      goto LABEL_74;
                  }
                  v135 = sub_250CC70(a1, v148.m128i_i64);
                  if ( !v135 )
                    goto LABEL_74;
                  if ( v127
                    && !*(_BYTE *)(a1 + 4296)
                    && !(unsigned __int8)sub_266EE70(*(_QWORD *)(a1 + 200), (__int64)v127) )
                  {
                    v117 = v148.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
                    if ( (v148.m128i_i8[0] & 3) == 3 )
                      v117 = *(_QWORD *)(v117 + 24);
                    v118 = *(_BYTE *)v117;
                    if ( *(_BYTE *)v117 )
                    {
                      if ( v118 == 22 )
                      {
                        v117 = *(_QWORD *)(v117 + 24);
                      }
                      else if ( v118 <= 0x1Cu )
                      {
                        v117 = 0;
                      }
                      else
                      {
                        v119 = sub_B43CB0(v117);
                        v41 = *(_QWORD *)(a1 + 200);
                        v117 = v119;
                      }
                    }
                    if ( *(_DWORD *)(v41 + 40) )
                    {
                      v120 = *(_DWORD *)(v41 + 24);
                      if ( !v120 )
                      {
LABEL_74:
                        v138 = sub_2565F60(&v148, a1);
                        *(_QWORD *)v149 = &unk_438A65A;
                        *(__m128i *)&v149[8] = _mm_loadu_si128((const __m128i *)(v138 + 72));
                        v42 = sub_2519B70(a1 + 136, (__int64)v149);
                        v46 = v138;
                        *v42 = v138;
                        if ( *(_DWORD *)(a1 + 3552) <= 1u )
                        {
                          *(_QWORD *)v149 = v138 & 0xFFFFFFFFFFFFFFFBLL;
                          sub_269CF50(a1 + 224, (unsigned __int64 *)v149, v43, v44, v45, v138);
                          v46 = v138;
                          if ( !*(_DWORD *)(a1 + 3552) )
                          {
                            v98 = sub_250E880(a1, v138);
                            v46 = v138;
                            if ( !v98 )
                              goto LABEL_148;
                          }
                        }
                        *(_QWORD *)v149 = v46;
                        v139 = v46;
                        v47 = sub_C99770(
                                "initialize",
                                10,
                                (void (__fastcall *)(__m128i **, __int64))sub_250BD00,
                                (__int64)v149);
                        v46 = v139;
                        v48 = v47;
                        v49 = *(_DWORD *)(a1 + 3556);
                        *(_DWORD *)(a1 + 3556) = v49 + 1;
                        v50 = *(void (**)())(*(_QWORD *)v139 + 24LL);
                        if ( v50 != nullsub_1516 )
                        {
                          v134 = v48;
                          ((void (__fastcall *)(__int64, __int64))v50)(v139, a1);
                          v48 = v134;
                          v46 = v139;
                          v49 = *(_DWORD *)(a1 + 3556) - 1;
                        }
                        *(_DWORD *)(a1 + 3556) = v49;
                        if ( v48 )
                        {
                          v140 = v46;
                          sub_C9AF60(v48);
                          v46 = v140;
                        }
                        if ( v129 )
                        {
                          v51 = *(_DWORD *)(a1 + 3552);
                          *(_DWORD *)(a1 + 3552) = 1;
                          v141 = v51;
                          sub_251C580(a1, v46);
                          *(_DWORD *)(a1 + 3552) = v141;
                        }
                        else
                        {
LABEL_148:
                          v99 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v46 + 40LL);
                          if ( v99 == sub_2505F20 )
                            v100 = v46 + 88;
                          else
                            v100 = v99(v46);
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v100 + 40LL))(v100);
                        }
                        goto LABEL_33;
                      }
                      v121 = v120 - 1;
                      v122 = 1;
                      for ( k = v121 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4)); ; k = v121 & v125 )
                      {
                        v124 = *(_QWORD *)(*(_QWORD *)(v41 + 8) + 8LL * k);
                        if ( v117 == v124 )
                          break;
                        if ( v124 == -4096 )
                          goto LABEL_74;
                        v125 = v122 + k;
                        ++v122;
                      }
                    }
                  }
                  v129 = v135;
                  goto LABEL_74;
                }
              }
            }
          }
        }
      }
    }
LABEL_33:
    LOBYTE(v8) = *(_BYTE *)(j - 24);
    goto LABEL_34;
  }
  return result;
}
