// Function: sub_1DA3ED0
// Address: 0x1da3ed0
//
__int64 __fastcall sub_1DA3ED0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int16 v10; // ax
  __int64 v11; // r14
  const __m128i *v12; // r12
  const __m128i *v13; // r13
  char v14; // cl
  unsigned int v15; // esi
  __int64 v16; // rdi
  int v17; // esi
  unsigned int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // r8
  _QWORD *v21; // r15
  _QWORD *v22; // rcx
  _QWORD *v23; // rbx
  int v24; // ecx
  _QWORD *v25; // r14
  _QWORD *v26; // rax
  unsigned __int32 v27; // eax
  bool v28; // zf
  __m128i *v29; // rax
  _QWORD *v30; // rdx
  unsigned int v31; // r14d
  const __m128i *v32; // rdi
  unsigned int v33; // eax
  unsigned int v34; // eax
  unsigned int v35; // ecx
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v40; // rax
  __int64 i; // rax
  __int64 v43; // rdi
  __int64 (*v44)(); // rax
  __int64 v45; // rax
  __int64 *v46; // rdi
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 (*v49)(); // rcx
  const __m128i *v50; // rsi
  const __m128i *v51; // r13
  __int32 v52; // r11d
  __int64 v53; // rdx
  const __m128i *v54; // rax
  const __m128i *v55; // r8
  __int32 v56; // r9d
  char v57; // cl
  __int64 v58; // rax
  __int64 *v59; // rdx
  __int64 v60; // rax
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rax
  const __m128i *v64; // rsi
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  __m128i *v67; // rdx
  __m128i *v68; // r14
  __m128i *v69; // rax
  unsigned __int64 v70; // rax
  bool v71; // cl
  unsigned __int64 v72; // rax
  __int64 v73; // rsi
  __int64 v74; // r8
  __int64 v75; // rdi
  __int64 v76; // rsi
  __int64 v77; // rdx
  int v78; // esi
  __int32 v79; // eax
  __int64 v82; // r10
  __int64 v83; // rax
  int v84; // edx
  unsigned __int64 v85; // rax
  unsigned int v86; // eax
  int v87; // edi
  unsigned int v88; // r8d
  _QWORD *v89; // r9
  unsigned __int16 v90; // r11
  __int64 v91; // r10
  __int64 v92; // r14
  unsigned __int16 v93; // dx
  _WORD *v94; // rcx
  unsigned __int16 *v95; // rax
  unsigned __int16 v96; // di
  unsigned __int16 *v97; // r8
  unsigned __int16 *v98; // rsi
  unsigned __int16 *v99; // rax
  unsigned __int16 v100; // cx
  __int16 *v101; // rax
  unsigned __int16 *v102; // rax
  __int16 v103; // r8
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  bool v106; // al
  unsigned int v107; // esi
  __m128i *v108; // r8
  __int64 v109; // rsi
  int v110; // r10d
  _QWORD *v111; // r9
  __int64 v112; // rdx
  int v113; // esi
  __int32 v114; // eax
  int v117; // r10d
  __int64 v118; // r11
  __int64 v119; // rax
  int v120; // edx
  unsigned __int64 v121; // rax
  unsigned int v122; // eax
  __int64 v123; // r12
  char v124; // al
  __int64 v125; // rdi
  __int64 v126; // rax
  _QWORD *v127; // rax
  __int64 v128; // rdx
  _QWORD *k; // rdx
  __int64 v130; // rdi
  int v131; // ecx
  __int64 v132; // r8
  __int64 v133; // rax
  __int64 v134; // rdi
  int v135; // ecx
  __int64 v136; // r8
  __int64 v137; // rax
  int v138; // esi
  _QWORD *v139; // r9
  int v140; // ecx
  int v141; // ecx
  unsigned __int64 v142; // rdx
  _QWORD *v143; // rbx
  __int64 v144; // rax
  _QWORD *v145; // rax
  int v146; // r9d
  unsigned int v147; // r10d
  _QWORD *v148; // r11
  unsigned __int16 v149; // cx
  int v150; // esi
  unsigned int v151; // [rsp+Ch] [rbp-E4h]
  __int64 *v152; // [rsp+10h] [rbp-E0h]
  int v153; // [rsp+10h] [rbp-E0h]
  __m128i *v154; // [rsp+10h] [rbp-E0h]
  __m128i *v155; // [rsp+10h] [rbp-E0h]
  __int64 v159; // [rsp+28h] [rbp-C8h]
  __int64 v160; // [rsp+30h] [rbp-C0h] BYREF
  __m128i *v161; // [rsp+38h] [rbp-B8h] BYREF
  __m128i v162; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v163; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v164; // [rsp+60h] [rbp-90h]
  __int64 v165; // [rsp+68h] [rbp-88h] BYREF
  _BYTE *v166; // [rsp+70h] [rbp-80h]
  _BYTE *v167; // [rsp+78h] [rbp-78h]
  __int64 v168; // [rsp+80h] [rbp-70h]
  int v169; // [rsp+88h] [rbp-68h]
  _BYTE v170[32]; // [rsp+90h] [rbp-60h] BYREF
  int v171; // [rsp+B0h] [rbp-40h]
  unsigned __int64 v172; // [rsp+B8h] [rbp-38h]

  if ( **(_WORD **)(a2 + 16) != 12 )
    goto LABEL_2;
  v152 = (__int64 *)sub_1E16500(a2);
  v58 = sub_15C70A0(a2 + 64);
  v59 = 0;
  if ( *(_DWORD *)(v58 + 8) == 2 )
    v59 = *(__int64 **)(v58 - 8);
  sub_1DA10C0((__int64 **)a3, v152, v59);
  v60 = *(_QWORD *)(a2 + 32);
  if ( *(_BYTE *)v60 || !*(_DWORD *)(v60 + 8) )
    goto LABEL_2;
  v61 = 0;
  v62 = sub_15C70A0(a2 + 64);
  if ( *(_DWORD *)(v62 + 8) == 2 )
    v61 = *(_QWORD *)(v62 - 8);
  v63 = sub_1E16500(a2);
  v64 = *(const __m128i **)(a2 + 64);
  v162.m128i_i64[1] = v61;
  v162.m128i_i64[0] = v63;
  v163.m128i_i64[0] = a2;
  v161 = (__m128i *)v64;
  if ( v64 )
  {
    sub_1623A60((__int64)&v161, (__int64)v64, 2);
    v163.m128i_i64[1] = (__int64)v161;
    if ( v161 )
      sub_1623210((__int64)&v161, (unsigned __int8 *)v161, (__int64)&v163.m128i_i64[1]);
  }
  else
  {
    v163.m128i_i64[1] = 0;
  }
  v168 = 4;
  v165 = 0;
  v164 = (__int64)(a1 + 35);
  v166 = v170;
  v167 = v170;
  v65 = *(_QWORD *)(a2 + 32);
  v169 = 0;
  v171 = 0;
  if ( !*(_BYTE *)v65 )
  {
    v66 = *(int *)(v65 + 8);
    if ( (_DWORD)v66 )
    {
      v171 = 1;
      v172 = v66;
    }
  }
  v67 = *(__m128i **)(a5 + 16);
  if ( v67 )
  {
    v68 = (__m128i *)(a5 + 8);
    while ( 1 )
    {
      v70 = v67[2].m128i_u64[0];
      v71 = v70 < v162.m128i_i64[0];
      if ( v70 == v162.m128i_i64[0] )
      {
        v72 = v67[2].m128i_u64[1];
        v71 = v72 < v162.m128i_i64[1];
        if ( v72 == v162.m128i_i64[1] )
          v71 = v67[9].m128i_i64[1] < v172;
      }
      v69 = (__m128i *)v67[1].m128i_i64[1];
      if ( !v71 )
      {
        v69 = (__m128i *)v67[1].m128i_i64[0];
        v68 = v67;
      }
      if ( !v69 )
        break;
      v67 = v69;
    }
    if ( (__m128i *)(a5 + 8) != v68 )
    {
      v105 = v68[2].m128i_u64[0];
      v106 = v162.m128i_i64[0] < v105;
      if ( v162.m128i_i64[0] == v105 )
      {
        v142 = v68[2].m128i_u64[1];
        v106 = v162.m128i_i64[1] < v142;
        if ( v162.m128i_i64[1] == v142 )
          v106 = v172 < v68[9].m128i_i64[1];
      }
      if ( !v106 )
        goto LABEL_134;
    }
  }
  else
  {
    v68 = (__m128i *)(a5 + 8);
  }
  v161 = &v162;
  v68 = sub_1DA0CF0((_QWORD *)a5, v68, (const __m128i **)&v161);
LABEL_134:
  v107 = v68[10].m128i_u32[0];
  if ( !v107 )
  {
    v68[10].m128i_i32[0] = ((__int64)(*(_QWORD *)(a5 + 56) - *(_QWORD *)(a5 + 48)) >> 7) + 1;
    v108 = *(__m128i **)(a5 + 56);
    if ( v108 == *(__m128i **)(a5 + 64) )
    {
      sub_1DA0880((const __m128i **)(a5 + 48), *(const __m128i **)(a5 + 56), &v162);
    }
    else
    {
      if ( v108 )
      {
        *v108 = _mm_loadu_si128(&v162);
        v108[1].m128i_i64[0] = v163.m128i_i64[0];
        v109 = v163.m128i_i64[1];
        v108[1].m128i_i64[1] = v163.m128i_i64[1];
        if ( v109 )
        {
          v154 = v108;
          sub_1623A60((__int64)&v108[1].m128i_i64[1], v109, 2);
          v108 = v154;
        }
        v155 = v108;
        v108[2].m128i_i64[0] = v164;
        sub_16CCCB0(&v108[2].m128i_i64[1], (__int64)v108[5].m128i_i64, (__int64)&v165);
        v155[7].m128i_i32[0] = v171;
        v155[7].m128i_i64[1] = v172;
        v108 = *(__m128i **)(a5 + 56);
      }
      *(_QWORD *)(a5 + 56) = v108 + 8;
    }
    v107 = v68[10].m128i_u32[0];
  }
  sub_1DA2AD0((__m128i *)a3, v107, v162.m128i_i64[0], v162.m128i_i64[1]);
  if ( v167 != v166 )
    _libc_free((unsigned __int64)v167);
  if ( v163.m128i_i64[1] )
    sub_161E7C0((__int64)&v163.m128i_i64[1], v163.m128i_i64[1]);
LABEL_2:
  sub_1DA18F0((__int64)a1, a2, a3, a5);
  if ( a7 )
  {
    v43 = a1[30];
    v44 = *(__int64 (**)())(*(_QWORD *)v43 + 400LL);
    if ( v44 != sub_1DA0690
      && ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __m128i **))v44)(v43, a2, &v160, &v161)
      && (((*(_BYTE *)(v160 + 3) & 0x40) != 0) & ((*(_BYTE *)(v160 + 3) >> 4) ^ 1)) != 0
      && (v161->m128i_i8[3] & 0x10) != 0 )
    {
      v89 = (_QWORD *)a1[29];
      if ( !v89 )
        goto LABEL_236;
      v90 = 0;
      v91 = v89[1];
      v92 = v89[7];
      v153 = *(_DWORD *)(v160 + 8);
      v151 = v161->m128i_u32[2];
      v93 = 0;
      v94 = (_WORD *)(v92 + 2LL * (*(_DWORD *)(v91 + 24LL * v151 + 16) >> 4));
      v95 = v94 + 1;
      v96 = *v94 + v161->m128i_i16[4] * (*(_WORD *)(v91 + 24LL * v151 + 16) & 0xF);
LABEL_114:
      v97 = v95;
      while ( 1 )
      {
        v98 = v97;
        if ( !v97 )
          break;
        v99 = (unsigned __int16 *)(v89[6] + 4LL * v96);
        v100 = *v99;
        v93 = v99[1];
        if ( *v99 )
        {
          while ( 1 )
          {
            v101 = (__int16 *)(v92 + 2LL * *(unsigned int *)(v91 + 24LL * v100 + 8));
            if ( v101 )
              goto LABEL_122;
            if ( !v93 )
              break;
            v100 = v93;
            v93 = 0;
          }
          v90 = v100;
        }
        v149 = *v97;
        v95 = 0;
        ++v97;
        if ( !v149 )
          goto LABEL_114;
        v96 += v149;
      }
      v100 = v90;
      v101 = 0;
LABEL_122:
      while ( v98 )
      {
        if ( (*(_QWORD *)(a1[32] + 8 * ((unsigned __int64)v100 >> 6)) & (1LL << v100)) != 0 )
        {
          v112 = *(_QWORD *)(a3 + 8);
          v162.m128i_i64[1] = a3;
          v163.m128i_i64[1] = 0xFFFFFFFF00000000LL;
          v163.m128i_i64[0] = v112;
          v164 = 0;
          v162.m128i_i8[0] = 0;
          if ( v112 != a3 + 8 )
          {
            v113 = 0;
            v114 = *(_DWORD *)(v112 + 16) << 7;
            v163.m128i_i32[2] = v114;
            _RCX = *(_QWORD *)(v112 + 24);
            if ( !_RCX )
            {
              _RCX = *(_QWORD *)(v112 + 32);
              v113 = 64;
              if ( !_RCX )
              {
                _RCX = *(_QWORD *)(v112 + 40);
                v113 = 128;
              }
            }
            __asm { tzcnt   rcx, rcx }
            v117 = v153;
            v118 = a5;
            LODWORD(_RCX) = v113 + _RCX;
            v163.m128i_i32[2] = _RCX + v114;
            v119 = ((unsigned int)(_RCX + v114) >> 6) & 1;
            v163.m128i_i32[3] = v119;
            v164 = *(_QWORD *)(v112 + 8 * v119 + 24) >> _RCX;
            while ( 1 )
            {
              v120 = 0;
              v121 = *(_QWORD *)(v118 + 48) + ((unsigned __int64)(unsigned int)(v163.m128i_i32[2] - 1) << 7);
              if ( *(_DWORD *)(v121 + 112) == 1 )
                v120 = *(_DWORD *)(v121 + 120);
              if ( v117 == v120 )
                break;
              sub_1DA06D0((__int64)&v162);
              if ( v162.m128i_i8[0] )
                goto LABEL_44;
            }
            sub_1DA2EF0((__int64)a1, a2, (__int64 **)a3, a6, a5, v163.m128i_i32[2], v151);
          }
          break;
        }
        v103 = *v101++;
        if ( v103 )
        {
          v100 += v103;
        }
        else if ( v93 )
        {
          v104 = v93;
          v100 = v93;
          v93 = 0;
          v101 = (__int16 *)(v92 + 2LL * *(unsigned int *)(v91 + 24 * v104 + 8));
        }
        else
        {
          v93 = *v98;
          v96 += *v98;
          if ( *v98 )
          {
            ++v98;
            v102 = (unsigned __int16 *)(v89[6] + 4LL * v96);
            v100 = *v102;
            v93 = v102[1];
            v101 = (__int16 *)(v92 + 2LL * *(unsigned int *)(v91 + 24LL * *v102 + 8));
          }
          else
          {
            v101 = 0;
            v98 = 0;
          }
        }
      }
    }
LABEL_44:
    v45 = sub_1E15F70(a2);
    if ( *(_BYTE *)(a2 + 49) == 1 )
    {
      v46 = (__int64 *)a1[30];
      v47 = *(_QWORD *)(v45 + 56);
      v48 = *v46;
      v49 = *(__int64 (**)())(*v46 + 96);
      if ( v49 != sub_1DA0680 )
      {
        if ( ((unsigned int (__fastcall *)(__int64 *, __int64, __int64 *))v49)(v46, a2, &v160) )
        {
LABEL_47:
          if ( *(_BYTE *)(*(_QWORD *)(v47 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v47 + 32) + v160) + 21) )
          {
            v50 = *(const __m128i **)(a2 + 32);
            v51 = (const __m128i *)((char *)v50 + 40 * *(unsigned int *)(a2 + 40));
            if ( v50 != v51 )
            {
              while ( 1 )
              {
                v162 = _mm_loadu_si128(v50);
                v163 = _mm_loadu_si128(v50 + 1);
                v164 = v50[2].m128i_i64[0];
                if ( !(v50->m128i_i8[0] | v162.m128i_i8[3] & 0x10) )
                {
                  v52 = v50->m128i_i32[2];
                  if ( (v50->m128i_i8[3] & 0x40) != 0 )
                    goto LABEL_89;
                  if ( v52 )
                  {
                    v53 = *(_QWORD *)(a2 + 8);
                    if ( v53 != *(_QWORD *)(a2 + 24) + 24LL )
                    {
                      v54 = *(const __m128i **)(v53 + 32);
                      v55 = (const __m128i *)((char *)v54 + 40 * *(unsigned int *)(v53 + 40));
                      if ( v54 != v55 )
                        break;
                    }
                  }
                }
LABEL_59:
                v50 = (const __m128i *)((char *)v50 + 40);
                if ( v51 == v50 )
                  goto LABEL_3;
              }
              while ( 1 )
              {
                v162 = _mm_loadu_si128(v54);
                v163 = _mm_loadu_si128(v54 + 1);
                v164 = v54[2].m128i_i64[0];
                if ( !v54->m128i_i8[0] )
                {
                  v56 = v54->m128i_i32[2];
                  v57 = (v54->m128i_i8[3] & 0x40) != 0;
                  v162.m128i_i8[3] = (v57 << 6) | v162.m128i_i8[3] & 0xBF;
                  if ( (v162.m128i_i8[3] & 0x10) == 0 && v52 == v56 && v57 )
                    break;
                }
                v54 = (const __m128i *)((char *)v54 + 40);
                if ( v55 == v54 )
                  goto LABEL_59;
              }
LABEL_89:
              v77 = *(_QWORD *)(a3 + 8);
              v162.m128i_i64[1] = a3;
              v163.m128i_i64[1] = 0xFFFFFFFF00000000LL;
              v163.m128i_i64[0] = v77;
              v164 = 0;
              v162.m128i_i8[0] = 0;
              if ( v77 != a3 + 8 )
              {
                v78 = 0;
                v79 = *(_DWORD *)(v77 + 16) << 7;
                v163.m128i_i32[2] = v79;
                _RCX = *(_QWORD *)(v77 + 24);
                if ( !_RCX )
                {
                  _RCX = *(_QWORD *)(v77 + 32);
                  v78 = 64;
                  if ( !_RCX )
                  {
                    _RCX = *(_QWORD *)(v77 + 40);
                    v78 = 128;
                  }
                }
                __asm { tzcnt   rcx, rcx }
                v82 = a5;
                LODWORD(_RCX) = v78 + _RCX;
                v163.m128i_i32[2] = _RCX + v79;
                v83 = ((unsigned int)(_RCX + v79) >> 6) & 1;
                v163.m128i_i32[3] = v83;
                v164 = *(_QWORD *)(v77 + 8 * v83 + 24) >> _RCX;
                while ( 1 )
                {
                  v84 = 0;
                  v85 = *(_QWORD *)(v82 + 48) + ((unsigned __int64)(unsigned int)(v163.m128i_i32[2] - 1) << 7);
                  if ( *(_DWORD *)(v85 + 112) == 1 )
                    v84 = *(_DWORD *)(v85 + 120);
                  if ( v84 == v52 )
                    break;
                  sub_1DA06D0((__int64)&v162);
                  if ( v162.m128i_i8[0] )
                    goto LABEL_3;
                }
                sub_1DA2EF0((__int64)a1, a2, (__int64 **)a3, a6, a5, v163.m128i_i32[2], 0);
              }
            }
          }
          goto LABEL_3;
        }
        v46 = (__int64 *)a1[30];
        v48 = *v46;
      }
      if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __m128i **, __int64 *))(v48 + 104))(
              v46,
              a2,
              &v161,
              &v160) )
        goto LABEL_3;
      goto LABEL_47;
    }
  }
LABEL_3:
  v10 = *(_WORD *)(a2 + 46);
  v11 = *(_QWORD *)(a2 + 24);
  if ( (v10 & 4) != 0 || (v10 & 8) == 0 )
  {
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) & 0x40LL) != 0 )
      goto LABEL_5;
  }
  else if ( (unsigned __int8)sub_1E15D00(a2, 64, 1) )
  {
    goto LABEL_5;
  }
  v40 = *(_QWORD *)(v11 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v40 )
    BUG();
  if ( (*(_QWORD *)v40 & 4) == 0 && (*(_BYTE *)(v40 + 46) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v40; ; i = *(_QWORD *)v40 )
    {
      v40 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v40 + 46) & 4) == 0 )
        break;
    }
  }
  if ( a2 != v40 )
    return 0;
LABEL_5:
  v12 = *(const __m128i **)(a3 + 8);
  v13 = (const __m128i *)(a3 + 8);
  if ( v12 == (const __m128i *)(a3 + 8) )
    return 0;
  v14 = *(_BYTE *)(a4 + 8) & 1;
  if ( v14 )
  {
    v16 = a4 + 16;
    v17 = 3;
  }
  else
  {
    v15 = *(_DWORD *)(a4 + 24);
    v16 = *(_QWORD *)(a4 + 16);
    if ( !v15 )
    {
      v19 = 0;
      ++*(_QWORD *)a4;
      v86 = *(_DWORD *)(a4 + 8);
      v87 = (v86 >> 1) + 1;
      goto LABEL_100;
    }
    v17 = v15 - 1;
  }
  v18 = v17 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v19 = (_QWORD *)(v16 + 40LL * v18);
  v20 = *v19;
  if ( v11 == *v19 )
    goto LABEL_10;
  v110 = 1;
  v111 = 0;
  while ( 1 )
  {
    if ( v20 == -8 )
    {
      v88 = 12;
      v15 = 4;
      if ( v111 )
        v19 = v111;
      ++*(_QWORD *)a4;
      v86 = *(_DWORD *)(a4 + 8);
      v87 = (v86 >> 1) + 1;
      if ( v14 )
      {
LABEL_101:
        if ( 4 * v87 < v88 )
        {
          if ( v15 - *(_DWORD *)(a4 + 12) - v87 > v15 >> 3 )
          {
LABEL_103:
            *(_DWORD *)(a4 + 8) = (2 * (v86 >> 1) + 2) | v86 & 1;
            if ( *v19 != -8 )
              --*(_DWORD *)(a4 + 12);
            v22 = v19 + 2;
            *v19 = v11;
            v19[3] = v19 + 2;
            v21 = v19 + 2;
            v19[2] = v19 + 2;
            v19[4] = 0;
            v19[1] = v19 + 2;
            if ( (_QWORD *)a3 == v19 + 1 )
            {
              v31 = 0;
              goto LABEL_20;
            }
            v12 = *(const __m128i **)(a3 + 8);
            v31 = 0;
            if ( v12 != v13 )
              goto LABEL_12;
            goto LABEL_22;
          }
          sub_1DA39D0(a4, v15);
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v134 = a4 + 16;
            v135 = 3;
            goto LABEL_186;
          }
          v141 = *(_DWORD *)(a4 + 24);
          v134 = *(_QWORD *)(a4 + 16);
          if ( v141 )
          {
            v135 = v141 - 1;
LABEL_186:
            v136 = v135 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v19 = (_QWORD *)(v134 + 40 * v136);
            v137 = *v19;
            if ( *v19 != v11 )
            {
              v138 = 1;
              v139 = 0;
              while ( v137 != -8 )
              {
                if ( v137 == -16 && !v139 )
                  v139 = v19;
                LODWORD(v136) = v135 & (v138 + v136);
                v19 = (_QWORD *)(v134 + 40LL * (unsigned int)v136);
                v137 = *v19;
                if ( v11 == *v19 )
                  goto LABEL_183;
                ++v138;
              }
LABEL_189:
              if ( v139 )
                v19 = v139;
              goto LABEL_183;
            }
            goto LABEL_183;
          }
          goto LABEL_235;
        }
        sub_1DA39D0(a4, 2 * v15);
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v130 = a4 + 16;
          v131 = 3;
        }
        else
        {
          v140 = *(_DWORD *)(a4 + 24);
          v130 = *(_QWORD *)(a4 + 16);
          if ( !v140 )
          {
LABEL_235:
            *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
LABEL_236:
            BUG();
          }
          v131 = v140 - 1;
        }
        v132 = v131 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v19 = (_QWORD *)(v130 + 40 * v132);
        v133 = *v19;
        if ( *v19 != v11 )
        {
          v150 = 1;
          v139 = 0;
          while ( v133 != -8 )
          {
            if ( !v139 && v133 == -16 )
              v139 = v19;
            LODWORD(v132) = v131 & (v150 + v132);
            v19 = (_QWORD *)(v130 + 40LL * (unsigned int)v132);
            v133 = *v19;
            if ( v11 == *v19 )
              goto LABEL_183;
            ++v150;
          }
          goto LABEL_189;
        }
LABEL_183:
        v86 = *(_DWORD *)(a4 + 8);
        goto LABEL_103;
      }
      v15 = *(_DWORD *)(a4 + 24);
LABEL_100:
      v88 = 3 * v15;
      goto LABEL_101;
    }
    if ( v111 || v20 != -16 )
      v19 = v111;
    v146 = v110 + 1;
    v147 = v18 + v110;
    v18 = v17 & v147;
    v148 = (_QWORD *)(v16 + 40LL * (v17 & v147));
    v20 = *v148;
    if ( v11 == *v148 )
      break;
    v110 = v146;
    v111 = v19;
    v19 = v148;
  }
  v19 = (_QWORD *)(v16 + 40LL * (v17 & v147));
LABEL_10:
  if ( (_QWORD *)a3 == v19 + 1 )
  {
    v31 = 0;
    goto LABEL_21;
  }
  v21 = (_QWORD *)v19[2];
  v22 = v19 + 2;
LABEL_12:
  v159 = a3;
  v23 = v22;
  v24 = 0;
  v25 = v19;
  do
  {
    while ( 1 )
    {
      if ( v21 != v23 )
      {
        v27 = v12[1].m128i_u32[0];
        v28 = *((_DWORD *)v21 + 4) == v27;
        if ( *((_DWORD *)v21 + 4) <= v27 )
          break;
      }
      v29 = (__m128i *)sub_22077B0(40);
      v29[1] = _mm_loadu_si128(v12 + 1);
      v29[2].m128i_i64[0] = v12[2].m128i_i64[0];
      sub_2208C80(v29, v21);
      ++v25[4];
      v12 = (const __m128i *)v12->m128i_i64[0];
      v24 = 1;
      if ( v13 == v12 )
        goto LABEL_19;
    }
    v26 = (_QWORD *)*v21;
    if ( v28 )
    {
      v73 = v21[3];
      v74 = v21[4];
      v75 = v73 | v12[1].m128i_i64[1];
      v21[3] = v75;
      if ( v75 == v73 )
      {
        v76 = v74 | v12[2].m128i_i64[0];
        v21[4] = v76;
        LOBYTE(v76) = v76 != v74;
        v24 |= v76;
      }
      else
      {
        v24 = 1;
        v21[4] = v12[2].m128i_i64[0] | v74;
      }
      v12 = (const __m128i *)v12->m128i_i64[0];
      v21 = v26;
    }
    else
    {
      v21 = (_QWORD *)*v21;
    }
  }
  while ( v13 != v12 );
LABEL_19:
  v30 = v25;
  a3 = v159;
  v31 = v24;
  v30[1] = v30[2];
LABEL_20:
  v12 = *(const __m128i **)(a3 + 8);
  while ( v13 != v12 )
  {
LABEL_21:
    v32 = v12;
    v12 = (const __m128i *)v12->m128i_i64[0];
    j_j___libc_free_0(v32, 40);
  }
LABEL_22:
  v33 = *(_DWORD *)(a3 + 40);
  ++*(_QWORD *)(a3 + 32);
  *(_QWORD *)(a3 + 16) = v13;
  v34 = v33 >> 1;
  *(_QWORD *)(a3 + 8) = v13;
  *(_QWORD *)(a3 + 24) = 0;
  if ( v34 )
  {
    if ( (*(_BYTE *)(a3 + 40) & 1) == 0 )
    {
      v35 = 4 * v34;
      goto LABEL_25;
    }
LABEL_87:
    v37 = (_QWORD *)(a3 + 48);
    v38 = 24;
LABEL_28:
    for ( j = &v37[v38]; j != v37; *(v37 - 2) = -8 )
    {
      *v37 = -8;
      v37 += 3;
    }
    *(_QWORD *)(a3 + 40) &= 1uLL;
    return v31;
  }
  if ( !*(_DWORD *)(a3 + 44) )
    return v31;
  v35 = 0;
  if ( (*(_BYTE *)(a3 + 40) & 1) != 0 )
    goto LABEL_87;
LABEL_25:
  v36 = *(unsigned int *)(a3 + 56);
  if ( v35 >= (unsigned int)v36 || (unsigned int)v36 <= 0x40 )
  {
    v37 = *(_QWORD **)(a3 + 48);
    v38 = 3 * v36;
    goto LABEL_28;
  }
  if ( !v34 || (v122 = v34 - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a3 + 48));
    *(_BYTE *)(a3 + 40) |= 1u;
    goto LABEL_173;
  }
  _BitScanReverse(&v122, v122);
  v123 = (unsigned int)(1 << (33 - (v122 ^ 0x1F)));
  if ( (unsigned int)(v123 - 9) <= 0x36 )
  {
    LODWORD(v123) = 64;
    j___libc_free_0(*(_QWORD *)(a3 + 48));
    v124 = *(_BYTE *)(a3 + 40);
    v125 = 1536;
LABEL_172:
    *(_BYTE *)(a3 + 40) = v124 & 0xFE;
    v126 = sub_22077B0(v125);
    *(_DWORD *)(a3 + 56) = v123;
    *(_QWORD *)(a3 + 48) = v126;
LABEL_173:
    v28 = (*(_QWORD *)(a3 + 40) & 1LL) == 0;
    *(_QWORD *)(a3 + 40) &= 1uLL;
    if ( v28 )
    {
      v127 = *(_QWORD **)(a3 + 48);
      v128 = 3LL * *(unsigned int *)(a3 + 56);
    }
    else
    {
      v127 = (_QWORD *)(a3 + 48);
      v128 = 24;
    }
    for ( k = &v127[v128]; k != v127; v127 += 3 )
    {
      if ( v127 )
      {
        *v127 = -8;
        v127[1] = -8;
      }
    }
  }
  else
  {
    if ( (_DWORD)v123 != (_DWORD)v36 )
    {
      j___libc_free_0(*(_QWORD *)(a3 + 48));
      v124 = *(_BYTE *)(a3 + 40) | 1;
      *(_BYTE *)(a3 + 40) = v124;
      if ( (unsigned int)v123 > 8 )
      {
        v125 = 24LL * (unsigned int)v123;
        goto LABEL_172;
      }
      goto LABEL_173;
    }
    v28 = (*(_QWORD *)(a3 + 40) & 1LL) == 0;
    *(_QWORD *)(a3 + 40) &= 1uLL;
    if ( v28 )
    {
      v143 = *(_QWORD **)(a3 + 48);
      v144 = 3 * v123;
    }
    else
    {
      v143 = (_QWORD *)(a3 + 48);
      v144 = 24;
    }
    v145 = &v143[v144];
    do
    {
      if ( v143 )
      {
        *v143 = -8;
        v143[1] = -8;
      }
      v143 += 3;
    }
    while ( v145 != v143 );
  }
  return v31;
}
