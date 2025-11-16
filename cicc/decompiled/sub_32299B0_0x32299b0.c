// Function: sub_32299B0
// Address: 0x32299b0
//
char __fastcall sub_32299B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rsi
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // r8d
  __int64 v13; // r9
  int v14; // r11d
  __int64 *v15; // rdx
  int v16; // r13d
  unsigned int j; // edi
  __int64 *v18; // rax
  __int64 v19; // rcx
  unsigned __int32 v20; // edx
  __int64 *v21; // rdi
  unsigned __int32 v22; // ecx
  unsigned int v23; // eax
  __int64 v24; // r8
  __m128i *v25; // r12
  __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rsi
  _QWORD *v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rdx
  int v32; // edx
  __int64 v33; // rcx
  int v34; // r8d
  int v35; // esi
  __int64 *v36; // rdi
  unsigned int i; // eax
  __int64 v38; // r9
  unsigned int v39; // eax
  __int64 v40; // r15
  __int64 *v41; // rcx
  int v42; // edx
  int v43; // r9d
  unsigned int m; // eax
  __int64 *v45; // rsi
  unsigned int v46; // eax
  __int64 v47; // rax
  char v48; // al
  __int64 v49; // rdx
  __int64 v50; // r9
  __int32 v51; // r10d
  __int64 v52; // rax
  char v53; // al
  __int64 v54; // r9
  unsigned int v55; // esi
  __int64 *v56; // rcx
  int v57; // esi
  unsigned int v58; // edx
  __int64 v59; // r10
  int v60; // edx
  __int64 v61; // r13
  char v62; // al
  __int64 v63; // r14
  __int32 v64; // r15d
  __int64 v65; // rdx
  __int32 v66; // esi
  __int32 v67; // eax
  void (*v68)(); // rdx
  __int64 v69; // r9
  __int64 v70; // rdx
  __int64 v71; // rax
  _QWORD *v72; // rdx
  _QWORD *v73; // rcx
  char v74; // r9
  __m128i *v75; // rsi
  char v76; // al
  bool v77; // al
  __int32 v78; // edx
  __int64 v79; // rsi
  __int64 v80; // rdi
  unsigned __int64 v81; // rax
  __int8 *v82; // r12
  unsigned int v83; // edi
  __int64 *v84; // rcx
  int v85; // edx
  unsigned int n; // eax
  __int64 v87; // rsi
  int v88; // eax
  __int64 *v89; // rcx
  int v90; // edx
  unsigned int ii; // eax
  __int64 v92; // rsi
  int v93; // eax
  int v94; // ecx
  int v95; // eax
  int v96; // eax
  __int64 v97; // rcx
  int v98; // edi
  __int64 *v99; // rsi
  unsigned int k; // r13d
  __int64 v101; // r8
  unsigned int v102; // r13d
  char v103; // al
  __int64 v106; // [rsp+8h] [rbp-F8h]
  __int64 v107; // [rsp+18h] [rbp-E8h]
  __int32 v108; // [rsp+28h] [rbp-D8h]
  _QWORD *v109; // [rsp+28h] [rbp-D8h]
  __int64 v110; // [rsp+30h] [rbp-D0h]
  __int32 v111; // [rsp+30h] [rbp-D0h]
  _QWORD *v112; // [rsp+30h] [rbp-D0h]
  __int64 v113; // [rsp+38h] [rbp-C8h]
  __int64 v114; // [rsp+38h] [rbp-C8h]
  char v115; // [rsp+38h] [rbp-C8h]
  _QWORD *v116; // [rsp+38h] [rbp-C8h]
  __m128i v119; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v120; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v121; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v122; // [rsp+78h] [rbp-88h]
  char v123; // [rsp+D0h] [rbp-30h] BYREF

  v3 = (__int64 *)&v121;
  v120.m128i_i64[0] = 0;
  v120.m128i_i64[1] = 1;
  do
  {
    *v3 = -4096;
    v3 += 3;
    *(v3 - 2) = -4096;
  }
  while ( v3 != (__int64 *)&v123 );
  v106 = a1 + 72;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL);
  v5 = *(_QWORD *)(v4 + 752);
  v6 = v5 + 32LL * *(unsigned int *)(v4 + 760);
  v107 = v6;
  if ( v5 == v6 )
    goto LABEL_29;
  while ( 2 )
  {
    while ( 2 )
    {
      v7 = *(_QWORD *)(v5 + 8);
      if ( !v7 )
        goto LABEL_28;
      v8 = *(_QWORD *)(v5 + 24);
      v9 = *(_BYTE *)(v8 - 16);
      if ( (v9 & 2) != 0 )
      {
        if ( *(_DWORD *)(v8 - 24) != 2 )
          goto LABEL_7;
        v31 = *(_QWORD *)(v8 - 32);
LABEL_34:
        v10 = a3;
        v11 = *(_QWORD *)(v31 + 8);
        v12 = *(_DWORD *)(a3 + 24);
        if ( !v12 )
        {
LABEL_35:
          ++*(_QWORD *)a3;
LABEL_36:
          sub_3201030(a3, 2 * v12);
          v32 = *(_DWORD *)(a3 + 24);
          if ( !v32 )
            goto LABEL_187;
          v34 = 1;
          v35 = v32 - 1;
          v36 = 0;
          for ( i = (v32 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
                      | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; i = v35 & v39 )
          {
            v33 = *(_QWORD *)(a3 + 8);
            v15 = (__int64 *)(v33 + 16LL * i);
            v38 = *v15;
            if ( v7 == *v15 && v11 == v15[1] )
              break;
            if ( v38 == -4096 )
            {
              if ( v15[1] == -4096 )
              {
                if ( v36 )
                  v15 = v36;
                v94 = *(_DWORD *)(a3 + 16) + 1;
                goto LABEL_144;
              }
            }
            else if ( v38 == -8192 && v15[1] == -8192 && !v36 )
            {
              v36 = (__int64 *)(v33 + 16LL * i);
            }
            v39 = v34 + i;
            ++v34;
          }
LABEL_150:
          v94 = *(_DWORD *)(a3 + 16) + 1;
          goto LABEL_144;
        }
        goto LABEL_8;
      }
      if ( ((*(_WORD *)(v8 - 16) >> 6) & 0xF) == 2 )
      {
        v31 = v8 - 16 - 8LL * ((v9 >> 2) & 0xF);
        goto LABEL_34;
      }
LABEL_7:
      v10 = a3;
      v11 = 0;
      v12 = *(_DWORD *)(a3 + 24);
      if ( !v12 )
        goto LABEL_35;
LABEL_8:
      v13 = *(_QWORD *)(v10 + 8);
      v14 = 1;
      v15 = 0;
      v16 = ((0xBF58476D1CE4E5B9LL
            * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
             | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
          ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)));
      for ( j = v16 & (v12 - 1); ; j = (v12 - 1) & v83 )
      {
        v18 = (__int64 *)(v13 + 16LL * j);
        v19 = *v18;
        if ( v7 == *v18 && v11 == v18[1] )
          goto LABEL_47;
        if ( v19 == -4096 )
          break;
        if ( v19 == -8192 && v18[1] == -8192 && !v15 )
          v15 = (__int64 *)(v13 + 16LL * j);
LABEL_108:
        v83 = v14 + j;
        ++v14;
      }
      if ( v18[1] != -4096 )
        goto LABEL_108;
      if ( !v15 )
        v15 = (__int64 *)(v13 + 16LL * j);
      ++*(_QWORD *)a3;
      v94 = *(_DWORD *)(a3 + 16) + 1;
      if ( 4 * v94 >= 3 * v12 )
        goto LABEL_36;
      if ( v12 - *(_DWORD *)(a3 + 20) - v94 <= v12 >> 3 )
      {
        sub_3201030(a3, v12);
        v95 = *(_DWORD *)(a3 + 24);
        if ( v95 )
        {
          v96 = v95 - 1;
          v98 = 1;
          v99 = 0;
          for ( k = v96 & v16; ; k = v96 & v102 )
          {
            v97 = *(_QWORD *)(a3 + 8);
            v15 = (__int64 *)(v97 + 16LL * k);
            v101 = *v15;
            if ( v7 == *v15 && v11 == v15[1] )
              break;
            if ( v101 == -4096 )
            {
              if ( v15[1] == -4096 )
              {
                if ( v99 )
                  v15 = v99;
                v94 = *(_DWORD *)(a3 + 16) + 1;
                goto LABEL_144;
              }
            }
            else if ( v101 == -8192 && v15[1] == -8192 && !v99 )
            {
              v99 = (__int64 *)(v97 + 16LL * k);
            }
            v102 = v98 + k;
            ++v98;
          }
          goto LABEL_150;
        }
LABEL_187:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
LABEL_144:
      *(_DWORD *)(a3 + 16) = v94;
      if ( *v15 != -4096 || v15[1] != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v15 = v7;
      v15[1] = v11;
      v8 = *(_QWORD *)(v5 + 24);
LABEL_47:
      v6 = sub_35051D0(v106, v8);
      v40 = v6;
      if ( !v6 )
        goto LABEL_28;
      sub_3223A50(a1, a2, v7, *(_QWORD *)(v6 + 8));
      if ( (v120.m128i_i8[8] & 1) != 0 )
      {
        v41 = (__int64 *)&v121;
        v42 = 3;
LABEL_50:
        v43 = 1;
        for ( m = v42
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
                    | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; m = v42 & v46 )
        {
          v45 = &v41[3 * m];
          if ( v7 == *v45 && v11 == v45[1] )
            break;
          if ( *v45 == -4096 && v45[1] == -4096 )
            goto LABEL_57;
          v46 = v43 + m;
          ++v43;
        }
        v61 = v45[2];
        if ( v61 )
        {
          LOBYTE(v6) = *(_BYTE *)(v61 + 88);
          if ( (_BYTE)v6 != 3 )
          {
            if ( (_BYTE)v6 != 4 || *(_BYTE *)(v5 + 4) != 1 )
              goto LABEL_28;
            v65 = *(_QWORD *)(v5 + 16);
            v66 = *(_DWORD *)v5;
            v5 += 32;
            LOBYTE(v6) = sub_321E020((_QWORD *)(v61 + 40), v66, v65);
            if ( v107 == v5 )
              goto LABEL_29;
            continue;
          }
          v62 = *(_BYTE *)(v5 + 4);
          v63 = *(_QWORD *)(v5 + 16);
          if ( !v62 )
          {
            v78 = *(_DWORD *)v5;
            v79 = *(_QWORD *)(v5 + 16);
            v5 += 32;
            LOBYTE(v6) = sub_3226480(v61 + 40, v79, v78);
            if ( v107 == v5 )
              goto LABEL_29;
            continue;
          }
          if ( v62 == 1 )
          {
            v64 = *(_DWORD *)v5;
            sub_321A810((__int64)&v119, v61 + 40);
            *(_QWORD *)(v61 + 64) = v61 + 48;
            *(_QWORD *)(v61 + 72) = v61 + 48;
            *(_BYTE *)(v61 + 88) = 4;
            *(_DWORD *)(v61 + 48) = 0;
            *(_QWORD *)(v61 + 56) = 0;
            *(_QWORD *)(v61 + 80) = 0;
            sub_321E020((_QWORD *)(v61 + 40), v64, v63);
            LOBYTE(v6) = *(_BYTE *)(v61 + 88);
            if ( (_BYTE)v6 != 4 )
              sub_435074((_BYTE)v6 == 0xFF);
            v5 += 32;
            if ( v107 == v5 )
              goto LABEL_29;
            continue;
          }
LABEL_184:
          abort();
        }
      }
      else
      {
        v41 = v121;
        if ( v122 )
        {
          v42 = v122 - 1;
          goto LABEL_50;
        }
      }
      break;
    }
LABEL_57:
    v47 = sub_22077B0(0x60u);
    v30 = v47;
    if ( v47 )
    {
      *(_QWORD *)(v47 + 8) = v7;
      *(_QWORD *)(v47 + 16) = v11;
      *(_QWORD *)(v47 + 24) = 0;
      *(_DWORD *)(v47 + 32) = 0;
      *(_QWORD *)v47 = &unk_4A35790;
      *(_WORD *)(v47 + 88) = 0;
      v48 = *(_BYTE *)(v5 + 4);
      if ( v48 )
      {
        v49 = *(_QWORD *)(v5 + 16);
        v50 = v30 + 40;
        if ( v48 != 1 )
          goto LABEL_184;
        v51 = *(_DWORD *)v5;
        v52 = *(char *)(v30 + 88);
        goto LABEL_61;
      }
      v67 = *(_DWORD *)v5;
      v68 = nullsub_1858;
      v69 = v30 + 40;
LABEL_88:
      v111 = v67;
      v114 = v69;
      v68();
      v67 = v111;
      v69 = v114;
      goto LABEL_89;
    }
    v103 = *(_BYTE *)(v5 + 4);
    v69 = 40;
    if ( v103 )
    {
      v49 = *(_QWORD *)(v5 + 16);
      v50 = 40;
      if ( v103 != 1 )
        goto LABEL_184;
      v52 = MEMORY[0x58];
      v51 = *(_DWORD *)v5;
      if ( MEMORY[0x58] != 0xFF )
      {
LABEL_61:
        v108 = v51;
        v110 = v49;
        v113 = v50;
        funcs_32198D3[v52]();
        v51 = v108;
        v49 = v110;
        v50 = v113;
      }
      *(_BYTE *)(v30 + 88) = 4;
      *(_QWORD *)(v30 + 64) = v30 + 48;
      *(_QWORD *)(v30 + 72) = v30 + 48;
      *(_DWORD *)(v30 + 48) = 0;
      *(_QWORD *)(v30 + 56) = 0;
      *(_QWORD *)(v30 + 80) = 0;
      sub_321E020((_QWORD *)v50, v51, v49);
      v53 = *(_BYTE *)(v30 + 88);
      if ( v53 != 4 )
        sub_435074(v53 == -1);
    }
    else
    {
      v67 = *(_DWORD *)v5;
      if ( MEMORY[0x58] != 0xFF )
      {
        v68 = (void (*)())funcs_32198D3[MEMORY[0x58]];
        goto LABEL_88;
      }
LABEL_89:
      *(_BYTE *)(v30 + 88) = 3;
      v70 = *(_QWORD *)(v5 + 16);
      *(_DWORD *)(v30 + 48) = 0;
      *(_QWORD *)(v30 + 56) = 0;
      *(_QWORD *)(v30 + 64) = v30 + 48;
      *(_QWORD *)(v30 + 72) = v30 + 48;
      *(_QWORD *)(v30 + 80) = 0;
      v119.m128i_i64[1] = v70;
      v119.m128i_i32[0] = v67;
      v71 = sub_3226CC0((_QWORD *)v69, v30 + 48, (__int64)&v119);
      if ( v72 )
      {
        v73 = (_QWORD *)(v30 + 48);
        if ( (_QWORD *)(v30 + 48) == v72 || v71 )
        {
          v74 = 1;
        }
        else
        {
          v116 = v72;
          v77 = sub_321DF40((__int64)&v119, (__int64)(v72 + 4));
          v72 = v116;
          v73 = (_QWORD *)(v30 + 48);
          v74 = v77;
        }
        v109 = v72;
        v112 = v73;
        v115 = v74;
        v75 = (__m128i *)sub_22077B0(0x30u);
        v75[2] = _mm_loadu_si128(&v119);
        sub_220F040(v115, (__int64)v75, v109, v112);
        ++*(_QWORD *)(v30 + 80);
      }
      v76 = *(_BYTE *)(v30 + 88);
      if ( v76 != 3 )
        sub_435074(v76 == -1);
    }
    sub_3245B60(a1 + 3080, v40, v30);
    if ( (v120.m128i_i8[8] & 1) == 0 )
    {
      v55 = v122;
      v56 = v121;
      if ( v122 )
      {
        v57 = v122 - 1;
        goto LABEL_66;
      }
      v20 = v120.m128i_u32[2];
      ++v120.m128i_i64[0];
      v21 = 0;
      v22 = ((unsigned __int32)v120.m128i_i32[2] >> 1) + 1;
LABEL_17:
      v23 = 3 * v55;
      goto LABEL_18;
    }
    v56 = (__int64 *)&v121;
    v57 = 3;
LABEL_66:
    v54 = 1;
    v21 = 0;
    v58 = v57
        & (((0xBF58476D1CE4E5B9LL
           * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
            | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
         ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4))));
    while ( 2 )
    {
      v24 = (__int64)&v56[3 * v58];
      v59 = *(_QWORD *)v24;
      if ( v7 == *(_QWORD *)v24 && v11 == *(_QWORD *)(v24 + 8) )
        goto LABEL_23;
      if ( v59 != -4096 )
      {
        if ( v59 == -8192 && *(_QWORD *)(v24 + 8) == -8192 && !v21 )
          v21 = &v56[3 * v58];
        goto LABEL_73;
      }
      if ( *(_QWORD *)(v24 + 8) != -4096 )
      {
LABEL_73:
        v60 = v54 + v58;
        v54 = (unsigned int)(v54 + 1);
        v58 = v57 & v60;
        continue;
      }
      break;
    }
    v20 = v120.m128i_u32[2];
    if ( !v21 )
      v21 = (__int64 *)v24;
    ++v120.m128i_i64[0];
    v22 = ((unsigned __int32)v120.m128i_i32[2] >> 1) + 1;
    if ( (v120.m128i_i8[8] & 1) == 0 )
    {
      v55 = v122;
      goto LABEL_17;
    }
    v23 = 12;
    v55 = 4;
LABEL_18:
    v24 = 4 * v22;
    if ( (unsigned int)v24 >= v23 )
    {
      sub_3229410(&v120, 2 * v55);
      if ( (v120.m128i_i8[8] & 1) != 0 )
      {
        v84 = (__int64 *)&v121;
        v85 = 3;
      }
      else
      {
        v84 = v121;
        if ( !v122 )
          goto LABEL_186;
        v85 = v122 - 1;
      }
      v54 = 1;
      v24 = 0;
      for ( n = v85
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
                  | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; n = v85 & v88 )
      {
        v21 = &v84[3 * n];
        v87 = *v21;
        if ( v7 == *v21 && v11 == v21[1] )
          break;
        if ( v87 == -4096 )
        {
          if ( v21[1] == -4096 )
          {
LABEL_173:
            if ( v24 )
              v21 = (__int64 *)v24;
            goto LABEL_139;
          }
        }
        else if ( v87 == -8192 && v21[1] == -8192 && !v24 )
        {
          v24 = (__int64)&v84[3 * n];
        }
        v88 = v54 + n;
        v54 = (unsigned int)(v54 + 1);
      }
      goto LABEL_139;
    }
    if ( v55 - v120.m128i_i32[3] - v22 > v55 >> 3 )
      goto LABEL_20;
    sub_3229410(&v120, v55);
    if ( (v120.m128i_i8[8] & 1) != 0 )
    {
      v89 = (__int64 *)&v121;
      v90 = 3;
      goto LABEL_128;
    }
    v89 = v121;
    if ( !v122 )
    {
LABEL_186:
      v120.m128i_i32[2] = (2 * ((unsigned __int32)v120.m128i_i32[2] >> 1) + 2) | v120.m128i_i8[8] & 1;
      BUG();
    }
    v90 = v122 - 1;
LABEL_128:
    v54 = 1;
    v24 = 0;
    for ( ii = v90
             & (((0xBF58476D1CE4E5B9LL
                * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
                 | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
              ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; ii = v90 & v93 )
    {
      v21 = &v89[3 * ii];
      v92 = *v21;
      if ( v7 == *v21 && v11 == v21[1] )
        break;
      if ( v92 == -4096 )
      {
        if ( v21[1] == -4096 )
          goto LABEL_173;
      }
      else if ( v92 == -8192 && v21[1] == -8192 && !v24 )
      {
        v24 = (__int64)&v89[3 * ii];
      }
      v93 = v54 + ii;
      v54 = (unsigned int)(v54 + 1);
    }
LABEL_139:
    v20 = v120.m128i_u32[2];
LABEL_20:
    v120.m128i_i32[2] = (2 * (v20 >> 1) + 2) | v20 & 1;
    if ( *v21 != -4096 || v21[1] != -4096 )
      --v120.m128i_i32[3];
    *v21 = v7;
    v21[1] = v11;
    v21[2] = v30;
LABEL_23:
    v119.m128i_i64[0] = v30;
    v25 = &v119;
    v26 = *(unsigned int *)(a1 + 768);
    v27 = *(_QWORD *)(a1 + 760);
    v28 = v26 + 1;
    LODWORD(v6) = *(_DWORD *)(a1 + 768);
    if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 772) )
    {
      v80 = a1 + 760;
      if ( v27 > (unsigned __int64)&v119 || (v81 = v27 + 8 * v26, v26 = (__int64)&v119, (unsigned __int64)&v119 >= v81) )
      {
        sub_3227150(v80, v28, v26, v27, v24, v54);
        v25 = &v119;
        v26 = *(unsigned int *)(a1 + 768);
        v27 = *(_QWORD *)(a1 + 760);
        LODWORD(v6) = *(_DWORD *)(a1 + 768);
      }
      else
      {
        v82 = &v119.m128i_i8[-v27];
        sub_3227150(v80, v28, (__int64)v119.m128i_i64 - v27, v27, v24, v54);
        v27 = *(_QWORD *)(a1 + 760);
        v26 = *(unsigned int *)(a1 + 768);
        v25 = (__m128i *)&v82[v27];
        LODWORD(v6) = *(_DWORD *)(a1 + 768);
      }
    }
    v29 = (_QWORD *)(v27 + 8 * v26);
    if ( v29 )
    {
      *v29 = v25->m128i_i64[0];
      v25->m128i_i64[0] = 0;
      v30 = v119.m128i_i64[0];
      LODWORD(v6) = *(_DWORD *)(a1 + 768);
    }
    LODWORD(v6) = v6 + 1;
    *(_DWORD *)(a1 + 768) = v6;
    if ( v30 )
      LOBYTE(v6) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
LABEL_28:
    v5 += 32;
    if ( v107 != v5 )
      continue;
    break;
  }
LABEL_29:
  if ( (v120.m128i_i8[8] & 1) == 0 )
    LOBYTE(v6) = sub_C7D6A0((__int64)v121, 24LL * v122, 8);
  return v6;
}
