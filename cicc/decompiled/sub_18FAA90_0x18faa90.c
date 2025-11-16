// Function: sub_18FAA90
// Address: 0x18faa90
//
__int64 __fastcall sub_18FAA90(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // r13
  char v14; // al
  __int64 v15; // r12
  __int64 *v16; // rax
  unsigned int v17; // esi
  __int64 v18; // rdx
  __int64 v19; // r15
  int v20; // r11d
  __int64 v21; // r10
  int v22; // r14d
  unsigned int n; // ecx
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned int v26; // ecx
  __int64 *v27; // rax
  unsigned int v28; // esi
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 v31; // r10
  int v32; // r11d
  unsigned int i; // ecx
  __int64 v34; // rax
  __int64 v35; // r9
  unsigned int v36; // ecx
  __int64 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  const __m128i *v40; // rdi
  const __m128i *v41; // rbx
  __int64 v42; // r9
  __int64 v43; // r14
  int v44; // r15d
  unsigned int ii; // edx
  __int64 v46; // rax
  __int64 v47; // r13
  unsigned int v48; // edx
  __int64 *v49; // rax
  __int64 *v50; // rax
  __int64 v51; // r8
  int v52; // r9d
  unsigned int m; // ecx
  __int64 v54; // r10
  unsigned int v55; // ecx
  __int64 v56; // rdx
  __int64 v57; // rsi
  int v58; // r8d
  unsigned int k; // edx
  __int64 v60; // r9
  unsigned int v61; // edx
  int v62; // edi
  __int64 v63; // r14
  __int64 v64; // r15
  char v65; // r13
  char v66; // r13
  __int64 v67; // rsi
  char v69; // r9
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 *v72; // r13
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r10
  __int64 v76; // r13
  double v77; // xmm4_8
  double v78; // xmm5_8
  __int64 v79; // r9
  __int64 v80; // r8
  int v81; // r14d
  unsigned int v82; // edx
  __int64 v83; // r11
  unsigned int v84; // edx
  int v85; // ecx
  int v86; // edi
  __int64 v87; // r13
  int v88; // r14d
  __int64 v89; // r8
  unsigned int jj; // edx
  __int64 v91; // r10
  unsigned int v92; // edx
  __int64 v93; // rdx
  int v94; // edi
  unsigned int v95; // r14d
  unsigned int v96; // r14d
  __int64 v97; // r8
  int v98; // edi
  unsigned int j; // edx
  __int64 v100; // r8
  unsigned int v101; // edx
  char v102; // al
  __int64 v103; // [rsp+8h] [rbp-E8h]
  __int64 v104; // [rsp+8h] [rbp-E8h]
  __int64 v105; // [rsp+10h] [rbp-E0h]
  __int64 v106; // [rsp+10h] [rbp-E0h]
  __int64 v109; // [rsp+28h] [rbp-C8h]
  char v110; // [rsp+28h] [rbp-C8h]
  __int64 v111; // [rsp+30h] [rbp-C0h]
  unsigned __int8 v112; // [rsp+30h] [rbp-C0h]
  __int64 v113; // [rsp+30h] [rbp-C0h]
  __int64 v114; // [rsp+30h] [rbp-C0h]
  const __m128i *v115; // [rsp+38h] [rbp-B8h]
  __m128i v116; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v117; // [rsp+50h] [rbp-A0h]
  __int64 v118; // [rsp+60h] [rbp-90h] BYREF
  __int64 v119; // [rsp+68h] [rbp-88h]
  __int64 v120; // [rsp+70h] [rbp-80h]
  unsigned int v121; // [rsp+78h] [rbp-78h]
  __int64 v122; // [rsp+80h] [rbp-70h] BYREF
  __int64 v123; // [rsp+88h] [rbp-68h]
  __int64 v124; // [rsp+90h] [rbp-60h]
  int v125; // [rsp+98h] [rbp-58h]
  const __m128i *v126; // [rsp+A0h] [rbp-50h]
  const __m128i *v127; // [rsp+A8h] [rbp-48h]
  __int64 v128; // [rsp+B0h] [rbp-40h]

  v11 = *(_QWORD *)(a1 + 80);
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v109 = a1 + 72;
  if ( v11 == a1 + 72 )
  {
    v112 = 0;
    goto LABEL_83;
  }
  do
  {
    if ( !v11 )
      BUG();
    v12 = *(_QWORD *)(v11 + 24);
    v13 = v11 + 16;
    if ( v12 != v11 + 16 )
    {
      v111 = v11;
      while ( 1 )
      {
        while ( 1 )
        {
          if ( !v12 )
            BUG();
          v14 = *(_BYTE *)(v12 - 8);
          v15 = v12 - 24;
          if ( v14 == 42 )
            break;
          switch ( v14 )
          {
            case ')':
              if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
              {
                v27 = *(__int64 **)(v12 - 32);
                v28 = v121;
                v29 = v27[3];
                v30 = *v27;
                if ( v121 )
                {
LABEL_19:
                  v31 = 0;
                  v32 = 1;
                  for ( i = (v28 - 1) & (v29 ^ v30); ; i = (v28 - 1) & v36 )
                  {
                    v34 = v119 + 32LL * i;
                    v35 = *(_QWORD *)(v34 + 8);
                    if ( *(_BYTE *)v34 )
                    {
                      if ( !v35 && !(*(_QWORD *)(v34 + 16) | v31) )
                        v31 = v119 + 32LL * i;
                    }
                    else
                    {
                      if ( v30 == v35 && v29 == *(_QWORD *)(v34 + 16) )
                      {
                        *(_QWORD *)(v34 + 24) = v15;
                        goto LABEL_8;
                      }
                      if ( !v35 && !*(_QWORD *)(v34 + 16) )
                      {
                        v103 = 0;
                        if ( v31 )
                          v34 = v31;
                        ++v118;
                        v62 = v120 + 1;
                        if ( 4 * ((int)v120 + 1) < 3 * v28 )
                        {
                          if ( v28 - HIDWORD(v120) - v62 <= v28 >> 3 )
                          {
                            sub_18FA270((__int64)&v118, v28);
                            if ( !v121 )
                            {
LABEL_219:
                              LODWORD(v120) = v120 + 1;
                              BUG();
                            }
                            v98 = 1;
                            for ( j = (v121 - 1) & (v29 ^ v30); ; j = (v121 - 1) & v101 )
                            {
                              v34 = v119 + 32LL * j;
                              v100 = *(_QWORD *)(v34 + 8);
                              if ( *(_BYTE *)v34 )
                              {
                                if ( !v100 )
                                {
                                  if ( *(_QWORD *)(v34 + 16) | v103 )
                                    v34 = v103;
                                  v103 = v34;
                                }
                              }
                              else
                              {
                                if ( v30 == v100 && v29 == *(_QWORD *)(v34 + 16) )
                                  goto LABEL_191;
                                if ( !v100 && !*(_QWORD *)(v34 + 16) )
                                {
                                  v62 = v120 + 1;
                                  if ( v103 )
                                    v34 = v103;
                                  break;
                                }
                              }
                              v101 = v98 + j;
                              ++v98;
                            }
                          }
LABEL_68:
                          LODWORD(v120) = v62;
                          if ( *(_BYTE *)v34 || *(_QWORD *)(v34 + 8) || *(_QWORD *)(v34 + 16) )
                            --HIDWORD(v120);
                          *(_QWORD *)(v34 + 24) = 0;
                          *(_BYTE *)v34 = 0;
                          *(_QWORD *)(v34 + 8) = v30;
                          *(_QWORD *)(v34 + 16) = v29;
                          *(_QWORD *)(v34 + 24) = v15;
                          goto LABEL_8;
                        }
LABEL_49:
                        sub_18FA270((__int64)&v118, 2 * v28);
                        if ( !v121 )
                          goto LABEL_219;
                        v57 = 0;
                        v58 = 1;
                        for ( k = (v121 - 1) & (v29 ^ v30); ; k = (v121 - 1) & v61 )
                        {
                          v34 = v119 + 32LL * k;
                          v60 = *(_QWORD *)(v34 + 8);
                          if ( *(_BYTE *)v34 )
                          {
                            if ( !v60 && !(*(_QWORD *)(v34 + 16) | v57) )
                              v57 = v119 + 32LL * k;
                          }
                          else
                          {
                            if ( v30 == v60 && v29 == *(_QWORD *)(v34 + 16) )
                            {
LABEL_191:
                              v62 = v120 + 1;
                              goto LABEL_68;
                            }
                            if ( !v60 && !*(_QWORD *)(v34 + 16) )
                            {
                              v62 = v120 + 1;
                              if ( v57 )
                                v34 = v57;
                              goto LABEL_68;
                            }
                          }
                          v61 = v58 + k;
                          ++v58;
                        }
                      }
                    }
                    v36 = v32 + i;
                    ++v32;
                  }
                }
              }
              else
              {
                v28 = v121;
                v56 = 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF);
                v29 = *(_QWORD *)(v12 - v56);
                v30 = *(_QWORD *)(v15 - v56);
                if ( v121 )
                  goto LABEL_19;
              }
              ++v118;
              goto LABEL_49;
            case '-':
              if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
                v37 = *(__int64 **)(v12 - 32);
              else
                v37 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
              v38 = v37[3];
              v39 = *v37;
              v116.m128i_i8[0] = 1;
              break;
            case ',':
              if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
                v49 = *(__int64 **)(v12 - 32);
              else
                v49 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
              v38 = v49[3];
              v39 = *v49;
              v116.m128i_i8[0] = 0;
              break;
            default:
              goto LABEL_8;
          }
          v116.m128i_i64[1] = v39;
          v117 = v38;
          *(_QWORD *)sub_18FA710((__int64)&v122, &v116) = v15;
          v12 = *(_QWORD *)(v12 + 8);
          if ( v13 == v12 )
            goto LABEL_27;
        }
        if ( (*(_BYTE *)(v12 - 1) & 0x40) != 0 )
        {
          v16 = *(__int64 **)(v12 - 32);
          v17 = v121;
          v18 = v16[3];
          v19 = *v16;
          if ( !v121 )
            goto LABEL_41;
        }
        else
        {
          v17 = v121;
          v50 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
          v18 = v50[3];
          v19 = *v50;
          if ( !v121 )
          {
LABEL_41:
            ++v118;
LABEL_42:
            v105 = v18;
            sub_18FA270((__int64)&v118, 2 * v17);
            if ( !v121 )
              goto LABEL_222;
            v18 = v105;
            v51 = 0;
            v52 = 1;
            for ( m = (v121 - 1) & (v105 ^ v19 ^ 1); ; m = (v121 - 1) & v55 )
            {
              v24 = v119 + 32LL * m;
              v54 = *(_QWORD *)(v24 + 8);
              if ( *(_BYTE *)v24 )
              {
                if ( v19 == v54 && v105 == *(_QWORD *)(v24 + 16) )
                  goto LABEL_189;
                if ( !v54 && !(*(_QWORD *)(v24 + 16) | v51) )
                  v51 = v119 + 32LL * m;
              }
              else if ( !v54 && !*(_QWORD *)(v24 + 16) )
              {
                v86 = v120 + 1;
                if ( v51 )
                  v24 = v51;
                goto LABEL_111;
              }
              v55 = v52 + m;
              ++v52;
            }
          }
        }
        v20 = 1;
        v21 = 0;
        v22 = v18 ^ v19 ^ 1;
        for ( n = (v17 - 1) & v22; ; n = (v17 - 1) & v26 )
        {
          v24 = v119 + 32LL * n;
          v25 = *(_QWORD *)(v24 + 8);
          if ( !*(_BYTE *)v24 )
            break;
          if ( v19 == v25 && v18 == *(_QWORD *)(v24 + 16) )
          {
            *(_QWORD *)(v24 + 24) = v15;
            goto LABEL_8;
          }
          if ( !v25 && !(*(_QWORD *)(v24 + 16) | v21) )
            v21 = v119 + 32LL * n;
LABEL_16:
          v26 = v20 + n;
          ++v20;
        }
        if ( v25 || *(_QWORD *)(v24 + 16) )
          goto LABEL_16;
        v104 = 0;
        if ( v21 )
          v24 = v21;
        ++v118;
        v86 = v120 + 1;
        if ( 4 * ((int)v120 + 1) >= 3 * v17 )
          goto LABEL_42;
        if ( v17 - HIDWORD(v120) - v86 <= v17 >> 3 )
        {
          v106 = v18;
          sub_18FA270((__int64)&v118, v17);
          if ( !v121 )
          {
LABEL_222:
            LODWORD(v120) = v120 + 1;
            BUG();
          }
          v94 = 1;
          v18 = v106;
          v95 = (v121 - 1) & v22;
          v24 = v119 + 32LL * v95;
          if ( *(_BYTE *)v24 )
            goto LABEL_169;
          while ( *(_QWORD *)(v24 + 8) || *(_QWORD *)(v24 + 16) )
          {
            while ( 1 )
            {
              v96 = v94 + v95;
              ++v94;
              v95 = (v121 - 1) & v96;
              v24 = v119 + 32LL * v95;
              if ( !*(_BYTE *)v24 )
                break;
LABEL_169:
              v97 = *(_QWORD *)(v24 + 8);
              if ( v19 == v97 && v106 == *(_QWORD *)(v24 + 16) )
                goto LABEL_189;
              if ( !v97 )
              {
                if ( *(_QWORD *)(v24 + 16) | v104 )
                  v24 = v104;
                v104 = v24;
              }
            }
          }
          if ( v104 )
          {
            v86 = v120 + 1;
            v24 = v104;
            goto LABEL_111;
          }
LABEL_189:
          v86 = v120 + 1;
        }
LABEL_111:
        LODWORD(v120) = v86;
        if ( *(_BYTE *)v24 || *(_QWORD *)(v24 + 8) || *(_QWORD *)(v24 + 16) )
          --HIDWORD(v120);
        *(_QWORD *)(v24 + 24) = 0;
        *(_BYTE *)v24 = 1;
        *(_QWORD *)(v24 + 8) = v19;
        *(_QWORD *)(v24 + 16) = v18;
        *(_QWORD *)(v24 + 24) = v15;
LABEL_8:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
        {
LABEL_27:
          v11 = v111;
          break;
        }
      }
    }
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v109 != v11 );
  v40 = v126;
  if ( v127 == v126 )
  {
    v112 = 0;
    v67 = v128 - (_QWORD)v126;
    goto LABEL_81;
  }
  v112 = 0;
  v41 = v126;
  v115 = v127;
  do
  {
    if ( !v121 )
    {
      ++v118;
      goto LABEL_97;
    }
    v42 = v41[1].m128i_i64[0];
    v43 = 0;
    v44 = 1;
    for ( ii = (v121 - 1) & (v41->m128i_u8[0] ^ v42 ^ v41->m128i_i64[1]); ; ii = (v121 - 1) & v48 )
    {
      v46 = v119 + 32LL * ii;
      v47 = *(_QWORD *)(v46 + 8);
      if ( v41->m128i_i8[0] == *(_BYTE *)v46 && v41->m128i_i64[1] == v47 && v42 == *(_QWORD *)(v46 + 16) )
      {
        v63 = *(_QWORD *)(v46 + 24);
        if ( v63 )
        {
          v64 = v41[1].m128i_i64[1];
          if ( (unsigned __int8)sub_14A2BF0(a2, *(_QWORD *)v63, *(_BYTE *)(v63 + 16) == 42) )
          {
            if ( *(_QWORD *)(v63 + 40) != *(_QWORD *)(v64 + 40) )
            {
              v65 = sub_15CCEE0(a3, v63, v64);
              if ( v65 )
              {
                sub_15F2300((_QWORD *)v64, v63);
                v112 = v65;
              }
              else
              {
                v66 = sub_15CCEE0(a3, v64, v63);
                if ( v66 )
                {
                  sub_15F2300((_QWORD *)v63, v64);
                  v112 = v66;
                }
              }
            }
          }
          else
          {
            v69 = sub_15CCEE0(a3, v63, v64);
            if ( v69 || (v102 = sub_15CCEE0(a3, v64, v63), v69 = 0, v102) )
            {
              if ( (*(_BYTE *)(v64 + 23) & 0x40) != 0 )
                v70 = *(_QWORD *)(v64 - 8);
              else
                v70 = v64 - 24LL * (*(_DWORD *)(v64 + 20) & 0xFFFFFFF);
              v71 = *(_QWORD *)(v70 + 24);
              v72 = *(__int64 **)v70;
              v110 = v69;
              LOWORD(v117) = 257;
              v73 = sub_15FB440(15, (__int64 *)v63, v71, (__int64)&v116, 0);
              LOWORD(v117) = 257;
              v113 = v73;
              v74 = sub_15FB440(13, v72, v73, (__int64)&v116, 0);
              v75 = v113;
              v76 = v74;
              if ( !v110 )
              {
                sub_15F22F0((_QWORD *)v63, v64);
                v75 = v113;
              }
              v114 = v75;
              sub_15F2180(v75, v64);
              sub_15F2180(v76, v114);
              sub_164D160(v64, v76, a4, a5, a6, a7, v77, v78, a10, a11);
              sub_15F20C0((_QWORD *)v64);
              v112 = 1;
            }
          }
        }
        goto LABEL_79;
      }
      if ( !*(_BYTE *)v46 )
        break;
      if ( !v47 && !(*(_QWORD *)(v46 + 16) | v43) )
        v43 = v119 + 32LL * ii;
LABEL_36:
      v48 = v44 + ii;
      ++v44;
    }
    if ( v47 )
      goto LABEL_36;
    v87 = *(_QWORD *)(v46 + 16);
    if ( v87 )
      goto LABEL_36;
    if ( v43 )
      v46 = v43;
    ++v118;
    v85 = v120 + 1;
    if ( 4 * ((int)v120 + 1) < 3 * v121 )
    {
      if ( v121 - HIDWORD(v120) - v85 > v121 >> 3 )
        goto LABEL_143;
      sub_18FA270((__int64)&v118, v121);
      if ( v121 )
      {
        v88 = 1;
        v89 = v41[1].m128i_i64[0];
        for ( jj = (v121 - 1) & (v41->m128i_u8[0] ^ v89 ^ v41->m128i_i64[1]); ; jj = (v121 - 1) & v92 )
        {
          v46 = v119 + 32LL * jj;
          v91 = *(_QWORD *)(v46 + 8);
          if ( v41->m128i_i8[0] == *(_BYTE *)v46 && v41->m128i_i64[1] == v91 && v89 == *(_QWORD *)(v46 + 16) )
          {
            v85 = v120 + 1;
            goto LABEL_143;
          }
          if ( *(_BYTE *)v46 )
          {
            if ( !v91 && !(*(_QWORD *)(v46 + 16) | v87) )
              v87 = v119 + 32LL * jj;
          }
          else if ( !v91 && !*(_QWORD *)(v46 + 16) )
          {
            if ( v87 )
              v46 = v87;
            v85 = v120 + 1;
            goto LABEL_143;
          }
          v92 = v88 + jj;
          ++v88;
        }
      }
LABEL_220:
      LODWORD(v120) = v120 + 1;
      BUG();
    }
LABEL_97:
    sub_18FA270((__int64)&v118, 2 * v121);
    if ( !v121 )
      goto LABEL_220;
    v79 = v41[1].m128i_i64[0];
    v80 = 0;
    v81 = 1;
    v82 = (v121 - 1) & (v41->m128i_u8[0] ^ v79 ^ v41->m128i_i64[1]);
    while ( 2 )
    {
      v46 = v119 + 32LL * v82;
      v83 = *(_QWORD *)(v46 + 8);
      if ( v41->m128i_i8[0] == *(_BYTE *)v46 && v83 == v41->m128i_i64[1] && v79 == *(_QWORD *)(v46 + 16) )
      {
        v85 = v120 + 1;
        goto LABEL_143;
      }
      if ( *(_BYTE *)v46 )
      {
        if ( !v83 && !(*(_QWORD *)(v46 + 16) | v80) )
          v80 = v119 + 32LL * v82;
        goto LABEL_102;
      }
      if ( v83 || *(_QWORD *)(v46 + 16) )
      {
LABEL_102:
        v84 = v81 + v82;
        ++v81;
        v82 = (v121 - 1) & v84;
        continue;
      }
      break;
    }
    if ( v80 )
      v46 = v80;
    v85 = v120 + 1;
LABEL_143:
    LODWORD(v120) = v85;
    if ( *(_BYTE *)v46 || *(_QWORD *)(v46 + 8) || *(_QWORD *)(v46 + 16) )
      --HIDWORD(v120);
    a4 = (__m128)_mm_loadu_si128(v41);
    *(__m128 *)v46 = a4;
    v93 = v41[1].m128i_i64[0];
    *(_QWORD *)(v46 + 24) = 0;
    *(_QWORD *)(v46 + 16) = v93;
LABEL_79:
    v41 += 2;
  }
  while ( v115 != v41 );
  v40 = v126;
  v67 = v128 - (_QWORD)v126;
LABEL_81:
  if ( v40 )
    j_j___libc_free_0(v40, v67);
LABEL_83:
  j___libc_free_0(v123);
  j___libc_free_0(v119);
  return v112;
}
