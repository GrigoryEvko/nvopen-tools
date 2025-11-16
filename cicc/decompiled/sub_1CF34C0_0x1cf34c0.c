// Function: sub_1CF34C0
// Address: 0x1cf34c0
//
unsigned __int64 __fastcall sub_1CF34C0(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 result; // rax
  __int64 v11; // r14
  __int64 v13; // rbx
  _QWORD *v15; // rax
  unsigned int v16; // esi
  _QWORD *v17; // rdx
  int v18; // r10d
  __int64 *v19; // r12
  unsigned int v20; // r8d
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // r10
  __int64 v27; // r13
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // r12
  __int64 v31; // rdi
  unsigned __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 *v34; // rdi
  __int64 v35; // rax
  __int64 *v36; // r12
  __int64 v37; // rbx
  __int64 *v38; // r15
  __int64 v39; // r8
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // rcx
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 *v46; // r14
  double v47; // xmm4_8
  double v48; // xmm5_8
  _QWORD *v49; // rax
  _QWORD *v50; // r13
  char **v51; // rax
  __int64 v52; // rcx
  int v53; // r8d
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 *v56; // r12
  double v57; // xmm4_8
  double v58; // xmm5_8
  _QWORD *v59; // rax
  __int64 v60; // rdx
  _QWORD *v61; // r11
  _QWORD *v62; // rbx
  _QWORD *v63; // r12
  __int64 v64; // rsi
  __int64 v65; // rax
  unsigned int v66; // esi
  __int64 v67; // r10
  unsigned int v68; // edi
  _QWORD *v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rcx
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rdi
  __int64 v75; // rdi
  _QWORD *v76; // rsi
  __int64 v77; // r9
  __int64 v78; // r8
  __int64 v79; // rdi
  __int64 v80; // rsi
  _QWORD *v81; // rsi
  __int64 v82; // r9
  __int64 v83; // r8
  int v84; // edx
  int v85; // esi
  __int64 v86; // r9
  int v87; // ecx
  unsigned int v88; // edi
  _QWORD *v89; // r11
  __int64 v90; // rdx
  int v91; // ecx
  int v92; // edx
  int v93; // eax
  _QWORD *v94; // rdx
  __int64 v95; // rsi
  int v96; // r9d
  __int64 v97; // rdi
  __int64 *v98; // rdx
  int v99; // esi
  __int64 v100; // rcx
  __int64 v101; // rdi
  int v102; // esi
  __int64 *v103; // rcx
  int v104; // [rsp+Ch] [rbp-134h]
  __int64 v105; // [rsp+10h] [rbp-130h]
  __int64 v106; // [rsp+18h] [rbp-128h]
  int v107; // [rsp+18h] [rbp-128h]
  __int64 v108; // [rsp+18h] [rbp-128h]
  __int64 v109; // [rsp+20h] [rbp-120h]
  _QWORD *v110; // [rsp+28h] [rbp-118h]
  __int64 *v111; // [rsp+30h] [rbp-110h]
  __int64 v112; // [rsp+40h] [rbp-100h]
  unsigned int v113; // [rsp+40h] [rbp-100h]
  __int64 v114; // [rsp+40h] [rbp-100h]
  int v115; // [rsp+40h] [rbp-100h]
  _QWORD *src; // [rsp+50h] [rbp-F0h]
  void *srca; // [rsp+50h] [rbp-F0h]
  __int64 *v118; // [rsp+58h] [rbp-E8h]
  __m128i v119; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v120; // [rsp+70h] [rbp-D0h]
  void *v121; // [rsp+88h] [rbp-B8h] BYREF
  __m128i v122; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v123; // [rsp+A0h] [rbp-A0h]
  char **v124; // [rsp+B0h] [rbp-90h] BYREF
  void **v125; // [rsp+B8h] [rbp-88h]
  __int64 v126; // [rsp+C0h] [rbp-80h]
  __int64 v127; // [rsp+D0h] [rbp-70h] BYREF
  _QWORD *v128; // [rsp+D8h] [rbp-68h]
  __int64 v129; // [rsp+E0h] [rbp-60h]
  unsigned int v130; // [rsp+E8h] [rbp-58h]
  __m128i v131; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v132; // [rsp+100h] [rbp-40h]
  char v133; // [rsp+108h] [rbp-38h]

  v110 = a2 + 5;
  result = a2[5] & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 5 != (_QWORD *)result )
  {
    v11 = a2[6];
    if ( !v11 )
LABEL_166:
      BUG();
    if ( *(_BYTE *)(v11 - 8) == 77 )
    {
      v13 = a2[1];
      v127 = 0;
      v128 = 0;
      v129 = 0;
      v130 = 0;
      if ( v13 )
      {
        while ( 1 )
        {
          v15 = sub_1648700(v13);
          if ( (unsigned __int8)(*((_BYTE *)v15 + 16) - 25) <= 9u )
            break;
          v13 = *(_QWORD *)(v13 + 8);
          if ( !v13 )
            goto LABEL_20;
        }
        v16 = 0;
        v17 = 0;
LABEL_12:
        v23 = v15[5];
        if ( v16 )
        {
          v18 = 1;
          v19 = 0;
          v20 = (v16 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v21 = &v17[4 * v20];
          v22 = *v21;
          if ( v23 == *v21 )
            goto LABEL_9;
          while ( v22 != -8 )
          {
            if ( !v19 && v22 == -16 )
              v19 = v21;
            v20 = (v16 - 1) & (v18 + v20);
            v21 = &v17[4 * v20];
            v22 = *v21;
            if ( v23 == *v21 )
            {
              do
              {
LABEL_9:
                v13 = *(_QWORD *)(v13 + 8);
                if ( !v13 )
                  goto LABEL_19;
LABEL_10:
                v15 = sub_1648700(v13);
              }
              while ( (unsigned __int8)(*((_BYTE *)v15 + 16) - 25) > 9u );
              v17 = v128;
              v16 = v130;
              goto LABEL_12;
            }
            ++v18;
          }
          if ( !v19 )
            v19 = v21;
          ++v127;
          v25 = v129 + 1;
          if ( 4 * ((int)v129 + 1) >= 3 * v16 )
            goto LABEL_14;
          if ( v16 - (v25 + HIDWORD(v129)) <= v16 >> 3 )
          {
            sub_1CF32F0((__int64)&v127, v16);
            if ( v130 )
            {
              v98 = 0;
              v99 = 1;
              LODWORD(v100) = (v130 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v25 = v129 + 1;
              v19 = &v128[4 * (unsigned int)v100];
              v101 = *v19;
              if ( v23 != *v19 )
              {
                while ( v101 != -8 )
                {
                  if ( v101 == -16 && !v98 )
                    v98 = v19;
                  v100 = (v130 - 1) & ((_DWORD)v100 + v99);
                  v19 = &v128[4 * v100];
                  v101 = *v19;
                  if ( v23 == *v19 )
                    goto LABEL_16;
                  ++v99;
                }
                if ( v98 )
                  v19 = v98;
              }
              goto LABEL_16;
            }
LABEL_167:
            LODWORD(v129) = v129 + 1;
            BUG();
          }
        }
        else
        {
          ++v127;
LABEL_14:
          sub_1CF32F0((__int64)&v127, 2 * v16);
          if ( !v130 )
            goto LABEL_167;
          LODWORD(v24) = (v130 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v25 = v129 + 1;
          v19 = &v128[4 * (unsigned int)v24];
          v26 = *v19;
          if ( v23 != *v19 )
          {
            v102 = 1;
            v103 = 0;
            while ( v26 != -8 )
            {
              if ( v26 == -16 && !v103 )
                v103 = v19;
              v24 = (v130 - 1) & ((_DWORD)v24 + v102);
              v19 = &v128[4 * v24];
              v26 = *v19;
              if ( v23 == *v19 )
                goto LABEL_16;
              ++v102;
            }
            if ( v103 )
              v19 = v103;
          }
        }
LABEL_16:
        LODWORD(v129) = v25;
        if ( *v19 != -8 )
          --HIDWORD(v129);
        *v19 = v23;
        v19[1] = 0;
        v19[2] = 0;
        *((_DWORD *)v19 + 6) = 0;
        v131.m128i_i8[8] = 0;
        sub_1CF2B90((__int64)&v119, a1, v23, (__int64)&v131);
        a4 = (__m128)_mm_loadu_si128(&v119);
        *(__m128 *)(v19 + 1) = a4;
        v19[3] = v120;
        v13 = *(_QWORD *)(v13 + 8);
        if ( v13 )
          goto LABEL_10;
LABEL_19:
        v11 = a2[6];
      }
LABEL_20:
      if ( v110 == (_QWORD *)v11 )
        return j___libc_free_0(v128);
      while ( 1 )
      {
        if ( !v11 )
          goto LABEL_166;
        v27 = v11 - 24;
        if ( *(_BYTE *)(v11 - 8) != 77 )
          return j___libc_free_0(v128);
        v133 = 0;
        v111 = (__int64 *)(v11 - 24);
        v118 = sub_1CF27D0(a1, v11 - 24, &v131);
        if ( (_DWORD)v129 )
        {
          v59 = v128;
          v60 = 4LL * v130;
          v61 = &v128[v60];
          if ( v128 != &v128[v60] )
          {
            while ( 1 )
            {
              v62 = v59;
              if ( *v59 != -8 && *v59 != -16 )
                break;
              v59 += 4;
              if ( v61 == v59 )
                goto LABEL_24;
            }
            if ( v61 != v59 )
            {
              v63 = &v128[v60];
              v109 = a1 + 56;
              while ( 1 )
              {
                LOBYTE(v125) = 0;
                sub_1CF2F90(&v122, a1, v11 - 24, (const __m128i *)(v62 + 1), (__int64)&v124);
                a3 = (__m128)_mm_loadu_si128(&v122);
                v132 = v123;
                v131 = (__m128i)a3;
                v64 = v118[1];
                if ( v64 == v118[2] )
                {
                  sub_1CF1160((__int64)v118, (_BYTE *)v64, &v131);
                  v66 = *(_DWORD *)(a1 + 80);
                  v65 = v118[1];
                  v67 = *(_QWORD *)(v65 - 24);
                  if ( !v66 )
                    goto LABEL_92;
                }
                else
                {
                  if ( v64 )
                  {
                    *(__m128 *)v64 = a3;
                    *(_QWORD *)(v64 + 16) = v132;
                    v64 = v118[1];
                  }
                  v65 = v64 + 24;
                  v118[1] = v64 + 24;
                  v66 = *(_DWORD *)(a1 + 80);
                  v67 = *(_QWORD *)(v65 - 24);
                  if ( !v66 )
                  {
LABEL_92:
                    ++*(_QWORD *)(a1 + 56);
                    goto LABEL_93;
                  }
                }
                v28 = *(_QWORD *)(a1 + 64);
                v113 = ((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4);
                v68 = (v66 - 1) & v113;
                v69 = (_QWORD *)(v28 + 16LL * v68);
                v70 = *v69;
                if ( v67 == *v69 )
                  goto LABEL_59;
                v107 = 1;
                v89 = 0;
                while ( v70 != -8 )
                {
                  if ( v70 != -16 || v89 )
                    v69 = v89;
                  v68 = (v66 - 1) & (v107 + v68);
                  v70 = *(_QWORD *)(v28 + 16LL * v68);
                  if ( v67 == v70 )
                    goto LABEL_59;
                  ++v107;
                  v89 = v69;
                  v69 = (_QWORD *)(v28 + 16LL * v68);
                }
                v91 = *(_DWORD *)(a1 + 72);
                if ( !v89 )
                  v89 = v69;
                ++*(_QWORD *)(a1 + 56);
                v87 = v91 + 1;
                if ( 4 * v87 < 3 * v66 )
                {
                  if ( v66 - *(_DWORD *)(a1 + 76) - v87 <= v66 >> 3 )
                  {
                    v105 = v65;
                    v108 = v67;
                    sub_1CF2610(v109, v66);
                    v92 = *(_DWORD *)(a1 + 80);
                    if ( !v92 )
                    {
LABEL_168:
                      ++*(_DWORD *)(a1 + 72);
                      BUG();
                    }
                    v93 = v92 - 1;
                    v94 = 0;
                    v95 = *(_QWORD *)(a1 + 64);
                    v67 = v108;
                    v104 = v93;
                    v96 = 1;
                    v28 = v93 & v113;
                    v87 = *(_DWORD *)(a1 + 72) + 1;
                    v65 = v105;
                    v89 = (_QWORD *)(v95 + 16LL * (unsigned int)v28);
                    v97 = *v89;
                    if ( v108 != *v89 )
                    {
                      while ( v97 != -8 )
                      {
                        if ( !v94 && v97 == -16 )
                          v94 = v89;
                        v28 = v104 & (unsigned int)(v96 + v28);
                        v89 = (_QWORD *)(v95 + 16LL * (unsigned int)v28);
                        v97 = *v89;
                        if ( v108 == *v89 )
                          goto LABEL_106;
                        ++v96;
                      }
                      if ( v94 )
                        v89 = v94;
                    }
                  }
                  goto LABEL_106;
                }
LABEL_93:
                v106 = v65;
                v114 = v67;
                sub_1CF2610(v109, 2 * v66);
                v84 = *(_DWORD *)(a1 + 80);
                if ( !v84 )
                  goto LABEL_168;
                v67 = v114;
                v85 = v84 - 1;
                v86 = *(_QWORD *)(a1 + 64);
                v87 = *(_DWORD *)(a1 + 72) + 1;
                v65 = v106;
                v88 = (v84 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
                v89 = (_QWORD *)(v86 + 16LL * v88);
                v90 = *v89;
                if ( v114 != *v89 )
                {
                  v115 = 1;
                  v28 = 0;
                  while ( v90 != -8 )
                  {
                    if ( v90 == -16 && !v28 )
                      v28 = (__int64)v89;
                    v88 = v85 & (v115 + v88);
                    v89 = (_QWORD *)(v86 + 16LL * v88);
                    v90 = *v89;
                    if ( v67 == *v89 )
                      goto LABEL_106;
                    ++v115;
                  }
                  if ( v28 )
                    v89 = (_QWORD *)v28;
                }
LABEL_106:
                *(_DWORD *)(a1 + 72) = v87;
                if ( *v89 != -8 )
                  --*(_DWORD *)(a1 + 76);
                *v89 = v67;
                v89[1] = v118;
LABEL_59:
                v29 = *(unsigned int *)(v11 - 4);
                v71 = 0;
                v72 = 0;
                v73 = *(_DWORD *)(v11 - 4) & 0xFFFFFFF;
                if ( v73 )
                {
                  do
                  {
                    if ( (*(_BYTE *)(v11 - 1) & 0x40) != 0 )
                      v74 = *(_QWORD *)(v11 - 32);
                    else
                      v74 = v27 - 24LL * v73;
                    v28 = 3LL * *(unsigned int *)(v11 + 32);
                    if ( *v62 == *(_QWORD *)(v74 + 8 * v72 + 24LL * *(unsigned int *)(v11 + 32) + 8) )
                    {
                      v28 = *(_QWORD *)(v65 - 24);
                      v75 = *(_QWORD *)(v74 + v71);
                      v76 = (_QWORD *)(v28 + 24 * (1LL - (*(_DWORD *)(v28 + 20) & 0xFFFFFFF)));
                      if ( *v76 )
                      {
                        v77 = v76[1];
                        v28 = v76[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v28 = v77;
                        if ( v77 )
                        {
                          v28 |= *(_QWORD *)(v77 + 16) & 3LL;
                          *(_QWORD *)(v77 + 16) = v28;
                        }
                      }
                      *v76 = v75;
                      if ( v75 )
                      {
                        v78 = *(_QWORD *)(v75 + 8);
                        v76[1] = v78;
                        if ( v78 )
                          *(_QWORD *)(v78 + 16) = (unsigned __int64)(v76 + 1) | *(_QWORD *)(v78 + 16) & 3LL;
                        v28 = (v75 + 8) | v76[2] & 3LL;
                        v76[2] = v28;
                        *(_QWORD *)(v75 + 8) = v76;
                      }
                      v79 = *(_QWORD *)(v65 - 24);
                      if ( (*(_BYTE *)(v11 - 1) & 0x40) != 0 )
                      {
                        v80 = *(_QWORD *)(v11 - 32);
                      }
                      else
                      {
                        v28 = 24LL * (*(_DWORD *)(v11 - 4) & 0xFFFFFFF);
                        v80 = v27 - v28;
                      }
                      v81 = (_QWORD *)(v71 + v80);
                      if ( *v81 )
                      {
                        v82 = v81[1];
                        v28 = v81[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v28 = v82;
                        if ( v82 )
                        {
                          v28 |= *(_QWORD *)(v82 + 16) & 3LL;
                          *(_QWORD *)(v82 + 16) = v28;
                        }
                      }
                      *v81 = v79;
                      if ( v79 )
                      {
                        v83 = *(_QWORD *)(v79 + 8);
                        v81[1] = v83;
                        if ( v83 )
                          *(_QWORD *)(v83 + 16) = (unsigned __int64)(v81 + 1) | *(_QWORD *)(v83 + 16) & 3LL;
                        v28 = (v79 + 8) | v81[2] & 3LL;
                        v81[2] = v28;
                        *(_QWORD *)(v79 + 8) = v81;
                      }
                      v29 = *(unsigned int *)(v11 - 4);
                    }
                    ++v72;
                    v71 += 24;
                    v73 = v29 & 0xFFFFFFF;
                  }
                  while ( ((unsigned int)v29 & 0xFFFFFFF) > (unsigned int)v72 );
                }
                v62 += 4;
                if ( v62 != v63 )
                {
                  while ( *v62 == -16 || *v62 == -8 )
                  {
                    v62 += 4;
                    if ( v63 == v62 )
                      goto LABEL_24;
                  }
                  if ( v62 != v63 )
                    continue;
                }
                break;
              }
            }
          }
        }
LABEL_24:
        v30 = (__int64 *)v118[1];
        v31 = *v118;
        if ( (__int64 *)*v118 != v30 )
        {
          src = (_QWORD *)*v118;
          _BitScanReverse64(&v32, 0xAAAAAAAAAAAAAAABLL * (((__int64)v30 - v31) >> 3));
          sub_1CF2270(v31, (__m128i *)v118[1], 2LL * (int)(63 - (v32 ^ 0x3F)), 0xAAAAAAAAAAAAAAABLL, v28, v29);
          if ( (__int64)v30 - v31 <= 384 )
          {
            sub_1CEFF80(src, v30);
          }
          else
          {
            v33 = src + 48;
            sub_1CEFF80(src, src + 48);
            if ( v30 != src + 48 )
            {
              do
              {
                v34 = v33;
                v33 += 3;
                sub_1CEFF30(v34);
              }
              while ( v30 != v33 );
            }
          }
        }
        if ( *(_BYTE *)(a1 + 32) )
          break;
LABEL_29:
        v11 = *(_QWORD *)(v11 + 8);
        if ( v110 == (_QWORD *)v11 )
          return j___libc_free_0(v128);
      }
      v35 = 3LL * (*(_DWORD *)(v11 - 4) & 0xFFFFFFF);
      v36 = (__int64 *)(v27 - v35 * 8);
      if ( (*(_BYTE *)(v11 - 1) & 0x40) != 0 )
      {
        v36 = *(__int64 **)(v11 - 32);
        v111 = &v36[v35];
      }
      if ( v36 == v111 )
      {
LABEL_44:
        v49 = sub_1648700(*(_QWORD *)(v11 - 16));
        v133 = 0;
        v50 = v49;
        v51 = (char **)sub_1CF27D0(a1, (__int64)v49, &v131);
        v121 = v50;
        v124 = v51;
        v56 = (__int64 *)v51;
        v125 = &v121;
        v126 = a1;
        if ( v51 == (char **)v118 )
        {
          sub_1CF0C90(&v124, a3, *(double *)a4.m128_u64, a5, a6, v54, v55, a9, a10);
        }
        else if ( !(unsigned __int8)sub_1CF0970(a1, v51, v118, v52, v53) )
        {
          sub_1CF0C90(&v124, a3, *(double *)a4.m128_u64, a5, a6, v57, v58, a9, a10);
          sub_1CF1300(a1, v56, v118);
        }
        goto LABEL_29;
      }
      v37 = a1;
      v38 = v36;
      v112 = v11;
      while ( 1 )
      {
        v39 = *v38;
        if ( *(_BYTE *)(*v38 + 16) != 78 )
          goto LABEL_35;
        v40 = *(_QWORD *)(v39 - 24);
        if ( *(_BYTE *)(v40 + 16) )
          goto LABEL_35;
        if ( (*(_BYTE *)(v40 + 33) & 0x20) == 0 )
          goto LABEL_35;
        srca = (void *)*v38;
        v41 = *(_QWORD *)(v39 + 24 * (1LL - (*(_DWORD *)(v39 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v41 + 16) <= 0x17u )
          goto LABEL_35;
        v133 = 0;
        v42 = sub_1CF27D0(v37, v41, &v131);
        v124 = (char **)v118;
        v46 = v42;
        v126 = v37;
        v121 = srca;
        v125 = &v121;
        if ( v42 == v118 )
          break;
        if ( (unsigned __int8)sub_1CF0970(v37, v118, v42, v43, (int)srca) )
        {
LABEL_35:
          v38 += 3;
          if ( v111 == v38 )
            goto LABEL_43;
        }
        else
        {
          v38 += 3;
          sub_1CF0C90(&v124, a3, *(double *)a4.m128_u64, a5, a6, v47, v48, a9, a10);
          sub_1CF1300(v37, v118, v46);
          if ( v111 == v38 )
          {
LABEL_43:
            a1 = v37;
            v11 = v112;
            goto LABEL_44;
          }
        }
      }
      sub_1CF0C90(&v124, a3, *(double *)a4.m128_u64, a5, a6, v44, v45, a9, a10);
      goto LABEL_35;
    }
  }
  return result;
}
