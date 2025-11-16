// Function: sub_204A2F0
// Address: 0x204a2f0
//
void __fastcall sub_204A2F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned int a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned __int8 a10,
        __int64 a11,
        __int64 a12,
        unsigned int a13)
{
  __int64 v13; // rax
  unsigned __int64 v16; // r13
  __int64 v18; // rbx
  __int64 v19; // rdx
  char v20; // si
  const void **v21; // rcx
  bool v22; // al
  int v23; // eax
  unsigned int v24; // eax
  unsigned int v25; // r10d
  __int64 v26; // rdx
  unsigned int v27; // r8d
  bool v28; // al
  char v29; // al
  const void **v30; // r8
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // eax
  char v34; // di
  const void **v35; // rax
  unsigned int v36; // esi
  unsigned int v37; // eax
  const void **v38; // r8
  unsigned int v39; // edi
  unsigned int v40; // r13d
  int v41; // edx
  unsigned int v42; // r11d
  unsigned int v43; // r10d
  unsigned int v44; // r12d
  char v45; // r10
  unsigned __int64 v46; // rbx
  __int128 v47; // rax
  int v48; // edx
  __int128 v49; // rax
  int v50; // edx
  unsigned int v51; // eax
  const void **v52; // rdx
  unsigned int v53; // eax
  int v54; // edx
  unsigned int v55; // eax
  __int64 v56; // rax
  int v57; // edx
  const __m128i *v58; // rdx
  unsigned __int64 v59; // rax
  __int64 v60; // rcx
  __m128i v61; // xmm0
  const void **v62; // rdx
  unsigned int v63; // ecx
  __int64 v64; // r12
  __int128 v65; // rax
  unsigned int v66; // eax
  int v67; // edx
  const __m128i *v68; // rbx
  char v69; // al
  const void **v70; // rdx
  const void **v71; // r8
  __int64 v72; // rax
  unsigned int v73; // edx
  bool v74; // al
  char v75; // al
  const void **v76; // r8
  unsigned int v77; // edx
  __int64 v78; // rdi
  __int64 v79; // rax
  unsigned int v80; // edx
  __int64 v81; // rcx
  __int64 v82; // rdx
  __int64 v83; // rsi
  __int64 v84; // rax
  unsigned int v85; // edx
  __m128i *v86; // rax
  const void **v87; // rdx
  const void **v88; // rdx
  char v89; // al
  const void **v90; // r8
  __int64 v91; // rax
  __int64 v92; // r12
  const void **v93; // rdx
  __int128 v94; // [rsp-10h] [rbp-140h]
  __int128 v95; // [rsp-10h] [rbp-140h]
  __int128 v96; // [rsp-10h] [rbp-140h]
  __int128 v97; // [rsp-10h] [rbp-140h]
  __int128 v98; // [rsp-10h] [rbp-140h]
  __int128 v99; // [rsp-10h] [rbp-140h]
  __int128 v100; // [rsp-10h] [rbp-140h]
  unsigned int v101; // [rsp+8h] [rbp-128h]
  unsigned int v102; // [rsp+10h] [rbp-120h]
  unsigned int v103; // [rsp+18h] [rbp-118h]
  unsigned int v105; // [rsp+24h] [rbp-10Ch]
  unsigned int v106; // [rsp+28h] [rbp-108h]
  unsigned int v107; // [rsp+28h] [rbp-108h]
  __int64 v109; // [rsp+38h] [rbp-F8h]
  __int64 v110; // [rsp+38h] [rbp-F8h]
  unsigned int v111; // [rsp+38h] [rbp-F8h]
  int v112; // [rsp+40h] [rbp-F0h]
  char v113; // [rsp+47h] [rbp-E9h]
  const void **v114; // [rsp+48h] [rbp-E8h]
  __int64 v115; // [rsp+48h] [rbp-E8h]
  unsigned int v116; // [rsp+48h] [rbp-E8h]
  unsigned int v117; // [rsp+48h] [rbp-E8h]
  unsigned int v118; // [rsp+48h] [rbp-E8h]
  __int64 v119; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v120; // [rsp+50h] [rbp-E0h]
  unsigned int v121; // [rsp+50h] [rbp-E0h]
  unsigned int v122; // [rsp+50h] [rbp-E0h]
  unsigned int v123; // [rsp+58h] [rbp-D8h]
  unsigned int v124; // [rsp+58h] [rbp-D8h]
  unsigned int v125; // [rsp+58h] [rbp-D8h]
  const void **v126; // [rsp+58h] [rbp-D8h]
  unsigned int v127; // [rsp+58h] [rbp-D8h]
  unsigned int v128; // [rsp+58h] [rbp-D8h]
  unsigned int v129; // [rsp+D0h] [rbp-60h] BYREF
  const void **v130; // [rsp+D8h] [rbp-58h]
  _QWORD v131[2]; // [rsp+E0h] [rbp-50h] BYREF
  char v132; // [rsp+F0h] [rbp-40h]
  char v133; // [rsp+F1h] [rbp-3Fh]

  v13 = (unsigned int)a4;
  v16 = a4;
  v18 = a3;
  v19 = *(_QWORD *)(a3 + 40) + 16LL * (unsigned int)a4;
  v20 = *(_BYTE *)v19;
  v21 = *(const void ***)(v19 + 8);
  v123 = v13;
  v119 = v13;
  LOBYTE(v129) = v20;
  v130 = v21;
  if ( v20 )
  {
    if ( (unsigned __int8)(v20 - 14) > 0x5Fu )
      goto LABEL_3;
  }
  else
  {
    v109 = v19;
    v114 = v21;
    v22 = sub_1F58D20((__int64)&v129);
    v21 = v114;
    v20 = 0;
    v19 = v109;
    if ( !v22 )
    {
LABEL_3:
      if ( !a6 )
        return;
      if ( v20 == a10 && (v20 || !v21) )
        goto LABEL_7;
      v115 = v19;
      v23 = sub_2045180(a10);
      v112 = v23;
      if ( v20 )
      {
        v33 = sub_2045180(v20);
        v26 = v115;
        v27 = v33;
        if ( v25 > v33 )
          goto LABEL_15;
      }
      else
      {
        v110 = v115;
        v116 = v23 * a6;
        v24 = sub_1F58D40((__int64)&v129);
        v25 = v116;
        v26 = v110;
        v27 = v24;
        v20 = 0;
        if ( v116 > v24 )
        {
LABEL_15:
          if ( (unsigned __int8)(a10 - 8) > 5u && (unsigned __int8)(a10 - 86) > 0x17u )
          {
            if ( v20 )
            {
              v28 = (unsigned __int8)(v20 - 8) <= 5u || (unsigned __int8)(v20 - 86) <= 0x17u;
            }
            else
            {
              v117 = v27;
              v124 = v25;
              v28 = sub_1F58CD0((__int64)&v129);
              v25 = v124;
              v27 = v117;
            }
            if ( v28 )
            {
              if ( v27 == 32 )
              {
                v29 = 5;
              }
              else if ( v27 > 0x20 )
              {
                if ( v27 == 64 )
                {
                  v29 = 6;
                }
                else
                {
                  if ( v27 != 128 )
                  {
LABEL_118:
                    v128 = v25;
                    v29 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v27);
                    v25 = v128;
                    v30 = v88;
                    goto LABEL_25;
                  }
                  v29 = 7;
                }
              }
              else if ( v27 == 8 )
              {
                v29 = 3;
              }
              else
              {
                v29 = 4;
                if ( v27 != 16 )
                {
                  v29 = 2;
                  if ( v27 != 1 )
                    goto LABEL_118;
                }
              }
              v30 = 0;
LABEL_25:
              *((_QWORD *)&v94 + 1) = v16;
              *(_QWORD *)&v94 = a3;
              LOBYTE(v129) = v29;
              v125 = v25;
              v130 = v30;
              v31 = sub_1D309E0(
                      (__int64 *)a1,
                      158,
                      a2,
                      v129,
                      v30,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      *(double *)a9.m128i_i64,
                      v94);
              v25 = v125;
              v18 = v31;
              v119 = v32;
            }
LABEL_87:
            if ( v25 == 32 )
            {
              v75 = 5;
            }
            else if ( v25 > 0x20 )
            {
              if ( v25 == 64 )
              {
                v75 = 6;
              }
              else
              {
                if ( v25 != 128 )
                {
LABEL_115:
                  v75 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v25);
                  v76 = v87;
                  goto LABEL_92;
                }
                v75 = 7;
              }
            }
            else if ( v25 == 8 )
            {
              v75 = 3;
            }
            else
            {
              v75 = 4;
              if ( v25 != 16 )
              {
                v75 = 2;
                if ( v25 != 1 )
                  goto LABEL_115;
              }
            }
            v76 = 0;
LABEL_92:
            LOBYTE(v129) = v75;
            v130 = v76;
            v16 = v16 & 0xFFFFFFFF00000000LL | v119;
            *((_QWORD *)&v97 + 1) = v16;
            *(_QWORD *)&v97 = v18;
            v18 = sub_1D309E0(
                    (__int64 *)a1,
                    a13,
                    a2,
                    v129,
                    v76,
                    0,
                    *(double *)a7.m128i_i64,
                    a8,
                    *(double *)a9.m128i_i64,
                    v97);
            v123 = v77;
            if ( a10 != 110 )
            {
LABEL_93:
              v26 = *(_QWORD *)(v18 + 40) + 16LL * v77;
              goto LABEL_29;
            }
            v92 = v18;
            v16 = v77 | v16 & 0xFFFFFFFF00000000LL;
            goto LABEL_129;
          }
          if ( v20 )
          {
            if ( (unsigned __int8)(v20 - 8) > 5u && (unsigned __int8)(v20 - 86) > 0x17u )
              goto LABEL_87;
          }
          else
          {
            v127 = v25;
            v74 = sub_1F58CD0((__int64)&v129);
            v25 = v127;
            if ( !v74 )
              goto LABEL_87;
          }
          *((_QWORD *)&v99 + 1) = v16;
          v81 = a10;
          *(_QWORD *)&v99 = a3;
          v82 = a2;
          v83 = 157;
LABEL_102:
          v84 = sub_1D309E0(
                  (__int64 *)a1,
                  v83,
                  v82,
                  v81,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  a8,
                  *(double *)a9.m128i_i64,
                  v99);
          v123 = v85;
          v18 = v84;
          v26 = *(_QWORD *)(v84 + 40) + 16LL * v85;
LABEL_29:
          v34 = *(_BYTE *)v26;
          v35 = *(const void ***)(v26 + 8);
          LOBYTE(v129) = *(_BYTE *)v26;
          v130 = v35;
          if ( a6 == 1 )
          {
            if ( v34 != a10 || !a10 && v35 )
            {
              v78 = *(_QWORD *)(a1 + 48);
              v133 = 1;
              v131[0] = "scalar-to-vector conversion failed";
              v132 = 3;
              sub_2046E60(v78, a11, (__int64)v131);
              *((_QWORD *)&v98 + 1) = v123 | v16 & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v98 = v18;
              v79 = sub_1D309E0(
                      (__int64 *)a1,
                      158,
                      a2,
                      a10,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      *(double *)a9.m128i_i64,
                      v98);
              v123 = v80;
              v18 = v79;
            }
LABEL_7:
            *(_QWORD *)a5 = v18;
            *(_DWORD *)(a5 + 8) = v123;
            return;
          }
          v105 = a6;
          if ( ((a6 - 1) & a6) == 0 )
          {
LABEL_31:
            if ( v34 )
              v36 = sub_2045180(v34);
            else
              v36 = sub_1F58D40((__int64)&v129);
            if ( v36 == 32 )
            {
              LOBYTE(v37) = 5;
            }
            else if ( v36 > 0x20 )
            {
              if ( v36 == 64 )
              {
                LOBYTE(v37) = 6;
              }
              else
              {
                if ( v36 != 128 )
                {
LABEL_65:
                  v37 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v36);
                  v106 = v37;
                  v38 = v62;
                  goto LABEL_38;
                }
                LOBYTE(v37) = 7;
              }
            }
            else if ( v36 == 8 )
            {
              LOBYTE(v37) = 3;
            }
            else
            {
              LOBYTE(v37) = 4;
              if ( v36 != 16 )
              {
                LOBYTE(v37) = 2;
                if ( v36 != 1 )
                  goto LABEL_65;
              }
            }
            v38 = 0;
LABEL_38:
            v39 = v106;
            LOBYTE(v39) = v37;
            *((_QWORD *)&v95 + 1) = v123 | v16 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v95 = v18;
            v40 = v101;
            *(_QWORD *)a5 = sub_1D309E0(
                              (__int64 *)a1,
                              158,
                              a2,
                              v39,
                              v38,
                              0,
                              *(double *)a7.m128i_i64,
                              a8,
                              *(double *)a9.m128i_i64,
                              v95);
            *(_DWORD *)(a5 + 8) = v41;
            v111 = v105;
            while ( 1 )
            {
              v42 = 0;
              v43 = v111 * v112;
              v107 = v111;
              v111 >>= 1;
              v44 = v43 >> 1;
              do
              {
                while ( 1 )
                {
                  if ( v44 == 32 )
                  {
                    v45 = 5;
                  }
                  else if ( v44 <= 0x20 )
                  {
                    if ( v44 == 8 )
                    {
                      v45 = 3;
                    }
                    else
                    {
                      v45 = 4;
                      if ( v44 != 16 )
                      {
                        v45 = 2;
                        if ( v44 != 1 )
                        {
LABEL_51:
                          v121 = v42;
                          v51 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v44);
                          v42 = v121;
                          v126 = v52;
                          v40 = v51;
                          v45 = v51;
                          goto LABEL_43;
                        }
                      }
                    }
                  }
                  else if ( v44 == 64 )
                  {
                    v45 = 6;
                  }
                  else
                  {
                    if ( v44 != 128 )
                      goto LABEL_51;
                    v45 = 7;
                  }
                  v126 = 0;
LABEL_43:
                  LOBYTE(v40) = v45;
                  v113 = v45;
                  v46 = a5 + 16LL * v42;
                  v118 = v42;
                  v120 = a5 + 16LL * (v42 + v111);
                  *(_QWORD *)&v47 = sub_1D38E70(a1, 1, a2, 0, a7, a8, a9);
                  *(_QWORD *)v120 = sub_1D332F0(
                                      (__int64 *)a1,
                                      49,
                                      a2,
                                      v40,
                                      v126,
                                      0,
                                      *(double *)a7.m128i_i64,
                                      a8,
                                      a9,
                                      *(_QWORD *)v46,
                                      *(_QWORD *)(v46 + 8),
                                      v47);
                  *(_DWORD *)(v120 + 8) = v48;
                  *(_QWORD *)&v49 = sub_1D38E70(a1, 0, a2, 0, a7, a8, a9);
                  *(_QWORD *)v46 = sub_1D332F0(
                                     (__int64 *)a1,
                                     49,
                                     a2,
                                     v40,
                                     v126,
                                     0,
                                     *(double *)a7.m128i_i64,
                                     a8,
                                     a9,
                                     *(_QWORD *)v46,
                                     *(_QWORD *)(v46 + 8),
                                     v49);
                  *(_DWORD *)(v46 + 8) = v50;
                  if ( v112 == v44 && (a10 != v113 || !a10 && v126) )
                    break;
                  v42 = v107 + v118;
                  if ( v105 <= v107 + v118 )
                    goto LABEL_57;
                }
                v53 = v102;
                LOBYTE(v53) = a10;
                *(_QWORD *)v46 = sub_1D309E0(
                                   (__int64 *)a1,
                                   158,
                                   a2,
                                   v53,
                                   0,
                                   0,
                                   *(double *)a7.m128i_i64,
                                   a8,
                                   *(double *)a9.m128i_i64,
                                   *(_OWORD *)v46);
                *(_DWORD *)(v46 + 8) = v54;
                v55 = v103;
                LOBYTE(v55) = a10;
                v56 = sub_1D309E0(
                        (__int64 *)a1,
                        158,
                        a2,
                        v55,
                        0,
                        0,
                        *(double *)a7.m128i_i64,
                        a8,
                        *(double *)a9.m128i_i64,
                        *(_OWORD *)v120);
                v42 = v107 + v118;
                *(_QWORD *)v120 = v56;
                *(_DWORD *)(v120 + 8) = v57;
              }
              while ( v105 > v107 + v118 );
LABEL_57:
              if ( v111 == 1 )
              {
                if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1 + 32)) )
                {
                  v58 = (const __m128i *)a5;
                  v59 = a5 + 16LL * a6 - 16;
                  if ( a5 < v59 )
                  {
                    do
                    {
                      v60 = *(_QWORD *)v59;
                      v61 = _mm_loadu_si128(v58);
                      v59 -= 16LL;
                      ++v58;
                      v58[-1].m128i_i64[0] = v60;
                      v58[-1].m128i_i32[2] = *(_DWORD *)(v59 + 24);
                      *(_QWORD *)(v59 + 16) = v61.m128i_i64[0];
                      *(_DWORD *)(v59 + 24) = v61.m128i_i32[2];
                    }
                    while ( (unsigned __int64)v58 < v59 );
                  }
                }
                return;
              }
            }
          }
          _BitScanReverse(&v63, a6);
          v64 = v18;
          v105 = 0x80000000 >> (v63 ^ 0x1F);
          v122 = v105 * v112;
          *(_QWORD *)&v65 = sub_1D38E70(a1, v105 * v112, a2, 0, a7, a8, a9);
          v16 = v123 | v16 & 0xFFFFFFFF00000000LL;
          v66 = (unsigned int)sub_1D332F0(
                                (__int64 *)a1,
                                124,
                                a2,
                                v129,
                                v130,
                                0,
                                *(double *)a7.m128i_i64,
                                a8,
                                a9,
                                v18,
                                v16,
                                v65);
          BYTE4(v131[0]) = *(_BYTE *)(a12 + 4);
          if ( BYTE4(v131[0]) )
            LODWORD(v131[0]) = *(_DWORD *)a12;
          v68 = (const __m128i *)(a5 + 16LL * v105);
          sub_204A2F0(a1, a2, v66, v67, a5 + 16 * v105, a6 - v105, a10, a11, (__int64)v131, 144);
          if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1 + 32)) )
          {
            v86 = (__m128i *)(a5 + 16LL * a6);
            if ( v68 != v86 )
            {
              while ( v68 < --v86 )
              {
                a7 = _mm_loadu_si128(v68++);
                v68[-1].m128i_i64[0] = v86->m128i_i64[0];
                v68[-1].m128i_i32[2] = v86->m128i_i32[2];
                v86->m128i_i64[0] = a7.m128i_i64[0];
                v86->m128i_i32[2] = a7.m128i_i32[2];
              }
            }
          }
          if ( v122 == 32 )
          {
            v69 = 5;
          }
          else if ( v122 > 0x20 )
          {
            if ( v122 == 64 )
            {
              v69 = 6;
            }
            else
            {
              if ( v122 != 128 )
              {
LABEL_79:
                v69 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v122);
                v71 = v70;
LABEL_80:
                *((_QWORD *)&v96 + 1) = v16;
                *(_QWORD *)&v96 = v64;
                LOBYTE(v129) = v69;
                v130 = v71;
                v72 = sub_1D309E0(
                        (__int64 *)a1,
                        145,
                        a2,
                        v129,
                        v71,
                        0,
                        *(double *)a7.m128i_i64,
                        a8,
                        *(double *)a9.m128i_i64,
                        v96);
                v34 = v129;
                v123 = v73;
                v18 = v72;
                goto LABEL_31;
              }
              v69 = 7;
            }
          }
          else if ( v122 == 8 )
          {
            v69 = 3;
          }
          else
          {
            v69 = 4;
            if ( v122 != 16 )
            {
              v69 = 2;
              if ( v122 != 1 )
                goto LABEL_79;
            }
          }
          v71 = 0;
          goto LABEL_80;
        }
      }
      if ( v112 == v27 )
      {
        *((_QWORD *)&v99 + 1) = v16;
        v81 = a10;
        *(_QWORD *)&v99 = a3;
        goto LABEL_130;
      }
      if ( v25 >= v27 )
        goto LABEL_29;
      if ( v25 == 32 )
      {
        v89 = 5;
      }
      else if ( v25 > 0x20 )
      {
        if ( v25 == 64 )
        {
          v89 = 6;
        }
        else
        {
          if ( v25 != 128 )
          {
LABEL_135:
            v89 = sub_1F58CC0(*(_QWORD **)(a1 + 48), v25);
            v90 = v93;
            goto LABEL_127;
          }
          v89 = 7;
        }
      }
      else if ( v25 == 8 )
      {
        v89 = 3;
      }
      else
      {
        v89 = 4;
        if ( v25 != 16 )
        {
          v89 = 2;
          if ( v25 != 1 )
            goto LABEL_135;
        }
      }
      v90 = 0;
LABEL_127:
      *((_QWORD *)&v100 + 1) = v16;
      *(_QWORD *)&v100 = a3;
      LOBYTE(v129) = v89;
      v130 = v90;
      v91 = sub_1D309E0(
              (__int64 *)a1,
              145,
              a2,
              v129,
              v90,
              0,
              *(double *)a7.m128i_i64,
              a8,
              *(double *)a9.m128i_i64,
              v100);
      v123 = v77;
      v18 = v91;
      if ( a10 != 110 )
        goto LABEL_93;
      v92 = v91;
      v16 = v77 | v16 & 0xFFFFFFFF00000000LL;
LABEL_129:
      *((_QWORD *)&v99 + 1) = v16;
      v81 = 110;
      *(_QWORD *)&v99 = v92;
LABEL_130:
      v82 = a2;
      v83 = 158;
      goto LABEL_102;
    }
  }
  BYTE4(v131[0]) = *(_BYTE *)(a12 + 4);
  if ( BYTE4(v131[0]) )
    LODWORD(v131[0]) = *(_DWORD *)a12;
  sub_2048D40(a1, a2, a3, v16, a5, a6, a7, a8, a9, a10, a11, (unsigned int *)v131);
}
