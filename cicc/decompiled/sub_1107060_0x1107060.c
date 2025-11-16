// Function: sub_1107060
// Address: 0x1107060
//
unsigned __int8 *__fastcall sub_1107060(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // rax
  unsigned __int8 *result; // rax
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rdi
  unsigned __int8 v14; // dl
  __int64 v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __m128i v18; // xmm1
  __int64 v19; // rax
  unsigned __int64 v20; // xmm2_8
  __m128i v21; // xmm3
  unsigned int v22; // eax
  __int64 v23; // r11
  __int64 v24; // rdx
  _BYTE *v25; // rsi
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // r10
  unsigned int **v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // r13
  __int64 v33; // r14
  __int64 v34; // rax
  unsigned int v35; // r14d
  unsigned int v36; // eax
  unsigned __int32 v37; // r15d
  unsigned int v38; // esi
  unsigned int v39; // edx
  __m128i v40; // xmm5
  __int64 v41; // rax
  unsigned __int64 v42; // xmm6_8
  __m128i v43; // xmm7
  char v44; // al
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // eax
  unsigned int v49; // edx
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // r14d
  unsigned int v54; // eax
  unsigned int v55; // ecx
  unsigned __int64 v56; // rax
  unsigned __int8 *v57; // rdi
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // r12
  __int64 v61; // r14
  __int64 v62; // rdx
  __int64 v63; // r13
  __int64 v64; // rdi
  unsigned __int64 v65; // rax
  __int64 v66; // r12
  __int64 v67; // r11
  __int64 v68; // r10
  __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // r10
  __int64 v72; // rbx
  unsigned int **v73; // rdi
  __int64 v74; // r12
  unsigned __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r10
  __int64 v78; // r11
  __int64 v79; // r12
  __int64 v80; // rax
  __int64 v81; // r13
  unsigned __int64 v82; // rax
  __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // r10
  __int64 v86; // rbx
  __int64 i; // r12
  __int64 v88; // rdx
  unsigned int v89; // esi
  _QWORD *v90; // rax
  __int64 v91; // r11
  __int64 v92; // r12
  __int64 j; // r14
  __int64 v94; // rdx
  unsigned int v95; // esi
  __int64 v96; // rbx
  __int64 v97; // r12
  __int64 v98; // rdx
  unsigned int v99; // esi
  __int64 v100; // rbx
  __int64 v101; // r12
  __int64 v102; // rdx
  unsigned int v103; // esi
  unsigned int v104; // [rsp+0h] [rbp-110h]
  char v105; // [rsp+0h] [rbp-110h]
  char v106; // [rsp+Ch] [rbp-104h]
  __int64 v107; // [rsp+10h] [rbp-100h]
  int v108; // [rsp+10h] [rbp-100h]
  __int64 v109; // [rsp+10h] [rbp-100h]
  __int64 v110; // [rsp+10h] [rbp-100h]
  __int64 v111; // [rsp+10h] [rbp-100h]
  __int64 v112; // [rsp+10h] [rbp-100h]
  __int64 v113; // [rsp+10h] [rbp-100h]
  unsigned int v114; // [rsp+18h] [rbp-F8h]
  __int64 v115; // [rsp+18h] [rbp-F8h]
  __int64 v116; // [rsp+18h] [rbp-F8h]
  unsigned int v117; // [rsp+18h] [rbp-F8h]
  __int64 v118; // [rsp+18h] [rbp-F8h]
  __int64 v119; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v120; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v121; // [rsp+18h] [rbp-F8h]
  __int64 v122; // [rsp+18h] [rbp-F8h]
  unsigned int v123; // [rsp+18h] [rbp-F8h]
  __int64 v124; // [rsp+18h] [rbp-F8h]
  __int64 v125; // [rsp+18h] [rbp-F8h]
  __int64 v126; // [rsp+18h] [rbp-F8h]
  __int64 v127; // [rsp+18h] [rbp-F8h]
  __int64 v128; // [rsp+18h] [rbp-F8h]
  __int64 v129; // [rsp+18h] [rbp-F8h]
  __int64 v130; // [rsp+18h] [rbp-F8h]
  __int64 v131; // [rsp+18h] [rbp-F8h]
  __int64 v132; // [rsp+18h] [rbp-F8h]
  int v133; // [rsp+2Ch] [rbp-E4h] BYREF
  __int64 v134; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v135; // [rsp+38h] [rbp-D8h]
  __int16 v136; // [rsp+50h] [rbp-C0h]
  __int64 v137; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v138; // [rsp+68h] [rbp-A8h]
  char *v139; // [rsp+70h] [rbp-A0h]
  __int16 v140; // [rsp+80h] [rbp-90h]
  __m128i v141; // [rsp+90h] [rbp-80h] BYREF
  __m128i v142; // [rsp+A0h] [rbp-70h]
  unsigned __int64 v143; // [rsp+B0h] [rbp-60h]
  __int64 v144; // [rsp+B8h] [rbp-58h]
  __m128i v145; // [rsp+C0h] [rbp-50h]
  __int64 v146; // [rsp+D0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 16);
  if ( v8 && !*(_QWORD *)(v8 + 8) && **(_BYTE **)(v8 + 24) == 67 && **(_BYTE **)(a2 - 32) > 0x15u )
    return 0;
  result = sub_11005E0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  if ( !result )
  {
    v10 = *(_QWORD *)(a2 - 32);
    v11 = *(_QWORD *)(a2 + 8);
    v12 = *(_QWORD *)(v10 + 8);
    v13 = v12;
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
      v13 = **(_QWORD **)(v12 + 16);
    if ( sub_BCAC40(v13, 1) && sub_B44910(a2) )
    {
      v45 = sub_AD6530(*(_QWORD *)(a2 + 8), 1);
      return sub_F162A0((__int64)a1, a2, v45);
    }
    if ( (unsigned __int8)sub_F0C890((__int64)a1, v12, v11)
      && (unsigned __int8)sub_10FF0F0((unsigned __int8 *)v10, v11, &v133, a1, a2) )
    {
      v116 = sub_1106750((__int64)a1, (unsigned __int8 *)v10, v11, 0);
      if ( *(_BYTE *)v10 > 0x1Cu )
      {
        v34 = *(_QWORD *)(v10 + 16);
        if ( v34 )
        {
          if ( !*(_QWORD *)(v34 + 8) )
            sub_F55740(v10, v116, a2, a1[5].m128i_i64[0]);
        }
      }
      v108 = sub_BCB060(v12);
      v106 = v133;
      v35 = v108 - v133;
      v36 = sub_BCB060(v11);
      LODWORD(v138) = v36;
      v37 = v36;
      if ( v36 > 0x40 )
      {
        sub_C43690((__int64)&v137, 0, 0);
        v39 = v138;
        v38 = v138 + v35 - v37;
      }
      else
      {
        v137 = 0;
        v38 = v35;
        v39 = v36;
      }
      if ( v39 != v38 )
      {
        if ( v38 > 0x3F || v39 > 0x40 )
          sub_C43C90(&v137, v38, v39);
        else
          v137 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v35 - (unsigned __int8)v37 + 64) << v38;
      }
      v40 = _mm_loadu_si128(a1 + 7);
      v41 = a1[10].m128i_i64[0];
      v42 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v141 = _mm_loadu_si128(a1 + 6);
      v43 = _mm_loadu_si128(a1 + 9);
      v146 = v41;
      v143 = v42;
      v142 = v40;
      v144 = a2;
      v145 = v43;
      v44 = sub_9AC230(v116, (__int64)&v137, &v141, 0);
      if ( (unsigned int)v138 > 0x40 && v137 )
      {
        v105 = v44;
        j_j___libc_free_0_0(v137);
        v44 = v105;
      }
      if ( v44 )
        return sub_F162A0((__int64)a1, a2, v116);
      v141.m128i_i32[2] = v37;
      if ( v37 > 0x40 )
        sub_C43690((__int64)&v141, 0, 0);
      else
        v141.m128i_i64[0] = 0;
      if ( v35 )
      {
        if ( v35 > 0x40 )
        {
          sub_C43C90(&v141, 0, v35);
        }
        else
        {
          v65 = 0xFFFFFFFFFFFFFFFFLL >> (v106 + 64 - v108);
          if ( v141.m128i_i32[2] > 0x40u )
            *(_QWORD *)v141.m128i_i64[0] |= v65;
          else
            v141.m128i_i64[0] |= v65;
        }
      }
      v66 = sub_AD8D80(*(_QWORD *)(v116 + 8), (__int64)&v141);
      if ( v141.m128i_i32[2] > 0x40u && v141.m128i_i64[0] )
        j_j___libc_free_0_0(v141.m128i_i64[0]);
      LOWORD(v143) = 257;
      return (unsigned __int8 *)sub_B504D0(28, v116, v66, (__int64)&v141, 0, 0);
    }
    v14 = *(_BYTE *)v10;
    if ( *(_BYTE *)v10 > 0x1Cu )
    {
      if ( v14 == 67 )
      {
        v107 = *(_QWORD *)(v10 - 32);
        v114 = sub_BCB060(*(_QWORD *)(v107 + 8));
        v104 = sub_BCB060(*(_QWORD *)(v10 + 8));
        v22 = sub_BCB060(v11);
        v23 = v107;
        v14 = 67;
        if ( v114 < v22 )
        {
          v135 = v114;
          if ( v114 > 0x40 )
          {
            sub_C43690((__int64)&v134, 0, 0);
            v23 = v107;
          }
          else
          {
            v134 = 0;
          }
          if ( v104 )
          {
            if ( v104 > 0x40 )
            {
              v132 = v23;
              sub_C43C90(&v134, 0, v104);
              v23 = v132;
            }
            else
            {
              v58 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v104);
              if ( v135 > 0x40 )
                *(_QWORD *)v134 |= v58;
              else
                v134 |= v58;
            }
          }
          v119 = v23;
          v59 = sub_AD8D80(*(_QWORD *)(v23 + 8), (__int64)&v134);
          v60 = a1[2].m128i_i64[0];
          v61 = v59;
          v137 = (__int64)sub_BD5D20(v10);
          v140 = 773;
          v138 = v62;
          v139 = ".mask";
          v63 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v60 + 80) + 16LL))(
                  *(_QWORD *)(v60 + 80),
                  28,
                  v119,
                  v61);
          if ( !v63 )
          {
            LOWORD(v143) = 257;
            v63 = sub_B504D0(28, v119, v61, (__int64)&v141, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v60 + 88) + 16LL))(
              *(_QWORD *)(v60 + 88),
              v63,
              &v137,
              *(_QWORD *)(v60 + 56),
              *(_QWORD *)(v60 + 64));
            v96 = *(_QWORD *)v60;
            v97 = *(_QWORD *)v60 + 16LL * *(unsigned int *)(v60 + 8);
            while ( v97 != v96 )
            {
              v98 = *(_QWORD *)(v96 + 8);
              v99 = *(_DWORD *)v96;
              v96 += 16;
              sub_B99FD0(v63, v99, v98);
            }
          }
          LOWORD(v143) = 257;
          result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
          if ( result )
          {
            v120 = result;
            sub_B515B0((__int64)result, v63, v11, (__int64)&v141, 0, 0);
            result = v120;
          }
          if ( v135 <= 0x40 )
            return result;
          v64 = v134;
          if ( !v134 )
            return result;
        }
        else
        {
          if ( v114 == v22 )
          {
            LODWORD(v138) = v114;
            if ( v114 > 0x40 )
            {
              sub_C43690((__int64)&v137, 0, 0);
              v23 = v107;
            }
            else
            {
              v137 = 0;
            }
            if ( v104 )
            {
              if ( v104 > 0x40 )
              {
                v113 = v23;
                sub_C43C90(&v137, 0, v104);
                v23 = v113;
              }
              else
              {
                v82 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v104);
                if ( (unsigned int)v138 > 0x40 )
                  *(_QWORD *)v137 |= v82;
                else
                  v137 |= v82;
              }
            }
            v125 = v23;
            LOWORD(v143) = 257;
            v83 = sub_AD8D80(*(_QWORD *)(v23 + 8), (__int64)&v137);
            result = (unsigned __int8 *)sub_B504D0(28, v125, v83, (__int64)&v141, 0, 0);
            if ( (unsigned int)v138 <= 0x40 )
              return result;
          }
          else
          {
            if ( v114 <= v22 )
            {
              v15 = *(_QWORD *)(v10 + 16);
              if ( !v15 )
                goto LABEL_16;
LABEL_29:
              if ( !*(_QWORD *)(v15 + 8) )
              {
                if ( v14 == 57 )
                {
                  v16 = *(_BYTE **)(v10 - 64);
                  if ( *v16 == 67 )
                  {
                    v77 = *((_QWORD *)v16 - 4);
                    if ( v77 )
                    {
                      v78 = *(_QWORD *)(v10 - 32);
                      if ( *(_BYTE *)v78 <= 0x15u && v11 == *(_QWORD *)(v77 + 8) )
                      {
                        v79 = a1[2].m128i_i64[0];
                        v140 = 257;
                        v136 = 257;
                        if ( v11 == *(_QWORD *)(v78 + 8) )
                        {
                          v81 = v78;
                        }
                        else
                        {
                          v110 = v77;
                          v124 = v78;
                          v80 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v79 + 80)
                                                                                             + 120LL))(
                                  *(_QWORD *)(v79 + 80),
                                  39,
                                  v78,
                                  v11);
                          v77 = v110;
                          v81 = v80;
                          if ( !v80 )
                          {
                            v111 = v124;
                            v126 = v77;
                            LOWORD(v143) = 257;
                            v84 = sub_BD2C40(72, unk_3F10A14);
                            v85 = v126;
                            v81 = (__int64)v84;
                            if ( v84 )
                            {
                              sub_B515B0((__int64)v84, v111, v11, (__int64)&v141, 0, 0);
                              v85 = v126;
                            }
                            v127 = v85;
                            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v79 + 88)
                                                                                               + 16LL))(
                              *(_QWORD *)(v79 + 88),
                              v81,
                              &v134,
                              *(_QWORD *)(v79 + 56),
                              *(_QWORD *)(v79 + 64));
                            v86 = *(_QWORD *)v79;
                            v77 = v127;
                            for ( i = *(_QWORD *)v79 + 16LL * *(unsigned int *)(v79 + 8); i != v86; v77 = v128 )
                            {
                              v88 = *(_QWORD *)(v86 + 8);
                              v89 = *(_DWORD *)v86;
                              v86 += 16;
                              v128 = v77;
                              sub_B99FD0(v81, v89, v88);
                            }
                          }
                        }
                        return (unsigned __int8 *)sub_B504D0(28, v77, v81, (__int64)&v137, 0, 0);
                      }
                    }
                  }
                  goto LABEL_15;
                }
                if ( v14 == 59 )
                {
                  v24 = *(_QWORD *)(v10 - 64);
                  if ( v24 )
                  {
                    v25 = *(_BYTE **)(v10 - 32);
                    if ( *v25 <= 0x15u )
                    {
                      v26 = *(_QWORD *)(v24 + 16);
                      if ( v26 )
                      {
                        if ( !*(_QWORD *)(v26 + 8) && *(_BYTE *)v24 == 57 )
                        {
                          v27 = *(_BYTE **)(v24 - 64);
                          if ( *v27 == 67 )
                          {
                            v28 = *((_QWORD *)v27 - 4);
                            if ( v28 )
                            {
                              if ( v25 == *(_BYTE **)(v24 - 32) && v11 == *(_QWORD *)(v28 + 8) )
                              {
                                v29 = (unsigned int **)a1[2].m128i_i64[0];
                                LOWORD(v143) = 257;
                                v115 = v28;
                                v30 = sub_A82F30(v29, (__int64)v25, v11, (__int64)&v141, 0);
                                v31 = a1[2].m128i_i64[0];
                                v136 = 257;
                                v32 = v30;
                                v140 = 257;
                                v33 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v31 + 80) + 16LL))(
                                        *(_QWORD *)(v31 + 80),
                                        28,
                                        v115,
                                        v30);
                                if ( !v33 )
                                {
                                  LOWORD(v143) = 257;
                                  v33 = sub_B504D0(28, v115, v32, (__int64)&v141, 0, 0);
                                  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v31 + 88) + 16LL))(
                                    *(_QWORD *)(v31 + 88),
                                    v33,
                                    &v134,
                                    *(_QWORD *)(v31 + 56),
                                    *(_QWORD *)(v31 + 64));
                                  v100 = *(_QWORD *)v31;
                                  v101 = *(_QWORD *)v31 + 16LL * *(unsigned int *)(v31 + 8);
                                  while ( v101 != v100 )
                                  {
                                    v102 = *(_QWORD *)(v100 + 8);
                                    v103 = *(_DWORD *)v100;
                                    v100 += 16;
                                    sub_B99FD0(v33, v103, v102);
                                  }
                                }
                                return (unsigned __int8 *)sub_B504D0(30, v33, v32, (__int64)&v137, 0, 0);
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
LABEL_16:
                if ( (unsigned __int8)sub_10FD760((unsigned __int8 *)v10)
                  && sub_B43CB0(a2)
                  && (v46 = sub_B43CB0(a2), (unsigned __int8)sub_B2D610(v46, 96))
                  && (v47 = sub_B43CB0(a2), v134 = sub_B2D7D0(v47, 96), v137 = sub_A71ED0(&v134), BYTE4(v137))
                  && (v117 = v137) != 0
                  && (v48 = sub_BCB060(*(_QWORD *)(v10 + 8)), _BitScanReverse(&v49, v117), v48 > 31 - (v49 ^ 0x1F)) )
                {
                  v50 = a1[2].m128i_i64[0];
                  LOWORD(v143) = 257;
                  v51 = sub_AD64C0(v11, 1, 0);
                  v52 = sub_B33D80(v50, v51, (__int64)&v141);
                  return sub_F162A0((__int64)a1, a2, v52);
                }
                else
                {
                  if ( sub_B44910(a2) )
                    return 0;
                  v17 = *(_QWORD *)(a2 + 16);
                  if ( v17 && !*(_QWORD *)(v17 + 8) )
                  {
                    v118 = *(_QWORD *)(a2 + 16);
                    v53 = sub_BCB060(v12);
                    v54 = sub_BCB060(v11);
                    v55 = 0;
                    v56 = v54 - 1LL;
                    if ( v56 )
                    {
                      _BitScanReverse64(&v56, v56);
                      v55 = 64 - (v56 ^ 0x3F);
                    }
                    if ( v55 < v53 )
                    {
                      v57 = *(unsigned __int8 **)(v118 + 24);
                      if ( v57 )
                      {
                        if ( (unsigned int)*v57 - 54 <= 2 && a2 == *(_QWORD *)(sub_986520((__int64)v57) + 32) )
                          goto LABEL_72;
                      }
                    }
                  }
                  v18 = _mm_loadu_si128(a1 + 7);
                  v19 = a1[10].m128i_i64[0];
                  v20 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
                  v141 = _mm_loadu_si128(a1 + 6);
                  v21 = _mm_loadu_si128(a1 + 9);
                  v146 = v19;
                  v143 = v20;
                  v142 = v18;
                  v144 = a2;
                  v145 = v21;
                  if ( (unsigned __int8)sub_9AC470(v10, &v141, 0) )
                  {
LABEL_72:
                    sub_B448D0(a2, 1);
                    return (unsigned __int8 *)a2;
                  }
                  else
                  {
                    return 0;
                  }
                }
              }
LABEL_13:
              if ( v14 == 57 )
              {
                v16 = *(_BYTE **)(v10 - 64);
LABEL_15:
                if ( *v16 == 67 )
                {
                  v67 = *((_QWORD *)v16 - 4);
                  if ( v67 )
                  {
                    v68 = *(_QWORD *)(v10 - 32);
                    if ( *(_BYTE *)v68 <= 0x15u && v11 == *(_QWORD *)(v67 + 8) )
                    {
                      v69 = a1[2].m128i_i64[0];
                      v140 = 257;
                      if ( v11 == *(_QWORD *)(v68 + 8) )
                      {
                        v72 = v68;
                      }
                      else
                      {
                        v109 = v67;
                        v122 = v68;
                        v70 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v69 + 80)
                                                                                           + 120LL))(
                                *(_QWORD *)(v69 + 80),
                                39,
                                v68,
                                v11);
                        v71 = v122;
                        v67 = v109;
                        v72 = v70;
                        if ( !v70 )
                        {
                          v129 = v109;
                          v112 = v71;
                          LOWORD(v143) = 257;
                          v90 = sub_BD2C40(72, unk_3F10A14);
                          v91 = v129;
                          v72 = (__int64)v90;
                          if ( v90 )
                          {
                            sub_B515B0((__int64)v90, v112, v11, (__int64)&v141, 0, 0);
                            v91 = v129;
                          }
                          v130 = v91;
                          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v69 + 88)
                                                                                             + 16LL))(
                            *(_QWORD *)(v69 + 88),
                            v72,
                            &v137,
                            *(_QWORD *)(v69 + 56),
                            *(_QWORD *)(v69 + 64));
                          v92 = *(_QWORD *)v69;
                          v67 = v130;
                          for ( j = *(_QWORD *)v69 + 16LL * *(unsigned int *)(v69 + 8); j != v92; v67 = v131 )
                          {
                            v94 = *(_QWORD *)(v92 + 8);
                            v95 = *(_DWORD *)v92;
                            v92 += 16;
                            v131 = v67;
                            sub_B99FD0(v72, v95, v94);
                          }
                        }
                      }
                      LOWORD(v143) = 257;
                      return (unsigned __int8 *)sub_B504D0(28, v67, v72, (__int64)&v141, 0, 0);
                    }
                  }
                }
                goto LABEL_16;
              }
              goto LABEL_16;
            }
            v73 = (unsigned int **)a1[2].m128i_i64[0];
            v123 = v22;
            LOWORD(v143) = 257;
            v74 = sub_A82DA0(v73, v107, v11, (__int64)&v141, 0, 0);
            LODWORD(v138) = v123;
            if ( v123 > 0x40 )
              sub_C43690((__int64)&v137, 0, 0);
            else
              v137 = 0;
            if ( v104 )
            {
              if ( v104 > 0x40 )
              {
                sub_C43C90(&v137, 0, v104);
              }
              else
              {
                v75 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v104);
                if ( (unsigned int)v138 > 0x40 )
                  *(_QWORD *)v137 |= v75;
                else
                  v137 |= v75;
              }
            }
            LOWORD(v143) = 257;
            v76 = sub_AD8D80(*(_QWORD *)(v74 + 8), (__int64)&v137);
            result = (unsigned __int8 *)sub_B504D0(28, v74, v76, (__int64)&v141, 0, 0);
            if ( (unsigned int)v138 <= 0x40 )
              return result;
          }
          v64 = v137;
          if ( !v137 )
            return result;
        }
        v121 = result;
        j_j___libc_free_0_0(v64);
        return v121;
      }
      if ( v14 == 82 )
        return sub_11052A0(a1, v10, a2);
    }
    v15 = *(_QWORD *)(v10 + 16);
    if ( !v15 )
      goto LABEL_13;
    goto LABEL_29;
  }
  return result;
}
