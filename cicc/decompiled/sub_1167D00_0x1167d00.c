// Function: sub_1167D00
// Address: 0x1167d00
//
unsigned __int8 *__fastcall sub_1167D00(__m128i *a1, unsigned __int8 *a2)
{
  __int64 v3; // r12
  __m128i v4; // xmm1
  unsigned __int64 v5; // xmm2_8
  __m128i v6; // xmm3
  __int64 v7; // rax
  char v8; // al
  __int64 v9; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int8 *v16; // rbx
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned int v20; // r8d
  __int64 v21; // rdx
  unsigned __int8 v22; // al
  unsigned __int8 v23; // cl
  unsigned int **v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r12
  _QWORD *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdi
  bool v30; // al
  char v31; // cl
  __int64 v32; // rdx
  unsigned int v33; // edi
  __int64 v34; // rdx
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rcx
  unsigned __int8 v39; // al
  unsigned __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned __int64 v42; // rcx
  unsigned int v43; // r8d
  __int64 *v44; // r9
  __int64 v45; // rax
  bool v46; // zf
  __int64 v47; // r12
  unsigned int **v48; // r13
  _BYTE *v49; // rax
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // r13
  __int64 v53; // rax
  int v54; // eax
  int v55; // eax
  __int64 v56; // rdx
  __int64 v57; // rax
  unsigned __int8 *v58; // rax
  __int64 v59; // r13
  __int64 v60; // rbx
  const char *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdi
  __int64 v64; // r12
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 *v68; // r13
  bool v69; // bl
  const char *v70; // rax
  __int64 v71; // rdx
  __m128i v72; // xmm4
  __m128i v73; // xmm5
  unsigned __int64 v74; // xmm6_8
  __m128i v75; // xmm7
  __int64 v76; // rdx
  __m128i *v77; // rsi
  __m128i *v78; // rdi
  __int64 j; // rcx
  __int64 v80; // r8
  __int64 v81; // rax
  unsigned __int8 *v82; // rax
  __int64 *v83; // r15
  __int64 v84; // r13
  bool v85; // bl
  __m128i v86; // rax
  __int64 v87; // rax
  unsigned int i; // r12d
  __int64 *v89; // rax
  unsigned int v90; // r8d
  __int64 v91; // rdx
  __int64 *v92; // rax
  bool v93; // dl
  __int64 v94; // rax
  unsigned int v95; // eax
  unsigned int v96; // edx
  __int64 v97; // rax
  int v98; // eax
  unsigned __int64 v99; // rsi
  __int64 v100; // rax
  __int64 *v101; // rdi
  __int64 v102; // rax
  __int64 v103; // r12
  _QWORD *v104; // rax
  unsigned int v105; // eax
  unsigned __int64 v106; // rcx
  __int32 v107; // eax
  __int64 v108; // r13
  bool v109; // al
  __m128i *v110; // rsi
  __int64 v111; // rcx
  __m128i *v112; // rdi
  unsigned int v113; // eax
  unsigned int **v114; // r13
  _BYTE *v115; // rax
  __int64 v116; // rax
  __int64 v117; // r13
  __int64 v118; // r15
  unsigned int v119; // edx
  unsigned __int64 v122; // rcx
  int v123; // eax
  __m128i v124; // rax
  bool v125; // al
  bool v126; // r13
  unsigned int v127; // r12d
  __int64 v128; // rax
  __int64 v129; // [rsp-8h] [rbp-128h]
  unsigned int v130; // [rsp+4h] [rbp-11Ch]
  const void **v131; // [rsp+8h] [rbp-118h]
  unsigned __int8 v132; // [rsp+10h] [rbp-110h]
  __int64 v133; // [rsp+10h] [rbp-110h]
  bool v134; // [rsp+10h] [rbp-110h]
  int v135; // [rsp+18h] [rbp-108h]
  bool v136; // [rsp+18h] [rbp-108h]
  __int64 **v137; // [rsp+18h] [rbp-108h]
  int v138; // [rsp+18h] [rbp-108h]
  unsigned __int8 v139; // [rsp+20h] [rbp-100h]
  __int64 v140; // [rsp+20h] [rbp-100h]
  bool v141; // [rsp+20h] [rbp-100h]
  __int64 v142; // [rsp+20h] [rbp-100h]
  char v143; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v144; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v145; // [rsp+28h] [rbp-F8h]
  __int64 v146; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v147; // [rsp+28h] [rbp-F8h]
  __int64 v148; // [rsp+28h] [rbp-F8h]
  unsigned int v149; // [rsp+28h] [rbp-F8h]
  __int64 v150; // [rsp+30h] [rbp-F0h]
  __int64 v151; // [rsp+38h] [rbp-E8h]
  __int64 v152; // [rsp+48h] [rbp-D8h] BYREF
  const void **v153; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v154; // [rsp+58h] [rbp-C8h] BYREF
  _QWORD *v155[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v156; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v157; // [rsp+78h] [rbp-A8h]
  char *v158; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v159; // [rsp+90h] [rbp-90h]
  __m128i v160; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v161; // [rsp+B0h] [rbp-70h]
  unsigned __int64 v162; // [rsp+C0h] [rbp-60h]
  __int64 *v163; // [rsp+C8h] [rbp-58h]
  __m128i v164; // [rsp+D0h] [rbp-50h]
  __int64 v165; // [rsp+E0h] [rbp-40h]

  v3 = (__int64)a2;
  v4 = _mm_loadu_si128(a1 + 7);
  v5 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v6 = _mm_loadu_si128(a1 + 9);
  v7 = a1[10].m128i_i64[0];
  v160 = _mm_loadu_si128(a1 + 6);
  v162 = v5;
  v165 = v7;
  v163 = (__int64 *)a2;
  v161 = v4;
  v164 = v6;
  v8 = sub_B44E60((__int64)a2);
  v9 = sub_101A9B0(*((_QWORD *)a2 - 8), *((unsigned __int8 **)a2 - 4), v8, &v160);
  if ( v9 )
    return sub_F162A0((__int64)a1, (__int64)a2, v9);
  v11 = (__int64)sub_F0F270((__int64)a1, a2);
  if ( !v11 )
  {
    v11 = (__int64)sub_1166190(a1, (__int64)a2);
    if ( !v11 )
    {
      v11 = (__int64)sub_1160A10((__int64)a1, a2, v12, v13, v14);
      if ( !v11 )
      {
        v15 = *((_QWORD *)a2 - 8);
        v16 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
        v160.m128i_i64[0] = 0;
        v151 = v15;
        v17 = (__int64)v16;
        v150 = *(_QWORD *)(v3 + 8);
        v18 = sub_995B10(&v160, (__int64)v16);
        v21 = v18;
        if ( (_BYTE)v18 )
          goto LABEL_18;
        if ( *v16 == 69 )
        {
          v28 = *((_QWORD *)v16 - 4);
          if ( v28 )
          {
            v29 = *(_QWORD *)(v28 + 8);
            v152 = *((_QWORD *)v16 - 4);
            if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 <= 1 )
              v29 = **(_QWORD **)(v29 + 16);
            v17 = 1;
            v144 = v21;
            v30 = sub_BCAC40(v29, 1);
            v21 = v144;
            if ( v30 )
            {
LABEL_18:
              LOWORD(v162) = 257;
              return sub_B505E0(v151, (__int64)&v160, 0, 0);
            }
          }
        }
        if ( *v16 == 17 )
        {
          v143 = v21;
          v22 = sub_986B30((__int64 *)v16 + 3, v17, v21, v19, v20);
          LOBYTE(v21) = v143;
          v23 = v22;
        }
        else
        {
          v146 = *((_QWORD *)v16 + 1);
          if ( (unsigned int)*(unsigned __int8 *)(v146 + 8) - 17 > 1 || *v16 > 0x15u )
            goto LABEL_20;
          v139 = v21;
          v37 = (__int64 *)sub_AD7630((__int64)v16, 0, v21);
          LOBYTE(v21) = v139;
          if ( v37 && *(_BYTE *)v37 == 17 )
          {
            v39 = sub_986B30(v37 + 3, 0, v139, v38, v146);
            LOBYTE(v21) = v139;
            v23 = v39;
          }
          else
          {
            if ( *(_BYTE *)(v146 + 8) != 17 )
              goto LABEL_20;
            v135 = *(_DWORD *)(v146 + 32);
            if ( !v135 )
              goto LABEL_20;
            v147 = v139;
            v23 = 0;
            v140 = v3;
            for ( i = 0; i != v135; ++i )
            {
              v132 = v23;
              v89 = (__int64 *)sub_AD69F0(v16, i);
              if ( v89 )
              {
                v91 = *(unsigned __int8 *)v89;
                v23 = v132;
                if ( (_BYTE)v91 == 13 )
                  continue;
                if ( (_BYTE)v91 == 17 )
                {
                  v23 = sub_986B30(v89 + 3, i, v91, v132, v90);
                  if ( v23 )
                    continue;
                }
              }
              LOBYTE(v21) = v147;
              v3 = v140;
              goto LABEL_20;
            }
            LOBYTE(v21) = v147;
            v3 = v140;
          }
        }
        if ( v23 )
        {
          v24 = (unsigned int **)a1[2].m128i_i64[0];
          v159 = 257;
          v25 = sub_92B530(v24, 0x20u, v151, v16, (__int64)&v156);
          LOWORD(v162) = 257;
          v26 = v25;
          v27 = sub_BD2C40(72, unk_3F10A14);
          v11 = (__int64)v27;
          if ( v27 )
            sub_B515B0((__int64)v27, v26, v150, (__int64)&v160, 0, 0);
          return (unsigned __int8 *)v11;
        }
LABEL_20:
        v145 = v21;
        if ( !sub_B44E60(v3) )
          goto LABEL_33;
        v160.m128i_i64[0] = 0;
        v31 = sub_1159330((__int64 **)&v160, (__int64)v16);
        if ( v31 )
        {
          v32 = v145;
          if ( *v16 == 17 )
          {
            v33 = *((_DWORD *)v16 + 8);
            v34 = *((_QWORD *)v16 + 3);
            if ( v33 > 0x40 )
              v34 = *(_QWORD *)(v34 + 8LL * ((v33 - 1) >> 6));
            if ( (v34 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
              goto LABEL_26;
          }
          else
          {
            v136 = v145;
            v141 = v31;
            v148 = *((_QWORD *)v16 + 1);
            if ( (unsigned int)*(unsigned __int8 *)(v148 + 8) - 17 > 1 || *v16 > 0x15u )
              goto LABEL_40;
            v92 = (__int64 *)sub_AD7630((__int64)v16, 0, v32);
            v93 = v136;
            if ( v92 && *(_BYTE *)v92 == 17 )
            {
              v93 = !sub_986C60(v92 + 3, *((_DWORD *)v92 + 8) - 1);
            }
            else
            {
              if ( *(_BYTE *)(v148 + 8) != 17 )
                goto LABEL_40;
              v126 = v141;
              v142 = v3;
              v127 = 0;
              v138 = *(_DWORD *)(v148 + 32);
              while ( v138 != v127 )
              {
                v134 = v93;
                v128 = sub_AD69F0(v16, v127);
                if ( !v128 )
                  goto LABEL_39;
                v93 = v134;
                if ( *(_BYTE *)v128 != 13 )
                {
                  if ( *(_BYTE *)v128 != 17 || sub_986C60((__int64 *)(v128 + 24), *(_DWORD *)(v128 + 32) - 1) )
                  {
LABEL_39:
                    v11 = 0;
                    v3 = v142;
                    goto LABEL_40;
                  }
                  v93 = v126;
                }
                ++v127;
              }
              v11 = 0;
              v3 = v142;
            }
            if ( v93 )
            {
LABEL_26:
              v35 = sub_AD8AC0((__int64)v16);
              LOWORD(v162) = 257;
              v36 = v35;
LABEL_27:
              v11 = sub_B504D0(27, v151, (__int64)v36, (__int64)&v160, 0, 0);
              sub_B448B0(v11, 1);
              return (unsigned __int8 *)v11;
            }
          }
        }
LABEL_40:
        v160.m128i_i64[0] = 0;
        v160.m128i_i64[1] = (__int64)v155;
        if ( (unsigned __int8)sub_987880(v16) )
        {
          v54 = *v16;
          v55 = (unsigned __int8)v54 <= 0x1Cu ? *((unsigned __int16 *)v16 + 1) : v54 - 29;
          if ( v55 == 25 && (v16[1] & 4) != 0 )
          {
            if ( (unsigned __int8)sub_993A50(&v160, *((_QWORD *)v16 - 8)) )
            {
              v56 = *((_QWORD *)v16 - 4);
              if ( v56 )
              {
                *(_QWORD *)v160.m128i_i64[1] = v56;
                v36 = (unsigned __int8 *)v155[0];
                LOWORD(v162) = 257;
                goto LABEL_27;
              }
            }
          }
        }
        v160.m128i_i64[0] = 0;
        if ( (unsigned __int8)sub_1157E10((__int64 **)&v160, (__int64)v16) )
        {
          v57 = sub_AD6890((__int64)v16, 0);
          v58 = sub_AD8AC0(v57);
          v59 = a1[2].m128i_i64[0];
          v60 = (__int64)v58;
          v61 = sub_BD5D20(v3);
          v159 = 773;
          v156 = (__int64)v61;
          v157 = v62;
          v158 = ".neg";
          v63 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(v59 + 80) + 24LL))(
                  *(_QWORD *)(v59 + 80),
                  27,
                  v151,
                  v60,
                  1);
          if ( !v63 )
          {
            LOWORD(v162) = 257;
            v64 = sub_B504D0(27, v151, v60, (__int64)&v160, 0, 0);
            sub_B448B0(v64, 1);
            v63 = sub_1157250((__int64 *)v59, v64, (__int64)&v156);
          }
          LOWORD(v162) = 257;
          return sub_B505E0(v63, (__int64)&v160, 0, 0);
        }
LABEL_33:
        v40 = (unsigned __int64)v16;
        v160.m128i_i8[8] = 0;
        v160.m128i_i64[0] = (__int64)&v153;
        if ( (unsigned __int8)sub_991580((__int64)&v160, (__int64)v16) )
        {
          v44 = (__int64 *)v153;
          v45 = *(_QWORD *)(v151 + 16);
          if ( v45 )
          {
            if ( !*(_QWORD *)(v45 + 8) && *(_BYTE *)v151 == 69 )
            {
              v94 = *(_QWORD *)(v151 - 32);
              v133 = v94;
              if ( v94 )
              {
                v131 = v153;
                v137 = *(__int64 ***)(v94 + 8);
                v95 = sub_BCB060((__int64)v137);
                v44 = (__int64 *)v131;
                v149 = v95;
                v96 = *((_DWORD *)v131 + 2);
                v40 = (unsigned __int64)*v131;
                v42 = v96 - 1;
                v97 = 1LL << ((unsigned __int8)v96 - 1);
                if ( v96 > 0x40 )
                {
                  v130 = *((_DWORD *)v131 + 2);
                  v98 = (*(_QWORD *)(v40 + 8LL * ((unsigned int)v42 >> 6)) & v97) != 0
                      ? sub_C44500((__int64)v131)
                      : sub_C444A0((__int64)v131);
                  v44 = (__int64 *)v131;
                  v96 = v130;
                }
                else if ( (v97 & v40) != 0 )
                {
                  if ( v96 )
                  {
                    v98 = 64;
                    v42 = ~(v40 << (64 - (unsigned __int8)v96));
                    _BitScanReverse64(&v99, v42);
                    v40 = v99 ^ 0x3F;
                    if ( v42 )
                      v98 = v40;
                  }
                  else
                  {
                    v98 = 0;
                  }
                }
                else
                {
                  _BitScanReverse64(&v122, v40);
                  v123 = 64;
                  v42 = v122 ^ 0x3F;
                  if ( v40 )
                    v123 = v42;
                  v98 = v96 + v123 - 64;
                }
                v41 = v96 + 1 - v98;
                if ( v149 >= (unsigned int)v41 )
                {
                  v100 = sub_AD4C30((unsigned __int64)v16, v137, 0);
                  v101 = (__int64 *)a1[2].m128i_i64[0];
                  LOWORD(v162) = 257;
                  v102 = sub_1156550(v101, v133, v100, (__int64)&v160, 0);
                  LOWORD(v162) = 257;
                  v103 = v102;
                  v104 = sub_BD2C40(72, unk_3F10A14);
                  v11 = (__int64)v104;
                  if ( v104 )
                    sub_B51650((__int64)v104, v103, v150, (__int64)&v160, 0, 0);
                  return (unsigned __int8 *)v11;
                }
              }
            }
          }
          if ( !(unsigned __int8)sub_986B30(v44, v40, v41, v42, v43) )
          {
            v160.m128i_i64[0] = 0;
            v160.m128i_i64[1] = (__int64)&v152;
            if ( (unsigned __int8)sub_10E40A0(&v160, (unsigned __int8 *)v151) )
            {
              v105 = *((_DWORD *)v153 + 2);
              LODWORD(v157) = v105;
              if ( v105 > 0x40 )
              {
                sub_C43780((__int64)&v156, v153);
                v105 = v157;
                if ( (unsigned int)v157 > 0x40 )
                {
                  sub_C43D10((__int64)&v156);
LABEL_100:
                  sub_C46250((__int64)&v156);
                  v107 = v157;
                  LODWORD(v157) = 0;
                  v160.m128i_i32[2] = v107;
                  v160.m128i_i64[0] = v156;
                  v108 = sub_AD8D80(v150, (__int64)&v160);
                  sub_969240(v160.m128i_i64);
                  sub_969240(&v156);
                  LOWORD(v162) = 257;
                  v11 = sub_B504D0(20, v152, v108, (__int64)&v160, 0, 0);
                  v109 = sub_B44E60(v3);
                  sub_B448B0(v11, v109);
                  return (unsigned __int8 *)v11;
                }
              }
              else
              {
                v156 = (__int64)*v153;
              }
              v106 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v105;
              if ( !v105 )
                v106 = 0;
              v156 = v106 & ~v156;
              goto LABEL_100;
            }
          }
        }
        v46 = *(_BYTE *)v3 == 49;
        v160.m128i_i64[0] = 0;
        v160.m128i_i64[1] = (__int64)&v152;
        v161.m128i_i64[0] = (__int64)&v154;
        if ( v46 )
        {
          v65 = *(_QWORD *)(v3 - 64);
          v66 = *(_QWORD *)(v65 + 16);
          if ( v66 )
          {
            if ( !*(_QWORD *)(v66 + 8) )
            {
              if ( (unsigned __int8)sub_10E40A0(&v160, (unsigned __int8 *)v65) )
              {
                v67 = *(_QWORD *)(v3 - 32);
                if ( v67 )
                {
                  *(_QWORD *)v161.m128i_i64[0] = v67;
                  v68 = (__int64 *)a1[2].m128i_i64[0];
                  LOWORD(v162) = 257;
                  v69 = sub_B44E60(v3);
                  v70 = sub_BD5D20(v3);
                  v157 = v71;
                  v159 = 261;
                  v156 = (__int64)v70;
                  v63 = sub_1156550(v68, v152, v154, (__int64)&v156, v69);
                  return sub_B505E0(v63, (__int64)&v160, 0, 0);
                }
              }
            }
          }
        }
        v161.m128i_i64[0] = (__int64)&v152;
        v160.m128i_i32[0] = 1;
        v160.m128i_i32[2] = 0;
        v161.m128i_i32[2] = 1;
        v162 = 0;
        v163 = &v152;
        if ( sub_11594B0((__int64)&v160, v3) )
        {
          v47 = v152;
          v48 = (unsigned int **)a1[2].m128i_i64[0];
          LOWORD(v162) = 257;
          v49 = (_BYTE *)sub_AD62B0(*(_QWORD *)(v152 + 8));
          v50 = sub_92B530(v48, 0x26u, v47, v49, (__int64)&v160);
          LOWORD(v162) = 257;
          v51 = v50;
          v52 = sub_AD62B0(v150);
          v53 = sub_AD64C0(v150, 1, 0);
          return sub_109FEA0(v51, v53, v52, (const char **)&v160, 0, 0, 0);
        }
        v72 = _mm_loadu_si128(a1 + 6);
        v73 = _mm_loadu_si128(a1 + 7);
        v74 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
        v165 = a1[10].m128i_i64[0];
        v75 = _mm_loadu_si128(a1 + 9);
        v162 = v74;
        v160 = v72;
        v163 = (__int64 *)v3;
        v161 = v73;
        v164 = v75;
        sub_9AC330((__int64)&v156, v151, 0, &v160);
        if ( !sub_B44E60(v3) )
        {
          v155[0] = &v153;
          if ( (unsigned __int8)sub_1007280(v155, (__int64)v16, v76)
            || (v160.m128i_i64[0] = (__int64)&v153, (unsigned __int8)sub_10E8150((__int64 **)&v160, (__int64)v16)) )
          {
            if ( (unsigned int)v157 > 0x40 )
            {
              v119 = sub_C445E0((__int64)&v156);
            }
            else
            {
              v119 = 64;
              _RAX = ~v156;
              __asm { tzcnt   rcx, rax }
              if ( v156 != -1 )
                v119 = _RCX;
            }
            if ( (unsigned int)sub_D949C0((__int64)v153) <= v119 )
            {
              v11 = v3;
              sub_B448B0(v3, 1);
              goto LABEL_66;
            }
          }
        }
        if ( !sub_986C60(&v156, v157 - 1) )
          goto LABEL_140;
        v77 = a1 + 6;
        v78 = &v160;
        for ( j = 18; j; --j )
        {
          v78->m128i_i32[0] = v77->m128i_i32[0];
          v77 = (__m128i *)((char *)v77 + 4);
          v78 = (__m128i *)((char *)v78 + 4);
        }
        v163 = (__int64 *)v3;
        if ( !(unsigned __int8)sub_9AC470((__int64)v16, &v160, 0) )
        {
          v160.m128i_i64[0] = 0;
          if ( (unsigned __int8)sub_1157E10((__int64 **)&v160, (__int64)v16) )
          {
            v81 = sub_AD6890((__int64)v16, 0);
            v82 = sub_AD8AC0(v81);
            v83 = (__int64 *)a1[2].m128i_i64[0];
            v84 = (__int64)v82;
            v85 = sub_B44E60(v3);
            v86.m128i_i64[0] = (__int64)sub_BD5D20(v3);
            LOWORD(v162) = 261;
            v160 = v86;
            v87 = sub_F94560(v83, v151, v84, (__int64)&v160, v85);
            LOWORD(v162) = 257;
            v11 = sub_B50550(v87, (__int64)&v160, 0, 0);
LABEL_66:
            sub_969240((__int64 *)&v158);
            sub_969240(&v156);
            return (unsigned __int8 *)v11;
          }
          v110 = a1 + 6;
          v111 = 18;
          v112 = &v160;
          while ( v111 )
          {
            v112->m128i_i32[0] = v110->m128i_i32[0];
            v110 = (__m128i *)((char *)v110 + 4);
            v112 = (__m128i *)((char *)v112 + 4);
            --v111;
          }
          v163 = (__int64 *)v3;
          if ( !(unsigned __int8)sub_9A1DB0(v16, 1, 0, (__int64)&v160, v80) )
          {
LABEL_140:
            if ( sub_98F660((unsigned __int8 *)v151, v16, 0, 1) )
            {
              v113 = sub_BCB060(v150);
              sub_986680((__int64)v155, v113);
              v114 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v162) = 257;
              v115 = (_BYTE *)sub_AD8D80(v150, (__int64)v155);
              v116 = sub_92B530(v114, 0x20u, v151, v115, (__int64)&v160);
              LOWORD(v162) = 257;
              v117 = v116;
              v118 = sub_AD62B0(v150);
              v129 = sub_AD64C0(v150, 1, 0);
              v11 = (__int64)sub_109FEA0(v117, v129, v118, (const char **)&v160, 0, 0, 0);
              sub_969240((__int64 *)v155);
            }
            goto LABEL_66;
          }
        }
        v124.m128i_i64[0] = (__int64)sub_BD5D20(v3);
        LOWORD(v162) = 261;
        v160 = v124;
        v11 = sub_B504D0(19, v151, (__int64)v16, (__int64)&v160, 0, 0);
        v125 = sub_B44E60(v3);
        sub_B448B0(v11, v125);
        goto LABEL_66;
      }
    }
  }
  return (unsigned __int8 *)v11;
}
