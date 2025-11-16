// Function: sub_1FD0C90
// Address: 0x1fd0c90
//
__int64 *__fastcall sub_1FD0C90(
        __int128 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __m128 a7,
        __m128i a8)
{
  __int64 v8; // r13
  __int64 *v9; // r12
  __int64 *v10; // r14
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  bool v20; // al
  unsigned int v21; // r15d
  bool (__fastcall *v22)(__int64, __int64, unsigned __int8); // rax
  __m128i v23; // xmm0
  __int64 (*v24)(); // rax
  __int64 *v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // rdi
  __int128 v35; // rax
  __int64 (__fastcall **v36)(); // rsi
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rdx
  bool v40; // al
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  int v45; // r9d
  __int64 v46; // rsi
  bool (__fastcall *v47)(__int64, unsigned int); // rax
  unsigned __int8 *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rax
  bool v51; // al
  bool (__fastcall *v52)(__int64, __int64, unsigned __int8); // rax
  __int64 (*v53)(); // rax
  int v54; // edx
  unsigned __int8 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  bool v58; // al
  unsigned int v59; // r15d
  bool (__fastcall *v60)(__int64, __int64, unsigned __int8); // rax
  __m128i v61; // xmm1
  __int64 (*v62)(); // rax
  char v63; // al
  __int16 v64; // r8
  __int64 v65; // rax
  unsigned int v66; // edx
  __int64 v67; // r11
  unsigned int v68; // eax
  __int64 (__fastcall **v69)(); // rsi
  __int128 v70; // rax
  __int64 (__fastcall **v71)(); // rsi
  __int64 v72; // rax
  __int64 v73; // rdx
  int v74; // ecx
  __int64 (__fastcall **v75)(); // rsi
  int v76; // eax
  int v77; // edi
  int v78; // edi
  __int16 v79; // ax
  __int64 v80; // rdi
  unsigned __int16 v81; // si
  __int64 v82; // rdx
  unsigned __int8 *v83; // rax
  __int64 v84; // r15
  __int64 v85; // rax
  bool v86; // al
  bool (__fastcall *v87)(__int64, __int64, unsigned __int8); // rax
  __m128i v88; // xmm2
  __int64 (*v89)(); // rax
  bool v90; // al
  bool v91; // al
  bool v92; // al
  bool v93; // al
  char v94; // al
  __int64 (__fastcall **v95)(); // rsi
  __int64 *v96; // r15
  __int128 *v97; // rcx
  __int64 v98; // rsi
  unsigned int v99; // edx
  __int64 v100; // rsi
  __int64 v101; // rdi
  __int64 v102; // rsi
  char v103; // r10
  __int128 v104; // rax
  __int64 v105; // r15
  unsigned int v106; // edx
  unsigned int v107; // r8d
  __int64 v108; // rdx
  __int64 v109; // rdi
  __int64 v110; // rcx
  __int64 v111; // rax
  __int128 v112; // [rsp-10h] [rbp-130h]
  __int64 v113; // [rsp+0h] [rbp-120h]
  __int64 v114; // [rsp+8h] [rbp-118h]
  __int64 v115; // [rsp+10h] [rbp-110h]
  unsigned __int64 v116; // [rsp+18h] [rbp-108h]
  int v117; // [rsp+24h] [rbp-FCh]
  int v118; // [rsp+28h] [rbp-F8h]
  char v119; // [rsp+2Fh] [rbp-F1h]
  __int64 v120; // [rsp+30h] [rbp-F0h]
  __int64 v121; // [rsp+30h] [rbp-F0h]
  __int64 v122; // [rsp+30h] [rbp-F0h]
  unsigned int v123; // [rsp+30h] [rbp-F0h]
  __int64 v124; // [rsp+38h] [rbp-E8h]
  __int16 v125; // [rsp+38h] [rbp-E8h]
  __int16 v126; // [rsp+38h] [rbp-E8h]
  unsigned int v127; // [rsp+38h] [rbp-E8h]
  __int64 v128; // [rsp+38h] [rbp-E8h]
  __int64 v129; // [rsp+40h] [rbp-E0h]
  __int128 v130; // [rsp+40h] [rbp-E0h]
  __int64 v131; // [rsp+40h] [rbp-E0h]
  __int64 v132; // [rsp+40h] [rbp-E0h]
  __int64 v133; // [rsp+40h] [rbp-E0h]
  __int128 *v134; // [rsp+40h] [rbp-E0h]
  __int64 v135; // [rsp+50h] [rbp-D0h]
  __int16 v136; // [rsp+50h] [rbp-D0h]
  __m128i v137; // [rsp+50h] [rbp-D0h]
  __int64 *v138; // [rsp+50h] [rbp-D0h]
  __int64 v139; // [rsp+50h] [rbp-D0h]
  __int16 v140; // [rsp+50h] [rbp-D0h]
  __int64 v141; // [rsp+50h] [rbp-D0h]
  unsigned __int64 v142; // [rsp+58h] [rbp-C8h]
  char v143; // [rsp+9Eh] [rbp-82h] BYREF
  char v144; // [rsp+9Fh] [rbp-81h] BYREF
  __m128i v145; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v146; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v147; // [rsp+C0h] [rbp-60h] BYREF
  __int64 (__fastcall **v148)(); // [rsp+D0h] [rbp-50h] BYREF
  __int64 v149; // [rsp+D8h] [rbp-48h]
  __int64 v150; // [rsp+E0h] [rbp-40h]
  __int64 *v151; // [rsp+E8h] [rbp-38h]

  v8 = *((_QWORD *)&a1 + 1);
  v9 = (__int64 *)a1;
  v10 = sub_1FCE100(a1, a2, a3, a4, a5, a6, a7, a8);
  if ( v10 )
    return v10;
  v12 = *(unsigned __int16 *)(*((_QWORD *)&a1 + 1) + 24LL);
  v13 = *(_QWORD *)(a1 + 8);
  *((_QWORD *)&a1 + 1) = (unsigned __int16)v12;
  if ( (unsigned __int16)v12 > 0x102u
    || (v54 = *(unsigned __int8 *)(v13 + ((unsigned __int64)(unsigned __int16)v12 >> 3) + 74015), _bittest(
                                                                                                    &v54,
                                                                                                    v12 & 7)) )
  {
    v14 = *(_DWORD *)(a1 + 16);
    v15 = *(_QWORD *)a1;
    v148 = (__int64 (__fastcall **)())a1;
    BYTE4(v149) = 0;
    LODWORD(v149) = v14;
    v150 = v15;
    v16 = (*(__int64 (__fastcall **)(__int64, __int64, __int64 (__fastcall ***)()))(*(_QWORD *)v13 + 1112LL))(
            v13,
            v8,
            &v148);
    if ( v16 )
      return (__int64 *)v16;
    v12 = *(unsigned __int16 *)(v8 + 24);
    v13 = *(_QWORD *)(a1 + 8);
    *((_QWORD *)&a1 + 1) = (unsigned __int16)v12;
  }
  if ( (__int16)v12 <= 144 )
  {
    if ( (__int16)v12 > 141 )
    {
      if ( !*(_BYTE *)(a1 + 24) )
        goto LABEL_80;
      v83 = *(unsigned __int8 **)(v8 + 40);
      v84 = *v83;
      v85 = *((_QWORD *)v83 + 1);
      v146.m128i_i8[0] = v84;
      v146.m128i_i64[1] = v85;
      if ( (_BYTE)v84 )
      {
        if ( (unsigned __int8)(v84 - 14) <= 0x5Fu )
          goto LABEL_80;
        v86 = (unsigned __int8)(v84 - 2) <= 5u;
      }
      else
      {
        v127 = v12;
        v133 = v13;
        v92 = sub_1F58D20((__int64)&v146);
        v13 = v133;
        LOWORD(v12) = v127;
        if ( v92 )
          goto LABEL_80;
        v86 = sub_1F58CF0((__int64)&v146);
        v12 = v127;
        v13 = v133;
      }
      if ( !v86 )
        goto LABEL_80;
      v87 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*(_QWORD *)v13 + 1136LL);
      if ( v87 == sub_1F6BB70 )
      {
        if ( (_BYTE)v84 && *(_QWORD *)(v13 + 8 * v84 + 120) )
          goto LABEL_80;
      }
      else
      {
        if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v87)(
               v13,
               (unsigned int)(__int16)v12,
               v146.m128i_u32[0],
               v146.m128i_i64[1]) )
        {
          goto LABEL_112;
        }
        v13 = *(_QWORD *)(a1 + 8);
      }
      v88 = _mm_loadu_si128(&v146);
      v147 = v88;
      v89 = *(__int64 (**)())(*(_QWORD *)v13 + 1152LL);
      if ( v89 != sub_1F6BB90 )
      {
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __m128i *, __int64))v89)(
                v13,
                v8,
                0,
                &v147,
                v12) )
          goto LABEL_112;
        v95 = *(__int64 (__fastcall ***)())(v8 + 72);
        v96 = *(__int64 **)a1;
        v97 = *(__int128 **)(v8 + 32);
        v148 = v95;
        if ( v95 )
        {
          v134 = v97;
          sub_1623A60((__int64)&v148, (__int64)v95, 2);
          v97 = v134;
        }
        v98 = *(unsigned __int16 *)(v8 + 24);
        LODWORD(v149) = *(_DWORD *)(v8 + 64);
        v16 = sub_1D309E0(
                v96,
                v98,
                (__int64)&v148,
                v146.m128i_u32[0],
                (const void **)v146.m128i_i64[1],
                0,
                *(double *)a6.m128i_i64,
                *(double *)a7.m128_u64,
                *(double *)v88.m128i_i64,
                *v97);
        v71 = v148;
        if ( !v148 )
          goto LABEL_126;
        v139 = v16;
        goto LABEL_125;
      }
    }
    else
    {
      if ( (__int16)v12 > 124 )
        goto LABEL_39;
      if ( (__int16)v12 <= 121 )
      {
        if ( (__int16)v12 <= 54 )
        {
          if ( (__int16)v12 > 51 )
          {
LABEL_12:
            if ( *(_BYTE *)(a1 + 24) )
            {
              v17 = *(unsigned __int8 **)(v8 + 40);
              v18 = *v17;
              v19 = *((_QWORD *)v17 + 1);
              v145.m128i_i8[0] = v18;
              v145.m128i_i64[1] = v19;
              if ( (_BYTE)v18 )
              {
                if ( (unsigned __int8)(v18 - 14) <= 0x5Fu )
                  goto LABEL_80;
                v20 = (unsigned __int8)(v18 - 2) <= 5u;
              }
              else
              {
                v126 = v12;
                v132 = v13;
                v91 = sub_1F58D20((__int64)&v145);
                v13 = v132;
                LOWORD(v12) = v126;
                if ( v91 )
                  goto LABEL_80;
                v20 = sub_1F58CF0((__int64)&v145);
                LOWORD(v12) = v126;
                v13 = v132;
                v18 = 0;
              }
              if ( v20 )
              {
                v21 = (__int16)v12;
                v22 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*(_QWORD *)v13 + 1136LL);
                if ( v22 == sub_1F6BB70 )
                {
                  if ( (_BYTE)v18 && *(_QWORD *)(v13 + 8 * v18 + 120) )
                    goto LABEL_80;
LABEL_20:
                  v23 = _mm_loadu_si128(&v145);
                  v146 = v23;
                  v24 = *(__int64 (**)())(*(_QWORD *)v13 + 1152LL);
                  if ( v24 != sub_1F6BB90 )
                  {
                    v119 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __m128i *))v24)(v13, v8, 0, &v146);
                    if ( v119 )
                    {
                      v25 = *(__int64 **)(v8 + 32);
                      v143 = 0;
                      v135 = *v25;
                      v118 = *((_DWORD *)v25 + 2);
                      v115 = sub_1F973B0(
                               (__int64 *)a1,
                               *v25,
                               v25[1],
                               v146.m128i_u32[0],
                               (const void **)v146.m128i_i64[1],
                               &v143,
                               v23,
                               *(double *)a7.m128_u64,
                               a8);
                      v124 = v115;
                      v26 = *(_QWORD *)(v8 + 32);
                      v116 = v27;
                      *((_QWORD *)&a1 + 1) = *(_QWORD *)(v26 + 40);
                      v28 = *(_QWORD *)(v26 + 48);
                      v144 = 0;
                      v129 = *((_QWORD *)&a1 + 1);
                      v117 = *(_DWORD *)(v26 + 48);
                      v29 = sub_1F973B0(
                              (__int64 *)a1,
                              *((__int64 *)&a1 + 1),
                              v28,
                              v146.m128i_u32[0],
                              (const void **)v146.m128i_i64[1],
                              &v144,
                              v23,
                              *(double *)a7.m128_u64,
                              a8);
                      v31 = *(_QWORD *)(v8 + 72);
                      v120 = v29;
                      v32 = v29;
                      v33 = v30;
                      v147.m128i_i64[0] = v31;
                      if ( v31 )
                      {
                        v114 = v30;
                        v113 = v29;
                        sub_1623A60((__int64)&v147, v31, 2);
                        v32 = v113;
                        v33 = v114;
                      }
                      *((_QWORD *)&v112 + 1) = v33;
                      v34 = *(__int64 **)a1;
                      *(_QWORD *)&v112 = v32;
                      v147.m128i_i32[2] = *(_DWORD *)(v8 + 64);
                      *(_QWORD *)&v35 = sub_1D332F0(
                                          v34,
                                          v21,
                                          (__int64)&v147,
                                          v146.m128i_u32[0],
                                          (const void **)v146.m128i_i64[1],
                                          0,
                                          *(double *)v23.m128i_i64,
                                          *(double *)a7.m128_u64,
                                          a8,
                                          v115,
                                          v116,
                                          v112);
                      v36 = (__int64 (__fastcall **)())sub_1D309E0(
                                                         v34,
                                                         145,
                                                         (__int64)&v147,
                                                         v145.m128i_u32[0],
                                                         (const void **)v145.m128i_i64[1],
                                                         0,
                                                         *(double *)v23.m128i_i64,
                                                         *(double *)a7.m128_u64,
                                                         *(double *)a8.m128i_i64,
                                                         v35);
                      v38 = v37;
                      v39 = *(_QWORD *)(v135 + 48);
                      v40 = 0;
                      if ( v39 )
                        v40 = *(_QWORD *)(v39 + 32) == 0;
                      v143 &= !v40;
                      if ( v118 == v117 && v135 == v129 )
                      {
                        v119 = 0;
                      }
                      else
                      {
                        v41 = *(_QWORD *)(v129 + 48);
                        if ( v41 )
                          v119 = *(_QWORD *)(v41 + 32) != 0;
                      }
                      v148 = v36;
                      v149 = v38;
                      v144 &= v119;
                      sub_1F994A0((__int64)v9, v8, (__int64 *)&v148, 1, 1);
                      if ( v143 )
                      {
                        if ( !v144 )
                          goto LABEL_32;
                        if ( (unsigned __int8)sub_1D19270(v129, v135, v42, v43, v44, v45) )
                        {
                          v110 = v120;
                          v120 = v115;
                          v111 = v135;
                          v124 = v110;
                          v135 = v129;
                          v129 = v111;
                        }
                        if ( v143 )
                        {
LABEL_32:
                          sub_1F81BC0((__int64)v9, v124);
                          sub_1F97190(
                            v9,
                            v135,
                            v124,
                            *(double *)v23.m128i_i64,
                            *(double *)a7.m128_u64,
                            *(double *)a8.m128i_i64);
                        }
                      }
                      if ( v144 )
                      {
                        sub_1F81BC0((__int64)v9, v120);
                        sub_1F97190(
                          v9,
                          v129,
                          v120,
                          *(double *)v23.m128i_i64,
                          *(double *)a7.m128_u64,
                          *(double *)a8.m128i_i64);
                      }
                      v46 = v147.m128i_i64[0];
                      if ( !v147.m128i_i64[0] )
                        return (__int64 *)v8;
                      goto LABEL_36;
                    }
                    goto LABEL_112;
                  }
                  goto LABEL_102;
                }
                if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v22)(
                        v13,
                        (unsigned int)(__int16)v12,
                        v145.m128i_u32[0],
                        v145.m128i_i64[1]) )
                {
                  v13 = *(_QWORD *)(a1 + 8);
                  goto LABEL_20;
                }
LABEL_112:
                LOWORD(v12) = *(_WORD *)(v8 + 24);
                v13 = *(_QWORD *)(a1 + 8);
              }
            }
LABEL_80:
            *(_QWORD *)&a1 = v13;
            *((_QWORD *)&a1 + 1) = (unsigned __int16)v12;
            goto LABEL_40;
          }
        }
        else if ( (unsigned __int16)(v12 - 118) <= 2u )
        {
          goto LABEL_12;
        }
LABEL_39:
        *(_QWORD *)&a1 = v13;
        goto LABEL_40;
      }
      if ( !*(_BYTE *)(a1 + 24) )
        goto LABEL_80;
      v55 = *(unsigned __int8 **)(v8 + 40);
      v56 = *v55;
      v57 = *((_QWORD *)v55 + 1);
      v146.m128i_i8[0] = v56;
      v146.m128i_i64[1] = v57;
      if ( (_BYTE)v56 )
      {
        if ( (unsigned __int8)(v56 - 14) <= 0x5Fu )
          goto LABEL_80;
        v58 = (unsigned __int8)(v56 - 2) <= 5u;
      }
      else
      {
        v125 = v12;
        v131 = v13;
        v90 = sub_1F58D20((__int64)&v146);
        v13 = v131;
        LOWORD(v12) = v125;
        if ( v90 )
          goto LABEL_80;
        v58 = sub_1F58CF0((__int64)&v146);
        LOWORD(v12) = v125;
        v13 = v131;
        v56 = 0;
      }
      if ( !v58 )
        goto LABEL_80;
      v59 = (__int16)v12;
      v60 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*(_QWORD *)v13 + 1136LL);
      if ( v60 == sub_1F6BB70 )
      {
        if ( (_BYTE)v56 && *(_QWORD *)(v13 + 8 * v56 + 120) )
          goto LABEL_80;
      }
      else
      {
        v140 = v12;
        v94 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v60)(
                v13,
                (unsigned int)(__int16)v12,
                v146.m128i_u32[0],
                v146.m128i_i64[1]);
        LOWORD(v12) = v140;
        if ( v94 )
          goto LABEL_112;
        v13 = *(_QWORD *)(a1 + 8);
      }
      v61 = _mm_loadu_si128(&v146);
      v147 = v61;
      v62 = *(__int64 (**)())(*(_QWORD *)v13 + 1152LL);
      if ( v62 != sub_1F6BB90 )
      {
        v136 = v12;
        v63 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __m128i *))v62)(v13, v8, 0, &v147);
        v64 = v136;
        if ( !v63 )
          goto LABEL_112;
        v65 = *(_QWORD *)(v8 + 32);
        v145.m128i_i8[0] = 0;
        v137 = _mm_loadu_si128((const __m128i *)v65);
        v130 = (__int128)_mm_loadu_si128((const __m128i *)(v65 + 40));
        if ( v64 == 123 )
        {
          v67 = (__int64)sub_1F97690(
                           (__int64 *)a1,
                           v137.m128i_i64[0],
                           v137.m128i_i64[1],
                           v147.m128i_u32[0],
                           (const void **)v147.m128i_i64[1],
                           a6,
                           *(double *)v61.m128i_i64,
                           a8);
          v68 = v99;
        }
        else
        {
          if ( v59 == 124 )
            v67 = (__int64)sub_1F972B0(
                             (__int64 *)a1,
                             v137.m128i_i64[0],
                             v137.m128i_i64[1],
                             v147.m128i_u32[0],
                             v147.m128i_i64[1],
                             a6,
                             *(double *)v61.m128i_i64,
                             a8);
          else
            v67 = sub_1F973B0(
                    (__int64 *)a1,
                    v137.m128i_i64[0],
                    v137.m128i_i64[1],
                    v147.m128i_u32[0],
                    (const void **)v147.m128i_i64[1],
                    &v145,
                    a6,
                    *(double *)v61.m128i_i64,
                    a8);
          v68 = v66;
        }
        v142 = v68 | v137.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        if ( !v67 )
        {
          LOWORD(v12) = *(_WORD *)(v8 + 24);
          v13 = *(_QWORD *)(a1 + 8);
          goto LABEL_80;
        }
        v69 = *(__int64 (__fastcall ***)())(v8 + 72);
        v148 = v69;
        if ( v69 )
        {
          v121 = v67;
          sub_1623A60((__int64)&v148, (__int64)v69, 2);
          v67 = v121;
        }
        v122 = v67;
        v138 = *(__int64 **)a1;
        LODWORD(v149) = *(_DWORD *)(v8 + 64);
        *(_QWORD *)&v70 = sub_1D332F0(
                            v138,
                            v59,
                            (__int64)&v148,
                            v147.m128i_u32[0],
                            (const void **)v147.m128i_i64[1],
                            0,
                            *(double *)a6.m128i_i64,
                            *(double *)v61.m128i_i64,
                            a8,
                            v67,
                            v142,
                            v130);
        v139 = sub_1D309E0(
                 v138,
                 145,
                 (__int64)&v148,
                 v146.m128i_u32[0],
                 (const void **)v146.m128i_i64[1],
                 0,
                 *(double *)a6.m128i_i64,
                 *(double *)v61.m128i_i64,
                 *(double *)a8.m128i_i64,
                 v70);
        sub_1F81BC0(a1, v122);
        if ( v145.m128i_i8[0] )
          sub_1F97190(
            (__int64 *)a1,
            **(_QWORD **)(v8 + 32),
            v122,
            *(double *)a6.m128i_i64,
            *(double *)v61.m128i_i64,
            *(double *)a8.m128i_i64);
        v71 = v148;
        if ( !*(_WORD *)(v8 + 24) )
        {
          if ( v148 )
          {
            sub_161E7C0((__int64)&v148, (__int64)v148);
            LOWORD(v12) = *(_WORD *)(v8 + 24);
            v13 = *(_QWORD *)(a1 + 8);
          }
          else
          {
            v13 = *(_QWORD *)(a1 + 8);
            LOWORD(v12) = 0;
          }
          goto LABEL_80;
        }
        v16 = v139;
        if ( !v148 )
        {
LABEL_126:
          if ( !v16 )
          {
LABEL_127:
            LOWORD(v12) = *(_WORD *)(v8 + 24);
            v13 = v9[1];
            goto LABEL_80;
          }
          return (__int64 *)v16;
        }
LABEL_125:
        sub_161E7C0((__int64)&v148, (__int64)v71);
        v16 = v139;
        goto LABEL_126;
      }
    }
LABEL_102:
    LOWORD(v12) = *(_WORD *)(v8 + 24);
    goto LABEL_80;
  }
  if ( (_WORD)v12 != 185 )
    goto LABEL_39;
  if ( !*(_BYTE *)(a1 + 24) || (*(_WORD *)(v8 + 26) & 0x380) != 0 )
    goto LABEL_90;
  v48 = *(unsigned __int8 **)(v8 + 40);
  v49 = *v48;
  v50 = *((_QWORD *)v48 + 1);
  v145.m128i_i8[0] = v49;
  v145.m128i_i64[1] = v50;
  if ( !(_BYTE)v49 )
  {
    v123 = v12;
    v128 = v13;
    v93 = sub_1F58D20((__int64)&v145);
    v13 = v128;
    if ( !v93 )
    {
      v51 = sub_1F58CF0((__int64)&v145);
      v12 = v123;
      v13 = v128;
      v49 = 0;
      goto LABEL_50;
    }
LABEL_90:
    *(_QWORD *)&a1 = v13;
    *((_QWORD *)&a1 + 1) = 185;
    v47 = *(bool (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v13 + 776LL);
    if ( v47 == sub_1D12DA0 )
      return v10;
    goto LABEL_91;
  }
  if ( (unsigned __int8)(v49 - 14) <= 0x5Fu )
    goto LABEL_90;
  v51 = (unsigned __int8)(v49 - 2) <= 5u;
LABEL_50:
  if ( !v51 )
    goto LABEL_90;
  *(_QWORD *)&a1 = v13;
  v52 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*(_QWORD *)v13 + 1136LL);
  if ( v52 == sub_1F6BB70 )
  {
    if ( (_BYTE)v49 && *(_QWORD *)(v13 + 8 * v49 + 120) )
      goto LABEL_80;
  }
  else
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v52)(
           v13,
           185,
           v145.m128i_u32[0],
           v145.m128i_i64[1],
           v12) )
    {
      goto LABEL_127;
    }
    *(_QWORD *)&a1 = v9[1];
  }
  v146 = _mm_loadu_si128(&v145);
  v53 = *(__int64 (**)())(*(_QWORD *)a1 + 1152LL);
  if ( v53 != sub_1F6BB90 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_QWORD, __int64, _QWORD, __m128i *, __int64))v53)(a1, v8, 0, &v146, v12) )
    {
      v100 = *(_QWORD *)(v8 + 72);
      v147.m128i_i64[0] = v100;
      if ( v100 )
        sub_1623A60((__int64)&v147, v100, 2);
      v101 = *(_QWORD *)(v8 + 96);
      v102 = *(unsigned __int8 *)(v8 + 88);
      v147.m128i_i32[2] = *(_DWORD *)(v8 + 64);
      if ( *(_WORD *)(v8 + 24) != 185 || (v103 = 1, ((*(_BYTE *)(v8 + 27) >> 2) & 3) != 0) )
        v103 = (*(_BYTE *)(v8 + 27) >> 2) & 3;
      *(_QWORD *)&v104 = sub_1D2B590(
                           (_QWORD *)*v9,
                           v103,
                           (__int64)&v147,
                           v146.m128i_u32[0],
                           v146.m128i_i64[1],
                           *(_QWORD *)(v8 + 104),
                           *(_OWORD *)*(_QWORD *)(v8 + 32),
                           *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v8 + 32) + 48LL),
                           v102,
                           v101);
      v141 = v104;
      v105 = sub_1D309E0(
               (__int64 *)*v9,
               145,
               (__int64)&v147,
               v145.m128i_u32[0],
               (const void **)v145.m128i_i64[1],
               0,
               *(double *)a6.m128i_i64,
               *(double *)a7.m128_u64,
               *(double *)a8.m128i_i64,
               v104);
      v107 = v106;
      v108 = *(_QWORD *)(*v9 + 664);
      v150 = *v9;
      v149 = v108;
      *(_QWORD *)(v150 + 664) = &v148;
      v109 = *v9;
      v148 = off_49FFF30;
      v151 = v9;
      sub_1D44C70(v109, v8, 0, v105, v107);
      sub_1D44C70(*v9, v8, 1, v141, 1u);
      sub_1F81E80(v9, v8);
      sub_1F81BC0((__int64)v9, v105);
      *(_QWORD *)(v150 + 664) = v149;
      v46 = v147.m128i_i64[0];
      if ( !v147.m128i_i64[0] )
        return (__int64 *)v8;
LABEL_36:
      sub_161E7C0((__int64)&v147, v46);
      return (__int64 *)v8;
    }
    *(_QWORD *)&a1 = v9[1];
  }
  *((_QWORD *)&a1 + 1) = *(unsigned __int16 *)(v8 + 24);
LABEL_40:
  v47 = *(bool (__fastcall **)(__int64, unsigned int))(*(_QWORD *)a1 + 776LL);
  if ( v47 != sub_1D12DA0 )
  {
LABEL_91:
    if ( !v47(a1, *((_QWORD *)&a1 + 1)) )
      return v10;
LABEL_82:
    if ( *(_DWORD *)(v8 + 60) == 1 )
    {
      v72 = *(_QWORD *)(v8 + 32);
      v73 = *(_QWORD *)v72;
      v74 = *(_DWORD *)(v72 + 8);
      v75 = *(__int64 (__fastcall ***)())(v72 + 40);
      v76 = *(_DWORD *)(v72 + 48);
      if ( v74 != v76 || (__int64 (__fastcall **)())v73 != v75 )
      {
        v77 = *(unsigned __int16 *)(v73 + 24);
        if ( v77 == 32 || v77 == 10 || (v78 = *((unsigned __int16 *)v75 + 12), v78 != 32) && v78 != 10 )
        {
          LODWORD(v149) = v76;
          v79 = *(_WORD *)(v8 + 80);
          v148 = v75;
          v80 = *v9;
          v81 = *(_WORD *)(v8 + 24);
          v150 = v73;
          v82 = *(_QWORD *)(v8 + 40);
          LODWORD(v151) = v74;
          return sub_1D197A0(v80, v81, v82, 1, (__int64 *)&v148, 2, v79);
        }
      }
    }
    return v10;
  }
  if ( SWORD4(a1) > 120 )
  {
    if ( (unsigned __int16)(WORD4(a1) - 180) > 3u )
      return v10;
    goto LABEL_82;
  }
  if ( SWORD4(a1) > 51 )
  {
    switch ( WORD4(a1) )
    {
      case '4':
      case '6':
      case ';':
      case '<':
      case '@':
      case 'B':
      case 'F':
      case 'G':
      case 'L':
      case 'N':
      case 'p':
      case 'q':
      case 'r':
      case 's':
      case 't':
      case 'u':
      case 'v':
      case 'w':
      case 'x':
        goto LABEL_82;
      default:
        return v10;
    }
  }
  return v10;
}
