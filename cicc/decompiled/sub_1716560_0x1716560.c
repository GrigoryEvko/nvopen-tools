// Function: sub_1716560
// Address: 0x1716560
//
__int64 __fastcall sub_1716560(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128i a9,
        __m128 a10)
{
  __int64 v11; // r13
  unsigned int v12; // eax
  unsigned __int64 v13; // r12
  int v14; // eax
  int v15; // esi
  __int64 v16; // rcx
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  _QWORD **v21; // r8
  int v22; // r9d
  double v23; // xmm4_8
  double v24; // xmm5_8
  unsigned __int64 v25; // rax
  __int64 ***v26; // r13
  char v27; // al
  __int64 v28; // rcx
  __int64 *v29; // r14
  __int64 *v30; // r13
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rsi
  __int64 *v35; // r13
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  __int64 v38; // rax
  double v39; // xmm4_8
  double v40; // xmm5_8
  __int64 v41; // r13
  __int64 v42; // rsi
  __int64 *v43; // r14
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  __int64 v46; // r14
  __int64 v47; // r15
  _QWORD *v48; // rax
  __int64 v49; // r15
  __int64 v50; // rcx
  __int64 v51; // rax
  int v52; // eax
  __int64 v53; // rdx
  _QWORD *v54; // rax
  _QWORD *i; // rdx
  int v56; // eax
  __int64 v57; // rdx
  _QWORD *v58; // rax
  _QWORD *k; // rdx
  _QWORD *v60; // r13
  _QWORD *v61; // r12
  __int64 v62; // rax
  __int64 result; // rax
  __int64 v64; // r15
  __int64 v65; // r14
  _QWORD *v66; // rax
  double v67; // xmm4_8
  double v68; // xmm5_8
  __int64 v69; // r13
  _QWORD *v70; // rax
  __int64 v71; // r13
  _QWORD *v72; // rax
  unsigned __int64 v73; // rax
  __int64 v74; // r13
  unsigned int v75; // r14d
  int v76; // eax
  unsigned __int64 v77; // rax
  __int64 v78; // rdi
  int v79; // r13d
  unsigned __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rax
  char v83; // al
  char v84; // al
  __int64 v85; // rcx
  unsigned __int64 v86; // rdx
  __int64 v87; // r13
  __int64 v88; // rdi
  int v89; // eax
  int v90; // r8d
  unsigned int v91; // ecx
  _QWORD *v92; // rdi
  unsigned int v93; // eax
  int v94; // eax
  unsigned __int64 v95; // rax
  unsigned __int64 v96; // rax
  int v97; // r13d
  __int64 v98; // r12
  _QWORD *v99; // rax
  __int64 v100; // rdx
  _QWORD *m; // rdx
  unsigned int v102; // ecx
  _QWORD *v103; // rdi
  unsigned int v104; // eax
  int v105; // eax
  unsigned __int64 v106; // rax
  unsigned __int64 v107; // rax
  int v108; // r14d
  __int64 v109; // r12
  _QWORD *v110; // rax
  __int64 v111; // rdx
  _QWORD *j; // rdx
  __int64 v113; // r14
  __int64 v114; // r13
  _QWORD *v115; // rax
  __int64 v116; // rax
  __int64 v117; // rcx
  unsigned __int64 v118; // rdx
  _QWORD *v119; // rax
  _QWORD *v120; // rax
  __int64 v121; // rax
  __int64 v122; // r13
  _BYTE *v123; // rdi
  __int64 v124; // rcx
  _QWORD **v125; // rdx
  __int64 v126; // r13
  __int64 v127; // rax
  __int64 *v128; // r13
  __int64 v129; // [rsp+0h] [rbp-70h]
  __int64 v130; // [rsp+8h] [rbp-68h]
  _QWORD **v131; // [rsp+8h] [rbp-68h]
  __int64 v132; // [rsp+10h] [rbp-60h]
  int v133; // [rsp+10h] [rbp-60h]
  __int64 v134; // [rsp+10h] [rbp-60h]
  __int64 v135; // [rsp+10h] [rbp-60h]
  unsigned __int64 v136; // [rsp+10h] [rbp-60h]
  __int64 v137; // [rsp+10h] [rbp-60h]
  _QWORD **v138; // [rsp+10h] [rbp-60h]
  __int64 *v139; // [rsp+18h] [rbp-58h]
  __int64 v140; // [rsp+18h] [rbp-58h]
  __int64 v141; // [rsp+18h] [rbp-58h]
  __int64 v142; // [rsp+18h] [rbp-58h]
  _BYTE *v143; // [rsp+20h] [rbp-50h] BYREF
  __int64 v144; // [rsp+28h] [rbp-48h]
  _BYTE v145[64]; // [rsp+30h] [rbp-40h] BYREF

  if ( byte_4FA18E0 )
    sub_17157C0(a1, a2, *(_QWORD *)(a1 + 2632));
LABEL_3:
  v11 = *(_QWORD *)a1;
  v12 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  if ( !v12 )
    goto LABEL_48;
  do
  {
    v13 = *(_QWORD *)(*(_QWORD *)v11 + 8LL * v12 - 8);
    *(_DWORD *)(v11 + 8) = v12 - 1;
    v14 = *(_DWORD *)(v11 + 2088);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(v11 + 2072);
      v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v13 == *v18 )
      {
LABEL_6:
        *v18 = -16;
        --*(_DWORD *)(v11 + 2080);
        ++*(_DWORD *)(v11 + 2084);
      }
      else
      {
        v89 = 1;
        while ( v19 != -8 )
        {
          v90 = v89 + 1;
          v17 = v15 & (v89 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v13 == *v18 )
            goto LABEL_6;
          v89 = v90;
        }
      }
    }
    if ( !v13 )
      goto LABEL_3;
    if ( (unsigned __int8)sub_1AE9990(v13, *(_QWORD *)(a1 + 2648)) )
    {
LABEL_81:
      sub_170BC50(a1, v13);
      goto LABEL_47;
    }
    if ( !*(_QWORD *)(v13 + 8)
      || (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) != 0
      && ((*(_BYTE *)(v13 + 23) & 0x40) == 0
        ? (v20 = 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF), v25 = v13 - v20)
        : (v25 = *(_QWORD *)(v13 - 8)),
          *(_BYTE *)(*(_QWORD *)v25 + 16LL) > 0x10u)
      || (v26 = (__int64 ***)sub_14DD210((__int64 *)v13, *(_BYTE **)(a1 + 2664), *(_QWORD *)(a1 + 2648))) == 0 )
    {
      if ( !byte_4FA1C60 )
      {
        if ( byte_4FA1B80 )
        {
          v27 = *(_BYTE *)(v13 + 16);
          if ( v27 == 77 || v27 == 26 )
            goto LABEL_19;
          if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) != 0 )
          {
            v21 = 0;
            v116 = 0;
            v117 = 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
            do
            {
              v118 = v13 - v117;
              if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
                v118 = *(_QWORD *)(v13 - 8);
              v20 = *(_QWORD *)(v118 + v116);
              if ( *(_BYTE *)(v20 + 16) > 0x10u )
              {
                if ( v21 )
                {
                  if ( (_QWORD **)v20 != v21 )
                    goto LABEL_19;
                }
                else
                {
                  v21 = (_QWORD **)v20;
                }
              }
              v116 += 24;
            }
            while ( v116 != v117 );
          }
        }
        v69 = *(_QWORD *)(v13 + 8);
        if ( v69 )
        {
          if ( !*(_QWORD *)(v69 + 8) )
          {
            v70 = sub_1648700(*(_QWORD *)(v13 + 8));
            if ( *((_BYTE *)v70 + 16) == 77 )
              goto LABEL_87;
            while ( 1 )
            {
              v69 = *(_QWORD *)(v69 + 8);
              if ( !v69 )
                break;
              while ( 1 )
              {
                v70 = sub_1648700(v69);
                if ( *((_BYTE *)v70 + 16) != 77 )
                  break;
LABEL_87:
                if ( (unsigned __int8)sub_1B47380(v70) )
                  goto LABEL_19;
                v69 = *(_QWORD *)(v69 + 8);
                if ( !v69 )
                  goto LABEL_89;
              }
            }
LABEL_89:
            v71 = *(_QWORD *)(v13 + 8);
            v140 = *(_QWORD *)(v13 + 40);
            v72 = sub_1648700(v71);
            v28 = v140;
            if ( *((_BYTE *)v72 + 16) == 77 )
            {
              if ( (*((_BYTE *)v72 + 23) & 0x40) != 0 )
                v20 = *(v72 - 1);
              else
                v20 = (__int64)&v72[-3 * (*((_DWORD *)v72 + 5) & 0xFFFFFFF)];
              v141 = *(_QWORD *)(v20
                               + 0xFFFFFFFD55555558LL * (unsigned int)((v71 - v20) >> 3)
                               + 24LL * *((unsigned int *)v72 + 14)
                               + 8);
            }
            else
            {
              v141 = v72[5];
            }
            v29 = (__int64 *)(v13 + 24);
            if ( v141 == v28 )
              goto LABEL_21;
            v132 = v28;
            v73 = sub_157EBA0(v28);
            v28 = v132;
            v74 = v73;
            if ( !v73 )
              goto LABEL_21;
            v133 = sub_15F4D60(v73);
            if ( !v133 )
            {
LABEL_119:
              v28 = *(_QWORD *)(v13 + 40);
              goto LABEL_21;
            }
            v75 = 0;
            while ( v141 != sub_15F4DF0(v74, v75) )
            {
              if ( v133 == ++v75 )
                goto LABEL_19;
            }
            if ( sub_157F120(v141) )
            {
              v76 = *(unsigned __int8 *)(v13 + 16);
              v28 = *(_QWORD *)(v13 + 40);
              if ( (_BYTE)v76 == 77 )
                goto LABEL_20;
              v77 = (unsigned int)(v76 - 34);
              if ( (unsigned int)v77 <= 0x36 )
              {
                v78 = 0x40018000000001LL;
                if ( _bittest64(&v78, v77) )
                  goto LABEL_20;
              }
              v134 = *(_QWORD *)(v13 + 40);
              if ( !(unsigned __int8)sub_15F3040(v13) && !sub_15F3330(v13) )
              {
                v79 = *(unsigned __int8 *)(v13 + 16);
                if ( (unsigned int)(v79 - 25) > 9 )
                {
                  if ( (_BYTE)v79 == 53 )
                  {
                    v126 = *(_QWORD *)(v13 + 40);
                    v127 = *(_QWORD *)(*(_QWORD *)(v141 + 56) + 80LL);
                    if ( v127 )
                      v127 -= 24;
                    if ( v126 != v127 && *(_BYTE *)(sub_157EBA0(v141) + 16) != 34 )
                    {
                      v81 = v134;
                      goto LABEL_112;
                    }
                  }
                  else
                  {
                    v80 = sub_157EBA0(v141);
                    v81 = v134;
                    if ( *(_BYTE *)(v80 + 16) != 34 )
                    {
                      if ( (_BYTE)v79 != 78
                        || !(unsigned __int8)sub_1560260((_QWORD *)(v13 + 56), -1, 8)
                        && ((v82 = *(_QWORD *)(v13 - 24), v81 = v134, *(_BYTE *)(v82 + 16))
                         || (v143 = *(_BYTE **)(v82 + 112), v83 = sub_1560260(&v143, -1, 8), v81 = v134, !v83)) )
                      {
LABEL_112:
                        v135 = v81;
                        v29 = (__int64 *)(v13 + 24);
                        v84 = sub_15F2ED0(v13);
                        v85 = v135;
                        if ( v84 && (v86 = v13 + 24, v87 = *(_QWORD *)(v13 + 40) + 40LL, v29 != (__int64 *)v87) )
                        {
                          while ( 1 )
                          {
                            v88 = v86 - 24;
                            v130 = v85;
                            if ( !v86 )
                              v88 = 0;
                            v136 = v86;
                            if ( (unsigned __int8)sub_15F3040(v88) )
                              break;
                            v85 = v130;
                            v86 = *(_QWORD *)(v136 + 8);
                            if ( v87 == v86 )
                              goto LABEL_184;
                          }
                        }
                        else
                        {
LABEL_184:
                          v137 = v85;
                          v121 = sub_157EE30(v141);
                          v122 = v121;
                          if ( v121 )
                            v122 = v121 - 24;
                          sub_15F22F0((_QWORD *)v13, v122);
                          v143 = v145;
                          v144 = 0x100000000LL;
                          sub_1AEA440(&v143, v13);
                          v123 = v143;
                          v124 = v137;
                          v21 = (_QWORD **)&v143[8 * (unsigned int)v144];
                          v125 = (_QWORD **)v143;
                          if ( v143 != (_BYTE *)v21 )
                          {
                            do
                            {
                              if ( v124 == (*v125)[5] )
                              {
                                v129 = v124;
                                v131 = v125;
                                v138 = v21;
                                sub_15F22F0(*v125, v122);
                                v124 = v129;
                                v125 = v131;
                                v21 = v138;
                              }
                              ++v125;
                            }
                            while ( v21 != v125 );
                            v123 = v143;
                          }
                          if ( v123 != v145 )
                            _libc_free((unsigned __int64)v123);
                          *(_BYTE *)(a1 + 2728) = 1;
                          if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
                          {
                            v128 = *(__int64 **)(v13 - 8);
                            v20 = (__int64)&v128[3 * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)];
                          }
                          else
                          {
                            v20 = v13;
                            v128 = (__int64 *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
                          }
                          for ( ; (__int64 *)v20 != v128; v128 += 3 )
                          {
                            if ( *(_BYTE *)(*v128 + 16) > 0x17u )
                            {
                              v142 = v20;
                              sub_170B990(*(_QWORD *)a1, *v128);
                              v20 = v142;
                            }
                          }
                        }
                        goto LABEL_119;
                      }
                      v28 = *(_QWORD *)(v13 + 40);
                      v29 = (__int64 *)(v13 + 24);
LABEL_21:
                      v30 = *(__int64 **)(a1 + 8);
                      v139 = v29;
                      v30[1] = v28;
                      v30[2] = (__int64)v29;
                      v31 = *(_QWORD *)(v13 + 48);
                      v143 = (_BYTE *)v31;
                      if ( v31 )
                      {
                        sub_1623A60((__int64)&v143, v31, 2);
                        v32 = *v30;
                        if ( !*v30 )
                          goto LABEL_24;
                      }
                      else
                      {
                        v32 = *v30;
                        if ( !*v30 )
                          goto LABEL_26;
                      }
                      sub_161E7C0((__int64)v30, v32);
LABEL_24:
                      v33 = v143;
                      *v30 = (__int64)v143;
                      if ( v33 )
                      {
                        sub_1623210((__int64)&v143, v33, (__int64)v30);
                      }
                      else if ( v143 )
                      {
                        sub_161E7C0((__int64)&v143, (__int64)v143);
                      }
LABEL_26:
                      v34 = *(_QWORD *)(v13 + 48);
                      v35 = *(__int64 **)(a1 + 8);
                      v143 = (_BYTE *)v34;
                      if ( v34 )
                      {
                        sub_1623A60((__int64)&v143, v34, 2);
                        if ( v35 != (__int64 *)&v143 )
                        {
                          v36 = *v35;
                          if ( !*v35 )
                            goto LABEL_30;
                          goto LABEL_29;
                        }
                      }
                      else
                      {
                        if ( v35 == (__int64 *)&v143 )
                          goto LABEL_32;
                        v36 = *v35;
                        if ( *v35 )
                        {
LABEL_29:
                          sub_161E7C0((__int64)v35, v36);
LABEL_30:
                          v37 = v143;
                          *v35 = (__int64)v143;
                          if ( v37 )
                          {
                            sub_1623210((__int64)&v143, v37, (__int64)v35);
                            goto LABEL_32;
                          }
                        }
                      }
                      if ( v143 )
                        sub_161E7C0((__int64)&v143, (__int64)v143);
LABEL_32:
                      v38 = sub_17153B0(
                              (__m128i *)a1,
                              v13,
                              v20,
                              v28,
                              (int)v21,
                              v22,
                              *(double *)a3.m128_u64,
                              a4,
                              a5,
                              a6,
                              v23,
                              v24,
                              a9,
                              a10);
                      v41 = v38;
                      if ( !v38 )
                        goto LABEL_3;
                      if ( v38 == v13 )
                      {
                        if ( (unsigned __int8)sub_1AE9990(v13, *(_QWORD *)(a1 + 2648)) )
                          goto LABEL_81;
                        v113 = *(_QWORD *)(v13 + 8);
                        v114 = *(_QWORD *)a1;
                        if ( v113 )
                        {
                          do
                          {
                            v115 = sub_1648700(v113);
                            sub_170B990(v114, (__int64)v115);
                            v113 = *(_QWORD *)(v113 + 8);
                          }
                          while ( v113 );
                          v114 = *(_QWORD *)a1;
                        }
                        sub_170B990(v114, v13);
                      }
                      else
                      {
                        v42 = *(_QWORD *)(v13 + 48);
                        if ( v42 )
                        {
                          v43 = (__int64 *)(v38 + 48);
                          v143 = *(_BYTE **)(v13 + 48);
                          sub_1623A60((__int64)&v143, v42, 2);
                          if ( v43 == (__int64 *)&v143 )
                          {
                            if ( v143 )
                              sub_161E7C0((__int64)&v143, (__int64)v143);
                          }
                          else
                          {
                            v44 = *(_QWORD *)(v41 + 48);
                            if ( v44 )
                              sub_161E7C0(v41 + 48, v44);
                            v45 = v143;
                            *(_QWORD *)(v41 + 48) = v143;
                            if ( v45 )
                              sub_1623210((__int64)&v143, v45, v41 + 48);
                          }
                        }
                        sub_164D160(v13, v41, a3, a4, a5, a6, v39, v40, *(double *)a9.m128i_i64, a10);
                        sub_164B7C0(v41, v13);
                        v46 = *(_QWORD *)(v41 + 8);
                        v47 = *(_QWORD *)a1;
                        if ( v46 )
                        {
                          do
                          {
                            v48 = sub_1648700(v46);
                            sub_170B990(v47, (__int64)v48);
                            v46 = *(_QWORD *)(v46 + 8);
                          }
                          while ( v46 );
                          v47 = *(_QWORD *)a1;
                        }
                        sub_170B990(v47, v41);
                        v49 = *(_QWORD *)(v13 + 40);
                        if ( *(_BYTE *)(v41 + 16) != 77 && *(_BYTE *)(v13 + 16) == 77 )
                          v139 = (__int64 *)sub_157EE30(*(_QWORD *)(v13 + 40));
                        sub_157E9D0(v49 + 40, v41);
                        v50 = *v139;
                        v51 = *(_QWORD *)(v41 + 24) & 7LL;
                        *(_QWORD *)(v41 + 32) = v139;
                        v50 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(v41 + 24) = v50 | v51;
                        *(_QWORD *)(v50 + 8) = v41 + 24;
                        *v139 = *v139 & 7 | (v41 + 24);
                        sub_170BC50(a1, v13);
                      }
                      goto LABEL_47;
                    }
                    v126 = *(_QWORD *)(v13 + 40);
                  }
                  v28 = v126;
                  v29 = (__int64 *)(v13 + 24);
                  goto LABEL_21;
                }
              }
            }
          }
        }
      }
LABEL_19:
      v28 = *(_QWORD *)(v13 + 40);
LABEL_20:
      v29 = (__int64 *)(v13 + 24);
      goto LABEL_21;
    }
    v64 = *(_QWORD *)(v13 + 8);
    if ( v64 )
    {
      v65 = *(_QWORD *)a1;
      do
      {
        v66 = sub_1648700(v64);
        sub_170B990(v65, (__int64)v66);
        v64 = *(_QWORD *)(v64 + 8);
      }
      while ( v64 );
      if ( v26 == (__int64 ***)v13 )
        v26 = (__int64 ***)sub_1599EF0(*v26);
      sub_164D160(v13, (__int64)v26, a3, a4, a5, a6, v67, v68, *(double *)a9.m128i_i64, a10);
    }
    if ( (unsigned __int8)sub_1AE9990(v13, *(_QWORD *)(a1 + 2648)) )
      goto LABEL_81;
LABEL_47:
    v11 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 2728) = 1;
    v12 = *(_DWORD *)(v11 + 8);
  }
  while ( v12 );
LABEL_48:
  v52 = *(_DWORD *)(v11 + 2080);
  ++*(_QWORD *)(v11 + 2064);
  if ( v52 )
  {
    v102 = 4 * v52;
    v53 = *(unsigned int *)(v11 + 2088);
    if ( (unsigned int)(4 * v52) < 0x40 )
      v102 = 64;
    if ( (unsigned int)v53 <= v102 )
    {
LABEL_51:
      v54 = *(_QWORD **)(v11 + 2072);
      for ( i = &v54[2 * v53]; i != v54; v54 += 2 )
        *v54 = -8;
      *(_QWORD *)(v11 + 2080) = 0;
      goto LABEL_54;
    }
    v103 = *(_QWORD **)(v11 + 2072);
    v104 = v52 - 1;
    if ( v104 )
    {
      _BitScanReverse(&v104, v104);
      v105 = 1 << (33 - (v104 ^ 0x1F));
      if ( v105 < 64 )
        v105 = 64;
      if ( (_DWORD)v53 == v105 )
      {
        *(_QWORD *)(v11 + 2080) = 0;
        v120 = &v103[2 * (unsigned int)v53];
        do
        {
          if ( v103 )
            *v103 = -8;
          v103 += 2;
        }
        while ( v120 != v103 );
        goto LABEL_54;
      }
      v106 = (4 * v105 / 3u + 1) | ((unsigned __int64)(4 * v105 / 3u + 1) >> 1);
      v107 = ((v106 | (v106 >> 2)) >> 4)
           | v106
           | (v106 >> 2)
           | ((((v106 | (v106 >> 2)) >> 4) | v106 | (v106 >> 2)) >> 8);
      v108 = (v107 | (v107 >> 16)) + 1;
      v109 = 16 * ((v107 | (v107 >> 16)) + 1);
    }
    else
    {
      v109 = 2048;
      v108 = 128;
    }
    j___libc_free_0(v103);
    *(_DWORD *)(v11 + 2088) = v108;
    v110 = (_QWORD *)sub_22077B0(v109);
    v111 = *(unsigned int *)(v11 + 2088);
    *(_QWORD *)(v11 + 2080) = 0;
    *(_QWORD *)(v11 + 2072) = v110;
    for ( j = &v110[2 * v111]; j != v110; v110 += 2 )
    {
      if ( v110 )
        *v110 = -8;
    }
    goto LABEL_54;
  }
  if ( *(_DWORD *)(v11 + 2084) )
  {
    v53 = *(unsigned int *)(v11 + 2088);
    if ( (unsigned int)v53 <= 0x40 )
      goto LABEL_51;
    j___libc_free_0(*(_QWORD *)(v11 + 2072));
    *(_QWORD *)(v11 + 2072) = 0;
    *(_QWORD *)(v11 + 2080) = 0;
    *(_DWORD *)(v11 + 2088) = 0;
  }
LABEL_54:
  v56 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  if ( v56 )
  {
    v91 = 4 * v56;
    v57 = *(unsigned int *)(a1 + 48);
    if ( (unsigned int)(4 * v56) < 0x40 )
      v91 = 64;
    if ( (unsigned int)v57 <= v91 )
    {
LABEL_57:
      v58 = *(_QWORD **)(a1 + 32);
      for ( k = &v58[v57]; k != v58; ++v58 )
        *v58 = -8;
      *(_QWORD *)(a1 + 40) = 0;
      goto LABEL_60;
    }
    v92 = *(_QWORD **)(a1 + 32);
    v93 = v56 - 1;
    if ( v93 )
    {
      _BitScanReverse(&v93, v93);
      v94 = 1 << (33 - (v93 ^ 0x1F));
      if ( v94 < 64 )
        v94 = 64;
      if ( (_DWORD)v57 == v94 )
      {
        *(_QWORD *)(a1 + 40) = 0;
        v119 = &v92[v57];
        do
        {
          if ( v92 )
            *v92 = -8;
          ++v92;
        }
        while ( v119 != v92 );
        goto LABEL_60;
      }
      v95 = (4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1);
      v96 = ((v95 | (v95 >> 2)) >> 4) | v95 | (v95 >> 2) | ((((v95 | (v95 >> 2)) >> 4) | v95 | (v95 >> 2)) >> 8);
      v97 = (v96 | (v96 >> 16)) + 1;
      v98 = 8 * ((v96 | (v96 >> 16)) + 1);
    }
    else
    {
      v98 = 1024;
      v97 = 128;
    }
    j___libc_free_0(v92);
    *(_DWORD *)(a1 + 48) = v97;
    v99 = (_QWORD *)sub_22077B0(v98);
    v100 = *(unsigned int *)(a1 + 48);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = v99;
    for ( m = &v99[v100]; m != v99; ++v99 )
    {
      if ( v99 )
        *v99 = -8;
    }
    goto LABEL_60;
  }
  if ( *(_DWORD *)(a1 + 44) )
  {
    v57 = *(unsigned int *)(a1 + 48);
    if ( (unsigned int)v57 <= 0x40 )
      goto LABEL_57;
    j___libc_free_0(*(_QWORD *)(a1 + 32));
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_DWORD *)(a1 + 48) = 0;
  }
LABEL_60:
  v60 = *(_QWORD **)(a1 + 56);
  v61 = &v60[5 * *(unsigned int *)(a1 + 64)];
  while ( v60 != v61 )
  {
    v62 = *(v61 - 2);
    v61 -= 5;
    *v61 = &unk_49EE2B0;
    if ( v62 != -8 && v62 != 0 && v62 != -16 )
      sub_1649B30(v61 + 1);
  }
  result = *(unsigned __int8 *)(a1 + 2728);
  *(_DWORD *)(a1 + 64) = 0;
  return result;
}
