// Function: sub_3279770
// Address: 0x3279770
//
__int64 __fastcall sub_3279770(__int64 *a1, __int64 a2, char a3, __int64 a4)
{
  unsigned int v4; // eax
  __int64 (__fastcall *v5)(__int64, char, int, int, int, int, int); // r12
  unsigned int v7; // ebx
  bool v8; // al
  __int64 v10; // r13
  _QWORD *v11; // rbx
  unsigned __int16 *v12; // rax
  unsigned int v13; // r14d
  int v14; // esi
  unsigned int v15; // esi
  unsigned __int16 v16; // ax
  __int64 v17; // rdx
  __int64 v18; // rsi
  _QWORD *v19; // rdi
  _QWORD *v20; // r9
  __int64 (*v21)(); // rax
  __int64 v22; // rdx
  __int64 (__fastcall *v23)(__int64, __int64, unsigned int); // rax
  __int16 *v24; // rcx
  unsigned int *v25; // r11
  unsigned __int16 v26; // r10
  __int64 (__fastcall *v27)(__int64, unsigned __int16); // r8
  __int64 (*v28)(); // r9
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // ebx
  __int64 *v34; // r15
  __m128i *v35; // rdi
  __int64 v36; // r13
  __m128i *v37; // r12
  unsigned __int64 v38; // rax
  __m128i *v39; // r13
  const __m128i *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rbx
  _QWORD *v44; // r13
  int v45; // r14d
  unsigned int v46; // r14d
  unsigned __int16 v47; // ax
  __int64 v48; // rdx
  unsigned __int16 v49; // r13
  int v50; // esi
  unsigned int v51; // esi
  __int16 v52; // ax
  __int64 v53; // rdx
  __int64 (*v54)(); // rax
  int v55; // eax
  unsigned __int16 v56; // ax
  char v57; // r14
  __int64 v58; // r13
  unsigned __int16 *v59; // rdx
  unsigned __int16 v60; // ax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // r13
  int v66; // r12d
  int v67; // eax
  __int64 (__fastcall *v68)(__int64, char, int, int, int, int, int); // r10
  int v69; // eax
  unsigned int v70; // ebx
  unsigned int v71; // eax
  __int64 v72; // r11
  unsigned __int16 *v73; // rax
  unsigned __int16 v74; // di
  __int64 (__fastcall *v75)(__int64, unsigned __int16); // rax
  unsigned int *v76; // rdx
  __int64 v77; // rsi
  unsigned __int16 *v78; // rdx
  __int64 v79; // rsi
  __int64 v80; // rsi
  __int64 v81; // rdi
  __int64 v82; // rdi
  __int64 v83; // rdi
  char *v84; // rax
  __int64 v85; // r11
  __int64 v86; // rsi
  unsigned __int16 *v87; // rax
  unsigned __int16 v88; // dx
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  unsigned __int64 v92; // rax
  __int64 v93; // r11
  unsigned __int64 v94; // rax
  unsigned __int64 v95; // rdx
  char v96; // dl
  bool v97; // zf
  char v98; // al
  __int64 v99; // rax
  __int64 v100; // r8
  int v101; // eax
  unsigned int v102; // eax
  __int16 v103; // ax
  __int64 v104; // rdx
  __int64 v105; // rcx
  char v106; // al
  __int64 v107; // r14
  __int64 v108; // r14
  unsigned int v109; // eax
  char v110; // al
  __int64 v111; // rax
  __int64 v112; // rax
  unsigned int v113; // eax
  __int64 v114; // rax
  char v115; // cl
  unsigned __int64 v116; // rax
  __int16 v117; // ax
  unsigned int v118; // [rsp+8h] [rbp-108h]
  __int64 v119; // [rsp+10h] [rbp-100h]
  __int64 v120; // [rsp+10h] [rbp-100h]
  __int64 v121; // [rsp+10h] [rbp-100h]
  char v122; // [rsp+18h] [rbp-F8h]
  unsigned int v123; // [rsp+18h] [rbp-F8h]
  __int64 v124; // [rsp+18h] [rbp-F8h]
  unsigned int v125; // [rsp+28h] [rbp-E8h]
  int v126; // [rsp+30h] [rbp-E0h]
  __int64 v127; // [rsp+30h] [rbp-E0h]
  unsigned __int8 (__fastcall *v128)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, int *, __int64); // [rsp+30h] [rbp-E0h]
  unsigned int v129; // [rsp+30h] [rbp-E0h]
  __int64 v130; // [rsp+38h] [rbp-D8h]
  __int64 v131; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v132; // [rsp+38h] [rbp-D8h]
  __int64 (*v133)(); // [rsp+38h] [rbp-D8h]
  __int64 v134; // [rsp+38h] [rbp-D8h]
  __int64 v135; // [rsp+40h] [rbp-D0h]
  unsigned int v136; // [rsp+40h] [rbp-D0h]
  __int64 v137; // [rsp+48h] [rbp-C8h]
  int v138; // [rsp+48h] [rbp-C8h]
  char v139; // [rsp+48h] [rbp-C8h]
  unsigned int v140; // [rsp+48h] [rbp-C8h]
  __int64 v141; // [rsp+48h] [rbp-C8h]
  __int64 v142; // [rsp+48h] [rbp-C8h]
  __int64 v143; // [rsp+50h] [rbp-C0h]
  _QWORD *v144; // [rsp+50h] [rbp-C0h]
  unsigned __int8 v145; // [rsp+50h] [rbp-C0h]
  unsigned int v146; // [rsp+50h] [rbp-C0h]
  _QWORD *v147; // [rsp+58h] [rbp-B8h]
  __int64 v148; // [rsp+58h] [rbp-B8h]
  int v149; // [rsp+58h] [rbp-B8h]
  __int64 v150; // [rsp+60h] [rbp-B0h]
  __int64 v151; // [rsp+60h] [rbp-B0h]
  __int16 src; // [rsp+68h] [rbp-A8h]
  char *srca; // [rsp+68h] [rbp-A8h]
  unsigned __int16 srcb; // [rsp+68h] [rbp-A8h]
  _QWORD *srcc; // [rsp+68h] [rbp-A8h]
  int v157; // [rsp+78h] [rbp-98h]
  int v159; // [rsp+80h] [rbp-90h]
  int v160; // [rsp+84h] [rbp-8Ch]
  int v161; // [rsp+88h] [rbp-88h]
  unsigned __int8 v162; // [rsp+9Bh] [rbp-75h] BYREF
  int v163; // [rsp+9Ch] [rbp-74h] BYREF
  unsigned __int64 v164; // [rsp+A0h] [rbp-70h]
  __int64 v165; // [rsp+A8h] [rbp-68h]
  __int64 v166; // [rsp+B0h] [rbp-60h]
  __int64 v167; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v168; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v169; // [rsp+C8h] [rbp-48h]
  unsigned __int64 v170; // [rsp+D0h] [rbp-40h] BYREF
  __int64 v171; // [rsp+D8h] [rbp-38h]

  v4 = *((_DWORD *)a1 + 2);
  LODWORD(v5) = (unsigned __int8)qword_5038328;
  if ( (_BYTE)qword_5038328 )
  {
    LOBYTE(v5) = v4 > 1;
    return (unsigned int)v5;
  }
  if ( v4 == 2 )
  {
    v7 = *(_DWORD *)(a2 + 8);
    if ( !v7
      || (v7 <= 0x40
        ? (v8 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == *(_QWORD *)a2)
        : (v8 = v7 == (unsigned int)sub_C445E0(a2)),
          v8 || (unsigned __int8)sub_3262120(a2)) )
    {
      v157 = 0;
      v10 = 0;
      v161 = 0;
      v159 = 0;
      v160 = 0;
      while ( 1 )
      {
        v11 = (_QWORD *)(*a1 + 32 * v10);
        v12 = *(unsigned __int16 **)(*v11 + 48LL);
        v143 = *((_QWORD *)v12 + 1);
        v13 = *v12;
        src = *v12;
        v147 = *(_QWORD **)(v11[3] + 64LL);
        sub_3266230((__int64)&v170, (__int64)v11);
        if ( (unsigned int)v171 > 0x40 )
        {
          v55 = sub_C44630((__int64)&v170);
          v14 = v55;
          if ( v170 )
          {
            v138 = v55;
            j_j___libc_free_0_0(v170);
            v14 = v138;
          }
        }
        else
        {
          v14 = sub_39FAC40(v170);
        }
        v15 = v14 & 0xFFFFFFF8;
        switch ( v15 )
        {
          case 8u:
            v56 = 5;
            break;
          case 0x10u:
            v56 = 6;
            break;
          case 0x20u:
            v56 = 7;
            break;
          case 0x40u:
            v56 = 8;
            break;
          case 0x80u:
            v56 = 9;
            break;
          default:
            v16 = sub_3007020(v147, v15);
            v18 = v16;
            v19 = *(_QWORD **)(v11[3] + 16LL);
            v20 = (_QWORD *)*v19;
            if ( src == v16 && (src || v143 == v17) )
              goto LABEL_22;
            goto LABEL_20;
        }
        v17 = 0;
        v18 = v56;
        v19 = *(_QWORD **)(v11[3] + 16LL);
        v20 = (_QWORD *)*v19;
        if ( src == v56 )
          goto LABEL_22;
LABEL_20:
        v21 = (__int64 (*)())v20[179];
        if ( v21 == sub_2FE34A0
          || (v110 = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, __int64))v21)(
                       v19,
                       v18,
                       v17,
                       v13,
                       v143),
              v19 = *(_QWORD **)(v11[3] + 16LL),
              v20 = (_QWORD *)*v19,
              !v110) )
        {
          ++v157;
        }
LABEL_22:
        v22 = *v11;
        v23 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v20[176];
        v24 = *(__int16 **)(*v11 + 48LL);
        v25 = *(unsigned int **)(*v11 + 40LL);
        v26 = *v24;
        v27 = (__int64 (__fastcall *)(__int64, unsigned __int16))*((_QWORD *)v24 + 1);
        if ( v23 != sub_2FE3A30 )
        {
          v105 = v135;
          LOWORD(v105) = v26;
          v135 = v105;
          v106 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD, __int64, __int64 (__fastcall *)(__int64, unsigned __int16)))v23)(
                   v19,
                   *(_QWORD *)v25,
                   *((_QWORD *)v25 + 1),
                   v105,
                   v27);
          goto LABEL_145;
        }
        v28 = (__int64 (*)())v20[174];
        v29 = v26;
        v30 = *(_QWORD *)(*(_QWORD *)v25 + 48LL) + 16LL * v25[2];
        v31 = v150;
        LOWORD(v31) = *(_WORD *)v30;
        v150 = v31;
        if ( v28 != sub_2FE3480 )
        {
          v106 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD, _QWORD, __int64 (__fastcall *)(__int64, unsigned __int16)))v28)(
                   v19,
                   (unsigned int)v31,
                   *(_QWORD *)(v30 + 8),
                   v26,
                   v27);
LABEL_145:
          v22 = *v11;
          if ( v106 )
            goto LABEL_25;
        }
        ++v160;
LABEL_25:
        v161 -= (*((_DWORD *)v11 + 4) == 0) - 1;
        if ( v22 )
        {
          v32 = *(_QWORD *)(v22 + 56);
          if ( v32 )
          {
            if ( !*(_QWORD *)(v32 + 32) )
            {
              v29 = *(_QWORD *)(v32 + 16);
              if ( *(_DWORD *)(v29 + 24) == 234 )
              {
                v27 = sub_2EC09E0;
                v72 = *(_QWORD *)(v11[3] + 16LL);
                v73 = *(unsigned __int16 **)(v29 + 48);
                v74 = *v73;
                v148 = *((_QWORD *)v73 + 1);
                srcb = *v73;
                v146 = *v73;
                v75 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v72 + 552LL);
                if ( v75 == sub_2EC09E0 )
                {
                  v28 = *(__int64 (**)())(v72 + 8LL * v74 + 112);
                }
                else
                {
                  v134 = v29;
                  v142 = *(_QWORD *)(v11[3] + 16LL);
                  v112 = ((__int64 (__fastcall *)(__int64, _QWORD, bool))v75)(
                           v72,
                           srcb,
                           (*(_BYTE *)(v29 + 32) & 4) != 0);
                  v72 = v142;
                  v27 = sub_2EC09E0;
                  v28 = (__int64 (*)())v112;
                  v29 = v134;
                  v75 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v142 + 552LL);
                }
                v76 = *(unsigned int **)(v29 + 40);
                v77 = *(_QWORD *)v76;
                v78 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v76 + 48LL) + 16LL * v76[2]);
                v29 = *(unsigned __int8 *)(v77 + 32);
                v79 = *v78;
                if ( v75 == sub_2EC09E0 )
                {
                  v80 = *(_QWORD *)(v72 + 8 * v79 + 112);
                }
                else
                {
                  v133 = v28;
                  v141 = v72;
                  v111 = ((__int64 (__fastcall *)(__int64, __int64, bool))v75)(v72, v79, (v29 & 4) != 0);
                  v28 = v133;
                  v72 = v141;
                  v80 = v111;
                }
                if ( v28 != (__int64 (*)())v80 )
                {
                  if ( srcb == 1 )
                  {
                    v140 = 1;
                  }
                  else
                  {
                    if ( !srcb )
                      goto LABEL_27;
                    v140 = srcb;
                    if ( !*(_QWORD *)(v72 + 8LL * srcb + 112) )
                      goto LABEL_27;
                  }
                  v127 = (__int64)v28;
                  v131 = v72;
                  if ( !*(_BYTE *)(v72 + 500LL * v140 + 6712) )
                  {
                    v81 = *(_QWORD *)(*(_QWORD *)(v11[3] + 40LL) + 16LL);
                    v82 = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int16 *, __int64))(*(_QWORD *)v81 + 200LL))(
                            v81,
                            v80,
                            v78,
                            v29);
                    if ( v82 )
                    {
                      if ( !sub_2FF6970(v82, v80, v127) )
                      {
                        v163 = 0;
                        v119 = v131;
                        v128 = *(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, int *, __int64))(*(_QWORD *)v131 + 824LL);
                        v83 = *(_QWORD *)(v11[1] + 112LL);
                        v125 = *(unsigned __int16 *)(v83 + 32);
                        v132 = sub_2EAC4F0(v83);
                        v84 = (char *)sub_2E79000(*(__int64 **)(v11[3] + 40LL));
                        v85 = v119;
                        v86 = *((_DWORD *)v11 + 4) >> 3;
                        v122 = *v84;
                        v87 = *(unsigned __int16 **)(v11[1] + 48LL);
                        v88 = *v87;
                        v89 = *((_QWORD *)v87 + 1);
                        LOWORD(v168) = v88;
                        v169 = v89;
                        if ( v88 )
                        {
                          if ( v88 == 1 || (unsigned __int16)(v88 - 504) <= 7u )
LABEL_192:
                            BUG();
                          v114 = 16LL * (v88 - 1);
                          v115 = byte_444C4A0[v114 + 8];
                          v116 = *(_QWORD *)&byte_444C4A0[v114];
                          LOBYTE(v165) = v115;
                          v164 = v116;
                        }
                        else
                        {
                          v90 = sub_3007260((__int64)&v168);
                          v85 = v119;
                          v164 = v90;
                          v165 = v91;
                        }
                        v120 = v85;
                        v170 = v164;
                        LOBYTE(v171) = v165;
                        v92 = sub_CA1930(&v170);
                        v93 = v120;
                        if ( v122 )
                        {
                          v124 = (unsigned int)(v92 >> 3) - v86;
                          sub_3266230((__int64)&v170, (__int64)v11);
                          if ( (unsigned int)v171 > 0x40 )
                          {
                            v113 = sub_C44630((__int64)&v170);
                            v93 = v120;
                            if ( v170 )
                            {
                              v118 = v113;
                              j_j___libc_free_0_0(v170);
                              v113 = v118;
                              v93 = v120;
                            }
                          }
                          else
                          {
                            v113 = sub_39FAC40(v170);
                            v93 = v120;
                          }
                          v86 = v124 - (v113 >> 3);
                        }
                        if ( v86 )
                        {
                          v94 = -((1LL << v132) | (v86 + (1LL << v132))) & ((1LL << v132) | (v86 + (1LL << v132)));
                          _BitScanReverse64(&v95, v94);
                          v96 = v95 ^ 0x3F;
                          v97 = v94 == 0;
                          v98 = 64;
                          if ( !v97 )
                            v98 = v96;
                          v132 = 63 - v98;
                        }
                        v121 = v93;
                        v123 = sub_2EAC1E0(*(_QWORD *)(v11[1] + 112LL));
                        v99 = sub_2E79000(*(__int64 **)(v11[3] + 40LL));
                        if ( v128(v121, *(_QWORD *)(v11[3] + 64LL), v99, v146, v148, v123, v132, v125, &v163, v100) )
                        {
                          if ( v163
                            && (srcb == 1 || *(_QWORD *)(v121 + 8LL * (int)v140 + 112))
                            && !*(_BYTE *)(v121 + 500LL * v140 + 6712) )
                          {
                            srcc = *(_QWORD **)(v11[3] + 64LL);
                            sub_3266230((__int64)&v170, (__int64)v11);
                            if ( (unsigned int)v171 > 0x40 )
                            {
                              v101 = sub_C44630((__int64)&v170);
                              if ( v170 )
                              {
                                v149 = v101;
                                j_j___libc_free_0_0(v170);
                                v101 = v149;
                              }
                            }
                            else
                            {
                              v101 = sub_39FAC40(v170);
                            }
                            v102 = v101 & 0xFFFFFFF8;
                            switch ( v102 )
                            {
                              case 8u:
                                v117 = 5;
                                goto LABEL_186;
                              case 0x10u:
                                v117 = 6;
                                goto LABEL_186;
                              case 0x20u:
                                v117 = 7;
                                goto LABEL_186;
                              case 0x40u:
                                v117 = 8;
LABEL_186:
                                if ( **(_WORD **)(*v11 + 48LL) == v117 )
                                  goto LABEL_140;
                                goto LABEL_27;
                              case 0x80u:
                                v117 = 9;
                                goto LABEL_186;
                            }
                            v103 = sub_3007020(srcc, v102);
                            v29 = *(_QWORD *)(*v11 + 48LL);
                            if ( *(_WORD *)v29 == v103 )
                            {
                              v29 = *(_QWORD *)(v29 + 8);
                              if ( v103 || v29 == v104 )
LABEL_140:
                                ++v159;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
LABEL_27:
        if ( v10 == 1 )
        {
          v33 = *((_DWORD *)a1 + 2);
          v136 = 2;
          v34 = a1;
          if ( v33 > 1 )
          {
            v35 = (__m128i *)*a1;
            v36 = 32LL * v33;
            v37 = (__m128i *)(*a1 + v36);
            _BitScanReverse64(&v38, v36 >> 5);
            sub_3274500(v35, v37, 2LL * (int)(63 - (v38 ^ 0x3F)), v29, (__int64)v27, (__int64)v28);
            if ( (unsigned __int64)v36 <= 0x200 )
            {
              sub_3266760(v35, v37);
            }
            else
            {
              v39 = v35 + 32;
              sub_3266760(v35, v35 + 32);
              if ( v37 != &v35[32] )
              {
                do
                {
                  v40 = v39;
                  v39 += 2;
                  sub_32664A0(v40);
                }
                while ( v37 != v39 );
              }
            }
            v41 = *v34;
            v42 = 0;
            srca = 0;
            v130 = *(_QWORD *)(*(_QWORD *)(*v34 + 24) + 16LL);
            v151 = 32LL * (v33 - 1);
            while ( 1 )
            {
              v43 = (__int64)&srca[v41];
              if ( !v42 )
                goto LABEL_54;
              v44 = *(_QWORD **)(*(_QWORD *)(v42 + 24) + 64LL);
              sub_3266230((__int64)&v170, v42);
              if ( (unsigned int)v171 > 0x40 )
              {
                v45 = sub_C44630((__int64)&v170);
                if ( v170 )
                  j_j___libc_free_0_0(v170);
              }
              else
              {
                v45 = sub_39FAC40(v170);
              }
              v46 = v45 & 0xFFFFFFF8;
              switch ( v46 )
              {
                case 8u:
                  v49 = 5;
                  break;
                case 0x10u:
                  v49 = 6;
                  break;
                case 0x20u:
                  v49 = 7;
                  break;
                case 0x40u:
                  v49 = 8;
                  break;
                case 0x80u:
                  v49 = 9;
                  break;
                default:
                  v47 = sub_3007020(v44, v46);
                  v137 = v48;
                  v49 = v47;
                  goto LABEL_42;
              }
              v137 = 0;
LABEL_42:
              v144 = *(_QWORD **)(*(_QWORD *)(v43 + 24) + 64LL);
              sub_3266230((__int64)&v170, v43);
              if ( (unsigned int)v171 > 0x40 )
              {
                v67 = sub_C44630((__int64)&v170);
                v50 = v67;
                if ( v170 )
                {
                  v126 = v67;
                  j_j___libc_free_0_0(v170);
                  v50 = v126;
                }
              }
              else
              {
                v50 = sub_39FAC40(v170);
              }
              v51 = v50 & 0xFFFFFFF8;
              switch ( v51 )
              {
                case 8u:
                  v52 = 5;
                  v53 = 0;
                  break;
                case 0x10u:
                  v52 = 6;
                  v53 = 0;
                  break;
                case 0x20u:
                  v52 = 7;
                  v53 = 0;
                  break;
                case 0x40u:
                  v52 = 8;
                  v53 = 0;
                  break;
                case 0x80u:
                  v52 = 9;
                  v53 = 0;
                  break;
                default:
                  v52 = sub_3007020(v144, v51);
                  break;
              }
              if ( v52 != v49 || !v52 && v53 != v137 )
                goto LABEL_54;
              v162 = 0;
              v54 = *(__int64 (**)())(*(_QWORD *)v130 + 1480LL);
              if ( v54 == sub_2FE34C0 )
                goto LABEL_53;
              v139 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, unsigned __int8 *))v54)(v130, v49, v137, &v162);
              if ( !v139 )
                goto LABEL_53;
              v145 = sub_2EAC4F0(*(_QWORD *)(*(_QWORD *)(v42 + 8) + 112LL));
              v57 = *(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(v42 + 24) + 40LL));
              v58 = *(_DWORD *)(v42 + 16) >> 3;
              v59 = *(unsigned __int16 **)(*(_QWORD *)(v42 + 8) + 48LL);
              v60 = *v59;
              v61 = *((_QWORD *)v59 + 1);
              LOWORD(v170) = v60;
              v171 = v61;
              if ( v60 )
              {
                if ( v60 == 1 || (unsigned __int16)(v60 - 504) <= 7u )
                  goto LABEL_192;
                v63 = 16LL * (v60 - 1);
                v62 = *(_QWORD *)&byte_444C4A0[v63];
                LOBYTE(v63) = byte_444C4A0[v63 + 8];
              }
              else
              {
                v62 = sub_3007260((__int64)&v170);
                v166 = v62;
                v167 = v63;
              }
              v170 = v62;
              LOBYTE(v171) = v63;
              v64 = sub_CA1930(&v170);
              if ( v57 )
              {
                v107 = (unsigned int)(v64 >> 3);
                sub_3266230((__int64)&v170, v42);
                v108 = v107 - v58;
                if ( (unsigned int)v171 > 0x40 )
                {
                  v109 = sub_C44630((__int64)&v170);
                  if ( v170 )
                  {
                    v129 = v109;
                    j_j___libc_free_0_0(v170);
                    v109 = v129;
                  }
                }
                else
                {
                  v109 = sub_39FAC40(v170);
                }
                v58 = v108 - (v109 >> 3);
              }
              if ( v58 )
              {
                v65 = -((1LL << v145) | ((1LL << v145) + v58)) & ((1LL << v145) | ((1LL << v145) + v58));
                if ( !v65 )
                  goto LABEL_70;
                _BitScanReverse64(&v65, v65);
                v145 = 63 - (v65 ^ 0x3F);
              }
              if ( v162 > v145 )
                goto LABEL_54;
LABEL_70:
              sub_3266230((__int64)&v168, v42);
              sub_3266230((__int64)&v170, v43);
              if ( (unsigned int)v169 > 0x40 )
                sub_C43BD0(&v168, (__int64 *)&v170);
              else
                v168 |= v170;
              if ( (unsigned int)v171 > 0x40 && v170 )
                j_j___libc_free_0_0(v170);
              v66 = v169;
              if ( !(_DWORD)v169 )
              {
LABEL_82:
                --v136;
LABEL_53:
                v43 = 0;
                goto LABEL_54;
              }
              if ( (unsigned int)v169 <= 0x40 )
              {
                if ( v168 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v169) )
                  goto LABEL_82;
LABEL_78:
                v139 = sub_3262120((__int64)&v168);
                if ( (unsigned int)v169 <= 0x40 )
                  goto LABEL_81;
                goto LABEL_79;
              }
              if ( v66 != (unsigned int)sub_C445E0((__int64)&v168) )
                goto LABEL_78;
LABEL_79:
              if ( v168 )
                j_j___libc_free_0_0(v168);
LABEL_81:
              if ( v139 )
                goto LABEL_82;
LABEL_54:
              if ( srca == (char *)v151 )
                break;
              v42 = v43;
              srca += 32;
              v41 = *v34;
            }
          }
          v5 = sub_2FE34D0;
          v68 = *(__int64 (__fastcall **)(__int64, char, int, int, int, int, int))(*(_QWORD *)a4 + 1488LL);
          if ( v68 == sub_2FE34D0 )
          {
            v69 = v159 + 1;
            if ( !a3 )
              v69 *= 20;
            v70 = v69 + v160 + v161;
          }
          else
          {
            v70 = v68(a4, a3, 1, v159, v160, 0, v161);
            v68 = *(__int64 (__fastcall **)(__int64, char, int, int, int, int, int))(*(_QWORD *)a4 + 1488LL);
          }
          if ( v68 == sub_2FE34D0 )
          {
            if ( !a3 )
              v136 *= 20;
            v71 = v157 + v136;
          }
          else
          {
            v71 = v68(a4, a3, v136, 0, 0, v157, 0);
          }
          LOBYTE(v5) = v71 < v70;
          return (unsigned int)v5;
        }
        v10 = 1;
      }
    }
  }
  return (unsigned int)v5;
}
