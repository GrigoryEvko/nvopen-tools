// Function: sub_33C6820
// Address: 0x33c6820
//
void __fastcall sub_33C6820(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  __int64 i; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  char *v16; // rax
  _BYTE *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  _BYTE *v25; // r13
  unsigned int v26; // esi
  int v27; // r12d
  __int64 v28; // r10
  __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 v31; // rax
  _BYTE *v32; // rdi
  int v33; // r12d
  int v34; // eax
  __int64 *v35; // rdi
  __int64 v36; // r13
  __int64 v37; // rsi
  __int64 v38; // rdx
  unsigned __int64 v39; // rcx
  __int64 v40; // r14
  __int64 (__fastcall *v41)(__int64, __int64, __int64, unsigned __int64); // rax
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // r14
  int v45; // r13d
  unsigned __int64 *v46; // rdi
  _BYTE *v47; // rax
  __m128i *v48; // rsi
  char v49; // dl
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 (__fastcall *v52)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 (__fastcall *v55)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 (__fastcall *v58)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  unsigned __int16 v61; // ax
  __int64 v62; // rax
  char v63; // cl
  __int64 v64; // rax
  unsigned int v65; // eax
  int v66; // edx
  __int64 v67; // rax
  _QWORD *v68; // rdx
  _QWORD *j; // rax
  unsigned int v70; // ecx
  unsigned int v71; // edx
  _QWORD *v72; // rdi
  int v73; // ebx
  _QWORD *v74; // rax
  __int64 v75; // r8
  unsigned int v76; // ecx
  __int64 v77; // rsi
  unsigned int v78; // edx
  __int64 v79; // rax
  _BYTE *v80; // rdi
  int v81; // eax
  int v82; // r10d
  __int64 v83; // r12
  __int64 v84; // rcx
  int v85; // edx
  _BYTE *v86; // r8
  int *v87; // rax
  __int64 v88; // rdx
  int v89; // ecx
  __int64 (*v90)(); // rax
  __int64 v91; // rdx
  __int64 v92; // rdx
  int v93; // ecx
  int v94; // eax
  int v95; // r10d
  __int64 v96; // rsi
  __int64 v97; // r12
  int v98; // edi
  __int64 v99; // rcx
  _BYTE *v100; // r8
  unsigned __int64 v101; // rdx
  unsigned __int64 v102; // rdx
  unsigned __int64 v103; // rax
  _QWORD *v104; // rax
  __int64 v105; // rdx
  _QWORD *k; // rdx
  int v107; // eax
  int v108; // r9d
  __int64 v109; // rax
  int v110; // edi
  __int128 v111; // [rsp-10h] [rbp-1F0h]
  __int64 v112; // [rsp+8h] [rbp-1D8h]
  unsigned int v113; // [rsp+10h] [rbp-1D0h]
  unsigned __int64 v114; // [rsp+10h] [rbp-1D0h]
  unsigned __int64 v115; // [rsp+18h] [rbp-1C8h]
  __int64 v117; // [rsp+30h] [rbp-1B0h]
  __int64 v118; // [rsp+38h] [rbp-1A8h]
  __int64 v119; // [rsp+40h] [rbp-1A0h]
  __int64 v120; // [rsp+48h] [rbp-198h]
  int v121; // [rsp+50h] [rbp-190h]
  __int64 v122; // [rsp+58h] [rbp-188h]
  __int64 v123; // [rsp+60h] [rbp-180h]
  __int64 *v124; // [rsp+78h] [rbp-168h]
  __int64 v125; // [rsp+80h] [rbp-160h]
  int v126; // [rsp+90h] [rbp-150h]
  unsigned int v127; // [rsp+94h] [rbp-14Ch]
  unsigned int v128; // [rsp+98h] [rbp-148h]
  int *v129; // [rsp+98h] [rbp-148h]
  __int64 v130; // [rsp+98h] [rbp-148h]
  __int64 *v131; // [rsp+98h] [rbp-148h]
  unsigned __int16 v132; // [rsp+A6h] [rbp-13Ah] BYREF
  unsigned int v133; // [rsp+A8h] [rbp-138h] BYREF
  unsigned int v134; // [rsp+ACh] [rbp-134h]
  __int64 v135; // [rsp+B0h] [rbp-130h] BYREF
  unsigned __int64 v136; // [rsp+B8h] [rbp-128h]
  __int64 v137; // [rsp+C0h] [rbp-120h] BYREF
  unsigned __int64 v138; // [rsp+C8h] [rbp-118h]
  __int64 v139; // [rsp+D0h] [rbp-110h]
  __int64 v140; // [rsp+D8h] [rbp-108h]
  __int64 v141; // [rsp+E0h] [rbp-100h] BYREF
  __int64 v142; // [rsp+E8h] [rbp-F8h]
  __int64 v143; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v144; // [rsp+F8h] [rbp-E8h]
  __int64 v145; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v146; // [rsp+108h] [rbp-D8h]
  unsigned __int64 v147; // [rsp+110h] [rbp-D0h]
  __int64 v148; // [rsp+120h] [rbp-C0h] BYREF
  char *v149; // [rsp+128h] [rbp-B8h]
  __int64 v150; // [rsp+130h] [rbp-B0h]
  int v151; // [rsp+138h] [rbp-A8h]
  char v152; // [rsp+13Ch] [rbp-A4h]
  char v153; // [rsp+140h] [rbp-A0h] BYREF
  __int64 *v154; // [rsp+160h] [rbp-80h] BYREF
  __int64 v155; // [rsp+168h] [rbp-78h]
  _BYTE v156[112]; // [rsp+170h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 864);
  v148 = 0;
  v4 = *(_QWORD *)(v3 + 16);
  v152 = 1;
  v150 = 4;
  v117 = v4;
  v149 = &v153;
  v5 = *(_QWORD *)(a2 + 48);
  v151 = 0;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 != a2 + 48 )
  {
    if ( !v6 )
      goto LABEL_92;
    v123 = v6 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 <= 0xA )
    {
      v126 = sub_B46E30(v6 - 24);
      if ( v126 )
      {
        v7 = a1;
        v127 = 0;
        for ( i = sub_B46EC0(v123, 0); ; i = sub_B46EC0(v123, v127) )
        {
          v12 = i;
          v13 = *(_QWORD *)(i + 56);
          if ( !v13 )
LABEL_170:
            BUG();
          if ( *(_BYTE *)(v13 - 24) != 84 )
            goto LABEL_6;
          v14 = *(unsigned int *)(v12 + 44);
          v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 960) + 56LL) + 8 * v14);
          if ( v152 )
          {
            v16 = v149;
            v9 = HIDWORD(v150);
            v14 = (__int64)&v149[8 * HIDWORD(v150)];
            if ( v149 != (char *)v14 )
            {
              while ( v15 != *(_QWORD *)v16 )
              {
                v16 += 8;
                if ( (char *)v14 == v16 )
                  goto LABEL_14;
              }
              goto LABEL_6;
            }
LABEL_14:
            if ( HIDWORD(v150) < (unsigned int)v150 )
              break;
          }
          sub_C8CC70((__int64)&v148, v15, v14, v9, v10, v11);
          if ( v49 )
            goto LABEL_16;
LABEL_6:
          if ( v126 == ++v127 )
          {
            v2 = v7;
            goto LABEL_83;
          }
        }
        ++HIDWORD(v150);
        *(_QWORD *)v14 = v15;
        ++v148;
LABEL_16:
        v17 = *(_BYTE **)(v15 + 56);
        v18 = sub_AA5930(v12);
        v122 = v19;
        v125 = v18;
        if ( v18 == v19 )
          goto LABEL_6;
        while ( 2 )
        {
          if ( *(_QWORD *)(v125 + 16) && !(unsigned __int8)sub_BCADB0(*(_QWORD *)(v125 + 8)) )
          {
            v21 = *(_QWORD *)(v125 - 8);
            v22 = 0x1FFFFFFFE0LL;
            v23 = *(_DWORD *)(v125 + 4) & 0x7FFFFFF;
            if ( v23 )
            {
              v24 = 0;
              do
              {
                if ( a2 == *(_QWORD *)(v21 + 32LL * *(unsigned int *)(v125 + 72) + 8 * v24) )
                {
                  v22 = 32 * v24;
                  goto LABEL_29;
                }
                ++v24;
              }
              while ( v23 != (_DWORD)v24 );
              v25 = *(_BYTE **)(v21 + 0x1FFFFFFFE0LL);
              if ( !v25 )
LABEL_92:
                BUG();
            }
            else
            {
LABEL_29:
              v25 = *(_BYTE **)(v21 + v22);
              if ( !v25 )
                goto LABEL_92;
            }
            if ( *v25 <= 0x15u )
            {
              v26 = *(_DWORD *)(v7 + 952);
              if ( v26 )
              {
                v27 = 1;
                v28 = *(_QWORD *)(v7 + 936);
                v29 = 0;
                v128 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
                v30 = (v26 - 1) & v128;
                v31 = v28 + 16LL * v30;
                v32 = *(_BYTE **)v31;
                if ( v25 == *(_BYTE **)v31 )
                {
LABEL_33:
                  v33 = *(_DWORD *)(v31 + 8);
                  v129 = (int *)(v31 + 8);
                  if ( v33 )
                    goto LABEL_34;
LABEL_116:
                  *v129 = sub_374D200(*(_QWORD *)(v7 + 960), v125);
                  v89 = 215;
                  if ( *v25 == 17 )
                  {
                    v90 = *(__int64 (**)())(*(_QWORD *)v117 + 1464LL);
                    if ( v90 == sub_2FE34B0
                      || !((unsigned __int8 (__fastcall *)(__int64, _BYTE *, __int64, __int64))v90)(v117, v25, v88, 215) )
                    {
                      v89 = 214;
                    }
                    else
                    {
                      v89 = 213;
                    }
                  }
                  sub_33BF9C0(v7, (__int64)v25, *v129, v89);
                  v33 = *v129;
                  goto LABEL_34;
                }
                while ( v32 != (_BYTE *)-4096LL )
                {
                  if ( !v29 && v32 == (_BYTE *)-8192LL )
                    v29 = v31;
                  v30 = (v26 - 1) & (v27 + v30);
                  v31 = v28 + 16LL * v30;
                  v32 = *(_BYTE **)v31;
                  if ( v25 == *(_BYTE **)v31 )
                    goto LABEL_33;
                  ++v27;
                }
                if ( v29 )
                  v31 = v29;
                v93 = *(_DWORD *)(v7 + 944);
                ++*(_QWORD *)(v7 + 928);
                v85 = v93 + 1;
                if ( 4 * (v93 + 1) < 3 * v26 )
                {
                  if ( v26 - *(_DWORD *)(v7 + 948) - v85 <= v26 >> 3 )
                  {
                    sub_3385880(v7 + 928, v26);
                    v94 = *(_DWORD *)(v7 + 952);
                    if ( !v94 )
                    {
LABEL_169:
                      ++*(_DWORD *)(v7 + 944);
                      BUG();
                    }
                    v95 = v94 - 1;
                    v96 = 0;
                    v97 = *(_QWORD *)(v7 + 936);
                    v98 = 1;
                    LODWORD(v99) = (v94 - 1) & v128;
                    v85 = *(_DWORD *)(v7 + 944) + 1;
                    v31 = v97 + 16LL * (unsigned int)v99;
                    v100 = *(_BYTE **)v31;
                    if ( v25 != *(_BYTE **)v31 )
                    {
                      while ( v100 != (_BYTE *)-4096LL )
                      {
                        if ( v100 == (_BYTE *)-8192LL && !v96 )
                          v96 = v31;
                        v99 = v95 & (unsigned int)(v99 + v98);
                        v31 = v97 + 16 * v99;
                        v100 = *(_BYTE **)v31;
                        if ( v25 == *(_BYTE **)v31 )
                          goto LABEL_113;
                        ++v98;
                      }
                      goto LABEL_140;
                    }
                  }
                  goto LABEL_113;
                }
              }
              else
              {
                ++*(_QWORD *)(v7 + 928);
              }
              sub_3385880(v7 + 928, 2 * v26);
              v81 = *(_DWORD *)(v7 + 952);
              if ( !v81 )
                goto LABEL_169;
              v82 = v81 - 1;
              v83 = *(_QWORD *)(v7 + 936);
              LODWORD(v84) = (v81 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v85 = *(_DWORD *)(v7 + 944) + 1;
              v31 = v83 + 16LL * (unsigned int)v84;
              v86 = *(_BYTE **)v31;
              if ( *(_BYTE **)v31 != v25 )
              {
                v110 = 1;
                v96 = 0;
                while ( v86 != (_BYTE *)-4096LL )
                {
                  if ( !v96 && v86 == (_BYTE *)-8192LL )
                    v96 = v31;
                  v84 = v82 & (unsigned int)(v84 + v110);
                  v31 = v83 + 16 * v84;
                  v86 = *(_BYTE **)v31;
                  if ( v25 == *(_BYTE **)v31 )
                    goto LABEL_113;
                  ++v110;
                }
LABEL_140:
                if ( v96 )
                  v31 = v96;
              }
LABEL_113:
              *(_DWORD *)(v7 + 944) = v85;
              if ( *(_QWORD *)v31 != -4096 )
                --*(_DWORD *)(v7 + 948);
              *(_QWORD *)v31 = v25;
              v87 = (int *)(v31 + 8);
              *v87 = 0;
              v129 = v87;
              goto LABEL_116;
            }
            v75 = *(_QWORD *)(v7 + 960);
            v76 = *(_DWORD *)(v75 + 144);
            v77 = *(_QWORD *)(v75 + 128);
            if ( v76 )
            {
              v78 = (v76 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v79 = v77 + 16LL * v78;
              v80 = *(_BYTE **)v79;
              if ( v25 == *(_BYTE **)v79 )
              {
LABEL_107:
                if ( v79 != v77 + 16LL * v76 )
                {
                  v33 = *(_DWORD *)(v79 + 8);
                  goto LABEL_34;
                }
              }
              else
              {
                v107 = 1;
                while ( v80 != (_BYTE *)-4096LL )
                {
                  v108 = v107 + 1;
                  v109 = (v76 - 1) & (v78 + v107);
                  v78 = v109;
                  v79 = v77 + 16 * v109;
                  v80 = *(_BYTE **)v79;
                  if ( v25 == *(_BYTE **)v79 )
                    goto LABEL_107;
                  v107 = v108;
                }
              }
            }
            v33 = sub_374D200(*(_QWORD *)(v7 + 960), v125);
            sub_33BF9C0(v7, (__int64)v25, v33, 215);
LABEL_34:
            v154 = (__int64 *)v156;
            v155 = 0x400000000LL;
            v130 = *(_QWORD *)(v125 + 8);
            v34 = sub_2E79000(*(__int64 **)(*(_QWORD *)(v7 + 864) + 40LL));
            LOBYTE(v146) = 0;
            *((_QWORD *)&v111 + 1) = v146;
            v145 = 0;
            *(_QWORD *)&v111 = 0;
            sub_34B8C80(v117, v34, v130, (unsigned int)&v154, 0, 0, v111);
            v35 = v154;
            v124 = &v154[2 * (unsigned int)v155];
            if ( v124 != v154 )
            {
              v131 = v154;
              v36 = v7;
              while ( 1 )
              {
                v37 = *(_QWORD *)(v36 + 864);
                BYTE2(v134) = 0;
                v38 = *v131;
                v39 = v131[1];
                v40 = *(_QWORD *)(v37 + 64);
                v41 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v117 + 736LL);
                if ( v41 == sub_2FEA1A0 )
                {
                  v135 = *v131;
                  v136 = v39;
                  if ( (_WORD)v38 )
                  {
                    v42 = *(unsigned __int16 *)(v117 + 2LL * (unsigned __int16)v38 + 2304);
                  }
                  else
                  {
                    if ( !sub_30070B0((__int64)&v135) )
                    {
                      if ( !sub_3007070((__int64)&v135) )
                        goto LABEL_170;
                      v139 = sub_3007260((__int64)&v135);
                      v140 = v50;
                      v145 = v139;
                      LOBYTE(v146) = v50;
                      v121 = sub_CA1930(&v145);
                      v51 = (unsigned __int16)v135;
                      v137 = v135;
                      v138 = v136;
                      if ( (_WORD)v135 )
                        goto LABEL_77;
                      v112 = v136;
                      v113 = v135;
                      if ( sub_30070B0((__int64)&v137) )
                      {
                        LOWORD(v145) = 0;
                        LOWORD(v141) = 0;
                        v146 = 0;
                        sub_2FE8D10(
                          v117,
                          v40,
                          (unsigned int)v137,
                          v138,
                          &v145,
                          (unsigned int *)&v143,
                          (unsigned __int16 *)&v141);
                        v61 = v141;
                      }
                      else
                      {
                        if ( !sub_3007070((__int64)&v137) )
                          goto LABEL_170;
                        v52 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v117 + 592LL);
                        if ( v52 == sub_2D56A50 )
                        {
                          sub_2FE6CC0((__int64)&v145, v117, v40, v113, v112);
                          v53 = v118;
                          LOWORD(v53) = v146;
                          v54 = v147;
                          v118 = v53;
                        }
                        else
                        {
                          v118 = v52(v117, v40, v137, v138);
                          v54 = v91;
                        }
                        v142 = v54;
                        v51 = (unsigned __int16)v118;
                        v141 = v118;
                        if ( (_WORD)v118 )
                          goto LABEL_77;
                        v114 = v54;
                        if ( sub_30070B0((__int64)&v141) )
                        {
                          LOWORD(v145) = 0;
                          v146 = 0;
                          LOWORD(v133) = 0;
                          sub_2FE8D10(
                            v117,
                            v40,
                            (unsigned int)v141,
                            v114,
                            &v145,
                            (unsigned int *)&v143,
                            (unsigned __int16 *)&v133);
                          v61 = v133;
                        }
                        else
                        {
                          if ( !sub_3007070((__int64)&v141) )
                            goto LABEL_170;
                          v55 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v117
                                                                                                  + 592LL);
                          if ( v55 == sub_2D56A50 )
                          {
                            sub_2FE6CC0((__int64)&v145, v117, v40, v141, v142);
                            v56 = v119;
                            LOWORD(v56) = v146;
                            v57 = v147;
                            v119 = v56;
                          }
                          else
                          {
                            v119 = v55(v117, v40, v141, v114);
                            v57 = v92;
                          }
                          v144 = v57;
                          v51 = (unsigned __int16)v119;
                          v143 = v119;
                          if ( !(_WORD)v119 )
                          {
                            v115 = v57;
                            if ( sub_30070B0((__int64)&v143) )
                            {
                              LOWORD(v145) = 0;
                              v132 = 0;
                              v146 = 0;
                              sub_2FE8D10(v117, v40, (unsigned int)v143, v115, &v145, &v133, &v132);
                              v61 = v132;
                            }
                            else
                            {
                              if ( !sub_3007070((__int64)&v143) )
                                goto LABEL_170;
                              v58 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v117 + 592LL);
                              if ( v58 == sub_2D56A50 )
                              {
                                sub_2FE6CC0((__int64)&v145, v117, v40, v143, v144);
                                v59 = v120;
                                LOWORD(v59) = v146;
                                v60 = v147;
                                v120 = v59;
                              }
                              else
                              {
                                v120 = v58(v117, v40, v143, v115);
                                v60 = v101;
                              }
                              v61 = sub_2FE98B0(v117, v40, (unsigned int)v120, v60);
                            }
                            goto LABEL_78;
                          }
LABEL_77:
                          v61 = *(_WORD *)(v117 + 2 * v51 + 2852);
                        }
                      }
LABEL_78:
                      if ( v61 <= 1u || (unsigned __int16)(v61 - 504) <= 7u )
                        goto LABEL_170;
                      v62 = 16LL * (v61 - 1);
                      v63 = byte_444C4A0[v62 + 8];
                      v64 = *(_QWORD *)&byte_444C4A0[v62];
                      LOBYTE(v146) = v63;
                      v145 = v64;
                      v65 = sub_CA1930(&v145);
                      v42 = (v121 + v65 - 1) / v65;
                      goto LABEL_39;
                    }
                    LOWORD(v145) = 0;
                    LOWORD(v141) = 0;
                    v146 = 0;
                    v42 = sub_2FE8D10(
                            v117,
                            v40,
                            (unsigned int)v135,
                            v136,
                            &v145,
                            (unsigned int *)&v143,
                            (unsigned __int16 *)&v141);
                  }
                }
                else
                {
                  v42 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, unsigned __int64, _QWORD))v41)(
                          v117,
                          *(_QWORD *)(v37 + 64),
                          v38,
                          v39,
                          v134);
                }
LABEL_39:
                if ( !v42 )
                  goto LABEL_54;
                v43 = (__int64)v17;
                v44 = v36;
                v45 = v33;
                v33 += v42;
                while ( 1 )
                {
                  LODWORD(v143) = v45;
                  v46 = *(unsigned __int64 **)(v44 + 960);
                  if ( !v17 )
                    BUG();
                  v47 = v17;
                  if ( (*v17 & 4) == 0 && (v17[44] & 8) != 0 )
                  {
                    do
                      v47 = (_BYTE *)*((_QWORD *)v47 + 1);
                    while ( (v47[44] & 8) != 0 );
                  }
                  v17 = (_BYTE *)*((_QWORD *)v47 + 1);
                  v145 = v43;
                  v48 = (__m128i *)v46[108];
                  if ( v48 != (__m128i *)v46[109] )
                    break;
                  ++v45;
                  sub_337AE20(v46 + 107, v48, &v145, &v143);
                  if ( v45 == v33 )
                    goto LABEL_53;
LABEL_49:
                  v43 = (__int64)v17;
                }
                if ( v48 )
                {
                  v48->m128i_i64[0] = v43;
                  v48->m128i_i32[2] = v143;
                  v48 = (__m128i *)v46[108];
                }
                ++v45;
                v46[108] = (unsigned __int64)&v48[1];
                if ( v45 != v33 )
                  goto LABEL_49;
LABEL_53:
                v36 = v44;
LABEL_54:
                v131 += 2;
                if ( v124 == v131 )
                {
                  v35 = v154;
                  v7 = v36;
                  break;
                }
              }
            }
            if ( v35 != (__int64 *)v156 )
              _libc_free((unsigned __int64)v35);
          }
          v20 = *(_QWORD *)(v125 + 32);
          if ( !v20 )
            goto LABEL_170;
          v125 = 0;
          if ( *(_BYTE *)(v20 - 24) == 84 )
            v125 = v20 - 24;
          if ( v122 == v125 )
            goto LABEL_6;
          continue;
        }
      }
    }
  }
LABEL_83:
  v66 = *(_DWORD *)(v2 + 944);
  ++*(_QWORD *)(v2 + 928);
  if ( !v66 )
  {
    if ( !*(_DWORD *)(v2 + 948) )
      goto LABEL_89;
    v67 = *(unsigned int *)(v2 + 952);
    if ( (unsigned int)v67 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(v2 + 936), 16 * v67, 8);
      *(_QWORD *)(v2 + 936) = 0;
      *(_QWORD *)(v2 + 944) = 0;
      *(_DWORD *)(v2 + 952) = 0;
      goto LABEL_89;
    }
    goto LABEL_86;
  }
  v70 = 4 * v66;
  v67 = *(unsigned int *)(v2 + 952);
  if ( (unsigned int)(4 * v66) < 0x40 )
    v70 = 64;
  if ( (unsigned int)v67 <= v70 )
  {
LABEL_86:
    v68 = *(_QWORD **)(v2 + 936);
    for ( j = &v68[2 * v67]; j != v68; v68 += 2 )
      *v68 = -4096;
    *(_QWORD *)(v2 + 944) = 0;
    goto LABEL_89;
  }
  v71 = v66 - 1;
  if ( v71 )
  {
    _BitScanReverse(&v71, v71);
    v72 = *(_QWORD **)(v2 + 936);
    v73 = 1 << (33 - (v71 ^ 0x1F));
    if ( v73 < 64 )
      v73 = 64;
    if ( v73 == (_DWORD)v67 )
    {
      *(_QWORD *)(v2 + 944) = 0;
      v74 = &v72[2 * (unsigned int)v73];
      do
      {
        if ( v72 )
          *v72 = -4096;
        v72 += 2;
      }
      while ( v74 != v72 );
      goto LABEL_89;
    }
  }
  else
  {
    v72 = *(_QWORD **)(v2 + 936);
    v73 = 64;
  }
  sub_C7D6A0((__int64)v72, 16 * v67, 8);
  v102 = ((((((((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
            | (4 * v73 / 3u + 1)
            | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 4)
          | (((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
          | (4 * v73 / 3u + 1)
          | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
          | (4 * v73 / 3u + 1)
          | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 4)
        | (((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
        | (4 * v73 / 3u + 1)
        | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 16;
  v103 = (v102
        | (((((((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
            | (4 * v73 / 3u + 1)
            | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 4)
          | (((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
          | (4 * v73 / 3u + 1)
          | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
          | (4 * v73 / 3u + 1)
          | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 4)
        | (((4 * v73 / 3u + 1) | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1)) >> 2)
        | (4 * v73 / 3u + 1)
        | ((unsigned __int64)(4 * v73 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(v2 + 952) = v103;
  v104 = (_QWORD *)sub_C7D670(16 * v103, 8);
  v105 = *(unsigned int *)(v2 + 952);
  *(_QWORD *)(v2 + 944) = 0;
  *(_QWORD *)(v2 + 936) = v104;
  for ( k = &v104[2 * v105]; k != v104; v104 += 2 )
  {
    if ( v104 )
      *v104 = -4096;
  }
LABEL_89:
  if ( !v152 )
    _libc_free((unsigned __int64)v149);
}
