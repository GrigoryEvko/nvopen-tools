// Function: sub_3255C80
// Address: 0x3255c80
//
void __fastcall sub_3255C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // eax
  __int64 v7; // r14
  void (__fastcall *v8)(__int64, _QWORD, _QWORD); // r15
  __int64 v9; // rax
  __int64 v10; // r14
  void (__fastcall *v11)(__int64, _QWORD, _QWORD); // r15
  __int64 v12; // rax
  __int64 v13; // r14
  void (__fastcall *v14)(__int64, _QWORD, _QWORD); // r15
  __int64 v15; // rax
  __int64 *v16; // r13
  int v17; // r15d
  __int64 *v18; // rbx
  size_t v19; // rdx
  __int64 v20; // r12
  __int64 v21; // r14
  __int64 v22; // rax
  int v23; // eax
  int v24; // esi
  __int64 **v25; // r14
  __int64 *v26; // r12
  __int64 v27; // rcx
  __int64 v28; // rax
  size_t v29; // rdx
  __int64 v30; // r13
  void (*v31)(); // r12
  __m128i v32; // rax
  void (*v33)(void); // rax
  __int64 *v34; // rdx
  __int64 *v35; // r13
  __int64 v36; // rax
  int v37; // esi
  __int64 v38; // r15
  __int64 v39; // r12
  __int64 *v40; // rax
  char v41; // al
  __int64 v42; // rdx
  _QWORD *v43; // rdx
  char v44; // cl
  _QWORD *v45; // rax
  char v46; // dl
  _QWORD *v47; // rsi
  char v48; // al
  __m128i *v49; // rsi
  const char **v50; // rcx
  char v51; // dl
  __m128i *v52; // rsi
  _QWORD **v53; // rcx
  char v54; // al
  __m128i *v55; // rsi
  __m128i *v56; // rcx
  __m128i v57; // rax
  char v58; // al
  const char **v59; // rdx
  char v60; // dl
  __m128i *v61; // rcx
  char v62; // al
  _QWORD **v63; // rsi
  char v64; // dl
  __m128i *v65; // rsi
  __m128i *v66; // rcx
  __m128i *v67; // rsi
  __int64 v68; // rcx
  __m128i *v69; // rdi
  __m128i *v70; // rsi
  __int64 v71; // rcx
  __m128i *v72; // rdi
  __m128i v73; // xmm7
  __m128i v74; // xmm3
  __m128i v75; // xmm5
  __m128i v76; // xmm1
  __m128i *v77; // rdi
  __m128i *v78; // rsi
  __int64 m; // rcx
  __m128i *v80; // rdi
  __int32 *v81; // rsi
  __int64 k; // rcx
  __m128i *v83; // rdi
  __m128i *v84; // rsi
  __int64 j; // rcx
  __m128i *v86; // rdi
  __int64 v87; // rcx
  __int32 *v88; // rsi
  __m128i *v89; // rdi
  const char **v90; // rsi
  __int64 i; // rcx
  _DWORD *v92; // rsi
  __int64 v93; // rcx
  _DWORD *v94; // rdi
  size_t v95; // [rsp+8h] [rbp-2E8h]
  __int64 v96; // [rsp+10h] [rbp-2E0h]
  __int64 v97; // [rsp+18h] [rbp-2D8h]
  __int64 v98; // [rsp+20h] [rbp-2D0h]
  __int64 v99; // [rsp+28h] [rbp-2C8h]
  __int64 v100; // [rsp+30h] [rbp-2C0h]
  __int64 v101; // [rsp+38h] [rbp-2B8h]
  __int64 v102; // [rsp+40h] [rbp-2B0h]
  __int64 v103; // [rsp+48h] [rbp-2A8h]
  __int64 v104; // [rsp+50h] [rbp-2A0h]
  __int64 v105; // [rsp+58h] [rbp-298h]
  __int64 v106; // [rsp+60h] [rbp-290h]
  __int64 v107; // [rsp+68h] [rbp-288h]
  __int64 v108; // [rsp+70h] [rbp-280h]
  __int64 v109; // [rsp+78h] [rbp-278h]
  __int64 v110; // [rsp+88h] [rbp-268h]
  __int64 **v111; // [rsp+88h] [rbp-268h]
  __int64 v112; // [rsp+90h] [rbp-260h]
  __int64 *v113; // [rsp+90h] [rbp-260h]
  unsigned int v114; // [rsp+9Ch] [rbp-254h]
  unsigned __int64 v115; // [rsp+A0h] [rbp-250h] BYREF
  __int64 *v116; // [rsp+A8h] [rbp-248h] BYREF
  _QWORD v117[4]; // [rsp+B0h] [rbp-240h] BYREF
  __int16 v118; // [rsp+D0h] [rbp-220h]
  _QWORD v119[4]; // [rsp+E0h] [rbp-210h] BYREF
  char v120; // [rsp+100h] [rbp-1F0h]
  char v121; // [rsp+101h] [rbp-1EFh]
  _QWORD v122[4]; // [rsp+110h] [rbp-1E0h] BYREF
  __int16 v123; // [rsp+130h] [rbp-1C0h]
  _QWORD v124[4]; // [rsp+140h] [rbp-1B0h] BYREF
  __int16 v125; // [rsp+160h] [rbp-190h]
  __m128i v126; // [rsp+170h] [rbp-180h] BYREF
  __m128i v127; // [rsp+180h] [rbp-170h] BYREF
  __int64 v128; // [rsp+190h] [rbp-160h]
  const char *v129; // [rsp+1A0h] [rbp-150h] BYREF
  __int64 v130; // [rsp+1A8h] [rbp-148h]
  char v131; // [rsp+1C0h] [rbp-130h]
  char v132; // [rsp+1C1h] [rbp-12Fh]
  __m128i v133; // [rsp+1D0h] [rbp-120h] BYREF
  __m128i v134; // [rsp+1E0h] [rbp-110h] BYREF
  __int64 v135; // [rsp+1F0h] [rbp-100h]
  __int64 **v136; // [rsp+200h] [rbp-F0h] BYREF
  __int64 v137; // [rsp+208h] [rbp-E8h]
  __int16 v138; // [rsp+220h] [rbp-D0h]
  __m128i v139; // [rsp+230h] [rbp-C0h] BYREF
  __m128i v140; // [rsp+240h] [rbp-B0h] BYREF
  __int64 v141; // [rsp+250h] [rbp-A0h]
  __m128i v142; // [rsp+260h] [rbp-90h] BYREF
  __m128i v143; // [rsp+270h] [rbp-80h] BYREF
  __int64 v144; // [rsp+280h] [rbp-70h]
  __m128i v145; // [rsp+290h] [rbp-60h] BYREF
  __m128i v146; // [rsp+2A0h] [rbp-50h]
  __int64 v147; // [rsp+2B0h] [rbp-40h]

  v6 = sub_AE4380(a2 + 312, 0);
  v7 = *(_QWORD *)(a4 + 224);
  v114 = v6;
  v8 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v7 + 176LL);
  v9 = sub_31DA6B0(a4);
  v8(v7, *(_QWORD *)(v9 + 24), 0);
  sub_3255900(a2, a4, "code_end");
  v10 = *(_QWORD *)(a4 + 224);
  v11 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v10 + 176LL);
  v12 = sub_31DA6B0(a4);
  v11(v10, *(_QWORD *)(v12 + 32), 0);
  sub_3255900(a2, a4, "data_end");
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a4 + 224) + 536LL))(*(_QWORD *)(a4 + 224), 0, v114);
  v13 = *(_QWORD *)(a4 + 224);
  v14 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v13 + 176LL);
  v15 = sub_31DA6B0(a4);
  v14(v13, *(_QWORD *)(v15 + 32), 0);
  sub_3255900(a2, a4, "frametable");
  v16 = *(__int64 **)(a3 + 224);
  if ( *(__int64 **)(a3 + 232) == v16 )
  {
    v17 = 0;
  }
  else
  {
    v112 = a3;
    v17 = 0;
    v110 = a4;
    v18 = *(__int64 **)(a3 + 232);
    v19 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL);
    v20 = *(_QWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v21 = *v16;
        v22 = *(_QWORD *)(*v16 + 8);
        if ( *(_QWORD *)(v22 + 16) == v19 )
        {
          if ( !v19 )
            break;
          v95 = v19;
          v23 = memcmp(*(const void **)(v22 + 8), *(const void **)(v20 + 8), v19);
          v19 = v95;
          if ( !v23 )
            break;
        }
        if ( v18 == ++v16 )
          goto LABEL_8;
      }
      ++v16;
      v17 += (__int64)(*(_QWORD *)(v21 + 56) - *(_QWORD *)(v21 + 48)) >> 4;
    }
    while ( v18 != v16 );
LABEL_8:
    a3 = v112;
    a4 = v110;
    if ( v17 > 0xFFFF )
      sub_C64ED0(" Too much descriptor for ocaml GC", 1u);
  }
  v24 = v17;
  sub_31DC9F0(a4, v17);
  LOBYTE(v24) = v114 != 4;
  sub_31DCA70(a4, v24 + 2, 0, 0);
  v25 = *(__int64 ***)(a3 + 224);
  v111 = *(__int64 ***)(a3 + 232);
  while ( v111 != v25 )
  {
    v26 = *v25;
    v27 = *(_QWORD *)(a1 + 8);
    v28 = (*v25)[1];
    v29 = *(_QWORD *)(v28 + 16);
    if ( v29 == *(_QWORD *)(v27 + 16) && (!v29 || !memcmp(*(const void **)(v28 + 8), *(const void **)(v27 + 8), v29)) )
    {
      v115 = v26[2];
      if ( v115 > 0xFFFF )
      {
        v142.m128i_i64[0] = (__int64)")";
        LOWORD(v144) = 259;
        v40 = *v25;
        v138 = 267;
        v116 = v40;
        v136 = &v116;
        v129 = ">= 65536.\n(";
        v124[0] = &v115;
        v132 = 1;
        v131 = 3;
        v125 = 267;
        v121 = 1;
        v119[0] = "' is too large for the ocaml GC! Frame size ";
        v120 = 3;
        v117[2] = sub_BD5D20(**v25);
        v41 = v120;
        v118 = 1283;
        v117[0] = "Function '";
        v117[3] = v42;
        if ( v120 )
        {
          if ( v120 == 1 )
          {
            v45 = v122;
            v92 = v117;
            v93 = 10;
            v94 = v122;
            while ( v93 )
            {
              *v94++ = *v92++;
              --v93;
            }
            v44 = v125;
            if ( (_BYTE)v125 )
            {
              if ( (_BYTE)v125 == 1 )
                goto LABEL_131;
              if ( HIBYTE(v123) == 1 )
              {
                v46 = 3;
                v108 = v122[1];
                v45 = (_QWORD *)v122[0];
              }
              else
              {
LABEL_36:
                v46 = 2;
              }
              if ( HIBYTE(v125) == 1 )
              {
                v47 = (_QWORD *)v124[0];
                v107 = v124[1];
              }
              else
              {
                v47 = v124;
                v44 = 2;
              }
              v126.m128i_i64[0] = (__int64)v45;
              v127.m128i_i64[0] = (__int64)v47;
              v126.m128i_i64[1] = v108;
              LOBYTE(v128) = v46;
              v127.m128i_i64[1] = v107;
              v48 = v131;
              BYTE1(v128) = v44;
              if ( v131 )
                goto LABEL_40;
              goto LABEL_109;
            }
          }
          else
          {
            if ( v121 == 1 )
            {
              v43 = (_QWORD *)v119[0];
              v109 = v119[1];
            }
            else
            {
              v43 = v119;
              v41 = 2;
            }
            v122[2] = v43;
            v122[0] = v117;
            LOBYTE(v123) = 2;
            v122[3] = v109;
            v44 = v125;
            HIBYTE(v123) = v41;
            if ( (_BYTE)v125 )
            {
              v45 = v122;
              if ( (_BYTE)v125 != 1 )
                goto LABEL_36;
LABEL_131:
              v86 = &v126;
              v87 = 10;
              v88 = (__int32 *)v122;
              while ( v87 )
              {
                v86->m128i_i32[0] = *v88++;
                v86 = (__m128i *)((char *)v86 + 4);
                --v87;
              }
              v46 = v128;
              if ( (_BYTE)v128 )
              {
                v48 = v131;
                if ( v131 )
                {
                  if ( (_BYTE)v128 == 1 )
                  {
                    v89 = &v133;
                    v90 = &v129;
                    for ( i = 10; i; --i )
                    {
                      v89->m128i_i32[0] = *(_DWORD *)v90;
                      v90 = (const char **)((char *)v90 + 4);
                      v89 = (__m128i *)((char *)v89 + 4);
                    }
                    goto LABEL_46;
                  }
LABEL_40:
                  if ( v48 == 1 )
                  {
                    v83 = &v133;
                    v84 = &v126;
                    for ( j = 10; j; --j )
                    {
                      v83->m128i_i32[0] = v84->m128i_i32[0];
                      v84 = (__m128i *)((char *)v84 + 4);
                      v83 = (__m128i *)((char *)v83 + 4);
                    }
                    v48 = v135;
                    if ( (_BYTE)v135 )
                      goto LABEL_46;
                  }
                  else
                  {
                    if ( BYTE1(v128) == 1 )
                    {
                      v106 = v126.m128i_i64[1];
                      v49 = (__m128i *)v126.m128i_i64[0];
                    }
                    else
                    {
                      v49 = &v126;
                      v46 = 2;
                    }
                    if ( v132 == 1 )
                    {
                      v105 = v130;
                      v50 = (const char **)v129;
                    }
                    else
                    {
                      v50 = &v129;
                      v48 = 2;
                    }
                    v134.m128i_i64[0] = (__int64)v50;
                    v133.m128i_i64[0] = (__int64)v49;
                    v133.m128i_i64[1] = v106;
                    v134.m128i_i64[1] = v105;
                    LOBYTE(v135) = v46;
                    BYTE1(v135) = v48;
                    v48 = v46;
LABEL_46:
                    v51 = v138;
                    if ( (_BYTE)v138 )
                    {
                      if ( v48 == 1 )
                      {
                        v80 = &v139;
                        v81 = (__int32 *)&v136;
                        for ( k = 10; k; --k )
                        {
                          v80->m128i_i32[0] = *v81++;
                          v80 = (__m128i *)((char *)v80 + 4);
                        }
                        goto LABEL_54;
                      }
                      if ( (_BYTE)v138 == 1 )
                      {
                        v77 = &v139;
                        v78 = &v133;
                        for ( m = 10; m; --m )
                        {
                          v77->m128i_i32[0] = v78->m128i_i32[0];
                          v78 = (__m128i *)((char *)v78 + 4);
                          v77 = (__m128i *)((char *)v77 + 4);
                        }
                        v51 = v141;
                        if ( (_BYTE)v141 )
                          goto LABEL_54;
                      }
                      else
                      {
                        if ( BYTE1(v135) == 1 )
                        {
                          v104 = v133.m128i_i64[1];
                          v52 = (__m128i *)v133.m128i_i64[0];
                        }
                        else
                        {
                          v52 = &v133;
                          v48 = 2;
                        }
                        if ( HIBYTE(v138) == 1 )
                        {
                          v103 = v137;
                          v53 = v136;
                        }
                        else
                        {
                          v53 = &v136;
                          v51 = 2;
                        }
                        v140.m128i_i64[0] = (__int64)v53;
                        v139.m128i_i64[0] = (__int64)v52;
                        v139.m128i_i64[1] = v104;
                        v140.m128i_i64[1] = v103;
                        LOBYTE(v141) = v48;
                        BYTE1(v141) = v51;
                        v51 = v48;
LABEL_54:
                        v54 = v144;
                        if ( (_BYTE)v144 )
                        {
                          if ( v51 == 1 )
                          {
                            v70 = &v142;
                            v71 = 10;
                            v72 = &v145;
                            while ( v71 )
                            {
                              v72->m128i_i32[0] = v70->m128i_i32[0];
                              v70 = (__m128i *)((char *)v70 + 4);
                              v72 = (__m128i *)((char *)v72 + 4);
                              --v71;
                            }
                          }
                          else if ( (_BYTE)v144 == 1 )
                          {
                            v67 = &v139;
                            v68 = 10;
                            v69 = &v145;
                            while ( v68 )
                            {
                              v69->m128i_i32[0] = v67->m128i_i32[0];
                              v67 = (__m128i *)((char *)v67 + 4);
                              v69 = (__m128i *)((char *)v69 + 4);
                              --v68;
                            }
                          }
                          else
                          {
                            if ( BYTE1(v141) == 1 )
                            {
                              v102 = v139.m128i_i64[1];
                              v55 = (__m128i *)v139.m128i_i64[0];
                            }
                            else
                            {
                              v55 = &v139;
                              v51 = 2;
                            }
                            if ( BYTE1(v144) == 1 )
                            {
                              v101 = v142.m128i_i64[1];
                              v56 = (__m128i *)v142.m128i_i64[0];
                            }
                            else
                            {
                              v56 = &v142;
                              v54 = 2;
                            }
                            v146.m128i_i64[0] = (__int64)v56;
                            v145.m128i_i64[0] = (__int64)v55;
                            v145.m128i_i64[1] = v102;
                            v146.m128i_i64[1] = v101;
                            LOBYTE(v147) = v51;
                            BYTE1(v147) = v54;
                          }
LABEL_62:
                          sub_C64D30((__int64)&v145, 1u);
                        }
                      }
LABEL_111:
                      LOWORD(v147) = 256;
                      goto LABEL_62;
                    }
                  }
LABEL_110:
                  LOWORD(v141) = 256;
                  goto LABEL_111;
                }
              }
LABEL_109:
              LOWORD(v135) = 256;
              goto LABEL_110;
            }
          }
        }
        else
        {
          v123 = 256;
        }
        LOWORD(v128) = 256;
        goto LABEL_109;
      }
      v30 = *(_QWORD *)(a4 + 224);
      v31 = *(void (**)())(*(_QWORD *)v30 + 120LL);
      v32.m128i_i64[0] = (__int64)sub_BD5D20(**v25);
      v145.m128i_i64[0] = (__int64)"live roots for ";
      v146 = v32;
      LOWORD(v147) = 1283;
      if ( v31 != nullsub_98 )
        ((void (__fastcall *)(__int64, __m128i *, __int64))v31)(v30, &v145, 1);
      v33 = *(void (**)(void))(**(_QWORD **)(a4 + 224) + 160LL);
      if ( v33 != nullsub_99 )
        v33();
      v34 = *v25;
      v35 = (__int64 *)(*v25)[6];
      v113 = (__int64 *)(*v25)[7];
      if ( v113 != v35 )
      {
        while ( 1 )
        {
          v36 = v34[4] - v34[3];
          v124[0] = v36 >> 4;
          if ( (unsigned __int64)v36 > 0xFFFF0 )
            break;
          sub_E9A500(*(_QWORD *)(a4 + 224), *v35, v114, 0);
          sub_31DC9F0(a4, v115);
          v37 = v124[0];
          sub_31DC9F0(a4, v124[0]);
          v38 = (*v25)[3];
          v39 = (*v25)[4];
          while ( v39 != v38 )
          {
            v37 = *(_DWORD *)(v38 + 4);
            if ( v37 > 0xFFFF )
              sub_C64ED0("GC root stack offset is outside of fixed stack frame and out of range for ocaml GC!", 1u);
            v38 += 16;
            sub_31DC9F0(a4, v37);
          }
          LOBYTE(v37) = v114 != 4;
          v35 += 2;
          sub_31DCA70(a4, v37 + 2, 0, 0);
          if ( v113 == v35 )
            goto LABEL_11;
          v34 = *v25;
        }
        v142.m128i_i64[0] = (__int64)" >= 65536.";
        v136 = (__int64 **)v124;
        LOWORD(v144) = 259;
        v138 = 267;
        v132 = 1;
        v129 = "' is too large for the ocaml GC! Live root count ";
        v131 = 3;
        v57.m128i_i64[0] = (__int64)sub_BD5D20(**v25);
        v127 = v57;
        v58 = v131;
        LOWORD(v128) = 1283;
        v126.m128i_i64[0] = (__int64)"Function '";
        if ( v131 )
        {
          if ( v131 == 1 )
          {
            v76 = _mm_loadu_si128(&v127);
            v60 = v138;
            v133 = _mm_loadu_si128(&v126);
            v135 = v128;
            v134 = v76;
            if ( (_BYTE)v138 )
            {
              if ( (_BYTE)v138 == 1 )
                goto LABEL_98;
              if ( BYTE1(v135) == 1 )
              {
                v99 = v133.m128i_i64[1];
                v61 = (__m128i *)v133.m128i_i64[0];
                v62 = 3;
              }
              else
              {
LABEL_70:
                v61 = &v133;
                v62 = 2;
              }
              if ( HIBYTE(v138) == 1 )
              {
                v63 = v136;
                v98 = v137;
              }
              else
              {
                v63 = &v136;
                v60 = 2;
              }
              v139.m128i_i64[0] = (__int64)v61;
              BYTE1(v141) = v60;
              v64 = v144;
              v139.m128i_i64[1] = v99;
              v140.m128i_i64[0] = (__int64)v63;
              v140.m128i_i64[1] = v98;
              LOBYTE(v141) = v62;
              if ( (_BYTE)v144 )
                goto LABEL_74;
              goto LABEL_96;
            }
          }
          else
          {
            if ( v132 == 1 )
            {
              v59 = (const char **)v129;
              v100 = v130;
            }
            else
            {
              v59 = &v129;
              v58 = 2;
            }
            v134.m128i_i64[0] = (__int64)v59;
            v60 = v138;
            v133.m128i_i64[0] = (__int64)&v126;
            LOBYTE(v135) = 2;
            v134.m128i_i64[1] = v100;
            BYTE1(v135) = v58;
            if ( (_BYTE)v138 )
            {
              if ( (_BYTE)v138 != 1 )
                goto LABEL_70;
LABEL_98:
              v74 = _mm_loadu_si128(&v134);
              v62 = v135;
              v139 = _mm_loadu_si128(&v133);
              v141 = v135;
              v140 = v74;
              if ( (_BYTE)v135 )
              {
                v64 = v144;
                if ( (_BYTE)v144 )
                {
                  if ( (_BYTE)v135 == 1 )
                  {
                    v75 = _mm_loadu_si128(&v143);
                    v145 = _mm_loadu_si128(&v142);
                    v147 = v144;
                    v146 = v75;
                    goto LABEL_80;
                  }
LABEL_74:
                  if ( v64 == 1 )
                  {
                    v73 = _mm_loadu_si128(&v140);
                    v145 = _mm_loadu_si128(&v139);
                    v147 = v141;
                    v146 = v73;
                  }
                  else
                  {
                    if ( BYTE1(v141) == 1 )
                    {
                      v97 = v139.m128i_i64[1];
                      v65 = (__m128i *)v139.m128i_i64[0];
                    }
                    else
                    {
                      v65 = &v139;
                      v62 = 2;
                    }
                    if ( BYTE1(v144) == 1 )
                    {
                      v96 = v142.m128i_i64[1];
                      v66 = (__m128i *)v142.m128i_i64[0];
                    }
                    else
                    {
                      v66 = &v142;
                      v64 = 2;
                    }
                    v146.m128i_i64[0] = (__int64)v66;
                    v145.m128i_i64[0] = (__int64)v65;
                    v145.m128i_i64[1] = v97;
                    v146.m128i_i64[1] = v96;
                    LOBYTE(v147) = v62;
                    BYTE1(v147) = v64;
                  }
LABEL_80:
                  sub_C64D30((__int64)&v145, 1u);
                }
              }
LABEL_96:
              LOWORD(v147) = 256;
              goto LABEL_80;
            }
          }
        }
        else
        {
          LOWORD(v135) = 256;
        }
        LOWORD(v141) = 256;
        goto LABEL_96;
      }
    }
LABEL_11:
    ++v25;
  }
}
