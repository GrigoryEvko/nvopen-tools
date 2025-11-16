// Function: sub_1AE6420
// Address: 0x1ae6420
//
__int64 __fastcall sub_1AE6420(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        double a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 *v16; // r15
  __int64 *v17; // r13
  unsigned int v18; // r14d
  unsigned int *v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rsi
  __m128 v22; // xmm0
  __int64 v23; // rdx
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v27; // rax
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  int v30; // eax
  __int64 *v31; // rdi
  double v32; // xmm0_8
  __m128 v33; // xmm0
  __int64 *v34; // r9
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 **v37; // rdx
  int v38; // eax
  unsigned int v39; // xmm0_4
  _QWORD *v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r11
  __int64 v47; // rdi
  bool v48; // al
  unsigned __int8 *v49; // rsi
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rbx
  __int64 v53; // r12
  __int64 v54; // rdx
  __int64 v55; // r13
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // r14
  __int64 v59; // r13
  __int64 v60; // r12
  __int64 v61; // rbx
  __int64 v62; // r9
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rsi
  __int64 v66; // rdx
  unsigned __int8 *v67; // rsi
  __int64 v68; // rbx
  __int64 v69; // r12
  __int64 v70; // rdx
  __int64 v71; // r13
  __int64 v72; // rbx
  __int64 v73; // r14
  __int64 v74; // r12
  __int64 v75; // rbx
  __int64 v76; // rbx
  __int64 v77; // r12
  __int64 v78; // rdx
  __int64 v79; // r13
  __int64 v80; // rbx
  __int64 v81; // r14
  __int64 v82; // r13
  __int64 v83; // r12
  __int64 v84; // rbx
  unsigned __int8 v85; // [rsp+Fh] [rbp-1E1h]
  unsigned __int8 v86; // [rsp+10h] [rbp-1E0h]
  __int64 v87; // [rsp+10h] [rbp-1E0h]
  __int64 v88; // [rsp+18h] [rbp-1D8h]
  unsigned int *v89; // [rsp+18h] [rbp-1D8h]
  unsigned int *v90; // [rsp+20h] [rbp-1D0h]
  __int64 *v91; // [rsp+20h] [rbp-1D0h]
  unsigned __int8 v92; // [rsp+28h] [rbp-1C8h]
  __int64 *v93; // [rsp+28h] [rbp-1C8h]
  __int64 v94; // [rsp+28h] [rbp-1C8h]
  __int16 *v95; // [rsp+30h] [rbp-1C0h]
  __int64 v96; // [rsp+30h] [rbp-1C0h]
  __int64 v97; // [rsp+30h] [rbp-1C0h]
  unsigned int *v98; // [rsp+38h] [rbp-1B8h]
  __int64 v99; // [rsp+38h] [rbp-1B8h]
  __int64 v100; // [rsp+40h] [rbp-1B0h]
  __int64 *v101; // [rsp+40h] [rbp-1B0h]
  __int64 v102; // [rsp+40h] [rbp-1B0h]
  __int64 v103; // [rsp+48h] [rbp-1A8h]
  __int64 v104; // [rsp+48h] [rbp-1A8h]
  void *v105; // [rsp+50h] [rbp-1A0h]
  __int16 *v106; // [rsp+50h] [rbp-1A0h]
  __int64 v107; // [rsp+50h] [rbp-1A0h]
  void *v108; // [rsp+58h] [rbp-198h]
  __int64 ***v109; // [rsp+60h] [rbp-190h]
  __int64 v110; // [rsp+60h] [rbp-190h]
  __int64 v111; // [rsp+60h] [rbp-190h]
  __int64 *v112; // [rsp+68h] [rbp-188h]
  _BYTE *v113; // [rsp+70h] [rbp-180h]
  __int64 v114; // [rsp+70h] [rbp-180h]
  __int64 v115; // [rsp+70h] [rbp-180h]
  __int64 *v116; // [rsp+70h] [rbp-180h]
  __int64 v117; // [rsp+70h] [rbp-180h]
  __int64 *v118; // [rsp+70h] [rbp-180h]
  __int64 *v119; // [rsp+70h] [rbp-180h]
  unsigned int v120; // [rsp+84h] [rbp-16Ch] BYREF
  unsigned __int8 *v121; // [rsp+88h] [rbp-168h] BYREF
  __int64 v122[2]; // [rsp+90h] [rbp-160h] BYREF
  __int16 v123; // [rsp+A0h] [rbp-150h]
  unsigned __int8 *v124; // [rsp+B0h] [rbp-140h] BYREF
  void *v125; // [rsp+B8h] [rbp-138h] BYREF
  __int64 v126; // [rsp+C0h] [rbp-130h]
  unsigned __int8 *v127; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v128; // [rsp+D8h] [rbp-118h]
  __int64 *v129; // [rsp+E0h] [rbp-110h]
  __int64 v130; // [rsp+E8h] [rbp-108h]
  __int64 v131; // [rsp+F0h] [rbp-100h]
  int v132; // [rsp+F8h] [rbp-F8h]
  __int64 v133; // [rsp+100h] [rbp-F0h]
  __int64 v134; // [rsp+108h] [rbp-E8h]
  _QWORD v135[2]; // [rsp+120h] [rbp-D0h] BYREF
  __int64 *v136; // [rsp+130h] [rbp-C0h]
  __int64 v137; // [rsp+138h] [rbp-B8h]
  _BYTE v138[176]; // [rsp+140h] [rbp-B0h] BYREF

  v11 = *(_QWORD *)(a1 + 80);
  v136 = (__int64 *)v138;
  v135[0] = a2;
  v135[1] = a3;
  v137 = 0x1000000000LL;
  if ( a1 + 72 == v11 )
    return 0;
  do
  {
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 8);
    v13 = *(_QWORD *)(v12 + 24);
    v14 = v12 + 16;
    while ( v14 != v13 )
    {
      while ( 1 )
      {
        v15 = v13;
        v13 = *(_QWORD *)(v13 + 8);
        if ( *(_BYTE *)(v15 - 8) != 78 )
          break;
        sub_1AE5870((__int64)v135, v15 - 24);
        if ( v14 == v13 )
          goto LABEL_6;
      }
    }
LABEL_6:
    ;
  }
  while ( a1 + 72 != v11 );
  v16 = &v136[(unsigned int)v137];
  if ( v136 == v16 )
  {
    v18 = 0;
    goto LABEL_22;
  }
  v17 = v136;
  v18 = 0;
  v19 = &v120;
  do
  {
    v20 = *v17;
    v21 = *(_QWORD *)(*v17 - 24);
    if ( *(_BYTE *)(v21 + 16) )
      v21 = 0;
    sub_149CB50(*(_QWORD *)v135[0], v21, v19);
    if ( v120 > 0xAA )
    {
      if ( v120 > 0x163 )
        goto LABEL_32;
      if ( v120 > 0x160 )
      {
        v22 = 0;
        v23 = sub_1AE5D50(v20, 4u, 0.0);
        goto LABEL_19;
      }
      if ( v120 > 0x159 )
      {
        if ( v120 == 349 )
        {
LABEL_28:
          a5.m128i_i64[0] = 4286578688LL;
          v22 = (__m128)0x7F800000u;
          v23 = sub_1AE5F80(v20, 1u, 1u, INFINITY, (__m128i)0xFF800000, a6);
          goto LABEL_19;
        }
LABEL_32:
        if ( v120 - 346 <= 2 )
        {
          if ( v120 == 347 )
          {
LABEL_60:
            a5 = (__m128i)0xC2B20000;
            v22 = (__m128)0x42B20000u;
          }
          else if ( v120 == 348 )
          {
LABEL_58:
            a5 = (__m128i)0xC6317400;
            v22 = (__m128)0x46317400u;
          }
          else
          {
LABEL_95:
            a5 = (__m128i)0xC4318000;
            v22 = (__m128)0x44318000u;
          }
LABEL_59:
          v23 = sub_1AE5F80(v20, 2u, 4u, v22.m128_f32[0], a5, a6);
          goto LABEL_19;
        }
LABEL_33:
        if ( v120 <= 0x112 )
        {
          if ( v120 > 0x10F )
          {
            v33 = (__m128)0xBF800000;
            v34 = (__int64 *)sub_1AE5D50(v20, 5u, -1.0);
            goto LABEL_53;
          }
          goto LABEL_50;
        }
        if ( v120 <= 0x11A )
          goto LABEL_62;
        if ( v120 - 313 > 2 || v120 != 313 )
          goto LABEL_20;
        v113 = *(_BYTE **)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
        v109 = *(__int64 ****)(v20 + 24 * (1LL - (*(_DWORD *)(v20 + 20) & 0xFFFFFFF)));
        v27 = sub_16498A0(v20);
        v133 = 0;
        v134 = 0;
        v28 = *(unsigned __int8 **)(v20 + 48);
        v130 = v27;
        v132 = 0;
        v29 = *(_QWORD *)(v20 + 40);
        v127 = 0;
        v128 = v29;
        v131 = 0;
        v129 = (__int64 *)(v20 + 24);
        v124 = v28;
        if ( v28 )
        {
          sub_1623A60((__int64)&v124, (__int64)v28, 2);
          if ( v127 )
            sub_161E7C0((__int64)&v127, (__int64)v127);
          v127 = v124;
          if ( v124 )
            sub_1623210((__int64)&v124, v124, (__int64)&v127);
        }
        v30 = (unsigned __int8)v113[16];
        if ( (_BYTE)v30 == 14 )
        {
          v108 = sub_16982C0();
          if ( *((void **)v113 + 4) == v108 )
            v31 = (__int64 *)(*((_QWORD *)v113 + 5) + 8LL);
          else
            v31 = (__int64 *)(v113 + 32);
          v32 = sub_169D8E0(v31);
          a5.m128i_i64[0] = 0x3FF0000000000000LL;
          if ( v32 < 1.0 || v32 > 255.0 )
          {
LABEL_46:
            if ( v127 )
              sub_161E7C0((__int64)&v127, (__int64)v127);
            goto LABEL_20;
          }
          v33 = (__m128)0x42FE0000u;
          v106 = (__int16 *)sub_1698270();
          sub_169D3B0((__int64)v122, (__m128i)0x42FE0000u);
          sub_169E320(&v125, v122, v106);
          sub_1698460((__int64)v122);
          v50 = (_QWORD *)sub_16498A0(v20);
          v115 = sub_159CCF0(v50, (__int64)&v124);
          if ( v108 == v125 )
          {
            v107 = v126;
            if ( v126 )
            {
              if ( v126 != v126 + 32LL * *(_QWORD *)(v126 - 8) )
              {
                v98 = v19;
                v52 = v126 + 32LL * *(_QWORD *)(v126 - 8);
                v104 = v20;
                v101 = v17;
                do
                {
                  v52 -= 32;
                  if ( v108 == *(void **)(v52 + 8) )
                  {
                    v53 = *(_QWORD *)(v52 + 16);
                    if ( v53 )
                    {
                      v54 = 32LL * *(_QWORD *)(v53 - 8);
                      v55 = v53 + v54;
                      if ( v53 != v53 + v54 )
                      {
                        v96 = v52;
                        do
                        {
                          v55 -= 32;
                          if ( v108 == *(void **)(v55 + 8) )
                          {
                            v56 = *(_QWORD *)(v55 + 16);
                            if ( v56 )
                            {
                              v57 = 32LL * *(_QWORD *)(v56 - 8);
                              if ( v56 != v56 + v57 )
                              {
                                v92 = v18;
                                v58 = v55;
                                v59 = v53;
                                v60 = v56;
                                v61 = v56 + v57;
                                do
                                {
                                  v61 -= 32;
                                  sub_127D120((_QWORD *)(v61 + 8));
                                }
                                while ( v60 != v61 );
                                v56 = v60;
                                v53 = v59;
                                v55 = v58;
                                v18 = v92;
                              }
                              j_j_j___libc_free_0_0(v56 - 8);
                            }
                          }
                          else
                          {
                            sub_1698460(v55 + 8);
                          }
                        }
                        while ( v53 != v55 );
                        v52 = v96;
                      }
                      j_j_j___libc_free_0_0(v53 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v52 + 8);
                  }
                }
                while ( v107 != v52 );
                v20 = v104;
                v17 = v101;
                v19 = v98;
              }
              j_j_j___libc_free_0_0(v107 - 8);
            }
          }
          else
          {
            sub_1698460((__int64)&v125);
          }
          if ( *((_BYTE *)*v109 + 8) != 2 )
            v115 = sub_15A3E10(v115, *v109, 0);
          LOWORD(v126) = 257;
          v51 = sub_1289B20((__int64 *)&v127, 2u, v109, v115, (__int64)&v124, 0);
          v49 = v127;
          v34 = (__int64 *)v51;
LABEL_103:
          if ( v49 )
          {
            v116 = v34;
            sub_161E7C0((__int64)&v127, (__int64)v49);
            v34 = v116;
          }
LABEL_105:
          if ( v34 )
            goto LABEL_53;
          goto LABEL_20;
        }
        if ( (unsigned __int8)v30 <= 0x17u || (unsigned int)(v30 - 65) > 1 )
          goto LABEL_46;
        if ( (v113[23] & 0x40) != 0 )
          v37 = (__int64 **)*((_QWORD *)v113 - 1);
        else
          v37 = (__int64 **)&v113[-24 * (*((_DWORD *)v113 + 5) & 0xFFFFFFF)];
        v38 = sub_1643030(**v37);
        switch ( v38 )
        {
          case 8:
            v39 = 1124073472;
            break;
          case 16:
            v39 = 1115684864;
            break;
          case 32:
            v39 = 1107296256;
            break;
          default:
            goto LABEL_46;
        }
        v95 = (__int16 *)sub_1698270();
        sub_169D3B0((__int64)v122, (__m128i)v39);
        sub_169E320(&v125, v122, v95);
        sub_1698460((__int64)v122);
        v40 = (_QWORD *)sub_16498A0(v20);
        v103 = sub_159CCF0(v40, (__int64)&v124);
        v105 = sub_16982C0();
        if ( v125 == v105 )
        {
          v102 = v126;
          if ( v126 )
          {
            if ( v126 != v126 + 32LL * *(_QWORD *)(v126 - 8) )
            {
              v89 = v19;
              v76 = v126 + 32LL * *(_QWORD *)(v126 - 8);
              v94 = v20;
              v91 = v17;
              do
              {
                v76 -= 32;
                if ( v105 == *(void **)(v76 + 8) )
                {
                  v77 = *(_QWORD *)(v76 + 16);
                  if ( v77 )
                  {
                    v78 = 32LL * *(_QWORD *)(v77 - 8);
                    v79 = v77 + v78;
                    if ( v77 != v77 + v78 )
                    {
                      v87 = v76;
                      do
                      {
                        v79 -= 32;
                        if ( v105 == *(void **)(v79 + 8) )
                        {
                          v80 = *(_QWORD *)(v79 + 16);
                          if ( v80 )
                          {
                            if ( v80 != v80 + 32LL * *(_QWORD *)(v80 - 8) )
                            {
                              v85 = v18;
                              v81 = v79;
                              v82 = v77;
                              v83 = v80;
                              v84 = v80 + 32LL * *(_QWORD *)(v80 - 8);
                              do
                              {
                                v84 -= 32;
                                sub_127D120((_QWORD *)(v84 + 8));
                              }
                              while ( v83 != v84 );
                              v80 = v83;
                              v77 = v82;
                              v79 = v81;
                              v18 = v85;
                            }
                            j_j_j___libc_free_0_0(v80 - 8);
                          }
                        }
                        else
                        {
                          sub_1698460(v79 + 8);
                        }
                      }
                      while ( v77 != v79 );
                      v76 = v87;
                    }
                    j_j_j___libc_free_0_0(v77 - 8);
                  }
                }
                else
                {
                  sub_1698460(v76 + 8);
                }
              }
              while ( v102 != v76 );
              v20 = v94;
              v17 = v91;
              v19 = v89;
            }
            j_j_j___libc_free_0_0(v102 - 8);
          }
        }
        else
        {
          sub_1698460((__int64)&v125);
        }
        v33 = 0;
        sub_169D3B0((__int64)v122, (__m128i)0LL);
        sub_169E320(&v125, v122, v95);
        sub_1698460((__int64)v122);
        v41 = (_QWORD *)sub_16498A0(v20);
        v100 = sub_159CCF0(v41, (__int64)&v124);
        if ( v125 == v105 )
        {
          v99 = v126;
          if ( v126 )
          {
            if ( v126 != v126 + 32LL * *(_QWORD *)(v126 - 8) )
            {
              v90 = v19;
              v68 = v126 + 32LL * *(_QWORD *)(v126 - 8);
              v97 = v20;
              v93 = v17;
              do
              {
                v68 -= 32;
                if ( v105 == *(void **)(v68 + 8) )
                {
                  v69 = *(_QWORD *)(v68 + 16);
                  if ( v69 )
                  {
                    v70 = 32LL * *(_QWORD *)(v69 - 8);
                    v71 = v69 + v70;
                    if ( v69 != v69 + v70 )
                    {
                      v88 = v68;
                      do
                      {
                        v71 -= 32;
                        if ( v105 == *(void **)(v71 + 8) )
                        {
                          v72 = *(_QWORD *)(v71 + 16);
                          if ( v72 )
                          {
                            if ( v72 != v72 + 32LL * *(_QWORD *)(v72 - 8) )
                            {
                              v86 = v18;
                              v73 = v69;
                              v74 = *(_QWORD *)(v71 + 16);
                              v75 = v72 + 32LL * *(_QWORD *)(v72 - 8);
                              do
                              {
                                v75 -= 32;
                                sub_127D120((_QWORD *)(v75 + 8));
                              }
                              while ( v74 != v75 );
                              v72 = v74;
                              v69 = v73;
                              v18 = v86;
                            }
                            j_j_j___libc_free_0_0(v72 - 8);
                          }
                        }
                        else
                        {
                          sub_1698460(v71 + 8);
                        }
                      }
                      while ( v69 != v71 );
                      v68 = v88;
                    }
                    j_j_j___libc_free_0_0(v69 - 8);
                  }
                }
                else
                {
                  sub_1698460(v68 + 8);
                }
              }
              while ( v99 != v68 );
              v20 = v97;
              v17 = v93;
              v19 = v90;
            }
            j_j_j___libc_free_0_0(v99 - 8);
          }
        }
        else
        {
          sub_1698460((__int64)&v125);
        }
        if ( *((_BYTE *)*v109 + 8) != 2 )
          v103 = sub_15A3E10(v103, *v109, 0);
        if ( *(_BYTE *)(*(_QWORD *)v113 + 8LL) != 2 )
          v100 = sub_15A3E10(v100, *v109, 0);
        LOWORD(v126) = 257;
        v42 = sub_1289B20((__int64 *)&v127, 2u, v109, v103, (__int64)&v124, 0);
        LOWORD(v126) = 257;
        v110 = v42;
        v43 = sub_1289B20((__int64 *)&v127, 5u, v113, v100, (__int64)&v124, 0);
        v46 = v110;
        v123 = 257;
        v34 = (__int64 *)v43;
        if ( *(_BYTE *)(v110 + 16) > 0x10u )
          goto LABEL_128;
        v47 = v110;
        v114 = v110;
        v111 = v43;
        v48 = sub_1593BB0(v47, 257, v44, v45);
        v46 = v114;
        v34 = (__int64 *)v111;
        if ( !v48 )
        {
          if ( *(_BYTE *)(v111 + 16) <= 0x10u )
          {
            v34 = (__int64 *)sub_15A2D10((__int64 *)v111, v114, 0.0, *(double *)a5.m128i_i64, a6);
            goto LABEL_85;
          }
LABEL_128:
          LOWORD(v126) = 257;
          v62 = sub_15FB440(27, v34, v46, (__int64)&v124, 0);
          if ( v128 )
          {
            v117 = v62;
            v112 = v129;
            sub_157E9D0(v128 + 40, v62);
            v62 = v117;
            v63 = *(_QWORD *)(v117 + 24);
            v64 = *v112;
            *(_QWORD *)(v117 + 32) = v112;
            v64 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v117 + 24) = v64 | v63 & 7;
            *(_QWORD *)(v64 + 8) = v117 + 24;
            *v112 = *v112 & 7 | (v117 + 24);
          }
          v118 = (__int64 *)v62;
          sub_164B780(v62, v122);
          v34 = v118;
          if ( !v127 )
            goto LABEL_105;
          v121 = v127;
          sub_1623A60((__int64)&v121, (__int64)v127, 2);
          v34 = v118;
          v65 = v118[6];
          v66 = (__int64)(v118 + 6);
          if ( v65 )
          {
            sub_161E7C0((__int64)(v118 + 6), v65);
            v34 = v118;
            v66 = (__int64)(v118 + 6);
          }
          v67 = v121;
          v34[6] = (__int64)v121;
          if ( v67 )
          {
            v119 = v34;
            sub_1623210((__int64)&v121, v67, v66);
            v34 = v119;
          }
        }
LABEL_85:
        v49 = v127;
        goto LABEL_103;
      }
      if ( v120 > 0x157 )
        goto LABEL_28;
      if ( v120 > 0xB7 )
        goto LABEL_33;
LABEL_17:
      if ( v120 > 0xB4 )
      {
        v22 = (__m128)dword_42C6340[v120 - 181];
        v23 = sub_1AE5D50(v20, 2u, v22.m128_f32[0]);
LABEL_19:
        v18 = 1;
        sub_1AE56C0((__int64)v135, (_QWORD *)v20, v23, v22, *(double *)a5.m128i_i64, a6, a7, v24, v25, a10, a11);
        goto LABEL_20;
      }
    }
    else if ( v120 > 0x77 )
    {
      switch ( v120 )
      {
        case 0x78u:
        case 0x79u:
        case 0x7Du:
        case 0x7Eu:
        case 0x7Fu:
        case 0x83u:
          a5.m128i_i64[0] = 1065353216;
          v22 = (__m128)0xBF800000;
          v23 = sub_1AE5F80(v20, 4u, 2u, -1.0, (__m128i)0x3F800000u, a6);
          goto LABEL_19;
        case 0x7Au:
        case 0x7Bu:
        case 0x7Cu:
          v22 = (__m128)0x3F800000u;
          v23 = sub_1AE5D50(v20, 4u, 1.0);
          goto LABEL_19;
        case 0xA5u:
        case 0xA6u:
        case 0xAAu:
          goto LABEL_28;
        default:
          goto LABEL_17;
      }
    }
    if ( v120 > 0xA9 )
    {
      if ( v120 - 172 <= 8 )
      {
LABEL_55:
        switch ( v120 )
        {
          case 0xA7u:
            goto LABEL_95;
          case 0xA8u:
            goto LABEL_60;
          case 0xA9u:
            goto LABEL_58;
          case 0xACu:
            a5 = (__m128i)0xC43A4000;
            v22 = (__m128)0x44314000u;
            goto LABEL_59;
          case 0xADu:
            a5 = (__m128i)0xC3A18000;
            v22 = (__m128)0x439A0000u;
            goto LABEL_59;
          case 0xAEu:
            a5 = (__m128i)0xC2340000;
            v22 = (__m128)0x42180000u;
            goto LABEL_59;
          case 0xAFu:
            a5 = (__m128i)0xC59AB000;
            v22 = (__m128)0x459A2000u;
            goto LABEL_59;
          case 0xB0u:
            a5 = (__m128i)0xC4864000;
            v22 = (__m128)0x447FC000u;
            goto LABEL_59;
          case 0xB1u:
            a5 = (__m128i)0xC3150000;
            v22 = (__m128)0x42FE0000u;
            goto LABEL_59;
          case 0xB2u:
            a5 = (__m128i)0xC6807A00;
            v22 = (__m128)0x4631DC00u;
            goto LABEL_59;
          case 0xB3u:
            a5 = (__m128i)0xC2CE0000;
            v22 = (__m128)0x42B00000u;
            goto LABEL_59;
          case 0xB4u:
            a5 = (__m128i)0xC6321C00;
            v22 = (__m128)0x46317000u;
            goto LABEL_59;
          default:
            BUG();
        }
      }
      goto LABEL_61;
    }
    if ( v120 > 0xA6 )
      goto LABEL_55;
LABEL_50:
    if ( v120 <= 0x8B )
    {
      if ( v120 <= 0x88 )
        goto LABEL_20;
      a5.m128i_i64[0] = 1065353216;
      v33 = (__m128)0xBF800000;
      v34 = (__int64 *)sub_1AE5F80(v20, 5u, 3u, -1.0, (__m128i)0x3F800000u, a6);
      goto LABEL_53;
    }
LABEL_61:
    if ( v120 - 268 <= 3 )
    {
LABEL_62:
      v33 = 0;
      v34 = (__int64 *)sub_1AE5D50(v20, 5u, 0.0);
LABEL_53:
      v18 = 1;
      sub_1AE56C0((__int64)v135, (_QWORD *)v20, (__int64)v34, v33, *(double *)a5.m128i_i64, a6, a7, v35, v36, a10, a11);
    }
LABEL_20:
    ++v17;
  }
  while ( v16 != v17 );
  v16 = v136;
LABEL_22:
  if ( v16 != (__int64 *)v138 )
    _libc_free((unsigned __int64)v16);
  return v18;
}
