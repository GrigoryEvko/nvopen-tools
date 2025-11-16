// Function: sub_21C72F0
// Address: 0x21c72f0
//
__int64 __fastcall sub_21C72F0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // r15
  bool v9; // zf
  unsigned int v10; // r13d
  char v11; // r14
  __int64 v12; // rdi
  unsigned int v14; // eax
  __int64 v15; // rdx
  int v16; // r8d
  __int64 v17; // rcx
  unsigned int v18; // eax
  bool v19; // dl
  int v20; // esi
  unsigned int v21; // eax
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rdx
  unsigned __int8 *v26; // rax
  __int64 v27; // r11
  __int16 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdi
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdi
  int v40; // edx
  __int64 v41; // rax
  __m128i v42; // xmm3
  int v43; // edx
  __m128i v44; // xmm4
  _QWORD *v45; // rdi
  __int64 v46; // r9
  __int64 v47; // r14
  _QWORD *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  unsigned int v53; // eax
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rdi
  int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rdi
  int v60; // edx
  __int64 v61; // rax
  __int64 v62; // rdi
  int v63; // edx
  __int64 v64; // rax
  __int64 v65; // rdi
  int v66; // edx
  __int64 v67; // rax
  __m128i v68; // xmm1
  __m128i v69; // xmm2
  int v70; // edx
  _QWORD *v71; // rdi
  __int64 v72; // r9
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdi
  int v76; // edx
  __int64 v77; // rax
  __int64 v78; // rdi
  int v79; // edx
  __int64 v80; // rax
  __int64 v81; // rdi
  int v82; // edx
  __int64 v83; // rax
  __int64 v84; // rdi
  int v85; // edx
  __int64 v86; // rax
  __m128i v87; // xmm0
  int v88; // edx
  _QWORD *v89; // rdi
  __int64 v90; // r9
  __int64 v91; // r11
  __int64 v92; // rax
  __int64 v93; // rdi
  int v94; // edx
  __int64 v95; // rax
  __int64 v96; // rdi
  int v97; // edx
  __int64 v98; // rax
  __int64 v99; // rdi
  int v100; // edx
  __int64 v101; // rax
  __int64 v102; // rdi
  int v103; // edx
  int v104; // edx
  _QWORD *v105; // rdi
  __int64 v106; // r9
  __int128 v107; // [rsp-10h] [rbp-180h]
  __int128 v108; // [rsp-10h] [rbp-180h]
  __int128 v109; // [rsp-10h] [rbp-180h]
  __int128 v110; // [rsp-10h] [rbp-180h]
  __int64 v111; // [rsp+20h] [rbp-150h]
  __int32 v112; // [rsp+2Ch] [rbp-144h]
  __int64 v113; // [rsp+30h] [rbp-140h]
  int v114; // [rsp+38h] [rbp-138h]
  __int16 v115; // [rsp+38h] [rbp-138h]
  unsigned int v116; // [rsp+40h] [rbp-130h]
  unsigned int v117; // [rsp+44h] [rbp-12Ch]
  __int64 v118; // [rsp+48h] [rbp-128h]
  __int64 v119; // [rsp+48h] [rbp-128h]
  __int64 v120; // [rsp+48h] [rbp-128h]
  __int16 v121; // [rsp+48h] [rbp-128h]
  __int16 v122; // [rsp+48h] [rbp-128h]
  unsigned int v123; // [rsp+50h] [rbp-120h]
  char v124; // [rsp+57h] [rbp-119h]
  unsigned __int8 v125; // [rsp+57h] [rbp-119h]
  bool v126; // [rsp+58h] [rbp-118h]
  __int64 v127; // [rsp+58h] [rbp-118h]
  __int64 v128; // [rsp+60h] [rbp-110h] BYREF
  __int64 v129; // [rsp+68h] [rbp-108h] BYREF
  __int64 v130; // [rsp+70h] [rbp-100h] BYREF
  __int64 v131; // [rsp+78h] [rbp-F8h] BYREF
  __int64 v132; // [rsp+80h] [rbp-F0h] BYREF
  int v133; // [rsp+88h] [rbp-E8h]
  __m128i v134; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v135; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v136; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v137; // [rsp+C0h] [rbp-B0h] BYREF
  int v138; // [rsp+C8h] [rbp-A8h]
  __int64 v139; // [rsp+D0h] [rbp-A0h]
  int v140; // [rsp+D8h] [rbp-98h]
  __int64 v141; // [rsp+E0h] [rbp-90h]
  int v142; // [rsp+E8h] [rbp-88h]
  __int64 v143; // [rsp+F0h] [rbp-80h]
  int v144; // [rsp+F8h] [rbp-78h]
  __int64 v145; // [rsp+100h] [rbp-70h]
  int v146; // [rsp+108h] [rbp-68h]
  __m128i v147; // [rsp+110h] [rbp-60h]
  __m128i v148; // [rsp+120h] [rbp-50h]
  __int64 v149; // [rsp+130h] [rbp-40h]
  __int32 v150; // [rsp+138h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v132 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v132, v7, 2);
  v8 = 0;
  v9 = *(_WORD *)(a2 + 24) == 185;
  v133 = *(_DWORD *)(a2 + 64);
  if ( !v9 || (v10 = 0, v8 = a2, (*(_WORD *)(a2 + 26) & 0x380) == 0) )
  {
    v11 = *(_BYTE *)(a2 + 88);
    v10 = 0;
    if ( v11 )
    {
      v12 = *(_QWORD *)(a2 + 104);
      v10 = (unsigned __int8)byte_42880A0[8 * (*(_BYTE *)(v12 + 37) & 0xF) + 2];
      v124 = *(_BYTE *)(v12 + 37) & 0xF;
      if ( (_BYTE)v10 )
      {
LABEL_7:
        v10 = 0;
        goto LABEL_8;
      }
      v14 = sub_21BD7A0((_QWORD *)v12);
      v17 = v14;
      v123 = v14;
      v126 = v14 != 1 || *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) <= 0x1Fu;
      if ( v126 )
      {
        v118 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
        v18 = sub_1E340A0(*(_QWORD *)(a2 + 104));
        v114 = 8 * sub_15A9520(v118, v18);
        if ( (*(_BYTE *)(a2 + 26) & 8) != 0 )
          goto LABEL_14;
      }
      else
      {
        if ( (*(_BYTE *)(a2 + 26) & 0x40) != 0 || (unsigned __int8)sub_21BD8E0(a2, *(__int64 **)(a1 + 256)) )
        {
          v10 = sub_21C5A60((__int64 *)a1, a2, a3, a4, a5, v15, v17, v16);
          goto LABEL_8;
        }
        v127 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
        v53 = sub_1E340A0(*(_QWORD *)(a2 + 104));
        v54 = sub_15A9520(v127, v53);
        v126 = 1;
        v114 = 8 * v54;
        if ( (*(_BYTE *)(a2 + 26) & 8) != 0 )
        {
LABEL_18:
          if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
          {
            switch ( v11 )
            {
              case 24:
              case 25:
              case 26:
              case 27:
              case 28:
              case 29:
              case 30:
              case 31:
              case 32:
              case 62:
              case 63:
              case 64:
              case 65:
              case 66:
              case 67:
                v11 = 3;
                break;
              case 33:
              case 34:
              case 35:
              case 36:
              case 37:
              case 38:
              case 39:
              case 40:
              case 68:
              case 69:
              case 70:
              case 71:
              case 72:
              case 73:
                v11 = 4;
                break;
              case 41:
              case 42:
              case 43:
              case 44:
              case 45:
              case 46:
              case 47:
              case 48:
              case 74:
              case 75:
              case 76:
              case 77:
              case 78:
              case 79:
                v11 = 5;
                break;
              case 49:
              case 50:
              case 51:
              case 52:
              case 53:
              case 54:
              case 80:
              case 81:
              case 82:
              case 83:
              case 84:
              case 85:
                v11 = 6;
                break;
              case 55:
                v11 = 7;
                break;
              case 86:
              case 87:
              case 88:
              case 98:
              case 99:
              case 100:
                v11 = 8;
                break;
              case 89:
              case 90:
              case 91:
              case 92:
              case 93:
              case 101:
              case 102:
              case 103:
              case 104:
              case 105:
                v11 = 9;
                break;
              case 94:
              case 95:
              case 96:
              case 97:
              case 106:
              case 107:
              case 108:
              case 109:
                v11 = 10;
                break;
              default:
                v11 = 2;
                break;
            }
            v116 = 32;
          }
          else
          {
            v20 = 8;
            v21 = sub_21BD810(v11);
            if ( v21 >= 8 )
              v20 = v21;
            v116 = v20;
          }
          if ( !v8 || (v117 = 1, ((*(_BYTE *)(v8 + 27) >> 2) & 3) != 2) )
          {
            if ( (unsigned __int8)(v11 - 8) <= 5u || (v117 = 0, (unsigned __int8)(v11 - 86) <= 0x17u) )
              v117 = (v11 == 8) + 2;
          }
          v22 = *(__int64 **)(a2 + 32);
          v23 = *v22;
          v24 = v22[5];
          v134.m128i_i64[0] = 0;
          v135.m128i_i64[0] = 0;
          v113 = v23;
          LODWORD(v23) = *((_DWORD *)v22 + 2);
          v134.m128i_i32[2] = 0;
          v112 = v23;
          v25 = v22[6];
          v135.m128i_i32[2] = 0;
          v26 = *(unsigned __int8 **)(a2 + 40);
          v136.m128i_i64[0] = 0;
          v136.m128i_i32[2] = 0;
          v119 = v25;
          v125 = *v26;
          if ( sub_21C2A00(a1, v24, v25, (__int64)&v134) )
          {
            v131 = 0x100000BE8LL;
            v130 = 0x100000BDCLL;
            v129 = 0x100000BD6LL;
            v128 = 0x100000BFALL;
            sub_21BD570(
              (__int64)&v137,
              v125,
              3072,
              3054,
              3060,
              (__int64)&v128,
              (__int64)&v129,
              (__int64)&v130,
              3042,
              (__int64)&v131);
            if ( !BYTE4(v137) )
              goto LABEL_7;
            v122 = v137;
            v74 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v126, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v75 = *(_QWORD *)(a1 + 272);
            v138 = v76;
            v137 = v74;
            v77 = sub_1D38BB0(v75, v123, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v78 = *(_QWORD *)(a1 + 272);
            v140 = v79;
            v139 = v77;
            v80 = sub_1D38BB0(v78, 1, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v81 = *(_QWORD *)(a1 + 272);
            v142 = v82;
            v141 = v80;
            v83 = sub_1D38BB0(v81, v117, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v84 = *(_QWORD *)(a1 + 272);
            v144 = v85;
            v143 = v83;
            v86 = sub_1D38BB0(v84, v116, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v87 = _mm_loadu_si128(&v134);
            v146 = v88;
            v89 = *(_QWORD **)(a1 + 272);
            v148.m128i_i64[0] = v113;
            *((_QWORD *)&v109 + 1) = 7;
            *(_QWORD *)&v109 = &v137;
            v148.m128i_i32[2] = v112;
            v145 = v86;
            v147 = v87;
            v73 = sub_1D25BD0(v89, v122, (__int64)&v132, v125, 0, v90, 1u, v109);
          }
          else
          {
            if ( v114 == 64 )
            {
              if ( !(unsigned __int8)sub_21C2BC0(
                                       a1,
                                       v24,
                                       v24,
                                       v119,
                                       (__int64)&v136,
                                       (__int64)&v135,
                                       a3,
                                       *(double *)a4.m128i_i64,
                                       a5) )
              {
                if ( (unsigned __int8)sub_21C2F80(
                                        a1,
                                        v24,
                                        v24,
                                        v119,
                                        (__int64)&v136,
                                        (__int64)&v135,
                                        a3,
                                        *(double *)a4.m128i_i64,
                                        a5) )
                {
                  v131 = 0x100000BE6LL;
                  v130 = 0x100000BDALL;
                  v129 = 0x100000BD4LL;
                  v128 = 0x100000BF8LL;
                  sub_21BD570(
                    (__int64)&v137,
                    v125,
                    3070,
                    3052,
                    3058,
                    (__int64)&v128,
                    (__int64)&v129,
                    (__int64)&v130,
                    3040,
                    (__int64)&v131);
                  goto LABEL_31;
                }
                v131 = 0x100000BE4LL;
                v130 = 0x100000BD8LL;
                v129 = 0x100000BD2LL;
                v128 = 0x100000BF6LL;
                sub_21BD570(
                  (__int64)&v137,
                  v125,
                  3068,
                  3050,
                  3056,
                  (__int64)&v128,
                  (__int64)&v129,
                  (__int64)&v130,
                  3038,
                  (__int64)&v131);
                goto LABEL_61;
              }
            }
            else if ( !(unsigned __int8)sub_21C2BA0(
                                          a1,
                                          v24,
                                          v24,
                                          v119,
                                          (__int64)&v136,
                                          (__int64)&v135,
                                          a3,
                                          *(double *)a4.m128i_i64,
                                          a5) )
            {
              if ( (unsigned __int8)sub_21C2F60(
                                      a1,
                                      v24,
                                      v24,
                                      v119,
                                      (__int64)&v136,
                                      (__int64)&v135,
                                      a3,
                                      *(double *)a4.m128i_i64,
                                      a5) )
              {
                v131 = 0x100000BE5LL;
                v130 = 0x100000BD9LL;
                v129 = 0x100000BD3LL;
                v128 = 0x100000BF7LL;
                sub_21BD570(
                  (__int64)&v137,
                  v125,
                  3069,
                  3051,
                  3057,
                  (__int64)&v128,
                  (__int64)&v129,
                  (__int64)&v130,
                  3039,
                  (__int64)&v131);
LABEL_31:
                if ( !BYTE4(v137) )
                  goto LABEL_7;
                v120 = v27;
                v28 = v137;
                v29 = sub_1D38BB0(
                        *(_QWORD *)(a1 + 272),
                        v126,
                        (__int64)&v132,
                        5,
                        0,
                        1,
                        a3,
                        *(double *)a4.m128i_i64,
                        a5,
                        0);
                v30 = *(_QWORD *)(a1 + 272);
                v138 = v31;
                v137 = v29;
                v32 = sub_1D38BB0(v30, v123, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
                v33 = *(_QWORD *)(a1 + 272);
                v140 = v34;
                v139 = v32;
                v35 = sub_1D38BB0(v33, 1, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
                v36 = *(_QWORD *)(a1 + 272);
                v142 = v37;
                v141 = v35;
                v38 = sub_1D38BB0(v36, v117, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
                v39 = *(_QWORD *)(a1 + 272);
                v144 = v40;
                v143 = v38;
                v41 = sub_1D38BB0(v39, v116, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
                v42 = _mm_loadu_si128(&v136);
                v145 = v41;
                v146 = v43;
                *((_QWORD *)&v107 + 1) = 8;
                v44 = _mm_loadu_si128(&v135);
                *(_QWORD *)&v107 = v120;
                v45 = *(_QWORD **)(a1 + 272);
                v149 = v113;
                v147 = v42;
                v150 = v112;
                v148 = v44;
                v47 = sub_1D25BD0(v45, v28, (__int64)&v132, v125, 0, v46, 1u, v107);
                goto LABEL_33;
              }
              v131 = 0x100000BE3LL;
              v130 = 0x100000BD7LL;
              v129 = 0x100000BD1LL;
              v128 = 0x100000BF5LL;
              sub_21BD570(
                (__int64)&v137,
                v125,
                3067,
                3049,
                3055,
                (__int64)&v128,
                (__int64)&v129,
                (__int64)&v130,
                3037,
                (__int64)&v131);
LABEL_61:
              if ( !BYTE4(v137) )
                goto LABEL_7;
              v111 = v91;
              v115 = v137;
              v92 = sub_1D38BB0(
                      *(_QWORD *)(a1 + 272),
                      v126,
                      (__int64)&v132,
                      5,
                      0,
                      1,
                      a3,
                      *(double *)a4.m128i_i64,
                      a5,
                      0);
              v93 = *(_QWORD *)(a1 + 272);
              v138 = v94;
              v137 = v92;
              v95 = sub_1D38BB0(v93, v123, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
              v96 = *(_QWORD *)(a1 + 272);
              v140 = v97;
              v139 = v95;
              v98 = sub_1D38BB0(v96, 1, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
              v99 = *(_QWORD *)(a1 + 272);
              v142 = v100;
              v141 = v98;
              v101 = sub_1D38BB0(v99, v117, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
              v102 = *(_QWORD *)(a1 + 272);
              v144 = v103;
              v143 = v101;
              v145 = sub_1D38BB0(v102, v116, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
              v146 = v104;
              v147.m128i_i64[1] = v119;
              v105 = *(_QWORD **)(a1 + 272);
              *((_QWORD *)&v110 + 1) = 7;
              *(_QWORD *)&v110 = v111;
              v148.m128i_i64[0] = v113;
              v147.m128i_i64[0] = v24;
              v148.m128i_i32[2] = v112;
              v47 = sub_1D25BD0(v105, v115, (__int64)&v132, v125, 0, v106, 1u, v110);
              goto LABEL_33;
            }
            v131 = 0x100000BE7LL;
            v130 = 0x100000BDBLL;
            v129 = 0x100000BD5LL;
            v128 = 0x100000BF9LL;
            sub_21BD570(
              (__int64)&v137,
              v125,
              3071,
              3053,
              3059,
              (__int64)&v128,
              (__int64)&v129,
              (__int64)&v130,
              3041,
              (__int64)&v131);
            if ( !BYTE4(v137) )
              goto LABEL_7;
            v121 = v137;
            v55 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v126, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v56 = *(_QWORD *)(a1 + 272);
            v138 = v57;
            v137 = v55;
            v58 = sub_1D38BB0(v56, v123, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v59 = *(_QWORD *)(a1 + 272);
            v140 = v60;
            v139 = v58;
            v61 = sub_1D38BB0(v59, 1, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v62 = *(_QWORD *)(a1 + 272);
            v142 = v63;
            v141 = v61;
            v64 = sub_1D38BB0(v62, v117, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v65 = *(_QWORD *)(a1 + 272);
            v144 = v66;
            v143 = v64;
            v67 = sub_1D38BB0(v65, v116, (__int64)&v132, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
            v68 = _mm_loadu_si128(&v136);
            v69 = _mm_loadu_si128(&v135);
            v145 = v67;
            v146 = v70;
            v149 = v113;
            *((_QWORD *)&v108 + 1) = 8;
            v71 = *(_QWORD **)(a1 + 272);
            *(_QWORD *)&v108 = &v137;
            v150 = v112;
            v147 = v68;
            v148 = v69;
            v73 = sub_1D25BD0(v71, v121, (__int64)&v132, v125, 0, v72, 1u, v108);
          }
          v47 = v73;
LABEL_33:
          if ( v47 )
          {
            v10 = 1;
            v48 = (_QWORD *)sub_1E0A240(*(_QWORD *)(a1 + 256), 1);
            *v48 = *(_QWORD *)(a2 + 104);
            *(_QWORD *)(v47 + 88) = v48;
            *(_QWORD *)(v47 + 96) = v48 + 1;
            sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v47);
            sub_1D49010(v47);
            sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v49, v50, v51, v52);
          }
          goto LABEL_8;
        }
      }
      v126 = v124 == 2;
LABEL_14:
      if ( v123 > 1 )
      {
        v19 = v126;
        if ( v123 != 3 )
          v19 = 0;
        v126 = v19;
      }
      goto LABEL_18;
    }
  }
LABEL_8:
  if ( v132 )
    sub_161E7C0((__int64)&v132, v132);
  return v10;
}
