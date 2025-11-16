// Function: sub_1FEECD0
// Address: 0x1feecd0
//
__int64 *__fastcall sub_1FEECD0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  unsigned __int8 *v8; // rax
  unsigned __int8 v9; // r15
  const void **v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // rax
  unsigned int v13; // eax
  const void **v14; // rdx
  __int64 *v15; // r15
  __int128 v16; // rax
  unsigned int v17; // edx
  __int128 v18; // rax
  unsigned int v19; // edx
  __int128 v20; // rax
  unsigned int v21; // edx
  __int128 v22; // rax
  unsigned int v23; // edx
  __int128 v24; // rax
  unsigned int v25; // edx
  __int128 v26; // rax
  unsigned int v27; // edx
  __int128 v28; // rax
  __int64 *v29; // rax
  const void **v30; // r8
  unsigned int v31; // edx
  __int128 v32; // rax
  unsigned int v33; // edx
  __int128 v34; // rax
  unsigned int v35; // edx
  __int128 v36; // rax
  unsigned int v37; // edx
  __int128 v38; // rax
  unsigned int v39; // edx
  __int128 v40; // rax
  unsigned int v41; // edx
  __int128 v42; // rax
  unsigned int v43; // edx
  __int128 v44; // rax
  unsigned int v45; // edx
  unsigned int v46; // edx
  unsigned int v47; // edx
  unsigned int v48; // edx
  unsigned int v49; // edx
  unsigned int v50; // edx
  unsigned int v51; // edx
  __int64 *v53; // r15
  __int128 v54; // rax
  unsigned int v55; // edx
  __int128 v56; // rax
  __int64 *v57; // rax
  unsigned int v58; // edx
  __int64 *v59; // r15
  __int128 v60; // rax
  unsigned int v61; // edx
  __int128 v62; // rax
  unsigned int v63; // edx
  __int128 v64; // rax
  unsigned int v65; // edx
  __int128 v66; // rax
  unsigned int v67; // edx
  __int128 v68; // rax
  unsigned int v69; // edx
  __int128 v70; // rax
  unsigned int v71; // edx
  unsigned int v72; // edx
  unsigned int v73; // edx
  __int128 v74; // [rsp-10h] [rbp-2D0h]
  __int128 v75; // [rsp-10h] [rbp-2D0h]
  __int128 v76; // [rsp-10h] [rbp-2D0h]
  __int128 v77; // [rsp-10h] [rbp-2D0h]
  __int64 *v78; // [rsp+10h] [rbp-2B0h]
  __int128 v79; // [rsp+10h] [rbp-2B0h]
  unsigned __int64 v80; // [rsp+18h] [rbp-2A8h]
  __int64 *v81; // [rsp+20h] [rbp-2A0h]
  __int128 v82; // [rsp+20h] [rbp-2A0h]
  unsigned __int64 v83; // [rsp+28h] [rbp-298h]
  __int64 *v84; // [rsp+30h] [rbp-290h]
  __int64 *v85; // [rsp+30h] [rbp-290h]
  __int64 *v86; // [rsp+30h] [rbp-290h]
  __int64 *v87; // [rsp+30h] [rbp-290h]
  __int64 *v88; // [rsp+30h] [rbp-290h]
  __int64 *v89; // [rsp+30h] [rbp-290h]
  unsigned int v90; // [rsp+38h] [rbp-288h]
  const void **v91; // [rsp+40h] [rbp-280h]
  __int64 *v92; // [rsp+40h] [rbp-280h]
  __int64 v93; // [rsp+48h] [rbp-278h]
  __int64 *v94; // [rsp+50h] [rbp-270h]
  __int64 *v95; // [rsp+50h] [rbp-270h]
  __int64 *v96; // [rsp+50h] [rbp-270h]
  unsigned __int64 v97; // [rsp+58h] [rbp-268h]
  unsigned __int64 v98; // [rsp+58h] [rbp-268h]
  unsigned __int64 v99; // [rsp+58h] [rbp-268h]
  __int64 *v100; // [rsp+60h] [rbp-260h]
  __int64 *v101; // [rsp+60h] [rbp-260h]
  __int128 v102; // [rsp+60h] [rbp-260h]
  __int64 *v103; // [rsp+60h] [rbp-260h]
  __int64 *v104; // [rsp+60h] [rbp-260h]
  __int64 *v105; // [rsp+60h] [rbp-260h]
  unsigned __int64 v106; // [rsp+68h] [rbp-258h]
  unsigned __int64 v107; // [rsp+68h] [rbp-258h]
  __int64 v108; // [rsp+68h] [rbp-258h]
  __int64 *v109; // [rsp+70h] [rbp-250h]
  __int128 v110; // [rsp+70h] [rbp-250h]
  __int64 *v111; // [rsp+70h] [rbp-250h]
  __int128 v112; // [rsp+70h] [rbp-250h]
  unsigned __int64 v113; // [rsp+78h] [rbp-248h]
  unsigned __int64 v114; // [rsp+78h] [rbp-248h]
  __int64 *v115; // [rsp+80h] [rbp-240h]
  __int64 *v116; // [rsp+80h] [rbp-240h]
  __int64 *v117; // [rsp+80h] [rbp-240h]
  __int64 *v118; // [rsp+80h] [rbp-240h]
  __int64 *v119; // [rsp+80h] [rbp-240h]
  __int64 *v120; // [rsp+80h] [rbp-240h]
  __int64 *v121; // [rsp+80h] [rbp-240h]
  __int64 *v122; // [rsp+80h] [rbp-240h]
  __int64 *v124; // [rsp+90h] [rbp-230h]
  __int64 *v125; // [rsp+90h] [rbp-230h]
  __int64 *v126; // [rsp+90h] [rbp-230h]
  __int64 *v127; // [rsp+90h] [rbp-230h]
  __int64 *v128; // [rsp+90h] [rbp-230h]
  __int64 *v129; // [rsp+90h] [rbp-230h]
  unsigned __int64 v130; // [rsp+98h] [rbp-228h]
  unsigned __int64 v131; // [rsp+98h] [rbp-228h]
  unsigned __int64 v132; // [rsp+98h] [rbp-228h]
  unsigned __int64 v133; // [rsp+98h] [rbp-228h]
  unsigned __int64 v134; // [rsp+98h] [rbp-228h]
  __int64 v135; // [rsp+A0h] [rbp-220h]
  __int64 v136; // [rsp+A0h] [rbp-220h]
  __int64 *v137; // [rsp+A0h] [rbp-220h]
  __int128 v138; // [rsp+A0h] [rbp-220h]
  __int128 v139; // [rsp+A0h] [rbp-220h]
  __int64 *v140; // [rsp+A0h] [rbp-220h]
  __int64 *v141; // [rsp+A0h] [rbp-220h]
  unsigned __int64 v142; // [rsp+A8h] [rbp-218h]
  unsigned __int64 v143; // [rsp+A8h] [rbp-218h]
  unsigned __int64 v144; // [rsp+A8h] [rbp-218h]
  unsigned __int64 v145; // [rsp+A8h] [rbp-218h]
  __int64 *v146; // [rsp+B0h] [rbp-210h]
  __int64 *v147; // [rsp+170h] [rbp-150h]
  __int64 *v148; // [rsp+1F0h] [rbp-D0h]
  __int64 *v149; // [rsp+230h] [rbp-90h]

  v135 = *(_QWORD *)(a1 + 8);
  v8 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v9 = *v8;
  v10 = (const void **)*((_QWORD *)v8 + 1);
  v11 = *v8;
  v12 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL));
  v13 = sub_1F40B60(v135, v9, (__int64)v10, v12, 1);
  v91 = v14;
  v90 = v13;
  if ( (unsigned __int8)(v9 - 14) <= 0x5Fu )
  {
    switch ( v9 )
    {
      case '!':
      case '"':
      case '#':
      case '$':
      case '%':
      case '&':
      case '\'':
      case '(':
      case 'D':
      case 'E':
      case 'F':
      case 'G':
      case 'H':
      case 'I':
        goto LABEL_6;
      case ')':
      case '*':
      case '+':
      case ',':
      case '-':
      case '.':
      case '/':
      case '0':
      case 'J':
      case 'K':
      case 'L':
      case 'M':
      case 'N':
      case 'O':
        goto LABEL_7;
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case ':':
      case ';':
      case '<':
      case '=':
      case '>':
      case '?':
      case '@':
      case 'A':
      case 'B':
      case 'C':
      case 'P':
      case 'Q':
      case 'R':
      case 'S':
      case 'T':
      case 'U':
        goto LABEL_4;
    }
  }
  if ( v9 == 5 )
  {
LABEL_7:
    v59 = *(__int64 **)(a1 + 16);
    *(_QWORD *)&v60 = sub_1D38BB0((__int64)v59, 24, a4, v13, v14, 0, a5, a6, a7, 0);
    v128 = sub_1D332F0(v59, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v60);
    v103 = *(__int64 **)(a1 + 16);
    v133 = v61;
    *(_QWORD *)&v62 = sub_1D38BB0((__int64)v103, 8, a4, v90, v91, 0, a5, a6, a7, 0);
    v111 = sub_1D332F0(v103, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v62);
    v104 = *(__int64 **)(a1 + 16);
    v114 = v63;
    *(_QWORD *)&v64 = sub_1D38BB0((__int64)v104, 8, a4, v90, v91, 0, a5, a6, a7, 0);
    v140 = sub_1D332F0(v104, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v64);
    v105 = *(__int64 **)(a1 + 16);
    v144 = v65;
    *(_QWORD *)&v66 = sub_1D38BB0((__int64)v105, 24, a4, v90, v91, 0, a5, a6, a7, 0);
    v149 = sub_1D332F0(v105, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v66);
    v121 = *(__int64 **)(a1 + 16);
    v108 = v67;
    *(_QWORD *)&v68 = sub_1D38BB0((__int64)v121, (__int64)&loc_FF0000, a4, v11, v10, 0, a5, a6, a7, 0);
    *(_QWORD *)&v112 = sub_1D332F0(
                         v121,
                         118,
                         a4,
                         v11,
                         v10,
                         0,
                         *(double *)a5.m128i_i64,
                         a6,
                         a7,
                         (__int64)v111,
                         v114,
                         v68);
    v122 = *(__int64 **)(a1 + 16);
    *((_QWORD *)&v112 + 1) = v69 | v114 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v70 = sub_1D38BB0((__int64)v122, 65280, a4, v11, v10, 0, a5, a6, a7, 0);
    v141 = sub_1D332F0(v122, 118, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v140, v144, v70);
    v145 = v71 | v144 & 0xFFFFFFFF00000000LL;
    v129 = sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v128,
             v133,
             v112);
    *((_QWORD *)&v76 + 1) = v108;
    *(_QWORD *)&v76 = v149;
    v134 = v72 | v133 & 0xFFFFFFFF00000000LL;
    v148 = sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v141,
             v145,
             v76);
    *((_QWORD *)&v77 + 1) = v73 | v145 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v77 = v148;
    return sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v129,
             v134,
             v77);
  }
  else if ( v9 == 6 )
  {
LABEL_4:
    v15 = *(__int64 **)(a1 + 16);
    *(_QWORD *)&v16 = sub_1D38BB0((__int64)v15, 56, a4, v13, v14, 0, a5, a6, a7, 0);
    v94 = sub_1D332F0(v15, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v16);
    v84 = *(__int64 **)(a1 + 16);
    v97 = v17;
    *(_QWORD *)&v18 = sub_1D38BB0((__int64)v84, 40, a4, v90, v91, 0, a5, a6, a7, 0);
    v78 = sub_1D332F0(v84, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v18);
    v85 = *(__int64 **)(a1 + 16);
    v80 = v19;
    *(_QWORD *)&v20 = sub_1D38BB0((__int64)v85, 24, a4, v90, v91, 0, a5, a6, a7, 0);
    v100 = sub_1D332F0(v85, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v20);
    v86 = *(__int64 **)(a1 + 16);
    v106 = v21;
    *(_QWORD *)&v22 = sub_1D38BB0((__int64)v86, 8, a4, v90, v91, 0, a5, a6, a7, 0);
    v81 = sub_1D332F0(v86, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v22);
    v87 = *(__int64 **)(a1 + 16);
    v83 = v23;
    *(_QWORD *)&v24 = sub_1D38BB0((__int64)v87, 8, a4, v90, v91, 0, a5, a6, a7, 0);
    v124 = sub_1D332F0(v87, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v24);
    v88 = *(__int64 **)(a1 + 16);
    v130 = v25;
    *(_QWORD *)&v26 = sub_1D38BB0((__int64)v88, 24, a4, v90, v91, 0, a5, a6, a7, 0);
    v109 = sub_1D332F0(v88, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v26);
    v89 = *(__int64 **)(a1 + 16);
    v113 = v27;
    *(_QWORD *)&v28 = sub_1D38BB0((__int64)v89, 40, a4, v90, v91, 0, a5, a6, a7, 0);
    v29 = sub_1D332F0(v89, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v28);
    v30 = v91;
    v136 = (__int64)v29;
    v92 = *(__int64 **)(a1 + 16);
    v142 = v31;
    *(_QWORD *)&v32 = sub_1D38BB0((__int64)v92, 56, a4, v90, v30, 0, a5, a6, a7, 0);
    v147 = sub_1D332F0(v92, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v32);
    v115 = *(__int64 **)(a1 + 16);
    v93 = v33;
    *(_QWORD *)&v34 = sub_1D38BB0((__int64)v115, 0xFF000000000000LL, a4, v11, v10, 0, a5, a6, a7, 0);
    *(_QWORD *)&v79 = sub_1D332F0(v115, 118, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v78, v80, v34);
    v116 = *(__int64 **)(a1 + 16);
    *((_QWORD *)&v79 + 1) = v35 | v80 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v36 = sub_1D38BB0((__int64)v116, 0xFF0000000000LL, a4, v11, v10, 0, a5, a6, a7, 0);
    v101 = sub_1D332F0(v116, 118, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v100, v106, v36);
    v117 = *(__int64 **)(a1 + 16);
    v107 = v37 | v106 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v38 = sub_1D38BB0((__int64)v117, 0xFF00000000LL, a4, v11, v10, 0, a5, a6, a7, 0);
    *(_QWORD *)&v82 = sub_1D332F0(v117, 118, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v81, v83, v38);
    v118 = *(__int64 **)(a1 + 16);
    *((_QWORD *)&v82 + 1) = v39 | v83 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v40 = sub_1D38BB0((__int64)v118, 4278190080LL, a4, v11, v10, 0, a5, a6, a7, 0);
    v125 = sub_1D332F0(v118, 118, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v124, v130, v40);
    v119 = *(__int64 **)(a1 + 16);
    v131 = v41 | v130 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v42 = sub_1D38BB0((__int64)v119, (__int64)&loc_FF0000, a4, v11, v10, 0, a5, a6, a7, 0);
    *(_QWORD *)&v110 = sub_1D332F0(
                         v119,
                         118,
                         a4,
                         v11,
                         v10,
                         0,
                         *(double *)a5.m128i_i64,
                         a6,
                         a7,
                         (__int64)v109,
                         v113,
                         v42);
    v120 = *(__int64 **)(a1 + 16);
    *((_QWORD *)&v110 + 1) = v43 | v113 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v44 = sub_1D38BB0((__int64)v120, 65280, a4, v11, v10, 0, a5, a6, a7, 0);
    v137 = sub_1D332F0(v120, 118, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, v136, v142, v44);
    v143 = v45 | v142 & 0xFFFFFFFF00000000LL;
    v95 = sub_1D332F0(
            *(__int64 **)(a1 + 16),
            119,
            a4,
            v11,
            v10,
            0,
            *(double *)a5.m128i_i64,
            a6,
            a7,
            (__int64)v94,
            v97,
            v79);
    v98 = v46 | v97 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v102 = sub_1D332F0(
                         *(__int64 **)(a1 + 16),
                         119,
                         a4,
                         v11,
                         v10,
                         0,
                         *(double *)a5.m128i_i64,
                         a6,
                         a7,
                         (__int64)v101,
                         v107,
                         v82);
    *((_QWORD *)&v102 + 1) = v47 | v107 & 0xFFFFFFFF00000000LL;
    v126 = sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v125,
             v131,
             v110);
    *((_QWORD *)&v74 + 1) = v93;
    *(_QWORD *)&v74 = v147;
    v132 = v48 | v131 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v138 = sub_1D332F0(
                         *(__int64 **)(a1 + 16),
                         119,
                         a4,
                         v11,
                         v10,
                         0,
                         *(double *)a5.m128i_i64,
                         a6,
                         a7,
                         (__int64)v137,
                         v143,
                         v74);
    *((_QWORD *)&v138 + 1) = v49 | v143 & 0xFFFFFFFF00000000LL;
    v96 = sub_1D332F0(
            *(__int64 **)(a1 + 16),
            119,
            a4,
            v11,
            v10,
            0,
            *(double *)a5.m128i_i64,
            a6,
            a7,
            (__int64)v95,
            v98,
            v102);
    v99 = v50 | v98 & 0xFFFFFFFF00000000LL;
    v146 = sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v126,
             v132,
             v138);
    *((_QWORD *)&v75 + 1) = v51 | v132 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v75 = v146;
    return sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v96,
             v99,
             v75);
  }
  else
  {
LABEL_6:
    v53 = *(__int64 **)(a1 + 16);
    *(_QWORD *)&v54 = sub_1D38BB0((__int64)v53, 8, a4, v13, v14, 0, a5, a6, a7, 0);
    *(_QWORD *)&v139 = sub_1D332F0(v53, 122, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v54);
    v127 = *(__int64 **)(a1 + 16);
    *((_QWORD *)&v139 + 1) = v55;
    *(_QWORD *)&v56 = sub_1D38BB0((__int64)v127, 8, a4, v90, v91, 0, a5, a6, a7, 0);
    v57 = sub_1D332F0(v127, 124, a4, v11, v10, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v56);
    return sub_1D332F0(
             *(__int64 **)(a1 + 16),
             119,
             a4,
             v11,
             v10,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v57,
             v58,
             v139);
  }
}
