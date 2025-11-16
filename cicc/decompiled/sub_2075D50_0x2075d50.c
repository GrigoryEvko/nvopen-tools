// Function: sub_2075D50
// Address: 0x2075d50
//
void __fastcall sub_2075D50(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 (*v7)(); // rax
  __int64 *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r12
  __int16 v11; // r13
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // r9d
  __int64 *v18; // rax
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // rsi
  __int64 (__fastcall *v22)(__int64, __int64); // rax
  bool v23; // al
  int v24; // r8d
  unsigned int v25; // r9d
  _BYTE *v26; // rax
  unsigned __int64 v27; // r12
  __int64 v28; // r10
  _BYTE *v29; // rdx
  int v30; // r8d
  __m128i *v31; // rax
  __int64 v32; // r14
  __int64 *v33; // r12
  const void ***v34; // rax
  int v35; // edx
  __int64 v36; // r9
  __int64 *v37; // rax
  __int64 *v38; // rbx
  int v39; // edx
  __int64 *v40; // rax
  __m128i *v41; // rdx
  const void **v42; // r15
  unsigned __int16 v43; // dx
  unsigned __int16 v44; // ax
  __int64 v45; // rbx
  unsigned int v46; // r12d
  __int64 v47; // rax
  unsigned __int64 v48; // r14
  __int128 v49; // rax
  __int64 *v50; // r8
  __int64 v51; // rdx
  __int64 v52; // r9
  __int64 (*v53)(); // rdx
  unsigned __int16 v54; // ax
  __int64 v55; // rdx
  _QWORD *v56; // rdi
  __int64 v57; // rdx
  __int64 v58; // rax
  int v59; // edx
  __int64 v60; // rcx
  int v61; // eax
  unsigned __int64 v62; // rax
  __int64 v63; // rax
  __m128i *v64; // rax
  unsigned int v65; // edx
  __int16 v66; // ax
  char v67; // al
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // rax
  char v71; // al
  __int64 v72; // r13
  __int64 v73; // rdx
  int v74; // edx
  __int64 v75; // rax
  __int64 *v76; // rax
  __int64 *v77; // rdx
  int v78; // r8d
  int v79; // r9d
  __int64 *v80; // rbx
  __int64 *v81; // r12
  __int64 v82; // r8
  unsigned int v83; // eax
  __int64 v84; // rax
  __int64 **v85; // rax
  __int64 v86; // rax
  unsigned int v87; // edx
  __int128 v88; // [rsp-10h] [rbp-2D0h]
  __int128 v89; // [rsp-10h] [rbp-2D0h]
  __int64 v90; // [rsp+30h] [rbp-290h]
  char v91; // [rsp+38h] [rbp-288h]
  bool v92; // [rsp+3Bh] [rbp-285h]
  unsigned int v93; // [rsp+48h] [rbp-278h]
  unsigned __int16 v94; // [rsp+4Eh] [rbp-272h]
  int v95; // [rsp+50h] [rbp-270h]
  __int64 v96; // [rsp+50h] [rbp-270h]
  __int64 v97; // [rsp+58h] [rbp-268h]
  __int64 v98; // [rsp+60h] [rbp-260h]
  unsigned __int64 v99; // [rsp+68h] [rbp-258h]
  __int64 *v100; // [rsp+70h] [rbp-250h]
  __int64 v101; // [rsp+78h] [rbp-248h]
  __int64 v102; // [rsp+78h] [rbp-248h]
  __int64 v103; // [rsp+78h] [rbp-248h]
  unsigned __int64 v104; // [rsp+80h] [rbp-240h]
  __int64 v105; // [rsp+80h] [rbp-240h]
  __int64 v106; // [rsp+88h] [rbp-238h]
  unsigned __int64 v107; // [rsp+90h] [rbp-230h]
  unsigned int v108; // [rsp+98h] [rbp-228h]
  __int64 v109; // [rsp+A0h] [rbp-220h]
  __int64 *v110; // [rsp+A0h] [rbp-220h]
  __int64 v111; // [rsp+A0h] [rbp-220h]
  __int64 *v112; // [rsp+A0h] [rbp-220h]
  __int64 v113; // [rsp+A8h] [rbp-218h]
  char v114; // [rsp+B0h] [rbp-210h]
  int v115; // [rsp+B0h] [rbp-210h]
  unsigned int v116; // [rsp+B8h] [rbp-208h]
  unsigned int v117; // [rsp+B8h] [rbp-208h]
  unsigned int v118; // [rsp+B8h] [rbp-208h]
  unsigned int v119; // [rsp+B8h] [rbp-208h]
  unsigned int v120; // [rsp+B8h] [rbp-208h]
  unsigned int v121; // [rsp+B8h] [rbp-208h]
  unsigned int v122; // [rsp+B8h] [rbp-208h]
  unsigned int v123; // [rsp+C0h] [rbp-200h]
  __int128 v124; // [rsp+C0h] [rbp-200h]
  __int64 v125; // [rsp+C0h] [rbp-200h]
  unsigned int v126; // [rsp+C0h] [rbp-200h]
  __int64 v127; // [rsp+C0h] [rbp-200h]
  __int64 v128; // [rsp+120h] [rbp-1A0h] BYREF
  int v129; // [rsp+128h] [rbp-198h]
  __m128i v130; // [rsp+130h] [rbp-190h] BYREF
  __int64 v131; // [rsp+140h] [rbp-180h]
  __int128 v132; // [rsp+150h] [rbp-170h] BYREF
  __int64 v133; // [rsp+160h] [rbp-160h]
  unsigned __int64 v134[2]; // [rsp+170h] [rbp-150h] BYREF
  _BYTE v135[32]; // [rsp+180h] [rbp-140h] BYREF
  unsigned __int8 *v136; // [rsp+1A0h] [rbp-120h] BYREF
  __int64 v137; // [rsp+1A8h] [rbp-118h]
  _BYTE v138[64]; // [rsp+1B0h] [rbp-110h] BYREF
  _BYTE *v139; // [rsp+1F0h] [rbp-D0h] BYREF
  __int64 v140; // [rsp+1F8h] [rbp-C8h]
  _BYTE v141[64]; // [rsp+200h] [rbp-C0h] BYREF
  __m128i *v142; // [rsp+240h] [rbp-80h] BYREF
  unsigned __int64 v143; // [rsp+248h] [rbp-78h]
  __m128i v144; // [rsp+250h] [rbp-70h] BYREF
  __int64 v145; // [rsp+260h] [rbp-60h]

  v92 = sub_15F32D0(a2);
  if ( v92 )
  {
    sub_20758C0(a1, a2, a3, a4, a5);
    return;
  }
  v107 = *(_QWORD *)(a2 - 24);
  v106 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  v101 = *(_QWORD *)v107;
  v7 = *(__int64 (**)())(*(_QWORD *)v106 + 1160LL);
  if ( v7 != sub_1D45FE0 && ((unsigned __int8 (__fastcall *)(__int64))v7)(v106) )
  {
    v67 = *(_BYTE *)(v107 + 16);
    if ( v67 == 17 )
    {
      if ( (unsigned __int8)sub_15E02D0(v107) )
      {
LABEL_74:
        sub_205FE20(a1, (_QWORD *)a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        return;
      }
      v67 = *(_BYTE *)(v107 + 16);
    }
    if ( v67 == 53 && (*(_BYTE *)(v107 + 18) & 0x40) != 0 )
      goto LABEL_74;
  }
  v8 = sub_20685E0(a1, (__int64 *)v107, a3, a4, a5);
  v9 = *(_QWORD *)(a2 + 48);
  v10 = *(_QWORD *)a2;
  v98 = (__int64)v8;
  v11 = *(_WORD *)(a2 + 18);
  v99 = v12;
  if ( v9 || (v109 = 0, v11 < 0) )
  {
    v13 = sub_1625790(a2, 9);
    v9 = *(_QWORD *)(a2 + 48);
    v109 = v13;
    if ( v9 || *(__int16 *)(a2 + 18) < 0 )
      v9 = sub_1625790(a2, 6);
  }
  v14 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v130 = 0u;
  v114 = sub_13F8680(v107, v14, 0, 0);
  LOWORD(v108) = *(_WORD *)(a2 + 18);
  v131 = 0;
  sub_14A8180(a2, v130.m128i_i64, 0);
  v97 = *(_QWORD *)(a2 + 48);
  if ( v97 || *(__int16 *)(a2 + 18) < 0 )
    v97 = sub_1625790(a2, 4);
  v136 = v138;
  v137 = 0x400000000LL;
  v134[1] = 0x400000000LL;
  v15 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
  v134[0] = (unsigned __int64)v135;
  v16 = sub_1E0A0C0(v15);
  sub_20C7CE0(v106, v16, v10, &v136, v134, 0);
  v17 = v137;
  if ( (_DWORD)v137 )
  {
    v91 = v11 & 1;
    if ( v11 & 1 | ((unsigned int)v137 > 0x40) )
    {
      v116 = v137;
      v18 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
      v17 = v116;
      v128 = 0;
      v100 = v18;
      v19 = *(_QWORD *)a1;
      v123 = v20;
      v104 = v20;
      v129 = *(_DWORD *)(a1 + 536);
      if ( !v19 || &v128 == (__int64 *)(v19 + 48) || (v21 = *(_QWORD *)(v19 + 48), (v128 = v21) == 0) )
      {
LABEL_14:
        if ( v91 )
        {
          v22 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v106 + 1288LL);
          if ( v22 != sub_2043C60 )
          {
            v122 = v17;
            v104 = v123 | v104 & 0xFFFFFFFF00000000LL;
            v86 = ((__int64 (__fastcall *)(__int64, __int64 *, unsigned __int64, __int64 *, _QWORD))v22)(
                    v106,
                    v100,
                    v104,
                    &v128,
                    *(_QWORD *)(a1 + 552));
            v17 = v122;
            v100 = (__int64 *)v86;
            v123 = v87;
          }
        }
        goto LABEL_17;
      }
    }
    else
    {
      v68 = *(_QWORD *)(a1 + 568);
      if ( !v68 )
        goto LABEL_77;
      v126 = v137;
      v69 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
      v70 = sub_127FA20(v69, v10);
      a3 = _mm_load_si128(&v130);
      v142 = (__m128i *)v107;
      v143 = (unsigned __int64)(v70 + 7) >> 3;
      v144 = a3;
      v145 = v131;
      v71 = sub_134CBB0(v68, (__int64)&v142, 0);
      v17 = v126;
      v92 = v71;
      if ( v71 )
      {
        v123 = 0;
        v100 = (__int64 *)(*(_QWORD *)(a1 + 552) + 88LL);
      }
      else
      {
LABEL_77:
        v72 = v101;
        if ( *(_BYTE *)(v101 + 8) == 16 )
          v72 = **(_QWORD **)(v101 + 16);
        v73 = *(_QWORD *)(a1 + 552);
        if ( *(_DWORD *)(v72 + 8) >> 8 == 101 )
        {
          v92 = 1;
          v100 = (__int64 *)(v73 + 88);
          v123 = 0;
        }
        else
        {
          v92 = 0;
          v104 = *(unsigned int *)(v73 + 184);
          v100 = *(__int64 **)(v73 + 176);
          v123 = *(_DWORD *)(v73 + 184);
        }
      }
      v74 = *(_DWORD *)(a1 + 536);
      v75 = *(_QWORD *)a1;
      v128 = 0;
      v129 = v74;
      if ( !v75 || &v128 == (__int64 *)(v75 + 48) || (v21 = *(_QWORD *)(v75 + 48), (v128 = v21) == 0) )
      {
LABEL_17:
        v118 = v17;
        v23 = sub_1C300D0(a2);
        v25 = v118;
        v93 = -1;
        if ( v23 )
        {
          v83 = sub_1C30110(a2);
          v25 = v118;
          v93 = v83;
        }
        v26 = v141;
        v27 = v25;
        v139 = v141;
        v140 = 0x400000000LL;
        if ( v25 > 4 )
        {
          v121 = v25;
          sub_16CD150((__int64)&v139, v141, v25, 16, v24, v25);
          v26 = v139;
          v25 = v121;
        }
        LODWORD(v140) = v25;
        v28 = 16 * v27;
        v29 = &v26[16 * v27];
        do
        {
          if ( v26 )
          {
            *(_QWORD *)v26 = 0;
            *((_DWORD *)v26 + 2) = 0;
          }
          v26 += 16;
        }
        while ( v29 != v26 );
        v142 = &v144;
        v143 = 0x400000000LL;
        if ( v25 <= 0x3F )
        {
          v30 = v25;
          if ( v25 <= 4 )
          {
            v31 = &v144;
LABEL_42:
            LODWORD(v143) = v30;
            v41 = (__m128i *)((char *)v31 + v28);
            do
            {
              if ( v31 )
              {
                v31->m128i_i64[0] = 0;
                v31->m128i_i32[2] = 0;
              }
              ++v31;
            }
            while ( v41 != v31 );
            v103 = a2;
            v120 = *(unsigned __int8 *)(*(_QWORD *)(v98 + 40) + 16LL * (unsigned int)v99);
            v42 = *(const void ***)(*(_QWORD *)(v98 + 40) + 16LL * (unsigned int)v99 + 8);
            v43 = (4 * v91) & 4;
            if ( v109 )
              v43 = (4 * v91) & 4 | 8;
            v44 = v43 | 0x20;
            if ( !v9 )
              v44 = v43;
            if ( v114 )
              v44 |= 0x10u;
            v45 = 0;
            v46 = 0;
            v94 = v44;
            v90 = 8LL * (v25 - 1);
            v47 = v123;
            v125 = a1;
            v48 = v104;
            v96 = v47;
            while ( 1 )
            {
              v110 = *(__int64 **)(v125 + 552);
              *(_QWORD *)&v49 = sub_1D38BB0(
                                  (__int64)v110,
                                  *(_QWORD *)(v134[0] + v45),
                                  (__int64)&v128,
                                  v120,
                                  v42,
                                  0,
                                  a3,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  0);
              v50 = sub_1D332F0(
                      v110,
                      52,
                      (__int64)&v128,
                      v120,
                      v42,
                      3u,
                      *(double *)a3.m128i_i64,
                      *(double *)a4.m128i_i64,
                      a5,
                      v98,
                      v99,
                      v49);
              v52 = v51;
              v53 = *(__int64 (**)())(*(_QWORD *)v106 + 1296LL);
              v54 = v94;
              if ( v53 != sub_2043C70 )
              {
                v112 = v50;
                v113 = v52;
                v66 = ((__int64 (__fastcall *)(__int64, __int64))v53)(v106, v103);
                v50 = v112;
                v52 = v113;
                v54 = v94 | v66;
              }
              v55 = *(_QWORD *)(v134[0] + v45);
              v56 = *(_QWORD **)(v125 + 552);
              LOBYTE(v133) = 0;
              *((_QWORD *)&v132 + 1) = v55;
              *(_QWORD *)&v132 = v107;
              v57 = *(_QWORD *)v107;
              if ( *(_BYTE *)(*(_QWORD *)v107 + 8LL) == 16 )
                v57 = **(_QWORD **)(v57 + 16);
              v48 = v96 | v48 & 0xFFFFFFFF00000000LL;
              HIDWORD(v133) = *(_DWORD *)(v57 + 8) >> 8;
              v58 = sub_1D2B730(
                      v56,
                      *(unsigned int *)&v136[2 * v45],
                      *(_QWORD *)&v136[2 * v45 + 8],
                      (__int64)&v128,
                      (__int64)v100,
                      v48,
                      (__int64)v50,
                      v52,
                      v132,
                      v133,
                      1 << (v108 >> 1) >> 1,
                      v54,
                      (__int64)&v130,
                      v97);
              v115 = v59;
              v111 = v58;
              v105 = *(_QWORD *)(v58 + 104);
              if ( sub_1C300D0(v103) )
              {
                v60 = *(_QWORD *)(v105 + 24);
                v61 = v93 & ~(-1 << v60);
                v93 >>= v60;
                *(_DWORD *)(v105 + 72) = v61;
              }
              v62 = (unsigned __int64)v139;
              *(_QWORD *)&v139[2 * v45] = v111;
              *(_DWORD *)(v62 + 2 * v45 + 8) = v115;
              v63 = v46++;
              v64 = &v142[v63];
              v64->m128i_i64[0] = v111;
              v64->m128i_i32[2] = 1;
              if ( v45 == v90 )
                break;
              if ( v46 == 64 )
              {
                v46 = 0;
                *((_QWORD *)&v88 + 1) = 64;
                *(_QWORD *)&v88 = v142;
                v100 = sub_1D359D0(
                         *(__int64 **)(v125 + 552),
                         2,
                         (__int64)&v128,
                         1,
                         0,
                         0,
                         *(double *)a3.m128i_i64,
                         *(double *)a4.m128i_i64,
                         a5,
                         v88);
                v96 = v65;
              }
              v45 += 8;
            }
            v32 = v125;
            if ( !v92 )
            {
              *((_QWORD *)&v89 + 1) = v46;
              *(_QWORD *)&v89 = v142;
              v76 = sub_1D359D0(
                      *(__int64 **)(v125 + 552),
                      2,
                      (__int64)&v128,
                      1,
                      0,
                      0,
                      *(double *)a3.m128i_i64,
                      *(double *)a4.m128i_i64,
                      a5,
                      v89);
              v80 = v76;
              v81 = v77;
              if ( v91 )
              {
                v82 = *(_QWORD *)(v125 + 552);
                if ( v76 )
                {
                  v127 = *(_QWORD *)(v125 + 552);
                  nullsub_686();
                  *(_QWORD *)(v127 + 176) = v80;
                  *(_DWORD *)(v127 + 184) = (_DWORD)v81;
                  sub_1D23870();
                }
                else
                {
                  *(_QWORD *)(v82 + 176) = 0;
                  *(_DWORD *)(v82 + 184) = (_DWORD)v77;
                }
              }
              else
              {
                v84 = *(unsigned int *)(v125 + 112);
                if ( (unsigned int)v84 >= *(_DWORD *)(v125 + 116) )
                {
                  sub_16CD150(v125 + 104, (const void *)(v125 + 120), 0, 16, v78, v79);
                  v84 = *(unsigned int *)(v125 + 112);
                }
                v85 = (__int64 **)(*(_QWORD *)(v125 + 104) + 16 * v84);
                *v85 = v80;
                v85[1] = v81;
                ++*(_DWORD *)(v125 + 112);
              }
            }
            v33 = *(__int64 **)(v32 + 552);
            *(_QWORD *)&v124 = v139;
            *((_QWORD *)&v124 + 1) = (unsigned int)v140;
            v34 = (const void ***)sub_1D25C30((__int64)v33, v136, (unsigned int)v137);
            v37 = sub_1D36D80(
                    v33,
                    51,
                    (__int64)&v128,
                    v34,
                    v35,
                    *(double *)a3.m128i_i64,
                    *(double *)a4.m128i_i64,
                    a5,
                    v36,
                    v124);
            *(_QWORD *)&v132 = v103;
            v38 = v37;
            LODWORD(v33) = v39;
            v40 = sub_205F5C0(v32 + 8, (__int64 *)&v132);
            v40[1] = (__int64)v38;
            *((_DWORD *)v40 + 4) = (_DWORD)v33;
            if ( v142 != &v144 )
              _libc_free((unsigned __int64)v142);
            if ( v139 != v141 )
              _libc_free((unsigned __int64)v139);
            if ( v128 )
              sub_161E7C0((__int64)&v128, v128);
            goto LABEL_35;
          }
        }
        else
        {
          v28 = 1024;
          v27 = 64;
          v30 = 64;
        }
        v95 = v30;
        v102 = v28;
        v119 = v25;
        sub_16CD150((__int64)&v142, &v144, v27, 16, v30, v25);
        v31 = v142;
        v25 = v119;
        v28 = v102;
        v30 = v95;
        goto LABEL_42;
      }
    }
    v117 = v17;
    sub_1623A60((__int64)&v128, v21, 2);
    v17 = v117;
    goto LABEL_14;
  }
LABEL_35:
  if ( (_BYTE *)v134[0] != v135 )
    _libc_free(v134[0]);
  if ( v136 != v138 )
    _libc_free((unsigned __int64)v136);
}
