// Function: sub_3743390
// Address: 0x3743390
//
__int64 __fastcall sub_3743390(_QWORD *a1, __int64 ***a2)
{
  __int64 **v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  _WORD *v10; // r12
  char v11; // al
  __int64 v12; // rbx
  unsigned __int64 v13; // r14
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 *v16; // r15
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 **v20; // rax
  __int64 *v21; // r10
  unsigned int v22; // r9d
  __int64 (__fastcall *v23)(__int64, __int64 *, __int64, __int64, unsigned __int64); // rax
  int v24; // eax
  __int64 v25; // r9
  int v26; // r10d
  __int64 v27; // rbx
  __int64 v28; // rax
  int v29; // r12d
  __int64 v30; // rdi
  __int16 v31; // r15
  __int64 v32; // r11
  __int64 v33; // r8
  char v34; // dl
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rcx
  const __m128i *v37; // rbx
  __m128i *v38; // rax
  __int64 **v39; // rdi
  __int64 **v40; // rax
  __int64 **v41; // r14
  bool v42; // r11
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // eax
  char v46; // al
  char v47; // r12
  unsigned __int8 v48; // dl
  bool v49; // r10
  __int64 v50; // rdi
  char v51; // r12
  char v52; // bl
  char v53; // al
  __int64 v54; // r9
  __int64 v55; // rcx
  unsigned __int8 v56; // bl
  __int64 v57; // r8
  _BOOL8 v58; // r10
  char v59; // al
  __int64 v60; // r11
  __int64 v61; // rdx
  unsigned __int64 v62; // rbx
  unsigned __int64 *v63; // rdx
  __int64 *v64; // rsi
  __int64 (*v65)(); // rax
  char v66; // bl
  bool v67; // r15
  char v68; // al
  bool v69; // si
  int v70; // eax
  char v71; // al
  const void *v72; // rsi
  char *v73; // rbx
  __int64 (__fastcall *v74)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // r10
  __int64 v78; // rax
  __int64 (__fastcall *v79)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // r10
  __int64 (__fastcall *v83)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v84; // rax
  __int64 v85; // rcx
  __int64 v86; // r10
  __int64 (__fastcall *v87)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v88; // rax
  unsigned __int64 v89; // rcx
  __int64 v90; // r10
  unsigned __int8 v91; // al
  __int64 (*v92)(); // rax
  unsigned int v93; // r12d
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rax
  __int64 v98; // rdx
  int v99; // ecx
  __int64 **v100; // rsi
  __int64 v101; // rdi
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rax
  unsigned __int64 v106; // rdx
  __int64 **v107; // [rsp+28h] [rbp-278h]
  bool v108; // [rsp+33h] [rbp-26Dh]
  bool v109; // [rsp+34h] [rbp-26Ch]
  char v110; // [rsp+34h] [rbp-26Ch]
  __int64 v111; // [rsp+38h] [rbp-268h]
  unsigned int v112; // [rsp+38h] [rbp-268h]
  __int64 v113; // [rsp+40h] [rbp-260h]
  bool v114; // [rsp+40h] [rbp-260h]
  __int64 v115; // [rsp+40h] [rbp-260h]
  __int64 v116; // [rsp+48h] [rbp-258h]
  char v117; // [rsp+48h] [rbp-258h]
  bool v118; // [rsp+48h] [rbp-258h]
  __int64 v119; // [rsp+50h] [rbp-250h]
  __int64 v120; // [rsp+58h] [rbp-248h]
  bool v121; // [rsp+58h] [rbp-248h]
  bool v122; // [rsp+60h] [rbp-240h]
  __int64 *v123; // [rsp+68h] [rbp-238h]
  bool v124; // [rsp+68h] [rbp-238h]
  bool v125; // [rsp+70h] [rbp-230h]
  __int64 v126; // [rsp+70h] [rbp-230h]
  bool v127; // [rsp+78h] [rbp-228h]
  __int64 v128; // [rsp+78h] [rbp-228h]
  unsigned __int8 v129; // [rsp+80h] [rbp-220h]
  int v130; // [rsp+80h] [rbp-220h]
  unsigned __int64 v131; // [rsp+80h] [rbp-220h]
  unsigned __int64 v132; // [rsp+80h] [rbp-220h]
  unsigned __int64 v133; // [rsp+80h] [rbp-220h]
  unsigned __int8 v134; // [rsp+88h] [rbp-218h]
  __int64 v135; // [rsp+88h] [rbp-218h]
  __int64 v136; // [rsp+88h] [rbp-218h]
  __int64 v137; // [rsp+88h] [rbp-218h]
  unsigned int v138; // [rsp+90h] [rbp-210h]
  __int64 *v139; // [rsp+90h] [rbp-210h]
  unsigned __int8 v140; // [rsp+90h] [rbp-210h]
  __int64 v141; // [rsp+90h] [rbp-210h]
  unsigned __int16 v143; // [rsp+AAh] [rbp-1F6h] BYREF
  unsigned int v144; // [rsp+ACh] [rbp-1F4h] BYREF
  __int64 v145; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v146; // [rsp+B8h] [rbp-1E8h]
  __int64 v147; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 v148; // [rsp+C8h] [rbp-1D8h]
  __int64 v149; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v150; // [rsp+D8h] [rbp-1C8h]
  __int64 v151; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v152; // [rsp+E8h] [rbp-1B8h]
  unsigned __int64 v153; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v154; // [rsp+F8h] [rbp-1A8h]
  _QWORD v155[3]; // [rsp+100h] [rbp-1A0h] BYREF
  bool v156; // [rsp+118h] [rbp-188h]
  __int64 *v157; // [rsp+130h] [rbp-170h] BYREF
  __int64 v158; // [rsp+138h] [rbp-168h]
  _BYTE v159[64]; // [rsp+140h] [rbp-160h] BYREF
  _BYTE *v160; // [rsp+180h] [rbp-120h] BYREF
  __int64 v161; // [rsp+188h] [rbp-118h]
  _BYTE v162[272]; // [rsp+190h] [rbp-110h] BYREF

  v4 = *a2;
  *((_DWORD *)a2 + 148) = 0;
  *((_DWORD *)a2 + 208) = 0;
  v5 = a1[14];
  LOBYTE(v161) = 0;
  v6 = a1[16];
  v160 = 0;
  v157 = (__int64 *)v159;
  v158 = 0x400000000LL;
  sub_34B8C80(v6, v5, (__int64)v4, (__int64)&v157, 0, 0, __PAIR128__(v161, 0));
  v9 = 0;
  v10 = (_WORD *)a1[16];
  v160 = v162;
  v154 = 0x200000000LL;
  v11 = *((_BYTE *)a2 + 8);
  v161 = 0x400000000LL;
  v12 = a1[14];
  v153 = (unsigned __int64)v155;
  if ( (v11 & 1) != 0 )
  {
    LODWORD(v155[0]) = 54;
    v9 = 1;
    LODWORD(v154) = 1;
  }
  if ( (v11 & 2) != 0 )
  {
    *((_DWORD *)v155 + v9) = 79;
    v9 = (unsigned int)(v154 + 1);
    LODWORD(v154) = v154 + 1;
    if ( (v11 & 8) == 0 )
      goto LABEL_5;
    if ( v9 + 1 > (unsigned __int64)HIDWORD(v154) )
    {
      sub_C8D5F0((__int64)&v153, v155, v9 + 1, 4u, v7, v8);
      v9 = (unsigned int)v154;
    }
LABEL_83:
    *(_DWORD *)(v153 + 4 * v9) = 15;
    v9 = (unsigned int)(v154 + 1);
    LODWORD(v154) = v154 + 1;
    goto LABEL_5;
  }
  if ( (v11 & 8) != 0 )
    goto LABEL_83;
LABEL_5:
  v13 = sub_A79F10(**a2, 0, (int *)v153, v9);
  if ( (_QWORD *)v153 != v155 )
    _libc_free(v153);
  sub_2FEA900(*((_DWORD *)a2 + 4), (__int64 *)*a2, v13, (__int64)&v160, v10, v12);
  v14 = a1[16];
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 2320LL);
  if ( v15 != sub_302E220
    && !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, bool, _BYTE **, __int64 *, __int64 **))v15)(
          v14,
          *((unsigned int *)a2 + 4),
          *(_QWORD *)(a1[5] + 8LL),
          ((_BYTE)a2[1] & 4) != 0,
          &v160,
          **a2,
          *a2) )
  {
    goto LABEL_74;
  }
  v123 = &v157[2 * (unsigned int)v158];
  if ( v157 != v123 )
  {
    v16 = v157;
    while ( 1 )
    {
      v17 = *v16;
      v18 = v16[1];
      v19 = a1[16];
      v20 = *a2;
      v145 = v17;
      v146 = v18;
      v21 = *v20;
      if ( !(_WORD)v17 )
        break;
      v22 = *(unsigned __int16 *)(v19 + 2LL * (unsigned __int16)v17 + 2852);
LABEL_12:
      v138 = v22;
      v23 = *(__int64 (__fastcall **)(__int64, __int64 *, __int64, __int64, unsigned __int64))(*(_QWORD *)v19 + 736LL);
      BYTE2(v153) = 0;
      v24 = v23(v19, v21, v17, v18, v153);
      v25 = v138;
      v26 = v24;
      if ( v24 )
      {
        v27 = v17;
        v28 = *((unsigned int *)a2 + 148);
        v29 = 0;
        v139 = v16;
        v30 = (__int64)(a2 + 73);
        v31 = v25;
        v32 = v18;
        v33 = v27;
        do
        {
          v34 = *((_BYTE *)a2 + 8);
          LOWORD(v155[0]) = v31;
          v153 = 0;
          v155[1] = v33;
          v155[2] = v32;
          v154 = 0;
          v156 = (v34 & 0x20) != 0;
          if ( (v34 & 1) != 0 )
            LOBYTE(v153) = v153 | 2;
          if ( (v34 & 2) != 0 )
            LOBYTE(v153) = v153 | 1;
          if ( (v34 & 8) != 0 )
            LOBYTE(v153) = v153 | 8;
          v35 = v28 + 1;
          v36 = (unsigned __int64)a2[73];
          v37 = (const __m128i *)&v153;
          if ( v28 + 1 > (unsigned __int64)*((unsigned int *)a2 + 149) )
          {
            v126 = v32;
            v72 = a2 + 75;
            v128 = v33;
            v130 = v26;
            if ( v36 > (unsigned __int64)&v153 || (unsigned __int64)&v153 >= v36 + 56 * v28 )
            {
              v37 = (const __m128i *)&v153;
              sub_C8D5F0(v30, v72, v35, 0x38u, v33, v25);
              v36 = (unsigned __int64)a2[73];
              v28 = *((unsigned int *)a2 + 148);
              v32 = v126;
              v33 = v128;
              v26 = v130;
            }
            else
            {
              v73 = (char *)&v153 - v36;
              sub_C8D5F0(v30, v72, v35, 0x38u, v33, v25);
              v36 = (unsigned __int64)a2[73];
              v28 = *((unsigned int *)a2 + 148);
              v26 = v130;
              v33 = v128;
              v32 = v126;
              v37 = (const __m128i *)&v73[v36];
            }
          }
          ++v29;
          v38 = (__m128i *)(v36 + 56 * v28);
          *v38 = _mm_loadu_si128(v37);
          v38[1] = _mm_loadu_si128(v37 + 1);
          v38[2] = _mm_loadu_si128(v37 + 2);
          v38[3].m128i_i64[0] = v37[3].m128i_i64[0];
          v28 = (unsigned int)(*((_DWORD *)a2 + 148) + 1);
          *((_DWORD *)a2 + 148) = v28;
        }
        while ( v26 != v29 );
        v16 = v139;
      }
      v16 += 2;
      if ( v123 == v16 )
        goto LABEL_24;
    }
    v141 = (__int64)*v20;
    if ( sub_30070B0((__int64)&v145) )
    {
      LOWORD(v153) = 0;
      LOWORD(v149) = 0;
      v154 = 0;
      sub_2FE8D10(
        v19,
        v141,
        (unsigned int)v145,
        v146,
        (__int64 *)&v153,
        (unsigned int *)&v151,
        (unsigned __int16 *)&v149);
      v22 = (unsigned __int16)v149;
      v19 = a1[16];
      v21 = **a2;
      goto LABEL_12;
    }
    if ( !sub_3007070((__int64)&v145) )
      goto LABEL_103;
    v74 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v19 + 592LL);
    if ( v74 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v153, v19, v141, v17, v18);
      v75 = v120;
      LOWORD(v75) = v154;
      v76 = v155[0];
      v77 = v141;
      v120 = v75;
    }
    else
    {
      v95 = v74(v19, v141, v145, v146);
      v77 = v141;
      v120 = v95;
      v76 = v96;
    }
    v148 = v76;
    v78 = (unsigned __int16)v120;
    v147 = v120;
    if ( !(_WORD)v120 )
    {
      v131 = v76;
      v135 = v77;
      if ( sub_30070B0((__int64)&v147) )
      {
        LOWORD(v153) = 0;
        LOWORD(v149) = 0;
        v154 = 0;
        sub_2FE8D10(
          v19,
          v135,
          (unsigned int)v147,
          v131,
          (__int64 *)&v153,
          (unsigned int *)&v151,
          (unsigned __int16 *)&v149);
        v22 = (unsigned __int16)v149;
        goto LABEL_71;
      }
      if ( !sub_3007070((__int64)&v147) )
        goto LABEL_103;
      v79 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v19 + 592LL);
      if ( v79 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v153, v19, v135, v147, v148);
        v80 = v119;
        LOWORD(v80) = v154;
        v81 = v155[0];
        v82 = v135;
        v119 = v80;
      }
      else
      {
        v97 = v79(v19, v135, v147, v131);
        v82 = v135;
        v119 = v97;
        v81 = v98;
      }
      v150 = v81;
      v78 = (unsigned __int16)v119;
      v149 = v119;
      if ( !(_WORD)v119 )
      {
        v132 = v81;
        v136 = v82;
        if ( sub_30070B0((__int64)&v149) )
        {
          LOWORD(v153) = 0;
          LOWORD(v144) = 0;
          v154 = 0;
          sub_2FE8D10(
            v19,
            v136,
            (unsigned int)v149,
            v132,
            (__int64 *)&v153,
            (unsigned int *)&v151,
            (unsigned __int16 *)&v144);
          v22 = (unsigned __int16)v144;
          goto LABEL_71;
        }
        if ( !sub_3007070((__int64)&v149) )
          goto LABEL_103;
        v83 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v19 + 592LL);
        if ( v83 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v153, v19, v136, v149, v150);
          v84 = v116;
          LOWORD(v84) = v154;
          v85 = v155[0];
          v86 = v136;
          v116 = v84;
        }
        else
        {
          v103 = v83(v19, v136, v149, v132);
          v86 = v136;
          v116 = v103;
          v85 = v104;
        }
        v152 = v85;
        v78 = (unsigned __int16)v116;
        v151 = v116;
        if ( !(_WORD)v116 )
        {
          v133 = v85;
          v137 = v86;
          if ( sub_30070B0((__int64)&v151) )
          {
            LOWORD(v153) = 0;
            v143 = 0;
            v154 = 0;
            sub_2FE8D10(v19, v137, (unsigned int)v151, v133, (__int64 *)&v153, &v144, &v143);
            v22 = v143;
          }
          else
          {
            if ( !sub_3007070((__int64)&v151) )
LABEL_103:
              BUG();
            v87 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v19 + 592LL);
            if ( v87 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v153, v19, v137, v151, v152);
              v88 = v113;
              LOWORD(v88) = v154;
              v89 = v155[0];
              v90 = v137;
              v113 = v88;
            }
            else
            {
              v105 = v87(v19, v137, v151, v133);
              v90 = v137;
              v113 = v105;
              v89 = v106;
            }
            v22 = sub_2FE98B0(v19, v90, (unsigned int)v113, v89);
          }
          goto LABEL_71;
        }
      }
    }
    v22 = *(unsigned __int16 *)(v19 + 2 * v78 + 2852);
LABEL_71:
    v19 = a1[16];
    v21 = **a2;
    goto LABEL_12;
  }
LABEL_24:
  v39 = a2[6];
  v40 = a2[5];
  *((_DWORD *)a2 + 24) = 0;
  *((_DWORD *)a2 + 60) = 0;
  *((_DWORD *)a2 + 128) = 0;
  v107 = v39;
  if ( v39 != v40 )
  {
    v41 = v40;
    while ( 1 )
    {
      v48 = *((_BYTE *)v41 + 32);
      v64 = v41[3];
      v49 = (v48 & 0x40) != 0;
      if ( (v48 & 0x40) != 0 )
        v64 = v41[5];
      v134 = 0;
      v50 = a1[14];
      v65 = *(__int64 (**)())(*(_QWORD *)a1[16] + 2368LL);
      if ( v65 != sub_302E240 )
      {
        v91 = ((__int64 (__fastcall *)(_QWORD, __int64 *, _QWORD, bool, _QWORD))v65)(
                a1[16],
                v64,
                *((unsigned int *)a2 + 4),
                ((_BYTE)a2[1] & 4) != 0,
                a1[14]);
        v48 = *((_BYTE *)v41 + 32);
        v134 = v91;
        v50 = a1[14];
        v49 = (v48 & 0x40) != 0;
      }
      v66 = *((_BYTE *)v41 + 35);
      v47 = *((_BYTE *)v41 + 34);
      v129 = v48 & 1;
      v67 = (v48 & 2) != 0;
      v127 = (v48 & 8) != 0;
      v125 = (v48 & 0x10) != 0;
      v68 = *((_BYTE *)v41 + 33);
      v42 = (v68 & 2) != 0;
      v124 = (v68 & 8) != 0;
      v122 = (v68 & 0x10) != 0;
      v121 = (v68 & 0x20) != 0;
      v69 = (v68 & 0x40) != 0;
      v70 = v68 & 1;
      v140 = v70;
      if ( v70 )
        break;
      if ( (*((_BYTE *)v41 + 33) & 2) != 0 )
        goto LABEL_28;
      if ( v49 )
        goto LABEL_29;
      v112 = 0;
      if ( !v66 )
      {
        v71 = sub_AE5020(v50, (__int64)v41[3]);
        v48 = *((_BYTE *)v41 + 32);
        v42 = 0;
        v47 = v71;
        v49 = 0;
        v50 = a1[14];
      }
LABEL_32:
      v51 = v47 + 1;
      v114 = v42;
      v118 = v49;
      v52 = v48 >> 5;
      v53 = sub_AE5020(v50, (__int64)v41[3]);
      v55 = *((unsigned int *)a2 + 24);
      v56 = v52 & 1;
      v57 = (__int64)*v41;
      v58 = v118;
      v59 = v53 + 1;
      v60 = v114;
      if ( v55 + 1 > (unsigned __int64)*((unsigned int *)a2 + 25) )
      {
        v108 = v114;
        v110 = v59;
        v115 = (__int64)*v41;
        sub_C8D5F0((__int64)(a2 + 11), a2 + 13, v55 + 1, 8u, v57, v54);
        v55 = *((unsigned int *)a2 + 24);
        v60 = v108;
        v59 = v110;
        v57 = v115;
        v58 = v118;
      }
      a2[11][v55] = (__int64 *)v57;
      ++*((_DWORD *)a2 + 24);
      v61 = *((unsigned int *)a2 + 60);
      v62 = (((unsigned __int64)v134 << 32)
           | ((unsigned __int64)(v59 & 0x1F) << 26)
           | ((unsigned __int64)(v51 & 0x3F) << 20)
           | ((unsigned __int64)v69 << 16)
           | ((unsigned __int64)v121 << 15)
           | ((unsigned __int64)v122 << 14)
           | ((unsigned __int64)v124 << 13)
           | (v60 << 11)
           | ((unsigned __int64)v140 << 10)
           | (16LL * v125)
           | (8LL * v127)
           | (2LL * v129)
           | v67
           | (32 * v58)
           | ((unsigned __int64)v56 << 7))
          & 0x7FFFFFFFFLL;
      if ( v61 + 1 > (unsigned __int64)*((unsigned int *)a2 + 61) )
      {
        sub_C8D5F0((__int64)(a2 + 29), a2 + 31, v61 + 1, 0x10u, v61 + 1, v54);
        v61 = *((unsigned int *)a2 + 60);
      }
      v63 = (unsigned __int64 *)&a2[29][2 * v61];
      v41 += 6;
      *v63 = v62;
      v63[1] = v112;
      ++*((_DWORD *)a2 + 60);
      if ( v107 == v41 )
        goto LABEL_73;
    }
    if ( (*((_BYTE *)v41 + 33) & 2) != 0 )
    {
      v140 = 1;
LABEL_28:
      v42 = 1;
    }
LABEL_29:
    v109 = v42;
    v111 = (__int64)v41[5];
    v117 = sub_AE5020(v50, v111);
    v43 = sub_9208B0(v50, v111);
    v154 = v44;
    v153 = (((unsigned __int64)(v43 + 7) >> 3) + (1LL << v117) - 1) >> v117 << v117;
    v45 = sub_CA1930(&v153);
    v42 = v109;
    v112 = v45;
    if ( !v66 )
    {
      v46 = (*(__int64 (__fastcall **)(_QWORD, __int64 *, _QWORD))(*(_QWORD *)a1[16] + 728LL))(a1[16], v41[5], a1[14]);
      v42 = v109;
      v47 = v46;
    }
    v48 = *((_BYTE *)v41 + 32);
    v49 = 1;
    v50 = a1[14];
    goto LABEL_32;
  }
LABEL_73:
  v92 = *(__int64 (**)())(*a1 + 40LL);
  if ( v92 == sub_3740EC0 || (v93 = ((__int64 (__fastcall *)(_QWORD *, __int64 ***))v92)(a1, a2), !(_BYTE)v93) )
  {
LABEL_74:
    v93 = 0;
    goto LABEL_75;
  }
  sub_2E8FB70((__int64)a2[9], (unsigned int *)a2[103], *((unsigned int *)a2 + 208), a1[17]);
  v99 = *((_DWORD *)a2 + 21);
  if ( v99 )
  {
    v100 = a2[8];
    if ( !v100 )
      goto LABEL_75;
    sub_3742B00((__int64)a1, v100, *((_DWORD *)a2 + 20), v99);
  }
  v101 = (__int64)a2[8];
  if ( v101 && (*(_QWORD *)(v101 + 48) || (*(_BYTE *)(v101 + 7) & 0x20) != 0) )
  {
    v102 = sub_B91F50(v101, "heapallocsite", 0xDu);
    if ( v102 )
      sub_2E880E0((__int64)a2[9], a1[6], v102);
  }
LABEL_75:
  if ( v160 != v162 )
    _libc_free((unsigned __int64)v160);
  if ( v157 != (__int64 *)v159 )
    _libc_free((unsigned __int64)v157);
  return v93;
}
