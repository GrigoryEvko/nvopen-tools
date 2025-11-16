// Function: sub_3392620
// Address: 0x3392620
//
void __fastcall sub_3392620(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  _WORD *v6; // r13
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int); // r12
  __int64 v8; // rax
  int v9; // edx
  unsigned __int16 v10; // ax
  unsigned int v11; // r14d
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int); // r12
  __int64 v13; // rax
  int v14; // edx
  unsigned __int16 v15; // ax
  int v16; // edx
  __int64 v17; // rax
  unsigned int v18; // r12d
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 **v22; // r14
  __int64 v23; // rbx
  __int64 v24; // rax
  __int16 v25; // ax
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rbx
  __int64 (*v31)(); // rdx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 (*v36)(); // rdx
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // r9
  __int32 v43; // edx
  __int16 v44; // ax
  __int64 v45; // rdx
  unsigned int v46; // edx
  __int64 v47; // r9
  __int64 v48; // r14
  __int64 v49; // rsi
  __int64 v50; // rax
  int v51; // ecx
  int v52; // edx
  __int128 v53; // rax
  int v54; // r9d
  __int128 v55; // rax
  __int64 v56; // r13
  __int128 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // r12
  __int64 v61; // rdx
  __int64 v62; // r13
  __int128 v63; // rax
  int v64; // r9d
  __int64 v65; // rax
  int v66; // edx
  __int64 v67; // r12
  __int64 v68; // rbx
  int v69; // r13d
  __int64 v70; // r14
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rax
  char v74; // al
  __int64 v75; // rcx
  __m128i *v76; // rsi
  __int64 v77; // rax
  __int64 v78; // rdx
  int v79; // esi
  __int64 v80; // rsi
  __int64 v81; // rsi
  __int64 v82; // rax
  int v83; // edx
  __int64 v84; // rax
  char v85; // al
  char v86; // al
  char v87; // al
  char v88; // al
  unsigned __int64 v89; // rdi
  __int64 v90; // rax
  const __m128i *v91; // rax
  __int64 v92; // r13
  __int64 v93; // r12
  int v94; // r15d
  unsigned int v95; // edx
  __int128 v96; // [rsp-30h] [rbp-1310h]
  __int128 v97; // [rsp-20h] [rbp-1300h]
  __int128 v98; // [rsp-20h] [rbp-1300h]
  char v99; // [rsp+8h] [rbp-12D8h]
  __int64 (__fastcall *v100)(_WORD *, __int64, __int64, __int64, __int64); // [rsp+8h] [rbp-12D8h]
  __int64 v101; // [rsp+8h] [rbp-12D8h]
  __int64 v102; // [rsp+10h] [rbp-12D0h]
  __int64 v103; // [rsp+10h] [rbp-12D0h]
  int v104; // [rsp+10h] [rbp-12D0h]
  __int64 v105; // [rsp+18h] [rbp-12C8h]
  __int64 v106; // [rsp+20h] [rbp-12C0h]
  __int16 v107; // [rsp+28h] [rbp-12B8h]
  __int64 v108; // [rsp+28h] [rbp-12B8h]
  __int64 v109; // [rsp+30h] [rbp-12B0h]
  int v110; // [rsp+40h] [rbp-12A0h]
  __int128 v111; // [rsp+40h] [rbp-12A0h]
  __int64 v112; // [rsp+40h] [rbp-12A0h]
  __int64 v113; // [rsp+40h] [rbp-12A0h]
  __int64 v114; // [rsp+40h] [rbp-12A0h]
  __int64 v115; // [rsp+40h] [rbp-12A0h]
  int v117; // [rsp+78h] [rbp-1268h]
  __int64 v118; // [rsp+D8h] [rbp-1208h] BYREF
  __int64 v119; // [rsp+E0h] [rbp-1200h] BYREF
  int v120; // [rsp+E8h] [rbp-11F8h]
  unsigned __int64 v121; // [rsp+F0h] [rbp-11F0h] BYREF
  __m128i *v122; // [rsp+F8h] [rbp-11E8h]
  const __m128i *v123; // [rsp+100h] [rbp-11E0h]
  __int128 v124; // [rsp+110h] [rbp-11D0h] BYREF
  __int64 v125; // [rsp+120h] [rbp-11C0h]
  __int128 v126; // [rsp+130h] [rbp-11B0h] BYREF
  __int64 v127; // [rsp+140h] [rbp-11A0h]
  __int64 v128; // [rsp+148h] [rbp-1198h]
  __m128i v129; // [rsp+150h] [rbp-1190h] BYREF
  __m128i v130; // [rsp+160h] [rbp-1180h] BYREF
  __m128i v131; // [rsp+170h] [rbp-1170h] BYREF
  __int64 v132; // [rsp+180h] [rbp-1160h] BYREF
  __int64 v133; // [rsp+188h] [rbp-1158h]
  __int64 v134; // [rsp+190h] [rbp-1150h]
  unsigned __int64 v135; // [rsp+198h] [rbp-1148h]
  __int64 v136; // [rsp+1A0h] [rbp-1140h]
  __int64 v137; // [rsp+1A8h] [rbp-1138h]
  __int64 v138; // [rsp+1B0h] [rbp-1130h]
  unsigned __int64 v139; // [rsp+1B8h] [rbp-1128h]
  __m128i *v140; // [rsp+1C0h] [rbp-1120h]
  const __m128i *v141; // [rsp+1C8h] [rbp-1118h]
  __int64 v142; // [rsp+1D0h] [rbp-1110h]
  __int64 v143; // [rsp+1D8h] [rbp-1108h] BYREF
  int v144; // [rsp+1E0h] [rbp-1100h]
  __int64 v145; // [rsp+1E8h] [rbp-10F8h]
  _BYTE *v146; // [rsp+1F0h] [rbp-10F0h]
  __int64 v147; // [rsp+1F8h] [rbp-10E8h]
  _BYTE v148[1792]; // [rsp+200h] [rbp-10E0h] BYREF
  _BYTE *v149; // [rsp+900h] [rbp-9E0h]
  __int64 v150; // [rsp+908h] [rbp-9D8h]
  _BYTE v151[512]; // [rsp+910h] [rbp-9D0h] BYREF
  _BYTE *v152; // [rsp+B10h] [rbp-7D0h]
  __int64 v153; // [rsp+B18h] [rbp-7C8h]
  _BYTE v154[1792]; // [rsp+B20h] [rbp-7C0h] BYREF
  _BYTE *v155; // [rsp+1220h] [rbp-C0h]
  __int64 v156; // [rsp+1228h] [rbp-B8h]
  _BYTE v157[64]; // [rsp+1230h] [rbp-B0h] BYREF
  __int64 v158; // [rsp+1270h] [rbp-70h]
  __int64 v159; // [rsp+1278h] [rbp-68h]
  int v160; // [rsp+1280h] [rbp-60h]
  char v161; // [rsp+12A0h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 864);
  v6 = *(_WORD **)(v5 + 16);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v6 + 32LL);
  v8 = sub_2E79000(*(__int64 **)(v5 + 40));
  if ( v7 == sub_2D42F30 )
  {
    v9 = sub_AE2980(v8, 0)[1];
    v10 = 2;
    if ( v9 != 1 )
    {
      v10 = 3;
      if ( v9 != 2 )
      {
        v10 = 4;
        if ( v9 != 4 )
        {
          v10 = 5;
          if ( v9 != 8 )
          {
            v10 = 6;
            if ( v9 != 16 )
            {
              v10 = 7;
              if ( v9 != 32 )
              {
                v10 = 8;
                if ( v9 != 64 )
                  v10 = 9 * (v9 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v10 = v7((__int64)v6, v8, 0);
  }
  v11 = v10;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v6 + 40LL);
  v13 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  if ( v12 == sub_2D42FA0 )
  {
    v14 = sub_AE2980(v13, 0)[1];
    v15 = 2;
    if ( v14 != 1 )
    {
      v15 = 3;
      if ( v14 != 2 )
      {
        v15 = 4;
        if ( v14 != 4 )
        {
          v15 = 5;
          if ( v14 != 8 )
          {
            v15 = 6;
            if ( v14 != 16 )
            {
              v15 = 7;
              if ( v14 != 32 )
              {
                v15 = 8;
                if ( v14 != 64 )
                  v15 = 9 * (v14 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v15 = v12((__int64)v6, v13, 0);
  }
  v16 = *(_DWORD *)(a1 + 848);
  v119 = 0;
  v110 = v15;
  v17 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 48LL);
  v120 = v16;
  v18 = *(_DWORD *)(v17 + 68);
  v19 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    if ( &v119 != (__int64 *)(v19 + 48) )
    {
      v20 = *(_QWORD *)(v19 + 48);
      v119 = v20;
      if ( v20 )
        sub_B96E90((__int64)&v119, v20, 1);
    }
  }
  v102 = sub_33EDBD0(*(_QWORD *)(a1 + 864), v18, v11, 0, 0);
  v105 = v21;
  v22 = *(__int64 ***)(**(_QWORD **)(a3 + 32) + 40LL);
  v23 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v24 = sub_BCE3C0(*v22, 0);
  LOBYTE(v25) = sub_AE5260(v23, v24);
  v26 = *(_QWORD *)(a1 + 864);
  v132 = 0;
  v99 = v25;
  HIBYTE(v25) = 1;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v107 = v25;
  sub_2EAC300((__int64)&v124, *(_QWORD *)(v26 + 40), v18, 0);
  v27 = sub_33F1F00(
          v26,
          v110,
          0,
          (unsigned int)&v119,
          (unsigned int)*(_QWORD *)(a1 + 864) + 288,
          0,
          v102,
          v105,
          v124,
          v125,
          v107,
          4,
          (__int64)&v132,
          0);
  v28 = *(_QWORD *)v6;
  v30 = v29;
  v31 = *(__int64 (**)())(*(_QWORD *)v6 + 944LL);
  if ( v31 != sub_2FC91F0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_WORD *))v31)(v6) )
    {
      v27 = (*(__int64 (__fastcall **)(_WORD *, _QWORD, __int64, unsigned __int64, __int64 *))(*(_QWORD *)v6 + 2624LL))(
              v6,
              *(_QWORD *)(a1 + 864),
              v27,
              v30,
              &v119);
      v30 = v95 | v30 & 0xFFFFFFFF00000000LL;
    }
    v28 = *(_QWORD *)v6;
  }
  v32 = (*(__int64 (__fastcall **)(_WORD *, __int64 **))(v28 + 952))(v6, v22);
  v33 = v32;
  if ( !v32 )
  {
    v34 = *(_QWORD *)(a1 + 864);
    DWORD2(v126) = 0;
    *(_QWORD *)&v126 = v34 + 288;
    v35 = *(_QWORD *)v6;
    v36 = *(__int64 (**)())(*(_QWORD *)v6 + 2616LL);
    if ( v36 != sub_302E2B0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(_WORD *, __int64 **, __int64 (*)(), _QWORD))v36)(v6, v22, v36, 0) )
      {
        v47 = sub_3367BF0(*(_QWORD *)(a1 + 864), (__int64)&v119, &v126);
LABEL_31:
        v48 = v46;
        v49 = *(_QWORD *)(a1 + 864);
        v106 = v47;
        v100 = *(__int64 (__fastcall **)(_WORD *, __int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 528LL);
        v109 = *(_QWORD *)(v49 + 64);
        v103 = *(_QWORD *)(*(_QWORD *)(v47 + 48) + 16LL * v46 + 8);
        v108 = *(unsigned __int16 *)(*(_QWORD *)(v47 + 48) + 16LL * v46);
        v50 = sub_2E79000(*(__int64 **)(v49 + 40));
        v51 = v100(v6, v50, v109, v108, v103);
        LODWORD(v103) = v52;
        LODWORD(v108) = v51;
        *(_QWORD *)&v53 = sub_33ED040(v49, 22);
        *((_QWORD *)&v97 + 1) = v30;
        *(_QWORD *)&v97 = v27;
        *((_QWORD *)&v96 + 1) = v48;
        *(_QWORD *)&v96 = v106;
        *(_QWORD *)&v55 = sub_340F900(v49, 208, (unsigned int)&v119, v108, v103, v54, v96, v97, v53);
        v56 = *(_QWORD *)(a1 + 864);
        v111 = v55;
        *(_QWORD *)&v57 = sub_33EEAD0(v56, *(_QWORD *)(a2 + 16));
        v58 = sub_340F900(
                v56,
                305,
                (unsigned int)&v119,
                1,
                0,
                DWORD2(v111),
                *(_OWORD *)*(_QWORD *)(v27 + 40),
                v111,
                v57);
        v59 = *(_QWORD *)(a1 + 864);
        v60 = v58;
        v62 = v61;
        *(_QWORD *)&v63 = sub_33EEAD0(v59, *(_QWORD *)(a2 + 8));
        *((_QWORD *)&v98 + 1) = v62;
        *(_QWORD *)&v98 = v60;
        v65 = sub_3406EB0(v59, 301, (unsigned int)&v119, 1, 0, v64, v98, v63);
        v67 = *(_QWORD *)(a1 + 864);
        v68 = v65;
        v69 = v66;
        if ( v65 )
        {
          nullsub_1875(v65, *(_QWORD *)(a1 + 864), 0);
          *(_QWORD *)(v67 + 384) = v68;
          *(_DWORD *)(v67 + 392) = v69;
          sub_33E2B60(v67, 0);
        }
        else
        {
          *(_QWORD *)(v67 + 384) = 0;
          *(_DWORD *)(v67 + 392) = v66;
        }
        goto LABEL_33;
      }
      v35 = *(_QWORD *)v6;
    }
    v37 = (*(__int64 (__fastcall **)(_WORD *, __int64 **, __int64 (*)(), __int64))(v35 + 936))(v6, v22, v36, v33);
    v38 = sub_338B750(a1, v37);
    v130.m128i_i8[4] = 0;
    v39 = *(_QWORD *)(a1 + 864);
    v40 = v38;
    v42 = v41;
    LOBYTE(v44) = v99;
    v132 = 0;
    v129.m128i_i64[0] = v37 & 0xFFFFFFFFFFFFFFFBLL;
    v43 = 0;
    v133 = 0;
    HIBYTE(v44) = 1;
    v134 = 0;
    v135 = 0;
    v129.m128i_i64[1] = 0;
    if ( v37 )
    {
      v45 = *(_QWORD *)(v37 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17 <= 1 )
        v45 = **(_QWORD **)(v45 + 16);
      v43 = *(_DWORD *)(v45 + 8) >> 8;
    }
    v130.m128i_i32[0] = v43;
    v47 = sub_33F1F00(
            v39,
            v110,
            0,
            (unsigned int)&v119,
            v126,
            DWORD2(v126),
            v40,
            v42,
            *(_OWORD *)&v129,
            v130.m128i_i64[0],
            v44,
            4,
            (__int64)&v132,
            0);
    goto LABEL_31;
  }
  v70 = *(_QWORD *)(v32 + 24);
  v71 = v32;
  v130.m128i_i64[1] = 0;
  v123 = 0;
  v129.m128i_i64[0] = 0;
  v131 = 0u;
  v129.m128i_i64[1] = v27;
  v130.m128i_i32[0] = v30;
  v72 = *(_QWORD *)(v70 + 16);
  v122 = 0;
  v73 = *(_QWORD *)(v72 + 8);
  v112 = v33;
  v121 = 0;
  v130.m128i_i64[1] = v73;
  v74 = sub_B2D640(v71, 0, 15);
  v75 = v112;
  if ( v74 )
    v131.m128i_i8[0] |= 8u;
  v76 = v122;
  if ( v122 == v123 )
  {
    sub_332CDC0(&v121, v122, &v129);
    v75 = v112;
  }
  else
  {
    if ( v122 )
    {
      *v122 = _mm_loadu_si128(&v129);
      v76[1] = _mm_loadu_si128(&v130);
      v76[2] = _mm_loadu_si128(&v131);
      v76 = v122;
    }
    v122 = v76 + 3;
  }
  v77 = *(_QWORD *)(a1 + 864);
  v135 = 0xFFFFFFFF00000020LL;
  v146 = v148;
  v147 = 0x2000000000LL;
  v149 = v151;
  v150 = 0x2000000000LL;
  v153 = 0x2000000000LL;
  v78 = *(_QWORD *)a1;
  v156 = 0x400000000LL;
  v79 = *(_DWORD *)(a1 + 848);
  v152 = v154;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = v77;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v155 = v157;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  *(_QWORD *)&v126 = 0;
  DWORD2(v126) = v79;
  if ( v78 )
  {
    if ( &v126 != (__int128 *)(v78 + 48) )
    {
      v80 = *(_QWORD *)(v78 + 48);
      *(_QWORD *)&v126 = v80;
      if ( v80 )
      {
        v113 = v75;
        sub_B96E90((__int64)&v126, v80, 1);
        v75 = v113;
        if ( v143 )
        {
          sub_B91220((__int64)&v143, v143);
          v81 = v126;
          v75 = v113;
        }
        else
        {
          v81 = v126;
        }
        v143 = v81;
        if ( v81 )
        {
          v114 = v75;
          sub_B96E90((__int64)&v143, v81, 1);
          v77 = *(_QWORD *)(a1 + 864);
          v75 = v114;
        }
        else
        {
          v77 = *(_QWORD *)(a1 + 864);
        }
      }
    }
  }
  v115 = v75;
  v144 = DWORD2(v126);
  v132 = v77 + 288;
  LODWORD(v133) = 0;
  v82 = sub_338B750(a1, v75);
  v104 = v83;
  LOWORD(v83) = *(_WORD *)(v115 + 2);
  v101 = v82;
  v84 = **(_QWORD **)(v70 + 16);
  v118 = 0;
  LODWORD(v115) = ((unsigned __int16)v83 >> 4) & 0x3FF;
  v134 = v84;
  v85 = sub_A73170(&v118, 15);
  LOBYTE(v135) = (8 * (v85 & 1)) | v135 & 0xF7;
  v86 = sub_A73170(&v118, 54);
  LOBYTE(v135) = v86 & 1 | v135 & 0xFE;
  v87 = sub_A73170(&v118, 79);
  LOBYTE(v135) = (2 * (v87 & 1)) | v135 & 0xFD;
  v88 = sub_A73170(&v118, 32);
  v89 = v139;
  LODWORD(v136) = v115;
  v137 = v101;
  BYTE1(v135) = (2 * (v88 & 1)) | BYTE1(v135) & 0xFD;
  v139 = v121;
  LODWORD(v138) = v104;
  v90 = (__int64)v122->m128i_i64 - v121;
  v140 = v122;
  v121 = 0;
  v122 = 0;
  HIDWORD(v135) = -1431655765 * (v90 >> 4);
  v91 = v123;
  v123 = 0;
  v141 = v91;
  if ( v89 )
    j_j___libc_free_0(v89);
  if ( (_QWORD)v126 )
    sub_B91220((__int64)&v126, v126);
  sub_3377410((__int64)&v126, v6, (__int64)&v132);
  v92 = v127;
  v93 = *(_QWORD *)(a1 + 864);
  v94 = v128;
  if ( v127 )
  {
    nullsub_1875(v127, v93, 0);
    *(_QWORD *)(v93 + 384) = v92;
    *(_DWORD *)(v93 + 392) = v94;
    sub_33E2B60(v93, 0);
  }
  else
  {
    v117 = v128;
    *(_QWORD *)(v93 + 384) = 0;
    *(_DWORD *)(v93 + 392) = v117;
  }
  if ( v155 != v157 )
    _libc_free((unsigned __int64)v155);
  if ( v152 != v154 )
    _libc_free((unsigned __int64)v152);
  if ( v149 != v151 )
    _libc_free((unsigned __int64)v149);
  if ( v146 != v148 )
    _libc_free((unsigned __int64)v146);
  if ( v143 )
    sub_B91220((__int64)&v143, v143);
  if ( v139 )
    j_j___libc_free_0(v139);
  if ( v121 )
    j_j___libc_free_0(v121);
LABEL_33:
  if ( v119 )
    sub_B91220((__int64)&v119, v119);
}
