// Function: sub_33A33F0
// Address: 0x33a33f0
//
void __fastcall sub_33A33F0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // r13d
  __int64 v12; // r14
  __m128i v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // r13d
  __int64 v21; // rax
  unsigned __int16 v22; // r14
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  char v27; // r10
  __int64 v28; // rax
  int v29; // r14d
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned __int64 v33; // r14
  unsigned __int16 *v34; // rdx
  int v35; // eax
  __int64 v36; // rdx
  __int16 v37; // ax
  __int64 v38; // rdx
  __int64 (*v39)(); // rax
  __int64 v40; // rax
  __int64 v41; // r15
  __m128i v42; // xmm0
  __m128i v43; // xmm1
  int v44; // edx
  __m128i v45; // xmm2
  __m128i v46; // xmm3
  int v47; // eax
  int v48; // edx
  __int64 *v49; // rsi
  __int64 v50; // r15
  __int64 v51; // rdx
  __int64 v52; // r14
  _QWORD *v53; // rax
  __int64 v54; // r12
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 *v58; // rdi
  __int64 (__fastcall *v59)(__int64, __int64, unsigned int); // r13
  __int64 v60; // rax
  _DWORD *v61; // rax
  int v62; // r10d
  int v63; // edx
  unsigned __int16 v64; // ax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // r10
  __int64 v68; // rdx
  __int64 *v69; // rdi
  __int64 v70; // rax
  _DWORD *v71; // rax
  int v72; // r10d
  int v73; // edx
  unsigned __int16 v74; // ax
  __int64 v75; // rax
  __int64 v76; // rdx
  int v77; // esi
  int v78; // eax
  int v79; // r9d
  int v80; // r8d
  int v81; // edx
  int v82; // ecx
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // [rsp-10h] [rbp-420h]
  __int64 v86; // [rsp-10h] [rbp-420h]
  __int64 v87; // [rsp-8h] [rbp-418h]
  __int16 v88; // [rsp+2h] [rbp-40Eh]
  char v89; // [rsp+Fh] [rbp-401h]
  _QWORD *v90; // [rsp+10h] [rbp-400h]
  __int64 v91; // [rsp+20h] [rbp-3F0h]
  __int64 v92; // [rsp+28h] [rbp-3E8h]
  __m128i v93; // [rsp+30h] [rbp-3E0h] BYREF
  __int64 (__fastcall *v94)(__int64, __int64, unsigned int); // [rsp+40h] [rbp-3D0h]
  int v95; // [rsp+48h] [rbp-3C8h]
  int v96; // [rsp+4Ch] [rbp-3C4h]
  __int64 v97; // [rsp+50h] [rbp-3C0h]
  __int64 v98; // [rsp+58h] [rbp-3B8h]
  __int64 *v99; // [rsp+60h] [rbp-3B0h]
  __int64 v100; // [rsp+68h] [rbp-3A8h]
  __int64 v101; // [rsp+70h] [rbp-3A0h]
  __int64 v102; // [rsp+78h] [rbp-398h]
  __int64 v103; // [rsp+80h] [rbp-390h]
  __int64 v104; // [rsp+88h] [rbp-388h]
  __int64 v105; // [rsp+90h] [rbp-380h]
  __int64 v106; // [rsp+98h] [rbp-378h]
  __int64 v107; // [rsp+A0h] [rbp-370h]
  __int64 v108; // [rsp+A8h] [rbp-368h]
  __int64 v109; // [rsp+B0h] [rbp-360h]
  __int64 v110; // [rsp+B8h] [rbp-358h]
  __int64 v111; // [rsp+C0h] [rbp-350h]
  __int64 v112; // [rsp+C8h] [rbp-348h]
  __int64 v113; // [rsp+D0h] [rbp-340h]
  __int64 v114; // [rsp+D8h] [rbp-338h]
  unsigned int v115; // [rsp+E4h] [rbp-32Ch] BYREF
  __int64 v116; // [rsp+E8h] [rbp-328h] BYREF
  __int64 v117; // [rsp+F0h] [rbp-320h] BYREF
  int v118; // [rsp+F8h] [rbp-318h]
  unsigned int v119; // [rsp+100h] [rbp-310h] BYREF
  __int64 v120; // [rsp+108h] [rbp-308h]
  __m128i v121; // [rsp+110h] [rbp-300h] BYREF
  __m128i v122; // [rsp+120h] [rbp-2F0h] BYREF
  __m128i v123; // [rsp+130h] [rbp-2E0h] BYREF
  unsigned int v124; // [rsp+140h] [rbp-2D0h] BYREF
  __int64 v125; // [rsp+148h] [rbp-2C8h]
  __int64 v126; // [rsp+150h] [rbp-2C0h]
  __int64 v127; // [rsp+158h] [rbp-2B8h]
  __int64 v128; // [rsp+160h] [rbp-2B0h] BYREF
  __int64 v129; // [rsp+168h] [rbp-2A8h]
  __int64 v130; // [rsp+170h] [rbp-2A0h]
  __int64 v131; // [rsp+180h] [rbp-290h] BYREF
  __int64 v132; // [rsp+188h] [rbp-288h]
  __int64 v133; // [rsp+190h] [rbp-280h]
  __int64 v134; // [rsp+198h] [rbp-278h]
  __m128i v135; // [rsp+1A0h] [rbp-270h]
  __m128i v136; // [rsp+1B0h] [rbp-260h]
  __m128i v137; // [rsp+1C0h] [rbp-250h]
  __m128i v138; // [rsp+1D0h] [rbp-240h]
  __int64 v139; // [rsp+1E0h] [rbp-230h]
  int v140; // [rsp+1E8h] [rbp-228h]
  _QWORD v141[7]; // [rsp+1F0h] [rbp-220h] BYREF
  char v142; // [rsp+228h] [rbp-1E8h] BYREF
  char *v143; // [rsp+230h] [rbp-1E0h]
  __int64 v144; // [rsp+238h] [rbp-1D8h]
  char v145; // [rsp+240h] [rbp-1D0h] BYREF
  char *v146; // [rsp+270h] [rbp-1A0h]
  __int64 v147; // [rsp+278h] [rbp-198h]
  char v148; // [rsp+280h] [rbp-190h] BYREF
  char *v149; // [rsp+2A0h] [rbp-170h]
  __int64 v150; // [rsp+2A8h] [rbp-168h]
  char v151; // [rsp+2B0h] [rbp-160h] BYREF
  char *v152; // [rsp+300h] [rbp-110h]
  __int64 v153; // [rsp+308h] [rbp-108h]
  char v154; // [rsp+310h] [rbp-100h] BYREF
  char *v155; // [rsp+3B0h] [rbp-60h]
  __int64 v156; // [rsp+3B8h] [rbp-58h]
  char v157; // [rsp+3C0h] [rbp-50h] BYREF
  __int16 v158; // [rsp+3D0h] [rbp-40h]
  __int64 v159; // [rsp+3D8h] [rbp-38h]

  v95 = a3;
  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 848);
  v117 = 0;
  v118 = v6;
  if ( v5 )
  {
    if ( &v117 != (__int64 *)(v5 + 48) )
    {
      v7 = *(_QWORD *)(v5 + 48);
      v117 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v117, v7, 1);
    }
  }
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v98 = *(_QWORD *)(a2 - 32 * v8);
  v9 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1 - v8)));
  v91 = v10;
  v11 = v10;
  v12 = v9;
  v92 = v9;
  v13.m128i_i64[0] = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v93 = v13;
  v13.m128i_i64[0] = *(_QWORD *)(a1 + 864);
  v14 = *(_QWORD *)(v13.m128i_i64[0] + 16);
  v15 = sub_2E79000(*(__int64 **)(v13.m128i_i64[0] + 40));
  memset(v141, 0, 32);
  v157 = 0;
  v141[4] = &v142;
  v143 = &v145;
  v144 = 0x600000000LL;
  v146 = &v148;
  v147 = 0x400000000LL;
  v149 = &v151;
  v150 = 0xA00000000LL;
  v152 = &v154;
  v153 = 0x800000000LL;
  v155 = &v157;
  v141[5] = 0;
  v141[6] = 8;
  v156 = 0;
  v158 = 768;
  v159 = 0;
  sub_AE1EA0((__int64)v141, v15);
  v16 = *(_QWORD *)(a1 + 864);
  v17 = *(_QWORD *)(v12 + 48) + 16LL * v11;
  v18 = *(_QWORD *)(v17 + 8);
  LOWORD(v119) = *(_WORD *)v17;
  v19 = v119;
  v120 = v18;
  LOBYTE(v94) = sub_33CC4A0(v16, v119, v18);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && (v19 = 29, sub_B91C10(a2, 29)) && (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v19 = 4;
    v20 = sub_B91C10(a2, 4);
  }
  else
  {
    v20 = 0;
  }
  v21 = *(_QWORD *)(a1 + 864);
  v22 = v119;
  v121.m128i_i64[0] = 0;
  v121.m128i_i32[2] = 0;
  v23 = *(_QWORD *)(v21 + 384);
  LODWORD(v21) = *(_DWORD *)(v21 + 392);
  v122.m128i_i64[0] = 0;
  v122.m128i_i32[2] = 0;
  v97 = v23;
  v96 = v21;
  v123.m128i_i64[0] = 0;
  v123.m128i_i32[2] = 0;
  if ( (_WORD)v119 )
  {
    if ( (unsigned __int16)(v119 - 17) > 0xD3u )
    {
LABEL_11:
      v24 = v120;
      goto LABEL_12;
    }
    v24 = 0;
    v22 = word_4456580[(unsigned __int16)v119 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v119) )
      goto LABEL_11;
    v22 = sub_3009970((__int64)&v119, v19, v55, v56, v57);
  }
LABEL_12:
  LOWORD(v131) = v22;
  v132 = v24;
  if ( v22 )
  {
    if ( v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
      BUG();
    v99 = &v131;
    v25 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
  }
  else
  {
    v99 = &v131;
    v25 = sub_3007260((__int64)&v131);
    v126 = v25;
    v127 = v26;
  }
  v27 = sub_339D300(
          v98,
          (__int64)&v121,
          (__int64)&v122,
          &v115,
          (__int64)&v123,
          a1,
          *(_QWORD *)(a2 + 40),
          (unsigned __int64)(v25 + 7) >> 3);
  v28 = *(_QWORD *)(v98 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 <= 1 )
  {
    v28 = **(_QWORD **)(v28 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 <= 1 )
      v28 = **(_QWORD **)(v28 + 16);
  }
  v89 = v27;
  v29 = *(_DWORD *)(v28 + 8) >> 8;
  v90 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  sub_B91FC0(v99, a2);
  v30 = 3;
  LODWORD(v130) = v29;
  v129 = 0;
  BYTE4(v130) = 0;
  v128 = 0;
  v33 = sub_2E7BD70(v90, 3u, -2, (unsigned __int8)v94, (int)v99, v20, 0, v130, 1u, 0, 0);
  if ( !v89 )
  {
    v58 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
    v59 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v14 + 32LL);
    v94 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(a1 + 864);
    v60 = sub_2E79000(v58);
    if ( v59 == sub_2D42F30 )
    {
      v61 = sub_AE2980(v60, 0);
      v62 = (int)v94;
      v63 = v61[1];
      v64 = 2;
      if ( v63 != 1 )
      {
        v64 = 3;
        if ( v63 != 2 )
        {
          v64 = 4;
          if ( v63 != 4 )
          {
            v64 = 5;
            if ( v63 != 8 )
            {
              v64 = 6;
              if ( v63 != 16 )
              {
                v64 = 7;
                if ( v63 != 32 )
                {
                  v64 = 8;
                  if ( v63 != 64 )
                    v64 = 9 * (v63 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v64 = v59(v14, v60, 0);
      v62 = (int)v94;
    }
    v113 = sub_3400BD0(v62, 0, (unsigned int)&v117, v64, 0, 0, 0);
    v114 = v65;
    v121.m128i_i64[0] = v113;
    v121.m128i_i32[2] = v65;
    v66 = sub_338B750(a1, v98);
    v67 = *(_QWORD *)(a1 + 864);
    v115 = 0;
    v111 = v66;
    v112 = v68;
    v69 = *(__int64 **)(v67 + 40);
    v122.m128i_i64[0] = v66;
    v98 = v67;
    v122.m128i_i32[2] = v68;
    v94 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v14 + 32LL);
    v70 = sub_2E79000(v69);
    if ( v94 == sub_2D42F30 )
    {
      v71 = sub_AE2980(v70, 0);
      v72 = v98;
      v73 = v71[1];
      v74 = 2;
      if ( v73 != 1 )
      {
        v74 = 3;
        if ( v73 != 2 )
        {
          v74 = 4;
          if ( v73 != 4 )
          {
            v74 = 5;
            if ( v73 != 8 )
            {
              v74 = 6;
              if ( v73 != 16 )
              {
                v74 = 7;
                if ( v73 != 32 )
                {
                  v74 = 8;
                  if ( v73 != 64 )
                    v74 = 9 * (v73 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v74 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64 (__fastcall *)(__int64, __int64, unsigned int), __int64, __int64))v94)(
              v14,
              v70,
              0,
              v94,
              v85,
              v87);
      v72 = v98;
    }
    v75 = sub_3400BD0(v72, 1, (unsigned int)&v117, v74, 0, 1, 0);
    v31 = v86;
    v30 = v87;
    v109 = v75;
    v110 = v76;
    v123.m128i_i64[0] = v75;
    v123.m128i_i32[2] = v76;
  }
  v34 = (unsigned __int16 *)(*(_QWORD *)(v122.m128i_i64[0] + 48) + 16LL * v122.m128i_u32[2]);
  v35 = *v34;
  v36 = *((_QWORD *)v34 + 1);
  LOWORD(v124) = v35;
  v125 = v36;
  if ( (_WORD)v35 )
  {
    v37 = word_4456580[v35 - 1];
    v38 = 0;
  }
  else
  {
    v37 = sub_3009970((__int64)&v124, v30, v36, v31, v32);
  }
  LOWORD(v128) = v37;
  v129 = v38;
  v39 = *(__int64 (**)())(*(_QWORD *)v14 + 696LL);
  if ( v39 != sub_2FE3240
    && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64 *))v39)(v14, v124, v125, &v128) )
  {
    if ( (_WORD)v124 )
    {
      v77 = word_4456340[(unsigned __int16)v124 - 1];
      if ( (unsigned __int16)(v124 - 176) > 0x34u )
        LOWORD(v78) = sub_2D43050(v128, v77);
      else
        LOWORD(v78) = sub_2D43AD0(v128, v77);
      v80 = 0;
    }
    else
    {
      v78 = sub_3009490((unsigned __int16 *)&v124, v128, v129);
      v88 = HIWORD(v78);
      v80 = v81;
    }
    HIWORD(v82) = v88;
    LOWORD(v82) = v78;
    v83 = sub_33FAF80(*(_QWORD *)(a1 + 864), 213, (unsigned int)&v117, v82, v80, v79);
    v108 = v84;
    v107 = v83;
    v122.m128i_i64[0] = v83;
    v122.m128i_i32[2] = v84;
  }
  v40 = sub_3400BD0(*(_QWORD *)(a1 + 864), v95, (unsigned int)&v117, 7, 0, 1, 0);
  v41 = *(_QWORD *)(a1 + 864);
  v42 = _mm_load_si128(&v93);
  v43 = _mm_loadu_si128(&v121);
  v140 = v44;
  v131 = v97;
  v45 = _mm_loadu_si128(&v122);
  v46 = _mm_loadu_si128(&v123);
  v135 = v42;
  LODWORD(v132) = v96;
  v136 = v43;
  v133 = v92;
  v137 = v45;
  v134 = v91;
  v138 = v46;
  LODWORD(v98) = v115;
  v100 = 7;
  v139 = v40;
  v47 = sub_33ED250(v41, 1, 0, v115);
  v49 = &v116;
  v50 = sub_33E74D0(v41, v47, v48, v119, v120, (unsigned int)&v117, (__int64)v99, v100, v33, v98);
  v52 = v51;
  v116 = a2;
  v53 = sub_337DC20(a1 + 8, &v116);
  v106 = v52;
  v105 = v50;
  *v53 = v50;
  *((_DWORD *)v53 + 2) = v106;
  v54 = *(_QWORD *)(a1 + 864);
  if ( v50 )
  {
    nullsub_1875(v50, *(_QWORD *)(a1 + 864), 0);
    v104 = v52;
    v49 = 0;
    v103 = v50;
    *(_QWORD *)(v54 + 384) = v50;
    *(_DWORD *)(v54 + 392) = v104;
    sub_33E2B60(v54, 0);
  }
  else
  {
    v102 = v52;
    v101 = 0;
    *(_QWORD *)(v54 + 384) = 0;
    *(_DWORD *)(v54 + 392) = v102;
  }
  sub_AE4030(v141, (__int64)v49);
  if ( v117 )
    sub_B91220((__int64)&v117, v117);
}
