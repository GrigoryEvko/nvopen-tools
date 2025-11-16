// Function: sub_339E9C0
// Address: 0x339e9c0
//
void __fastcall sub_339E9C0(__int64 a1, __int64 a2)
{
  int v2; // r13d
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __m128i v11; // rax
  __int64 *v12; // r14
  __int64 *v13; // rdi
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  bool v18; // cc
  unsigned __int64 v19; // rax
  char v20; // r14
  unsigned __int16 v21; // r15
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rax
  unsigned __int16 v27; // r14
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  char v32; // r10
  __int64 v33; // rax
  int v34; // r14d
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // r8
  unsigned __int64 v38; // r13
  unsigned __int16 *v39; // rdx
  int v40; // eax
  __int64 v41; // rdx
  __int16 v42; // ax
  __int64 v43; // rdx
  __int64 (*v44)(); // rax
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __m128i v48; // xmm3
  __m128i v49; // xmm4
  __int64 v50; // rdi
  int v51; // eax
  int v52; // edx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // r15
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r13
  __int64 *v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  bool v65; // al
  __int64 v66; // rdx
  __int64 v67; // rcx
  unsigned __int16 v68; // ax
  __int64 *v69; // rdi
  __int64 v70; // rax
  __int64 (__fastcall *v71)(__int64, __int64, unsigned int); // r14
  __int64 v72; // rax
  _DWORD *v73; // rax
  int v74; // r10d
  int v75; // edx
  unsigned __int16 v76; // ax
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // r10
  __int64 v80; // rdx
  __int64 *v81; // rdi
  __int64 v82; // rax
  _DWORD *v83; // rax
  int v84; // r10d
  int v85; // edx
  unsigned __int16 v86; // ax
  __int64 v87; // rax
  __int64 v88; // rdx
  int v89; // esi
  int v90; // eax
  int v91; // r9d
  int v92; // r8d
  int v93; // edx
  int v94; // ecx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // [rsp-10h] [rbp-200h]
  __int64 v98; // [rsp-10h] [rbp-200h]
  __int64 v99; // [rsp-8h] [rbp-1F8h]
  __int16 v100; // [rsp+Ah] [rbp-1E6h]
  char v101; // [rsp+17h] [rbp-1D9h]
  _QWORD *v102; // [rsp+18h] [rbp-1D8h]
  __m128i v103; // [rsp+20h] [rbp-1D0h] BYREF
  __m128i v104; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v105; // [rsp+40h] [rbp-1B0h]
  __int64 (__fastcall *v106)(__int64, __int64, unsigned int); // [rsp+48h] [rbp-1A8h]
  __int64 *v107; // [rsp+50h] [rbp-1A0h]
  __int64 v108; // [rsp+58h] [rbp-198h]
  __int64 (__fastcall *v109)(__int64, __int64, unsigned int); // [rsp+60h] [rbp-190h]
  __int64 *v110; // [rsp+68h] [rbp-188h]
  __int64 v111; // [rsp+70h] [rbp-180h]
  __int64 v112; // [rsp+78h] [rbp-178h]
  __int64 v113; // [rsp+80h] [rbp-170h]
  __int64 v114; // [rsp+88h] [rbp-168h]
  __int64 v115; // [rsp+90h] [rbp-160h]
  __int64 v116; // [rsp+98h] [rbp-158h]
  __int64 v117; // [rsp+A0h] [rbp-150h]
  __int64 v118; // [rsp+A8h] [rbp-148h]
  __int64 v119; // [rsp+B0h] [rbp-140h]
  __int64 v120; // [rsp+B8h] [rbp-138h]
  int v121; // [rsp+C4h] [rbp-12Ch] BYREF
  __int64 v122; // [rsp+C8h] [rbp-128h] BYREF
  __int64 v123; // [rsp+D0h] [rbp-120h] BYREF
  int v124; // [rsp+D8h] [rbp-118h]
  unsigned int v125; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v126; // [rsp+E8h] [rbp-108h]
  __m128i v127; // [rsp+F0h] [rbp-100h] BYREF
  __m128i v128; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v129; // [rsp+110h] [rbp-E0h] BYREF
  unsigned int v130; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v131; // [rsp+128h] [rbp-C8h]
  __int64 v132; // [rsp+130h] [rbp-C0h]
  __int64 v133; // [rsp+138h] [rbp-B8h]
  __int64 v134; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v135; // [rsp+148h] [rbp-A8h]
  __int64 v136; // [rsp+150h] [rbp-A0h]
  __int64 v137; // [rsp+160h] [rbp-90h] BYREF
  __int64 v138; // [rsp+168h] [rbp-88h]
  __m128i v139; // [rsp+170h] [rbp-80h]
  __m128i v140; // [rsp+180h] [rbp-70h]
  __m128i v141; // [rsp+190h] [rbp-60h]
  __m128i v142; // [rsp+1A0h] [rbp-50h]
  __m128i v143; // [rsp+1B0h] [rbp-40h]

  v5 = *(_DWORD *)(a1 + 848);
  v6 = *(_QWORD *)a1;
  v123 = 0;
  v124 = v5;
  if ( v6 )
  {
    if ( &v123 != (__int64 *)(v6 + 48) )
    {
      v7 = *(_QWORD *)(v6 + 48);
      v123 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v123, v7, 1);
    }
  }
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v107 = *(__int64 **)(a2 - 32 * v8);
  v9 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (3 - v8)));
  v104.m128i_i64[1] = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 4);
  v104.m128i_i64[0] = v9;
  v11.m128i_i64[0] = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (2 - (v10 & 0x7FFFFFF))));
  v12 = *(__int64 **)(a2 + 8);
  v103 = v11;
  v11.m128i_i64[0] = *(_QWORD *)(a1 + 864);
  v13 = *(__int64 **)(v11.m128i_i64[0] + 40);
  v110 = *(__int64 **)(v11.m128i_i64[0] + 16);
  v14 = sub_2E79000(v13);
  v15 = sub_2D5BAE0((__int64)v110, v14, v12, 0);
  v126 = v16;
  LODWORD(v16) = *(_DWORD *)(a2 + 4);
  v125 = v15;
  v17 = *(_QWORD *)(a2 + 32 * (1 - (v16 & 0x7FFFFFF)));
  v18 = *(_DWORD *)(v17 + 32) <= 0x40u;
  v19 = *(_QWORD *)(v17 + 24);
  if ( !v18 )
    v19 = *(_QWORD *)v19;
  v20 = 0;
  if ( v19 )
  {
    _BitScanReverse64(&v19, v19);
    v20 = 1;
    v2 = 63 - (v19 ^ 0x3F);
  }
  v21 = v125;
  v22 = *(_QWORD *)(a1 + 864);
  if ( (_WORD)v125 )
  {
    if ( (unsigned __int16)(v125 - 17) > 0xD3u )
    {
LABEL_11:
      v23 = v126;
      goto LABEL_12;
    }
    v23 = 0;
    v21 = word_4456580[(unsigned __int16)v125 - 1];
  }
  else
  {
    v106 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(a1 + 864);
    v109 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))&v125;
    v65 = sub_30070B0((__int64)&v125);
    v22 = (__int64)v106;
    if ( !v65 )
      goto LABEL_11;
    v109 = v106;
    v68 = sub_3009970((__int64)&v125, v14, v66, v67, (__int64)v106);
    v22 = (__int64)v106;
    v21 = v68;
  }
LABEL_12:
  v24 = v21;
  v25 = sub_33CC4A0(v22, v21, v23);
  if ( !v20 )
    v2 = v25;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && (v24 = 29, sub_B91C10(a2, 29)) && (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v24 = 4;
    v109 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))sub_B91C10(a2, 4);
  }
  else
  {
    v109 = 0;
  }
  v26 = *(_QWORD *)(a1 + 864);
  v27 = v125;
  v127.m128i_i64[0] = 0;
  v127.m128i_i32[2] = 0;
  v28 = *(_QWORD *)(v26 + 384);
  LODWORD(v26) = *(_DWORD *)(v26 + 392);
  v128.m128i_i64[0] = 0;
  v128.m128i_i32[2] = 0;
  v105 = v28;
  LODWORD(v106) = v26;
  v129.m128i_i64[0] = 0;
  v129.m128i_i32[2] = 0;
  if ( (_WORD)v125 )
  {
    if ( (unsigned __int16)(v125 - 17) > 0xD3u )
    {
LABEL_20:
      v29 = v126;
      goto LABEL_21;
    }
    v29 = 0;
    v27 = word_4456580[(unsigned __int16)v125 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v125) )
      goto LABEL_20;
    v27 = sub_3009970((__int64)&v125, v24, v62, v63, v64);
  }
LABEL_21:
  LOWORD(v137) = v27;
  v138 = v29;
  if ( v27 )
  {
    if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
      BUG();
    v30 = *(_QWORD *)&byte_444C4A0[16 * v27 - 16];
  }
  else
  {
    v30 = sub_3007260((__int64)&v137);
    v132 = v30;
    v133 = v31;
  }
  v32 = sub_339D300(
          (__int64)v107,
          (__int64)&v127,
          (__int64)&v128,
          &v121,
          (__int64)&v129,
          a1,
          *(_QWORD *)(a2 + 40),
          (unsigned __int64)(v30 + 7) >> 3);
  v33 = v107[1];
  if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 <= 1 )
  {
    v33 = **(_QWORD **)(v33 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 <= 1 )
      v33 = **(_QWORD **)(v33 + 16);
  }
  v101 = v32;
  v34 = *(_DWORD *)(v33 + 8) >> 8;
  v102 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  sub_B91FC0(&v137, a2);
  v35 = 1;
  LODWORD(v136) = v34;
  v135 = 0;
  BYTE4(v136) = 0;
  v134 = 0;
  v38 = sub_2E7BD70(v102, 1u, -1, v2, (int)&v137, (int)v109, 0, v136, 1u, 0, 0);
  if ( !v101 )
  {
    v69 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
    v70 = *v110;
    v109 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(a1 + 864);
    v71 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v70 + 32);
    v72 = sub_2E79000(v69);
    if ( v71 == sub_2D42F30 )
    {
      v73 = sub_AE2980(v72, 0);
      v74 = (int)v109;
      v75 = v73[1];
      v76 = 2;
      if ( v75 != 1 )
      {
        v76 = 3;
        if ( v75 != 2 )
        {
          v76 = 4;
          if ( v75 != 4 )
          {
            v76 = 5;
            if ( v75 != 8 )
            {
              v76 = 6;
              if ( v75 != 16 )
              {
                v76 = 7;
                if ( v75 != 32 )
                {
                  v76 = 8;
                  if ( v75 != 64 )
                    v76 = 9 * (v75 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v76 = v71((__int64)v110, v72, 0);
      v74 = (int)v109;
    }
    v119 = sub_3400BD0(v74, 0, (unsigned int)&v123, v76, 0, 0, 0);
    v120 = v77;
    v127.m128i_i64[0] = v119;
    v127.m128i_i32[2] = v77;
    v78 = sub_338B750(a1, (__int64)v107);
    v79 = *(_QWORD *)(a1 + 864);
    v121 = 0;
    v117 = v78;
    v118 = v80;
    v81 = *(__int64 **)(v79 + 40);
    v128.m128i_i64[0] = v78;
    v107 = (__int64 *)v79;
    v128.m128i_i32[2] = v80;
    v109 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*v110 + 32);
    v82 = sub_2E79000(v81);
    if ( v109 == sub_2D42F30 )
    {
      v83 = sub_AE2980(v82, 0);
      v84 = (int)v107;
      v85 = v83[1];
      v86 = 2;
      if ( v85 != 1 )
      {
        v86 = 3;
        if ( v85 != 2 )
        {
          v86 = 4;
          if ( v85 != 4 )
          {
            v86 = 5;
            if ( v85 != 8 )
            {
              v86 = 6;
              if ( v85 != 16 )
              {
                v86 = 7;
                if ( v85 != 32 )
                {
                  v86 = 8;
                  if ( v85 != 64 )
                    v86 = 9 * (v85 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v86 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, __int64 (__fastcall *)(__int64, __int64, unsigned int), __int64, __int64))v109)(
              v110,
              v82,
              0,
              v109,
              v97,
              v99);
      v84 = (int)v107;
    }
    v87 = sub_3400BD0(v84, 1, (unsigned int)&v123, v86, 0, 1, 0);
    v36 = v98;
    v35 = v99;
    v115 = v87;
    v116 = v88;
    v129.m128i_i64[0] = v87;
    v129.m128i_i32[2] = v88;
  }
  v39 = (unsigned __int16 *)(*(_QWORD *)(v128.m128i_i64[0] + 48) + 16LL * v128.m128i_u32[2]);
  v40 = *v39;
  v41 = *((_QWORD *)v39 + 1);
  LOWORD(v130) = v40;
  v131 = v41;
  if ( (_WORD)v40 )
  {
    v42 = word_4456580[v40 - 1];
    v43 = 0;
  }
  else
  {
    v42 = sub_3009970((__int64)&v130, v35, v41, v36, v37);
  }
  LOWORD(v134) = v42;
  v135 = v43;
  v44 = *(__int64 (**)())(*v110 + 696);
  if ( v44 != sub_2FE3240
    && ((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, __int64, __int64 *))v44)(v110, v130, v131, &v134) )
  {
    if ( (_WORD)v130 )
    {
      v89 = word_4456340[(unsigned __int16)v130 - 1];
      if ( (unsigned __int16)(v130 - 176) > 0x34u )
        LOWORD(v90) = sub_2D43050(v134, v89);
      else
        LOWORD(v90) = sub_2D43AD0(v134, v89);
      v92 = 0;
    }
    else
    {
      v90 = sub_3009490((unsigned __int16 *)&v130, v134, v135);
      v100 = HIWORD(v90);
      v92 = v93;
    }
    HIWORD(v94) = v100;
    LOWORD(v94) = v90;
    v95 = sub_33FAF80(*(_QWORD *)(a1 + 864), 213, (unsigned int)&v123, v94, v92, v91);
    v114 = v96;
    v113 = v95;
    v128.m128i_i64[0] = v95;
    v128.m128i_i32[2] = v96;
  }
  v45 = _mm_load_si128(&v104);
  v46 = _mm_load_si128(&v103);
  v47 = _mm_load_si128(&v127);
  v107 = &v137;
  v48 = _mm_load_si128(&v128);
  v49 = _mm_load_si128(&v129);
  v137 = v105;
  v50 = *(_QWORD *)(a1 + 864);
  v108 = 6;
  v139 = v45;
  v110 = (__int64 *)v50;
  LODWORD(v109) = v121;
  v140 = v46;
  v141 = v47;
  v142 = v48;
  v143 = v49;
  LODWORD(v138) = (_DWORD)v106;
  v51 = sub_33E5110(v50, v125, v126, 1, 0);
  v55 = sub_33E8420((_DWORD)v110, v51, v52, v125, v126, (unsigned int)&v123, (__int64)v107, v108, v38, (__int16)v109, 0);
  v56 = *(unsigned int *)(a1 + 136);
  v58 = v57;
  if ( v56 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
  {
    sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v56 + 1, 0x10u, v53, v54);
    v56 = *(unsigned int *)(a1 + 136);
  }
  v59 = (__int64 *)(*(_QWORD *)(a1 + 128) + 16 * v56);
  *v59 = v55;
  v59[1] = 1;
  ++*(_DWORD *)(a1 + 136);
  v122 = a2;
  v60 = sub_337DC20(a1 + 8, &v122);
  v112 = v58;
  v111 = v55;
  *v60 = v55;
  v61 = v123;
  *((_DWORD *)v60 + 2) = v112;
  if ( v61 )
    sub_B91220((__int64)&v123, v61);
}
