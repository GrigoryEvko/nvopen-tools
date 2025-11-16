// Function: sub_339D900
// Address: 0x339d900
//
void __fastcall sub_339D900(__int64 a1, __int64 a2)
{
  int v2; // r15d
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // r13d
  __int64 v13; // r14
  __int64 v14; // rsi
  __m128i v15; // rax
  int v16; // ecx
  int v17; // r14d
  __int64 v18; // rdx
  bool v19; // cc
  unsigned __int64 v20; // rax
  char v21; // r13
  __int64 v22; // r8
  int v23; // eax
  bool v24; // zf
  unsigned __int16 v25; // r13
  __int64 v26; // rax
  int *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  char v31; // r10
  __int64 v32; // rax
  int v33; // r13d
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 v36; // r8
  unsigned __int64 v37; // r15
  unsigned __int16 *v38; // rdx
  int v39; // eax
  __int64 v40; // rdx
  __int16 v41; // ax
  __int64 v42; // rdx
  __int64 (*v43)(); // rax
  __int64 v44; // rax
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __int64 v48; // rdx
  __m128i v49; // xmm3
  int *v50; // rdi
  int v51; // eax
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r8
  __int64 v56; // r14
  __int64 v57; // r15
  int *v58; // r8
  _QWORD *v59; // rax
  __int64 v60; // rsi
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  bool v64; // al
  __int64 v65; // rcx
  __int16 v66; // ax
  __int64 (__fastcall *v67)(__int64, __int64, unsigned int); // r13
  __int64 v68; // rax
  _DWORD *v69; // rax
  int v70; // r10d
  int v71; // edx
  unsigned __int16 v72; // ax
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // r10
  __int64 v76; // rdx
  __int64 *v77; // rdi
  __int64 v78; // rax
  _DWORD *v79; // rax
  int v80; // r10d
  int v81; // edx
  unsigned __int16 v82; // ax
  __int64 v83; // rax
  __int64 v84; // rdx
  int v85; // esi
  int v86; // eax
  int v87; // r9d
  int v88; // r8d
  int v89; // edx
  int v90; // ecx
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // [rsp-10h] [rbp-200h]
  __int64 v94; // [rsp-10h] [rbp-200h]
  __int64 v95; // [rsp-8h] [rbp-1F8h]
  __int16 v96; // [rsp+2h] [rbp-1EEh]
  char v97; // [rsp+8h] [rbp-1E8h]
  __int64 v98; // [rsp+8h] [rbp-1E8h]
  _QWORD *v99; // [rsp+10h] [rbp-1E0h]
  __int64 v100; // [rsp+10h] [rbp-1E0h]
  __int64 v101; // [rsp+10h] [rbp-1E0h]
  __int64 (__fastcall *v102)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-1E0h]
  __int64 v103; // [rsp+18h] [rbp-1D8h]
  __m128i v104; // [rsp+20h] [rbp-1D0h] BYREF
  __int64 *v105; // [rsp+30h] [rbp-1C0h]
  __int64 v106; // [rsp+38h] [rbp-1B8h]
  __int64 v107; // [rsp+40h] [rbp-1B0h]
  int *v108; // [rsp+48h] [rbp-1A8h]
  __int64 v109; // [rsp+50h] [rbp-1A0h]
  __int64 v110; // [rsp+58h] [rbp-198h]
  __int64 v111; // [rsp+60h] [rbp-190h]
  __int64 v112; // [rsp+68h] [rbp-188h]
  __int64 v113; // [rsp+70h] [rbp-180h]
  __int64 v114; // [rsp+78h] [rbp-178h]
  __int64 v115; // [rsp+80h] [rbp-170h]
  __int64 v116; // [rsp+88h] [rbp-168h]
  __int64 v117; // [rsp+90h] [rbp-160h]
  __int64 v118; // [rsp+98h] [rbp-158h]
  __int64 v119; // [rsp+A0h] [rbp-150h]
  __int64 v120; // [rsp+A8h] [rbp-148h]
  __int64 v121; // [rsp+B0h] [rbp-140h]
  __int64 v122; // [rsp+B8h] [rbp-138h]
  unsigned __int32 v123; // [rsp+C4h] [rbp-12Ch] BYREF
  __int64 v124; // [rsp+C8h] [rbp-128h] BYREF
  __int64 v125; // [rsp+D0h] [rbp-120h] BYREF
  int v126; // [rsp+D8h] [rbp-118h]
  int v127; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v128; // [rsp+E8h] [rbp-108h]
  __m128i v129; // [rsp+F0h] [rbp-100h] BYREF
  __m128i v130; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v131; // [rsp+110h] [rbp-E0h] BYREF
  unsigned int v132; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v133; // [rsp+128h] [rbp-C8h]
  __int64 v134; // [rsp+130h] [rbp-C0h]
  __int64 v135; // [rsp+138h] [rbp-B8h]
  __int64 v136; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v137; // [rsp+148h] [rbp-A8h]
  __int64 v138; // [rsp+150h] [rbp-A0h]
  __int64 v139; // [rsp+160h] [rbp-90h] BYREF
  __int64 v140; // [rsp+168h] [rbp-88h]
  __int64 v141; // [rsp+170h] [rbp-80h]
  __int64 v142; // [rsp+178h] [rbp-78h]
  __m128i v143; // [rsp+180h] [rbp-70h]
  __m128i v144; // [rsp+190h] [rbp-60h]
  __m128i v145; // [rsp+1A0h] [rbp-50h]
  __m128i v146; // [rsp+1B0h] [rbp-40h]

  v5 = *(_DWORD *)(a1 + 848);
  v6 = *(_QWORD *)a1;
  v125 = 0;
  v126 = v5;
  if ( v6 )
  {
    if ( &v125 != (__int64 *)(v6 + 48) )
    {
      v7 = *(_QWORD *)(v6 + 48);
      v125 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v125, v7, 1);
    }
  }
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v9 = *(_QWORD *)(a2 - 32 * v8);
  v105 = *(__int64 **)(a2 + 32 * (1 - v8));
  v10 = sub_338B750(a1, v9);
  v103 = v11;
  v12 = v11;
  LODWORD(v11) = *(_DWORD *)(a2 + 4);
  v13 = v10;
  v107 = v10;
  v14 = *(_QWORD *)(a2 + 32 * (3 - (v11 & 0x7FFFFFF)));
  v15.m128i_i64[0] = sub_338B750(a1, v14);
  v16 = *(_DWORD *)(a2 + 4);
  v104 = v15;
  v15.m128i_i64[0] = *(_QWORD *)(v13 + 48) + 16LL * v12;
  v17 = *(unsigned __int16 *)v15.m128i_i64[0];
  v18 = *(_QWORD *)(v15.m128i_i64[0] + 8);
  LOWORD(v127) = *(_WORD *)v15.m128i_i64[0];
  v15.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (v16 & 0x7FFFFFF)));
  v128 = v18;
  v19 = *(_DWORD *)(v15.m128i_i64[0] + 32) <= 0x40u;
  v20 = *(_QWORD *)(v15.m128i_i64[0] + 24);
  if ( !v19 )
    v20 = *(_QWORD *)v20;
  v21 = 0;
  if ( v20 )
  {
    _BitScanReverse64(&v20, v20);
    v21 = 1;
    v2 = 63 - (v20 ^ 0x3F);
  }
  v22 = *(_QWORD *)(a1 + 864);
  if ( (_WORD)v17 )
  {
    if ( (unsigned __int16)(v17 - 17) <= 0xD3u )
    {
      v18 = 0;
      LOWORD(v17) = word_4456580[v17 - 1];
    }
  }
  else
  {
    v98 = v18;
    v100 = *(_QWORD *)(a1 + 864);
    v108 = &v127;
    v64 = sub_30070B0((__int64)&v127);
    v22 = v100;
    v18 = v98;
    if ( v64 )
    {
      v108 = (int *)v100;
      v66 = sub_3009970((__int64)&v127, v14, v98, v65, v100);
      v22 = v100;
      LOWORD(v17) = v66;
    }
  }
  v23 = sub_33CC4A0(v22, (unsigned __int16)v17, v18);
  v24 = v21 == 0;
  v25 = v127;
  v129.m128i_i64[0] = 0;
  v129.m128i_i32[2] = 0;
  if ( v24 )
    v2 = v23;
  v26 = *(_QWORD *)(a1 + 864);
  v130.m128i_i64[0] = 0;
  v130.m128i_i32[2] = 0;
  v27 = *(int **)(v26 + 16);
  v131.m128i_i64[0] = 0;
  v108 = v27;
  v131.m128i_i32[2] = 0;
  if ( (_WORD)v127 )
  {
    if ( (unsigned __int16)(v127 - 17) > 0xD3u )
    {
LABEL_16:
      v28 = v128;
      goto LABEL_17;
    }
    v28 = 0;
    v25 = word_4456580[(unsigned __int16)v127 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v127) )
      goto LABEL_16;
    v25 = sub_3009970((__int64)&v127, (unsigned __int16)v17, v61, v62, v63);
  }
LABEL_17:
  LOWORD(v139) = v25;
  v140 = v28;
  if ( v25 )
  {
    if ( v25 == 1 || (unsigned __int16)(v25 - 504) <= 7u )
      BUG();
    v29 = *(_QWORD *)&byte_444C4A0[16 * v25 - 16];
  }
  else
  {
    v29 = sub_3007260((__int64)&v139);
    v134 = v29;
    v135 = v30;
  }
  v31 = sub_339D300(
          (__int64)v105,
          (__int64)&v129,
          (__int64)&v130,
          &v123,
          (__int64)&v131,
          a1,
          *(_QWORD *)(a2 + 40),
          (unsigned __int64)(v29 + 7) >> 3);
  v32 = v105[1];
  if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
  {
    v32 = **(_QWORD **)(v32 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
      v32 = **(_QWORD **)(v32 + 16);
  }
  v97 = v31;
  v33 = *(_DWORD *)(v32 + 8) >> 8;
  v99 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  sub_B91FC0(&v139, a2);
  v34 = 2;
  LODWORD(v138) = v33;
  v137 = 0;
  BYTE4(v138) = 0;
  v136 = 0;
  v37 = sub_2E7BD70(v99, 2u, -1, v2, (int)&v139, 0, 0, v138, 1u, 0, 0);
  if ( !v97 )
  {
    v101 = *(_QWORD *)(a1 + 864);
    v67 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v108 + 32LL);
    v68 = sub_2E79000(*(__int64 **)(v101 + 40));
    if ( v67 == sub_2D42F30 )
    {
      v69 = sub_AE2980(v68, 0);
      v70 = v101;
      v71 = v69[1];
      v72 = 2;
      if ( v71 != 1 )
      {
        v72 = 3;
        if ( v71 != 2 )
        {
          v72 = 4;
          if ( v71 != 4 )
          {
            v72 = 5;
            if ( v71 != 8 )
            {
              v72 = 6;
              if ( v71 != 16 )
              {
                v72 = 7;
                if ( v71 != 32 )
                {
                  v72 = 8;
                  if ( v71 != 64 )
                    v72 = 9 * (v71 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v72 = v67((__int64)v108, v68, 0);
      v70 = v101;
    }
    v121 = sub_3400BD0(v70, 0, (unsigned int)&v125, v72, 0, 0, 0);
    v122 = v73;
    v129.m128i_i64[0] = v121;
    v129.m128i_i32[2] = v73;
    v74 = sub_338B750(a1, (__int64)v105);
    v75 = *(_QWORD *)(a1 + 864);
    v123 = 0;
    v119 = v74;
    v120 = v76;
    v77 = *(__int64 **)(v75 + 40);
    v130.m128i_i64[0] = v74;
    v105 = (__int64 *)v75;
    v130.m128i_i32[2] = v76;
    v102 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v108 + 32LL);
    v78 = sub_2E79000(v77);
    if ( v102 == sub_2D42F30 )
    {
      v79 = sub_AE2980(v78, 0);
      v80 = (int)v105;
      v81 = v79[1];
      v82 = 2;
      if ( v81 != 1 )
      {
        v82 = 3;
        if ( v81 != 2 )
        {
          v82 = 4;
          if ( v81 != 4 )
          {
            v82 = 5;
            if ( v81 != 8 )
            {
              v82 = 6;
              if ( v81 != 16 )
              {
                v82 = 7;
                if ( v81 != 32 )
                {
                  v82 = 8;
                  if ( v81 != 64 )
                    v82 = 9 * (v81 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v82 = ((__int64 (__fastcall *)(int *, __int64, _QWORD, __int64 (__fastcall *)(__int64, __int64, unsigned int), __int64, __int64))v102)(
              v108,
              v78,
              0,
              v102,
              v93,
              v95);
      v80 = (int)v105;
    }
    v83 = sub_3400BD0(v80, 1, (unsigned int)&v125, v82, 0, 1, 0);
    v35 = v94;
    v34 = v95;
    v117 = v83;
    v118 = v84;
    v131.m128i_i64[0] = v83;
    v131.m128i_i32[2] = v84;
  }
  v38 = (unsigned __int16 *)(*(_QWORD *)(v130.m128i_i64[0] + 48) + 16LL * v130.m128i_u32[2]);
  v39 = *v38;
  v40 = *((_QWORD *)v38 + 1);
  LOWORD(v132) = v39;
  v133 = v40;
  if ( (_WORD)v39 )
  {
    v41 = word_4456580[v39 - 1];
    v42 = 0;
  }
  else
  {
    v41 = sub_3009970((__int64)&v132, v34, v40, v35, v36);
  }
  LOWORD(v136) = v41;
  v137 = v42;
  v43 = *(__int64 (**)())(*(_QWORD *)v108 + 696LL);
  if ( v43 != sub_2FE3240
    && ((unsigned __int8 (__fastcall *)(int *, _QWORD, __int64, __int64 *))v43)(v108, v132, v133, &v136) )
  {
    if ( (_WORD)v132 )
    {
      v85 = word_4456340[(unsigned __int16)v132 - 1];
      if ( (unsigned __int16)(v132 - 176) > 0x34u )
        LOWORD(v86) = sub_2D43050(v136, v85);
      else
        LOWORD(v86) = sub_2D43AD0(v136, v85);
      v88 = 0;
    }
    else
    {
      v86 = sub_3009490((unsigned __int16 *)&v132, v136, v137);
      v96 = HIWORD(v86);
      v88 = v89;
    }
    HIWORD(v90) = v96;
    LOWORD(v90) = v86;
    v91 = sub_33FAF80(*(_QWORD *)(a1 + 864), 213, (unsigned int)&v125, v90, v88, v87);
    v116 = v92;
    v115 = v91;
    v130.m128i_i64[0] = v91;
    v130.m128i_i32[2] = v92;
  }
  v44 = sub_33738A0(a1);
  v45 = _mm_load_si128(&v104);
  v46 = _mm_loadu_si128(&v129);
  v139 = v44;
  v47 = _mm_loadu_si128(&v130);
  v140 = v48;
  v49 = _mm_loadu_si128(&v131);
  v50 = *(int **)(a1 + 864);
  v141 = v107;
  v104.m128i_i32[0] = v123;
  v105 = &v139;
  v106 = 6;
  v108 = v50;
  v143 = v45;
  v144 = v46;
  v145 = v47;
  v146 = v49;
  v142 = v103;
  v51 = sub_33ED250(v50, 1, 0, v123);
  v53 = sub_33E7ED0(
          (_DWORD)v108,
          v51,
          v52,
          v127,
          v128,
          (unsigned int)&v125,
          (__int64)v105,
          v106,
          v37,
          v104.m128i_i16[0],
          0);
  v55 = *(_QWORD *)(a1 + 864);
  v56 = v53;
  v57 = v54;
  if ( v53 )
  {
    v108 = *(int **)(a1 + 864);
    nullsub_1875(v53, v108, 0);
    v58 = v108;
    v114 = v57;
    v113 = v56;
    *((_QWORD *)v108 + 48) = v56;
    v58[98] = v114;
    sub_33E2B60(v58, 0);
  }
  else
  {
    v110 = v54;
    v109 = 0;
    *(_QWORD *)(v55 + 384) = 0;
    *(_DWORD *)(v55 + 392) = v110;
  }
  v124 = a2;
  v59 = sub_337DC20(a1 + 8, &v124);
  v112 = v57;
  v111 = v56;
  *v59 = v56;
  v60 = v125;
  *((_DWORD *)v59 + 2) = v112;
  if ( v60 )
    sub_B91220((__int64)&v125, v60);
}
