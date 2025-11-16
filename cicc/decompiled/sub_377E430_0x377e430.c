// Function: sub_377E430
// Address: 0x377e430
//
void __fastcall sub_377E430(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int16 *v7; // rax
  __int16 v8; // dx
  __m128i v9; // xmm0
  unsigned __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rdi
  __m128i *v13; // r10
  char v14; // cl
  __int128 *v15; // r13
  __int64 v16; // rsi
  __int128 v17; // xmm7
  __int64 v18; // rbx
  __m128i v19; // xmm1
  __int32 v20; // eax
  __int64 v21; // rax
  __int16 v22; // dx
  __m128i v23; // xmm2
  __int64 v24; // rbx
  __int64 v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 v28; // r14
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned int v32; // ecx
  __int64 v33; // r8
  int v34; // edx
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rsi
  int v38; // edx
  unsigned __int64 v39; // rdx
  __int64 v40; // r14
  unsigned int v41; // r13d
  __int64 v42; // rsi
  __int64 v43; // rdx
  char v44; // al
  _QWORD *v45; // r13
  __int64 v46; // rdx
  __int64 v47; // r14
  int v48; // r9d
  int v49; // r9d
  __int128 v50; // rax
  unsigned int v51; // edx
  _QWORD *v52; // r14
  __int64 v53; // r8
  unsigned __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rax
  unsigned __int8 v58; // al
  __m128i *v59; // rax
  _QWORD *v60; // r14
  unsigned __int64 v61; // r13
  unsigned int v62; // edx
  unsigned __int64 v63; // rbx
  __m128i v64; // xmm4
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 v67; // rcx
  unsigned __int8 v68; // al
  __m128i *v69; // rax
  __int64 *v70; // rdi
  unsigned int v71; // edx
  __int128 v72; // rax
  __int64 v73; // r9
  int v74; // edx
  _QWORD *v75; // r13
  __int64 v76; // rax
  __int16 v77; // dx
  __int64 v78; // rax
  __m128i v79; // xmm6
  __int64 v80; // rsi
  __int64 *v81; // r13
  __int64 v82; // r12
  unsigned __int16 v83; // cx
  bool v84; // di
  unsigned int v85; // esi
  unsigned int v86; // r14d
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  unsigned int v90; // eax
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int128 v94; // rax
  _QWORD *v95; // rsi
  __int64 v96; // rsi
  __int64 v97; // rdx
  unsigned __int64 v98; // rax
  __int128 v99; // [rsp-20h] [rbp-260h]
  __int128 v100; // [rsp-20h] [rbp-260h]
  __int64 v101; // [rsp+28h] [rbp-218h]
  __int128 v102; // [rsp+50h] [rbp-1F0h]
  __int64 *v103; // [rsp+50h] [rbp-1F0h]
  unsigned __int64 v104; // [rsp+50h] [rbp-1F0h]
  unsigned __int64 v105; // [rsp+58h] [rbp-1E8h]
  __int64 v107; // [rsp+60h] [rbp-1E0h]
  __int64 v108; // [rsp+60h] [rbp-1E0h]
  unsigned __int64 v109; // [rsp+68h] [rbp-1D8h]
  unsigned __int64 v112; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v113; // [rsp+88h] [rbp-1B8h]
  __m128i *v114; // [rsp+90h] [rbp-1B0h]
  __int64 *v115; // [rsp+90h] [rbp-1B0h]
  unsigned __int8 *v116; // [rsp+90h] [rbp-1B0h]
  unsigned __int16 v117; // [rsp+90h] [rbp-1B0h]
  __int64 v118; // [rsp+98h] [rbp-1A8h]
  unsigned __int64 v119; // [rsp+98h] [rbp-1A8h]
  int v120; // [rsp+C8h] [rbp-178h]
  __int64 v121; // [rsp+F0h] [rbp-150h] BYREF
  int v122; // [rsp+F8h] [rbp-148h]
  __int64 v123; // [rsp+100h] [rbp-140h] BYREF
  unsigned __int64 v124; // [rsp+108h] [rbp-138h]
  __m128i v125; // [rsp+110h] [rbp-130h] BYREF
  __int128 v126; // [rsp+120h] [rbp-120h] BYREF
  unsigned __int64 v127; // [rsp+130h] [rbp-110h]
  __int64 v128; // [rsp+138h] [rbp-108h]
  __int64 v129; // [rsp+140h] [rbp-100h]
  __int64 v130; // [rsp+148h] [rbp-F8h]
  __m128i v131; // [rsp+150h] [rbp-F0h] BYREF
  __m128i v132; // [rsp+160h] [rbp-E0h] BYREF
  __m128i v133; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v134; // [rsp+180h] [rbp-C0h]
  __m128i v135; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v136; // [rsp+1A0h] [rbp-A0h]
  __m128i v137; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v138; // [rsp+1C0h] [rbp-80h]
  __m128i v139; // [rsp+1D0h] [rbp-70h] BYREF
  unsigned int v140; // [rsp+1E0h] [rbp-60h] BYREF
  __int64 v141; // [rsp+1E8h] [rbp-58h]
  __m128i v142; // [rsp+1F0h] [rbp-50h] BYREF
  __m128i v143; // [rsp+200h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v121 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v121, v5, 1);
  v6 = a1[1];
  v122 = *(_DWORD *)(a2 + 72);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v124 = *((_QWORD *)v7 + 1);
  LOWORD(v123) = v8;
  sub_33D0340((__int64)&v139, v6, &v123);
  v9 = _mm_loadu_si128(&v139);
  LOWORD(v10) = v139.m128i_i16[0];
  v125 = v9;
  if ( !v139.m128i_i16[0] )
    goto LABEL_27;
LABEL_4:
  v11 = (unsigned __int16)v10;
  v12 = *a1;
  v13 = (__m128i *)a1[1];
  if ( word_4456340[(unsigned __int16)v10 - 1] <= 1u )
  {
LABEL_28:
    *(_QWORD *)&v94 = sub_346E5E0(v12, a2, v13, v9);
    v95 = (_QWORD *)a1[1];
    v137 = (__m128i)v94;
    sub_3408290((__int64)&v142, v95, (__int128 *)v137.m128i_i8, (__int64)&v121, (unsigned int *)&v139, &v140, v9);
    v96 = v121;
    *a3 = v142.m128i_i64[0];
    *((_DWORD *)a3 + 2) = v142.m128i_i32[2];
    *(_QWORD *)a4 = v143.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v143.m128i_i32[2];
    if ( v96 )
      sub_B91220((__int64)&v121, v96);
  }
  else
  {
    if ( (_WORD)v10 != 1
      && (v14 = *(_BYTE *)(v12 + 500LL * (unsigned __int16)v10 + 6585),
          !*(_QWORD *)(v12 + 8LL * (unsigned __int16)v10 + 112))
      || (v14 = *(_BYTE *)(v12 + 500LL * (unsigned __int16)v10 + 6585)) != 0 )
    {
      if ( v14 != 4 )
      {
        v81 = (__int64 *)v13[4].m128i_i64[0];
        v82 = 0;
        v83 = word_4456580[(unsigned __int16)v10 - 1];
        while ( 1 )
        {
          v84 = (unsigned __int16)(v10 - 176) <= 0x34u;
          LOBYTE(v10) = v84;
          v85 = word_4456340[v11 - 1];
          while ( 1 )
          {
            v6 = v85 >> 1;
            v142.m128i_i8[4] = v10;
            v86 = v83;
            v142.m128i_i32[0] = v6;
            if ( v84 )
            {
              LOWORD(v10) = sub_2D43AD0(v83, v6);
              v89 = 0;
              if ( (_WORD)v10 )
                goto LABEL_26;
            }
            else
            {
              LOWORD(v10) = sub_2D43050(v83, v6);
              v89 = 0;
              if ( (_WORD)v10 )
                goto LABEL_26;
            }
            v6 = v86;
            LOWORD(v10) = sub_3009450(v81, v86, v82, v142.m128i_i64[0], v87, v88);
LABEL_26:
            v125.m128i_i16[0] = v10;
            v125.m128i_i64[1] = v89;
            if ( (_WORD)v10 )
              goto LABEL_4;
LABEL_27:
            v90 = sub_3007240((__int64)&v125);
            v12 = *a1;
            v13 = (__m128i *)a1[1];
            if ( v90 <= 1 )
              goto LABEL_28;
            v81 = (__int64 *)v13[4].m128i_i64[0];
            v83 = sub_3009970((__int64)&v125, v6, v91, v92, v93);
            LOWORD(v10) = v125.m128i_i16[0];
            v82 = v97;
            if ( v125.m128i_i16[0] )
              break;
            v117 = v83;
            v98 = sub_3007240((__int64)&v125);
            v83 = v117;
            v85 = v98;
            v10 = HIDWORD(v98);
            v84 = v10;
          }
          v11 = v125.m128i_u16[0];
        }
      }
    }
    v15 = *(__int128 **)(a2 + 40);
    v16 = *(_QWORD *)(a2 + 80);
    v17 = (__int128)_mm_loadu_si128((const __m128i *)v15 + 5);
    v18 = *((_QWORD *)v15 + 10);
    v19 = _mm_loadu_si128((const __m128i *)((char *)v15 + 40));
    v132.m128i_i64[0] = v16;
    v101 = v18;
    if ( v16 )
    {
      v114 = v13;
      sub_B96E90((__int64)&v132, v16, 1);
      v13 = v114;
      v15 = *(__int128 **)(a2 + 40);
    }
    v133.m128i_i16[0] = 0;
    v20 = *(_DWORD *)(a2 + 72);
    v135.m128i_i16[0] = 0;
    v133.m128i_i64[1] = 0;
    v132.m128i_i32[2] = v20;
    v135.m128i_i64[1] = 0;
    v115 = (__int64 *)v13;
    v21 = *(_QWORD *)(*(_QWORD *)v15 + 48LL) + 16LL * *((unsigned int *)v15 + 2);
    v22 = *(_WORD *)v21;
    v137.m128i_i64[1] = *(_QWORD *)(v21 + 8);
    v137.m128i_i16[0] = v22;
    sub_33D0340((__int64)&v142, (__int64)v13, v137.m128i_i64);
    v23 = _mm_loadu_si128(&v143);
    v133 = _mm_loadu_si128(&v142);
    v135 = v23;
    sub_3408290((__int64)&v142, v115, v15, (__int64)&v132, (unsigned int *)&v133, (unsigned int *)&v135, v9);
    if ( v132.m128i_i64[0] )
      sub_B91220((__int64)&v132, v132.m128i_i64[0]);
    *a3 = v142.m128i_i64[0];
    *((_DWORD *)a3 + 2) = v142.m128i_i32[2];
    *(_QWORD *)a4 = v143.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v143.m128i_i32[2];
    sub_3777990(&v142, a1, v19.m128i_u64[0], v19.m128i_u64[1], v9);
    v24 = v142.m128i_u32[2];
    v25 = v142.m128i_i64[0];
    v26 = (_QWORD *)a1[1];
    v142.m128i_i64[0] = 0;
    v27 = v143.m128i_i64[0];
    v142.m128i_i32[2] = 0;
    v107 = v25;
    v28 = v143.m128i_u32[2];
    *(_QWORD *)&v102 = sub_33F17F0(v26, 51, (__int64)&v142, v139.m128i_u32[0], v139.m128i_i64[1]);
    *((_QWORD *)&v102 + 1) = v30;
    if ( v142.m128i_i64[0] )
      sub_B91220((__int64)&v142, v142.m128i_i64[0]);
    *((_QWORD *)&v99 + 1) = v24;
    *(_QWORD *)&v99 = v107;
    v31 = sub_340F900(
            (_QWORD *)a1[1],
            0xABu,
            (__int64)&v121,
            v139.m128i_u32[0],
            v139.m128i_i64[1],
            v29,
            *(_OWORD *)a3,
            v99,
            v102);
    v32 = v140;
    v33 = v141;
    *a3 = v31;
    *((_DWORD *)a3 + 2) = v34;
    *((_QWORD *)&v100 + 1) = v28;
    *(_QWORD *)&v100 = v27;
    v36 = sub_340F900((_QWORD *)a1[1], 0xABu, (__int64)&v121, v32, v33, v35, *(_OWORD *)a4, v100, v102);
    v37 = (unsigned int)v123;
    v120 = v38;
    v39 = v124;
    *(_QWORD *)a4 = v36;
    *(_DWORD *)(a4 + 8) = v120;
    v40 = a1[1];
    v41 = sub_33CD850(v40, v37, v39, 0);
    if ( (_WORD)v123 )
    {
      if ( (_WORD)v123 == 1 || (unsigned __int16)(v123 - 504) <= 7u )
        BUG();
      v42 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v123 - 16];
      v44 = byte_444C4A0[16 * (unsigned __int16)v123 - 8];
    }
    else
    {
      v129 = sub_3007260((__int64)&v123);
      v42 = v129;
      v130 = v43;
      v44 = v43;
    }
    LOBYTE(v128) = v44;
    v127 = (unsigned __int64)(v42 + 7) >> 3;
    v45 = sub_33EDE90(v40, v127, v128, v41);
    v47 = v46;
    v103 = *(__int64 **)(a1[1] + 40);
    sub_2EAC300((__int64)&v133, (__int64)v103, *((_DWORD *)v45 + 24), 0);
    sub_33FAF80(
      a1[1],
      214,
      (__int64)&v121,
      *(unsigned __int16 *)(*(_QWORD *)(v107 + 48) + 16 * v24),
      *(_QWORD *)(*(_QWORD *)(v107 + 48) + 16 * v24 + 8),
      v48,
      v9);
    *(_QWORD *)&v50 = sub_33FAF80(a1[1], 382, (__int64)&v121, 7, 0, v49, v9);
    v118 = *((_QWORD *)&v50 + 1);
    v108 = (__int64)v45;
    v109 = v47;
    v116 = sub_3466750(*a1, (_QWORD *)a1[1], (__int64)v45, v47, (unsigned int)v123, v124, v9, v50);
    v52 = (_QWORD *)a1[1];
    v137 = _mm_loadu_si128(&v133);
    v53 = *a3;
    v54 = v51 | v118 & 0xFFFFFFFF00000000LL;
    v55 = *a3;
    v142 = 0u;
    v119 = v54;
    v56 = a3[1];
    v138 = v134;
    v57 = *((unsigned int *)a3 + 2);
    v143 = 0u;
    v112 = v53;
    v113 = v56;
    v58 = sub_33CC4A0(
            (__int64)v52,
            *(unsigned __int16 *)(*(_QWORD *)(v55 + 48) + 16 * v57),
            *(_QWORD *)(*(_QWORD *)(v55 + 48) + 16 * v57 + 8),
            0xFFFFFFFF00000000LL,
            v53,
            v56);
    v59 = sub_33F4560(
            v52,
            (unsigned __int64)(v52 + 36),
            0,
            (__int64)&v121,
            v112,
            v113,
            (unsigned __int64)v45,
            v109,
            *(_OWORD *)&v137,
            v138,
            v58,
            0,
            (__int64)&v142);
    v60 = (_QWORD *)a1[1];
    v61 = (unsigned __int64)v59;
    v142 = 0u;
    v63 = v62;
    v143 = 0u;
    sub_2EAC3A0((__int64)&v135, v103);
    v64 = _mm_loadu_si128(&v135);
    v65 = *(_QWORD *)a4;
    v138 = v136;
    v66 = *(unsigned int *)(a4 + 8);
    v137 = v64;
    v104 = v65;
    v105 = *(_QWORD *)(a4 + 8);
    v68 = sub_33CC4A0(
            (__int64)v60,
            *(unsigned __int16 *)(*(_QWORD *)(v65 + 48) + 16 * v66),
            *(_QWORD *)(*(_QWORD *)(v65 + 48) + 16 * v66 + 8),
            v67,
            v65,
            v105);
    v69 = sub_33F4560(
            v60,
            v61,
            v63,
            (__int64)&v121,
            v104,
            v105,
            (unsigned __int64)v116,
            v119,
            *(_OWORD *)&v137,
            v138,
            v68,
            0,
            (__int64)&v142);
    v70 = (__int64 *)a1[1];
    v142 = 0u;
    v143 = 0u;
    *(_QWORD *)&v72 = sub_33F1F00(
                        v70,
                        (unsigned int)v123,
                        v124,
                        (__int64)&v121,
                        (__int64)v69,
                        v71,
                        v108,
                        v109,
                        *(_OWORD *)&v133,
                        v134,
                        0,
                        0,
                        (__int64)&v142,
                        0);
    v126 = v72;
    if ( *(_DWORD *)(v101 + 24) != 51 )
    {
      *(_QWORD *)&v126 = sub_340F900((_QWORD *)a1[1], 0xCEu, (__int64)&v121, v123, v124, v73, *(_OWORD *)&v19, v72, v17);
      DWORD2(v126) = v74;
    }
    v75 = (_QWORD *)a1[1];
    v132.m128i_i16[0] = 0;
    v131.m128i_i16[0] = 0;
    v132.m128i_i64[1] = 0;
    v131.m128i_i64[1] = 0;
    v76 = *(_QWORD *)(v126 + 48) + 16LL * DWORD2(v126);
    v77 = *(_WORD *)v76;
    v78 = *(_QWORD *)(v76 + 8);
    v137.m128i_i16[0] = v77;
    v137.m128i_i64[1] = v78;
    sub_33D0340((__int64)&v142, (__int64)v75, v137.m128i_i64);
    v79 = _mm_loadu_si128(&v143);
    v131 = _mm_loadu_si128(&v142);
    v132 = v79;
    sub_3408290((__int64)&v142, v75, &v126, (__int64)&v121, (unsigned int *)&v131, (unsigned int *)&v132, v9);
    v80 = v121;
    *a3 = v142.m128i_i64[0];
    *((_DWORD *)a3 + 2) = v142.m128i_i32[2];
    *(_QWORD *)a4 = v143.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v143.m128i_i32[2];
    if ( v80 )
      sub_B91220((__int64)&v121, v80);
  }
}
