// Function: sub_3824710
// Address: 0x3824710
//
void __fastcall sub_3824710(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 v7; // rax
  __int128 v8; // xmm0
  __m128i v9; // xmm1
  unsigned __int8 *v10; // r15
  __int64 v11; // rdx
  __int16 v12; // cx
  __int64 v13; // rdx
  unsigned __int16 *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  __int32 v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // r8
  int v24; // r9d
  unsigned int v25; // r14d
  unsigned int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rdi
  unsigned int v31; // esi
  __int16 v32; // ax
  unsigned __int64 v33; // rdx
  __int64 v34; // rdx
  unsigned __int8 v35; // al
  __int64 v36; // r14
  __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned __int16 *v39; // rax
  unsigned int v40; // esi
  __int64 v41; // rax
  unsigned __int64 v42; // r14
  int v43; // r9d
  int v44; // eax
  unsigned int v45; // edx
  unsigned __int8 *v46; // r8
  _QWORD *v47; // rdi
  unsigned int v48; // edx
  __int128 v49; // rax
  unsigned __int8 *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  __int64 v53; // rsi
  unsigned __int8 *v54; // r14
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned __int8 *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r15
  unsigned __int8 *v60; // r14
  __int128 v61; // rax
  __int128 v62; // rax
  _QWORD *v63; // r14
  __int128 v64; // rax
  __int64 v65; // r9
  unsigned int v66; // edx
  int v67; // r14d
  bool v68; // al
  unsigned __int8 *v69; // rax
  unsigned int v70; // edx
  unsigned __int8 *v71; // rax
  unsigned int v72; // edx
  __int64 *v73; // r14
  __int64 v74; // rdx
  __int64 v75; // rax
  char v76; // cl
  __int64 v77; // rdx
  unsigned __int64 v78; // rax
  __int16 v79; // ax
  __int16 v80; // r15
  __m128i *v81; // r14
  unsigned __int64 v82; // rdx
  unsigned __int64 v83; // r15
  unsigned int v84; // edx
  __int128 v85; // rax
  __int64 v86; // r9
  __int128 v87; // rax
  __int64 v88; // r9
  unsigned int v89; // edx
  _QWORD *v90; // r15
  unsigned __int8 *v91; // rax
  __int64 v92; // rdx
  unsigned int v93; // edx
  unsigned int v94; // edx
  __int128 v95; // rax
  __int64 v96; // r9
  __int128 v97; // [rsp-20h] [rbp-280h]
  __int128 v98; // [rsp-20h] [rbp-280h]
  __int128 v99; // [rsp-10h] [rbp-270h]
  __int64 v100; // [rsp+18h] [rbp-248h]
  _QWORD *v101; // [rsp+28h] [rbp-238h]
  __m128i *v102; // [rsp+30h] [rbp-230h]
  __int64 v103; // [rsp+40h] [rbp-220h]
  _QWORD *v106; // [rsp+60h] [rbp-200h]
  unsigned __int64 v107; // [rsp+68h] [rbp-1F8h]
  unsigned int v108; // [rsp+70h] [rbp-1F0h]
  __int64 v109; // [rsp+78h] [rbp-1E8h]
  unsigned int v110; // [rsp+80h] [rbp-1E0h]
  unsigned int v111; // [rsp+88h] [rbp-1D8h]
  unsigned __int8 v112; // [rsp+8Fh] [rbp-1D1h]
  unsigned int v113; // [rsp+90h] [rbp-1D0h]
  unsigned int v114; // [rsp+94h] [rbp-1CCh]
  __int128 v116; // [rsp+A0h] [rbp-1C0h]
  unsigned __int64 v117; // [rsp+A8h] [rbp-1B8h]
  unsigned int v118; // [rsp+B8h] [rbp-1A8h]
  __int64 v119; // [rsp+C0h] [rbp-1A0h]
  unsigned __int8 *v120; // [rsp+C0h] [rbp-1A0h]
  __int64 v121; // [rsp+C0h] [rbp-1A0h]
  __int64 v122; // [rsp+C8h] [rbp-198h]
  unsigned __int64 v123; // [rsp+C8h] [rbp-198h]
  __int64 v124; // [rsp+D0h] [rbp-190h]
  _QWORD *v125; // [rsp+D0h] [rbp-190h]
  _QWORD *v126; // [rsp+D0h] [rbp-190h]
  __int128 v127; // [rsp+D0h] [rbp-190h]
  unsigned __int8 *v128; // [rsp+D0h] [rbp-190h]
  _QWORD *v129; // [rsp+D0h] [rbp-190h]
  unsigned __int64 v130; // [rsp+D8h] [rbp-188h]
  __int64 v131; // [rsp+170h] [rbp-F0h] BYREF
  int v132; // [rsp+178h] [rbp-E8h]
  __m128i v133; // [rsp+180h] [rbp-E0h] BYREF
  __m128i v134; // [rsp+190h] [rbp-D0h] BYREF
  unsigned int v135; // [rsp+1A0h] [rbp-C0h] BYREF
  unsigned __int64 v136; // [rsp+1A8h] [rbp-B8h]
  unsigned __int64 v137; // [rsp+1B0h] [rbp-B0h] BYREF
  char v138; // [rsp+1B8h] [rbp-A8h]
  unsigned __int64 v139; // [rsp+1C0h] [rbp-A0h]
  __int64 v140; // [rsp+1C8h] [rbp-98h]
  __int128 v141; // [rsp+1D0h] [rbp-90h] BYREF
  __int64 v142; // [rsp+1E0h] [rbp-80h]
  __int128 v143; // [rsp+1F0h] [rbp-70h] BYREF
  __int64 v144; // [rsp+200h] [rbp-60h]
  unsigned __int64 v145; // [rsp+210h] [rbp-50h] BYREF
  __int64 v146; // [rsp+218h] [rbp-48h]
  unsigned __int64 v147; // [rsp+220h] [rbp-40h]
  __int64 v148; // [rsp+228h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v131 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v131, v5, 1);
  v6 = *a1;
  v132 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = (__int128)_mm_loadu_si128((const __m128i *)v7);
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v10 = *(unsigned __int8 **)(v7 + 40);
  v11 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * *(unsigned int *)(v7 + 8);
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v119 = *(unsigned int *)(v7 + 48);
  v14 = (unsigned __int16 *)(*((_QWORD *)v10 + 6) + 16 * v119);
  v117 = v9.m128i_u64[1];
  v133.m128i_i16[0] = v12;
  v133.m128i_i64[1] = v13;
  v15 = *((_QWORD *)v14 + 1);
  v118 = *v14;
  v134 = _mm_loadu_si128(&v133);
  do
  {
    v16 = a1[1];
    v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
    if ( v17 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v145, v6, *(_QWORD *)(v16 + 64), v134.m128i_i64[0], v134.m128i_i64[1]);
      LOWORD(v18) = v146;
      v134.m128i_i16[0] = v146;
      v134.m128i_i64[1] = v147;
    }
    else
    {
      v18 = v17(v6, *(_QWORD *)(v16 + 64), v134.m128i_u32[0], v134.m128i_i64[1]);
      v134.m128i_i32[0] = v18;
      v134.m128i_i64[1] = v34;
    }
    v6 = *a1;
  }
  while ( !(_WORD)v18 || !*(_QWORD *)(v6 + 8LL * (unsigned __int16)v18 + 112) );
  v19 = sub_2D5B750((unsigned __int16 *)&v134);
  v146 = v20;
  v145 = (v19 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v21 = sub_CA1930(&v145);
  v22 = a1[1];
  v110 = v21;
  sub_33DD090((__int64)&v145, v22, v9.m128i_i64[0], v9.m128i_i64[1], 0);
  v25 = v146;
  if ( (unsigned int)v146 <= 0x40 )
  {
    _RDX = ~v145;
    __asm { tzcnt   rax, rdx }
    if ( v145 == -1 )
      LODWORD(_RAX) = 64;
    v114 = _RAX;
    v28 = v110;
    if ( v110 )
      goto LABEL_12;
LABEL_28:
    v113 = -1;
    v124 = 0xFFFFFFFFLL;
    goto LABEL_13;
  }
  v114 = sub_C445E0((__int64)&v145);
  v28 = v110;
  if ( !v110 )
    goto LABEL_28;
LABEL_12:
  _BitScanReverse(&v28, v28);
  v113 = 31 - (v28 ^ 0x1F);
  v124 = (int)v113;
LABEL_13:
  if ( (unsigned int)v148 > 0x40 && v147 )
  {
    j_j___libc_free_0_0(v147);
    v25 = v146;
  }
  if ( v25 > 0x40 && v145 )
    j_j___libc_free_0_0(v145);
  if ( v114 < v113 )
  {
    v22 = 0xFFFFFFFF00000000LL;
    v10 = sub_33FB960(a1[1], v9.m128i_i64[0], v9.m128i_u32[2], (__m128i)v8, v113, v23, v24);
    v117 = v84 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v119 = v84;
  }
  v29 = sub_32844A0((unsigned __int16 *)&v133, v22);
  v30 = a1[1];
  v111 = v29 >> 3;
  v31 = 16 * (v29 >> 3);
  switch ( v31 )
  {
    case 0x10u:
      v32 = 6;
      break;
    case 0x20u:
      v32 = 7;
      break;
    case 0x40u:
      v32 = 8;
      break;
    case 0x80u:
      v32 = 9;
      break;
    default:
      v32 = sub_3007020(*(_QWORD **)(v30 + 64), v31);
      v30 = a1[1];
      goto LABEL_31;
  }
  v33 = 0;
LABEL_31:
  LOWORD(v135) = v32;
  v136 = v33;
  v35 = sub_33CD850(v30, v135, v33, 0);
  v36 = a1[1];
  v112 = v35;
  v145 = sub_2D5B750((unsigned __int16 *)&v135);
  v146 = v37;
  LOBYTE(v140) = v37;
  v139 = (v145 + 7) >> 3;
  v106 = sub_33EDE90(v36, v139, v140, v112);
  v100 = (unsigned int)v38;
  v39 = (unsigned __int16 *)(v106[6] + 16LL * (unsigned int)v38);
  v107 = v38;
  v109 = *((_QWORD *)v39 + 1);
  v40 = *v39;
  v41 = a1[1];
  v108 = v40;
  v42 = v41 + 288;
  sub_2EAC300((__int64)&v141, *(_QWORD *)(v41 + 40), *((_DWORD *)v106 + 24), 0);
  v44 = *(_DWORD *)(a2 + 24);
  if ( v44 == 190 )
  {
    *(_QWORD *)&v95 = sub_3400BD0(a1[1], 0, (__int64)&v131, v133.m128i_u32[0], v133.m128i_i64[1], 0, (__m128i)v8, 0);
    v46 = sub_3406EB0((_QWORD *)a1[1], 0x36u, (__int64)&v131, v135, v136, v96, v95, v8);
  }
  else
  {
    v46 = sub_33FAF80(a1[1], (unsigned int)(v44 != 191) + 213, (__int64)&v131, v135, v136, v43, (__m128i)v8);
  }
  v47 = (_QWORD *)a1[1];
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v102 = sub_33F4560(
           v47,
           v42,
           0,
           (__int64)&v131,
           (unsigned __int64)v46,
           v45,
           (unsigned __int64)v106,
           v107,
           v141,
           v142,
           v112,
           0,
           (__int64)&v145);
  v101 = (_QWORD *)a1[1];
  v103 = v48;
  *(_QWORD *)&v49 = sub_3400BD0((__int64)v101, v124, (__int64)&v131, v118, v15, 0, (__m128i)v8, 0);
  *(_QWORD *)&v116 = v10;
  *((_QWORD *)&v116 + 1) = v119 | v117 & 0xFFFFFFFF00000000LL;
  v50 = sub_3405C90(
          v101,
          0xC0u,
          (__int64)&v131,
          v118,
          v15,
          4 * (unsigned int)(v114 >= v113),
          (__m128i)v8,
          __PAIR128__(*((unsigned __int64 *)&v116 + 1), (unsigned __int64)v10),
          v49);
  v52 = v51;
  v53 = v124;
  v54 = v50;
  v125 = (_QWORD *)a1[1];
  *(_QWORD *)&v55 = sub_3400BD0((__int64)v125, v53, (__int64)&v131, v118, v15, 0, (__m128i)v8, 0);
  *((_QWORD *)&v97 + 1) = v52;
  *(_QWORD *)&v97 = v54;
  v57 = sub_3406EB0(v125, 0xBEu, (__int64)&v131, v118, v15, v56, v97, v55);
  v59 = v58;
  v60 = v57;
  v126 = (_QWORD *)a1[1];
  *(_QWORD *)&v61 = sub_3400BD0((__int64)v126, 3, (__int64)&v131, v118, v15, 0, (__m128i)v8, 0);
  *((_QWORD *)&v98 + 1) = v59;
  *(_QWORD *)&v98 = v60;
  *(_QWORD *)&v62 = sub_3405C90(v126, 0xC0u, (__int64)&v131, v118, v15, 4, (__m128i)v8, v98, v61);
  v63 = (_QWORD *)a1[1];
  v127 = v62;
  *(_QWORD *)&v64 = sub_3400BD0((__int64)v63, v111 - 1, (__int64)&v131, v118, v15, 0, (__m128i)v8, 0);
  v128 = sub_3406EB0(v63, 0xBAu, (__int64)&v131, v118, v15, v65, v127, v64);
  v130 = v66 | *((_QWORD *)&v127 + 1) & 0xFFFFFFFF00000000LL;
  v67 = *(_DWORD *)(a2 + 24);
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
    v68 = v67 == 190;
  else
    v68 = v67 != 190;
  if ( v68 )
  {
    v122 = v100;
    v120 = (unsigned __int8 *)v106;
  }
  else
  {
    v90 = (_QWORD *)a1[1];
    v91 = sub_3400BD0((__int64)v90, v111, (__int64)&v131, v108, v109, 0, (__m128i)v8, 0);
    v120 = sub_34092D0(v90, (__int64)v106, v107, (__int64)v91, v92, (__int64)&v131, (__m128i)v8, 0);
    v122 = v93;
    v128 = sub_3407430((_QWORD *)a1[1], (__int64)v128, v130, (__int64)&v131, v118, v15, (__m128i)v8);
    v130 = v94 | v130 & 0xFFFFFFFF00000000LL;
  }
  v69 = sub_33FB160(a1[1], (__int64)v128, v130, (__int64)&v131, v108, v109, (__m128i)v8);
  v71 = sub_34092D0(
          (_QWORD *)a1[1],
          (__int64)v120,
          v122,
          (__int64)v69,
          v70 | v130 & 0xFFFFFFFF00000000LL,
          (__int64)&v131,
          (__m128i)v8,
          0);
  v145 = 0;
  v121 = (__int64)v71;
  v146 = 0;
  v147 = 0;
  v123 = v72 | v122 & 0xFFFFFFFF00000000LL;
  v73 = (__int64 *)a1[1];
  v148 = 0;
  *(_QWORD *)&v143 = sub_2D5B750((unsigned __int16 *)&v134);
  *((_QWORD *)&v143 + 1) = v74;
  v137 = (unsigned __int64)(v143 + 7) >> 3;
  v138 = v74;
  v75 = sub_CA1930(&v137);
  v76 = -1;
  v77 = v75 | (1LL << v112);
  if ( (v77 & -v77) != 0 )
  {
    _BitScanReverse64(&v78, v77 & -v77);
    v76 = 63 - (v78 ^ 0x3F);
  }
  LOBYTE(v79) = v76;
  HIBYTE(v79) = 1;
  v80 = v79;
  sub_2EAC3A0((__int64)&v143, *(__int64 **)(a1[1] + 40));
  v81 = sub_33F1F00(
          v73,
          v133.m128i_u32[0],
          v133.m128i_i64[1],
          (__int64)&v131,
          (__int64)v102,
          v103,
          v121,
          v123,
          v143,
          v144,
          v80,
          0,
          (__int64)&v145,
          0);
  v83 = v82;
  if ( v114 < v113 )
  {
    v129 = (_QWORD *)a1[1];
    *(_QWORD *)&v85 = sub_3400BD0((__int64)v129, v110 - 1, (__int64)&v131, v118, v15, 0, (__m128i)v8, 0);
    *(_QWORD *)&v87 = sub_3406EB0(v129, 0xBAu, (__int64)&v131, v118, v15, v86, v116, v85);
    *((_QWORD *)&v99 + 1) = v83;
    *(_QWORD *)&v99 = v81;
    v81 = (__m128i *)sub_3406EB0(
                       (_QWORD *)a1[1],
                       *(_DWORD *)(a2 + 24),
                       (__int64)&v131,
                       v133.m128i_u32[0],
                       v133.m128i_i64[1],
                       v88,
                       v99,
                       v87);
    v83 = v89 | v83 & 0xFFFFFFFF00000000LL;
  }
  sub_375BC20(a1, (__int64)v81, v83, a3, a4, (__m128i)v8);
  if ( v131 )
    sub_B91220((__int64)&v131, v131);
}
