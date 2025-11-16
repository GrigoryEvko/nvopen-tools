// Function: sub_202B3A0
// Address: 0x202b3a0
//
__int64 __fastcall sub_202B3A0(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rsi
  char *v6; // rax
  char v7; // dl
  __int64 v8; // rax
  __m128i v9; // kr00_16
  __m128i v10; // kr10_16
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __m128i v14; // xmm0
  __int64 v15; // rdx
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // rsi
  unsigned __int8 *v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __m128i v24; // xmm5
  char v25; // dl
  __int64 v26; // rax
  __int64 v27; // rsi
  __m128i v28; // xmm4
  __m128i v29; // xmm3
  unsigned __int8 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // r8
  __int64 v34; // rdx
  __int64 *v35; // r14
  __int64 v36; // rax
  char v37; // dl
  __int64 v38; // rax
  __m128i v39; // xmm7
  __int64 v40; // rsi
  unsigned __int8 *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 *v44; // r14
  __int64 v45; // rax
  char v46; // dl
  __int64 v47; // rax
  __m128i v48; // xmm6
  __int64 v49; // rcx
  __int64 v50; // r14
  __int64 v51; // r9
  int v52; // edx
  __int64 v53; // rcx
  int v54; // r9d
  __int64 v55; // r14
  __m128i v56; // xmm6
  __m128i v57; // xmm7
  _QWORD *v58; // rdi
  __m128i v59; // xmm0
  __int64 v60; // rax
  int v61; // edx
  __int64 *v62; // rax
  __int64 v63; // r15
  __m128i v64; // xmm1
  __int64 v65; // r9
  unsigned __int64 v66; // rdx
  __int64 v67; // r14
  int v68; // edx
  int v69; // r9d
  __m128i v70; // xmm2
  __m128i v71; // xmm3
  __m128i v72; // xmm4
  _QWORD *v73; // rdi
  __int64 v74; // rax
  int v75; // edx
  __int64 *v76; // r14
  __int64 v77; // rdx
  __int64 v78; // r15
  __int64 *v79; // rax
  unsigned int v80; // edx
  const __m128i *v81; // r9
  __int64 *v82; // rax
  __m128i *v83; // rdx
  const __m128i *v84; // r9
  int v86; // eax
  int v87; // eax
  __int128 v88; // [rsp-10h] [rbp-280h]
  __int128 v89; // [rsp-10h] [rbp-280h]
  __int64 v90; // [rsp+0h] [rbp-270h]
  int v91; // [rsp+10h] [rbp-260h]
  int v92; // [rsp+28h] [rbp-248h]
  __int64 v93; // [rsp+30h] [rbp-240h]
  unsigned __int64 v94; // [rsp+38h] [rbp-238h]
  __int64 v95; // [rsp+40h] [rbp-230h]
  int v96; // [rsp+4Ch] [rbp-224h]
  unsigned int v97; // [rsp+50h] [rbp-220h]
  __int64 v98; // [rsp+50h] [rbp-220h]
  __int64 v99; // [rsp+58h] [rbp-218h]
  int v100; // [rsp+60h] [rbp-210h]
  __int64 v101; // [rsp+68h] [rbp-208h]
  __int64 v102; // [rsp+70h] [rbp-200h]
  __int64 v103; // [rsp+90h] [rbp-1E0h] BYREF
  int v104; // [rsp+98h] [rbp-1D8h]
  __m128i v105; // [rsp+A0h] [rbp-1D0h] BYREF
  __m128i v106; // [rsp+B0h] [rbp-1C0h] BYREF
  __m128i v107; // [rsp+C0h] [rbp-1B0h] BYREF
  __m128i v108; // [rsp+D0h] [rbp-1A0h] BYREF
  __m128i v109; // [rsp+E0h] [rbp-190h] BYREF
  _QWORD v110[2]; // [rsp+F0h] [rbp-180h] BYREF
  __m128i v111; // [rsp+100h] [rbp-170h] BYREF
  __m128i v112; // [rsp+110h] [rbp-160h] BYREF
  __m128i v113; // [rsp+120h] [rbp-150h] BYREF
  __m128i v114; // [rsp+130h] [rbp-140h] BYREF
  __m128i v115; // [rsp+140h] [rbp-130h] BYREF
  __m128i v116; // [rsp+150h] [rbp-120h] BYREF
  __m128i v117; // [rsp+160h] [rbp-110h] BYREF
  __m128i v118; // [rsp+170h] [rbp-100h] BYREF
  __int64 v119; // [rsp+180h] [rbp-F0h] BYREF
  __int64 v120; // [rsp+188h] [rbp-E8h]
  __m128i v121; // [rsp+190h] [rbp-E0h]
  __m128i v122; // [rsp+1A0h] [rbp-D0h]
  __int64 v123; // [rsp+1B0h] [rbp-C0h]
  int v124; // [rsp+1B8h] [rbp-B8h]
  __m128i v125; // [rsp+1C0h] [rbp-B0h]
  __int64 v126; // [rsp+1D0h] [rbp-A0h]
  int v127; // [rsp+1D8h] [rbp-98h]
  __m128i v128; // [rsp+1E0h] [rbp-90h] BYREF
  __m128i v129[2]; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 v130; // [rsp+210h] [rbp-60h]
  int v131; // [rsp+218h] [rbp-58h]
  __m128i v132; // [rsp+220h] [rbp-50h]
  __int64 v133; // [rsp+230h] [rbp-40h]
  int v134; // [rsp+238h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 72);
  v103 = v4;
  if ( v4 )
    sub_1623A60((__int64)&v103, v4, 2);
  v5 = *(_QWORD *)(a1 + 8);
  v104 = *(_DWORD *)(a2 + 64);
  v6 = *(char **)(a2 + 40);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOBYTE(v119) = v7;
  v120 = v8;
  sub_1D19A30((__int64)&v128, v5, &v119);
  v9 = v128;
  v10 = v129[0];
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(_QWORD *)v11;
  v13 = *(_QWORD *)(v11 + 120);
  v108.m128i_i64[0] = 0;
  v14 = _mm_loadu_si128((const __m128i *)(v11 + 160));
  v15 = *(_QWORD *)(v11 + 80);
  v95 = v12;
  v16 = _mm_loadu_si128((const __m128i *)(v11 + 80));
  v99 = v13;
  v17 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v100 = *(_DWORD *)(v11 + 128);
  v102 = *(_QWORD *)(v11 + 8);
  v96 = *(_DWORD *)(v11 + 208);
  v101 = *(_QWORD *)(v11 + 200);
  LOWORD(v13) = *(_WORD *)(*(_QWORD *)(a2 + 104) + 34LL);
  v105 = v14;
  v106 = v16;
  v107 = v17;
  v97 = (unsigned int)(1 << v13) >> 1;
  v108.m128i_i32[2] = 0;
  v109.m128i_i32[2] = 0;
  v18 = *(_QWORD *)a1;
  v109.m128i_i64[0] = 0;
  v19 = (unsigned __int8 *)(*(_QWORD *)(v15 + 40) + 16LL * v16.m128i_u32[2]);
  sub_1F40D10((__int64)&v128, v18, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v19, *((_QWORD *)v19 + 1));
  if ( v128.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v106.m128i_u64[0], v106.m128i_i64[1], &v108, &v109);
  }
  else
  {
    v20 = *(__int64 **)(a1 + 8);
    v117.m128i_i8[0] = 0;
    v117.m128i_i64[1] = 0;
    v118.m128i_i8[0] = 0;
    v118.m128i_i64[1] = 0;
    v21 = *(_QWORD *)(v106.m128i_i64[0] + 40) + 16LL * v106.m128i_u32[2];
    v22 = *(_BYTE *)v21;
    v23 = *(_QWORD *)(v21 + 8);
    LOBYTE(v119) = v22;
    v120 = v23;
    sub_1D19A30((__int64)&v128, (__int64)v20, &v119);
    v24 = _mm_loadu_si128(&v128);
    v118 = _mm_loadu_si128(v129);
    v117 = v24;
    sub_1D40600(
      (__int64)&v128,
      v20,
      (__int64)&v106,
      (__int64)&v103,
      (const void ***)&v117,
      (const void ***)&v118,
      v14,
      *(double *)v16.m128i_i64,
      v17);
    v108.m128i_i64[0] = v128.m128i_i64[0];
    v108.m128i_i32[2] = v128.m128i_i32[2];
    v109.m128i_i64[0] = v129[0].m128i_i64[0];
    v109.m128i_i32[2] = v129[0].m128i_i32[2];
  }
  v25 = *(_BYTE *)(a2 + 88);
  v26 = *(_QWORD *)(a2 + 96);
  v111.m128i_i8[0] = 0;
  v27 = *(_QWORD *)(a1 + 8);
  v111.m128i_i64[1] = 0;
  LOBYTE(v110[0]) = v25;
  v110[1] = v26;
  v112.m128i_i8[0] = 0;
  v112.m128i_i64[1] = 0;
  sub_1D19A30((__int64)&v128, v27, v110);
  v28 = _mm_loadu_si128(v129);
  v113.m128i_i32[2] = 0;
  v29 = _mm_loadu_si128(&v128);
  v114.m128i_i32[2] = 0;
  v30 = (unsigned __int8 *)(*(_QWORD *)(v107.m128i_i64[0] + 40) + 16LL * v107.m128i_u32[2]);
  v31 = *(_QWORD *)(a1 + 8);
  v112 = v28;
  v111 = v29;
  v32 = *(_QWORD *)a1;
  v33 = *((_QWORD *)v30 + 1);
  v113.m128i_i64[0] = 0;
  v34 = *(_QWORD *)(v31 + 48);
  v114.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v128, v32, v34, *v30, v33);
  if ( v128.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v107.m128i_u64[0], v107.m128i_i64[1], &v113, &v114);
  }
  else
  {
    v35 = *(__int64 **)(a1 + 8);
    v117.m128i_i8[0] = 0;
    v117.m128i_i64[1] = 0;
    v118.m128i_i8[0] = 0;
    v118.m128i_i64[1] = 0;
    v36 = *(_QWORD *)(v107.m128i_i64[0] + 40) + 16LL * v107.m128i_u32[2];
    v37 = *(_BYTE *)v36;
    v38 = *(_QWORD *)(v36 + 8);
    LOBYTE(v119) = v37;
    v120 = v38;
    sub_1D19A30((__int64)&v128, (__int64)v35, &v119);
    v39 = _mm_loadu_si128(&v128);
    v118 = _mm_loadu_si128(v129);
    v117 = v39;
    sub_1D40600(
      (__int64)&v128,
      v35,
      (__int64)&v107,
      (__int64)&v103,
      (const void ***)&v117,
      (const void ***)&v118,
      v14,
      *(double *)v16.m128i_i64,
      v17);
    v113.m128i_i64[0] = v128.m128i_i64[0];
    v113.m128i_i32[2] = v128.m128i_i32[2];
    v114.m128i_i64[0] = v129[0].m128i_i64[0];
    v114.m128i_i32[2] = v129[0].m128i_i32[2];
  }
  v40 = *(_QWORD *)a1;
  v115.m128i_i32[2] = 0;
  v116.m128i_i32[2] = 0;
  v115.m128i_i64[0] = 0;
  v41 = (unsigned __int8 *)(*(_QWORD *)(v105.m128i_i64[0] + 40) + 16LL * v105.m128i_u32[2]);
  v42 = *(_QWORD *)(a1 + 8);
  v43 = *((_QWORD *)v41 + 1);
  v116.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v128, v40, *(_QWORD *)(v42 + 48), *v41, v43);
  if ( v128.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v105.m128i_u64[0], v105.m128i_i64[1], &v116, &v115);
  }
  else
  {
    v44 = *(__int64 **)(a1 + 8);
    v117.m128i_i8[0] = 0;
    v117.m128i_i64[1] = 0;
    v118.m128i_i8[0] = 0;
    v118.m128i_i64[1] = 0;
    v45 = *(_QWORD *)(v105.m128i_i64[0] + 40) + 16LL * v105.m128i_u32[2];
    v46 = *(_BYTE *)v45;
    v47 = *(_QWORD *)(v45 + 8);
    LOBYTE(v119) = v46;
    v120 = v47;
    sub_1D19A30((__int64)&v128, (__int64)v44, &v119);
    v48 = _mm_loadu_si128(&v128);
    v118 = _mm_loadu_si128(v129);
    v117 = v48;
    sub_1D40600(
      (__int64)&v128,
      v44,
      (__int64)&v105,
      (__int64)&v103,
      (const void ***)&v117,
      (const void ***)&v118,
      v14,
      *(double *)v16.m128i_i64,
      v17);
    v116.m128i_i64[0] = v128.m128i_i64[0];
    v116.m128i_i32[2] = v128.m128i_i32[2];
    v115.m128i_i64[0] = v129[0].m128i_i64[0];
    v115.m128i_i32[2] = v129[0].m128i_i32[2];
  }
  v49 = *(_QWORD *)(a2 + 104);
  v50 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v51 = *(_QWORD *)(v49 + 64);
  v128 = _mm_loadu_si128((const __m128i *)(v49 + 40));
  v129[0].m128i_i64[0] = *(_QWORD *)(v49 + 56);
  if ( v111.m128i_i8[0] )
  {
    v52 = sub_2021900(v111.m128i_i8[0]);
  }
  else
  {
    v91 = v51;
    v90 = v49;
    v87 = sub_1F58D40((__int64)&v111);
    v54 = v91;
    v53 = v90;
    v52 = v87;
  }
  v55 = sub_1E0B8E0(
          v50,
          1u,
          (unsigned int)(v52 + 7) >> 3,
          v97,
          (int)&v128,
          v54,
          *(_OWORD *)v53,
          *(_QWORD *)(v53 + 16),
          1u,
          0,
          0);
  v56 = _mm_loadu_si128(&v113);
  v57 = _mm_loadu_si128(&v108);
  v58 = *(_QWORD **)(a1 + 8);
  v119 = v95;
  v59 = _mm_loadu_si128(&v116);
  v120 = v102;
  v123 = v99;
  v121 = v56;
  v124 = v100;
  v122 = v57;
  v126 = v101;
  v125 = v59;
  v127 = v96;
  v60 = sub_1D252B0((__int64)v58, v9.m128i_u32[0], v9.m128i_i64[1], 1, 0);
  v62 = sub_1D24AE0(v58, v60, v61, v9.m128i_u8[0], v9.m128i_i64[1], (__int64)&v103, &v119, 6, v55);
  v63 = *(_QWORD *)(a2 + 104);
  v93 = (__int64)v62;
  v64 = _mm_loadu_si128((const __m128i *)(v63 + 40));
  v65 = *(_QWORD *)(v63 + 64);
  v94 = v66;
  v67 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v128 = v64;
  v129[0].m128i_i64[0] = *(_QWORD *)(v63 + 56);
  if ( v112.m128i_i8[0] )
  {
    v68 = sub_2021900(v112.m128i_i8[0]);
  }
  else
  {
    v92 = v65;
    v86 = sub_1F58D40((__int64)&v112);
    v69 = v92;
    v68 = v86;
  }
  v98 = sub_1E0B8E0(
          v67,
          1u,
          (unsigned int)(v68 + 7) >> 3,
          v97,
          (int)&v128,
          v69,
          *(_OWORD *)v63,
          *(_QWORD *)(v63 + 16),
          1u,
          0,
          0);
  v130 = v99;
  v128.m128i_i64[0] = v95;
  v70 = _mm_loadu_si128(&v114);
  v134 = v96;
  v71 = _mm_loadu_si128(&v109);
  v72 = _mm_loadu_si128(&v115);
  v131 = v100;
  v73 = *(_QWORD **)(a1 + 8);
  v133 = v101;
  v129[0] = v70;
  v129[1] = v71;
  v132 = v72;
  v128.m128i_i64[1] = v102;
  v74 = sub_1D252B0((__int64)v73, v10.m128i_u32[0], v10.m128i_i64[1], 1, 0);
  v76 = sub_1D24AE0(v73, v74, v75, v10.m128i_u8[0], v10.m128i_i64[1], (__int64)&v103, v128.m128i_i64, 6, v98);
  v78 = v77;
  *((_QWORD *)&v88 + 1) = 1;
  *(_QWORD *)&v88 = v76;
  v79 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          2,
          (__int64)&v103,
          1,
          0,
          0,
          *(double *)v59.m128i_i64,
          *(double *)v64.m128i_i64,
          v70,
          v93,
          1u,
          v88);
  sub_2013400(a1, a2, 1, (__int64)v79, (__m128i *)(v102 & 0xFFFFFFFF00000000LL | v80), v81);
  *((_QWORD *)&v89 + 1) = v78;
  *(_QWORD *)&v89 = v76;
  v82 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          107,
          (__int64)&v103,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          *(double *)v59.m128i_i64,
          *(double *)v64.m128i_i64,
          v70,
          v93,
          v94,
          v89);
  sub_2013400(a1, a2, 0, (__int64)v82, v83, v84);
  if ( v103 )
    sub_161E7C0((__int64)&v103, v103);
  return 0;
}
