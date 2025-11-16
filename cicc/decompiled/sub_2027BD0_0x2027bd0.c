// Function: sub_2027BD0
// Address: 0x2027bd0
//
void __fastcall sub_2027BD0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  char *v8; // rax
  char v9; // dl
  __int64 v10; // rax
  __m128i v11; // kr00_16
  __m128i v12; // kr10_16
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rdx
  __m128i v17; // xmm0
  __int64 v18; // rcx
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rax
  __int64 *v24; // rsi
  __int64 v25; // rax
  char v26; // dl
  __int64 v27; // rax
  __m128i v28; // xmm4
  char v29; // dl
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __m128i v33; // xmm3
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r8
  __int64 *v37; // rsi
  __int64 v38; // rax
  char v39; // dl
  __int64 v40; // rax
  __m128i v41; // xmm6
  __int64 v42; // rsi
  unsigned __int8 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r8
  __int64 *v46; // rsi
  __int64 v47; // rax
  char v48; // dl
  __int64 v49; // rax
  __m128i v50; // xmm3
  __int64 v51; // rcx
  __int64 v52; // r10
  __int64 v53; // r9
  int v54; // edx
  __int64 v55; // rcx
  int v56; // r9d
  __int64 v57; // r10
  __int64 v58; // rax
  __m128i v59; // xmm5
  __m128i v60; // xmm6
  __m128i v61; // xmm7
  _QWORD *v62; // rdi
  __int64 v63; // rax
  int v64; // edx
  int v65; // edx
  __m128i v66; // xmm0
  _QWORD *v67; // rdi
  __m128i v68; // xmm1
  __m128i v69; // xmm2
  __int64 v70; // rax
  int v71; // edx
  int v72; // edx
  unsigned int v73; // edx
  const __m128i *v74; // r9
  int v75; // eax
  __int128 v76; // [rsp-10h] [rbp-2A0h]
  __int64 v77; // [rsp+8h] [rbp-288h]
  __int64 v78; // [rsp+10h] [rbp-280h]
  int v79; // [rsp+20h] [rbp-270h]
  __int64 v81; // [rsp+50h] [rbp-240h]
  unsigned int v82; // [rsp+58h] [rbp-238h]
  __int64 v83; // [rsp+58h] [rbp-238h]
  __int64 v84; // [rsp+60h] [rbp-230h]
  int v85; // [rsp+68h] [rbp-228h]
  int v86; // [rsp+70h] [rbp-220h]
  __int64 v87; // [rsp+78h] [rbp-218h]
  __int64 *v89; // [rsp+90h] [rbp-200h]
  __int64 *v90; // [rsp+A0h] [rbp-1F0h]
  __int64 v91; // [rsp+C0h] [rbp-1D0h] BYREF
  int v92; // [rsp+C8h] [rbp-1C8h]
  __m128i v93; // [rsp+D0h] [rbp-1C0h] BYREF
  __m128i v94; // [rsp+E0h] [rbp-1B0h] BYREF
  __m128i v95; // [rsp+F0h] [rbp-1A0h] BYREF
  __m128i v96; // [rsp+100h] [rbp-190h] BYREF
  __m128i v97; // [rsp+110h] [rbp-180h] BYREF
  _QWORD v98[2]; // [rsp+120h] [rbp-170h] BYREF
  __m128i v99; // [rsp+130h] [rbp-160h] BYREF
  __m128i v100; // [rsp+140h] [rbp-150h] BYREF
  __m128i v101; // [rsp+150h] [rbp-140h] BYREF
  __m128i v102; // [rsp+160h] [rbp-130h] BYREF
  __m128i v103; // [rsp+170h] [rbp-120h] BYREF
  __m128i v104; // [rsp+180h] [rbp-110h] BYREF
  __m128i v105; // [rsp+190h] [rbp-100h] BYREF
  __int64 v106; // [rsp+1A0h] [rbp-F0h] BYREF
  __int64 v107; // [rsp+1A8h] [rbp-E8h]
  __m128i v108; // [rsp+1B0h] [rbp-E0h]
  __m128i v109; // [rsp+1C0h] [rbp-D0h]
  __int64 v110; // [rsp+1D0h] [rbp-C0h]
  int v111; // [rsp+1D8h] [rbp-B8h]
  __m128i v112; // [rsp+1E0h] [rbp-B0h]
  __int64 v113; // [rsp+1F0h] [rbp-A0h]
  int v114; // [rsp+1F8h] [rbp-98h]
  __m128i v115; // [rsp+200h] [rbp-90h] BYREF
  __m128i v116[2]; // [rsp+210h] [rbp-80h] BYREF
  __int64 v117; // [rsp+230h] [rbp-60h]
  int v118; // [rsp+238h] [rbp-58h]
  __m128i v119; // [rsp+240h] [rbp-50h]
  __int64 v120; // [rsp+250h] [rbp-40h]
  int v121; // [rsp+258h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 72);
  v91 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v91, v6, 2);
  v7 = *(_QWORD *)(a1 + 8);
  v92 = *(_DWORD *)(a2 + 64);
  v8 = *(char **)(a2 + 40);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOBYTE(v106) = v9;
  v107 = v10;
  sub_1D19A30((__int64)&v115, v7, &v106);
  v11 = v115;
  v12 = v116[0];
  v13 = *(_QWORD *)(a2 + 32);
  v14 = *(_QWORD *)(v13 + 8);
  v15 = *(_QWORD *)v13;
  v96.m128i_i64[0] = 0;
  v16 = *(_QWORD *)(v13 + 80);
  v17 = _mm_loadu_si128((const __m128i *)(v13 + 80));
  v96.m128i_i32[2] = 0;
  v87 = v14;
  v18 = *(_QWORD *)(v13 + 120);
  v19 = _mm_loadu_si128((const __m128i *)(v13 + 40));
  v20 = _mm_loadu_si128((const __m128i *)(v13 + 160));
  v93 = v17;
  v21 = *(_QWORD *)(v13 + 200);
  v84 = v18;
  LODWORD(v18) = *(_DWORD *)(v13 + 128);
  LODWORD(v13) = *(_DWORD *)(v13 + 208);
  v94 = v19;
  v81 = v21;
  v86 = v13;
  v85 = v18;
  LOWORD(v18) = *(_WORD *)(*(_QWORD *)(a2 + 104) + 34LL);
  v95 = v20;
  v82 = (unsigned int)(1 << v18) >> 1;
  v97.m128i_i64[0] = 0;
  v22 = *(_QWORD *)a1;
  v97.m128i_i32[2] = 0;
  v23 = (unsigned __int8 *)(*(_QWORD *)(v16 + 40) + 16LL * v17.m128i_u32[2]);
  sub_1F40D10((__int64)&v115, v22, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v23, *((_QWORD *)v23 + 1));
  if ( v115.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v93.m128i_u64[0], v93.m128i_i64[1], &v96, &v97);
  }
  else
  {
    v24 = *(__int64 **)(a1 + 8);
    v104.m128i_i8[0] = 0;
    v104.m128i_i64[1] = 0;
    v105.m128i_i8[0] = 0;
    v105.m128i_i64[1] = 0;
    v25 = *(_QWORD *)(v93.m128i_i64[0] + 40) + 16LL * v93.m128i_u32[2];
    v26 = *(_BYTE *)v25;
    v27 = *(_QWORD *)(v25 + 8);
    LOBYTE(v106) = v26;
    v107 = v27;
    sub_1D19A30((__int64)&v115, (__int64)v24, &v106);
    v28 = _mm_loadu_si128(v116);
    v104 = _mm_loadu_si128(&v115);
    v105 = v28;
    sub_1D40600(
      (__int64)&v115,
      v24,
      (__int64)&v93,
      (__int64)&v91,
      (const void ***)&v104,
      (const void ***)&v105,
      v17,
      *(double *)v19.m128i_i64,
      v20);
    v96.m128i_i64[0] = v115.m128i_i64[0];
    v96.m128i_i32[2] = v115.m128i_i32[2];
    v97.m128i_i64[0] = v116[0].m128i_i64[0];
    v97.m128i_i32[2] = v116[0].m128i_i32[2];
  }
  v29 = *(_BYTE *)(a2 + 88);
  v30 = *(_QWORD *)(a2 + 96);
  v99.m128i_i8[0] = 0;
  v99.m128i_i64[1] = 0;
  v31 = *(_QWORD *)(a1 + 8);
  LOBYTE(v98[0]) = v29;
  v98[1] = v30;
  sub_1D19A30((__int64)&v115, v31, v98);
  v32 = *(_QWORD *)a1;
  v100.m128i_i32[2] = 0;
  v33 = _mm_loadu_si128(&v115);
  v101.m128i_i32[2] = 0;
  v100.m128i_i64[0] = 0;
  v34 = (unsigned __int8 *)(*(_QWORD *)(v94.m128i_i64[0] + 40) + 16LL * v94.m128i_u32[2]);
  v35 = *(_QWORD *)(a1 + 8);
  v99 = v33;
  v36 = *((_QWORD *)v34 + 1);
  v101.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v115, v32, *(_QWORD *)(v35 + 48), *v34, v36);
  if ( v115.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v94.m128i_u64[0], v94.m128i_i64[1], &v100, &v101);
  }
  else
  {
    v37 = *(__int64 **)(a1 + 8);
    v104.m128i_i8[0] = 0;
    v104.m128i_i64[1] = 0;
    v105.m128i_i8[0] = 0;
    v105.m128i_i64[1] = 0;
    v38 = *(_QWORD *)(v94.m128i_i64[0] + 40) + 16LL * v94.m128i_u32[2];
    v39 = *(_BYTE *)v38;
    v40 = *(_QWORD *)(v38 + 8);
    LOBYTE(v106) = v39;
    v107 = v40;
    sub_1D19A30((__int64)&v115, (__int64)v37, &v106);
    v41 = _mm_loadu_si128(v116);
    v104 = _mm_loadu_si128(&v115);
    v105 = v41;
    sub_1D40600(
      (__int64)&v115,
      v37,
      (__int64)&v94,
      (__int64)&v91,
      (const void ***)&v104,
      (const void ***)&v105,
      v17,
      *(double *)v19.m128i_i64,
      v20);
    v100.m128i_i64[0] = v115.m128i_i64[0];
    v100.m128i_i32[2] = v115.m128i_i32[2];
    v101.m128i_i64[0] = v116[0].m128i_i64[0];
    v101.m128i_i32[2] = v116[0].m128i_i32[2];
  }
  v42 = *(_QWORD *)a1;
  v102.m128i_i32[2] = 0;
  v103.m128i_i32[2] = 0;
  v102.m128i_i64[0] = 0;
  v43 = (unsigned __int8 *)(*(_QWORD *)(v95.m128i_i64[0] + 40) + 16LL * v95.m128i_u32[2]);
  v44 = *(_QWORD *)(a1 + 8);
  v45 = *((_QWORD *)v43 + 1);
  v103.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v115, v42, *(_QWORD *)(v44 + 48), *v43, v45);
  if ( v115.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v95.m128i_u64[0], v95.m128i_i64[1], &v103, &v102);
  }
  else
  {
    v46 = *(__int64 **)(a1 + 8);
    v104.m128i_i8[0] = 0;
    v104.m128i_i64[1] = 0;
    v105.m128i_i8[0] = 0;
    v105.m128i_i64[1] = 0;
    v47 = *(_QWORD *)(v95.m128i_i64[0] + 40) + 16LL * v95.m128i_u32[2];
    v48 = *(_BYTE *)v47;
    v49 = *(_QWORD *)(v47 + 8);
    LOBYTE(v106) = v48;
    v107 = v49;
    sub_1D19A30((__int64)&v115, (__int64)v46, &v106);
    v50 = _mm_loadu_si128(v116);
    v104 = _mm_loadu_si128(&v115);
    v105 = v50;
    sub_1D40600(
      (__int64)&v115,
      v46,
      (__int64)&v95,
      (__int64)&v91,
      (const void ***)&v104,
      (const void ***)&v105,
      v17,
      *(double *)v19.m128i_i64,
      v20);
    v103.m128i_i64[0] = v115.m128i_i64[0];
    v103.m128i_i32[2] = v115.m128i_i32[2];
    v102.m128i_i64[0] = v116[0].m128i_i64[0];
    v102.m128i_i32[2] = v116[0].m128i_i32[2];
  }
  v51 = *(_QWORD *)(a2 + 104);
  v52 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v53 = *(_QWORD *)(v51 + 64);
  v115 = _mm_loadu_si128((const __m128i *)(v51 + 40));
  v116[0].m128i_i64[0] = *(_QWORD *)(v51 + 56);
  if ( v99.m128i_i8[0] )
  {
    v54 = sub_2021900(v99.m128i_i8[0]);
  }
  else
  {
    v77 = v52;
    v79 = v53;
    v78 = v51;
    v75 = sub_1F58D40((__int64)&v99);
    v57 = v77;
    v56 = v79;
    v55 = v78;
    v54 = v75;
  }
  v58 = sub_1E0B8E0(
          v57,
          1u,
          (unsigned int)(v54 + 7) >> 3,
          v82,
          (int)&v115,
          v56,
          *(_OWORD *)v55,
          *(_QWORD *)(v55 + 16),
          1u,
          0,
          0);
  v110 = v84;
  v59 = _mm_loadu_si128(&v100);
  v107 = v87;
  v60 = _mm_loadu_si128(&v96);
  v61 = _mm_loadu_si128(&v103);
  v111 = v85;
  v62 = *(_QWORD **)(a1 + 8);
  v113 = v81;
  v114 = v86;
  v108 = v59;
  v109 = v60;
  v112 = v61;
  v106 = v15;
  v83 = v58;
  v63 = sub_1D252B0((__int64)v62, v11.m128i_u32[0], v11.m128i_i64[1], 1, 0);
  *(_QWORD *)a3 = sub_1D24AE0(v62, v63, v64, v11.m128i_u8[0], v11.m128i_i64[1], (__int64)&v91, &v106, 6, v83);
  v115.m128i_i64[1] = v87;
  *(_DWORD *)(a3 + 8) = v65;
  v66 = _mm_loadu_si128(&v101);
  v67 = *(_QWORD **)(a1 + 8);
  v120 = v81;
  v68 = _mm_loadu_si128(&v97);
  v69 = _mm_loadu_si128(&v102);
  v117 = v84;
  v121 = v86;
  v116[0] = v66;
  v116[1] = v68;
  v119 = v69;
  v115.m128i_i64[0] = v15;
  v118 = v85;
  v70 = sub_1D252B0((__int64)v67, v12.m128i_u32[0], v12.m128i_i64[1], 1, 0);
  v90 = sub_1D24AE0(v67, v70, v71, v12.m128i_u8[0], v12.m128i_i64[1], (__int64)&v91, v115.m128i_i64, 6, v83);
  *(_QWORD *)a4 = v90;
  *(_DWORD *)(a4 + 8) = v72;
  *((_QWORD *)&v76 + 1) = 1;
  *(_QWORD *)&v76 = v90;
  v89 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          2,
          (__int64)&v91,
          1,
          0,
          0,
          *(double *)v66.m128i_i64,
          *(double *)v68.m128i_i64,
          v69,
          *(_QWORD *)a3,
          1u,
          v76);
  sub_2013400(a1, a2, 1, (__int64)v89, (__m128i *)(v73 | v87 & 0xFFFFFFFF00000000LL), v74);
  if ( v91 )
    sub_161E7C0((__int64)&v91, v91);
}
