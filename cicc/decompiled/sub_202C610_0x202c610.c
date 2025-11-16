// Function: sub_202C610
// Address: 0x202c610
//
__int64 *__fastcall sub_202C610(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edi
  __int64 v6; // r14
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __int64 v9; // rdi
  __m128i v10; // xmm2
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rsi
  __m128i v16; // xmm4
  __m128i v17; // xmm3
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 *v22; // rsi
  __int64 v23; // rax
  char v24; // dl
  __int64 v25; // rax
  __m128i v26; // xmm6
  __int64 v27; // rsi
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 *v31; // rsi
  __int64 v32; // rax
  char v33; // dl
  __int64 v34; // rax
  __m128i v35; // xmm5
  __int64 v36; // rsi
  unsigned __int8 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r8
  __int64 *v40; // rsi
  __int64 v41; // rax
  char v42; // dl
  __int64 v43; // rax
  __m128i v44; // xmm7
  __int64 v45; // rcx
  __int64 v46; // r10
  __int64 v47; // r9
  int v48; // edx
  __int64 v49; // rcx
  int v50; // r9d
  __int64 v51; // r10
  __int64 v52; // rax
  __m128i v53; // xmm6
  __m128i v54; // xmm7
  __m128i v55; // xmm0
  _QWORD *v56; // r14
  unsigned __int8 v57; // r15
  __int64 v58; // rcx
  __int64 v59; // r9
  __int64 v60; // rax
  int v61; // edx
  __int64 *v62; // rax
  __int64 v63; // r13
  __int64 v64; // r15
  __int32 v65; // edx
  __int32 v66; // r14d
  __int64 v67; // r9
  __int64 v68; // r10
  int v69; // edx
  int v70; // r9d
  __int64 v71; // r10
  __int64 v72; // rax
  __m128i v73; // xmm2
  _QWORD *v74; // rbx
  __int64 v75; // r13
  __m128i v76; // xmm3
  __m128i v77; // xmm4
  __int64 v78; // r12
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rax
  int v82; // edx
  __int64 *v83; // r12
  int v85; // eax
  int v86; // eax
  __int64 v87; // [rsp+0h] [rbp-230h]
  __int64 v88; // [rsp+8h] [rbp-228h]
  int v89; // [rsp+8h] [rbp-228h]
  __int64 v90; // [rsp+10h] [rbp-220h]
  __int64 v91; // [rsp+10h] [rbp-220h]
  int v92; // [rsp+20h] [rbp-210h]
  __int64 v93; // [rsp+20h] [rbp-210h]
  int v94; // [rsp+20h] [rbp-210h]
  __int64 v95; // [rsp+30h] [rbp-200h]
  unsigned int v96; // [rsp+38h] [rbp-1F8h]
  int v97; // [rsp+3Ch] [rbp-1F4h]
  int v98; // [rsp+40h] [rbp-1F0h]
  unsigned __int8 v99; // [rsp+40h] [rbp-1F0h]
  __m128i v100; // [rsp+50h] [rbp-1E0h] BYREF
  __m128i v101; // [rsp+60h] [rbp-1D0h] BYREF
  __m128i v102; // [rsp+70h] [rbp-1C0h] BYREF
  _QWORD v103[2]; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v104; // [rsp+90h] [rbp-1A0h] BYREF
  int v105; // [rsp+98h] [rbp-198h]
  __m128i v106; // [rsp+A0h] [rbp-190h] BYREF
  __m128i v107; // [rsp+B0h] [rbp-180h] BYREF
  __m128i v108; // [rsp+C0h] [rbp-170h] BYREF
  __m128i v109; // [rsp+D0h] [rbp-160h] BYREF
  __m128i v110; // [rsp+E0h] [rbp-150h] BYREF
  __m128i v111; // [rsp+F0h] [rbp-140h] BYREF
  __m128i v112; // [rsp+100h] [rbp-130h] BYREF
  __m128i v113; // [rsp+110h] [rbp-120h] BYREF
  __m128i v114; // [rsp+120h] [rbp-110h] BYREF
  __m128i v115; // [rsp+130h] [rbp-100h] BYREF
  __int64 v116; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+148h] [rbp-E8h]
  __m128i v118; // [rsp+150h] [rbp-E0h]
  __m128i v119; // [rsp+160h] [rbp-D0h]
  __int64 v120; // [rsp+170h] [rbp-C0h]
  int v121; // [rsp+178h] [rbp-B8h]
  __m128i v122; // [rsp+180h] [rbp-B0h]
  __int64 v123; // [rsp+190h] [rbp-A0h]
  int v124; // [rsp+198h] [rbp-98h]
  __m128i v125; // [rsp+1A0h] [rbp-90h] BYREF
  __m128i v126[2]; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v127; // [rsp+1D0h] [rbp-60h]
  int v128; // [rsp+1D8h] [rbp-58h]
  __m128i v129; // [rsp+1E0h] [rbp-50h]
  __int64 v130; // [rsp+1F0h] [rbp-40h]
  int v131; // [rsp+1F8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 32);
  v5 = *(_DWORD *)(v4 + 8);
  v6 = *(_QWORD *)v4;
  LOBYTE(v103[0]) = *(_BYTE *)(a2 + 88);
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 80));
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 160));
  v92 = v5;
  v9 = *(_QWORD *)(v4 + 120);
  v10 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v100 = v7;
  v95 = v9;
  LODWORD(v9) = *(_DWORD *)(v4 + 128);
  v101 = v8;
  v98 = v9;
  v11 = *(_QWORD *)(v4 + 200);
  v102 = v10;
  v97 = *(_DWORD *)(v4 + 208);
  v103[1] = *(_QWORD *)(a2 + 96);
  v96 = (unsigned int)(1 << *(_WORD *)(*(_QWORD *)(a2 + 104) + 34LL)) >> 1;
  v12 = *(_QWORD *)(a2 + 72);
  v104 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v104, v12, 2);
  v13 = *(_DWORD *)(a2 + 64);
  v14 = a1[1];
  v106.m128i_i8[0] = 0;
  v105 = v13;
  v106.m128i_i64[1] = 0;
  v107.m128i_i8[0] = 0;
  v107.m128i_i64[1] = 0;
  sub_1D19A30((__int64)&v125, v14, v103);
  v15 = *a1;
  v16 = _mm_loadu_si128(v126);
  v108.m128i_i32[2] = 0;
  v17 = _mm_loadu_si128(&v125);
  v109.m128i_i32[2] = 0;
  v18 = (unsigned __int8 *)(*(_QWORD *)(v102.m128i_i64[0] + 40) + 16LL * v102.m128i_u32[2]);
  v19 = a1[1];
  v107 = v16;
  v106 = v17;
  v20 = *((_QWORD *)v18 + 1);
  v108.m128i_i64[0] = 0;
  v21 = *(_QWORD *)(v19 + 48);
  v109.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v125, v15, v21, *v18, v20);
  if ( v125.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v102.m128i_u64[0], v102.m128i_i64[1], &v108, &v109);
  }
  else
  {
    v22 = (__int64 *)a1[1];
    v114.m128i_i8[0] = 0;
    v114.m128i_i64[1] = 0;
    v115.m128i_i8[0] = 0;
    v115.m128i_i64[1] = 0;
    v23 = *(_QWORD *)(v102.m128i_i64[0] + 40) + 16LL * v102.m128i_u32[2];
    v24 = *(_BYTE *)v23;
    v25 = *(_QWORD *)(v23 + 8);
    LOBYTE(v116) = v24;
    v117 = v25;
    sub_1D19A30((__int64)&v125, (__int64)v22, &v116);
    v26 = _mm_loadu_si128(v126);
    v114 = _mm_loadu_si128(&v125);
    v115 = v26;
    sub_1D40600(
      (__int64)&v125,
      v22,
      (__int64)&v102,
      (__int64)&v104,
      (const void ***)&v114,
      (const void ***)&v115,
      v7,
      *(double *)v8.m128i_i64,
      v10);
    v108.m128i_i64[0] = v125.m128i_i64[0];
    v108.m128i_i32[2] = v125.m128i_i32[2];
    v109.m128i_i64[0] = v126[0].m128i_i64[0];
    v109.m128i_i32[2] = v126[0].m128i_i32[2];
  }
  v27 = *a1;
  v110.m128i_i32[2] = 0;
  v111.m128i_i32[2] = 0;
  v110.m128i_i64[0] = 0;
  v28 = (unsigned __int8 *)(*(_QWORD *)(v100.m128i_i64[0] + 40) + 16LL * v100.m128i_u32[2]);
  v29 = a1[1];
  v30 = *((_QWORD *)v28 + 1);
  v111.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v125, v27, *(_QWORD *)(v29 + 48), *v28, v30);
  if ( v125.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v100.m128i_u64[0], v100.m128i_i64[1], &v110, &v111);
  }
  else
  {
    v31 = (__int64 *)a1[1];
    v114.m128i_i8[0] = 0;
    v114.m128i_i64[1] = 0;
    v115.m128i_i8[0] = 0;
    v115.m128i_i64[1] = 0;
    v32 = *(_QWORD *)(v100.m128i_i64[0] + 40) + 16LL * v100.m128i_u32[2];
    v33 = *(_BYTE *)v32;
    v34 = *(_QWORD *)(v32 + 8);
    LOBYTE(v116) = v33;
    v117 = v34;
    sub_1D19A30((__int64)&v125, (__int64)v31, &v116);
    v35 = _mm_loadu_si128(v126);
    v114 = _mm_loadu_si128(&v125);
    v115 = v35;
    sub_1D40600(
      (__int64)&v125,
      v31,
      (__int64)&v100,
      (__int64)&v104,
      (const void ***)&v114,
      (const void ***)&v115,
      v7,
      *(double *)v8.m128i_i64,
      v10);
    v110.m128i_i64[0] = v125.m128i_i64[0];
    v110.m128i_i32[2] = v125.m128i_i32[2];
    v111.m128i_i64[0] = v126[0].m128i_i64[0];
    v111.m128i_i32[2] = v126[0].m128i_i32[2];
  }
  v36 = *a1;
  v112.m128i_i32[2] = 0;
  v113.m128i_i32[2] = 0;
  v112.m128i_i64[0] = 0;
  v37 = (unsigned __int8 *)(*(_QWORD *)(v101.m128i_i64[0] + 40) + 16LL * v101.m128i_u32[2]);
  v38 = a1[1];
  v39 = *((_QWORD *)v37 + 1);
  v113.m128i_i64[0] = 0;
  sub_1F40D10((__int64)&v125, v36, *(_QWORD *)(v38 + 48), *v37, v39);
  if ( v125.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v101.m128i_u64[0], v101.m128i_i64[1], &v113, &v112);
  }
  else
  {
    v40 = (__int64 *)a1[1];
    v114.m128i_i8[0] = 0;
    v114.m128i_i64[1] = 0;
    v115.m128i_i8[0] = 0;
    v115.m128i_i64[1] = 0;
    v41 = *(_QWORD *)(v101.m128i_i64[0] + 40) + 16LL * v101.m128i_u32[2];
    v42 = *(_BYTE *)v41;
    v43 = *(_QWORD *)(v41 + 8);
    LOBYTE(v116) = v42;
    v117 = v43;
    sub_1D19A30((__int64)&v125, (__int64)v40, &v116);
    v44 = _mm_loadu_si128(v126);
    v114 = _mm_loadu_si128(&v125);
    v115 = v44;
    sub_1D40600(
      (__int64)&v125,
      v40,
      (__int64)&v101,
      (__int64)&v104,
      (const void ***)&v114,
      (const void ***)&v115,
      v7,
      *(double *)v8.m128i_i64,
      v10);
    v113.m128i_i64[0] = v125.m128i_i64[0];
    v113.m128i_i32[2] = v125.m128i_i32[2];
    v112.m128i_i64[0] = v126[0].m128i_i64[0];
    v112.m128i_i32[2] = v126[0].m128i_i32[2];
  }
  v45 = *(_QWORD *)(a2 + 104);
  v46 = *(_QWORD *)(a1[1] + 32);
  v47 = *(_QWORD *)(v45 + 64);
  v125 = _mm_loadu_si128((const __m128i *)(v45 + 40));
  v126[0].m128i_i64[0] = *(_QWORD *)(v45 + 56);
  if ( v106.m128i_i8[0] )
  {
    v48 = sub_2021900(v106.m128i_i8[0]);
  }
  else
  {
    v87 = v46;
    v89 = v47;
    v91 = v45;
    v86 = sub_1F58D40((__int64)&v106);
    v51 = v87;
    v50 = v89;
    v49 = v91;
    v48 = v86;
  }
  v52 = sub_1E0B8E0(
          v51,
          2u,
          (unsigned int)(v48 + 7) >> 3,
          v96,
          (int)&v125,
          v50,
          *(_OWORD *)v49,
          *(_QWORD *)(v49 + 16),
          1u,
          0,
          0);
  v53 = _mm_loadu_si128(&v108);
  v88 = v52;
  v54 = _mm_loadu_si128(&v110);
  v55 = _mm_loadu_si128(&v113);
  v116 = v6;
  LODWORD(v117) = v92;
  v118 = v53;
  v56 = (_QWORD *)a1[1];
  v120 = v95;
  v119 = v54;
  v121 = v98;
  v122 = v55;
  v123 = v11;
  v124 = v97;
  v57 = *(_BYTE *)(*(_QWORD *)(v108.m128i_i64[0] + 40) + 16LL * v108.m128i_u32[2]);
  v93 = *(_QWORD *)(*(_QWORD *)(v108.m128i_i64[0] + 40) + 16LL * v108.m128i_u32[2] + 8);
  v60 = sub_1D29190((__int64)v56, 1u, 0, v58, v93, v59);
  v62 = sub_1D24800(v56, v60, v61, v57, v93, (__int64)&v104, &v116, 6, v88);
  v63 = *(_QWORD *)(a2 + 104);
  v64 = (__int64)v62;
  v66 = v65;
  v67 = *(_QWORD *)(v63 + 64);
  v68 = *(_QWORD *)(a1[1] + 32);
  v125 = _mm_loadu_si128((const __m128i *)(v63 + 40));
  v126[0].m128i_i64[0] = *(_QWORD *)(v63 + 56);
  if ( v107.m128i_i8[0] )
  {
    v69 = sub_2021900(v107.m128i_i8[0]);
  }
  else
  {
    v90 = v68;
    v94 = v67;
    v85 = sub_1F58D40((__int64)&v107);
    v71 = v90;
    v70 = v94;
    v69 = v85;
  }
  v72 = sub_1E0B8E0(
          v71,
          2u,
          (unsigned int)(v69 + 7) >> 3,
          v96,
          (int)&v125,
          v70,
          *(_OWORD *)v63,
          *(_QWORD *)(v63 + 16),
          1u,
          0,
          0);
  v73 = _mm_loadu_si128(&v109);
  v74 = (_QWORD *)a1[1];
  v125.m128i_i64[0] = v64;
  v75 = v72;
  v76 = _mm_loadu_si128(&v111);
  v126[0] = v73;
  v77 = _mm_loadu_si128(&v112);
  v127 = v95;
  v125.m128i_i32[2] = v66;
  v128 = v98;
  v126[1] = v76;
  v130 = v11;
  v129 = v77;
  v131 = v97;
  v78 = *(_QWORD *)(*(_QWORD *)(v109.m128i_i64[0] + 40) + 16LL * v109.m128i_u32[2] + 8);
  v99 = *(_BYTE *)(*(_QWORD *)(v109.m128i_i64[0] + 40) + 16LL * v109.m128i_u32[2]);
  v81 = sub_1D29190((__int64)v74, 1u, 0, v99, v79, v80);
  v83 = sub_1D24800(v74, v81, v82, v99, v78, (__int64)&v104, v125.m128i_i64, 6, v75);
  if ( v104 )
    sub_161E7C0((__int64)&v104, v104);
  return v83;
}
