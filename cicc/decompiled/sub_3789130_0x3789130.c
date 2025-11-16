// Function: sub_3789130
// Address: 0x3789130
//
__m128i *__fastcall sub_3789130(__int64 *a1, __int64 a2, int a3, __m128i a4)
{
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rcx
  __int16 v9; // si
  int v10; // ebx
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rcx
  int v14; // r14d
  __m128i v15; // xmm3
  __m128i v16; // xmm7
  __m128i v17; // xmm4
  __int64 v18; // rsi
  unsigned __int16 *v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  __m128i v24; // xmm4
  __int64 v25; // rsi
  unsigned __int16 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r8
  __m128i *v29; // rsi
  __int64 v30; // rax
  __int16 v31; // dx
  __int64 v32; // rax
  __m128i v33; // xmm6
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // r9
  const __m128i *v37; // rax
  int v38; // edx
  char v39; // si
  __m128i v40; // xmm5
  __m128i v41; // xmm6
  __m128i v42; // xmm0
  __m128i v43; // xmm1
  __int64 *v44; // rdi
  unsigned __int16 v45; // r14
  __m128i *v46; // rax
  __int32 v47; // edx
  __m128i *v48; // rax
  __m128i v49; // xmm2
  __m128i v50; // xmm7
  __m128i v51; // xmm3
  _QWORD *v52; // rdi
  __m128i v53; // xmm4
  unsigned __int16 v54; // r15
  __int32 v55; // edx
  __m128i *v56; // rax
  __int32 v57; // edx
  __m128i *v58; // r14
  __int64 v60; // rax
  __m128i v61; // xmm1
  __m128i v62; // xmm2
  __m128i v63; // xmm5
  unsigned __int16 v64; // r14
  __int64 *v65; // rdi
  __m128i v66; // xmm0
  __m128i *v67; // rax
  __int32 v68; // edx
  __m128i *v69; // rax
  unsigned __int16 v70; // r14
  _QWORD *v71; // r15
  __m128i v72; // xmm6
  __int32 v73; // edx
  __m128i v74; // xmm7
  __m128i v75; // xmm3
  __m128i v76; // xmm4
  __m128i *v77; // rax
  __int32 v78; // edx
  __m128i v79; // xmm1
  __int64 v80; // rax
  __m128i v81; // xmm5
  unsigned __int64 v82; // [rsp+0h] [rbp-260h]
  __int64 v83; // [rsp+8h] [rbp-258h]
  unsigned __int64 v84; // [rsp+10h] [rbp-250h]
  __int64 v85; // [rsp+18h] [rbp-248h]
  unsigned __int64 v86; // [rsp+20h] [rbp-240h]
  __int64 v87; // [rsp+28h] [rbp-238h]
  unsigned __int64 *v88; // [rsp+30h] [rbp-230h]
  __int64 v89; // [rsp+38h] [rbp-228h]
  unsigned __int64 *v90; // [rsp+40h] [rbp-220h]
  __int64 v91; // [rsp+48h] [rbp-218h]
  __int64 *v92; // [rsp+50h] [rbp-210h]
  const __m128i *v93; // [rsp+58h] [rbp-208h]
  _QWORD *v94; // [rsp+60h] [rbp-200h]
  int v95; // [rsp+6Ch] [rbp-1F4h]
  __int64 v96[2]; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v97; // [rsp+80h] [rbp-1E0h] BYREF
  int v98; // [rsp+88h] [rbp-1D8h]
  __m128i v99; // [rsp+90h] [rbp-1D0h] BYREF
  __m128i v100; // [rsp+A0h] [rbp-1C0h] BYREF
  __m128i v101; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i v102; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v103; // [rsp+D0h] [rbp-190h] BYREF
  __m128i v104; // [rsp+E0h] [rbp-180h] BYREF
  __m128i v105; // [rsp+F0h] [rbp-170h] BYREF
  __m128i v106; // [rsp+100h] [rbp-160h] BYREF
  __m128i v107; // [rsp+110h] [rbp-150h]
  __m128i v108; // [rsp+120h] [rbp-140h] BYREF
  __m128i v109; // [rsp+130h] [rbp-130h] BYREF
  __m128i v110; // [rsp+140h] [rbp-120h] BYREF
  unsigned __int64 *v111; // [rsp+150h] [rbp-110h] BYREF
  __int64 v112; // [rsp+158h] [rbp-108h]
  __m128i v113; // [rsp+160h] [rbp-100h]
  __m128i v114; // [rsp+170h] [rbp-F0h]
  __m128i v115; // [rsp+180h] [rbp-E0h]
  __m128i v116; // [rsp+190h] [rbp-D0h]
  __m128i v117; // [rsp+1A0h] [rbp-C0h]
  __int64 v118; // [rsp+1B0h] [rbp-B0h]
  __int64 v119; // [rsp+1B8h] [rbp-A8h]
  __m128i v120; // [rsp+1C0h] [rbp-A0h] BYREF
  __m128i v121; // [rsp+1D0h] [rbp-90h] BYREF
  __m128i v122; // [rsp+1E0h] [rbp-80h]
  __m128i v123; // [rsp+1F0h] [rbp-70h]
  __m128i v124; // [rsp+200h] [rbp-60h]
  __m128i v125; // [rsp+210h] [rbp-50h]
  __int64 v126; // [rsp+220h] [rbp-40h]
  unsigned __int64 v127; // [rsp+228h] [rbp-38h]

  LODWORD(v93) = a3;
  v6 = *(_QWORD *)(a2 + 40);
  v88 = *(unsigned __int64 **)v6;
  LODWORD(v90) = *(_DWORD *)(v6 + 8);
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 <= 365 )
  {
    if ( v7 <= 363 )
    {
      if ( v7 != 339 && (v7 & 0xFFFFFFBF) != 0x12B )
        goto LABEL_5;
      goto LABEL_26;
    }
LABEL_24:
    v8 = v6 + 120;
    goto LABEL_6;
  }
  if ( v7 > 467 )
  {
    if ( v7 == 497 )
      goto LABEL_24;
LABEL_5:
    v8 = v6 + 40;
    goto LABEL_6;
  }
  if ( v7 <= 464 )
    goto LABEL_5;
LABEL_26:
  v8 = v6 + 80;
LABEL_6:
  v9 = *(_WORD *)(a2 + 96);
  v94 = *(_QWORD **)v8;
  v10 = *(_DWORD *)(v8 + 8);
  v11 = *(_QWORD *)(a2 + 104);
  LOWORD(v96[0]) = v9;
  v12 = *(_QWORD *)(a2 + 80);
  v95 = v10;
  v96[1] = v11;
  v13 = *(_QWORD *)(a2 + 112);
  v97 = v12;
  v14 = *(unsigned __int8 *)(v13 + 34);
  if ( v12 )
  {
    sub_B96E90((__int64)&v97, v12, 1);
    v7 = *(_DWORD *)(a2 + 24);
    v6 = *(_QWORD *)(a2 + 40);
  }
  v98 = *(_DWORD *)(a2 + 72);
  if ( v7 == 365 )
  {
    v15 = _mm_loadu_si128((const __m128i *)(v6 + 200));
    v107 = _mm_loadu_si128((const __m128i *)(v6 + 80));
    v16 = _mm_loadu_si128((const __m128i *)(v6 + 160));
    v109 = v15;
    v17 = _mm_loadu_si128((const __m128i *)(v6 + 40));
    v108 = v16;
    v110 = v17;
  }
  else
  {
    if ( v7 == 470 )
    {
      v81 = _mm_loadu_si128((const __m128i *)(v6 + 80));
      v80 = 120;
      v107 = _mm_loadu_si128((const __m128i *)(v6 + 160));
      v108 = v81;
    }
    else
    {
      a4 = _mm_loadu_si128((const __m128i *)(v6 + 200));
      v79 = _mm_loadu_si128((const __m128i *)(v6 + 120));
      v80 = 160;
      v107 = a4;
      v108 = v79;
    }
    v109 = _mm_loadu_si128((const __m128i *)(v6 + v80));
    v110 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  }
  sub_33D0340((__int64)&v120, a1[1], v96);
  v18 = *a1;
  v99.m128i_i32[2] = 0;
  v100.m128i_i32[2] = 0;
  v87 = v120.m128i_i64[0];
  v99.m128i_i64[0] = 0;
  v86 = v120.m128i_u64[1];
  v100.m128i_i64[0] = 0;
  v84 = v121.m128i_u64[1];
  v85 = v121.m128i_i64[0];
  v19 = (unsigned __int16 *)(*(_QWORD *)(v110.m128i_i64[0] + 48) + 16LL * v110.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v120, v18, *(_QWORD *)(a1[1] + 64), *v19, *((_QWORD *)v19 + 1));
  if ( v120.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v110.m128i_u64[0], v110.m128i_i64[1], (__int64)&v99, (__int64)&v100);
    v92 = (__int64 *)&v111;
  }
  else
  {
    v20 = (_QWORD *)a1[1];
    v105.m128i_i16[0] = 0;
    v106.m128i_i16[0] = 0;
    v105.m128i_i64[1] = 0;
    v106.m128i_i64[1] = 0;
    v21 = *(_QWORD *)(v110.m128i_i64[0] + 48) + 16LL * v110.m128i_u32[2];
    v22 = *(_WORD *)v21;
    v23 = *(_QWORD *)(v21 + 8);
    v83 = (__int64)v20;
    v112 = v23;
    LOWORD(v111) = v22;
    v92 = (__int64 *)&v111;
    sub_33D0340((__int64)&v120, (__int64)v20, (__int64 *)&v111);
    v24 = _mm_loadu_si128(&v121);
    v105 = _mm_loadu_si128(&v120);
    v106 = v24;
    sub_3408290(
      (__int64)&v120,
      v20,
      (__int128 *)v110.m128i_i8,
      (__int64)&v97,
      (unsigned int *)&v105,
      (unsigned int *)&v106,
      a4);
    v99.m128i_i64[0] = v120.m128i_i64[0];
    v99.m128i_i32[2] = v120.m128i_i32[2];
    v100.m128i_i64[0] = v121.m128i_i64[0];
    v100.m128i_i32[2] = v121.m128i_i32[2];
  }
  v101.m128i_i64[0] = 0;
  v101.m128i_i32[2] = 0;
  v102.m128i_i64[0] = 0;
  v102.m128i_i32[2] = 0;
  if ( (_DWORD)v93 == 1 && *(_DWORD *)(v107.m128i_i64[0] + 24) == 208 )
  {
    sub_377EF80(a1, v107.m128i_i64[0], (__int64)&v101, (__int64)&v102, a4);
  }
  else
  {
    sub_3777810(&v120, a1, v107.m128i_u64[0], v107.m128i_u64[1], (__int64)&v97, a4);
    v101.m128i_i64[0] = v120.m128i_i64[0];
    v101.m128i_i32[2] = v120.m128i_i32[2];
    v102.m128i_i64[0] = v121.m128i_i64[0];
    v102.m128i_i32[2] = v121.m128i_i32[2];
  }
  v25 = *a1;
  v103.m128i_i32[2] = 0;
  v104.m128i_i32[2] = 0;
  v103.m128i_i64[0] = 0;
  v26 = (unsigned __int16 *)(*(_QWORD *)(v108.m128i_i64[0] + 48) + 16LL * v108.m128i_u32[2]);
  v27 = a1[1];
  v28 = *((_QWORD *)v26 + 1);
  v104.m128i_i64[0] = 0;
  sub_2FE6CC0((__int64)&v120, v25, *(_QWORD *)(v27 + 64), *v26, v28);
  if ( v120.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v108.m128i_u64[0], v108.m128i_i64[1], (__int64)&v104, (__int64)&v103);
  }
  else
  {
    v29 = (__m128i *)a1[1];
    v105.m128i_i16[0] = 0;
    v106.m128i_i16[0] = 0;
    v105.m128i_i64[1] = 0;
    v106.m128i_i64[1] = 0;
    v30 = *(_QWORD *)(v108.m128i_i64[0] + 48) + 16LL * v108.m128i_u32[2];
    v31 = *(_WORD *)v30;
    v32 = *(_QWORD *)(v30 + 8);
    v93 = v29;
    LOWORD(v111) = v31;
    v112 = v32;
    sub_33D0340((__int64)&v120, (__int64)v29, v92);
    v33 = _mm_loadu_si128(&v121);
    v105 = _mm_loadu_si128(&v120);
    v106 = v33;
    sub_3408290(
      (__int64)&v120,
      v29,
      (__int128 *)v108.m128i_i8,
      (__int64)&v97,
      (unsigned int *)&v105,
      (unsigned int *)&v106,
      a4);
    v104.m128i_i64[0] = v120.m128i_i64[0];
    v104.m128i_i32[2] = v120.m128i_i32[2];
    v103.m128i_i64[0] = v121.m128i_i64[0];
    v103.m128i_i32[2] = v121.m128i_i32[2];
  }
  v34 = *(_QWORD **)(a1[1] + 40);
  v35 = *(_QWORD *)(a2 + 112);
  v36 = *(_QWORD *)(v35 + 72);
  v120 = _mm_loadu_si128((const __m128i *)(v35 + 40));
  v121 = _mm_loadu_si128((const __m128i *)(v35 + 56));
  v37 = (const __m128i *)sub_2E7BD70(v34, 2u, -1, v14, (int)&v120, v36, *(_OWORD *)v35, *(_QWORD *)(v35 + 16), 1u, 0, 0);
  v38 = *(_DWORD *)(a2 + 24);
  v93 = v37;
  if ( v38 == 365 )
  {
    v39 = *(_BYTE *)(a2 + 33);
    v40 = _mm_loadu_si128(&v99);
    v41 = _mm_loadu_si128(&v101);
    v89 = 6;
    v111 = v88;
    v42 = _mm_loadu_si128(&v104);
    v43 = _mm_loadu_si128(&v109);
    v113 = v40;
    LODWORD(v112) = (_DWORD)v90;
    v114 = v41;
    v115.m128i_i64[0] = (__int64)v94;
    v44 = (__int64 *)a1[1];
    LODWORD(v90) = (v39 & 4) != 0;
    v45 = *(_WORD *)(a2 + 32);
    v116 = v42;
    v117 = v43;
    v115.m128i_i32[2] = v95;
    v88 = (unsigned __int64 *)v92;
    v92 = v44;
    v46 = sub_33ED250((__int64)v44, 1, 0);
    v48 = sub_33E7ED0(v44, (unsigned __int64)v46, v47, v87, v86, (__int64)&v97, v88, 6, v93, (v45 >> 7) & 7, (char)v90);
    LOBYTE(v45) = *(_BYTE *)(a2 + 33);
    v49 = _mm_loadu_si128(&v100);
    v120.m128i_i64[0] = (__int64)v48;
    v50 = _mm_loadu_si128(&v102);
    v123.m128i_i64[0] = (__int64)v94;
    v51 = _mm_loadu_si128(&v103);
    v52 = (_QWORD *)a1[1];
    v123.m128i_i32[2] = v95;
    v53 = _mm_loadu_si128(&v109);
    v54 = *(_WORD *)(a2 + 32);
    v122 = v50;
    v120.m128i_i32[2] = v55;
    v94 = v52;
    v121 = v49;
    v124 = v51;
    v125 = v53;
    v95 = (v45 & 4) != 0;
    v56 = sub_33ED250((__int64)v52, 1, 0);
    v58 = sub_33E7ED0(
            v52,
            (unsigned __int64)v56,
            v57,
            v85,
            v84,
            (__int64)&v97,
            (unsigned __int64 *)&v120,
            6,
            v93,
            (v54 >> 7) & 7,
            v95);
  }
  else
  {
    v60 = 200;
    if ( v38 != 470 )
      v60 = 240;
    sub_3408380(
      &v120,
      (_QWORD *)a1[1],
      *(_QWORD *)(v60 + *(_QWORD *)(a2 + 40)),
      *(_QWORD *)(v60 + *(_QWORD *)(a2 + 40) + 8),
      *(unsigned __int16 *)(*(_QWORD *)(v110.m128i_i64[0] + 48) + 16LL * v110.m128i_u32[2]),
      *(_QWORD *)(*(_QWORD *)(v110.m128i_i64[0] + 48) + 16LL * v110.m128i_u32[2] + 8),
      a4,
      (__int64)&v97);
    v61 = _mm_loadu_si128(&v104);
    v111 = v88;
    v83 = v121.m128i_i64[0];
    LODWORD(v112) = (_DWORD)v90;
    v62 = _mm_loadu_si128(&v109);
    v63 = _mm_loadu_si128(&v101);
    v114.m128i_i64[0] = (__int64)v94;
    v64 = *(_WORD *)(a2 + 32);
    v114.m128i_i32[2] = v95;
    v65 = (__int64 *)a1[1];
    v82 = _mm_cvtsi32_si128(v121.m128i_u32[2]).m128i_u64[0];
    v66 = _mm_loadu_si128(&v99);
    v115 = v61;
    v116 = v62;
    v117 = v63;
    v113 = v66;
    v119 = v120.m128i_u32[2];
    v90 = (unsigned __int64 *)v92;
    v91 = 7;
    v92 = v65;
    v118 = v120.m128i_i64[0];
    v67 = sub_33ED250((__int64)v65, 1, 0);
    v69 = sub_33E6FD0(v65, (unsigned __int64)v67, v68, v87, v86, (__int64)&v97, v90, 7, v93, (v64 >> 7) & 7);
    v70 = *(_WORD *)(a2 + 32);
    v120.m128i_i64[0] = (__int64)v69;
    v71 = (_QWORD *)a1[1];
    v72 = _mm_loadu_si128(&v100);
    v120.m128i_i32[2] = v73;
    v122.m128i_i64[0] = (__int64)v94;
    v74 = _mm_loadu_si128(&v103);
    v122.m128i_i32[2] = v95;
    v75 = _mm_loadu_si128(&v109);
    v76 = _mm_loadu_si128(&v102);
    v126 = v83;
    v123 = v74;
    v121 = v72;
    v124 = v75;
    v125 = v76;
    v127 = v82;
    v95 = (v70 >> 7) & 7;
    v77 = sub_33ED250((__int64)v71, 1, 0);
    v58 = sub_33E6FD0(v71, (unsigned __int64)v77, v78, v85, v84, (__int64)&v97, (unsigned __int64 *)&v120, 7, v93, v95);
  }
  if ( v97 )
    sub_B91220((__int64)&v97, v97);
  return v58;
}
