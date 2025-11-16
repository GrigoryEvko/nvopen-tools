// Function: sub_20272D0
// Address: 0x20272d0
//
void __fastcall sub_20272D0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int8 v9; // dl
  __m128i v10; // kr00_16
  __m128i v11; // kr10_16
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r15
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // rax
  __int64 v19; // rax
  char v20; // di
  __int64 v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // rsi
  unsigned __int8 *v25; // rax
  __int64 *v26; // rsi
  __int64 v27; // rax
  __int8 v28; // dl
  __int64 v29; // rax
  char v30; // dl
  __int64 v31; // rax
  __int64 v32; // rsi
  __m128i v33; // xmm3
  __m128i v34; // xmm4
  unsigned __int8 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // r8
  __int64 v39; // rdx
  __int64 *v40; // rsi
  __int64 v41; // rax
  __int8 v42; // dl
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r10
  __int64 v46; // r9
  int v47; // edx
  __int64 v48; // rcx
  int v49; // r9d
  __int64 v50; // r10
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // ecx
  int v54; // r8d
  int v55; // edx
  unsigned int v56; // edx
  int v57; // ebx
  __int64 v58; // rcx
  __int64 v59; // r10
  __int64 v60; // r9
  int v61; // edx
  __int64 v62; // rcx
  int v63; // r9d
  __int64 v64; // r10
  unsigned int v65; // edx
  unsigned __int64 v66; // rdi
  __int32 v67; // eax
  __int64 v68; // rax
  int v69; // edx
  unsigned int v70; // edx
  const __m128i *v71; // r9
  __int8 v72; // r8
  __int64 v73; // rbx
  __int64 v74; // rax
  int v75; // eax
  int v76; // eax
  __int128 v77; // [rsp-10h] [rbp-200h]
  __int64 v78; // [rsp+8h] [rbp-1E8h]
  int v79; // [rsp+10h] [rbp-1E0h]
  __int64 v80; // [rsp+18h] [rbp-1D8h]
  __int64 v81; // [rsp+30h] [rbp-1C0h]
  int v82; // [rsp+38h] [rbp-1B8h]
  unsigned int v85; // [rsp+50h] [rbp-1A0h]
  int v86; // [rsp+50h] [rbp-1A0h]
  char v87; // [rsp+54h] [rbp-19Ch]
  unsigned int v88; // [rsp+58h] [rbp-198h]
  __int64 v89; // [rsp+58h] [rbp-198h]
  __int64 v90; // [rsp+68h] [rbp-188h]
  __int128 v91; // [rsp+70h] [rbp-180h]
  __int64 *v93; // [rsp+80h] [rbp-170h]
  __int64 v94; // [rsp+90h] [rbp-160h]
  __int64 v95; // [rsp+C0h] [rbp-130h] BYREF
  int v96; // [rsp+C8h] [rbp-128h]
  __m128i v97; // [rsp+D0h] [rbp-120h] BYREF
  __m128i v98; // [rsp+E0h] [rbp-110h] BYREF
  __int128 v99; // [rsp+F0h] [rbp-100h] BYREF
  __int128 v100; // [rsp+100h] [rbp-F0h] BYREF
  _QWORD v101[2]; // [rsp+110h] [rbp-E0h] BYREF
  __m128i v102; // [rsp+120h] [rbp-D0h] BYREF
  __m128i v103; // [rsp+130h] [rbp-C0h] BYREF
  __int128 v104; // [rsp+140h] [rbp-B0h] BYREF
  __int128 v105; // [rsp+150h] [rbp-A0h] BYREF
  __m128i v106; // [rsp+160h] [rbp-90h] BYREF
  __m128i v107; // [rsp+170h] [rbp-80h] BYREF
  __m128i v108; // [rsp+180h] [rbp-70h] BYREF
  __int64 v109; // [rsp+190h] [rbp-60h]
  __m128i v110; // [rsp+1A0h] [rbp-50h] BYREF
  __m128i v111; // [rsp+1B0h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 72);
  v95 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v95, v6, 2);
  v7 = a1[1];
  v96 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_BYTE *)v8;
  v108.m128i_i64[1] = *(_QWORD *)(v8 + 8);
  v108.m128i_i8[0] = v9;
  sub_1D19A30((__int64)&v110, v7, &v108);
  v10 = v110;
  v11 = v111;
  v12 = *(_QWORD *)(a2 + 32);
  v13 = *(_QWORD *)(v12 + 8);
  v14 = *(_QWORD *)v12;
  v15 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v16 = _mm_loadu_si128((const __m128i *)(v12 + 80));
  v17 = _mm_loadu_si128((const __m128i *)(v12 + 120));
  v18 = *(_QWORD *)(a2 + 104);
  v90 = v13;
  v97 = v16;
  LOWORD(v13) = *(_WORD *)(v18 + 34);
  v98 = v17;
  v91 = (__int128)v15;
  v85 = 1 << v13;
  v88 = (unsigned int)(1 << v13) >> 1;
  v87 = (*(_BYTE *)(a2 + 27) >> 2) & 3;
  v19 = *(_QWORD *)(a2 + 40);
  v20 = *(_BYTE *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  v110.m128i_i8[0] = v20;
  v110.m128i_i64[1] = v21;
  if ( v20 )
    v22 = sub_2021900(v20);
  else
    v22 = sub_1F58D40((__int64)&v110);
  v23 = v85 >> 2;
  v24 = *a1;
  DWORD2(v99) = 0;
  DWORD2(v100) = 0;
  *(_QWORD *)&v99 = 0;
  if ( v22 >> 3 != v88 )
    v23 = v88;
  *(_QWORD *)&v100 = 0;
  v86 = v23;
  v25 = (unsigned __int8 *)(*(_QWORD *)(v97.m128i_i64[0] + 40) + 16LL * v97.m128i_u32[2]);
  sub_1F40D10((__int64)&v110, v24, *(_QWORD *)(a1[1] + 48), *v25, *((_QWORD *)v25 + 1));
  if ( v110.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v97.m128i_u64[0], v97.m128i_i64[1], &v99, &v100);
  }
  else
  {
    v107.m128i_i8[0] = 0;
    v106.m128i_i8[0] = 0;
    v106.m128i_i64[1] = 0;
    v26 = (__int64 *)a1[1];
    v107.m128i_i64[1] = 0;
    v27 = *(_QWORD *)(v97.m128i_i64[0] + 40) + 16LL * v97.m128i_u32[2];
    v28 = *(_BYTE *)v27;
    v29 = *(_QWORD *)(v27 + 8);
    v108.m128i_i8[0] = v28;
    v108.m128i_i64[1] = v29;
    sub_1D19A30((__int64)&v110, (__int64)v26, &v108);
    v106 = _mm_loadu_si128(&v110);
    v107 = _mm_loadu_si128(&v111);
    sub_1D40600(
      (__int64)&v110,
      v26,
      (__int64)&v97,
      (__int64)&v95,
      (const void ***)&v106,
      (const void ***)&v107,
      v15,
      *(double *)v16.m128i_i64,
      v17);
    *(_QWORD *)&v99 = v110.m128i_i64[0];
    DWORD2(v99) = v110.m128i_i32[2];
    *(_QWORD *)&v100 = v111.m128i_i64[0];
    DWORD2(v100) = v111.m128i_i32[2];
  }
  v30 = *(_BYTE *)(a2 + 88);
  v31 = *(_QWORD *)(a2 + 96);
  v102.m128i_i8[0] = 0;
  v32 = a1[1];
  v102.m128i_i64[1] = 0;
  LOBYTE(v101[0]) = v30;
  v101[1] = v31;
  v103.m128i_i8[0] = 0;
  v103.m128i_i64[1] = 0;
  sub_1D19A30((__int64)&v110, v32, v101);
  v33 = _mm_loadu_si128(&v110);
  DWORD2(v104) = 0;
  v34 = _mm_loadu_si128(&v111);
  DWORD2(v105) = 0;
  v35 = (unsigned __int8 *)(*(_QWORD *)(v98.m128i_i64[0] + 40) + 16LL * v98.m128i_u32[2]);
  v36 = a1[1];
  v102 = v33;
  v103 = v34;
  v37 = *a1;
  v38 = *((_QWORD *)v35 + 1);
  *(_QWORD *)&v104 = 0;
  v39 = *(_QWORD *)(v36 + 48);
  *(_QWORD *)&v105 = 0;
  sub_1F40D10((__int64)&v110, v37, v39, *v35, v38);
  if ( v110.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v98.m128i_u64[0], v98.m128i_i64[1], &v104, &v105);
  }
  else
  {
    v107.m128i_i8[0] = 0;
    v106.m128i_i8[0] = 0;
    v106.m128i_i64[1] = 0;
    v40 = (__int64 *)a1[1];
    v107.m128i_i64[1] = 0;
    v41 = *(_QWORD *)(v98.m128i_i64[0] + 40) + 16LL * v98.m128i_u32[2];
    v42 = *(_BYTE *)v41;
    v43 = *(_QWORD *)(v41 + 8);
    v108.m128i_i8[0] = v42;
    v108.m128i_i64[1] = v43;
    sub_1D19A30((__int64)&v110, (__int64)v40, &v108);
    v15 = _mm_loadu_si128(&v110);
    v16 = _mm_loadu_si128(&v111);
    v106 = v15;
    v107 = v16;
    sub_1D40600(
      (__int64)&v110,
      v40,
      (__int64)&v98,
      (__int64)&v95,
      (const void ***)&v106,
      (const void ***)&v107,
      v15,
      *(double *)v16.m128i_i64,
      v17);
    *(_QWORD *)&v104 = v110.m128i_i64[0];
    DWORD2(v104) = v110.m128i_i32[2];
    *(_QWORD *)&v105 = v111.m128i_i64[0];
    DWORD2(v105) = v111.m128i_i32[2];
  }
  v44 = *(_QWORD *)(a2 + 104);
  v45 = *(_QWORD *)(a1[1] + 32);
  v46 = *(_QWORD *)(v44 + 64);
  v110 = _mm_loadu_si128((const __m128i *)(v44 + 40));
  v111.m128i_i64[0] = *(_QWORD *)(v44 + 56);
  if ( v102.m128i_i8[0] )
  {
    v47 = sub_2021900(v102.m128i_i8[0]);
  }
  else
  {
    v78 = v45;
    v79 = v46;
    v80 = v44;
    v76 = sub_1F58D40((__int64)&v102);
    v50 = v78;
    v49 = v79;
    v48 = v80;
    v47 = v76;
  }
  v51 = sub_1E0B8E0(
          v50,
          1u,
          (unsigned int)(v47 + 7) >> 3,
          v88,
          (int)&v110,
          v49,
          *(_OWORD *)v48,
          *(_QWORD *)(v48 + 16),
          1u,
          0,
          0);
  v52 = sub_1D257D0(
          (_QWORD *)a1[1],
          v10.m128i_i64[0],
          v10.m128i_i64[1],
          (__int64)&v95,
          v14,
          v90,
          v91,
          v99,
          v104,
          v102.m128i_i64[0],
          v102.m128i_i64[1],
          v51,
          v87,
          (*(_BYTE *)(a2 + 27) & 0x10) != 0);
  v53 = v99;
  *(_QWORD *)a3 = v52;
  v54 = DWORD2(v99);
  *(_DWORD *)(a3 + 8) = v55;
  *(_QWORD *)&v91 = sub_20BCE60(
                      *a1,
                      v91,
                      DWORD2(v91),
                      v53,
                      v54,
                      (unsigned int)&v95,
                      v102.m128i_i8[0],
                      v102.m128i_i64[1],
                      a1[1],
                      (*(_BYTE *)(a2 + 27) & 0x10) != 0);
  *((_QWORD *)&v91 + 1) = v56 | *((_QWORD *)&v91 + 1) & 0xFFFFFFFF00000000LL;
  if ( v102.m128i_i8[0] )
    v57 = sub_2021900(v102.m128i_i8[0]);
  else
    v57 = sub_1F58D40((__int64)&v102);
  v58 = *(_QWORD *)(a2 + 104);
  v59 = *(_QWORD *)(a1[1] + 32);
  v60 = *(_QWORD *)(v58 + 64);
  v108 = _mm_loadu_si128((const __m128i *)(v58 + 40));
  v109 = *(_QWORD *)(v58 + 56);
  if ( v103.m128i_i8[0] )
  {
    v61 = sub_2021900(v103.m128i_i8[0]);
  }
  else
  {
    v81 = v59;
    v82 = v60;
    v89 = v58;
    v75 = sub_1F58D40((__int64)&v103);
    v64 = v81;
    v63 = v82;
    v62 = v89;
    v61 = v75;
  }
  v65 = (unsigned int)(v61 + 7) >> 3;
  v66 = *(_QWORD *)v62 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v66 )
  {
    v72 = *(_BYTE *)(v62 + 16);
    v73 = *(_QWORD *)(v62 + 8) + ((unsigned int)(v57 + 7) >> 3);
    if ( (*(_QWORD *)v62 & 4) != 0 )
    {
      v110.m128i_i64[1] = v73;
      v111.m128i_i8[0] = v72;
      v110.m128i_i64[0] = v66 | 4;
      v111.m128i_i32[1] = *(_DWORD *)(v66 + 12);
    }
    else
    {
      v110.m128i_i64[0] = *(_QWORD *)v62 & 0xFFFFFFFFFFFFFFF8LL;
      v110.m128i_i64[1] = v73;
      v111.m128i_i8[0] = v72;
      v74 = *(_QWORD *)v66;
      if ( *(_BYTE *)(*(_QWORD *)v66 + 8LL) == 16 )
        v74 = **(_QWORD **)(v74 + 16);
      v111.m128i_i32[1] = *(_DWORD *)(v74 + 8) >> 8;
    }
  }
  else
  {
    v67 = *(_DWORD *)(v62 + 20);
    v111.m128i_i32[0] = 0;
    v110 = 0u;
    v111.m128i_i32[1] = v67;
  }
  v68 = sub_1E0B8E0(v64, 1u, v65, v86, (int)&v108, v63, *(_OWORD *)&v110, v111.m128i_i64[0], 1u, 0, 0);
  v94 = sub_1D257D0(
          (_QWORD *)a1[1],
          v11.m128i_i64[0],
          v11.m128i_i64[1],
          (__int64)&v95,
          v14,
          v90,
          v91,
          v100,
          v105,
          v103.m128i_i64[0],
          v103.m128i_i64[1],
          v68,
          v87,
          (*(_BYTE *)(a2 + 27) & 0x10) != 0);
  *(_QWORD *)a4 = v94;
  *(_DWORD *)(a4 + 8) = v69;
  *((_QWORD *)&v77 + 1) = 1;
  *(_QWORD *)&v77 = v94;
  v93 = sub_1D332F0(
          (__int64 *)a1[1],
          2,
          (__int64)&v95,
          1,
          0,
          0,
          *(double *)v15.m128i_i64,
          *(double *)v16.m128i_i64,
          v17,
          *(_QWORD *)a3,
          1u,
          v77);
  sub_2013400((__int64)a1, a2, 1, (__int64)v93, (__m128i *)(v70 | v90 & 0xFFFFFFFF00000000LL), v71);
  if ( v95 )
    sub_161E7C0((__int64)&v95, v95);
}
