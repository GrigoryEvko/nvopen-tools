// Function: sub_202BDB0
// Address: 0x202bdb0
//
__int64 *__fastcall sub_202BDB0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rsi
  __int64 v7; // rdi
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __int64 v11; // rsi
  __int64 v12; // rax
  __int16 v13; // cx
  unsigned int v14; // r15d
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rsi
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  unsigned __int8 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 *v24; // rsi
  __int64 v25; // rax
  __int8 v26; // dl
  __int64 v27; // rsi
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 *v31; // rsi
  __int64 v32; // rax
  __int8 v33; // dl
  __int64 v34; // rax
  __int64 v35; // rax
  char v36; // di
  __int64 v37; // rax
  unsigned int v38; // eax
  __int64 v39; // rcx
  int v40; // r15d
  __m128i v41; // xmm5
  __int64 v42; // r9
  __int64 v43; // r10
  int v44; // edx
  __int64 v45; // rcx
  int v46; // r9d
  __int64 v47; // r10
  __int64 v48; // rax
  unsigned int v49; // edx
  unsigned int v50; // edx
  int v51; // r13d
  __int64 v52; // rcx
  __int64 v53; // r10
  __int64 v54; // r9
  int v55; // edx
  __int64 v56; // rcx
  int v57; // r9d
  __int64 v58; // r10
  unsigned int v59; // edx
  unsigned __int64 v60; // rdi
  __int32 v61; // eax
  __int64 v62; // rax
  unsigned int v63; // edx
  __int64 *v64; // r12
  __int8 v66; // r8
  __int64 v67; // r13
  __int64 v68; // rax
  int v69; // eax
  int v70; // eax
  __int128 v71; // [rsp-10h] [rbp-1C0h]
  __int64 v72; // [rsp+0h] [rbp-1B0h]
  __int64 v73; // [rsp+0h] [rbp-1B0h]
  int v74; // [rsp+8h] [rbp-1A8h]
  int v75; // [rsp+8h] [rbp-1A8h]
  __int64 v76; // [rsp+10h] [rbp-1A0h]
  __int64 v77; // [rsp+10h] [rbp-1A0h]
  __int64 v78; // [rsp+20h] [rbp-190h]
  __int64 v79; // [rsp+28h] [rbp-188h]
  unsigned int v80; // [rsp+30h] [rbp-180h]
  __int64 v81; // [rsp+30h] [rbp-180h]
  unsigned __int64 v82; // [rsp+38h] [rbp-178h]
  __int128 v83; // [rsp+40h] [rbp-170h]
  __int64 v85; // [rsp+50h] [rbp-160h]
  __m128i v86; // [rsp+80h] [rbp-130h] BYREF
  __m128i v87; // [rsp+90h] [rbp-120h] BYREF
  _QWORD v88[2]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v89; // [rsp+B0h] [rbp-100h] BYREF
  int v90; // [rsp+B8h] [rbp-F8h]
  __m128i v91; // [rsp+C0h] [rbp-F0h] BYREF
  __m128i v92; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v93; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v94; // [rsp+E8h] [rbp-C8h]
  __int64 v95; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v96; // [rsp+F8h] [rbp-B8h]
  __int128 v97; // [rsp+100h] [rbp-B0h] BYREF
  __int128 v98; // [rsp+110h] [rbp-A0h] BYREF
  __m128i v99; // [rsp+120h] [rbp-90h] BYREF
  __m128i v100; // [rsp+130h] [rbp-80h] BYREF
  __m128i v101; // [rsp+140h] [rbp-70h] BYREF
  __int64 v102; // [rsp+150h] [rbp-60h]
  __m128i v103; // [rsp+160h] [rbp-50h] BYREF
  __m128i v104; // [rsp+170h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 32);
  v5 = *(_BYTE *)(a2 + 88);
  v6 = *(_QWORD *)v4;
  v7 = *(_QWORD *)(v4 + 8);
  LOBYTE(v88[0]) = v5;
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v9 = _mm_loadu_si128((const __m128i *)(v4 + 80));
  v10 = _mm_loadu_si128((const __m128i *)(v4 + 120));
  v79 = v6;
  v11 = *(_QWORD *)(a2 + 72);
  v78 = v7;
  v88[1] = *(_QWORD *)(a2 + 96);
  v12 = *(_QWORD *)(a2 + 104);
  v89 = v11;
  v13 = *(_WORD *)(v12 + 34);
  v83 = (__int128)v8;
  v86 = v9;
  v14 = 1 << v13;
  v87 = v10;
  v80 = (unsigned int)(1 << v13) >> 1;
  if ( v11 )
    sub_1623A60((__int64)&v89, v11, 2);
  v15 = *(_DWORD *)(a2 + 64);
  v16 = a1[1];
  v91.m128i_i8[0] = 0;
  v90 = v15;
  v91.m128i_i64[1] = 0;
  v92.m128i_i8[0] = 0;
  v92.m128i_i64[1] = 0;
  sub_1D19A30((__int64)&v103, v16, v88);
  v17 = *a1;
  v18 = _mm_loadu_si128(&v103);
  LODWORD(v94) = 0;
  v19 = _mm_loadu_si128(&v104);
  LODWORD(v96) = 0;
  v20 = (unsigned __int8 *)(*(_QWORD *)(v87.m128i_i64[0] + 40) + 16LL * v87.m128i_u32[2]);
  v21 = a1[1];
  v91 = v18;
  v92 = v19;
  v22 = *((_QWORD *)v20 + 1);
  v93 = 0;
  v23 = *(_QWORD *)(v21 + 48);
  v95 = 0;
  sub_1F40D10((__int64)&v103, v17, v23, *v20, v22);
  if ( v103.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v87.m128i_u64[0], v87.m128i_i64[1], &v93, &v95);
  }
  else
  {
    v100.m128i_i8[0] = 0;
    v99.m128i_i8[0] = 0;
    v99.m128i_i64[1] = 0;
    v24 = (__int64 *)a1[1];
    v100.m128i_i64[1] = 0;
    v25 = *(_QWORD *)(v87.m128i_i64[0] + 40) + 16LL * v87.m128i_u32[2];
    v26 = *(_BYTE *)v25;
    v101.m128i_i64[1] = *(_QWORD *)(v25 + 8);
    v101.m128i_i8[0] = v26;
    sub_1D19A30((__int64)&v103, (__int64)v24, &v101);
    v99 = _mm_loadu_si128(&v103);
    v100 = _mm_loadu_si128(&v104);
    sub_1D40600(
      (__int64)&v103,
      v24,
      (__int64)&v87,
      (__int64)&v89,
      (const void ***)&v99,
      (const void ***)&v100,
      v8,
      *(double *)v9.m128i_i64,
      v10);
    v93 = v103.m128i_i64[0];
    LODWORD(v94) = v103.m128i_i32[2];
    v95 = v104.m128i_i64[0];
    LODWORD(v96) = v104.m128i_i32[2];
  }
  v27 = *a1;
  DWORD2(v97) = 0;
  DWORD2(v98) = 0;
  *(_QWORD *)&v97 = 0;
  v28 = (unsigned __int8 *)(*(_QWORD *)(v86.m128i_i64[0] + 40) + 16LL * v86.m128i_u32[2]);
  v29 = a1[1];
  v30 = *((_QWORD *)v28 + 1);
  *(_QWORD *)&v98 = 0;
  sub_1F40D10((__int64)&v103, v27, *(_QWORD *)(v29 + 48), *v28, v30);
  if ( v103.m128i_i8[0] == 6 )
  {
    sub_2017DE0((__int64)a1, v86.m128i_u64[0], v86.m128i_i64[1], &v97, &v98);
  }
  else
  {
    v100.m128i_i8[0] = 0;
    v99.m128i_i8[0] = 0;
    v99.m128i_i64[1] = 0;
    v31 = (__int64 *)a1[1];
    v100.m128i_i64[1] = 0;
    v32 = *(_QWORD *)(v86.m128i_i64[0] + 40) + 16LL * v86.m128i_u32[2];
    v33 = *(_BYTE *)v32;
    v34 = *(_QWORD *)(v32 + 8);
    v101.m128i_i8[0] = v33;
    v101.m128i_i64[1] = v34;
    sub_1D19A30((__int64)&v103, (__int64)v31, &v101);
    v8 = _mm_loadu_si128(&v103);
    v9 = _mm_loadu_si128(&v104);
    v99 = v8;
    v100 = v9;
    sub_1D40600(
      (__int64)&v103,
      v31,
      (__int64)&v86,
      (__int64)&v89,
      (const void ***)&v99,
      (const void ***)&v100,
      v8,
      *(double *)v9.m128i_i64,
      v10);
    *(_QWORD *)&v97 = v103.m128i_i64[0];
    DWORD2(v97) = v103.m128i_i32[2];
    *(_QWORD *)&v98 = v104.m128i_i64[0];
    DWORD2(v98) = v104.m128i_i32[2];
  }
  v35 = *(_QWORD *)(v87.m128i_i64[0] + 40);
  v36 = *(_BYTE *)v35;
  v37 = *(_QWORD *)(v35 + 8);
  v103.m128i_i8[0] = v36;
  v103.m128i_i64[1] = v37;
  if ( v36 )
    v38 = sub_2021900(v36);
  else
    v38 = sub_1F58D40((__int64)&v103);
  v39 = *(_QWORD *)(a2 + 104);
  v40 = v14 >> 2;
  v41 = _mm_loadu_si128((const __m128i *)(v39 + 40));
  v42 = *(_QWORD *)(v39 + 64);
  if ( v38 >> 3 != v80 )
    v40 = v80;
  v43 = *(_QWORD *)(a1[1] + 32);
  v103 = v41;
  v104.m128i_i64[0] = *(_QWORD *)(v39 + 56);
  if ( v91.m128i_i8[0] )
  {
    v44 = sub_2021900(v91.m128i_i8[0]);
  }
  else
  {
    v73 = v43;
    v75 = v42;
    v77 = v39;
    v70 = sub_1F58D40((__int64)&v91);
    v47 = v73;
    v46 = v75;
    v45 = v77;
    v44 = v70;
  }
  v48 = sub_1E0B8E0(
          v47,
          2u,
          (unsigned int)(v44 + 7) >> 3,
          v80,
          (int)&v103,
          v46,
          *(_OWORD *)v45,
          *(_QWORD *)(v45 + 16),
          1u,
          0,
          0);
  v81 = sub_1D2C870(
          (_QWORD *)a1[1],
          v79,
          v78,
          (__int64)&v89,
          v93,
          v94,
          v83,
          v97,
          v91.m128i_i64[0],
          v91.m128i_i64[1],
          v48,
          (*(_BYTE *)(a2 + 27) & 4) != 0,
          (*(_BYTE *)(a2 + 27) & 8) != 0);
  v82 = v49;
  *(_QWORD *)&v83 = sub_20BCE60(
                      *a1,
                      v83,
                      DWORD2(v83),
                      v97,
                      DWORD2(v97),
                      (unsigned int)&v89,
                      v91.m128i_i8[0],
                      v91.m128i_i64[1],
                      a1[1],
                      (*(_BYTE *)(a2 + 27) & 8) != 0);
  *((_QWORD *)&v83 + 1) = v50 | *((_QWORD *)&v83 + 1) & 0xFFFFFFFF00000000LL;
  if ( v91.m128i_i8[0] )
    v51 = sub_2021900(v91.m128i_i8[0]);
  else
    v51 = sub_1F58D40((__int64)&v91);
  v52 = *(_QWORD *)(a2 + 104);
  v53 = *(_QWORD *)(a1[1] + 32);
  v54 = *(_QWORD *)(v52 + 64);
  v101 = _mm_loadu_si128((const __m128i *)(v52 + 40));
  v102 = *(_QWORD *)(v52 + 56);
  if ( v92.m128i_i8[0] )
  {
    v55 = sub_2021900(v92.m128i_i8[0]);
  }
  else
  {
    v72 = v53;
    v74 = v54;
    v76 = v52;
    v69 = sub_1F58D40((__int64)&v92);
    v58 = v72;
    v57 = v74;
    v56 = v76;
    v55 = v69;
  }
  v59 = (unsigned int)(v55 + 7) >> 3;
  v60 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v60 )
  {
    v66 = *(_BYTE *)(v56 + 16);
    v67 = *(_QWORD *)(v56 + 8) + ((unsigned int)(v51 + 7) >> 3);
    if ( (*(_QWORD *)v56 & 4) != 0 )
    {
      v103.m128i_i64[1] = v67;
      v104.m128i_i8[0] = v66;
      v103.m128i_i64[0] = v60 | 4;
      v104.m128i_i32[1] = *(_DWORD *)(v60 + 12);
    }
    else
    {
      v103.m128i_i64[0] = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
      v103.m128i_i64[1] = v67;
      v104.m128i_i8[0] = v66;
      v68 = *(_QWORD *)v60;
      if ( *(_BYTE *)(*(_QWORD *)v60 + 8LL) == 16 )
        v68 = **(_QWORD **)(v68 + 16);
      v104.m128i_i32[1] = *(_DWORD *)(v68 + 8) >> 8;
    }
  }
  else
  {
    v61 = *(_DWORD *)(v56 + 20);
    v104.m128i_i32[0] = 0;
    v103 = 0u;
    v104.m128i_i32[1] = v61;
  }
  v62 = sub_1E0B8E0(v58, 2u, v59, v40, (int)&v101, v57, *(_OWORD *)&v103, v104.m128i_i64[0], 1u, 0, 0);
  v85 = sub_1D2C870(
          (_QWORD *)a1[1],
          v79,
          v78,
          (__int64)&v89,
          v95,
          v96,
          v83,
          v98,
          v92.m128i_i64[0],
          v92.m128i_i64[1],
          v62,
          (*(_BYTE *)(a2 + 27) & 4) != 0,
          (*(_BYTE *)(a2 + 27) & 8) != 0);
  *((_QWORD *)&v71 + 1) = v63;
  *(_QWORD *)&v71 = v85;
  v64 = sub_1D332F0(
          (__int64 *)a1[1],
          2,
          (__int64)&v89,
          1,
          0,
          0,
          *(double *)v8.m128i_i64,
          *(double *)v9.m128i_i64,
          v10,
          v81,
          v82,
          v71);
  if ( v89 )
    sub_161E7C0((__int64)&v89, v89);
  return v64;
}
