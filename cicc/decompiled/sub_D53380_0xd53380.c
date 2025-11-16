// Function: sub_D53380
// Address: 0xd53380
//
__int64 __fastcall sub_D53380(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  __m128i v11; // xmm4
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rsi
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  __m128i v18; // xmm1
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // r10
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  __m128i v32; // xmm5
  __m128i v33; // xmm6
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // r11
  __int64 v38; // r9
  __int64 v39; // r8
  __int64 v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  __m128i v44; // xmm7
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __m128i v48; // xmm3
  __int64 v49; // rdi
  __int64 *v50; // rbx
  unsigned __int64 v51; // r13
  __int64 v52; // rdi
  __int64 v53; // rdi
  __int64 *v54; // rbx
  unsigned __int64 v55; // r13
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 *v58; // rbx
  unsigned __int64 v59; // r13
  __int64 v60; // rdi
  __int64 v61; // rdi
  __int64 *v62; // rbx
  unsigned __int64 v63; // r13
  __int64 v64; // rdi
  __int64 v66; // [rsp+10h] [rbp-330h] BYREF
  _QWORD *v67; // [rsp+18h] [rbp-328h]
  __int64 v68; // [rsp+20h] [rbp-320h]
  int v69; // [rsp+28h] [rbp-318h]
  char v70; // [rsp+2Ch] [rbp-314h]
  _QWORD v71[8]; // [rsp+30h] [rbp-310h] BYREF
  __m128i v72; // [rsp+70h] [rbp-2D0h] BYREF
  __m128i v73; // [rsp+80h] [rbp-2C0h] BYREF
  __m128i v74; // [rsp+90h] [rbp-2B0h] BYREF
  __m128i v75; // [rsp+A0h] [rbp-2A0h] BYREF
  __m128i v76; // [rsp+B0h] [rbp-290h] BYREF
  int v77; // [rsp+C0h] [rbp-280h]
  __int64 v78; // [rsp+D0h] [rbp-270h] BYREF
  _BYTE *v79; // [rsp+D8h] [rbp-268h]
  __int64 v80; // [rsp+E0h] [rbp-260h]
  int v81; // [rsp+E8h] [rbp-258h]
  char v82; // [rsp+ECh] [rbp-254h]
  _BYTE v83[64]; // [rsp+F0h] [rbp-250h] BYREF
  __m128i v84; // [rsp+130h] [rbp-210h] BYREF
  __m128i v85; // [rsp+140h] [rbp-200h] BYREF
  __m128i v86; // [rsp+150h] [rbp-1F0h] BYREF
  __m128i v87; // [rsp+160h] [rbp-1E0h] BYREF
  __m128i v88; // [rsp+170h] [rbp-1D0h] BYREF
  __int64 v89; // [rsp+180h] [rbp-1C0h]
  _BYTE v90[8]; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v91; // [rsp+198h] [rbp-1A8h]
  char v92; // [rsp+1ACh] [rbp-194h]
  _BYTE v93[64]; // [rsp+1B0h] [rbp-190h] BYREF
  __m128i v94; // [rsp+1F0h] [rbp-150h] BYREF
  __m128i v95; // [rsp+200h] [rbp-140h] BYREF
  __m128i v96; // [rsp+210h] [rbp-130h] BYREF
  __m128i v97; // [rsp+220h] [rbp-120h] BYREF
  __m128i v98; // [rsp+230h] [rbp-110h] BYREF
  int v99; // [rsp+240h] [rbp-100h]
  __m128i v100; // [rsp+250h] [rbp-F0h] BYREF
  char v101; // [rsp+260h] [rbp-E0h]
  char v102; // [rsp+268h] [rbp-D8h]
  char v103; // [rsp+26Ch] [rbp-D4h]
  _BYTE v104[64]; // [rsp+270h] [rbp-D0h] BYREF
  __m128i v105; // [rsp+2B0h] [rbp-90h] BYREF
  __m128i v106; // [rsp+2C0h] [rbp-80h] BYREF
  __m128i v107; // [rsp+2D0h] [rbp-70h] BYREF
  __m128i v108; // [rsp+2E0h] [rbp-60h] BYREF
  __m128i v109; // [rsp+2F0h] [rbp-50h] BYREF
  int v110; // [rsp+300h] [rbp-40h]

  v79 = v83;
  v89 = 0;
  v78 = 0;
  v80 = 8;
  v81 = 0;
  v82 = 1;
  v84 = 0u;
  v85 = 0u;
  v86 = 0u;
  v87 = 0u;
  v88 = 0u;
  sub_D53100(v84.m128i_i64, 0, a3);
  v4 = *a2;
  LODWORD(v89) = 0;
  v66 = 0;
  v67 = v71;
  v68 = 8;
  v69 = 0;
  v70 = 1;
  v72 = 0u;
  v73 = 0u;
  v74 = 0u;
  v75 = 0u;
  v76 = 0u;
  sub_D53100(v72.m128i_i64, 0, v5);
  v77 = 0;
  ++HIDWORD(v68);
  v71[0] = v4;
  ++v66;
  v100.m128i_i64[0] = v4;
  v101 = 0;
  v77 = 0;
  v102 = 1;
  sub_D52090(v72.m128i_i64, &v100);
  v102 = 0;
  sub_D52090(v72.m128i_i64, &v100);
  sub_C8CF70((__int64)&v100, v104, 8, (__int64)v83, (__int64)&v78);
  v105 = 0u;
  v106 = 0u;
  v107 = 0u;
  v108 = 0u;
  v109 = 0u;
  sub_D53100(v105.m128i_i64, 0, v6);
  if ( v84.m128i_i64[0] )
  {
    v7 = _mm_loadu_si128(&v84);
    v8 = _mm_loadu_si128(&v85);
    v84 = v105;
    v9 = _mm_loadu_si128(&v86);
    v10 = _mm_loadu_si128(&v87);
    v11 = _mm_loadu_si128(&v88);
    v85 = v106;
    v86 = v107;
    v87 = v108;
    v88 = v109;
    v105 = v7;
    v106 = v8;
    v107 = v9;
    v108 = v10;
    v109 = v11;
  }
  v110 = v89;
  sub_C8CF70((__int64)v90, v93, 8, (__int64)v71, (__int64)&v66);
  v94 = 0u;
  v95 = 0u;
  v96 = 0u;
  v97 = 0u;
  v98 = 0u;
  sub_D53100(v94.m128i_i64, 0, v12);
  if ( v72.m128i_i64[0] )
  {
    v13 = _mm_loadu_si128(&v75);
    v75.m128i_i64[1] = v97.m128i_i64[1];
    v14 = v97.m128i_i64[0];
    v97 = v13;
    v15 = _mm_loadu_si128(&v72);
    v72 = v94;
    v16 = _mm_loadu_si128(&v73);
    v17 = _mm_loadu_si128(&v74);
    v18 = _mm_loadu_si128(&v76);
    v73 = v95;
    v74 = v96;
    v75.m128i_i64[0] = v14;
    v76 = v98;
    v94 = v15;
    v95 = v16;
    v96 = v17;
    v98 = v18;
  }
  v99 = v77;
  sub_C8CF70(a1, (void *)(a1 + 32), 8, (__int64)v93, (__int64)v90);
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  sub_D53100((__int64 *)(a1 + 96), 0, v19);
  if ( v94.m128i_i64[0] )
  {
    v20 = *(_QWORD *)(a1 + 104);
    v21 = *(_QWORD *)(a1 + 112);
    v22 = *(_QWORD *)(a1 + 120);
    v23 = *(_QWORD *)(a1 + 128);
    v24 = *(_QWORD *)(a1 + 136);
    v25 = *(_QWORD *)(a1 + 144);
    v26 = *(_QWORD *)(a1 + 152);
    v27 = *(_QWORD *)(a1 + 160);
    v28 = *(_QWORD *)(a1 + 168);
    v29 = _mm_loadu_si128(&v94);
    v30 = _mm_loadu_si128(&v95);
    v94.m128i_i64[0] = *(_QWORD *)(a1 + 96);
    v31 = _mm_loadu_si128(&v96);
    v32 = _mm_loadu_si128(&v97);
    v94.m128i_i64[1] = v20;
    v33 = _mm_loadu_si128(&v98);
    v95.m128i_i64[0] = v21;
    v95.m128i_i64[1] = v22;
    v96.m128i_i64[0] = v23;
    v96.m128i_i64[1] = v24;
    v97.m128i_i64[0] = v25;
    v97.m128i_i64[1] = v26;
    v98.m128i_i64[0] = v27;
    v98.m128i_i64[1] = v28;
    *(__m128i *)(a1 + 96) = v29;
    *(__m128i *)(a1 + 112) = v30;
    *(__m128i *)(a1 + 128) = v31;
    *(__m128i *)(a1 + 144) = v32;
    *(__m128i *)(a1 + 160) = v33;
  }
  *(_DWORD *)(a1 + 176) = v99;
  sub_C8CF70(a1 + 184, (void *)(a1 + 216), 8, (__int64)v104, (__int64)&v100);
  v34 = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  sub_D53100((__int64 *)(a1 + 280), 0, v35);
  if ( v105.m128i_i64[0] )
  {
    v36 = *(_QWORD *)(a1 + 280);
    v37 = *(_QWORD *)(a1 + 288);
    v38 = *(_QWORD *)(a1 + 304);
    v39 = *(_QWORD *)(a1 + 312);
    v40 = *(_QWORD *)(a1 + 320);
    v34 = *(_QWORD *)(a1 + 328);
    v41 = *(_QWORD *)(a1 + 336);
    v42 = *(_QWORD *)(a1 + 344);
    v43 = *(_QWORD *)(a1 + 352);
    v44 = _mm_loadu_si128(&v105);
    v45 = _mm_loadu_si128(&v106);
    v106.m128i_i64[0] = *(_QWORD *)(a1 + 296);
    v46 = _mm_loadu_si128(&v107);
    v47 = _mm_loadu_si128(&v108);
    v105.m128i_i64[0] = v36;
    v48 = _mm_loadu_si128(&v109);
    v105.m128i_i64[1] = v37;
    v106.m128i_i64[1] = v38;
    v107.m128i_i64[0] = v39;
    v107.m128i_i64[1] = v40;
    v108.m128i_i64[0] = v34;
    v108.m128i_i64[1] = v41;
    v109.m128i_i64[0] = v42;
    v109.m128i_i64[1] = v43;
    *(__m128i *)(a1 + 280) = v44;
    *(__m128i *)(a1 + 296) = v45;
    *(__m128i *)(a1 + 312) = v46;
    *(__m128i *)(a1 + 328) = v47;
    *(__m128i *)(a1 + 344) = v48;
  }
  v49 = v94.m128i_i64[0];
  *(_DWORD *)(a1 + 360) = v110;
  if ( v49 )
  {
    v50 = (__int64 *)v96.m128i_i64[1];
    v51 = v98.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v98.m128i_i64[1] + 8) > v96.m128i_i64[1] )
    {
      do
      {
        v52 = *v50++;
        j_j___libc_free_0(v52, 512);
      }
      while ( v51 > (unsigned __int64)v50 );
      v49 = v94.m128i_i64[0];
    }
    v34 = 8 * v94.m128i_i64[1];
    j_j___libc_free_0(v49, 8 * v94.m128i_i64[1]);
  }
  if ( !v92 )
    _libc_free(v91, v34);
  v53 = v105.m128i_i64[0];
  if ( v105.m128i_i64[0] )
  {
    v54 = (__int64 *)v107.m128i_i64[1];
    v55 = v109.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v109.m128i_i64[1] + 8) > v107.m128i_i64[1] )
    {
      do
      {
        v56 = *v54++;
        j_j___libc_free_0(v56, 512);
      }
      while ( v55 > (unsigned __int64)v54 );
      v53 = v105.m128i_i64[0];
    }
    v34 = 8 * v105.m128i_i64[1];
    j_j___libc_free_0(v53, 8 * v105.m128i_i64[1]);
  }
  if ( !v103 )
    _libc_free(v100.m128i_i64[1], v34);
  v57 = v72.m128i_i64[0];
  if ( v72.m128i_i64[0] )
  {
    v58 = (__int64 *)v74.m128i_i64[1];
    v59 = v76.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v76.m128i_i64[1] + 8) > v74.m128i_i64[1] )
    {
      do
      {
        v60 = *v58++;
        j_j___libc_free_0(v60, 512);
      }
      while ( v59 > (unsigned __int64)v58 );
      v57 = v72.m128i_i64[0];
    }
    v34 = 8 * v72.m128i_i64[1];
    j_j___libc_free_0(v57, 8 * v72.m128i_i64[1]);
  }
  if ( !v70 )
    _libc_free(v67, v34);
  v61 = v84.m128i_i64[0];
  if ( v84.m128i_i64[0] )
  {
    v62 = (__int64 *)v86.m128i_i64[1];
    v63 = v88.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v88.m128i_i64[1] + 8) > v86.m128i_i64[1] )
    {
      do
      {
        v64 = *v62++;
        j_j___libc_free_0(v64, 512);
      }
      while ( v63 > (unsigned __int64)v62 );
      v61 = v84.m128i_i64[0];
    }
    v34 = 8 * v84.m128i_i64[1];
    j_j___libc_free_0(v61, 8 * v84.m128i_i64[1]);
  }
  if ( !v82 )
    _libc_free(v79, v34);
  return a1;
}
