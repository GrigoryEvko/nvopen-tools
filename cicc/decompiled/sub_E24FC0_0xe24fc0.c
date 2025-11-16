// Function: sub_E24FC0
// Address: 0xe24fc0
//
__int64 __fastcall sub_E24FC0(__int64 a1, size_t *a2, char a3)
{
  __m128i v4; // xmm7
  __m128i v5; // xmm10
  __m128i v6; // xmm9
  __m128i v7; // xmm6
  __m128i v8; // xmm8
  __m128i v9; // xmm5
  __m128i v10; // xmm4
  __m128i v11; // xmm3
  __m128i v12; // xmm2
  __m128i v13; // xmm1
  __m128i v14; // xmm0
  __m128i v15; // xmm11
  __m128i v16; // xmm12
  __m128i v17; // xmm13
  __m128i v18; // xmm14
  __m128i v19; // xmm15
  __m128i v20; // xmm11
  __m128i v21; // xmm12
  __m128i v22; // xmm13
  __m128i v23; // xmm14
  __m128i v24; // xmm15
  __m128i v25; // xmm11
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __m128i v30; // xmm1
  __m128i v31; // xmm2
  __m128i v32; // xmm3
  __m128i v33; // xmm4
  __m128i v34; // xmm5
  __m128i v35; // xmm6
  __m128i v36; // xmm7
  __m128i v37; // xmm0
  __m128i v38; // xmm1
  __m128i v39; // xmm2
  __int64 v40; // r14
  unsigned __int64 v42; // rax
  __m128i v43; // xmm7
  __m128i v44; // xmm10
  __m128i v45; // xmm9
  __m128i v46; // xmm6
  __m128i v47; // xmm8
  __m128i v48; // xmm5
  __m128i v49; // xmm4
  __m128i v50; // xmm3
  __m128i v51; // xmm2
  __m128i v52; // xmm0
  __m128i v53; // xmm1
  bool v54; // zf
  __m128i v55; // [rsp+0h] [rbp-180h] BYREF
  __m128i v56; // [rsp+10h] [rbp-170h] BYREF
  __m128i v57; // [rsp+20h] [rbp-160h] BYREF
  __m128i v58; // [rsp+30h] [rbp-150h] BYREF
  __m128i v59; // [rsp+40h] [rbp-140h] BYREF
  __m128i v60; // [rsp+50h] [rbp-130h] BYREF
  __m128i v61; // [rsp+60h] [rbp-120h] BYREF
  __m128i v62; // [rsp+70h] [rbp-110h] BYREF
  __m128i v63; // [rsp+80h] [rbp-100h] BYREF
  __m128i v64; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v65; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v66; // [rsp+B0h] [rbp-D0h]
  __m128i v67; // [rsp+C0h] [rbp-C0h]
  __m128i v68; // [rsp+D0h] [rbp-B0h]
  __m128i v69; // [rsp+E0h] [rbp-A0h]
  __m128i v70; // [rsp+F0h] [rbp-90h]
  __m128i v71; // [rsp+100h] [rbp-80h]
  __m128i v72; // [rsp+110h] [rbp-70h]
  __m128i v73; // [rsp+120h] [rbp-60h]
  __m128i v74; // [rsp+130h] [rbp-50h]
  __m128i v75; // [rsp+140h] [rbp-40h]
  __m128i v76; // [rsp+150h] [rbp-30h]

  sub_E20730(a2, 2u, "?$");
  v4 = _mm_loadu_si128(&v58);
  v5 = _mm_loadu_si128(&v55);
  v6 = _mm_loadu_si128(&v56);
  v7 = _mm_loadu_si128(&v59);
  v60.m128i_i64[0] = 0;
  v8 = _mm_loadu_si128(&v57);
  v9 = _mm_loadu_si128(&v60);
  v65.m128i_i64[1] = 0;
  v10 = _mm_loadu_si128(&v61);
  v11 = _mm_loadu_si128(&v62);
  v66 = v5;
  v12 = _mm_loadu_si128(&v63);
  v13 = _mm_loadu_si128(&v64);
  v67 = v6;
  v14 = _mm_loadu_si128(&v65);
  v15 = _mm_loadu_si128((const __m128i *)(a1 + 24));
  v68 = v8;
  v16 = _mm_loadu_si128((const __m128i *)(a1 + 40));
  v17 = _mm_loadu_si128((const __m128i *)(a1 + 56));
  v69 = v4;
  v18 = _mm_loadu_si128((const __m128i *)(a1 + 72));
  v70 = v7;
  v71 = v9;
  v72 = v10;
  v73 = v11;
  v74 = v12;
  v75 = v13;
  v76 = v14;
  v55 = v15;
  v56 = v16;
  v57 = v17;
  v58 = v18;
  v19 = _mm_loadu_si128((const __m128i *)(a1 + 88));
  v20 = _mm_loadu_si128((const __m128i *)(a1 + 104));
  v21 = _mm_loadu_si128((const __m128i *)(a1 + 120));
  *(__m128i *)(a1 + 24) = v5;
  v22 = _mm_loadu_si128((const __m128i *)(a1 + 136));
  v23 = _mm_loadu_si128((const __m128i *)(a1 + 152));
  v59 = v19;
  v24 = _mm_loadu_si128((const __m128i *)(a1 + 168));
  v60 = v20;
  v25 = _mm_loadu_si128((const __m128i *)(a1 + 184));
  *(__m128i *)(a1 + 40) = v6;
  *(__m128i *)(a1 + 56) = v8;
  *(__m128i *)(a1 + 72) = v4;
  *(__m128i *)(a1 + 88) = v7;
  *(__m128i *)(a1 + 104) = v9;
  *(__m128i *)(a1 + 120) = v10;
  *(__m128i *)(a1 + 136) = v11;
  *(__m128i *)(a1 + 152) = v12;
  *(__m128i *)(a1 + 168) = v13;
  *(__m128i *)(a1 + 184) = v14;
  v29 = sub_E25570(
          a1,
          a2,
          2,
          v26,
          v27,
          v28,
          v55.m128i_i64[0],
          v55.m128i_i64[1],
          v56.m128i_i64[0],
          v56.m128i_i64[1],
          v57.m128i_i64[0],
          v57.m128i_i64[1],
          v58.m128i_i64[0],
          v58.m128i_i64[1],
          v59.m128i_i64[0],
          v59.m128i_i64[1],
          v60.m128i_i64[0],
          v60.m128i_i64[1],
          v21.m128i_i64[0],
          v21.m128i_i64[1],
          v22.m128i_i64[0],
          v22.m128i_i64[1],
          v23.m128i_i64[0],
          v23.m128i_i64[1],
          v24.m128i_i64[0],
          v24.m128i_i64[1],
          v25.m128i_i64[0],
          v25.m128i_i64[1],
          v66.m128i_i64[0],
          v66.m128i_i64[1],
          v67.m128i_i64[0],
          v67.m128i_i64[1],
          v68.m128i_i64[0],
          v68.m128i_i64[1],
          v69.m128i_i64[0],
          v69.m128i_i64[1],
          v70.m128i_i64[0],
          v70.m128i_i64[1],
          v71.m128i_i64[0],
          v71.m128i_i64[1],
          v72.m128i_i64[0],
          v72.m128i_i64[1],
          v73.m128i_i64[0],
          v73.m128i_i64[1],
          v74.m128i_i64[0],
          v74.m128i_i64[1],
          v75.m128i_i64[0],
          v75.m128i_i64[1],
          v76.m128i_i64[0],
          v76.m128i_i64[1]);
  if ( *(_BYTE *)(a1 + 8) )
  {
    v30 = _mm_loadu_si128(&v56);
    v31 = _mm_loadu_si128(&v57);
    v32 = _mm_loadu_si128(&v58);
    v33 = _mm_loadu_si128(&v59);
    v34 = _mm_loadu_si128(&v60);
    *(__m128i *)(a1 + 24) = _mm_loadu_si128(&v55);
    v35 = _mm_loadu_si128(&v61);
    v36 = _mm_loadu_si128(&v62);
    *(__m128i *)(a1 + 40) = v30;
    v37 = _mm_loadu_si128(&v63);
    v38 = _mm_loadu_si128(&v64);
    *(__m128i *)(a1 + 56) = v31;
    v39 = _mm_loadu_si128(&v65);
    *(__m128i *)(a1 + 72) = v32;
    *(__m128i *)(a1 + 88) = v33;
    *(__m128i *)(a1 + 104) = v34;
    *(__m128i *)(a1 + 120) = v35;
    *(__m128i *)(a1 + 136) = v36;
    *(__m128i *)(a1 + 152) = v37;
    *(__m128i *)(a1 + 168) = v38;
    *(__m128i *)(a1 + 184) = v39;
    return 0;
  }
  v40 = v29;
  v42 = sub_E246E0(a1, a2);
  v43 = _mm_loadu_si128(&v58);
  v44 = _mm_loadu_si128(&v55);
  v45 = _mm_loadu_si128(&v56);
  v46 = _mm_loadu_si128(&v59);
  *(_QWORD *)(v40 + 16) = v42;
  v47 = _mm_loadu_si128(&v57);
  v48 = _mm_loadu_si128(&v60);
  v66 = v44;
  v49 = _mm_loadu_si128(&v61);
  v50 = _mm_loadu_si128(&v62);
  v67 = v45;
  v51 = _mm_loadu_si128(&v63);
  v52 = _mm_loadu_si128(&v65);
  v68 = v47;
  v53 = _mm_loadu_si128(&v64);
  v69 = v43;
  v70 = v46;
  v71 = v48;
  v72 = v49;
  v73 = v50;
  v74 = v51;
  v75 = v53;
  v76 = v52;
  *(__m128i *)(a1 + 24) = v44;
  *(__m128i *)(a1 + 40) = v45;
  *(__m128i *)(a1 + 56) = v47;
  *(__m128i *)(a1 + 72) = v43;
  *(__m128i *)(a1 + 88) = v46;
  *(__m128i *)(a1 + 104) = v48;
  *(__m128i *)(a1 + 120) = v49;
  *(__m128i *)(a1 + 136) = v50;
  *(__m128i *)(a1 + 152) = v51;
  *(__m128i *)(a1 + 168) = v53;
  v54 = *(_BYTE *)(a1 + 8) == 0;
  *(__m128i *)(a1 + 184) = v52;
  if ( !v54 )
    return 0;
  if ( (a3 & 1) != 0 )
  {
    if ( (*(_DWORD *)(v40 + 8) & 0xFFFFFFFD) == 9 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
    }
    else
    {
      sub_E21CC0(a1, (__int64 *)v40);
    }
  }
  return v40;
}
