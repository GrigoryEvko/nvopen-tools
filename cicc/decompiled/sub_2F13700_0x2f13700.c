// Function: sub_2F13700
// Address: 0x2f13700
//
unsigned __int64 *__fastcall sub_2F13700(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  __m128i v10; // xmm6
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __m128i v13; // xmm6
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __m128i v17; // xmm0
  __m128i v18; // xmm7
  _BYTE *v19; // rsi
  __int64 v20; // rdx
  __m128i v21; // xmm6
  _BYTE *v22; // rsi
  __int64 v23; // rdx
  __m128i v24; // xmm0
  _BYTE *v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // r15
  unsigned __int64 v28; // r14
  unsigned __int64 v29; // r13
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rbx
  __m128i v32; // xmm2
  __int64 v33; // rcx
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  __int64 v36; // rcx
  __m128i v37; // xmm5
  __int64 v38; // rcx
  __m128i v39; // xmm6
  __int64 v40; // rcx
  __m128i v41; // xmm7
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // r9
  __m128i v56; // xmm2
  __int64 v57; // r9
  __int64 v58; // r9
  __int64 v59; // r9
  __m128i v60; // xmm3
  __m128i v61; // xmm4
  __int64 v62; // r9
  __int64 v63; // r9
  __m128i v64; // xmm5
  __int64 v65; // r9
  __int64 v66; // r9
  __m128i v67; // xmm6
  __int64 v68; // r9
  __int64 v69; // r9
  __m128i v70; // xmm0
  __m128i v71; // xmm1
  __int64 v72; // r9
  unsigned __int64 v74; // rbx
  unsigned __int64 v75; // [rsp+0h] [rbp-60h]
  unsigned __int64 v76; // [rsp+8h] [rbp-58h]
  unsigned __int64 v78; // [rsp+18h] [rbp-48h]
  __int64 v79; // [rsp+20h] [rbp-40h]

  v75 = a1[1];
  v78 = *a1;
  v4 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v75 - *a1) >> 6);
  if ( v4 == 0x66666666666666LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v75 - *a1) >> 6);
  v6 = __CFADD__(v5, v4);
  v7 = v5 - 0x3333333333333333LL * ((__int64)(v75 - *a1) >> 6);
  if ( v6 )
  {
    v74 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v7 )
    {
      v76 = 0;
      v8 = 320;
      v79 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0x66666666666666LL )
      v7 = 0x66666666666666LL;
    v74 = 320 * v7;
  }
  v79 = sub_22077B0(v74);
  v76 = v79 + v74;
  v8 = v79 + 320;
LABEL_7:
  v9 = v79 + a2 - v78;
  if ( v9 )
  {
    v10 = _mm_loadu_si128((const __m128i *)a3);
    v11 = *(_BYTE **)(a3 + 24);
    v12 = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(v9 + 16) = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v9 + 24) = v9 + 40;
    *(__m128i *)v9 = v10;
    sub_2F07250((__int64 *)(v9 + 24), v11, (__int64)&v11[v12]);
    v13 = _mm_loadu_si128((const __m128i *)(a3 + 56));
    v14 = *(_BYTE **)(a3 + 104);
    v15 = *(_QWORD *)(a3 + 112);
    *(_DWORD *)(v9 + 72) = *(_DWORD *)(a3 + 72);
    v16 = *(_QWORD *)(a3 + 80);
    *(__m128i *)(v9 + 56) = v13;
    *(_QWORD *)(v9 + 80) = v16;
    *(_QWORD *)(v9 + 88) = *(_QWORD *)(a3 + 88);
    *(_WORD *)(v9 + 96) = *(_WORD *)(a3 + 96);
    *(_DWORD *)(v9 + 100) = *(_DWORD *)(a3 + 100);
    *(_QWORD *)(v9 + 104) = v9 + 120;
    sub_2F07250((__int64 *)(v9 + 104), v14, (__int64)&v14[v15]);
    v17 = _mm_loadu_si128((const __m128i *)(a3 + 136));
    v18 = _mm_loadu_si128((const __m128i *)(a3 + 160));
    v19 = *(_BYTE **)(a3 + 176);
    *(_BYTE *)(v9 + 152) = *(_BYTE *)(a3 + 152);
    v20 = *(_QWORD *)(a3 + 184);
    *(_QWORD *)(v9 + 176) = v9 + 192;
    *(__m128i *)(v9 + 136) = v17;
    *(__m128i *)(v9 + 160) = v18;
    sub_2F07250((__int64 *)(v9 + 176), v19, (__int64)&v19[v20]);
    v21 = _mm_loadu_si128((const __m128i *)(a3 + 208));
    v22 = *(_BYTE **)(a3 + 224);
    *(_QWORD *)(v9 + 224) = v9 + 240;
    v23 = *(_QWORD *)(a3 + 232);
    *(__m128i *)(v9 + 208) = v21;
    sub_2F07250((__int64 *)(v9 + 224), v22, (__int64)&v22[v23]);
    v24 = _mm_loadu_si128((const __m128i *)(a3 + 256));
    v25 = *(_BYTE **)(a3 + 272);
    v26 = *(_QWORD *)(a3 + 280);
    *(_QWORD *)(v9 + 272) = v9 + 288;
    *(__m128i *)(v9 + 256) = v24;
    sub_2F07250((__int64 *)(v9 + 272), v25, (__int64)&v25[v26]);
    *(__m128i *)(v9 + 304) = _mm_loadu_si128((const __m128i *)(a3 + 304));
  }
  if ( a2 != v78 )
  {
    v27 = v79;
    v28 = v78 + 40;
    v29 = v78 + 120;
    v30 = v78 + 192;
    v31 = v78 + 240;
    while ( 1 )
    {
      v47 = v28 + 248;
      if ( v27 )
      {
        *(__m128i *)v27 = _mm_loadu_si128((const __m128i *)(v28 - 40));
        *(_QWORD *)(v27 + 16) = *(_QWORD *)(v28 - 24);
        *(_QWORD *)(v27 + 24) = v27 + 40;
        v48 = *(_QWORD *)(v28 - 16);
        if ( v48 == v28 )
        {
          *(__m128i *)(v27 + 40) = _mm_loadu_si128((const __m128i *)v28);
        }
        else
        {
          *(_QWORD *)(v27 + 24) = v48;
          *(_QWORD *)(v27 + 40) = *(_QWORD *)v28;
        }
        *(_QWORD *)(v27 + 32) = *(_QWORD *)(v28 - 8);
        v32 = _mm_loadu_si128((const __m128i *)(v28 + 16));
        *(_QWORD *)(v28 - 16) = v28;
        *(_QWORD *)(v28 - 8) = 0;
        *(_BYTE *)v28 = 0;
        *(__m128i *)(v27 + 56) = v32;
        *(_DWORD *)(v27 + 72) = *(_DWORD *)(v28 + 32);
        *(_QWORD *)(v27 + 80) = *(_QWORD *)(v28 + 40);
        *(_QWORD *)(v27 + 88) = *(_QWORD *)(v28 + 48);
        *(_WORD *)(v27 + 96) = *(_WORD *)(v28 + 56);
        *(_DWORD *)(v27 + 100) = *(_DWORD *)(v28 + 60);
        *(_QWORD *)(v27 + 104) = v27 + 120;
        v33 = *(_QWORD *)(v28 + 64);
        if ( v33 == v29 )
        {
          *(__m128i *)(v27 + 120) = _mm_loadu_si128((const __m128i *)(v28 + 80));
        }
        else
        {
          *(_QWORD *)(v27 + 104) = v33;
          *(_QWORD *)(v27 + 120) = *(_QWORD *)(v28 + 80);
        }
        *(_QWORD *)(v27 + 112) = *(_QWORD *)(v28 + 72);
        v34 = _mm_loadu_si128((const __m128i *)(v28 + 96));
        *(_QWORD *)(v28 + 64) = v29;
        *(_QWORD *)(v28 + 72) = 0;
        *(_BYTE *)(v28 + 80) = 0;
        *(__m128i *)(v27 + 136) = v34;
        *(_BYTE *)(v27 + 152) = *(_BYTE *)(v28 + 112);
        v35 = _mm_loadu_si128((const __m128i *)(v28 + 120));
        *(_QWORD *)(v27 + 176) = v27 + 192;
        *(__m128i *)(v27 + 160) = v35;
        v36 = *(_QWORD *)(v28 + 136);
        if ( v36 == v30 )
        {
          *(__m128i *)(v27 + 192) = _mm_loadu_si128((const __m128i *)(v28 + 152));
        }
        else
        {
          *(_QWORD *)(v27 + 176) = v36;
          *(_QWORD *)(v27 + 192) = *(_QWORD *)(v28 + 152);
        }
        *(_QWORD *)(v27 + 184) = *(_QWORD *)(v28 + 144);
        v37 = _mm_loadu_si128((const __m128i *)(v28 + 168));
        *(_QWORD *)(v28 + 136) = v30;
        *(_QWORD *)(v28 + 144) = 0;
        *(_BYTE *)(v28 + 152) = 0;
        *(_QWORD *)(v27 + 224) = v27 + 240;
        *(__m128i *)(v27 + 208) = v37;
        v38 = *(_QWORD *)(v28 + 184);
        if ( v38 == v31 )
        {
          *(__m128i *)(v27 + 240) = _mm_loadu_si128((const __m128i *)(v28 + 200));
        }
        else
        {
          *(_QWORD *)(v27 + 224) = v38;
          *(_QWORD *)(v27 + 240) = *(_QWORD *)(v28 + 200);
        }
        *(_QWORD *)(v27 + 232) = *(_QWORD *)(v28 + 192);
        v39 = _mm_loadu_si128((const __m128i *)(v28 + 216));
        *(_QWORD *)(v28 + 184) = v31;
        *(_QWORD *)(v28 + 192) = 0;
        *(_BYTE *)(v28 + 200) = 0;
        *(_QWORD *)(v27 + 272) = v27 + 288;
        *(__m128i *)(v27 + 256) = v39;
        v40 = *(_QWORD *)(v28 + 232);
        if ( v40 == v47 )
        {
          *(__m128i *)(v27 + 288) = _mm_loadu_si128((const __m128i *)(v28 + 248));
        }
        else
        {
          *(_QWORD *)(v27 + 272) = v40;
          *(_QWORD *)(v27 + 288) = *(_QWORD *)(v28 + 248);
        }
        *(_QWORD *)(v27 + 280) = *(_QWORD *)(v28 + 240);
        v41 = _mm_loadu_si128((const __m128i *)(v28 + 264));
        *(_QWORD *)(v28 + 232) = v47;
        *(_QWORD *)(v28 + 240) = 0;
        *(_BYTE *)(v28 + 248) = 0;
        *(__m128i *)(v27 + 304) = v41;
      }
      v42 = *(_QWORD *)(v28 + 232);
      if ( v42 != v47 )
        j_j___libc_free_0(v42);
      v43 = *(_QWORD *)(v28 + 184);
      if ( v43 != v31 )
        j_j___libc_free_0(v43);
      v44 = *(_QWORD *)(v28 + 136);
      if ( v44 != v30 )
        j_j___libc_free_0(v44);
      v45 = *(_QWORD *)(v28 + 64);
      if ( v45 != v29 )
        j_j___libc_free_0(v45);
      v46 = *(_QWORD *)(v28 - 16);
      if ( v46 != v28 )
        j_j___libc_free_0(v46);
      v29 += 320LL;
      v30 += 320LL;
      v31 += 320LL;
      if ( a2 == v28 + 280 )
        break;
      v28 += 320LL;
      v27 += 320;
    }
    v8 = v27 + 640;
  }
  v49 = a2;
  if ( a2 != v75 )
  {
    v50 = a2 + 40;
    v51 = a2 + 240;
    v52 = v8;
    v53 = a2 + 192;
    v54 = a2 + 120;
    do
    {
      v71 = _mm_loadu_si128((const __m128i *)(v50 - 40));
      *(_QWORD *)(v52 + 16) = *(_QWORD *)(v50 - 24);
      *(_QWORD *)(v52 + 24) = v52 + 40;
      v72 = *(_QWORD *)(v50 - 16);
      *(__m128i *)v52 = v71;
      if ( v72 == v50 )
      {
        *(__m128i *)(v52 + 40) = _mm_loadu_si128((const __m128i *)v50);
      }
      else
      {
        *(_QWORD *)(v52 + 24) = v72;
        *(_QWORD *)(v52 + 40) = *(_QWORD *)v50;
      }
      v55 = *(_QWORD *)(v50 - 8);
      v56 = _mm_loadu_si128((const __m128i *)(v50 + 16));
      *(_QWORD *)(v50 - 16) = v50;
      *(_QWORD *)(v50 - 8) = 0;
      *(_QWORD *)(v52 + 32) = v55;
      LODWORD(v55) = *(_DWORD *)(v50 + 32);
      *(_BYTE *)v50 = 0;
      *(_DWORD *)(v52 + 72) = v55;
      v57 = *(_QWORD *)(v50 + 40);
      *(__m128i *)(v52 + 56) = v56;
      *(_QWORD *)(v52 + 80) = v57;
      *(_QWORD *)(v52 + 88) = *(_QWORD *)(v50 + 48);
      *(_WORD *)(v52 + 96) = *(_WORD *)(v50 + 56);
      *(_DWORD *)(v52 + 100) = *(_DWORD *)(v50 + 60);
      *(_QWORD *)(v52 + 104) = v52 + 120;
      v58 = *(_QWORD *)(v50 + 64);
      if ( v58 == v54 )
      {
        *(__m128i *)(v52 + 120) = _mm_loadu_si128((const __m128i *)(v50 + 80));
      }
      else
      {
        *(_QWORD *)(v52 + 104) = v58;
        *(_QWORD *)(v52 + 120) = *(_QWORD *)(v50 + 80);
      }
      v59 = *(_QWORD *)(v50 + 72);
      v60 = _mm_loadu_si128((const __m128i *)(v50 + 96));
      *(_QWORD *)(v50 + 64) = v54;
      v61 = _mm_loadu_si128((const __m128i *)(v50 + 120));
      *(_QWORD *)(v50 + 72) = 0;
      *(_QWORD *)(v52 + 112) = v59;
      LOBYTE(v59) = *(_BYTE *)(v50 + 112);
      *(_BYTE *)(v50 + 80) = 0;
      *(_BYTE *)(v52 + 152) = v59;
      *(_QWORD *)(v52 + 176) = v52 + 192;
      v62 = *(_QWORD *)(v50 + 136);
      *(__m128i *)(v52 + 136) = v60;
      *(__m128i *)(v52 + 160) = v61;
      if ( v62 == v53 )
      {
        *(__m128i *)(v52 + 192) = _mm_loadu_si128((const __m128i *)(v50 + 152));
      }
      else
      {
        *(_QWORD *)(v52 + 176) = v62;
        *(_QWORD *)(v52 + 192) = *(_QWORD *)(v50 + 152);
      }
      v63 = *(_QWORD *)(v50 + 144);
      v64 = _mm_loadu_si128((const __m128i *)(v50 + 168));
      *(_QWORD *)(v50 + 136) = v53;
      *(_QWORD *)(v50 + 144) = 0;
      *(_QWORD *)(v52 + 184) = v63;
      *(_QWORD *)(v52 + 224) = v52 + 240;
      v65 = *(_QWORD *)(v50 + 184);
      *(_BYTE *)(v50 + 152) = 0;
      *(__m128i *)(v52 + 208) = v64;
      if ( v65 == v51 )
      {
        *(__m128i *)(v52 + 240) = _mm_loadu_si128((const __m128i *)(v50 + 200));
      }
      else
      {
        *(_QWORD *)(v52 + 224) = v65;
        *(_QWORD *)(v52 + 240) = *(_QWORD *)(v50 + 200);
      }
      v66 = *(_QWORD *)(v50 + 192);
      v67 = _mm_loadu_si128((const __m128i *)(v50 + 216));
      *(_QWORD *)(v50 + 184) = v51;
      *(_QWORD *)(v50 + 192) = 0;
      *(_QWORD *)(v52 + 232) = v66;
      *(_QWORD *)(v52 + 272) = v52 + 288;
      v68 = *(_QWORD *)(v50 + 232);
      *(_BYTE *)(v50 + 200) = 0;
      *(__m128i *)(v52 + 256) = v67;
      if ( v68 == v49 + 288 )
      {
        *(__m128i *)(v52 + 288) = _mm_loadu_si128((const __m128i *)(v50 + 248));
      }
      else
      {
        *(_QWORD *)(v52 + 272) = v68;
        *(_QWORD *)(v52 + 288) = *(_QWORD *)(v50 + 248);
      }
      v69 = *(_QWORD *)(v50 + 240);
      v70 = _mm_loadu_si128((const __m128i *)(v50 + 264));
      v49 += 320;
      v52 += 320;
      v50 += 320;
      v51 += 320;
      v53 += 320;
      v54 += 320;
      *(_QWORD *)(v52 - 40) = v69;
      *(__m128i *)(v52 - 16) = v70;
    }
    while ( v49 != v75 );
    v8 += (5 * ((0xCCCCCCCCCCCCCDLL * ((unsigned __int64)(v49 - a2 - 320) >> 6)) & 0x3FFFFFFFFFFFFFFLL) + 5) << 6;
  }
  if ( v78 )
    j_j___libc_free_0(v78);
  a1[1] = v8;
  *a1 = v79;
  a1[2] = v76;
  return a1;
}
