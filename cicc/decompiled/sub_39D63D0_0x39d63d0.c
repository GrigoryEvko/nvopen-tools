// Function: sub_39D63D0
// Address: 0x39d63d0
//
unsigned __int64 *__fastcall sub_39D63D0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  bool v5; // zf
  __int64 v6; // rcx
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 v11; // r12
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rsi
  int v15; // eax
  __m128i v16; // xmm6
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  __m128i v19; // xmm7
  _BYTE *v20; // rsi
  __int64 v21; // rdx
  __m128i v22; // xmm6
  _BYTE *v23; // rsi
  __int64 v24; // rdx
  __m128i v25; // xmm7
  _BYTE *v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // r15
  unsigned __int64 v29; // r14
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // rbx
  __m128i v33; // xmm2
  __int64 v34; // rcx
  __m128i v35; // xmm3
  __int64 v36; // rcx
  __m128i v37; // xmm4
  __int64 v38; // rcx
  __m128i v39; // xmm5
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rdi
  __m128i v50; // xmm7
  __int64 v51; // rdi
  __int64 v52; // rdi
  __m128i v53; // xmm1
  __int64 v54; // rdi
  __int64 v55; // rdi
  __m128i v56; // xmm2
  __int64 v57; // rdi
  __int64 v58; // rdi
  __m128i v59; // xmm0
  __m128i v60; // xmm6
  int v61; // edi
  __int64 v62; // rdi
  unsigned __int64 v64; // rbx
  unsigned __int64 v65; // [rsp+0h] [rbp-60h]
  unsigned __int64 v67; // [rsp+10h] [rbp-50h]
  unsigned __int64 v68; // [rsp+18h] [rbp-48h]
  __int64 v69; // [rsp+20h] [rbp-40h]

  v67 = a1[1];
  v68 = *a1;
  v3 = (__int64)(v67 - *a1) >> 8;
  if ( v3 == 0x7FFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = v3 == 0;
  v6 = (__int64)(v67 - *a1) >> 8;
  v7 = 1;
  if ( !v5 )
    v7 = (__int64)(v67 - *a1) >> 8;
  v8 = __CFADD__(v6, v7);
  v9 = v6 + v7;
  if ( v8 )
  {
    v64 = 0x7FFFFFFFFFFFFF00LL;
  }
  else
  {
    if ( !v9 )
    {
      v65 = 0;
      v10 = 256;
      v69 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x7FFFFFFFFFFFFFLL )
      v9 = 0x7FFFFFFFFFFFFFLL;
    v64 = v9 << 8;
  }
  v69 = sub_22077B0(v64);
  v65 = v69 + v64;
  v10 = v69 + 256;
LABEL_7:
  v11 = v69 + a2 - v68;
  if ( v11 )
  {
    v12 = *(_DWORD *)(a3 + 48);
    v13 = *(_QWORD *)(a3 + 40);
    v14 = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(v11 + 16) = *(_QWORD *)(a3 + 16);
    v15 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(v11 + 48) = v12;
    LOWORD(v12) = *(_WORD *)(a3 + 52);
    *(_DWORD *)(v11 + 24) = v15;
    LOBYTE(v15) = *(_BYTE *)(a3 + 54);
    *(_QWORD *)(v11 + 40) = v13;
    v16 = _mm_loadu_si128((const __m128i *)a3);
    *(_BYTE *)(v11 + 54) = v15;
    *(_QWORD *)(v11 + 32) = v14;
    v17 = *(_BYTE **)(a3 + 56);
    *(_WORD *)(v11 + 52) = v12;
    v18 = *(_QWORD *)(a3 + 64);
    *(_QWORD *)(v11 + 56) = v11 + 72;
    *(__m128i *)v11 = v16;
    sub_39CF630((__int64 *)(v11 + 56), v17, (__int64)&v17[v18]);
    v19 = _mm_loadu_si128((const __m128i *)(a3 + 88));
    v20 = *(_BYTE **)(a3 + 112);
    v21 = *(_QWORD *)(a3 + 120);
    *(_BYTE *)(v11 + 104) = *(_BYTE *)(a3 + 104);
    *(_QWORD *)(v11 + 112) = v11 + 128;
    *(__m128i *)(v11 + 88) = v19;
    sub_39CF630((__int64 *)(v11 + 112), v20, (__int64)&v20[v21]);
    v22 = _mm_loadu_si128((const __m128i *)(a3 + 144));
    v23 = *(_BYTE **)(a3 + 160);
    *(_QWORD *)(v11 + 160) = v11 + 176;
    v24 = *(_QWORD *)(a3 + 168);
    *(__m128i *)(v11 + 144) = v22;
    sub_39CF630((__int64 *)(v11 + 160), v23, (__int64)&v23[v24]);
    v25 = _mm_loadu_si128((const __m128i *)(a3 + 192));
    v26 = *(_BYTE **)(a3 + 208);
    v27 = *(_QWORD *)(a3 + 216);
    *(_QWORD *)(v11 + 208) = v11 + 224;
    *(__m128i *)(v11 + 192) = v25;
    sub_39CF630((__int64 *)(v11 + 208), v26, (__int64)&v26[v27]);
    *(__m128i *)(v11 + 240) = _mm_loadu_si128((const __m128i *)(a3 + 240));
  }
  if ( a2 != v68 )
  {
    v28 = v69;
    v29 = v68 + 72;
    v30 = v68 + 128;
    v31 = v68 + 176;
    v32 = v68 + 224;
    while ( 1 )
    {
      if ( v28 )
      {
        *(__m128i *)v28 = _mm_loadu_si128((const __m128i *)(v29 - 72));
        *(_QWORD *)(v28 + 16) = *(_QWORD *)(v29 - 56);
        *(_DWORD *)(v28 + 24) = *(_DWORD *)(v29 - 48);
        *(_QWORD *)(v28 + 32) = *(_QWORD *)(v29 - 40);
        *(_QWORD *)(v28 + 40) = *(_QWORD *)(v29 - 32);
        *(_DWORD *)(v28 + 48) = *(_DWORD *)(v29 - 24);
        *(_BYTE *)(v28 + 52) = *(_BYTE *)(v29 - 20);
        *(_BYTE *)(v28 + 53) = *(_BYTE *)(v29 - 19);
        *(_BYTE *)(v28 + 54) = *(_BYTE *)(v29 - 18);
        *(_QWORD *)(v28 + 56) = v28 + 72;
        v44 = *(_QWORD *)(v29 - 16);
        if ( v44 == v29 )
        {
          *(__m128i *)(v28 + 72) = _mm_loadu_si128((const __m128i *)v29);
        }
        else
        {
          *(_QWORD *)(v28 + 56) = v44;
          *(_QWORD *)(v28 + 72) = *(_QWORD *)v29;
        }
        *(_QWORD *)(v28 + 64) = *(_QWORD *)(v29 - 8);
        v33 = _mm_loadu_si128((const __m128i *)(v29 + 16));
        *(_QWORD *)(v29 - 16) = v29;
        *(_QWORD *)(v29 - 8) = 0;
        *(_BYTE *)v29 = 0;
        *(__m128i *)(v28 + 88) = v33;
        *(_BYTE *)(v28 + 104) = *(_BYTE *)(v29 + 32);
        *(_QWORD *)(v28 + 112) = v28 + 128;
        v34 = *(_QWORD *)(v29 + 40);
        if ( v34 == v30 )
        {
          *(__m128i *)(v28 + 128) = _mm_loadu_si128((const __m128i *)(v29 + 56));
        }
        else
        {
          *(_QWORD *)(v28 + 112) = v34;
          *(_QWORD *)(v28 + 128) = *(_QWORD *)(v29 + 56);
        }
        *(_QWORD *)(v28 + 120) = *(_QWORD *)(v29 + 48);
        v35 = _mm_loadu_si128((const __m128i *)(v29 + 72));
        *(_QWORD *)(v29 + 40) = v30;
        *(_QWORD *)(v29 + 48) = 0;
        *(_BYTE *)(v29 + 56) = 0;
        *(_QWORD *)(v28 + 160) = v28 + 176;
        *(__m128i *)(v28 + 144) = v35;
        v36 = *(_QWORD *)(v29 + 88);
        if ( v36 == v31 )
        {
          *(__m128i *)(v28 + 176) = _mm_loadu_si128((const __m128i *)(v29 + 104));
        }
        else
        {
          *(_QWORD *)(v28 + 160) = v36;
          *(_QWORD *)(v28 + 176) = *(_QWORD *)(v29 + 104);
        }
        *(_QWORD *)(v28 + 168) = *(_QWORD *)(v29 + 96);
        v37 = _mm_loadu_si128((const __m128i *)(v29 + 120));
        *(_QWORD *)(v29 + 88) = v31;
        *(_QWORD *)(v29 + 96) = 0;
        *(_BYTE *)(v29 + 104) = 0;
        *(_QWORD *)(v28 + 208) = v28 + 224;
        *(__m128i *)(v28 + 192) = v37;
        v38 = *(_QWORD *)(v29 + 136);
        if ( v38 == v32 )
        {
          *(__m128i *)(v28 + 224) = _mm_loadu_si128((const __m128i *)(v29 + 152));
        }
        else
        {
          *(_QWORD *)(v28 + 208) = v38;
          *(_QWORD *)(v28 + 224) = *(_QWORD *)(v29 + 152);
        }
        *(_QWORD *)(v28 + 216) = *(_QWORD *)(v29 + 144);
        v39 = _mm_loadu_si128((const __m128i *)(v29 + 168));
        *(_QWORD *)(v29 + 136) = v32;
        *(_QWORD *)(v29 + 144) = 0;
        *(_BYTE *)(v29 + 152) = 0;
        *(__m128i *)(v28 + 240) = v39;
      }
      v40 = *(_QWORD *)(v29 + 136);
      if ( v40 != v32 )
        j_j___libc_free_0(v40);
      v41 = *(_QWORD *)(v29 + 88);
      if ( v41 != v31 )
        j_j___libc_free_0(v41);
      v42 = *(_QWORD *)(v29 + 40);
      if ( v42 != v30 )
        j_j___libc_free_0(v42);
      v43 = *(_QWORD *)(v29 - 16);
      if ( v43 != v29 )
        j_j___libc_free_0(v43);
      v30 += 256LL;
      v31 += 256LL;
      v32 += 256LL;
      if ( a2 == v29 + 184 )
        break;
      v29 += 256LL;
      v28 += 256;
    }
    v10 = v28 + 512;
  }
  if ( a2 != v67 )
  {
    v45 = a2 + 72;
    v46 = a2 + 176;
    v47 = a2 + 128;
    v48 = v10;
    do
    {
      v60 = _mm_loadu_si128((const __m128i *)(v45 - 72));
      *(_QWORD *)(v48 + 16) = *(_QWORD *)(v45 - 56);
      v61 = *(_DWORD *)(v45 - 48);
      *(__m128i *)v48 = v60;
      *(_DWORD *)(v48 + 24) = v61;
      *(_QWORD *)(v48 + 32) = *(_QWORD *)(v45 - 40);
      *(_QWORD *)(v48 + 40) = *(_QWORD *)(v45 - 32);
      *(_DWORD *)(v48 + 48) = *(_DWORD *)(v45 - 24);
      *(_BYTE *)(v48 + 52) = *(_BYTE *)(v45 - 20);
      *(_BYTE *)(v48 + 53) = *(_BYTE *)(v45 - 19);
      *(_BYTE *)(v48 + 54) = *(_BYTE *)(v45 - 18);
      *(_QWORD *)(v48 + 56) = v48 + 72;
      v62 = *(_QWORD *)(v45 - 16);
      if ( v62 == v45 )
      {
        *(__m128i *)(v48 + 72) = _mm_loadu_si128((const __m128i *)v45);
      }
      else
      {
        *(_QWORD *)(v48 + 56) = v62;
        *(_QWORD *)(v48 + 72) = *(_QWORD *)v45;
      }
      v49 = *(_QWORD *)(v45 - 8);
      v50 = _mm_loadu_si128((const __m128i *)(v45 + 16));
      *(_QWORD *)(v45 - 16) = v45;
      *(_QWORD *)(v45 - 8) = 0;
      *(_QWORD *)(v48 + 64) = v49;
      LOBYTE(v49) = *(_BYTE *)(v45 + 32);
      *(_BYTE *)v45 = 0;
      *(_BYTE *)(v48 + 104) = v49;
      *(_QWORD *)(v48 + 112) = v48 + 128;
      v51 = *(_QWORD *)(v45 + 40);
      *(__m128i *)(v48 + 88) = v50;
      if ( v51 == v47 )
      {
        *(__m128i *)(v48 + 128) = _mm_loadu_si128((const __m128i *)(v45 + 56));
      }
      else
      {
        *(_QWORD *)(v48 + 112) = v51;
        *(_QWORD *)(v48 + 128) = *(_QWORD *)(v45 + 56);
      }
      v52 = *(_QWORD *)(v45 + 48);
      v53 = _mm_loadu_si128((const __m128i *)(v45 + 72));
      *(_QWORD *)(v45 + 40) = v47;
      *(_QWORD *)(v45 + 48) = 0;
      *(_QWORD *)(v48 + 120) = v52;
      *(_QWORD *)(v48 + 160) = v48 + 176;
      v54 = *(_QWORD *)(v45 + 88);
      *(_BYTE *)(v45 + 56) = 0;
      *(__m128i *)(v48 + 144) = v53;
      if ( v54 == v46 )
      {
        *(__m128i *)(v48 + 176) = _mm_loadu_si128((const __m128i *)(v45 + 104));
      }
      else
      {
        *(_QWORD *)(v48 + 160) = v54;
        *(_QWORD *)(v48 + 176) = *(_QWORD *)(v45 + 104);
      }
      v55 = *(_QWORD *)(v45 + 96);
      v56 = _mm_loadu_si128((const __m128i *)(v45 + 120));
      *(_QWORD *)(v45 + 88) = v46;
      *(_QWORD *)(v45 + 96) = 0;
      *(_QWORD *)(v48 + 168) = v55;
      *(_QWORD *)(v48 + 208) = v48 + 224;
      v57 = *(_QWORD *)(v45 + 136);
      *(_BYTE *)(v45 + 104) = 0;
      *(__m128i *)(v48 + 192) = v56;
      if ( v57 == v45 + 152 )
      {
        *(__m128i *)(v48 + 224) = _mm_loadu_si128((const __m128i *)(v45 + 152));
      }
      else
      {
        *(_QWORD *)(v48 + 208) = v57;
        *(_QWORD *)(v48 + 224) = *(_QWORD *)(v45 + 152);
      }
      v58 = *(_QWORD *)(v45 + 144);
      v59 = _mm_loadu_si128((const __m128i *)(v45 + 168));
      v45 += 256;
      v48 += 256;
      v46 += 256;
      v47 += 256;
      *(_QWORD *)(v48 - 40) = v58;
      *(__m128i *)(v48 - 16) = v59;
    }
    while ( v45 != v67 + 72 );
    v10 += v67 - a2;
  }
  if ( v68 )
    j_j___libc_free_0(v68);
  a1[1] = v10;
  *a1 = v69;
  a1[2] = v65;
  return a1;
}
