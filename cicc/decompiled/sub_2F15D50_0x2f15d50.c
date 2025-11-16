// Function: sub_2F15D50
// Address: 0x2f15d50
//
unsigned __int64 *__fastcall sub_2F15D50(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  __m128i v11; // xmm6
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rcx
  char v15; // dl
  __m128i v16; // xmm7
  __int64 v17; // rcx
  __int64 v18; // rcx
  __m128i v19; // xmm6
  __int64 v20; // rcx
  __int64 v21; // rcx
  __m128i v22; // xmm7
  __int64 v23; // rcx
  __int64 v24; // rcx
  __m128i v25; // xmm6
  __int64 v26; // r15
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rbx
  __m128i v31; // xmm2
  __int64 v32; // rcx
  __m128i v33; // xmm3
  __int64 v34; // rcx
  __m128i v35; // xmm4
  __int64 v36; // rcx
  __m128i v37; // xmm5
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 v48; // r8
  __m128i v49; // xmm7
  __int64 v50; // r8
  __int64 v51; // r8
  __m128i v52; // xmm1
  __int64 v53; // r8
  __int64 v54; // r8
  __m128i v55; // xmm2
  __int64 v56; // r8
  __int64 v57; // r8
  __m128i v58; // xmm0
  __m128i v59; // xmm6
  int v60; // r8d
  __int64 v61; // r8
  unsigned __int64 v63; // rbx
  unsigned __int64 v64; // [rsp+0h] [rbp-60h]
  unsigned __int64 v66; // [rsp+10h] [rbp-50h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  unsigned __int64 v68; // [rsp+20h] [rbp-40h]

  v68 = a1[1];
  v66 = *a1;
  v4 = 0xF83E0F83E0F83E1LL * ((__int64)(v68 - *a1) >> 3);
  if ( v4 == 0x7C1F07C1F07C1FLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xF83E0F83E0F83E1LL * ((__int64)(v68 - *a1) >> 3);
  v6 = __CFADD__(v5, v4);
  v7 = v5 + v4;
  v8 = a2 - v66;
  if ( v6 )
  {
    v63 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v7 )
    {
      v64 = 0;
      v9 = 264;
      v67 = 0;
      goto LABEL_7;
    }
    if ( v7 > 0x7C1F07C1F07C1FLL )
      v7 = 0x7C1F07C1F07C1FLL;
    v63 = 264 * v7;
  }
  v67 = sub_22077B0(v63);
  v64 = v67 + v63;
  v9 = v67 + 264;
LABEL_7:
  v10 = v67 + v8;
  if ( v67 + v8 )
  {
    v11 = _mm_loadu_si128((const __m128i *)a3);
    v12 = *(_QWORD *)(a3 + 64);
    *(_QWORD *)(v10 + 16) = *(_QWORD *)(a3 + 16);
    v13 = *(_DWORD *)(a3 + 24);
    *(__m128i *)v10 = v11;
    *(_DWORD *)(v10 + 24) = v13;
    *(_QWORD *)(v10 + 32) = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(v10 + 40) = *(_QWORD *)(a3 + 40);
    *(_WORD *)(v10 + 48) = *(_WORD *)(a3 + 48);
    *(_DWORD *)(v10 + 52) = *(_DWORD *)(a3 + 52);
    *(_WORD *)(v10 + 56) = *(_WORD *)(a3 + 56);
    *(_QWORD *)(v10 + 64) = v10 + 80;
    if ( v12 == a3 + 80 )
    {
      *(__m128i *)(v10 + 80) = _mm_loadu_si128((const __m128i *)(a3 + 80));
    }
    else
    {
      *(_QWORD *)(v10 + 64) = v12;
      *(_QWORD *)(v10 + 80) = *(_QWORD *)(a3 + 80);
    }
    v14 = *(_QWORD *)(a3 + 72);
    *(_QWORD *)(a3 + 64) = a3 + 80;
    v15 = *(_BYTE *)(a3 + 112);
    v16 = _mm_loadu_si128((const __m128i *)(a3 + 96));
    *(_QWORD *)(a3 + 72) = 0;
    *(_QWORD *)(v10 + 72) = v14;
    v17 = *(_QWORD *)(a3 + 120);
    *(_BYTE *)(v10 + 112) = v15;
    *(_QWORD *)(v10 + 120) = v10 + 136;
    *(_BYTE *)(a3 + 80) = 0;
    *(__m128i *)(v10 + 96) = v16;
    if ( v17 == a3 + 136 )
    {
      *(__m128i *)(v10 + 136) = _mm_loadu_si128((const __m128i *)(a3 + 136));
    }
    else
    {
      *(_QWORD *)(v10 + 120) = v17;
      *(_QWORD *)(v10 + 136) = *(_QWORD *)(a3 + 136);
    }
    v18 = *(_QWORD *)(a3 + 128);
    *(_QWORD *)(a3 + 120) = a3 + 136;
    v19 = _mm_loadu_si128((const __m128i *)(a3 + 152));
    *(_QWORD *)(v10 + 168) = v10 + 184;
    *(_QWORD *)(v10 + 128) = v18;
    v20 = *(_QWORD *)(a3 + 168);
    *(_QWORD *)(a3 + 128) = 0;
    *(_BYTE *)(a3 + 136) = 0;
    *(__m128i *)(v10 + 152) = v19;
    if ( v20 == a3 + 184 )
    {
      *(__m128i *)(v10 + 184) = _mm_loadu_si128((const __m128i *)(a3 + 184));
    }
    else
    {
      *(_QWORD *)(v10 + 168) = v20;
      *(_QWORD *)(v10 + 184) = *(_QWORD *)(a3 + 184);
    }
    v21 = *(_QWORD *)(a3 + 176);
    *(_QWORD *)(a3 + 168) = a3 + 184;
    v22 = _mm_loadu_si128((const __m128i *)(a3 + 200));
    *(_QWORD *)(v10 + 216) = v10 + 232;
    *(_QWORD *)(v10 + 176) = v21;
    v23 = *(_QWORD *)(a3 + 216);
    *(_QWORD *)(a3 + 176) = 0;
    *(_BYTE *)(a3 + 184) = 0;
    *(__m128i *)(v10 + 200) = v22;
    if ( v23 == a3 + 232 )
    {
      *(__m128i *)(v10 + 232) = _mm_loadu_si128((const __m128i *)(a3 + 232));
    }
    else
    {
      *(_QWORD *)(v10 + 216) = v23;
      *(_QWORD *)(v10 + 232) = *(_QWORD *)(a3 + 232);
    }
    v24 = *(_QWORD *)(a3 + 224);
    v25 = _mm_loadu_si128((const __m128i *)(a3 + 248));
    *(_QWORD *)(a3 + 216) = a3 + 232;
    *(_QWORD *)(a3 + 224) = 0;
    *(_QWORD *)(v10 + 224) = v24;
    *(_BYTE *)(a3 + 232) = 0;
    *(__m128i *)(v10 + 248) = v25;
  }
  if ( a2 != v66 )
  {
    v26 = v67;
    v27 = v66 + 80;
    v28 = v66 + 136;
    v29 = v66 + 184;
    v30 = v66 + 232;
    while ( 1 )
    {
      if ( v26 )
      {
        *(__m128i *)v26 = _mm_loadu_si128((const __m128i *)(v27 - 80));
        *(_QWORD *)(v26 + 16) = *(_QWORD *)(v27 - 64);
        *(_DWORD *)(v26 + 24) = *(_DWORD *)(v27 - 56);
        *(_QWORD *)(v26 + 32) = *(_QWORD *)(v27 - 48);
        *(_QWORD *)(v26 + 40) = *(_QWORD *)(v27 - 40);
        *(_WORD *)(v26 + 48) = *(_WORD *)(v27 - 32);
        *(_DWORD *)(v26 + 52) = *(_DWORD *)(v27 - 28);
        *(_BYTE *)(v26 + 56) = *(_BYTE *)(v27 - 24);
        *(_BYTE *)(v26 + 57) = *(_BYTE *)(v27 - 23);
        *(_QWORD *)(v26 + 64) = v26 + 80;
        v42 = *(_QWORD *)(v27 - 16);
        if ( v42 == v27 )
        {
          *(__m128i *)(v26 + 80) = _mm_loadu_si128((const __m128i *)v27);
        }
        else
        {
          *(_QWORD *)(v26 + 64) = v42;
          *(_QWORD *)(v26 + 80) = *(_QWORD *)v27;
        }
        *(_QWORD *)(v26 + 72) = *(_QWORD *)(v27 - 8);
        v31 = _mm_loadu_si128((const __m128i *)(v27 + 16));
        *(_QWORD *)(v27 - 16) = v27;
        *(_QWORD *)(v27 - 8) = 0;
        *(_BYTE *)v27 = 0;
        *(__m128i *)(v26 + 96) = v31;
        *(_BYTE *)(v26 + 112) = *(_BYTE *)(v27 + 32);
        *(_QWORD *)(v26 + 120) = v26 + 136;
        v32 = *(_QWORD *)(v27 + 40);
        if ( v32 == v28 )
        {
          *(__m128i *)(v26 + 136) = _mm_loadu_si128((const __m128i *)(v27 + 56));
        }
        else
        {
          *(_QWORD *)(v26 + 120) = v32;
          *(_QWORD *)(v26 + 136) = *(_QWORD *)(v27 + 56);
        }
        *(_QWORD *)(v26 + 128) = *(_QWORD *)(v27 + 48);
        v33 = _mm_loadu_si128((const __m128i *)(v27 + 72));
        *(_QWORD *)(v27 + 40) = v28;
        *(_QWORD *)(v27 + 48) = 0;
        *(_BYTE *)(v27 + 56) = 0;
        *(_QWORD *)(v26 + 168) = v26 + 184;
        *(__m128i *)(v26 + 152) = v33;
        v34 = *(_QWORD *)(v27 + 88);
        if ( v34 == v29 )
        {
          *(__m128i *)(v26 + 184) = _mm_loadu_si128((const __m128i *)(v27 + 104));
        }
        else
        {
          *(_QWORD *)(v26 + 168) = v34;
          *(_QWORD *)(v26 + 184) = *(_QWORD *)(v27 + 104);
        }
        *(_QWORD *)(v26 + 176) = *(_QWORD *)(v27 + 96);
        v35 = _mm_loadu_si128((const __m128i *)(v27 + 120));
        *(_QWORD *)(v27 + 88) = v29;
        *(_QWORD *)(v27 + 96) = 0;
        *(_BYTE *)(v27 + 104) = 0;
        *(_QWORD *)(v26 + 216) = v26 + 232;
        *(__m128i *)(v26 + 200) = v35;
        v36 = *(_QWORD *)(v27 + 136);
        if ( v36 == v30 )
        {
          *(__m128i *)(v26 + 232) = _mm_loadu_si128((const __m128i *)(v27 + 152));
        }
        else
        {
          *(_QWORD *)(v26 + 216) = v36;
          *(_QWORD *)(v26 + 232) = *(_QWORD *)(v27 + 152);
        }
        *(_QWORD *)(v26 + 224) = *(_QWORD *)(v27 + 144);
        v37 = _mm_loadu_si128((const __m128i *)(v27 + 168));
        *(_QWORD *)(v27 + 136) = v30;
        *(_QWORD *)(v27 + 144) = 0;
        *(_BYTE *)(v27 + 152) = 0;
        *(__m128i *)(v26 + 248) = v37;
      }
      v38 = *(_QWORD *)(v27 + 136);
      if ( v38 != v30 )
        j_j___libc_free_0(v38);
      v39 = *(_QWORD *)(v27 + 88);
      if ( v39 != v29 )
        j_j___libc_free_0(v39);
      v40 = *(_QWORD *)(v27 + 40);
      if ( v40 != v28 )
        j_j___libc_free_0(v40);
      v41 = *(_QWORD *)(v27 - 16);
      if ( v41 != v27 )
        j_j___libc_free_0(v41);
      v28 += 264LL;
      v29 += 264LL;
      v30 += 264LL;
      if ( a2 == v27 + 184 )
        break;
      v27 += 264LL;
      v26 += 264;
    }
    v9 = v26 + 528;
  }
  v43 = a2;
  if ( a2 != v68 )
  {
    v44 = a2 + 80;
    v45 = a2 + 184;
    v46 = v9;
    v47 = a2 + 136;
    do
    {
      v59 = _mm_loadu_si128((const __m128i *)(v44 - 80));
      *(_QWORD *)(v46 + 16) = *(_QWORD *)(v44 - 64);
      v60 = *(_DWORD *)(v44 - 56);
      *(__m128i *)v46 = v59;
      *(_DWORD *)(v46 + 24) = v60;
      *(_QWORD *)(v46 + 32) = *(_QWORD *)(v44 - 48);
      *(_QWORD *)(v46 + 40) = *(_QWORD *)(v44 - 40);
      *(_WORD *)(v46 + 48) = *(_WORD *)(v44 - 32);
      *(_DWORD *)(v46 + 52) = *(_DWORD *)(v44 - 28);
      *(_BYTE *)(v46 + 56) = *(_BYTE *)(v44 - 24);
      *(_BYTE *)(v46 + 57) = *(_BYTE *)(v44 - 23);
      *(_QWORD *)(v46 + 64) = v46 + 80;
      v61 = *(_QWORD *)(v44 - 16);
      if ( v61 == v44 )
      {
        *(__m128i *)(v46 + 80) = _mm_loadu_si128((const __m128i *)v44);
      }
      else
      {
        *(_QWORD *)(v46 + 64) = v61;
        *(_QWORD *)(v46 + 80) = *(_QWORD *)v44;
      }
      v48 = *(_QWORD *)(v44 - 8);
      v49 = _mm_loadu_si128((const __m128i *)(v44 + 16));
      *(_QWORD *)(v44 - 16) = v44;
      *(_QWORD *)(v44 - 8) = 0;
      *(_QWORD *)(v46 + 72) = v48;
      LOBYTE(v48) = *(_BYTE *)(v44 + 32);
      *(_BYTE *)v44 = 0;
      *(_BYTE *)(v46 + 112) = v48;
      *(_QWORD *)(v46 + 120) = v46 + 136;
      v50 = *(_QWORD *)(v44 + 40);
      *(__m128i *)(v46 + 96) = v49;
      if ( v50 == v47 )
      {
        *(__m128i *)(v46 + 136) = _mm_loadu_si128((const __m128i *)(v44 + 56));
      }
      else
      {
        *(_QWORD *)(v46 + 120) = v50;
        *(_QWORD *)(v46 + 136) = *(_QWORD *)(v44 + 56);
      }
      v51 = *(_QWORD *)(v44 + 48);
      v52 = _mm_loadu_si128((const __m128i *)(v44 + 72));
      *(_QWORD *)(v44 + 40) = v47;
      *(_QWORD *)(v44 + 48) = 0;
      *(_QWORD *)(v46 + 128) = v51;
      *(_QWORD *)(v46 + 168) = v46 + 184;
      v53 = *(_QWORD *)(v44 + 88);
      *(_BYTE *)(v44 + 56) = 0;
      *(__m128i *)(v46 + 152) = v52;
      if ( v53 == v45 )
      {
        *(__m128i *)(v46 + 184) = _mm_loadu_si128((const __m128i *)(v44 + 104));
      }
      else
      {
        *(_QWORD *)(v46 + 168) = v53;
        *(_QWORD *)(v46 + 184) = *(_QWORD *)(v44 + 104);
      }
      v54 = *(_QWORD *)(v44 + 96);
      v55 = _mm_loadu_si128((const __m128i *)(v44 + 120));
      *(_QWORD *)(v44 + 88) = v45;
      *(_QWORD *)(v44 + 96) = 0;
      *(_QWORD *)(v46 + 176) = v54;
      *(_QWORD *)(v46 + 216) = v46 + 232;
      v56 = *(_QWORD *)(v44 + 136);
      *(_BYTE *)(v44 + 104) = 0;
      *(__m128i *)(v46 + 200) = v55;
      if ( v56 == v43 + 232 )
      {
        *(__m128i *)(v46 + 232) = _mm_loadu_si128((const __m128i *)(v44 + 152));
      }
      else
      {
        *(_QWORD *)(v46 + 216) = v56;
        *(_QWORD *)(v46 + 232) = *(_QWORD *)(v44 + 152);
      }
      v57 = *(_QWORD *)(v44 + 144);
      v58 = _mm_loadu_si128((const __m128i *)(v44 + 168));
      v43 += 264;
      v46 += 264;
      v44 += 264;
      v45 += 264;
      v47 += 264;
      *(_QWORD *)(v46 - 40) = v57;
      *(__m128i *)(v46 - 16) = v58;
    }
    while ( v43 != v68 );
    v9 += 264 * (((0xF83E0F83E0F83E1LL * ((unsigned __int64)(v43 - a2 - 264) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
  }
  if ( v66 )
    j_j___libc_free_0(v66);
  a1[1] = v9;
  *a1 = v67;
  a1[2] = v64;
  return a1;
}
