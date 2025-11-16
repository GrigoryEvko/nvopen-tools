// Function: sub_2F0C9E0
// Address: 0x2f0c9e0
//
void __fastcall sub_2F0C9E0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // r15
  __int64 v12; // rbx
  unsigned __int64 v13; // r14
  __m128i v14; // xmm5
  unsigned __int64 *v15; // rsi
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  char *v22; // r13
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 i; // r14
  __m128i v26; // xmm6
  __m128i v27; // xmm7
  __m128i v28; // xmm3
  _QWORD *v29; // rbx
  _QWORD *v30; // r14
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r14
  _QWORD *v37; // r13
  __int64 v38; // rbx
  __m128i v39; // xmm4
  unsigned __int64 *v40; // rsi
  unsigned __int64 *v41; // rdi
  __int64 v42; // rbx
  __m128i v43; // xmm7
  __m128i v44; // xmm0
  __m128i v45; // xmm1
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+10h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a2[1];
    v4 = *a2;
    v5 = (_QWORD *)*a1;
    v6 = v2 - *a2;
    v7 = a1[2] - *a1;
    v49 = v6;
    if ( v7 < v6 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFF80LL )
          sub_4261EA(a1, v6, v7);
        v46 = v2;
        v23 = sub_22077B0(v6);
        v2 = v46;
        v24 = v23;
      }
      else
      {
        v24 = 0;
      }
      for ( i = v24; v2 != v4; i += 320 )
      {
        if ( i )
        {
          v47 = v2;
          *(__m128i *)i = _mm_loadu_si128((const __m128i *)v4);
          *(_QWORD *)(i + 16) = *(_QWORD *)(v4 + 16);
          *(_QWORD *)(i + 24) = i + 40;
          sub_2F07250((__int64 *)(i + 24), *(_BYTE **)(v4 + 24), *(_QWORD *)(v4 + 24) + *(_QWORD *)(v4 + 32));
          *(__m128i *)(i + 56) = _mm_loadu_si128((const __m128i *)(v4 + 56));
          *(_DWORD *)(i + 72) = *(_DWORD *)(v4 + 72);
          *(_QWORD *)(i + 80) = *(_QWORD *)(v4 + 80);
          *(_QWORD *)(i + 88) = *(_QWORD *)(v4 + 88);
          *(_WORD *)(i + 96) = *(_WORD *)(v4 + 96);
          *(_DWORD *)(i + 100) = *(_DWORD *)(v4 + 100);
          *(_QWORD *)(i + 104) = i + 120;
          sub_2F07250((__int64 *)(i + 104), *(_BYTE **)(v4 + 104), *(_QWORD *)(v4 + 104) + *(_QWORD *)(v4 + 112));
          *(__m128i *)(i + 136) = _mm_loadu_si128((const __m128i *)(v4 + 136));
          *(_BYTE *)(i + 152) = *(_BYTE *)(v4 + 152);
          v26 = _mm_loadu_si128((const __m128i *)(v4 + 160));
          *(_QWORD *)(i + 176) = i + 192;
          *(__m128i *)(i + 160) = v26;
          sub_2F07250((__int64 *)(i + 176), *(_BYTE **)(v4 + 176), *(_QWORD *)(v4 + 176) + *(_QWORD *)(v4 + 184));
          v27 = _mm_loadu_si128((const __m128i *)(v4 + 208));
          *(_QWORD *)(i + 224) = i + 240;
          *(__m128i *)(i + 208) = v27;
          sub_2F07250((__int64 *)(i + 224), *(_BYTE **)(v4 + 224), *(_QWORD *)(v4 + 224) + *(_QWORD *)(v4 + 232));
          v28 = _mm_loadu_si128((const __m128i *)(v4 + 256));
          *(_QWORD *)(i + 272) = i + 288;
          *(__m128i *)(i + 256) = v28;
          sub_2F07250((__int64 *)(i + 272), *(_BYTE **)(v4 + 272), *(_QWORD *)(v4 + 272) + *(_QWORD *)(v4 + 280));
          v2 = v47;
          *(__m128i *)(i + 304) = _mm_loadu_si128((const __m128i *)(v4 + 304));
        }
        v4 += 320;
      }
      v29 = (_QWORD *)a1[1];
      v30 = (_QWORD *)*a1;
      if ( v29 != (_QWORD *)*a1 )
      {
        do
        {
          v31 = v30[34];
          if ( (_QWORD *)v31 != v30 + 36 )
            j_j___libc_free_0(v31);
          v32 = v30[28];
          if ( (_QWORD *)v32 != v30 + 30 )
            j_j___libc_free_0(v32);
          v33 = v30[22];
          if ( (_QWORD *)v33 != v30 + 24 )
            j_j___libc_free_0(v33);
          v34 = v30[13];
          if ( (_QWORD *)v34 != v30 + 15 )
            j_j___libc_free_0(v34);
          v35 = v30[3];
          if ( (_QWORD *)v35 != v30 + 5 )
            j_j___libc_free_0(v35);
          v30 += 40;
        }
        while ( v29 != v30 );
        v30 = (_QWORD *)*a1;
      }
      if ( v30 )
        j_j___libc_free_0((unsigned __int64)v30);
      *a1 = v24;
      v22 = (char *)(v6 + v24);
      a1[2] = (__int64)v22;
      goto LABEL_21;
    }
    v8 = a1[1];
    v9 = v8 - (_QWORD)v5;
    v10 = v8 - (_QWORD)v5;
    if ( v49 > (unsigned __int64)(v8 - (_QWORD)v5) )
    {
      v36 = 0xCCCCCCCCCCCCCCCDLL * (v9 >> 6);
      if ( v9 > 0 )
      {
        v37 = v5 + 3;
        v38 = v4 + 24;
        do
        {
          *(__m128i *)(v37 - 3) = _mm_loadu_si128((const __m128i *)(v38 - 24));
          *(v37 - 1) = *(_QWORD *)(v38 - 8);
          sub_2240AE0(v37, (unsigned __int64 *)v38);
          *((__m128i *)v37 + 2) = _mm_loadu_si128((const __m128i *)(v38 + 32));
          *((_DWORD *)v37 + 12) = *(_DWORD *)(v38 + 48);
          v37[7] = *(_QWORD *)(v38 + 56);
          v37[8] = *(_QWORD *)(v38 + 64);
          *((_WORD *)v37 + 36) = *(_WORD *)(v38 + 72);
          *((_DWORD *)v37 + 19) = *(_DWORD *)(v38 + 76);
          sub_2240AE0(v37 + 10, (unsigned __int64 *)(v38 + 80));
          *((__m128i *)v37 + 7) = _mm_loadu_si128((const __m128i *)(v38 + 112));
          *((_BYTE *)v37 + 128) = *(_BYTE *)(v38 + 128);
          *(__m128i *)(v37 + 17) = _mm_loadu_si128((const __m128i *)(v38 + 136));
          sub_2240AE0(v37 + 19, (unsigned __int64 *)(v38 + 152));
          *(__m128i *)(v37 + 23) = _mm_loadu_si128((const __m128i *)(v38 + 184));
          sub_2240AE0(v37 + 25, (unsigned __int64 *)(v38 + 200));
          v39 = _mm_loadu_si128((const __m128i *)(v38 + 232));
          v40 = (unsigned __int64 *)(v38 + 248);
          v41 = v37 + 31;
          v38 += 320;
          v37 += 40;
          *(__m128i *)(v37 - 11) = v39;
          sub_2240AE0(v41, v40);
          *(__m128i *)(v37 - 5) = _mm_loadu_si128((const __m128i *)(v38 - 40));
          --v36;
        }
        while ( v36 );
        v2 = a2[1];
        v4 = *a2;
        v8 = a1[1];
        v5 = (_QWORD *)*a1;
        v10 = v8 - *a1;
      }
      v42 = v10 + v4;
      v22 = (char *)v5 + v49;
      if ( v42 == v2 )
        goto LABEL_21;
      do
      {
        if ( v8 )
        {
          v48 = v2;
          *(__m128i *)v8 = _mm_loadu_si128((const __m128i *)v42);
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v42 + 16);
          *(_QWORD *)(v8 + 24) = v8 + 40;
          sub_2F07250((__int64 *)(v8 + 24), *(_BYTE **)(v42 + 24), *(_QWORD *)(v42 + 24) + *(_QWORD *)(v42 + 32));
          *(__m128i *)(v8 + 56) = _mm_loadu_si128((const __m128i *)(v42 + 56));
          *(_DWORD *)(v8 + 72) = *(_DWORD *)(v42 + 72);
          *(_QWORD *)(v8 + 80) = *(_QWORD *)(v42 + 80);
          *(_QWORD *)(v8 + 88) = *(_QWORD *)(v42 + 88);
          *(_WORD *)(v8 + 96) = *(_WORD *)(v42 + 96);
          *(_DWORD *)(v8 + 100) = *(_DWORD *)(v42 + 100);
          *(_QWORD *)(v8 + 104) = v8 + 120;
          sub_2F07250((__int64 *)(v8 + 104), *(_BYTE **)(v42 + 104), *(_QWORD *)(v42 + 104) + *(_QWORD *)(v42 + 112));
          *(__m128i *)(v8 + 136) = _mm_loadu_si128((const __m128i *)(v42 + 136));
          *(_BYTE *)(v8 + 152) = *(_BYTE *)(v42 + 152);
          v43 = _mm_loadu_si128((const __m128i *)(v42 + 160));
          *(_QWORD *)(v8 + 176) = v8 + 192;
          *(__m128i *)(v8 + 160) = v43;
          sub_2F07250((__int64 *)(v8 + 176), *(_BYTE **)(v42 + 176), *(_QWORD *)(v42 + 176) + *(_QWORD *)(v42 + 184));
          v44 = _mm_loadu_si128((const __m128i *)(v42 + 208));
          *(_QWORD *)(v8 + 224) = v8 + 240;
          *(__m128i *)(v8 + 208) = v44;
          sub_2F07250((__int64 *)(v8 + 224), *(_BYTE **)(v42 + 224), *(_QWORD *)(v42 + 224) + *(_QWORD *)(v42 + 232));
          v45 = _mm_loadu_si128((const __m128i *)(v42 + 256));
          *(_QWORD *)(v8 + 272) = v8 + 288;
          *(__m128i *)(v8 + 256) = v45;
          sub_2F07250((__int64 *)(v8 + 272), *(_BYTE **)(v42 + 272), *(_QWORD *)(v42 + 272) + *(_QWORD *)(v42 + 280));
          v2 = v48;
          *(__m128i *)(v8 + 304) = _mm_loadu_si128((const __m128i *)(v42 + 304));
        }
        v42 += 320;
        v8 += 320;
      }
      while ( v42 != v2 );
    }
    else
    {
      if ( v49 <= 0 )
        goto LABEL_19;
      v11 = v5 + 3;
      v12 = v4 + 24;
      v13 = 0xCCCCCCCCCCCCCCCDLL * (v49 >> 6);
      do
      {
        *(__m128i *)(v11 - 3) = _mm_loadu_si128((const __m128i *)(v12 - 24));
        *(v11 - 1) = *(_QWORD *)(v12 - 8);
        sub_2240AE0(v11, (unsigned __int64 *)v12);
        *((__m128i *)v11 + 2) = _mm_loadu_si128((const __m128i *)(v12 + 32));
        *((_DWORD *)v11 + 12) = *(_DWORD *)(v12 + 48);
        v11[7] = *(_QWORD *)(v12 + 56);
        v11[8] = *(_QWORD *)(v12 + 64);
        *((_WORD *)v11 + 36) = *(_WORD *)(v12 + 72);
        *((_DWORD *)v11 + 19) = *(_DWORD *)(v12 + 76);
        sub_2240AE0(v11 + 10, (unsigned __int64 *)(v12 + 80));
        *((__m128i *)v11 + 7) = _mm_loadu_si128((const __m128i *)(v12 + 112));
        *((_BYTE *)v11 + 128) = *(_BYTE *)(v12 + 128);
        *(__m128i *)(v11 + 17) = _mm_loadu_si128((const __m128i *)(v12 + 136));
        sub_2240AE0(v11 + 19, (unsigned __int64 *)(v12 + 152));
        *(__m128i *)(v11 + 23) = _mm_loadu_si128((const __m128i *)(v12 + 184));
        sub_2240AE0(v11 + 25, (unsigned __int64 *)(v12 + 200));
        v14 = _mm_loadu_si128((const __m128i *)(v12 + 232));
        v15 = (unsigned __int64 *)(v12 + 248);
        v16 = v11 + 31;
        v12 += 320;
        v11 += 40;
        *(__m128i *)(v11 - 11) = v14;
        sub_2240AE0(v16, v15);
        *(__m128i *)(v11 - 5) = _mm_loadu_si128((const __m128i *)(v12 - 40));
        --v13;
      }
      while ( v13 );
      v5 = (_QWORD *)((char *)v5 + v49);
      while ( (_QWORD *)v8 != v5 )
      {
        v17 = v5[34];
        if ( (_QWORD *)v17 != v5 + 36 )
          j_j___libc_free_0(v17);
        v18 = v5[28];
        if ( (_QWORD *)v18 != v5 + 30 )
          j_j___libc_free_0(v18);
        v19 = v5[22];
        if ( (_QWORD *)v19 != v5 + 24 )
          j_j___libc_free_0(v19);
        v20 = v5[13];
        if ( (_QWORD *)v20 != v5 + 15 )
          j_j___libc_free_0(v20);
        v21 = v5[3];
        if ( (_QWORD *)v21 != v5 + 5 )
          j_j___libc_free_0(v21);
        v5 += 40;
LABEL_19:
        ;
      }
    }
    v22 = (char *)(*a1 + v49);
LABEL_21:
    a1[1] = (__int64)v22;
  }
}
