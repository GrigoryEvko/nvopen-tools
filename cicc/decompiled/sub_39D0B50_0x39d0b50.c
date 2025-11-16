// Function: sub_39D0B50
// Address: 0x39d0b50
//
void __fastcall sub_39D0B50(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  _QWORD *v4; // r14
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rdi
  _QWORD *v11; // r13
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  __m128i v14; // xmm4
  unsigned __int64 *v15; // rsi
  unsigned __int64 *v16; // rdi
  bool v17; // zf
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  char *v23; // r14
  __int64 v24; // r14
  __int64 v25; // r12
  __int64 i; // r15
  char v27; // dl
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  _QWORD *v30; // rbx
  _QWORD *v31; // r15
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // r13
  _QWORD *v38; // r14
  __int64 v39; // rbx
  __m128i v40; // xmm2
  unsigned __int64 *v41; // rsi
  unsigned __int64 *v42; // rdi
  __int64 v43; // r13
  char v44; // al
  __m128i v45; // xmm7
  __m128i v46; // xmm4
  signed __int64 v47; // [rsp+10h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a2[1];
    v4 = (_QWORD *)*a1;
    v5 = *a2;
    v6 = v2 - v5;
    v7 = a1[2] - *a1;
    v47 = v2 - v5;
    if ( v7 < v2 - v5 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFF80LL )
          sub_4261EA(a1, v5, v7);
        v24 = sub_22077B0(v47);
      }
      else
      {
        v24 = 0;
      }
      v25 = v5;
      for ( i = v24; v2 != v25; i += 320 )
      {
        if ( i )
        {
          *(__m128i *)i = _mm_loadu_si128((const __m128i *)v25);
          *(_QWORD *)(i + 16) = *(_QWORD *)(v25 + 16);
          *(_QWORD *)(i + 24) = i + 40;
          sub_39CF630((__int64 *)(i + 24), *(_BYTE **)(v25 + 24), *(_QWORD *)(v25 + 24) + *(_QWORD *)(v25 + 32));
          *(__m128i *)(i + 56) = _mm_loadu_si128((const __m128i *)(v25 + 56));
          *(_DWORD *)(i + 72) = *(_DWORD *)(v25 + 72);
          *(_QWORD *)(i + 80) = *(_QWORD *)(v25 + 80);
          *(_QWORD *)(i + 88) = *(_QWORD *)(v25 + 88);
          *(_DWORD *)(i + 96) = *(_DWORD *)(v25 + 96);
          *(_BYTE *)(i + 100) = *(_BYTE *)(v25 + 100);
          *(_QWORD *)(i + 104) = i + 120;
          sub_39CF630((__int64 *)(i + 104), *(_BYTE **)(v25 + 104), *(_QWORD *)(v25 + 104) + *(_QWORD *)(v25 + 112));
          *(__m128i *)(i + 136) = _mm_loadu_si128((const __m128i *)(v25 + 136));
          *(_BYTE *)(i + 152) = *(_BYTE *)(v25 + 152);
          v27 = *(_BYTE *)(v25 + 168);
          *(_BYTE *)(i + 168) = v27;
          if ( v27 )
            *(_QWORD *)(i + 160) = *(_QWORD *)(v25 + 160);
          *(_QWORD *)(i + 176) = i + 192;
          sub_39CF630((__int64 *)(i + 176), *(_BYTE **)(v25 + 176), *(_QWORD *)(v25 + 176) + *(_QWORD *)(v25 + 184));
          v28 = _mm_loadu_si128((const __m128i *)(v25 + 208));
          *(_QWORD *)(i + 224) = i + 240;
          *(__m128i *)(i + 208) = v28;
          sub_39CF630((__int64 *)(i + 224), *(_BYTE **)(v25 + 224), *(_QWORD *)(v25 + 224) + *(_QWORD *)(v25 + 232));
          v29 = _mm_loadu_si128((const __m128i *)(v25 + 256));
          *(_QWORD *)(i + 272) = i + 288;
          *(__m128i *)(i + 256) = v29;
          sub_39CF630((__int64 *)(i + 272), *(_BYTE **)(v25 + 272), *(_QWORD *)(v25 + 272) + *(_QWORD *)(v25 + 280));
          *(__m128i *)(i + 304) = _mm_loadu_si128((const __m128i *)(v25 + 304));
        }
        v25 += 320;
      }
      v30 = (_QWORD *)a1[1];
      v31 = (_QWORD *)*a1;
      if ( v30 != (_QWORD *)*a1 )
      {
        do
        {
          v32 = v31[34];
          if ( (_QWORD *)v32 != v31 + 36 )
            j_j___libc_free_0(v32);
          v33 = v31[28];
          if ( (_QWORD *)v33 != v31 + 30 )
            j_j___libc_free_0(v33);
          v34 = v31[22];
          if ( (_QWORD *)v34 != v31 + 24 )
            j_j___libc_free_0(v34);
          v35 = v31[13];
          if ( (_QWORD *)v35 != v31 + 15 )
            j_j___libc_free_0(v35);
          v36 = v31[3];
          if ( (_QWORD *)v36 != v31 + 5 )
            j_j___libc_free_0(v36);
          v31 += 40;
        }
        while ( v30 != v31 );
        v31 = (_QWORD *)*a1;
      }
      if ( v31 )
        j_j___libc_free_0((unsigned __int64)v31);
      *a1 = v24;
      v23 = (char *)(v47 + v24);
      a1[2] = (__int64)v23;
      goto LABEL_25;
    }
    v8 = a1[1];
    v9 = v8 - (_QWORD)v4;
    v10 = v8 - (_QWORD)v4;
    if ( v47 > (unsigned __int64)(v8 - (_QWORD)v4) )
    {
      v37 = 0xCCCCCCCCCCCCCCCDLL * (v9 >> 6);
      if ( v9 > 0 )
      {
        v38 = v4 + 3;
        v39 = v5 + 24;
        do
        {
          *(__m128i *)(v38 - 3) = _mm_loadu_si128((const __m128i *)(v39 - 24));
          *(v38 - 1) = *(_QWORD *)(v39 - 8);
          sub_2240AE0(v38, (unsigned __int64 *)v39);
          *((__m128i *)v38 + 2) = _mm_loadu_si128((const __m128i *)(v39 + 32));
          *((_DWORD *)v38 + 12) = *(_DWORD *)(v39 + 48);
          v38[7] = *(_QWORD *)(v39 + 56);
          v38[8] = *(_QWORD *)(v39 + 64);
          *((_DWORD *)v38 + 18) = *(_DWORD *)(v39 + 72);
          *((_BYTE *)v38 + 76) = *(_BYTE *)(v39 + 76);
          sub_2240AE0(v38 + 10, (unsigned __int64 *)(v39 + 80));
          *((__m128i *)v38 + 7) = _mm_loadu_si128((const __m128i *)(v39 + 112));
          *((_BYTE *)v38 + 128) = *(_BYTE *)(v39 + 128);
          if ( *(_BYTE *)(v39 + 144) )
          {
            v17 = *((_BYTE *)v38 + 144) == 0;
            v38[17] = *(_QWORD *)(v39 + 136);
            if ( v17 )
              *((_BYTE *)v38 + 144) = 1;
          }
          else if ( *((_BYTE *)v38 + 144) )
          {
            *((_BYTE *)v38 + 144) = 0;
          }
          sub_2240AE0(v38 + 19, (unsigned __int64 *)(v39 + 152));
          *(__m128i *)(v38 + 23) = _mm_loadu_si128((const __m128i *)(v39 + 184));
          sub_2240AE0(v38 + 25, (unsigned __int64 *)(v39 + 200));
          v40 = _mm_loadu_si128((const __m128i *)(v39 + 232));
          v41 = (unsigned __int64 *)(v39 + 248);
          v42 = v38 + 31;
          v39 += 320;
          v38 += 40;
          *(__m128i *)(v38 - 11) = v40;
          sub_2240AE0(v42, v41);
          *(__m128i *)(v38 - 5) = _mm_loadu_si128((const __m128i *)(v39 - 40));
          --v37;
        }
        while ( v37 );
        v2 = a2[1];
        v5 = *a2;
        v8 = a1[1];
        v4 = (_QWORD *)*a1;
        v10 = v8 - *a1;
      }
      v43 = v5 + v10;
      v23 = (char *)v4 + v47;
      if ( v5 + v10 == v2 )
        goto LABEL_25;
      do
      {
        if ( v8 )
        {
          *(__m128i *)v8 = _mm_loadu_si128((const __m128i *)v43);
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v43 + 16);
          *(_QWORD *)(v8 + 24) = v8 + 40;
          sub_39CF630((__int64 *)(v8 + 24), *(_BYTE **)(v43 + 24), *(_QWORD *)(v43 + 24) + *(_QWORD *)(v43 + 32));
          *(__m128i *)(v8 + 56) = _mm_loadu_si128((const __m128i *)(v43 + 56));
          *(_DWORD *)(v8 + 72) = *(_DWORD *)(v43 + 72);
          *(_QWORD *)(v8 + 80) = *(_QWORD *)(v43 + 80);
          *(_QWORD *)(v8 + 88) = *(_QWORD *)(v43 + 88);
          *(_DWORD *)(v8 + 96) = *(_DWORD *)(v43 + 96);
          *(_BYTE *)(v8 + 100) = *(_BYTE *)(v43 + 100);
          *(_QWORD *)(v8 + 104) = v8 + 120;
          sub_39CF630((__int64 *)(v8 + 104), *(_BYTE **)(v43 + 104), *(_QWORD *)(v43 + 104) + *(_QWORD *)(v43 + 112));
          *(__m128i *)(v8 + 136) = _mm_loadu_si128((const __m128i *)(v43 + 136));
          *(_BYTE *)(v8 + 152) = *(_BYTE *)(v43 + 152);
          v44 = *(_BYTE *)(v43 + 168);
          *(_BYTE *)(v8 + 168) = v44;
          if ( v44 )
            *(_QWORD *)(v8 + 160) = *(_QWORD *)(v43 + 160);
          *(_QWORD *)(v8 + 176) = v8 + 192;
          sub_39CF630((__int64 *)(v8 + 176), *(_BYTE **)(v43 + 176), *(_QWORD *)(v43 + 176) + *(_QWORD *)(v43 + 184));
          v45 = _mm_loadu_si128((const __m128i *)(v43 + 208));
          *(_QWORD *)(v8 + 224) = v8 + 240;
          *(__m128i *)(v8 + 208) = v45;
          sub_39CF630((__int64 *)(v8 + 224), *(_BYTE **)(v43 + 224), *(_QWORD *)(v43 + 224) + *(_QWORD *)(v43 + 232));
          v46 = _mm_loadu_si128((const __m128i *)(v43 + 256));
          *(_QWORD *)(v8 + 272) = v8 + 288;
          *(__m128i *)(v8 + 256) = v46;
          sub_39CF630((__int64 *)(v8 + 272), *(_BYTE **)(v43 + 272), *(_QWORD *)(v43 + 272) + *(_QWORD *)(v43 + 280));
          *(__m128i *)(v8 + 304) = _mm_loadu_si128((const __m128i *)(v43 + 304));
        }
        v43 += 320;
        v8 += 320;
      }
      while ( v43 != v2 );
    }
    else
    {
      if ( v47 > 0 )
      {
        v11 = v4 + 3;
        v12 = v5 + 24;
        v13 = 0xCCCCCCCCCCCCCCCDLL * (v47 >> 6);
        do
        {
          *(__m128i *)(v11 - 3) = _mm_loadu_si128((const __m128i *)(v12 - 24));
          *(v11 - 1) = *(_QWORD *)(v12 - 8);
          sub_2240AE0(v11, (unsigned __int64 *)v12);
          *((__m128i *)v11 + 2) = _mm_loadu_si128((const __m128i *)(v12 + 32));
          *((_DWORD *)v11 + 12) = *(_DWORD *)(v12 + 48);
          v11[7] = *(_QWORD *)(v12 + 56);
          v11[8] = *(_QWORD *)(v12 + 64);
          *((_DWORD *)v11 + 18) = *(_DWORD *)(v12 + 72);
          *((_BYTE *)v11 + 76) = *(_BYTE *)(v12 + 76);
          sub_2240AE0(v11 + 10, (unsigned __int64 *)(v12 + 80));
          *((__m128i *)v11 + 7) = _mm_loadu_si128((const __m128i *)(v12 + 112));
          *((_BYTE *)v11 + 128) = *(_BYTE *)(v12 + 128);
          if ( *(_BYTE *)(v12 + 144) )
          {
            v17 = *((_BYTE *)v11 + 144) == 0;
            v11[17] = *(_QWORD *)(v12 + 136);
            if ( v17 )
              *((_BYTE *)v11 + 144) = 1;
          }
          else if ( *((_BYTE *)v11 + 144) )
          {
            *((_BYTE *)v11 + 144) = 0;
          }
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
        v4 = (_QWORD *)((char *)v4 + v47);
      }
      while ( (_QWORD *)v8 != v4 )
      {
        v18 = v4[34];
        if ( (_QWORD *)v18 != v4 + 36 )
          j_j___libc_free_0(v18);
        v19 = v4[28];
        if ( (_QWORD *)v19 != v4 + 30 )
          j_j___libc_free_0(v19);
        v20 = v4[22];
        if ( (_QWORD *)v20 != v4 + 24 )
          j_j___libc_free_0(v20);
        v21 = v4[13];
        if ( (_QWORD *)v21 != v4 + 15 )
          j_j___libc_free_0(v21);
        v22 = v4[3];
        if ( (_QWORD *)v22 != v4 + 5 )
          j_j___libc_free_0(v22);
        v4 += 40;
      }
    }
    v23 = (char *)(*a1 + v47);
LABEL_25:
    a1[1] = (__int64)v23;
  }
}
