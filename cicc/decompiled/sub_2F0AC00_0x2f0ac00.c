// Function: sub_2F0AC00
// Address: 0x2f0ac00
//
void __fastcall sub_2F0AC00(__int64 *a1, __int64 *a2)
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
  __int64 v13; // r14
  __m128i v14; // xmm3
  unsigned __int64 *v15; // rsi
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  char *v21; // r13
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 i; // r14
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  _QWORD *v27; // rbx
  _QWORD *v28; // r14
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  __int64 v33; // r14
  _QWORD *v34; // r13
  __int64 v35; // rbx
  __m128i v36; // xmm0
  unsigned __int64 *v37; // rsi
  unsigned __int64 *v38; // rdi
  __int64 v39; // rbx
  __m128i v40; // xmm4
  __m128i v41; // xmm5
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+8h] [rbp-48h]
  __int64 v45; // [rsp+10h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a2[1];
    v4 = *a2;
    v5 = (_QWORD *)*a1;
    v6 = v2 - *a2;
    v7 = a1[2] - *a1;
    v45 = v6;
    if ( v7 < v6 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, v6, v7);
        v42 = v2;
        v22 = sub_22077B0(v6);
        v2 = v42;
        v23 = v22;
      }
      else
      {
        v23 = 0;
      }
      for ( i = v23; v2 != v4; i += 264 )
      {
        if ( i )
        {
          v43 = v2;
          *(__m128i *)i = _mm_loadu_si128((const __m128i *)v4);
          *(_QWORD *)(i + 16) = *(_QWORD *)(v4 + 16);
          *(_DWORD *)(i + 24) = *(_DWORD *)(v4 + 24);
          *(_QWORD *)(i + 32) = *(_QWORD *)(v4 + 32);
          *(_QWORD *)(i + 40) = *(_QWORD *)(v4 + 40);
          *(_WORD *)(i + 48) = *(_WORD *)(v4 + 48);
          *(_DWORD *)(i + 52) = *(_DWORD *)(v4 + 52);
          *(_BYTE *)(i + 56) = *(_BYTE *)(v4 + 56);
          *(_BYTE *)(i + 57) = *(_BYTE *)(v4 + 57);
          *(_QWORD *)(i + 64) = i + 80;
          sub_2F07250((__int64 *)(i + 64), *(_BYTE **)(v4 + 64), *(_QWORD *)(v4 + 64) + *(_QWORD *)(v4 + 72));
          *(__m128i *)(i + 96) = _mm_loadu_si128((const __m128i *)(v4 + 96));
          *(_BYTE *)(i + 112) = *(_BYTE *)(v4 + 112);
          *(_QWORD *)(i + 120) = i + 136;
          sub_2F07250((__int64 *)(i + 120), *(_BYTE **)(v4 + 120), *(_QWORD *)(v4 + 120) + *(_QWORD *)(v4 + 128));
          v25 = _mm_loadu_si128((const __m128i *)(v4 + 152));
          *(_QWORD *)(i + 168) = i + 184;
          *(__m128i *)(i + 152) = v25;
          sub_2F07250((__int64 *)(i + 168), *(_BYTE **)(v4 + 168), *(_QWORD *)(v4 + 168) + *(_QWORD *)(v4 + 176));
          v26 = _mm_loadu_si128((const __m128i *)(v4 + 200));
          *(_QWORD *)(i + 216) = i + 232;
          *(__m128i *)(i + 200) = v26;
          sub_2F07250((__int64 *)(i + 216), *(_BYTE **)(v4 + 216), *(_QWORD *)(v4 + 216) + *(_QWORD *)(v4 + 224));
          v2 = v43;
          *(__m128i *)(i + 248) = _mm_loadu_si128((const __m128i *)(v4 + 248));
        }
        v4 += 264;
      }
      v27 = (_QWORD *)a1[1];
      v28 = (_QWORD *)*a1;
      if ( v27 != (_QWORD *)*a1 )
      {
        do
        {
          v29 = v28[27];
          if ( (_QWORD *)v29 != v28 + 29 )
            j_j___libc_free_0(v29);
          v30 = v28[21];
          if ( (_QWORD *)v30 != v28 + 23 )
            j_j___libc_free_0(v30);
          v31 = v28[15];
          if ( (_QWORD *)v31 != v28 + 17 )
            j_j___libc_free_0(v31);
          v32 = v28[8];
          if ( (_QWORD *)v32 != v28 + 10 )
            j_j___libc_free_0(v32);
          v28 += 33;
        }
        while ( v27 != v28 );
        v28 = (_QWORD *)*a1;
      }
      if ( v28 )
        j_j___libc_free_0((unsigned __int64)v28);
      *a1 = v23;
      v21 = (char *)(v6 + v23);
      a1[2] = (__int64)v21;
      goto LABEL_19;
    }
    v8 = a1[1];
    v9 = v8 - (_QWORD)v5;
    v10 = v8 - (_QWORD)v5;
    if ( v45 > (unsigned __int64)(v8 - (_QWORD)v5) )
    {
      v33 = 0xF83E0F83E0F83E1LL * (v9 >> 3);
      if ( v9 > 0 )
      {
        v34 = v5 + 8;
        v35 = v4 + 64;
        do
        {
          *((__m128i *)v34 - 4) = _mm_loadu_si128((const __m128i *)(v35 - 64));
          *(v34 - 6) = *(_QWORD *)(v35 - 48);
          *((_DWORD *)v34 - 10) = *(_DWORD *)(v35 - 40);
          *(v34 - 4) = *(_QWORD *)(v35 - 32);
          *(v34 - 3) = *(_QWORD *)(v35 - 24);
          *((_WORD *)v34 - 8) = *(_WORD *)(v35 - 16);
          *((_DWORD *)v34 - 3) = *(_DWORD *)(v35 - 12);
          *((_BYTE *)v34 - 8) = *(_BYTE *)(v35 - 8);
          *((_BYTE *)v34 - 7) = *(_BYTE *)(v35 - 7);
          sub_2240AE0(v34, (unsigned __int64 *)v35);
          *((__m128i *)v34 + 2) = _mm_loadu_si128((const __m128i *)(v35 + 32));
          *((_BYTE *)v34 + 48) = *(_BYTE *)(v35 + 48);
          sub_2240AE0(v34 + 7, (unsigned __int64 *)(v35 + 56));
          *(__m128i *)(v34 + 11) = _mm_loadu_si128((const __m128i *)(v35 + 88));
          sub_2240AE0(v34 + 13, (unsigned __int64 *)(v35 + 104));
          v36 = _mm_loadu_si128((const __m128i *)(v35 + 136));
          v37 = (unsigned __int64 *)(v35 + 152);
          v38 = v34 + 19;
          v35 += 264;
          v34 += 33;
          *((__m128i *)v34 - 8) = v36;
          sub_2240AE0(v38, v37);
          *((__m128i *)v34 - 5) = _mm_loadu_si128((const __m128i *)(v35 - 80));
          --v33;
        }
        while ( v33 );
        v2 = a2[1];
        v4 = *a2;
        v8 = a1[1];
        v5 = (_QWORD *)*a1;
        v10 = v8 - *a1;
      }
      v39 = v10 + v4;
      v21 = (char *)v5 + v45;
      if ( v39 == v2 )
        goto LABEL_19;
      do
      {
        if ( v8 )
        {
          v44 = v2;
          *(__m128i *)v8 = _mm_loadu_si128((const __m128i *)v39);
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v39 + 16);
          *(_DWORD *)(v8 + 24) = *(_DWORD *)(v39 + 24);
          *(_QWORD *)(v8 + 32) = *(_QWORD *)(v39 + 32);
          *(_QWORD *)(v8 + 40) = *(_QWORD *)(v39 + 40);
          *(_WORD *)(v8 + 48) = *(_WORD *)(v39 + 48);
          *(_DWORD *)(v8 + 52) = *(_DWORD *)(v39 + 52);
          *(_BYTE *)(v8 + 56) = *(_BYTE *)(v39 + 56);
          *(_BYTE *)(v8 + 57) = *(_BYTE *)(v39 + 57);
          *(_QWORD *)(v8 + 64) = v8 + 80;
          sub_2F07250((__int64 *)(v8 + 64), *(_BYTE **)(v39 + 64), *(_QWORD *)(v39 + 64) + *(_QWORD *)(v39 + 72));
          *(__m128i *)(v8 + 96) = _mm_loadu_si128((const __m128i *)(v39 + 96));
          *(_BYTE *)(v8 + 112) = *(_BYTE *)(v39 + 112);
          *(_QWORD *)(v8 + 120) = v8 + 136;
          sub_2F07250((__int64 *)(v8 + 120), *(_BYTE **)(v39 + 120), *(_QWORD *)(v39 + 120) + *(_QWORD *)(v39 + 128));
          v40 = _mm_loadu_si128((const __m128i *)(v39 + 152));
          *(_QWORD *)(v8 + 168) = v8 + 184;
          *(__m128i *)(v8 + 152) = v40;
          sub_2F07250((__int64 *)(v8 + 168), *(_BYTE **)(v39 + 168), *(_QWORD *)(v39 + 168) + *(_QWORD *)(v39 + 176));
          v41 = _mm_loadu_si128((const __m128i *)(v39 + 200));
          *(_QWORD *)(v8 + 216) = v8 + 232;
          *(__m128i *)(v8 + 200) = v41;
          sub_2F07250((__int64 *)(v8 + 216), *(_BYTE **)(v39 + 216), *(_QWORD *)(v39 + 216) + *(_QWORD *)(v39 + 224));
          v2 = v44;
          *(__m128i *)(v8 + 248) = _mm_loadu_si128((const __m128i *)(v39 + 248));
        }
        v39 += 264;
        v8 += 264;
      }
      while ( v39 != v2 );
    }
    else
    {
      if ( v45 <= 0 )
        goto LABEL_17;
      v11 = v5 + 8;
      v12 = v4 + 64;
      v13 = 0xF83E0F83E0F83E1LL * (v45 >> 3);
      do
      {
        *((__m128i *)v11 - 4) = _mm_loadu_si128((const __m128i *)(v12 - 64));
        *(v11 - 6) = *(_QWORD *)(v12 - 48);
        *((_DWORD *)v11 - 10) = *(_DWORD *)(v12 - 40);
        *(v11 - 4) = *(_QWORD *)(v12 - 32);
        *(v11 - 3) = *(_QWORD *)(v12 - 24);
        *((_WORD *)v11 - 8) = *(_WORD *)(v12 - 16);
        *((_DWORD *)v11 - 3) = *(_DWORD *)(v12 - 12);
        *((_BYTE *)v11 - 8) = *(_BYTE *)(v12 - 8);
        *((_BYTE *)v11 - 7) = *(_BYTE *)(v12 - 7);
        sub_2240AE0(v11, (unsigned __int64 *)v12);
        *((__m128i *)v11 + 2) = _mm_loadu_si128((const __m128i *)(v12 + 32));
        *((_BYTE *)v11 + 48) = *(_BYTE *)(v12 + 48);
        sub_2240AE0(v11 + 7, (unsigned __int64 *)(v12 + 56));
        *(__m128i *)(v11 + 11) = _mm_loadu_si128((const __m128i *)(v12 + 88));
        sub_2240AE0(v11 + 13, (unsigned __int64 *)(v12 + 104));
        v14 = _mm_loadu_si128((const __m128i *)(v12 + 136));
        v15 = (unsigned __int64 *)(v12 + 152);
        v16 = v11 + 19;
        v12 += 264;
        v11 += 33;
        *((__m128i *)v11 - 8) = v14;
        sub_2240AE0(v16, v15);
        *((__m128i *)v11 - 5) = _mm_loadu_si128((const __m128i *)(v12 - 80));
        --v13;
      }
      while ( v13 );
      v5 = (_QWORD *)((char *)v5 + v45);
      while ( (_QWORD *)v8 != v5 )
      {
        v17 = v5[27];
        if ( (_QWORD *)v17 != v5 + 29 )
          j_j___libc_free_0(v17);
        v18 = v5[21];
        if ( (_QWORD *)v18 != v5 + 23 )
          j_j___libc_free_0(v18);
        v19 = v5[15];
        if ( (_QWORD *)v19 != v5 + 17 )
          j_j___libc_free_0(v19);
        v20 = v5[8];
        if ( (_QWORD *)v20 != v5 + 10 )
          j_j___libc_free_0(v20);
        v5 += 33;
LABEL_17:
        ;
      }
    }
    v21 = (char *)(*a1 + v45);
LABEL_19:
    a1[1] = (__int64)v21;
  }
}
