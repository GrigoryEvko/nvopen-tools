// Function: sub_2F0B340
// Address: 0x2f0b340
//
void __fastcall sub_2F0B340(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v4; // rcx
  _QWORD *v5; // r15
  unsigned __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __m128i *v10; // r13
  unsigned __int64 v11; // r12
  __m128i *v12; // r14
  __m128i v13; // xmm4
  unsigned __int64 **v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 *v16; // r14
  unsigned __int64 *v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  char *v20; // r15
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 i; // r13
  __m128i v24; // xmm6
  unsigned __int64 v25; // rbx
  __m128i *v26; // r12
  __int64 v27; // rbx
  __int64 v28; // r15
  _QWORD *v29; // r14
  _QWORD *v30; // r13
  unsigned __int64 *v31; // rbx
  unsigned __int64 *v32; // r12
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // r14
  __m128i *v36; // r15
  __m128i *v37; // rbx
  __m128i v38; // xmm7
  unsigned __int64 **v39; // rsi
  __int64 m128i_i64; // rdi
  __int64 v41; // r14
  __m128i v42; // xmm3
  unsigned __int64 v43; // r13
  __m128i *v44; // r15
  __int64 v45; // r13
  __int64 v46; // r12
  __int64 v47; // [rsp+8h] [rbp-58h]
  __int64 v48; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+18h] [rbp-48h]
  __int64 *v50; // [rsp+20h] [rbp-40h]
  __int64 v51; // [rsp+28h] [rbp-38h]

  v50 = a1;
  if ( a2 != a1 )
  {
    v3 = a2;
    v4 = *a2;
    v5 = (_QWORD *)*a1;
    v51 = a2[1];
    v6 = v51 - *a2;
    v49 = v6;
    if ( a1[2] - *a1 < v6 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFFB0LL )
LABEL_76:
          sub_4261EA(a1, a2, a3);
        v47 = *a2;
        v21 = sub_22077B0(v6);
        v4 = v47;
        v48 = v21;
      }
      else
      {
        v48 = 0;
      }
      v22 = v48;
      for ( i = v4; v51 != i; i += 144 )
      {
        if ( v22 )
        {
          *(__m128i *)v22 = _mm_loadu_si128((const __m128i *)i);
          *(_QWORD *)(v22 + 16) = *(_QWORD *)(i + 16);
          *(_QWORD *)(v22 + 24) = v22 + 40;
          sub_2F07250((__int64 *)(v22 + 24), *(_BYTE **)(i + 24), *(_QWORD *)(i + 24) + *(_QWORD *)(i + 32));
          v24 = _mm_loadu_si128((const __m128i *)(i + 56));
          a1 = (__int64 *)(v22 + 72);
          *(_QWORD *)(v22 + 72) = v22 + 88;
          *(__m128i *)(v22 + 56) = v24;
          a2 = *(__int64 **)(i + 72);
          sub_2F07250((__int64 *)(v22 + 72), a2, (__int64)a2 + *(_QWORD *)(i + 80));
          *(__m128i *)(v22 + 104) = _mm_loadu_si128((const __m128i *)(i + 104));
          v25 = *(_QWORD *)(i + 128) - *(_QWORD *)(i + 120);
          *(_QWORD *)(v22 + 120) = 0;
          *(_QWORD *)(v22 + 128) = 0;
          *(_QWORD *)(v22 + 136) = 0;
          if ( v25 )
          {
            if ( v25 > 0x7FFFFFFFFFFFFFE0LL )
              goto LABEL_76;
            v26 = (__m128i *)sub_22077B0(v25);
          }
          else
          {
            v25 = 0;
            v26 = 0;
          }
          *(_QWORD *)(v22 + 120) = v26;
          *(_QWORD *)(v22 + 128) = v26;
          *(_QWORD *)(v22 + 136) = (char *)v26 + v25;
          v27 = *(_QWORD *)(i + 128);
          if ( v27 != *(_QWORD *)(i + 120) )
          {
            v28 = *(_QWORD *)(i + 120);
            do
            {
              if ( v26 )
              {
                v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
                sub_2F07250(v26->m128i_i64, *(_BYTE **)v28, *(_QWORD *)v28 + *(_QWORD *)(v28 + 8));
                v26[2] = _mm_loadu_si128((const __m128i *)(v28 + 32));
              }
              v28 += 48;
              v26 += 3;
            }
            while ( v27 != v28 );
          }
          *(_QWORD *)(v22 + 128) = v26;
        }
        v22 += 144;
      }
      v29 = (_QWORD *)v50[1];
      v30 = (_QWORD *)*v50;
      if ( v29 != (_QWORD *)*v50 )
      {
        do
        {
          v31 = (unsigned __int64 *)v30[16];
          v32 = (unsigned __int64 *)v30[15];
          if ( v31 != v32 )
          {
            do
            {
              if ( (unsigned __int64 *)*v32 != v32 + 2 )
                j_j___libc_free_0(*v32);
              v32 += 6;
            }
            while ( v31 != v32 );
            v32 = (unsigned __int64 *)v30[15];
          }
          if ( v32 )
            j_j___libc_free_0((unsigned __int64)v32);
          v33 = v30[9];
          if ( (_QWORD *)v33 != v30 + 11 )
            j_j___libc_free_0(v33);
          v34 = v30[3];
          if ( (_QWORD *)v34 != v30 + 5 )
            j_j___libc_free_0(v34);
          v30 += 18;
        }
        while ( v29 != v30 );
        v30 = (_QWORD *)*v50;
      }
      if ( v30 )
        j_j___libc_free_0((unsigned __int64)v30);
      v20 = (char *)(v48 + v49);
      *v50 = v48;
      v50[2] = v48 + v49;
      goto LABEL_22;
    }
    v7 = a1[1];
    v8 = v7 - (_QWORD)v5;
    v9 = v7 - (_QWORD)v5;
    if ( v49 > (unsigned __int64)(v7 - (_QWORD)v5) )
    {
      v35 = 0x8E38E38E38E38E39LL * (v8 >> 4);
      if ( v8 > 0 )
      {
        v36 = (__m128i *)(v5 + 3);
        v37 = (__m128i *)(v4 + 24);
        do
        {
          *(__m128i *)((char *)v36 - 24) = _mm_loadu_si128((__m128i *)((char *)v37 - 24));
          v36[-1].m128i_i64[1] = v37[-1].m128i_i64[1];
          sub_2240AE0((unsigned __int64 *)v36, (unsigned __int64 *)v37);
          v36[2] = _mm_loadu_si128(v37 + 2);
          sub_2240AE0((unsigned __int64 *)&v36[3], (unsigned __int64 *)&v37[3]);
          v38 = _mm_loadu_si128(v37 + 5);
          v39 = (unsigned __int64 **)&v37[6];
          m128i_i64 = (__int64)v36[6].m128i_i64;
          v37 += 9;
          v36 += 9;
          v36[-4] = v38;
          sub_2F08860(m128i_i64, v39);
          --v35;
        }
        while ( v35 );
        v7 = v50[1];
        v5 = (_QWORD *)*v50;
        v51 = v3[1];
        v4 = *v3;
        v9 = v7 - *v50;
      }
      v41 = v4 + v9;
      v20 = (char *)v5 + v49;
      if ( v4 + v9 == v51 )
        goto LABEL_22;
      do
      {
        if ( v7 )
        {
          *(__m128i *)v7 = _mm_loadu_si128((const __m128i *)v41);
          *(_QWORD *)(v7 + 16) = *(_QWORD *)(v41 + 16);
          *(_QWORD *)(v7 + 24) = v7 + 40;
          sub_2F07250((__int64 *)(v7 + 24), *(_BYTE **)(v41 + 24), *(_QWORD *)(v41 + 24) + *(_QWORD *)(v41 + 32));
          v42 = _mm_loadu_si128((const __m128i *)(v41 + 56));
          a1 = (__int64 *)(v7 + 72);
          *(_QWORD *)(v7 + 72) = v7 + 88;
          *(__m128i *)(v7 + 56) = v42;
          a2 = *(__int64 **)(v41 + 72);
          sub_2F07250((__int64 *)(v7 + 72), a2, (__int64)a2 + *(_QWORD *)(v41 + 80));
          *(__m128i *)(v7 + 104) = _mm_loadu_si128((const __m128i *)(v41 + 104));
          v43 = *(_QWORD *)(v41 + 128) - *(_QWORD *)(v41 + 120);
          *(_QWORD *)(v7 + 120) = 0;
          *(_QWORD *)(v7 + 128) = 0;
          *(_QWORD *)(v7 + 136) = 0;
          if ( v43 )
          {
            if ( v43 > 0x7FFFFFFFFFFFFFE0LL )
              goto LABEL_76;
            v44 = (__m128i *)sub_22077B0(v43);
          }
          else
          {
            v44 = 0;
          }
          *(_QWORD *)(v7 + 120) = v44;
          *(_QWORD *)(v7 + 128) = v44;
          *(_QWORD *)(v7 + 136) = (char *)v44 + v43;
          v45 = *(_QWORD *)(v41 + 128);
          if ( v45 != *(_QWORD *)(v41 + 120) )
          {
            v46 = *(_QWORD *)(v41 + 120);
            do
            {
              if ( v44 )
              {
                v44->m128i_i64[0] = (__int64)v44[1].m128i_i64;
                sub_2F07250(v44->m128i_i64, *(_BYTE **)v46, *(_QWORD *)v46 + *(_QWORD *)(v46 + 8));
                v44[2] = _mm_loadu_si128((const __m128i *)(v46 + 32));
              }
              v46 += 48;
              v44 += 3;
            }
            while ( v45 != v46 );
          }
          *(_QWORD *)(v7 + 128) = v44;
        }
        v41 += 144;
        v7 += 144;
      }
      while ( v41 != v51 );
    }
    else
    {
      if ( v49 > 0 )
      {
        v10 = (__m128i *)(v4 + 24);
        v11 = 0x8E38E38E38E38E39LL * (v49 >> 4);
        v12 = (__m128i *)(v5 + 3);
        do
        {
          *(__m128i *)((char *)v12 - 24) = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
          v12[-1].m128i_i64[1] = v10[-1].m128i_i64[1];
          sub_2240AE0((unsigned __int64 *)v12, (unsigned __int64 *)v10);
          v12[2] = _mm_loadu_si128(v10 + 2);
          sub_2240AE0((unsigned __int64 *)&v12[3], (unsigned __int64 *)&v10[3]);
          v13 = _mm_loadu_si128(v10 + 5);
          v14 = (unsigned __int64 **)&v10[6];
          v15 = (__int64)v12[6].m128i_i64;
          v10 += 9;
          v12 += 9;
          v12[-4] = v13;
          sub_2F08860(v15, v14);
          --v11;
        }
        while ( v11 );
        v5 = (_QWORD *)((char *)v5 + v49);
      }
      for ( ; (_QWORD *)v7 != v5; v5 += 18 )
      {
        v16 = (unsigned __int64 *)v5[16];
        v17 = (unsigned __int64 *)v5[15];
        if ( v16 != v17 )
        {
          do
          {
            if ( (unsigned __int64 *)*v17 != v17 + 2 )
              j_j___libc_free_0(*v17);
            v17 += 6;
          }
          while ( v16 != v17 );
          v17 = (unsigned __int64 *)v5[15];
        }
        if ( v17 )
          j_j___libc_free_0((unsigned __int64)v17);
        v18 = v5[9];
        if ( (_QWORD *)v18 != v5 + 11 )
          j_j___libc_free_0(v18);
        v19 = v5[3];
        if ( (_QWORD *)v19 != v5 + 5 )
          j_j___libc_free_0(v19);
      }
    }
    v20 = (char *)(*v50 + v49);
LABEL_22:
    v50[1] = (__int64)v20;
  }
}
