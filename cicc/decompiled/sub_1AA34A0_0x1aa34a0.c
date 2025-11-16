// Function: sub_1AA34A0
// Address: 0x1aa34a0
//
const __m128i *__fastcall sub_1AA34A0(const __m128i *src, const __m128i *a2, const __m128i *a3)
{
  char *v3; // rbx
  char *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r8
  const __m128i *v10; // rdx
  __int64 *v11; // rcx
  __int64 v12; // rdi
  __m128i v13; // xmm0
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // r11
  __int64 v19; // r10
  __int32 v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rdx
  __m128i *v23; // rdx
  __m128i *v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // r15
  __int64 v27; // r14
  __m128i v28; // xmm3
  __int64 v29; // r13
  __int64 v30; // r12
  __int64 v31; // r11
  __int64 v32; // r10
  __int32 v33; // r9d
  __int64 v34; // r9
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // r13
  int v41; // r12d
  size_t v42; // rdx
  const __m128i *v44; // rdx
  const __m128i *v45; // rax
  __m128i v46; // xmm6
  __int64 v47; // r12
  __int64 v48; // r11
  __int64 v49; // r10
  __int64 v50; // r9
  __int64 v51; // r8
  __int64 v52; // rdi
  __int32 v53; // esi
  size_t v54; // rdx
  __int64 v55; // [rsp+8h] [rbp-58h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  char *v60; // [rsp+20h] [rbp-40h]
  size_t v61; // [rsp+28h] [rbp-38h]
  __int64 v62; // [rsp+28h] [rbp-38h]

  if ( src == a2 )
    return a3;
  v3 = (char *)src;
  if ( a2 == a3 )
    return src;
  v6 = &src->m128i_i8[(char *)a3 - (char *)a2];
  v7 = 0x6DB6DB6DB6DB6DB7LL * (((char *)a2 - (char *)src) >> 3);
  v60 = v6;
  v8 = 0x6DB6DB6DB6DB6DB7LL * (((char *)a3 - (char *)src) >> 3);
  if ( v7 == v8 - v7 )
  {
    v44 = a2;
    v45 = src;
    do
    {
      v46 = _mm_loadu_si128(v44);
      v47 = v45->m128i_i64[0];
      v45 = (const __m128i *)((char *)v45 + 56);
      v44 = (const __m128i *)((char *)v44 + 56);
      v48 = v45[-3].m128i_i64[0];
      v49 = v45[-3].m128i_i64[1];
      *(__m128i *)((char *)v45 - 56) = v46;
      v50 = v45[-2].m128i_i64[0];
      v51 = v45[-2].m128i_i64[1];
      v52 = v45[-1].m128i_i64[0];
      *(const __m128i *)((char *)&v45[-3] + 8) = _mm_loadu_si128((const __m128i *)((char *)v44 - 40));
      v53 = v45[-1].m128i_i32[2];
      *(const __m128i *)((char *)&v45[-2] + 8) = _mm_loadu_si128((const __m128i *)((char *)v44 - 24));
      v45[-1].m128i_i64[1] = v44[-1].m128i_i64[1];
      v44[-4].m128i_i64[1] = v47;
      v44[-3].m128i_i64[0] = v48;
      v44[-3].m128i_i64[1] = v49;
      v44[-2].m128i_i64[0] = v50;
      v44[-2].m128i_i64[1] = v51;
      v44[-1].m128i_i64[0] = v52;
      v44[-1].m128i_i32[2] = v53;
    }
    while ( a2 != v45 );
    return (const __m128i *)&v3[56
                              * ((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)&a2[-4].m128i_u64[1] - v3) >> 3))
                               & 0x1FFFFFFFFFFFFFFFLL)
                              + 56];
  }
  else
  {
    while ( 1 )
    {
      v9 = v8 - v7;
      if ( v7 < v8 - v7 )
        break;
LABEL_12:
      v22 = 56 * v8;
      if ( v9 == 1 )
      {
        v54 = v22 - 56;
        v34 = *(_QWORD *)&v3[v54];
        v35 = *(_QWORD *)&v3[v54 + 8];
        v36 = *(_QWORD *)&v3[v54 + 16];
        v37 = *(_QWORD *)&v3[v54 + 24];
        v39 = *(_QWORD *)&v3[v54 + 32];
        v40 = *(_QWORD *)&v3[v54 + 40];
        v41 = *(_DWORD *)&v3[v54 + 48];
        if ( v3 != &v3[v54] )
        {
          v57 = *(_QWORD *)&v3[v54 + 16];
          v59 = *(_QWORD *)&v3[v54 + 8];
          v62 = *(_QWORD *)&v3[v54];
          memmove(v3 + 56, v3, v54);
          v36 = v57;
          v35 = v59;
          v34 = v62;
        }
        goto LABEL_22;
      }
      v23 = (__m128i *)&v3[v22];
      v3 = &v23->m128i_i8[-56 * v9];
      if ( v7 > 0 )
      {
        v24 = (__m128i *)((char *)v23 - 56 * v9);
        v25 = 0;
        do
        {
          v26 = v24[-4].m128i_i64[1];
          v27 = v24[-3].m128i_i64[0];
          ++v25;
          v24 = (__m128i *)((char *)v24 - 56);
          v28 = _mm_loadu_si128((__m128i *)((char *)v23 - 56));
          v29 = v24[1].m128i_i64[0];
          v23 = (__m128i *)((char *)v23 - 56);
          v30 = v24[1].m128i_i64[1];
          v31 = v24[2].m128i_i64[0];
          *v24 = v28;
          v32 = v24[2].m128i_i64[1];
          v33 = v24[3].m128i_i32[0];
          v24[1] = _mm_loadu_si128(v23 + 1);
          v24[2] = _mm_loadu_si128(v23 + 2);
          v24[3].m128i_i64[0] = v23[3].m128i_i64[0];
          v23->m128i_i64[0] = v26;
          v23->m128i_i64[1] = v27;
          v23[1].m128i_i64[0] = v29;
          v23[1].m128i_i64[1] = v30;
          v23[2].m128i_i64[0] = v31;
          v23[2].m128i_i64[1] = v32;
          v23[3].m128i_i32[0] = v33;
        }
        while ( v7 != v25 );
        v3 -= 56 * v7;
      }
      v7 = v8 % v9;
      if ( !(v8 % v9) )
        return (const __m128i *)v60;
      v8 = v9;
    }
    while ( v7 != 1 )
    {
      v10 = (const __m128i *)&v3[56 * v7];
      if ( v9 > 0 )
      {
        v11 = (__int64 *)v3;
        v12 = 0;
        do
        {
          v13 = _mm_loadu_si128(v10);
          v14 = *v11;
          ++v12;
          v11 += 7;
          v15 = *(v11 - 6);
          v16 = *(v11 - 5);
          v10 = (const __m128i *)((char *)v10 + 56);
          *(__m128i *)(v11 - 7) = v13;
          v17 = *(v11 - 4);
          v18 = *(v11 - 3);
          v19 = *(v11 - 2);
          *(__m128i *)(v11 - 5) = _mm_loadu_si128((const __m128i *)((char *)v10 - 40));
          v20 = *((_DWORD *)v11 - 2);
          *(__m128i *)(v11 - 3) = _mm_loadu_si128((const __m128i *)((char *)v10 - 24));
          *(v11 - 1) = v10[-1].m128i_i64[1];
          v10[-4].m128i_i64[1] = v14;
          v10[-3].m128i_i64[0] = v15;
          v10[-3].m128i_i64[1] = v16;
          v10[-2].m128i_i64[0] = v17;
          v10[-2].m128i_i64[1] = v18;
          v10[-1].m128i_i64[0] = v19;
          v10[-1].m128i_i32[2] = v20;
        }
        while ( v9 != v12 );
        v3 += 56 * v9;
      }
      v21 = v8 % v7;
      if ( !(v8 % v7) )
        return (const __m128i *)v60;
      v8 = v7;
      v7 -= v21;
      v9 = v8 - v7;
      if ( v7 >= v8 - v7 )
        goto LABEL_12;
    }
    v34 = *(_QWORD *)v3;
    v35 = *((_QWORD *)v3 + 1);
    v36 = *((_QWORD *)v3 + 2);
    v37 = *((_QWORD *)v3 + 3);
    v38 = 56 * v8;
    v39 = *((_QWORD *)v3 + 4);
    v40 = *((_QWORD *)v3 + 5);
    v41 = *((_DWORD *)v3 + 12);
    v42 = v38 - 56;
    if ( v3 + 56 != &v3[v38] )
    {
      v55 = *((_QWORD *)v3 + 2);
      v56 = *((_QWORD *)v3 + 1);
      v58 = *(_QWORD *)v3;
      v61 = v38 - 56;
      memmove(v3, v3 + 56, v42);
      v36 = v55;
      v35 = v56;
      v34 = v58;
      v42 = v61;
    }
    v3 += v42;
LABEL_22:
    *(_QWORD *)v3 = v34;
    *((_QWORD *)v3 + 1) = v35;
    *((_QWORD *)v3 + 2) = v36;
    *((_QWORD *)v3 + 3) = v37;
    *((_QWORD *)v3 + 4) = v39;
    *((_QWORD *)v3 + 5) = v40;
    *((_DWORD *)v3 + 12) = v41;
  }
  return (const __m128i *)v60;
}
