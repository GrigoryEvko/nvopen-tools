// Function: sub_27B9290
// Address: 0x27b9290
//
void __fastcall sub_27B9290(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v8; // rdx
  __m128i *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r12
  int v12; // eax
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // r15
  __m128i *v17; // r12
  __m128i *v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __m128i v23; // xmm6
  __m128i v24; // xmm6
  __int64 v25; // rbx
  __int64 v26; // r12
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // [rsp+20h] [rbp-80h]
  __int64 v34; // [rsp+28h] [rbp-78h]
  __m128i *v35; // [rsp+30h] [rbp-70h]
  __int64 v37; // [rsp+48h] [rbp-58h]
  __int64 *v38; // [rsp+48h] [rbp-58h]

  v35 = a2;
  v6 = (char *)a2 - (char *)a1;
  v34 = a3;
  if ( v6 <= 512 )
    return;
  v8 = v6;
  if ( !a3 )
  {
    v38 = (__int64 *)v35;
    goto LABEL_22;
  }
  while ( 2 )
  {
    --v34;
    v9 = &a1[2 * (v8 >> 6)];
    v10 = a1[2].m128i_i64[1] + 24;
    v11 = v9->m128i_i64[1] + 24;
    v12 = sub_C4C880(v10, v11);
    v13 = a1->m128i_i64[1];
    v33 = a1[1].m128i_i64[0];
    v14 = v35[-2].m128i_i64[1] + 24;
    v15 = a1[1].m128i_i64[1];
    v37 = a1->m128i_i64[0];
    if ( v12 < 0 )
    {
      if ( (int)sub_C4C880(v11, v14) < 0 )
        goto LABEL_6;
      if ( (int)sub_C4C880(v10, v14) < 0 )
      {
LABEL_27:
        *a1 = _mm_loadu_si128(v35 - 2);
        a1[1] = _mm_loadu_si128(v35 - 1);
        v35[-2].m128i_i64[1] = v13;
        v16 = v13;
        v35[-1].m128i_i64[1] = v15;
        v35[-2].m128i_i64[0] = v37;
        v35[-1].m128i_i64[0] = v33;
        v13 = a1[2].m128i_i64[1];
        goto LABEL_7;
      }
LABEL_20:
      v23 = _mm_loadu_si128(a1 + 2);
      a1[2].m128i_i64[0] = v37;
      a1[2].m128i_i64[1] = v13;
      *a1 = v23;
      v24 = _mm_loadu_si128(a1 + 3);
      a1[3].m128i_i64[0] = v33;
      a1[3].m128i_i64[1] = v15;
      a1[1] = v24;
      v16 = v35[-2].m128i_i64[1];
      goto LABEL_7;
    }
    if ( (int)sub_C4C880(v10, v14) < 0 )
      goto LABEL_20;
    if ( (int)sub_C4C880(v11, v14) < 0 )
      goto LABEL_27;
LABEL_6:
    *a1 = _mm_loadu_si128(v9);
    a1[1] = _mm_loadu_si128(v9 + 1);
    v9->m128i_i64[0] = v37;
    v9->m128i_i64[1] = v13;
    v9[1].m128i_i64[0] = v33;
    v9[1].m128i_i64[1] = v15;
    v13 = a1[2].m128i_i64[1];
    v16 = v35[-2].m128i_i64[1];
LABEL_7:
    v17 = a1 + 2;
    v18 = v35;
    v19 = a1->m128i_i64[1] + 24;
    while ( 1 )
    {
      v38 = (__int64 *)v17;
      if ( (int)sub_C4C880(v13 + 24, v19) < 0 )
        goto LABEL_14;
      v18 -= 2;
      while ( (int)sub_C4C880(v19, v16 + 24) < 0 )
      {
        v16 = v18[-2].m128i_i64[1];
        v18 -= 2;
      }
      if ( v17 >= v18 )
        break;
      v20 = v17[1].m128i_i64[1];
      v21 = v17->m128i_i64[0];
      v22 = v17[1].m128i_i64[0];
      *v17 = _mm_loadu_si128(v18);
      v17[1] = _mm_loadu_si128(v18 + 1);
      v16 = v18[-2].m128i_i64[1];
      v18[1].m128i_i64[1] = v20;
      v18->m128i_i64[0] = v21;
      v18->m128i_i64[1] = v13;
      v18[1].m128i_i64[0] = v22;
      v19 = a1->m128i_i64[1] + 24;
LABEL_14:
      v13 = v17[2].m128i_i64[1];
      v17 += 2;
    }
    sub_27B9290(v17, v35, v34);
    v8 = (char *)v17 - (char *)a1;
    if ( (char *)v17 - (char *)a1 > 512 )
    {
      if ( v34 )
      {
        v35 = v17;
        continue;
      }
LABEL_22:
      v25 = v8 >> 5;
      v26 = ((v8 >> 5) - 2) >> 1;
      sub_27B8ED0(
        (__int64)a1,
        (v25 - 2) >> 1,
        v25,
        (__int64)a1,
        a5,
        a6,
        a1[2 * v26].m128i_i64[0],
        a1[2 * v26].m128i_i64[1],
        a1[2 * v26 + 1].m128i_i64[0],
        a1[2 * v26 + 1].m128i_i64[1]);
      do
      {
        --v26;
        sub_27B8ED0(
          (__int64)a1,
          v26,
          v25,
          32 * v26,
          v27,
          v28,
          a1[2 * v26].m128i_i64[0],
          a1[2 * v26].m128i_i64[1],
          a1[2 * v26 + 1].m128i_i64[0],
          a1[2 * v26 + 1].m128i_i64[1]);
      }
      while ( v26 );
      do
      {
        v38 -= 4;
        v29 = v38[1];
        v30 = *v38;
        *(__m128i *)v38 = _mm_loadu_si128(a1);
        v31 = v38[2];
        v32 = v38[3];
        *((__m128i *)v38 + 1) = _mm_loadu_si128(a1 + 1);
        sub_27B8ED0(
          (__int64)a1,
          0,
          ((char *)v38 - (char *)a1) >> 5,
          (char *)v38 - (char *)a1,
          ((char *)v38 - (char *)a1) >> 5,
          (__int64)a1,
          v30,
          v29,
          v31,
          v32);
      }
      while ( (char *)v38 - (char *)a1 > 32 );
    }
    break;
  }
}
