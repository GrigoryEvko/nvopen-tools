// Function: sub_19E3890
// Address: 0x19e3890
//
void __fastcall sub_19E3890(__m128i *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 m128i_i64; // r13
  bool v8; // al
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 v11; // rsi
  __int32 v12; // r15d
  __int32 v13; // r14d
  __int32 v14; // r8d
  __m128i *v15; // r10
  __m128i *v16; // r11
  __int32 v17; // r8d
  __int32 v18; // edx
  __int32 v19; // esi
  __m128i *v20; // rdi
  __int64 *v21; // r15
  __m128i *v22; // rax
  __int64 *v23; // r14
  __int32 v24; // ecx
  __int32 v25; // ecx
  __int64 v26; // rcx
  __int32 v27; // ecx
  __int32 v28; // ecx
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int32 v32; // r10d
  __int32 v33; // r9d
  __int64 v34; // r10
  __m128i v35; // xmm6
  __m128i v36; // xmm7
  __int64 v37; // r12
  __int64 v38; // rsi
  __m128i *v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  __m128i *v46; // r10
  __m128i *v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+18h] [rbp-58h]

  v5 = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 > 512 )
  {
    if ( a3 )
    {
      m128i_i64 = (__int64)a1[2].m128i_i64;
      v47 = a1 + 4;
      while ( 1 )
      {
        --a3;
        v8 = sub_19E1390(m128i_i64, (__int64)a1[2 * (v5 >> 6)].m128i_i64);
        v11 = v9;
        v12 = a1->m128i_i32[1];
        v48 = a1[1].m128i_i64[0];
        v13 = a1->m128i_i32[2];
        v49 = a1[1].m128i_i64[1];
        if ( !v8 )
          break;
        if ( !sub_19E1390(v10, v9) )
        {
          if ( sub_19E1390(m128i_i64, v11) )
            goto LABEL_41;
LABEL_34:
          v35 = _mm_loadu_si128(a1 + 2);
          a1[2].m128i_i32[0] = v17;
          v36 = _mm_loadu_si128(a1 + 3);
          a1[2].m128i_i32[1] = v12;
          a1[3].m128i_i64[0] = v48;
          a1[2].m128i_i32[2] = v13;
          a1[3].m128i_i64[1] = v49;
          *a1 = v35;
          a1[1] = v36;
          v18 = v16[-2].m128i_i32[0];
          goto LABEL_7;
        }
        *a1 = _mm_loadu_si128(v15);
        a1[1] = _mm_loadu_si128(v15 + 1);
        v15->m128i_i32[0] = v14;
        v15->m128i_i32[1] = v12;
        v15->m128i_i32[2] = v13;
        v15[1].m128i_i64[0] = v48;
        v15[1].m128i_i64[1] = v49;
        v17 = a1[2].m128i_i32[0];
        v18 = v16[-2].m128i_i32[0];
LABEL_7:
        v19 = a1->m128i_i32[0];
        v20 = v47;
        v21 = (__int64 *)m128i_i64;
        v22 = v16;
        while ( 1 )
        {
          v23 = v21;
          if ( v19 <= v17 )
          {
            if ( v19 != v17 )
              break;
            v24 = a1->m128i_i32[1];
            if ( v20[-2].m128i_i32[1] >= v24 )
            {
              if ( v20[-2].m128i_i32[1] != v24 )
                break;
              v25 = a1->m128i_i32[2];
              if ( v20[-2].m128i_i32[2] >= v25 )
              {
                if ( v20[-2].m128i_i32[2] != v25 )
                  break;
                v26 = a1[1].m128i_i64[0];
                if ( v20[-1].m128i_i64[0] >= v26
                  && (v20[-1].m128i_i64[0] != v26 || v20[-1].m128i_i64[1] >= (unsigned __int64)a1[1].m128i_i64[1]) )
                {
                  break;
                }
              }
            }
          }
LABEL_28:
          v17 = v20->m128i_i32[0];
          v21 += 4;
          v20 += 2;
        }
        for ( v22 -= 2; ; v18 = v22->m128i_i32[0] )
        {
          if ( v19 >= v18 )
          {
            if ( v19 != v18 )
              break;
            v27 = v22->m128i_i32[1];
            if ( a1->m128i_i32[1] >= v27 )
            {
              if ( a1->m128i_i32[1] != v27 )
                break;
              v28 = v22->m128i_i32[2];
              if ( a1->m128i_i32[2] >= v28 )
              {
                if ( a1->m128i_i32[2] != v28 )
                  break;
                v29 = v22[1].m128i_i64[0];
                if ( a1[1].m128i_i64[0] >= v29
                  && (a1[1].m128i_i64[0] != v29 || a1[1].m128i_i64[1] >= (unsigned __int64)v22[1].m128i_i64[1]) )
                {
                  break;
                }
              }
            }
          }
          v22 -= 2;
        }
        if ( v21 < (__int64 *)v22 )
        {
          v30 = v20[-1].m128i_i64[0];
          v31 = v20[-1].m128i_i64[1];
          v32 = v20[-2].m128i_i32[1];
          v33 = v20[-2].m128i_i32[2];
          v20[-2] = _mm_loadu_si128(v22);
          v20[-1] = _mm_loadu_si128(v22 + 1);
          v22[1].m128i_i64[1] = v31;
          v18 = v22[-2].m128i_i32[0];
          v22->m128i_i32[0] = v17;
          v22->m128i_i32[1] = v32;
          v22->m128i_i32[2] = v33;
          v22[1].m128i_i64[0] = v30;
          v19 = a1->m128i_i32[0];
          goto LABEL_28;
        }
        sub_19E3890(v21, v16, a3);
        v5 = (char *)v21 - (char *)a1;
        if ( (char *)v21 - (char *)a1 <= 512 )
          return;
        if ( !a3 )
          goto LABEL_36;
      }
      if ( !sub_19E1390(m128i_i64, v9) )
      {
        if ( !sub_19E1390(v34, v11) )
        {
          *a1 = _mm_loadu_si128(v46);
          a1[1] = _mm_loadu_si128(v46 + 1);
          v46[1].m128i_i64[0] = v48;
          v46->m128i_i32[0] = v17;
          v46->m128i_i32[1] = v12;
          v46->m128i_i32[2] = v13;
          v46[1].m128i_i64[1] = v49;
          v17 = a1[2].m128i_i32[0];
          v18 = v16[-2].m128i_i32[0];
          goto LABEL_7;
        }
LABEL_41:
        v18 = v17;
        *a1 = _mm_loadu_si128(v16 - 2);
        a1[1] = _mm_loadu_si128(v16 - 1);
        v16[-1].m128i_i64[0] = v48;
        v16[-2].m128i_i32[0] = v17;
        v16[-2].m128i_i32[1] = v12;
        v16[-2].m128i_i32[2] = v13;
        v16[-1].m128i_i64[1] = v49;
        v17 = a1[2].m128i_i32[0];
        goto LABEL_7;
      }
      goto LABEL_34;
    }
    v23 = a2;
LABEL_36:
    v37 = v5 >> 5;
    v38 = ((v5 >> 5) - 2) >> 1;
    v39 = &a1[2 * v38];
    sub_19E3650(
      (__int64)a1,
      v38,
      v5 >> 5,
      a4,
      v39->m128i_i64[0],
      v39->m128i_i64[1],
      v39->m128i_i64[0],
      v39->m128i_i64[1],
      v39[1].m128i_i64[0],
      v39[1].m128i_u64[1]);
    do
    {
      --v38;
      sub_19E3650(
        (__int64)a1,
        v38,
        v37,
        32 * v38,
        a1[2 * v38 + 1].m128i_i64[0],
        a1[2 * v38 + 1].m128i_i64[1],
        a1[2 * v38].m128i_i64[0],
        a1[2 * v38].m128i_i64[1],
        a1[2 * v38 + 1].m128i_i64[0],
        a1[2 * v38 + 1].m128i_u64[1]);
    }
    while ( v38 );
    do
    {
      v23 -= 4;
      v42 = v23[1];
      v43 = v23[2];
      v44 = v23[3];
      v45 = *v23;
      *(__m128i *)v23 = _mm_loadu_si128(a1);
      *((__m128i *)v23 + 1) = _mm_loadu_si128(a1 + 1);
      sub_19E3650(
        (__int64)a1,
        0,
        ((char *)v23 - (char *)a1) >> 5,
        v40,
        ((char *)v23 - (char *)a1) >> 5,
        v41,
        v45,
        v42,
        v43,
        v44);
    }
    while ( (char *)v23 - (char *)a1 > 32 );
  }
}
