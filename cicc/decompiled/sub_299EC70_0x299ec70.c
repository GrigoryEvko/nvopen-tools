// Function: sub_299EC70
// Address: 0x299ec70
//
void __fastcall sub_299EC70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 (__fastcall *a6)(__int64, __int64))
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  const __m128i *v12; // r8
  const __m128i *v13; // r11
  const __m128i *v14; // r10
  __int64 v15; // r14
  __m128i *v16; // r10
  __int64 v17; // rax
  const __m128i *v18; // rdx
  __int64 v19; // r11
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int32 v25; // eax
  const __m128i *v27; // [rsp+10h] [rbp-50h]
  const __m128i *src; // [rsp+18h] [rbp-48h]
  const __m128i *srcb; // [rsp+18h] [rbp-48h]
  __m128i *srca; // [rsp+18h] [rbp-48h]
  const __m128i *v31; // [rsp+20h] [rbp-40h]
  const __m128i *v32; // [rsp+20h] [rbp-40h]
  const __m128i *v33; // [rsp+20h] [rbp-40h]
  __m128i *v34; // [rsp+20h] [rbp-40h]

  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a4;
      if ( a5 + a4 == 2 )
      {
        v16 = (__m128i *)a2;
        v18 = (const __m128i *)a1;
LABEL_12:
        srca = (__m128i *)v18;
        v34 = v16;
        if ( a6((__int64)v16, (__int64)v18) )
        {
          v19 = srca->m128i_i64[0];
          v20 = srca->m128i_i64[1];
          v21 = srca[1].m128i_i64[0];
          *srca = _mm_loadu_si128(v34);
          v22 = srca[1].m128i_i64[1];
          v23 = srca[2].m128i_i64[0];
          v24 = srca[2].m128i_i64[1];
          srca[1] = _mm_loadu_si128(v34 + 1);
          v25 = srca[3].m128i_i32[0];
          srca[2] = _mm_loadu_si128(v34 + 2);
          srca[3].m128i_i64[0] = v34[3].m128i_i64[0];
          v34->m128i_i64[0] = v19;
          v34->m128i_i64[1] = v20;
          v34[1].m128i_i64[0] = v21;
          v34[1].m128i_i64[1] = v22;
          v34[2].m128i_i64[0] = v23;
          v34[2].m128i_i64[1] = v24;
          v34[3].m128i_i32[0] = v25;
        }
      }
      else
      {
        v8 = a2;
        v9 = a1;
        if ( a4 <= v6 )
          goto LABEL_10;
LABEL_5:
        v31 = (const __m128i *)v8;
        v10 = v7 / 2;
        v11 = sub_299EB50(v8, a3, v9 + 56 * (v7 / 2), a6);
        v12 = v31;
        v13 = (const __m128i *)(v9 + 56 * (v7 / 2));
        v14 = (const __m128i *)v11;
        v15 = 0x6DB6DB6DB6DB6DB7LL * ((v11 - (__int64)v31) >> 3);
        while ( 1 )
        {
          v27 = v14;
          v32 = v13;
          v6 -= v15;
          src = sub_299DCF0(v13, v12, v14);
          sub_299EC70(v9, v32, src, v10, v15, a6);
          v7 -= v10;
          if ( !v7 )
            break;
          v16 = (__m128i *)v27;
          if ( !v6 )
            break;
          if ( v6 + v7 == 2 )
          {
            v18 = src;
            goto LABEL_12;
          }
          v8 = (__int64)v27;
          v9 = (__int64)src;
          if ( v7 > v6 )
            goto LABEL_5;
LABEL_10:
          v33 = (const __m128i *)v8;
          v15 = v6 / 2;
          srcb = (const __m128i *)(v8 + 56 * (v6 / 2));
          v17 = sub_299EBE0(v9, v8, (__int64)srcb, a6);
          v14 = srcb;
          v12 = v33;
          v13 = (const __m128i *)v17;
          v10 = 0x6DB6DB6DB6DB6DB7LL * ((v17 - v9) >> 3);
        }
      }
    }
  }
}
