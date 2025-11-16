// Function: sub_39867E0
// Address: 0x39867e0
//
void __fastcall sub_39867E0(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 *v7; // r11
  __int64 *v8; // r10
  __int64 v9; // rbx
  __int64 v10; // rcx
  __m128i *v11; // r14
  __int64 *v12; // rax
  const __m128i *v13; // r10
  __int64 *v14; // r11
  __m128i *v15; // r15
  __int64 v16; // r13
  __int64 *v17; // rdx
  __m128i *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 *v22; // [rsp+10h] [rbp-50h]
  const __m128i *v23; // [rsp+10h] [rbp-50h]
  const __m128i *v24; // [rsp+18h] [rbp-48h]
  __int64 *v25; // [rsp+18h] [rbp-48h]
  const __m128i *v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 *v28; // [rsp+20h] [rbp-40h]
  __int64 *v29; // [rsp+20h] [rbp-40h]
  __int64 v30[7]; // [rsp+28h] [rbp-38h] BYREF

  v30[0] = a6;
  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a1;
      v8 = (__int64 *)a2;
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v15 = a2;
        v17 = a1;
LABEL_12:
        v29 = v17;
        if ( (unsigned __int8)sub_3985080((__int64)v30, v15->m128i_i64[0], v17) )
        {
          v19 = *v29;
          v20 = v29[1];
          *(__m128i *)v29 = _mm_loadu_si128(v15);
          v15->m128i_i64[0] = v19;
          v15->m128i_i64[1] = v20;
        }
      }
      else
      {
        v10 = v30[0];
        if ( v9 <= a5 )
          goto LABEL_10;
LABEL_5:
        v22 = v7;
        v24 = (const __m128i *)v8;
        v27 = v9 / 2;
        v11 = (__m128i *)&v7[2 * (v9 / 2)];
        v12 = sub_3985650(v8, a3, v11->m128i_i64, v10);
        v13 = v24;
        v14 = v22;
        v15 = (__m128i *)v12;
        v16 = ((char *)v12 - (char *)v24) >> 4;
        while ( 1 )
        {
          v25 = v14;
          v23 = sub_3984680(v11, v13, v15);
          v6 -= v16;
          sub_39867E0(v25, v11, v23, v27, v16, v30[0]);
          v9 -= v27;
          if ( !v9 )
            break;
          v17 = (__int64 *)v23;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v10 = v30[0];
          v8 = (__int64 *)v15;
          v7 = (__int64 *)v23;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v26 = (const __m128i *)v8;
          v28 = v7;
          v16 = v6 / 2;
          v15 = (__m128i *)&v8[2 * (v6 / 2)];
          v18 = (__m128i *)sub_39865E0(v7, (__int64)v8, v15->m128i_i64, v10);
          v14 = v28;
          v13 = v26;
          v11 = v18;
          v27 = ((char *)v18 - (char *)v28) >> 4;
        }
      }
    }
  }
}
