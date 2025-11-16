// Function: sub_28EB0B0
// Address: 0x28eb0b0
//
void __fastcall sub_28EB0B0(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  const __m128i *v10; // r9
  const __m128i *v11; // r10
  __int64 v12; // r11
  __m128i *v13; // r15
  __int64 v14; // r14
  __m128i *v15; // rax
  __m128i v16; // xmm0
  const __m128i *v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  const __m128i *v20; // [rsp+18h] [rbp-48h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = (__int64)a1;
      v7 = (__int64)a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = a1;
        v13 = a2;
LABEL_12:
        if ( v13->m128i_i32[2] > (unsigned __int32)v15->m128i_i32[2] )
        {
          v16 = _mm_loadu_si128(v15);
          v15->m128i_i64[0] = v13->m128i_i64[0];
          v15->m128i_i32[2] = v13->m128i_i32[2];
          v13->m128i_i64[0] = v16.m128i_i64[0];
          v13->m128i_i32[2] = v16.m128i_i32[2];
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = (__m128i *)sub_28EA1F0(v7, a3, v6 + 16 * (v5 / 2));
        v14 = v13 - v10;
        while ( 1 )
        {
          v19 = v12;
          v20 = v11;
          v8 -= v14;
          v18 = sub_28E9610(v11, v10, v13);
          sub_28EB0B0(v19, v20, v18, v9, v14);
          v5 -= v9;
          if ( !v5 )
            break;
          v15 = (__m128i *)v18;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = (__int64)v18;
          v7 = (__int64)v13;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v8 / 2;
          v13 = (__m128i *)(v7 + 16 * (v8 / 2));
          v11 = (const __m128i *)sub_28EA240(v6, v7, (__int64)v13);
          v9 = ((__int64)v11->m128i_i64 - v12) >> 4;
        }
      }
    }
  }
}
