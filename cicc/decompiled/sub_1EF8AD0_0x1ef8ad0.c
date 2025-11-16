// Function: sub_1EF8AD0
// Address: 0x1ef8ad0
//
void __fastcall sub_1EF8AD0(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  const __m128i *v10; // r9
  const __m128i *v11; // r10
  const __m128i *v12; // r11
  __int64 v13; // r14
  __m128i *v14; // rax
  __m128i *v15; // r10
  __m128i v16; // xmm0
  __int32 v17; // ebx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned __int64 v21; // rdi
  __m128i v22; // xmm1
  __m128i *v24; // [rsp+8h] [rbp-78h]
  const __m128i *v25; // [rsp+10h] [rbp-70h]
  const __m128i *v26; // [rsp+18h] [rbp-68h]
  __m128i *v27; // [rsp+18h] [rbp-68h]
  __m128i v28[5]; // [rsp+30h] [rbp-50h] BYREF

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
        v14 = a1;
        v15 = a2;
LABEL_12:
        if ( v15->m128i_i32[2] > (unsigned __int32)v14->m128i_i32[2] )
        {
          v16 = _mm_loadu_si128(v14 + 1);
          v17 = v14[2].m128i_i32[0];
          v14[1].m128i_i64[0] = 0;
          v18 = v14->m128i_i64[0];
          v19 = v14->m128i_i64[1];
          v14[1].m128i_i64[1] = 0;
          v14[2].m128i_i32[0] = 0;
          v20 = v15->m128i_i64[0];
          v28[0] = v16;
          v14->m128i_i64[0] = v20;
          v14->m128i_i32[2] = v15->m128i_i32[2];
          v14->m128i_i32[3] = v15->m128i_i32[3];
          if ( &v15[1] == &v14[1] )
          {
            v21 = v15[1].m128i_u64[0];
          }
          else
          {
            v21 = 0;
            v14[1] = _mm_loadu_si128(v15 + 1);
            v14[2].m128i_i32[0] = v15[2].m128i_i32[0];
          }
          v15->m128i_i64[0] = v18;
          v15->m128i_i64[1] = v19;
          v27 = v15;
          _libc_free(v21);
          v22 = _mm_loadu_si128(v28);
          v27[2].m128i_i32[0] = v17;
          v27[1] = v22;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v11 = (const __m128i *)sub_1EF7FF0(v7, a3, v6 + 40 * (v5 / 2));
        v13 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v11 - (char *)v10) >> 3);
        while ( 1 )
        {
          v24 = (__m128i *)v11;
          v26 = v12;
          v8 -= v13;
          v25 = sub_1EF8700(v12, v10, v11);
          sub_1EF8AD0(v6, v26, v25, v9, v13);
          v5 -= v9;
          if ( !v5 )
            break;
          v14 = (__m128i *)v25;
          v15 = v24;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = (__int64)v25;
          v7 = (__int64)v24;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v13 = v8 / 2;
          v12 = (const __m128i *)sub_1EF8050(v6, v7, v7 + 40 * (v8 / 2));
          v9 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v12->m128i_i64 - v6) >> 3);
        }
      }
    }
  }
}
