// Function: sub_1078420
// Address: 0x1078420
//
void __fastcall sub_1078420(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __m128i *v6; // r15
  __m128i *v7; // r10
  __int64 v8; // rbx
  __int64 v9; // r13
  const __m128i *v10; // rax
  const __m128i *v11; // r10
  const __m128i *v12; // r9
  const __m128i *v13; // r11
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int32 v19; // eax
  const __m128i *v20; // [rsp+0h] [rbp-50h]
  const __m128i *v21; // [rsp+8h] [rbp-48h]
  __m128i *v22; // [rsp+10h] [rbp-40h]
  const __m128i *v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  while ( 1 )
  {
    v24 = a3;
    if ( !a4 )
      break;
    v5 = a5;
    if ( !a5 )
      break;
    v6 = a1;
    v7 = a2;
    v8 = a4;
    if ( a4 + a5 == 2 )
    {
      v15 = a1[2].m128i_i64[0];
      v16 = a1->m128i_i64[0];
      if ( *(_QWORD *)(a2[2].m128i_i64[0] + 160) + a2->m128i_i64[0] < (unsigned __int64)(a1->m128i_i64[0]
                                                                                       + *(_QWORD *)(v15 + 160)) )
      {
        v17 = a1->m128i_i64[1];
        v18 = v6[1].m128i_i64[0];
        v19 = v6[1].m128i_i32[2];
        *v6 = _mm_loadu_si128(v7);
        v6[1] = _mm_loadu_si128(v7 + 1);
        v6[2].m128i_i64[0] = v7[2].m128i_i64[0];
        v7->m128i_i64[0] = v16;
        v7->m128i_i64[1] = v17;
        v7[1].m128i_i64[0] = v18;
        v7[1].m128i_i32[2] = v19;
        v7[2].m128i_i64[0] = v15;
      }
      return;
    }
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v12 = (const __m128i *)sub_1077A50(a2, a3, &a1->m128i_i64[5 * (a4 / 2)]);
      v9 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v12 - (char *)v11) >> 3);
    }
    else
    {
      v9 = a5 / 2;
      v22 = (__m128i *)((char *)a2 + 40 * (a5 / 2));
      v10 = (const __m128i *)sub_1077AC0(a1, (__int64)a2, v22);
      v12 = v22;
      v13 = v10;
      v14 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v10 - (char *)a1) >> 3);
    }
    v20 = v12;
    v21 = v13;
    v23 = sub_1077D10(v13, v11, v12);
    sub_1078420(a1, v21, v23, v14, v9);
    a3 = v24;
    a4 = v8 - v14;
    a5 = v5 - v9;
    a2 = (__m128i *)v20;
    a1 = (__m128i *)v23;
  }
}
