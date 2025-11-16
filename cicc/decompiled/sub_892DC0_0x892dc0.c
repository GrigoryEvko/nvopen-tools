// Function: sub_892DC0
// Address: 0x892dc0
//
__m128i **__fastcall sub_892DC0(__int64 a1, __int64 **a2, __m128i **a3, __int64 a4, char a5)
{
  const __m128i *v5; // r15
  __int64 *v8; // rax
  _QWORD *m128i_i64; // r12
  __int32 *v10; // rax
  __m128i **result; // rax
  __int8 v12; // al
  const __m128i *v13; // r13
  __m128i *v14; // r13
  __m128i *v15; // r13
  _DWORD *v16; // rax
  const __m128i *v17; // r13
  __m128i *v18; // rax
  __m128i *v19; // r14
  const void *v20; // rsi
  void *v21; // rax
  __int64 v22; // r11
  __m128i *v23; // r13
  __int8 v24; // cl
  __int64 v25; // rcx
  const __m128i *v26; // [rsp+8h] [rbp-68h]
  const __m128i *v27; // [rsp+10h] [rbp-60h]
  char v28; // [rsp+1Fh] [rbp-51h]
  const __m128i *v30; // [rsp+28h] [rbp-48h]
  __m128i *v31; // [rsp+28h] [rbp-48h]
  __m128i *v32; // [rsp+28h] [rbp-48h]
  int v34; // [rsp+38h] [rbp-38h]
  __int32 v35; // [rsp+3Ch] [rbp-34h]

  v5 = (const __m128i *)a1;
  *a3 = 0;
  v8 = *a2;
  if ( *a2 )
  {
    do
    {
      m128i_i64 = v8;
      v8 = (__int64 *)*v8;
    }
    while ( v8 );
    v10 = (__int32 *)sub_892BC0((__int64)m128i_i64);
    v34 = v10[1];
    v35 = *v10;
  }
  else
  {
    m128i_i64 = 0;
    v35 = 0;
    v34 = *(_DWORD *)(sub_892BC0(a1) + 4) + 2;
  }
  result = (__m128i **)(a5 & 1);
  v28 = a5 & 1;
  if ( a1 )
  {
    do
    {
      while ( 1 )
      {
        v17 = (const __m128i *)v5->m128i_i64[1];
        result = (__m128i **)*(unsigned int *)sub_892BC0((__int64)v5);
        if ( !a4
          || (unsigned int)result <= *(_DWORD *)(a4 + 16)
          && (result = (__m128i **)*(unsigned int *)(*(_QWORD *)a4 + 4LL * (unsigned int)((_DWORD)result - 1)),
              (_DWORD)result) )
        {
          v18 = (__m128i *)sub_87EBB0(v17[5].m128i_u8[0], v17->m128i_i64[0], (const __m128i *)v17[3].m128i_i64);
          v19 = v18;
          if ( v17[5].m128i_i8[0] == 19 )
          {
            v20 = (const void *)v17[5].m128i_i64[1];
            v21 = (void *)v18[5].m128i_i64[1];
            *v19 = _mm_loadu_si128(v17);
            v19[1] = _mm_loadu_si128(v17 + 1);
            v19[2] = _mm_loadu_si128(v17 + 2);
            v19[3] = _mm_loadu_si128(v17 + 3);
            v19[4] = _mm_loadu_si128(v17 + 4);
            v19[5] = _mm_loadu_si128(v17 + 5);
            v19[6] = _mm_loadu_si128(v17 + 6);
            if ( v20 )
            {
              v19[5].m128i_i64[1] = (__int64)v21;
              qmemcpy(v21, v20, 0x1B0u);
            }
          }
          else
          {
            *v18 = _mm_loadu_si128(v17);
            v18[1] = _mm_loadu_si128(v17 + 1);
            v18[2] = _mm_loadu_si128(v17 + 2);
            v18[3] = _mm_loadu_si128(v17 + 3);
            v18[4] = _mm_loadu_si128(v17 + 4);
            v18[5] = _mm_loadu_si128(v17 + 5);
            v18[6] = _mm_loadu_si128(v17 + 6);
          }
          v12 = v19[5].m128i_i8[0];
          v19->m128i_i64[1] = 0;
          v19[1].m128i_i64[0] = 0;
          v19[1].m128i_i64[1] = 0;
          if ( v12 == 3 )
          {
            v22 = v5[4].m128i_i64[0];
            v27 = (const __m128i *)v22;
            v26 = *(const __m128i **)(v22 + 168);
            v23 = (__m128i *)sub_7259C0(*(_BYTE *)(v22 + 140));
            v32 = (__m128i *)v23[10].m128i_i64[1];
            sub_73C230(v27, v23);
            v24 = v23[10].m128i_i8[1];
            v23[10].m128i_i64[1] = (__int64)v32;
            v23[10].m128i_i8[1] = (16 * v28) | v24 & 0xEF;
            *v32 = _mm_loadu_si128(v26);
            v32[1] = _mm_loadu_si128(v26 + 1);
            v25 = v26[2].m128i_i64[0];
            v32->m128i_i64[0] = 0;
            v32[2].m128i_i64[0] = v25;
            sub_877D80((__int64)v23, v19->m128i_i64);
            v19[5].m128i_i64[1] = (__int64)v23;
          }
          else
          {
            v13 = (const __m128i *)v5[4].m128i_i64[0];
            if ( v12 == 2 )
            {
              v31 = (__m128i *)sub_724D50(v13[10].m128i_i8[13]);
              sub_72A510(v13, v31);
              sub_877D80((__int64)v31, v19->m128i_i64);
              v19[5].m128i_i64[1] = (__int64)v31;
            }
            else
            {
              v30 = (const __m128i *)v13[6].m128i_i64[1];
              v14 = (__m128i *)sub_727340();
              sub_72A5D0(v30, v14);
              v14[10].m128i_i64[1] = v19[5].m128i_i64[1];
              sub_877D80((__int64)v14, v19->m128i_i64);
              *(_QWORD *)(v19[5].m128i_i64[1] + 104) = v14;
            }
          }
          ++v35;
          v15 = sub_880B70(v5, (__int64)v19);
          v15[3].m128i_i32[3] = v35;
          v16 = (_DWORD *)sub_892BC0((__int64)v15);
          *v16 = v35;
          v16[1] = v34;
          if ( m128i_i64 )
            *m128i_i64 = v15;
          else
            *a2 = (__int64 *)v15;
          result = a3;
          m128i_i64 = v15->m128i_i64;
          if ( !*a3 )
            break;
        }
        v5 = (const __m128i *)v5->m128i_i64[0];
        if ( !v5 )
          return result;
      }
      *a3 = v15;
      v5 = (const __m128i *)v5->m128i_i64[0];
    }
    while ( v5 );
  }
  return result;
}
