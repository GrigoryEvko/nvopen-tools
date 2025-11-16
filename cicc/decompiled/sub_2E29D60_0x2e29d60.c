// Function: sub_2E29D60
// Address: 0x2e29d60
//
unsigned __int64 __fastcall sub_2E29D60(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  unsigned int v7; // ebx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v11; // rsi
  __m128i *v12; // r13
  _QWORD **v13; // r14
  __m128i *v14; // rbx
  unsigned __int64 v15; // rcx
  _QWORD *v16; // r15
  const __m128i *v17; // r14
  _QWORD *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  char *v22; // rdi
  size_t v23; // rdx
  char *v24; // rax
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r15
  unsigned __int64 v27; // rdi
  __int8 *v28; // rbx
  unsigned __int64 v29; // [rsp+8h] [rbp-48h]
  unsigned __int64 v30; // [rsp+10h] [rbp-40h]
  size_t v31; // [rsp+10h] [rbp-40h]
  unsigned __int64 v32; // [rsp+18h] [rbp-38h]
  _QWORD **v33; // [rsp+18h] [rbp-38h]

  v6 = a2 & 0x7FFFFFFF;
  v7 = (a2 & 0x7FFFFFFF) + 1;
  v8 = a1->m128i_u32[2];
  v9 = a1->m128i_i64[0];
  if ( v7 > (unsigned int)v8 )
  {
    v11 = v7;
    if ( v7 != v8 )
    {
      v12 = a1;
      v13 = (_QWORD **)(v9 + 56 * v8);
      if ( v7 < v8 )
      {
        v33 = (_QWORD **)(v9 + 56LL * v7);
        if ( v33 != v13 )
        {
          do
          {
            v25 = (unsigned __int64)*(v13 - 3);
            v13 -= 7;
            if ( v25 )
              j_j___libc_free_0(v25);
            v26 = *v13;
            while ( v13 != v26 )
            {
              v27 = (unsigned __int64)v26;
              v26 = (_QWORD *)*v26;
              j_j___libc_free_0(v27);
            }
          }
          while ( v33 != v13 );
          v9 = v12->m128i_i64[0];
        }
        v12->m128i_i32[2] = v7;
      }
      else
      {
        v14 = a1 + 1;
        v29 = v11 - v8;
        v15 = a1->m128i_u32[3];
        if ( v11 > v15 )
        {
          if ( v9 > (unsigned __int64)v14 || v13 <= (_QWORD **)v14 )
          {
            sub_239A190((__int64)a1, v11, v9, v15, a5, a6);
            v9 = a1->m128i_i64[0];
            v8 = a1->m128i_u32[2];
          }
          else
          {
            v28 = &v14->m128i_i8[-v9];
            sub_239A190((__int64)a1, v11, v9, v15, a5, a6);
            v9 = a1->m128i_i64[0];
            v8 = a1->m128i_u32[2];
            v14 = (__m128i *)&v28[a1->m128i_i64[0]];
          }
        }
        v16 = (_QWORD *)(v9 + 56 * v8);
        v32 = v29;
        do
        {
          if ( v16 )
          {
            v16[1] = v16;
            *v16 = v16;
            v16[2] = 0;
            v17 = (const __m128i *)v14->m128i_i64[0];
            if ( v14 == (__m128i *)v14->m128i_i64[0] )
            {
              v18 = v16;
            }
            else
            {
              do
              {
                v11 = (unsigned __int64)v16;
                a1 = (__m128i *)sub_22077B0(0x28u);
                a1[1] = _mm_loadu_si128(v17 + 1);
                a1[2].m128i_i64[0] = v17[2].m128i_i64[0];
                sub_2208C80(a1, (__int64)v16);
                ++v16[2];
                v17 = (const __m128i *)v17->m128i_i64[0];
              }
              while ( v14 != v17 );
              v18 = (_QWORD *)*v16;
            }
            v16[3] = v18;
            v19 = v14[2].m128i_i64[1] - v14[2].m128i_i64[0];
            v16[4] = 0;
            v16[5] = 0;
            v16[6] = 0;
            if ( v19 )
            {
              if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
                sub_4261EA(a1, v11, v19);
              v30 = v19;
              v20 = sub_22077B0(v19);
              v21 = v30;
              v22 = (char *)v20;
            }
            else
            {
              v21 = 0;
              v22 = 0;
            }
            v16[4] = v22;
            v16[6] = &v22[v21];
            v16[5] = v22;
            v11 = v14[2].m128i_u64[0];
            v23 = v14[2].m128i_i64[1] - v11;
            if ( v14[2].m128i_i64[1] != v11 )
            {
              v31 = v14[2].m128i_i64[1] - v11;
              v24 = (char *)memmove(v22, (const void *)v11, v23);
              v23 = v31;
              v22 = v24;
            }
            a1 = (__m128i *)&v22[v23];
            v16[5] = a1;
          }
          v16 += 7;
          --v32;
        }
        while ( v32 );
        v9 = v12->m128i_i64[0];
        v12->m128i_i32[2] += v29;
      }
    }
  }
  return v9 + 56LL * v6;
}
