// Function: sub_AE2A10
// Address: 0xae2a10
//
__int64 __fastcall sub_AE2A10(
        __int64 a1,
        unsigned int a2,
        __int32 a3,
        __int8 a4,
        __int8 a5,
        __int32 a6,
        unsigned __int8 a7)
{
  __int64 v10; // r10
  __m128i *v11; // r13
  unsigned int v12; // r9d
  __int64 v13; // rdi
  __m128i *v14; // rbx
  __m128i *v15; // r15
  __int64 i; // rax
  __int32 *v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  __m128i *v20; // r14
  const __m128i *v21; // rcx
  const __m128i *v22; // rdx
  __int64 v23; // r9
  __int64 result; // rax
  __int64 v25; // rdx
  const __m128i *v26; // r14
  unsigned __int64 v27; // rax
  __int64 v28; // rdi
  char *v29; // rbx
  __int64 v30; // rsi
  char *v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rsi
  __int64 v34; // rax
  const __m128i *v36; // [rsp+8h] [rbp-58h]
  unsigned int v37; // [rsp+10h] [rbp-50h] BYREF
  __int32 v38; // [rsp+14h] [rbp-4Ch]
  __int8 v39; // [rsp+18h] [rbp-48h]
  __int8 v40; // [rsp+19h] [rbp-47h]
  __int32 v41; // [rsp+1Ch] [rbp-44h]
  unsigned __int8 v42; // [rsp+20h] [rbp-40h]

  v10 = *(unsigned int *)(a1 + 280);
  v11 = *(__m128i **)(a1 + 272);
  v12 = *(_DWORD *)(a1 + 280);
  v13 = 20 * v10;
  v14 = v11;
  v15 = (__m128i *)((char *)v11 + 20 * v10);
  for ( i = 0xCCCCCCCCCCCCCCCDLL * ((20 * v10) >> 2); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v17 = &v14->m128i_i32[5 * (i >> 1)];
      if ( *v17 >= a2 )
        break;
      v14 = (__m128i *)(v17 + 5);
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        goto LABEL_5;
    }
  }
LABEL_5:
  if ( v15 == v14 )
  {
    v25 = v10 + 1;
    v39 = a4;
    v26 = (const __m128i *)&v37;
    v37 = a2;
    v41 = a6;
    v38 = a3;
    v42 = a7;
    v27 = *(unsigned int *)(a1 + 284);
    v40 = a5;
    if ( v10 + 1 > v27 )
    {
      v32 = a1 + 272;
      v33 = a1 + 288;
      if ( v11 > (__m128i *)&v37 || v14 <= (__m128i *)&v37 )
      {
        sub_C8D5F0(v32, v33, v25, 20);
        v14 = (__m128i *)(*(_QWORD *)(a1 + 272) + 20LL * *(unsigned int *)(a1 + 280));
      }
      else
      {
        sub_C8D5F0(v32, v33, v25, 20);
        v34 = *(_QWORD *)(a1 + 272);
        v26 = (const __m128i *)(v34 + (char *)&v37 - (char *)v11);
        v14 = (__m128i *)(v34 + 20LL * *(unsigned int *)(a1 + 280));
      }
    }
    *v14 = _mm_loadu_si128(v26);
    result = v26[1].m128i_u32[0];
    v14[1].m128i_i32[0] = result;
    ++*(_DWORD *)(a1 + 280);
  }
  else if ( v14->m128i_i32[0] == a2 )
  {
    v14->m128i_i32[1] = a3;
    v14->m128i_i8[8] = a4;
    v14->m128i_i32[3] = a6;
    v14->m128i_i8[9] = a5;
    v14[1].m128i_i8[0] = a7;
    return a7;
  }
  else
  {
    v37 = a2;
    v18 = v10 + 1;
    v19 = *(unsigned int *)(a1 + 284);
    v39 = a4;
    v41 = a6;
    v38 = a3;
    v42 = a7;
    v40 = a5;
    v20 = (__m128i *)&v37;
    v21 = (const __m128i *)&v37;
    if ( v10 + 1 > v19 )
    {
      v28 = a1 + 272;
      v29 = (char *)((char *)v14 - (char *)v11);
      v30 = a1 + 288;
      if ( v11 > (__m128i *)&v37 || v15 <= (__m128i *)&v37 )
      {
        sub_C8D5F0(v28, v30, v18, 20);
        v11 = *(__m128i **)(a1 + 272);
        v12 = *(_DWORD *)(a1 + 280);
        v21 = (const __m128i *)&v37;
        v13 = 20LL * v12;
        v14 = (__m128i *)&v29[(_QWORD)v11];
        v15 = (__m128i *)((char *)v11 + v13);
      }
      else
      {
        v31 = (char *)((char *)&v37 - (char *)v11);
        sub_C8D5F0(v28, v30, v18, 20);
        v11 = *(__m128i **)(a1 + 272);
        v21 = (const __m128i *)&v31[(_QWORD)v11];
        v14 = (__m128i *)&v29[(_QWORD)v11];
        v12 = *(_DWORD *)(a1 + 280);
        v13 = 20LL * v12;
        v20 = (__m128i *)&v31[(_QWORD)v11];
        v15 = (__m128i *)((char *)v11 + v13);
      }
    }
    v22 = (__m128i *)((char *)v11 + v13 - 20);
    if ( v15 )
    {
      *v15 = _mm_loadu_si128(v22);
      v15[1].m128i_i32[0] = v22[1].m128i_i32[0];
      v11 = *(__m128i **)(a1 + 272);
      v12 = *(_DWORD *)(a1 + 280);
      v13 = 20LL * v12;
      v22 = (__m128i *)((char *)v11 + v13 - 20);
    }
    if ( v14 != v22 )
    {
      v36 = v21;
      memmove(&v11->m128i_i8[v13 - ((char *)v22 - (char *)v14)], v14, (char *)v22 - (char *)v14);
      v12 = *(_DWORD *)(a1 + 280);
      v21 = v36;
      v11 = *(__m128i **)(a1 + 272);
    }
    v23 = v12 + 1;
    *(_DWORD *)(a1 + 280) = v23;
    if ( v14 <= v20 && v20 < (__m128i *)((char *)v11 + 20 * v23) )
      v21 = (const __m128i *)((char *)v21 + 20);
    *v14 = _mm_loadu_si128(v21);
    result = v21[1].m128i_u8[0];
    v14[1].m128i_i8[0] = result;
  }
  return result;
}
