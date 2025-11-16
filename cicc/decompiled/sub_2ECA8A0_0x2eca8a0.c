// Function: sub_2ECA8A0
// Address: 0x2eca8a0
//
__int64 __fastcall sub_2ECA8A0(__int64 a1, const __m128i *a2)
{
  __int64 v4; // r9
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rbx
  _DWORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  int v15; // edi
  __int64 v16; // rsi
  __m128i v17; // xmm2
  __int64 v18; // rcx
  int v19; // r9d
  __int64 v20; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdi
  const __m128i *v24; // r14
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __m128i *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  __int8 *v30; // r14
  __m128i v31; // [rsp+20h] [rbp-50h] BYREF
  __m128i v32; // [rsp+30h] [rbp-40h]
  __int64 v33; // [rsp+40h] [rbp-30h]

  v4 = a2->m128i_u32[0];
  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(_QWORD *)a1;
  v7 = v4 & 0x7FFFFFFF;
  v8 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 336) + v7);
  if ( (unsigned int)v8 < (unsigned int)v5 )
  {
    while ( 1 )
    {
      v9 = (_DWORD *)(v6 + 40LL * (unsigned int)v8);
      if ( (v4 & 0x7FFFFFFF) == (*v9 & 0x7FFFFFFF) )
      {
        v10 = (unsigned int)v9[8];
        if ( (_DWORD)v10 != -1 && *(_DWORD *)(v6 + 40 * v10 + 36) == -1 )
          break;
      }
      v8 = (unsigned int)(v8 + 256);
      if ( (unsigned int)v5 <= (unsigned int)v8 )
        goto LABEL_12;
    }
    v11 = a2[1].m128i_i64[1];
    if ( !*(_DWORD *)(a1 + 356) )
      goto LABEL_13;
LABEL_8:
    v12 = *(unsigned int *)(a1 + 352);
    v13 = _mm_loadu_si128(a2);
    v14 = _mm_loadu_si128(a2 + 1);
    v31 = v13;
    v15 = v12;
    v16 = 40 * v12;
    v32 = v14;
    v17 = _mm_loadu_si128((const __m128i *)&v31.m128i_u64[1]);
    v18 = 40 * v12 + v6;
    v19 = *(_DWORD *)(v18 + 36);
    *(_QWORD *)(v18 + 24) = v11;
    *(_QWORD *)(v18 + 32) = -1;
    *(_DWORD *)v18 = _mm_cvtsi128_si32(v13);
    *(__m128i *)(v18 + 8) = v17;
    --*(_DWORD *)(a1 + 356);
    *(_DWORD *)(a1 + 352) = v19;
    goto LABEL_9;
  }
LABEL_12:
  v11 = a2[1].m128i_i64[1];
  v8 = 0xFFFFFFFFLL;
  if ( *(_DWORD *)(a1 + 356) )
    goto LABEL_8;
LABEL_13:
  v22 = a2[1].m128i_i64[0];
  v23 = a2->m128i_i64[1];
  v32.m128i_i64[1] = v11;
  v24 = &v31;
  v25 = *(unsigned int *)(a1 + 12);
  v31.m128i_i32[0] = v4;
  v32.m128i_i64[0] = v22;
  v26 = v5 + 1;
  v31.m128i_i64[1] = v23;
  v33 = -1;
  if ( v5 + 1 > v25 )
  {
    v29 = a1 + 16;
    if ( v6 > (unsigned __int64)&v31 || (unsigned __int64)&v31 >= v6 + 40 * v5 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v26, 0x28u, v29, v4);
      v6 = *(_QWORD *)a1;
      v5 = *(unsigned int *)(a1 + 8);
    }
    else
    {
      v30 = &v31.m128i_i8[-v6];
      sub_C8D5F0(a1, (const void *)(a1 + 16), v26, 0x28u, v29, v4);
      v6 = *(_QWORD *)a1;
      v5 = *(unsigned int *)(a1 + 8);
      v24 = (const __m128i *)&v30[*(_QWORD *)a1];
    }
  }
  v27 = (__m128i *)(v6 + 40 * v5);
  *v27 = _mm_loadu_si128(v24);
  v27[1] = _mm_loadu_si128(v24 + 1);
  v27[2].m128i_i64[0] = v24[2].m128i_i64[0];
  v28 = *(unsigned int *)(a1 + 8);
  v15 = *(_DWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 8) = v28 + 1;
  v16 = 40 * v28;
LABEL_9:
  if ( (_DWORD)v8 == -1 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 336) + v7) = v15;
    *(_DWORD *)(*(_QWORD *)a1 + v16 + 32) = v15;
  }
  else
  {
    v20 = *(unsigned int *)(*(_QWORD *)a1 + 40 * v8 + 32);
    *(_DWORD *)(*(_QWORD *)a1 + 40 * v20 + 36) = v15;
    *(_DWORD *)(*(_QWORD *)a1 + 40 * v8 + 32) = v15;
    *(_DWORD *)(*(_QWORD *)a1 + v16 + 32) = v20;
  }
  return a1;
}
