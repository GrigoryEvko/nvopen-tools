// Function: sub_2E0E0B0
// Address: 0x2e0e0b0
//
unsigned __int64 __fastcall sub_2E0E0B0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 v7; // r13
  __int64 v9; // r9
  __m128i *v10; // r15
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // r8d
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // rcx
  __m128i *v25; // rdx
  __m128i *v26; // rax
  __int64 v27; // rax
  __int64 v28; // r14
  const void *v29; // rsi
  int v30; // [rsp+8h] [rbp-58h]
  unsigned int v31; // [rsp+8h] [rbp-58h]
  unsigned __int64 v32; // [rsp+8h] [rbp-58h]
  __m128i v33; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v34; // [rsp+20h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 96) )
  {
    v33.m128i_i64[0] = a1;
    return sub_2E0DC20(v33.m128i_i64, a2, a3, 0, a5, a6);
  }
  v10 = (__m128i *)sub_2E09D00((__int64 *)a1, a2);
  v11 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8)) )
  {
    v17 = *a3;
    v18 = *(unsigned int *)(a1 + 72);
    a3[10] += 16;
    v19 = (v17 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( a3[1] >= v19 + 16 && v17 )
    {
      *a3 = v19 + 16;
      v7 = (v17 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( !v19 )
      {
LABEL_17:
        v20 = *(unsigned int *)(a1 + 72);
        if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
        {
          sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v20 + 1, 8u, v18, v9);
          v20 = *(unsigned int *)(a1 + 72);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v20) = v19;
        v21 = *(unsigned int *)(a1 + 8);
        v22 = *(unsigned int *)(a1 + 12);
        ++*(_DWORD *)(a1 + 72);
        v23 = v21 + 1;
        v33.m128i_i64[0] = a2;
        v33.m128i_i64[1] = v11 | 6;
        v34 = v7;
        if ( v21 + 1 > v22 )
        {
          v28 = *(_QWORD *)a1;
          v29 = (const void *)(a1 + 16);
          if ( *(_QWORD *)a1 <= (unsigned __int64)&v33 && (unsigned __int64)&v33 < v28 + 24 * v21 )
          {
            sub_C8D5F0(a1, v29, v23, 0x18u, v18, v9);
            v24 = *(_QWORD *)a1;
            v21 = *(unsigned int *)(a1 + 8);
            v25 = (__m128i *)((char *)&v33 + *(_QWORD *)a1 - v28);
          }
          else
          {
            sub_C8D5F0(a1, v29, v23, 0x18u, v18, v9);
            v24 = *(_QWORD *)a1;
            v21 = *(unsigned int *)(a1 + 8);
            v25 = &v33;
          }
        }
        else
        {
          v24 = *(_QWORD *)a1;
          v25 = &v33;
        }
        v26 = (__m128i *)(v24 + 24 * v21);
        *v26 = _mm_loadu_si128(v25);
        v26[1].m128i_i64[0] = v25[1].m128i_i64[0];
        ++*(_DWORD *)(a1 + 8);
        return v7;
      }
    }
    else
    {
      v31 = v18;
      v27 = sub_9D1E70((__int64)a3, 16, 16, 4);
      v18 = v31;
      v7 = v27;
      v19 = v27;
    }
    *(_DWORD *)v7 = v18;
    *(_QWORD *)(v7 + 8) = a2;
    goto LABEL_17;
  }
  v12 = v10->m128i_i64[0];
  if ( v11 != (v10->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v13 = *a3;
    v14 = *(_DWORD *)(a1 + 72);
    a3[10] += 16;
    v15 = (v13 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( a3[1] >= v15 + 16 && v13 )
    {
      *a3 = v15 + 16;
      v7 = (v13 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( !v15 )
      {
LABEL_10:
        v16 = *(unsigned int *)(a1 + 72);
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
        {
          v32 = v15;
          sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v16 + 1, 8u, v16 + 1, v9);
          v16 = *(unsigned int *)(a1 + 72);
          v15 = v32;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v16) = v15;
        ++*(_DWORD *)(a1 + 72);
        v33.m128i_i64[0] = a2;
        v33.m128i_i64[1] = v11 | 6;
        v34 = v7;
        sub_2E0C1A0(a1, v10, &v33);
        return v7;
      }
    }
    else
    {
      v30 = v14;
      v15 = sub_9D1E70((__int64)a3, 16, 16, 4);
      v14 = v30;
      v7 = v15;
    }
    *(_DWORD *)v7 = v14;
    *(_QWORD *)(v7 + 8) = a2;
    goto LABEL_10;
  }
  v7 = v10[1].m128i_u64[0];
  if ( (*(_DWORD *)(v11 + 24) | (unsigned int)(v12 >> 1) & 3) >= (*(_DWORD *)(v11 + 24) | (unsigned int)(a2 >> 1) & 3)
    && v12 != a2 )
  {
    *(_QWORD *)(v7 + 8) = a2;
    v7 = v10[1].m128i_u64[0];
    v10->m128i_i64[0] = *(_QWORD *)(v7 + 8);
  }
  return v7;
}
