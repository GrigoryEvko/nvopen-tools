// Function: sub_3074D80
// Address: 0x3074d80
//
void __fastcall sub_3074D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r9
  __int64 v6; // r9
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  const __m128i *v9; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // r8
  __m128i *v12; // rax
  unsigned __int64 v13; // rsi
  const __m128i *v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // r8
  __m128i *v18; // rax
  __int64 v19; // rax
  bool v20; // cc
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // r8
  __int64 v23; // rcx
  const __m128i *v24; // rdx
  unsigned __int64 v25; // rcx
  __m128i *v26; // rax
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  const __m128i *v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // r8
  __m128i *v32; // rax
  const void *v33; // rsi
  char *v34; // r13
  const void *v35; // rsi
  char *v36; // r12
  const void *v37; // rsi
  char *v38; // r14
  const void *v39; // rsi
  char *v40; // r13
  const char *v41; // [rsp+0h] [rbp-60h] BYREF
  __int64 v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+10h] [rbp-50h]
  char *v44; // [rsp+20h] [rbp-40h] BYREF
  __int64 v45; // [rsp+28h] [rbp-38h]
  __int64 v46; // [rsp+30h] [rbp-30h] BYREF

  v41 = (const char *)sub_CE9030(a2);
  if ( BYTE4(v41) )
  {
    v7 = *(unsigned int *)(a3 + 12);
    v8 = *(_QWORD *)a3;
    v9 = (const __m128i *)&v44;
    v44 = "maxclusterrank";
    v45 = 14;
    v46 = (unsigned int)v41;
    v10 = *(unsigned int *)(a3 + 8);
    v11 = v10 + 1;
    if ( v10 + 1 > v7 )
    {
      v37 = (const void *)(a3 + 16);
      if ( v8 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v8 + 24 * v10 )
      {
        sub_C8D5F0(a3, v37, v11, 0x18u, v11, v5);
        v8 = *(_QWORD *)a3;
        v10 = *(unsigned int *)(a3 + 8);
      }
      else
      {
        v38 = (char *)&v44 - v8;
        sub_C8D5F0(a3, v37, v11, 0x18u, v11, v5);
        v8 = *(_QWORD *)a3;
        v10 = *(unsigned int *)(a3 + 8);
        v9 = (const __m128i *)&v38[*(_QWORD *)a3];
      }
    }
    v12 = (__m128i *)(v8 + 24 * v10);
    *v12 = _mm_loadu_si128(v9);
    v12[1].m128i_i64[0] = v9[1].m128i_i64[0];
    ++*(_DWORD *)(a3 + 8);
  }
  sub_CE8D40((__int64)&v44, a2);
  if ( (_DWORD)v45 )
  {
    v13 = *(unsigned int *)(a3 + 12);
    v42 = 8;
    v41 = "maxntidx";
    v14 = (const __m128i *)&v41;
    v15 = *(_QWORD *)a3;
    v43 = *(unsigned int *)v44;
    v16 = *(unsigned int *)(a3 + 8);
    v17 = v16 + 1;
    if ( v16 + 1 > v13 )
    {
      v39 = (const void *)(a3 + 16);
      if ( v15 > (unsigned __int64)&v41 || (unsigned __int64)&v41 >= v15 + 24 * v16 )
      {
        sub_C8D5F0(a3, v39, v17, 0x18u, v17, v6);
        v15 = *(_QWORD *)a3;
        v16 = *(unsigned int *)(a3 + 8);
        v14 = (const __m128i *)&v41;
      }
      else
      {
        v40 = (char *)&v41 - v15;
        sub_C8D5F0(a3, v39, v17, 0x18u, v17, v6);
        v15 = *(_QWORD *)a3;
        v16 = *(unsigned int *)(a3 + 8);
        v14 = (const __m128i *)&v40[*(_QWORD *)a3];
      }
    }
    v18 = (__m128i *)(v15 + 24 * v16);
    *v18 = _mm_loadu_si128(v14);
    v18[1].m128i_i64[0] = v14[1].m128i_i64[0];
    v19 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    v20 = (unsigned int)v45 <= 1;
    *(_DWORD *)(a3 + 8) = v19;
    if ( !v20 )
    {
      v21 = *(unsigned int *)(a3 + 12);
      v22 = v19 + 1;
      v41 = "maxntidy";
      v23 = *((unsigned int *)v44 + 1);
      v42 = 8;
      v24 = (const __m128i *)&v41;
      v43 = v23;
      v25 = *(_QWORD *)a3;
      if ( v19 + 1 > v21 )
      {
        v33 = (const void *)(a3 + 16);
        if ( v25 > (unsigned __int64)&v41 || (unsigned __int64)&v41 >= v25 + 24 * v19 )
        {
          sub_C8D5F0(a3, v33, v22, 0x18u, v22, v6);
          v25 = *(_QWORD *)a3;
          v19 = *(unsigned int *)(a3 + 8);
          v24 = (const __m128i *)&v41;
        }
        else
        {
          v34 = (char *)&v41 - v25;
          sub_C8D5F0(a3, v33, v22, 0x18u, v22, v6);
          v25 = *(_QWORD *)a3;
          v19 = *(unsigned int *)(a3 + 8);
          v24 = (const __m128i *)&v34[*(_QWORD *)a3];
        }
      }
      v26 = (__m128i *)(v25 + 24 * v19);
      *v26 = _mm_loadu_si128(v24);
      v26[1].m128i_i64[0] = v24[1].m128i_i64[0];
      ++*(_DWORD *)(a3 + 8);
      if ( (unsigned int)v45 > 2 )
      {
        v27 = *(unsigned int *)(a3 + 12);
        v28 = *(_QWORD *)a3;
        v29 = (const __m128i *)&v41;
        v41 = "maxntidz";
        v42 = 8;
        v43 = *((unsigned int *)v44 + 2);
        v30 = *(unsigned int *)(a3 + 8);
        v31 = v30 + 1;
        if ( v30 + 1 > v27 )
        {
          v35 = (const void *)(a3 + 16);
          if ( v28 > (unsigned __int64)&v41 || (unsigned __int64)&v41 >= v28 + 24 * v30 )
          {
            sub_C8D5F0(a3, v35, v31, 0x18u, v31, v6);
            v28 = *(_QWORD *)a3;
            v30 = *(unsigned int *)(a3 + 8);
            v29 = (const __m128i *)&v41;
          }
          else
          {
            v36 = (char *)&v41 - v28;
            sub_C8D5F0(a3, v35, v31, 0x18u, v31, v6);
            v28 = *(_QWORD *)a3;
            v30 = *(unsigned int *)(a3 + 8);
            v29 = (const __m128i *)&v36[*(_QWORD *)a3];
          }
        }
        v32 = (__m128i *)(v28 + 24 * v30);
        *v32 = _mm_loadu_si128(v29);
        v32[1].m128i_i64[0] = v29[1].m128i_i64[0];
        ++*(_DWORD *)(a3 + 8);
      }
    }
  }
  if ( v44 != (char *)&v46 )
    _libc_free((unsigned __int64)v44);
}
