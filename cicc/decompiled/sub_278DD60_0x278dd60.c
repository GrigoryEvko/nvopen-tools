// Function: sub_278DD60
// Address: 0x278dd60
//
__int64 __fastcall sub_278DD60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 result; // rax
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // r8
  const __m128i *v18; // rbx
  __m128i *v19; // rdx
  int v20; // ecx
  __int64 v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // rdx
  __m128i v24; // xmm0
  __m128i v25; // xmm1
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // rcx
  const __m128i *v29; // rbx
  __m128i *v30; // rdx
  __int64 v31; // rdx
  int v32; // r8d
  const void *v33; // rsi
  char *v34; // rbx
  const void *v35; // rsi
  char *v36; // rbx
  const void *v37; // [rsp+8h] [rbp-B8h]
  __int64 v39; // [rsp+20h] [rbp-A0h]
  __m128i v40; // [rsp+30h] [rbp-90h] BYREF
  __m128i v41; // [rsp+40h] [rbp-80h] BYREF
  char v42; // [rsp+50h] [rbp-70h]
  __int64 v43; // [rsp+60h] [rbp-60h] BYREF
  __m128i v44; // [rsp+68h] [rbp-58h]
  __m128i v45; // [rsp+78h] [rbp-48h]

  v6 = *(__int64 **)a3;
  v37 = (const void *)(a5 + 16);
  result = (__int64)&v43;
  v39 = *(_QWORD *)a3 + 24LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v39 )
  {
    while ( 1 )
    {
      v20 = *(_DWORD *)(a1 + 72);
      v21 = *v6;
      v22 = *(_QWORD *)(a1 + 56);
      if ( v20 )
      {
        v11 = v20 - 1;
        v12 = v11 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v13 = *(_QWORD *)(v22 + 8LL * v12);
        if ( v21 == v13 )
        {
LABEL_4:
          v14 = *(unsigned int *)(a4 + 8);
          v15 = *(unsigned int *)(a4 + 12);
          v43 = *v6;
          v44.m128i_i64[0] = 0;
          v16 = *(_QWORD *)a4;
          v17 = v14 + 1;
          v44.m128i_i64[1] = 3;
          v18 = (const __m128i *)&v43;
          v45 = 0u;
          if ( v14 + 1 > v15 )
          {
            v35 = (const void *)(a4 + 16);
            if ( v16 > (unsigned __int64)&v43 || (unsigned __int64)&v43 >= v16 + 40 * v14 )
            {
              result = sub_C8D5F0(a4, v35, v17, 0x28u, v17, a6);
              v16 = *(_QWORD *)a4;
              v14 = *(unsigned int *)(a4 + 8);
              v18 = (const __m128i *)&v43;
            }
            else
            {
              v36 = (char *)&v43 - v16;
              result = sub_C8D5F0(a4, v35, v17, 0x28u, v17, a6);
              v16 = *(_QWORD *)a4;
              v14 = *(unsigned int *)(a4 + 8);
              v18 = (const __m128i *)&v36[*(_QWORD *)a4];
            }
          }
          v19 = (__m128i *)(v16 + 40 * v14);
          *v19 = _mm_loadu_si128(v18);
          v19[1] = _mm_loadu_si128(v18 + 1);
          v19[2].m128i_i64[0] = v18[2].m128i_i64[0];
          ++*(_DWORD *)(a4 + 8);
          goto LABEL_6;
        }
        v32 = 1;
        while ( v13 != -4096 )
        {
          a6 = (unsigned int)(v32 + 1);
          v12 = v11 & (v32 + v12);
          result = v12;
          v13 = *(_QWORD *)(v22 + 8LL * v12);
          if ( v21 == v13 )
            goto LABEL_4;
          ++v32;
        }
      }
      if ( (unsigned int)(v6[1] & 7) - 1 <= 1 && (result = sub_278D170((__int64)&v40, a1, a2, v6[1], v6[2]), v42) )
      {
        v23 = *(unsigned int *)(a4 + 8);
        v24 = _mm_loadu_si128(&v40);
        v43 = v21;
        v25 = _mm_loadu_si128(&v41);
        v26 = *(unsigned int *)(a4 + 12);
        v27 = v23 + 1;
        v28 = *(_QWORD *)a4;
        v29 = (const __m128i *)&v43;
        v44 = v24;
        v45 = v25;
        if ( v23 + 1 > v26 )
        {
          v33 = (const void *)(a4 + 16);
          if ( v28 > (unsigned __int64)&v43 || (unsigned __int64)&v43 >= v28 + 40 * v23 )
          {
            result = sub_C8D5F0(a4, v33, v27, 0x28u, v27, a6);
            v28 = *(_QWORD *)a4;
            v23 = *(unsigned int *)(a4 + 8);
            v29 = (const __m128i *)&v43;
          }
          else
          {
            v34 = (char *)&v43 - v28;
            result = sub_C8D5F0(a4, v33, v27, 0x28u, v27, a6);
            v28 = *(_QWORD *)a4;
            v23 = *(unsigned int *)(a4 + 8);
            v29 = (const __m128i *)&v34[*(_QWORD *)a4];
          }
        }
        v6 += 3;
        v30 = (__m128i *)(v28 + 40 * v23);
        *v30 = _mm_loadu_si128(v29);
        v30[1] = _mm_loadu_si128(v29 + 1);
        v30[2].m128i_i64[0] = v29[2].m128i_i64[0];
        ++*(_DWORD *)(a4 + 8);
        if ( (__int64 *)v39 == v6 )
          return result;
      }
      else
      {
        v31 = *(unsigned int *)(a5 + 8);
        if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          result = sub_C8D5F0(a5, v37, v31 + 1, 8u, v31 + 1, a6);
          v31 = *(unsigned int *)(a5 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v31) = v21;
        ++*(_DWORD *)(a5 + 8);
LABEL_6:
        v6 += 3;
        if ( (__int64 *)v39 == v6 )
          return result;
      }
    }
  }
  return result;
}
