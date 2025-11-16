// Function: sub_2F91D20
// Address: 0x2f91d20
//
__int64 __fastcall sub_2F91D20(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, unsigned __int64 a5)
{
  unsigned int v5; // r10d
  __int64 v6; // r15
  __int64 v7; // r14
  _DWORD *v10; // rsi
  unsigned int v11; // esi
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rax
  __int64 v15; // rbx
  _DWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edi
  __m128i v22; // xmm0
  __int64 v23; // r10
  __int64 v24; // r11
  unsigned int v25; // r10d
  __int64 result; // rax
  __int64 v27; // rsi
  __int64 v28; // rdx
  _DWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // r11
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rdx
  const __m128i *v37; // r10
  __m128i *v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rdi
  const void *v42; // rsi
  char *v44; // [rsp+10h] [rbp-70h]
  unsigned int v45; // [rsp+1Ch] [rbp-64h]
  unsigned __int64 v46; // [rsp+20h] [rbp-60h] BYREF
  __m128i v47; // [rsp+28h] [rbp-58h] BYREF
  __int64 v48; // [rsp+38h] [rbp-48h]
  unsigned int v49; // [rsp+40h] [rbp-40h]
  __int64 v50; // [rsp+48h] [rbp-38h]

  v5 = a3;
  v6 = -1;
  v7 = -1;
  v10 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 32LL) + 40LL * a3);
  v45 = v10[2];
  if ( *(_BYTE *)(a1 + 899) )
  {
    v39 = sub_2F91CB0(a1, v10);
    v5 = a3;
    v7 = v39;
    v6 = v40;
  }
  v11 = *(_DWORD *)(a1 + 1800);
  v12 = *(_QWORD *)(a1 + 1792);
  v13 = v45 & 0x7FFFFFFF;
  v14 = v13;
  v15 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 2192) + v13);
  if ( (unsigned int)v15 < v11 )
  {
    while ( 1 )
    {
      v16 = (_DWORD *)(v12 + 48LL * (unsigned int)v15);
      if ( (_DWORD)v13 == (*v16 & 0x7FFFFFFF) )
      {
        v17 = (unsigned int)v16[10];
        if ( (_DWORD)v17 != -1 && *(_DWORD *)(v12 + 48 * v17 + 44) == -1 )
          break;
      }
      v15 = (unsigned int)(v15 + 256);
      if ( v11 <= (unsigned int)v15 )
        goto LABEL_27;
    }
    if ( !*(_DWORD *)(a1 + 2212) )
      goto LABEL_28;
LABEL_10:
    a5 = v45;
    v18 = *(unsigned int *)(a1 + 2208);
    v19 = 48 * v18;
    v20 = 48 * v18 + v12;
    v21 = *(_DWORD *)(v20 + 44);
    v47.m128i_i64[0] = v7;
    v47.m128i_i64[1] = v6;
    LODWORD(v46) = v45;
    *(_DWORD *)v20 = v45;
    v22 = _mm_loadu_si128(&v47);
    *(_QWORD *)(v20 + 24) = a2;
    *(_DWORD *)(v20 + 32) = v5;
    *(_QWORD *)(v20 + 40) = -1;
    *(__m128i *)(v20 + 8) = v22;
    --*(_DWORD *)(a1 + 2212);
    *(_DWORD *)(a1 + 2208) = v21;
    goto LABEL_11;
  }
LABEL_27:
  v15 = 0xFFFFFFFFLL;
  if ( *(_DWORD *)(a1 + 2212) )
    goto LABEL_10;
LABEL_28:
  v34 = v11;
  v35 = *(unsigned int *)(a1 + 1804);
  v49 = v5;
  v36 = v34 + 1;
  v47.m128i_i64[0] = v7;
  v37 = (const __m128i *)&v46;
  LODWORD(v46) = v45;
  v47.m128i_i64[1] = v6;
  v48 = a2;
  v50 = -1;
  if ( v34 + 1 > v35 )
  {
    v41 = a1 + 1792;
    v42 = (const void *)(a1 + 1808);
    if ( v12 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v12 + 48 * v34 )
    {
      sub_C8D5F0(v41, v42, v36, 0x30u, a5, v13);
      v13 = v45 & 0x7FFFFFFF;
      v14 = v13;
      v12 = *(_QWORD *)(a1 + 1792);
      v34 = *(unsigned int *)(a1 + 1800);
      v37 = (const __m128i *)&v46;
    }
    else
    {
      v44 = (char *)&v46 - v12;
      sub_C8D5F0(v41, v42, v36, 0x30u, a5, v13);
      v14 = v45 & 0x7FFFFFFF;
      v12 = *(_QWORD *)(a1 + 1792);
      v34 = *(unsigned int *)(a1 + 1800);
      v13 = v14;
      v37 = (const __m128i *)&v44[v12];
    }
  }
  v38 = (__m128i *)(48 * v34 + v12);
  *v38 = _mm_loadu_si128(v37);
  v38[1] = _mm_loadu_si128(v37 + 1);
  v38[2] = _mm_loadu_si128(v37 + 2);
  LODWORD(v18) = *(_DWORD *)(a1 + 1800);
  *(_DWORD *)(a1 + 1800) = v18 + 1;
  v19 = 48LL * (unsigned int)v18;
LABEL_11:
  if ( (_DWORD)v15 == -1 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 2192) + v14) = v18;
    *(_DWORD *)(*(_QWORD *)(a1 + 1792) + v19 + 40) = v18;
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 1792);
    v24 = *(unsigned int *)(v23 + 48 * v15 + 40);
    *(_DWORD *)(v23 + 48 * v24 + 44) = v18;
    *(_DWORD *)(*(_QWORD *)(a1 + 1792) + 48 * v15 + 40) = v18;
    *(_DWORD *)(*(_QWORD *)(a1 + 1792) + v19 + 40) = v24;
  }
  v25 = *(_DWORD *)(a1 + 1440);
  result = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1768) + v14);
  if ( (unsigned int)result < v25 )
  {
    v27 = *(_QWORD *)(a1 + 1432);
    while ( 1 )
    {
      v28 = (unsigned int)result;
      v29 = (_DWORD *)(v27 + 40LL * (unsigned int)result);
      if ( (*v29 & 0x7FFFFFFF) == (_DWORD)v13 )
      {
        v30 = (unsigned int)v29[8];
        if ( (_DWORD)v30 != -1 && *(_DWORD *)(v27 + 40 * v30 + 36) == -1 )
          break;
      }
      result = (unsigned int)(result + 256);
      if ( v25 <= (unsigned int)result )
        return result;
    }
    if ( (_DWORD)result != -1 )
    {
      while ( 1 )
      {
        v31 = 40 * v28;
        v32 = v27 + 40 * v28;
        if ( v6 & *(_QWORD *)(v32 + 16) | v7 & *(_QWORD *)(v32 + 8) )
        {
          v33 = *(_QWORD *)(v32 + 24);
          if ( v33 != a2 )
          {
            v46 = a2 & 0xFFFFFFFFFFFFFFF9LL | 2;
            v47.m128i_i64[0] = v45;
            sub_2F8F1B0(v33, (__int64)&v46, 1u, 5 * v28, a5, v13);
            v27 = *(_QWORD *)(a1 + 1432);
            v32 = v27 + v31;
          }
        }
        result = *(unsigned int *)(v32 + 36);
        if ( (_DWORD)result == -1 )
          break;
        v28 = (unsigned int)result;
      }
    }
  }
  return result;
}
