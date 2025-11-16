// Function: sub_37DE250
// Address: 0x37de250
//
__int64 __fastcall sub_37DE250(__int64 a1, int *a2, const __m128i *a3)
{
  int v6; // eax
  char v7; // r9
  int v8; // r9d
  __int64 v9; // r8
  int v10; // esi
  unsigned int v11; // edx
  int *v12; // rcx
  int v13; // edi
  __int64 result; // rax
  unsigned int v15; // esi
  unsigned int v16; // edx
  int v17; // ecx
  unsigned int v18; // r8d
  int *v19; // rdx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rcx
  int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  __m128i v27; // xmm5
  __m128i v28; // xmm6
  __int64 v29; // rax
  int v30; // r11d
  int *v31; // r10
  int v32; // eax
  unsigned __int64 v33; // r8
  unsigned __int64 v34; // rsi
  __m128i v35; // xmm7
  const __m128i *v36; // rax
  __m128i *v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // r9
  char *v40; // r12
  int *v41; // [rsp+8h] [rbp-78h] BYREF
  int v42; // [rsp+10h] [rbp-70h] BYREF
  int v43; // [rsp+14h] [rbp-6Ch]
  __m128i v44; // [rsp+18h] [rbp-68h]
  __m128i v45; // [rsp+28h] [rbp-58h]
  __m128i v46; // [rsp+38h] [rbp-48h]
  __m128i v47; // [rsp+48h] [rbp-38h]

  v6 = *a2;
  v7 = *(_BYTE *)(a1 + 8);
  v43 = 0;
  v42 = v6;
  v8 = v7 & 1;
  if ( v8 )
  {
    v9 = a1 + 16;
    v10 = 7;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 24);
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v41 = 0;
      v17 = (v16 >> 1) + 1;
LABEL_9:
      v18 = 3 * v15;
      goto LABEL_10;
    }
    v10 = v15 - 1;
  }
  v11 = v10 & (37 * v6);
  v12 = (int *)(v9 + 8LL * v11);
  v13 = *v12;
  if ( v6 == *v12 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 80) + 72LL * (unsigned int)v12[1];
    *(__m128i *)(result + 8) = _mm_loadu_si128(a3);
    *(__m128i *)(result + 24) = _mm_loadu_si128(a3 + 1);
    *(__m128i *)(result + 40) = _mm_loadu_si128(a3 + 2);
    *(_QWORD *)(result + 56) = a3[3].m128i_i64[0];
    *(_DWORD *)(result + 64) = a3[3].m128i_i32[2];
    return result;
  }
  v30 = 1;
  v31 = 0;
  while ( v13 != -1 )
  {
    if ( !v31 && v13 == -2 )
      v31 = v12;
    v11 = v10 & (v30 + v11);
    v12 = (int *)(v9 + 8LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
      goto LABEL_4;
    ++v30;
  }
  v16 = *(_DWORD *)(a1 + 8);
  if ( !v31 )
    v31 = v12;
  ++*(_QWORD *)a1;
  v41 = v31;
  v17 = (v16 >> 1) + 1;
  if ( !(_BYTE)v8 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v18 = 24;
  v15 = 8;
LABEL_10:
  if ( v18 <= 4 * v17 )
  {
    v15 *= 2;
    goto LABEL_26;
  }
  if ( v15 - *(_DWORD *)(a1 + 12) - v17 <= v15 >> 3 )
  {
LABEL_26:
    sub_375BDE0(a1, v15);
    sub_37C5ED0(a1, &v42, &v41);
    v6 = v42;
    v16 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
  v19 = v41;
  if ( *v41 != -1 )
    --*(_DWORD *)(a1 + 12);
  *v19 = v6;
  v19[1] = v43;
  v19[1] = *(_DWORD *)(a1 + 88);
  v20 = *(unsigned int *)(a1 + 88);
  v21 = *(unsigned int *)(a1 + 92);
  v22 = *(_DWORD *)(a1 + 88);
  if ( v20 >= v21 )
  {
    v32 = *a2;
    v33 = v20 + 1;
    v34 = *(_QWORD *)(a1 + 80);
    v44 = _mm_loadu_si128(a3);
    v35 = _mm_loadu_si128(a3 + 1);
    v42 = v32;
    v36 = (const __m128i *)&v42;
    v45 = v35;
    v46 = _mm_loadu_si128(a3 + 2);
    v47 = _mm_loadu_si128(a3 + 3);
    if ( v21 < v20 + 1 )
    {
      v38 = a1 + 80;
      v39 = a1 + 96;
      if ( v34 > (unsigned __int64)&v42 || (unsigned __int64)&v42 >= v34 + 72 * v20 )
      {
        sub_C8D5F0(v38, (const void *)(a1 + 96), v33, 0x48u, v33, v39);
        v34 = *(_QWORD *)(a1 + 80);
        v20 = *(unsigned int *)(a1 + 88);
        v36 = (const __m128i *)&v42;
      }
      else
      {
        v40 = (char *)&v42 - v34;
        sub_C8D5F0(v38, (const void *)(a1 + 96), v33, 0x48u, v33, v39);
        v34 = *(_QWORD *)(a1 + 80);
        v20 = *(unsigned int *)(a1 + 88);
        v36 = (const __m128i *)&v40[v34];
      }
    }
    v37 = (__m128i *)(v34 + 72 * v20);
    *v37 = _mm_loadu_si128(v36);
    v37[1] = _mm_loadu_si128(v36 + 1);
    v37[2] = _mm_loadu_si128(v36 + 2);
    v37[3] = _mm_loadu_si128(v36 + 3);
    v37[4].m128i_i64[0] = v36[4].m128i_i64[0];
    v23 = *(_QWORD *)(a1 + 80);
    v29 = (unsigned int)(*(_DWORD *)(a1 + 88) + 1);
    *(_DWORD *)(a1 + 88) = v29;
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 80);
    v24 = v23 + 72 * v20;
    if ( v24 )
    {
      v25 = _mm_loadu_si128(a3);
      v26 = _mm_loadu_si128(a3 + 1);
      v27 = _mm_loadu_si128(a3 + 2);
      v28 = _mm_loadu_si128(a3 + 3);
      *(_DWORD *)v24 = *a2;
      *(__m128i *)(v24 + 8) = v25;
      *(__m128i *)(v24 + 24) = v26;
      *(__m128i *)(v24 + 40) = v27;
      *(__m128i *)(v24 + 56) = v28;
      v22 = *(_DWORD *)(a1 + 88);
      v23 = *(_QWORD *)(a1 + 80);
    }
    v29 = (unsigned int)(v22 + 1);
    *(_DWORD *)(a1 + 88) = v29;
  }
  return v23 + 72 * v29 - 72;
}
