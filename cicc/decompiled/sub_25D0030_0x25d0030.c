// Function: sub_25D0030
// Address: 0x25d0030
//
__int64 __fastcall sub_25D0030(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __m128i v6; // xmm0
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rax
  int v11; // ecx
  unsigned int v12; // esi
  int v13; // edx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  int v16; // eax
  __int64 v17; // rcx
  __m128i *v18; // rdx
  __m128i v19; // xmm2
  __int64 v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm3
  unsigned __int64 v23; // r8
  __m128i *v24; // rsi
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __m128i *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // r9
  __int8 *v30; // r12
  __int64 v31; // [rsp+0h] [rbp-70h] BYREF
  __int64 v32; // [rsp+8h] [rbp-68h] BYREF
  __m128i v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+20h] [rbp-50h]
  __m128i v35; // [rsp+30h] [rbp-40h] BYREF
  __int64 v36; // [rsp+40h] [rbp-30h]
  int v37; // [rsp+48h] [rbp-28h]

  v6 = _mm_loadu_si128(a2);
  v7 = a2[1].m128i_i64[0];
  v37 = 0;
  v34 = v7;
  v36 = v7;
  v33 = v6;
  v35 = v6;
  v8 = (unsigned __int8)sub_25CE2B0(a1, (char **)&v35, &v31) == 0;
  v9 = v31;
  if ( !v8 )
    return *(_QWORD *)(a1 + 32) + 32LL * *(unsigned int *)(v31 + 24);
  v11 = *(_DWORD *)(a1 + 16);
  v12 = *(_DWORD *)(a1 + 24);
  v32 = v31;
  ++*(_QWORD *)a1;
  v13 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v12 )
  {
    v12 *= 2;
  }
  else if ( v12 - *(_DWORD *)(a1 + 20) - v13 > v12 >> 3 )
  {
    goto LABEL_5;
  }
  sub_25CFE60(a1, v12);
  sub_25CE2B0(a1, (char **)&v35, &v32);
  v13 = *(_DWORD *)(a1 + 16) + 1;
  v9 = v32;
LABEL_5:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *(_QWORD *)v9 != -1 || *(_QWORD *)(v9 + 16) != -1 )
    --*(_DWORD *)(a1 + 20);
  *(__m128i *)v9 = _mm_loadu_si128(&v35);
  *(_QWORD *)(v9 + 16) = v36;
  *(_DWORD *)(v9 + 24) = v37;
  *(_DWORD *)(v9 + 24) = *(_DWORD *)(a1 + 40);
  v14 = *(unsigned int *)(a1 + 40);
  v15 = *(unsigned int *)(a1 + 44);
  v16 = *(_DWORD *)(a1 + 40);
  if ( v14 >= v15 )
  {
    v22 = _mm_loadu_si128(a2);
    v23 = v14 + 1;
    v24 = &v35;
    v36 = a2[1].m128i_i64[0];
    v25 = *a3;
    v35 = v22;
    v37 = v25;
    v26 = *(_QWORD *)(a1 + 32);
    if ( v15 < v14 + 1 )
    {
      v28 = a1 + 32;
      v29 = a1 + 48;
      if ( v26 > (unsigned __int64)&v35 || (unsigned __int64)&v35 >= v26 + 32 * v14 )
      {
        sub_C8D5F0(v28, (const void *)(a1 + 48), v23, 0x20u, v23, v29);
        v26 = *(_QWORD *)(a1 + 32);
        v14 = *(unsigned int *)(a1 + 40);
        v24 = &v35;
      }
      else
      {
        v30 = &v35.m128i_i8[-v26];
        sub_C8D5F0(v28, (const void *)(a1 + 48), v23, 0x20u, v23, v29);
        v26 = *(_QWORD *)(a1 + 32);
        v14 = *(unsigned int *)(a1 + 40);
        v24 = (__m128i *)&v30[v26];
      }
    }
    v27 = (__m128i *)(v26 + 32 * v14);
    *v27 = _mm_loadu_si128(v24);
    v27[1] = _mm_loadu_si128(v24 + 1);
    v17 = *(_QWORD *)(a1 + 32);
    v21 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v21;
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 32);
    v18 = (__m128i *)(v17 + 32 * v14);
    if ( v18 )
    {
      v19 = _mm_loadu_si128(a2);
      v18[1].m128i_i64[0] = a2[1].m128i_i64[0];
      v20 = *a3;
      *v18 = v19;
      v18[1].m128i_i32[2] = v20;
      v16 = *(_DWORD *)(a1 + 40);
      v17 = *(_QWORD *)(a1 + 32);
    }
    v21 = (unsigned int)(v16 + 1);
    *(_DWORD *)(a1 + 40) = v21;
  }
  return v17 + 32 * v21 - 32;
}
