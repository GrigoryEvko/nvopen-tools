// Function: sub_B97400
// Address: 0xb97400
//
__int64 *__fastcall sub_B97400(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r8
  __int64 v5; // r9
  int v6; // edi
  unsigned int v7; // eax
  __int64 *v8; // rcx
  __int64 v9; // r10
  unsigned int v10; // eax
  int v11; // ecx
  int v12; // eax
  int v13; // esi
  int v14; // edi
  int v15; // r9d
  __int64 *result; // rax
  __int64 v17; // r11
  int v18; // r8d
  unsigned int v19; // esi
  _QWORD *v20; // r12
  __int64 v21; // r10
  __int64 v22; // rax
  unsigned int v23; // esi
  int v24; // ecx
  unsigned int v25; // r8d
  __int64 v26; // rcx
  int v27; // ecx
  int v28; // r14d
  _QWORD *v29; // r13
  int v30; // r11d
  __int64 *v31; // [rsp+8h] [rbp-58h] BYREF
  __int64 v32; // [rsp+10h] [rbp-50h] BYREF
  __m128i v33[4]; // [rsp+18h] [rbp-48h] BYREF

  v4 = *(_BYTE *)(a1 + 24) & 1;
  if ( v4 )
  {
    v5 = a1 + 32;
    v6 = 3;
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 40);
    v5 = *(_QWORD *)(a1 + 32);
    if ( !(_DWORD)v22 )
      goto LABEL_20;
    v6 = v22 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 24LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
    goto LABEL_4;
  v27 = 1;
  while ( v9 != -4096 )
  {
    v30 = v27 + 1;
    v7 = v6 & (v27 + v7);
    v8 = (__int64 *)(v5 + 24LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      goto LABEL_4;
    v27 = v30;
  }
  if ( v4 )
  {
    v26 = 96;
    goto LABEL_21;
  }
  v22 = *(unsigned int *)(a1 + 40);
LABEL_20:
  v26 = 24 * v22;
LABEL_21:
  v8 = (__int64 *)(v5 + v26);
LABEL_4:
  *v8 = -8192;
  v10 = *(_DWORD *)(a1 + 24);
  v32 = a3;
  v33[0] = _mm_loadu_si128((const __m128i *)(v8 + 1));
  v11 = (v10 >> 1) - 1;
  v12 = (2 * v11) | v10 & 1;
  v13 = *(_DWORD *)(a1 + 28);
  v14 = v11 & 0x7FFFFFFF;
  *(_DWORD *)(a1 + 24) = v12;
  v15 = v13 + 1;
  *(_DWORD *)(a1 + 28) = v13 + 1;
  result = (__int64 *)(v12 & 1);
  if ( (_DWORD)result )
  {
    v17 = a1 + 32;
    v18 = 3;
  }
  else
  {
    v23 = *(_DWORD *)(a1 + 40);
    v17 = *(_QWORD *)(a1 + 32);
    if ( !v23 )
    {
      ++*(_QWORD *)(a1 + 16);
      v24 = v14 + 1;
      v31 = 0;
LABEL_13:
      v25 = 3 * v23;
      goto LABEL_14;
    }
    v18 = v23 - 1;
  }
  v19 = v18 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v20 = (_QWORD *)(v17 + 24LL * v19);
  v21 = *v20;
  if ( a3 == *v20 )
    return result;
  v28 = 1;
  v29 = 0;
  while ( v21 != -4096 )
  {
    if ( v21 != -8192 || v29 )
      v20 = v29;
    v19 = v18 & (v28 + v19);
    v21 = *(_QWORD *)(v17 + 24LL * v19);
    if ( a3 == v21 )
      return result;
    ++v28;
    v29 = v20;
    v20 = (_QWORD *)(v17 + 24LL * v19);
  }
  if ( !v29 )
    v29 = v20;
  ++*(_QWORD *)(a1 + 16);
  v24 = (v11 & 0x7FFFFFFF) + 1;
  v31 = v29;
  if ( !(_BYTE)result )
  {
    v23 = *(_DWORD *)(a1 + 40);
    goto LABEL_13;
  }
  v25 = 12;
  v23 = 4;
LABEL_14:
  if ( v25 <= 4 * v24 )
  {
    v23 *= 2;
    goto LABEL_33;
  }
  if ( v23 - v15 - v24 <= v23 >> 3 )
  {
LABEL_33:
    sub_B95E60(a1 + 16, v23);
    sub_B926F0(a1 + 16, &v32, &v31);
    a3 = v32;
    v14 = *(_DWORD *)(a1 + 24) >> 1;
  }
  *(_DWORD *)(a1 + 24) = (2 * v14 + 2) | *(_DWORD *)(a1 + 24) & 1;
  result = v31;
  if ( *v31 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *result = a3;
  *(__m128i *)(result + 1) = _mm_loadu_si128(v33);
  return result;
}
