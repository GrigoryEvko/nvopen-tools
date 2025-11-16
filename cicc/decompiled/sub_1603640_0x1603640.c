// Function: sub_1603640
// Address: 0x1603640
//
__int64 __fastcall sub_1603640(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // r9d
  unsigned int v8; // edx
  __int64 *v9; // r12
  __int64 v10; // rdi
  const void *v11; // r8
  const void *v12; // rdx
  __int64 result; // rax
  __int64 *v14; // rdi
  size_t v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned int v18; // r8d
  __int64 v19; // r10
  _QWORD *v20; // r9
  __int64 v21; // rcx
  int v22; // ebx
  _QWORD *v23; // rdx
  int v24; // eax
  int v25; // ecx
  int v26; // eax
  int v27; // edi
  __int64 v28; // r9
  unsigned int v29; // eax
  __int64 v30; // r8
  int v31; // r11d
  _QWORD *v32; // r10
  int v33; // eax
  int v34; // edi
  int v35; // r11d
  __int64 v36; // r9
  unsigned int v37; // eax
  __int64 v38; // r8
  __m128i *v39; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  __m128i v41[4]; // [rsp+18h] [rbp-48h] BYREF

  v4 = *a1;
  v5 = *(unsigned int *)(*a1 + 2952);
  if ( !(_DWORD)v5 )
  {
LABEL_18:
    v12 = (const void *)(a3 + 16);
    goto LABEL_19;
  }
  v6 = *(_QWORD *)(v4 + 2936);
  v7 = 1;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 40LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    while ( v10 != -8 )
    {
      v8 = (v5 - 1) & (v7 + v8);
      v9 = (__int64 *)(v6 + 40LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      ++v7;
    }
    goto LABEL_18;
  }
LABEL_3:
  v11 = *(const void **)a3;
  v12 = (const void *)(a3 + 16);
  result = v6 + 40 * v5;
  if ( v9 != (__int64 *)result )
  {
    v14 = (__int64 *)v9[1];
    if ( v11 == v12 )
    {
      v15 = *(_QWORD *)(a3 + 8);
      if ( v15 )
      {
        if ( v15 == 1 )
        {
          result = *(unsigned __int8 *)(a3 + 16);
          *(_BYTE *)v14 = result;
        }
        else
        {
          result = (__int64)memcpy(v14, *(const void **)a3, v15);
        }
        v15 = *(_QWORD *)(a3 + 8);
        v14 = (__int64 *)v9[1];
      }
      v9[2] = v15;
      *((_BYTE *)v14 + v15) = 0;
      v14 = *(__int64 **)a3;
      goto LABEL_8;
    }
    if ( v14 == v9 + 3 )
    {
      v9[1] = (__int64)v11;
      v9[2] = *(_QWORD *)(a3 + 8);
      result = *(_QWORD *)(a3 + 16);
      v9[3] = result;
    }
    else
    {
      v9[1] = (__int64)v11;
      result = v9[3];
      v9[2] = *(_QWORD *)(a3 + 8);
      v9[3] = *(_QWORD *)(a3 + 16);
      if ( v14 )
      {
        *(_QWORD *)a3 = v14;
        *(_QWORD *)(a3 + 16) = result;
LABEL_8:
        *(_QWORD *)(a3 + 8) = 0;
        *(_BYTE *)v14 = 0;
        return result;
      }
    }
    *(_QWORD *)a3 = v12;
    v14 = (__int64 *)(a3 + 16);
    goto LABEL_8;
  }
LABEL_19:
  v16 = v4 + 2928;
  v39 = v41;
  if ( *(const void **)a3 == v12 )
  {
    v41[0] = _mm_loadu_si128((const __m128i *)(a3 + 16));
  }
  else
  {
    v39 = *(__m128i **)a3;
    v41[0].m128i_i64[0] = *(_QWORD *)(a3 + 16);
  }
  v17 = *(_QWORD *)(a3 + 8);
  *(_QWORD *)a3 = v12;
  *(_QWORD *)(a3 + 8) = 0;
  *(_BYTE *)(a3 + 16) = 0;
  v18 = *(_DWORD *)(v4 + 2952);
  v40 = v17;
  if ( !v18 )
  {
    ++*(_QWORD *)(v4 + 2928);
    goto LABEL_42;
  }
  v19 = *(_QWORD *)(v4 + 2936);
  result = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (_QWORD *)(v19 + 40 * result);
  v21 = *v20;
  if ( a2 == *v20 )
  {
LABEL_23:
    if ( v39 != v41 )
      return j_j___libc_free_0(v39, v41[0].m128i_i64[0] + 1);
    return result;
  }
  v22 = 1;
  v23 = 0;
  while ( v21 != -8 )
  {
    if ( v21 != -16 || v23 )
      v20 = v23;
    result = (v18 - 1) & (v22 + (_DWORD)result);
    v21 = *(_QWORD *)(v19 + 40LL * (unsigned int)result);
    if ( a2 == v21 )
      goto LABEL_23;
    v23 = v20;
    ++v22;
    v20 = (_QWORD *)(v19 + 40LL * (unsigned int)result);
  }
  v24 = *(_DWORD *)(v4 + 2944);
  if ( !v23 )
    v23 = v20;
  ++*(_QWORD *)(v4 + 2928);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v18 )
  {
LABEL_42:
    sub_1603400(v16, 2 * v18);
    v26 = *(_DWORD *)(v4 + 2952);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(v4 + 2936);
      v25 = *(_DWORD *)(v4 + 2944) + 1;
      v29 = (v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = (_QWORD *)(v28 + 40LL * v29);
      v30 = *v23;
      if ( *v23 == a2 )
        goto LABEL_34;
      v31 = 1;
      v32 = 0;
      while ( v30 != -8 )
      {
        if ( !v32 && v30 == -16 )
          v32 = v23;
        v29 = v27 & (v31 + v29);
        v23 = (_QWORD *)(v28 + 40LL * v29);
        v30 = *v23;
        if ( a2 == *v23 )
          goto LABEL_34;
        ++v31;
      }
LABEL_46:
      if ( v32 )
        v23 = v32;
      goto LABEL_34;
    }
LABEL_67:
    ++*(_DWORD *)(v4 + 2944);
    BUG();
  }
  if ( v18 - *(_DWORD *)(v4 + 2948) - v25 <= v18 >> 3 )
  {
    sub_1603400(v16, v18);
    v33 = *(_DWORD *)(v4 + 2952);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = 1;
      v32 = 0;
      v36 = *(_QWORD *)(v4 + 2936);
      v25 = *(_DWORD *)(v4 + 2944) + 1;
      v37 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = (_QWORD *)(v36 + 40LL * v37);
      v38 = *v23;
      if ( *v23 == a2 )
        goto LABEL_34;
      while ( v38 != -8 )
      {
        if ( v38 == -16 && !v32 )
          v32 = v23;
        v37 = v34 & (v35 + v37);
        v23 = (_QWORD *)(v36 + 40LL * v37);
        v38 = *v23;
        if ( a2 == *v23 )
          goto LABEL_34;
        ++v35;
      }
      goto LABEL_46;
    }
    goto LABEL_67;
  }
LABEL_34:
  *(_DWORD *)(v4 + 2944) = v25;
  if ( *v23 != -8 )
    --*(_DWORD *)(v4 + 2948);
  *v23 = a2;
  v23[1] = v23 + 3;
  if ( v39 == v41 )
  {
    *(__m128i *)(v23 + 3) = _mm_loadu_si128(v41);
  }
  else
  {
    v23[1] = v39;
    v23[3] = v41[0].m128i_i64[0];
  }
  v23[2] = v40;
  return v40;
}
