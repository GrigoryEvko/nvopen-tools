// Function: sub_18FA710
// Address: 0x18fa710
//
__int64 __fastcall sub_18FA710(__int64 a1, const __m128i *a2)
{
  unsigned __int8 v4; // r14
  __int64 v5; // r15
  __int64 v6; // r13
  unsigned int v7; // esi
  __m128i v8; // xmm2
  __int64 v9; // rcx
  __int64 v10; // r10
  unsigned int i; // r8d
  __int64 v12; // rax
  __int64 v13; // r11
  unsigned int v14; // r8d
  __int64 v15; // rax
  int v16; // eax
  int v17; // edi
  __m128i v18; // xmm3
  __m128i v19; // xmm0
  __m128i *v20; // rsi
  __m128i *v21; // rax
  int v23; // eax
  int v24; // edx
  __int64 v25; // rsi
  int v26; // r9d
  __int64 v27; // r8
  unsigned int j; // eax
  __int64 v29; // r10
  unsigned int v30; // eax
  int v31; // eax
  int v32; // eax
  int v33; // r8d
  unsigned int k; // edx
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // [rsp+0h] [rbp-80h]
  int v38; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+8h] [rbp-78h]
  __m128i v40; // [rsp+10h] [rbp-70h] BYREF
  __int64 v41; // [rsp+20h] [rbp-60h]
  __m128i v42; // [rsp+30h] [rbp-50h] BYREF
  __m128i v43[4]; // [rsp+40h] [rbp-40h] BYREF

  v4 = a2->m128i_i8[0];
  v5 = a2->m128i_i64[1];
  v6 = a2[1].m128i_i64[0];
  v42 = _mm_loadu_si128(a2);
  v7 = *(_DWORD *)(a1 + 24);
  v42.m128i_i8[0] = v4;
  v42.m128i_i64[1] = v5;
  v8 = _mm_loadu_si128(&v42);
  v43[0].m128i_i64[0] = v6;
  v41 = v6;
  v40 = v8;
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_29;
  }
  v38 = 1;
  v9 = 0;
  v10 = *(_QWORD *)(a1 + 8);
  for ( i = (v7 - 1) & (v4 ^ v6 ^ v5); ; i = (v7 - 1) & v14 )
  {
    v12 = v10 + 32LL * i;
    v13 = *(_QWORD *)(v12 + 8);
    if ( v4 == *(_BYTE *)v12 && v5 == v13 && *(_QWORD *)(v12 + 16) == v6 )
    {
      v15 = *(unsigned int *)(v12 + 24);
      return *(_QWORD *)(a1 + 32) + 32 * v15 + 24;
    }
    if ( !*(_BYTE *)v12 )
      break;
    if ( !v13 && !(*(_QWORD *)(v12 + 16) | v9) )
      v9 = v10 + 32LL * i;
LABEL_6:
    v14 = v38 + i;
    ++v38;
  }
  if ( v13 || *(_QWORD *)(v12 + 16) )
    goto LABEL_6;
  v37 = 0;
  if ( !v9 )
    v9 = v10 + 32LL * i;
  v16 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v7 )
  {
LABEL_29:
    sub_18FA4C0(a1, 2 * v7);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v26 = 1;
      v27 = 0;
      for ( j = (v23 - 1) & (v4 ^ v6 ^ v5); ; j = v24 & v30 )
      {
        v25 = *(_QWORD *)(a1 + 8);
        v9 = v25 + 32LL * j;
        v29 = *(_QWORD *)(v9 + 8);
        if ( v4 == *(_BYTE *)v9 && v5 == v29 && v6 == *(_QWORD *)(v9 + 16) )
          break;
        if ( *(_BYTE *)v9 )
        {
          if ( !v29 && !(*(_QWORD *)(v9 + 16) | v27) )
            v27 = v25 + 32LL * j;
        }
        else if ( !v29 && !*(_QWORD *)(v9 + 16) )
        {
          if ( v27 )
            v9 = v27;
          v17 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_16;
        }
        v30 = v26 + j;
        ++v26;
      }
      goto LABEL_43;
    }
LABEL_66:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 20) - v17 <= v7 >> 3 )
  {
    sub_18FA4C0(a1, v7);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = 1;
      for ( k = v32 & (v4 ^ v6 ^ v5); ; k = v32 & v36 )
      {
        v9 = *(_QWORD *)(a1 + 8) + 32LL * k;
        v35 = *(_QWORD *)(v9 + 8);
        if ( v4 == *(_BYTE *)v9 && v5 == v35 && v6 == *(_QWORD *)(v9 + 16) )
          break;
        if ( *(_BYTE *)v9 )
        {
          if ( !v35 )
          {
            if ( *(_QWORD *)(v9 + 16) | v37 )
              v9 = v37;
            v37 = v9;
          }
        }
        else if ( !v35 && !*(_QWORD *)(v9 + 16) )
        {
          v17 = *(_DWORD *)(a1 + 16) + 1;
          if ( v37 )
            v9 = v37;
          goto LABEL_16;
        }
        v36 = v33 + k;
        ++v33;
      }
LABEL_43:
      v17 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_16;
    }
    goto LABEL_66;
  }
LABEL_16:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_BYTE *)v9 || *(_QWORD *)(v9 + 8) || *(_QWORD *)(v9 + 16) )
    --*(_DWORD *)(a1 + 20);
  v40.m128i_i8[0] = v4;
  v40.m128i_i64[1] = v5;
  v18 = _mm_loadu_si128(&v40);
  *(_QWORD *)(v9 + 16) = v6;
  *(_DWORD *)(v9 + 24) = 0;
  *(__m128i *)v9 = v18;
  v19 = _mm_loadu_si128(a2);
  v20 = *(__m128i **)(a1 + 40);
  v43[0] = (__m128i)a2[1].m128i_u64[0];
  v42 = v19;
  if ( v20 == *(__m128i **)(a1 + 48) )
  {
    v39 = v9;
    sub_18FA0E0((const __m128i **)(a1 + 32), v20, &v42);
    v21 = *(__m128i **)(a1 + 40);
    v9 = v39;
  }
  else
  {
    if ( v20 )
    {
      *v20 = v19;
      v20[1] = _mm_loadu_si128(v43);
      v20 = *(__m128i **)(a1 + 40);
    }
    v21 = v20 + 2;
    *(_QWORD *)(a1 + 40) = v20 + 2;
  }
  v15 = (unsigned int)(((__int64)v21->m128i_i64 - *(_QWORD *)(a1 + 32)) >> 5) - 1;
  *(_DWORD *)(v9 + 24) = v15;
  return *(_QWORD *)(a1 + 32) + 32 * v15 + 24;
}
