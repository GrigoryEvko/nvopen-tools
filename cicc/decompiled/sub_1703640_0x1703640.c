// Function: sub_1703640
// Address: 0x1703640
//
__int64 __fastcall sub_1703640(__int64 a1, unsigned __int64 *a2)
{
  __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r12
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v14; // eax
  int v15; // edx
  __m128i *v16; // rsi
  __int8 *v17; // rsi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r9d
  __int64 *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  unsigned int v29; // r15d
  __int64 *v30; // rdi
  __int64 v31; // rcx
  __m128i v32; // [rsp+0h] [rbp-50h] BYREF
  __int64 v33; // [rsp+10h] [rbp-40h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
LABEL_3:
    v12 = *((unsigned int *)v10 + 2);
    return *(_QWORD *)(a1 + 32) + 24 * v12 + 8;
  }
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_24:
    sub_14672C0(a1, 2 * v5);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 8);
      v21 = (v18 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v8;
      if ( v4 != *v8 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v8;
          v21 = v19 & (v23 + v21);
          v8 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v23;
        }
        if ( v24 )
          v8 = v24;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
    sub_14672C0(a1, v5);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = 1;
      v29 = v26 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v30 = 0;
      v8 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v8;
      if ( v4 != *v8 )
      {
        while ( v31 != -8 )
        {
          if ( !v30 && v31 == -16 )
            v30 = v8;
          v29 = v26 & (v28 + v29);
          v8 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v28;
        }
        if ( v30 )
          v8 = v30;
      }
      goto LABEL_15;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = 0;
  v16 = *(__m128i **)(a1 + 40);
  v32 = (__m128i)*a2;
  v33 = 0;
  if ( v16 == *(__m128i **)(a1 + 48) )
  {
    sub_1702BB0((const __m128i **)(a1 + 32), v16, &v32);
    v17 = *(__int8 **)(a1 + 40);
  }
  else
  {
    if ( v16 )
    {
      *v16 = _mm_loadu_si128(&v32);
      v16[1].m128i_i64[0] = v33;
      v16 = *(__m128i **)(a1 + 40);
    }
    v17 = &v16[1].m128i_i8[8];
    *(_QWORD *)(a1 + 40) = v17;
  }
  v12 = -1431655765 * (unsigned int)((__int64)&v17[-*(_QWORD *)(a1 + 32)] >> 3) - 1;
  *((_DWORD *)v8 + 2) = v12;
  return *(_QWORD *)(a1 + 32) + 24 * v12 + 8;
}
