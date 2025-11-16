// Function: sub_137C180
// Address: 0x137c180
//
__int64 __fastcall sub_137C180(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int32 v4; // eax
  unsigned int v5; // esi
  __int32 v6; // ecx
  __int64 v7; // r10
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  _BYTE *v12; // rsi
  __int64 v13; // r12
  __int32 v14; // r13d
  __int64 result; // rax
  __m128i *v16; // rsi
  int v17; // r13d
  __int64 *v18; // r12
  int v19; // ecx
  int v20; // ecx
  int v21; // eax
  int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // edx
  __int64 v25; // r8
  int v26; // r11d
  __int64 *v27; // r10
  int v28; // eax
  int v29; // esi
  __int64 v30; // r9
  __int64 *v31; // r10
  int v32; // r11d
  unsigned int v33; // edx
  __int64 v34; // [rsp+8h] [rbp-48h] BYREF
  __m128i v35; // [rsp+10h] [rbp-40h] BYREF
  __m128i v36; // [rsp+20h] [rbp-30h] BYREF

  v2 = a1 + 8;
  v4 = *(_DWORD *)a1;
  v34 = a2;
  v5 = *(_DWORD *)(a1 + 32);
  v6 = v4 + 1;
  *(_DWORD *)a1 = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_23;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = v34;
  v9 = (v5 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v34 == *v10 )
    goto LABEL_3;
  v17 = 1;
  v18 = 0;
  while ( v11 != -8 )
  {
    if ( !v18 && v11 == -16 )
      v18 = v10;
    v9 = (v5 - 1) & (v17 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v34 == *v10 )
      goto LABEL_3;
    ++v17;
  }
  v19 = *(_DWORD *)(a1 + 24);
  if ( v18 )
    v10 = v18;
  ++*(_QWORD *)(a1 + 8);
  v20 = v19 + 1;
  if ( 4 * v20 >= 3 * v5 )
  {
LABEL_23:
    sub_137BFC0(v2, 2 * v5);
    v21 = *(_DWORD *)(a1 + 32);
    if ( v21 )
    {
      v8 = v34;
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 16);
      v20 = *(_DWORD *)(a1 + 24) + 1;
      v24 = (v21 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v10 = (__int64 *)(v23 + 16LL * v24);
      v25 = *v10;
      if ( *v10 != v34 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -8 )
        {
          if ( !v27 && v25 == -16 )
            v27 = v10;
          v24 = v22 & (v26 + v24);
          v10 = (__int64 *)(v23 + 16LL * v24);
          v25 = *v10;
          if ( v34 == *v10 )
            goto LABEL_19;
          ++v26;
        }
        if ( v27 )
          v10 = v27;
      }
      goto LABEL_19;
    }
    goto LABEL_51;
  }
  if ( v5 - *(_DWORD *)(a1 + 28) - v20 <= v5 >> 3 )
  {
    sub_137BFC0(v2, v5);
    v28 = *(_DWORD *)(a1 + 32);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 16);
      v31 = 0;
      v32 = 1;
      v20 = *(_DWORD *)(a1 + 24) + 1;
      v33 = (v28 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v10 = (__int64 *)(v30 + 16LL * v33);
      v8 = *v10;
      if ( v34 != *v10 )
      {
        while ( v8 != -8 )
        {
          if ( !v31 && v8 == -16 )
            v31 = v10;
          v33 = v29 & (v32 + v33);
          v10 = (__int64 *)(v30 + 16LL * v33);
          v8 = *v10;
          if ( v34 == *v10 )
            goto LABEL_19;
          ++v32;
        }
        v8 = v34;
        if ( v31 )
          v10 = v31;
      }
      goto LABEL_19;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_19:
  *(_DWORD *)(a1 + 24) = v20;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 28);
  *v10 = v8;
  *((_DWORD *)v10 + 2) = 0;
  v6 = *(_DWORD *)a1;
LABEL_3:
  *((_DWORD *)v10 + 2) = v6;
  v12 = *(_BYTE **)(a1 + 48);
  if ( v12 == *(_BYTE **)(a1 + 56) )
  {
    sub_136D8A0(a1 + 40, v12, &v34);
    v13 = v34;
  }
  else
  {
    v13 = v34;
    if ( v12 )
    {
      *(_QWORD *)v12 = v34;
      v12 = *(_BYTE **)(a1 + 48);
      v13 = v34;
    }
    *(_QWORD *)(a1 + 48) = v12 + 8;
  }
  v14 = *(_DWORD *)a1;
  result = sub_157EBA0(v13);
  v35.m128i_i64[0] = v13;
  v16 = *(__m128i **)(a1 + 96);
  v35.m128i_i64[1] = result;
  v36.m128i_i32[0] = 0;
  v36.m128i_i32[2] = v14;
  if ( v16 == *(__m128i **)(a1 + 104) )
    return sub_137BE30((const __m128i **)(a1 + 88), v16, &v35);
  if ( v16 )
  {
    *v16 = _mm_loadu_si128(&v35);
    v16[1] = _mm_loadu_si128(&v36);
    v16 = *(__m128i **)(a1 + 96);
  }
  *(_QWORD *)(a1 + 96) = v16 + 2;
  return result;
}
