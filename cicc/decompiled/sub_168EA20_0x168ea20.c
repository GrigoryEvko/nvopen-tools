// Function: sub_168EA20
// Address: 0x168ea20
//
_QWORD *__fastcall sub_168EA20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned int v7; // esi
  __int64 v8; // r9
  unsigned int v9; // r13d
  unsigned int v10; // ecx
  _QWORD *result; // rax
  __int64 v12; // rdx
  __m128i *v13; // rsi
  int v14; // r11d
  _QWORD *v15; // rdi
  int v16; // eax
  int v17; // edx
  const __m128i **v18; // rdi
  int v19; // eax
  int v20; // eax
  __int64 v21; // r8
  unsigned int v22; // ecx
  __int64 v23; // rsi
  int v24; // r10d
  _QWORD *v25; // r9
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  _QWORD *v29; // r8
  unsigned int v30; // r13d
  int v31; // r9d
  __int64 v32; // rcx
  __m128i v33; // [rsp+0h] [rbp-30h] BYREF

  v4 = a1 + 304;
  v33.m128i_i64[0] = a2;
  v7 = *(_DWORD *)(a1 + 328);
  v33.m128i_i64[1] = a3;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 304);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(a1 + 312);
  v9 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  v10 = (v7 - 1) & v9;
  result = (_QWORD *)(v8 + 32LL * v10);
  v12 = *result;
  if ( *result != a4 )
  {
    v14 = 1;
    v15 = 0;
    while ( v12 != -8 )
    {
      if ( !v15 && v12 == -16 )
        v15 = result;
      v10 = (v7 - 1) & (v14 + v10);
      result = (_QWORD *)(v8 + 32LL * v10);
      v12 = *result;
      if ( *result == a4 )
        goto LABEL_3;
      ++v14;
    }
    if ( !v15 )
      v15 = result;
    v16 = *(_DWORD *)(a1 + 320);
    ++*(_QWORD *)(a1 + 304);
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 324) - v17 > v7 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(a1 + 320) = v17;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 324);
        *v15 = a4;
        v13 = 0;
        v18 = (const __m128i **)(v15 + 1);
        *v18 = 0;
        v18[1] = 0;
        v18[2] = 0;
        return (_QWORD *)sub_1516D00(v18, v13, &v33);
      }
      sub_168E800(v4, v7);
      v26 = *(_DWORD *)(a1 + 328);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a1 + 312);
        v29 = 0;
        v30 = v27 & v9;
        v31 = 1;
        v17 = *(_DWORD *)(a1 + 320) + 1;
        v15 = (_QWORD *)(v28 + 32LL * v30);
        v32 = *v15;
        if ( *v15 != a4 )
        {
          while ( v32 != -8 )
          {
            if ( !v29 && v32 == -16 )
              v29 = v15;
            v30 = v27 & (v31 + v30);
            v15 = (_QWORD *)(v28 + 32LL * v30);
            v32 = *v15;
            if ( *v15 == a4 )
              goto LABEL_13;
            ++v31;
          }
          if ( v29 )
            v15 = v29;
        }
        goto LABEL_13;
      }
LABEL_47:
      ++*(_DWORD *)(a1 + 320);
      BUG();
    }
LABEL_18:
    sub_168E800(v4, 2 * v7);
    v19 = *(_DWORD *)(a1 + 328);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 312);
      v22 = v20 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v17 = *(_DWORD *)(a1 + 320) + 1;
      v15 = (_QWORD *)(v21 + 32LL * v22);
      v23 = *v15;
      if ( *v15 != a4 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( !v25 && v23 == -16 )
            v25 = v15;
          v22 = v20 & (v24 + v22);
          v15 = (_QWORD *)(v21 + 32LL * v22);
          v23 = *v15;
          if ( *v15 == a4 )
            goto LABEL_13;
          ++v24;
        }
        if ( v25 )
          v15 = v25;
      }
      goto LABEL_13;
    }
    goto LABEL_47;
  }
LABEL_3:
  v13 = (__m128i *)result[2];
  if ( v13 == (__m128i *)result[3] )
  {
    v18 = (const __m128i **)(result + 1);
    return (_QWORD *)sub_1516D00(v18, v13, &v33);
  }
  if ( v13 )
  {
    *v13 = _mm_load_si128(&v33);
    v13 = (__m128i *)result[2];
  }
  result[2] = v13 + 1;
  return result;
}
