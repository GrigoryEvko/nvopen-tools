// Function: sub_1E0F850
// Address: 0x1e0f850
//
__int64 __fastcall sub_1E0F850(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // edi
  __int64 *v10; // rcx
  __int64 v11; // rdx
  int v12; // r14d
  __int64 *v13; // r11
  int v14; // ecx
  int v15; // ecx
  int v16; // edx
  int v17; // edx
  __int64 v18; // rdi
  __int64 *v19; // r8
  unsigned int v20; // ebx
  int v21; // r9d
  __int64 v22; // rsi
  __m128i *v23; // rsi
  __m128i *v24; // rsi
  int v25; // edx
  int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // edx
  __int64 v29; // rdi
  int v30; // r10d
  __int64 *v31; // r9
  unsigned int v32; // [rsp+Ch] [rbp-44h]
  unsigned int v33; // [rsp+Ch] [rbp-44h]
  __m128i v34; // [rsp+10h] [rbp-40h] BYREF

  if ( *(_DWORD *)a1 < a3 )
    *(_DWORD *)a1 = a3;
  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL))(a2, a1, a3);
  if ( (_DWORD)result != -1 )
  {
    v7 = *(_DWORD *)(a1 + 56);
    if ( v7 )
    {
      v8 = *(_QWORD *)(a1 + 40);
      v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (__int64 *)(v8 + 8LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        return result;
      v12 = 1;
      v13 = 0;
      while ( v11 != -8 )
      {
        if ( v13 || v11 != -16 )
          v10 = v13;
        v9 = (v7 - 1) & (v12 + v9);
        v11 = *(_QWORD *)(v8 + 8LL * v9);
        if ( v11 == a2 )
          return result;
        ++v12;
        v13 = v10;
        v10 = (__int64 *)(v8 + 8LL * v9);
      }
      if ( !v13 )
        v13 = v10;
      v14 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v15 = v14 + 1;
      if ( 4 * v15 < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a1 + 52) - v15 > v7 >> 3 )
        {
LABEL_28:
          *(_DWORD *)(a1 + 48) = v15;
          if ( *v13 != -8 )
            --*(_DWORD *)(a1 + 52);
          *v13 = a2;
          return result;
        }
        v32 = result;
        sub_1E0F6A0(a1 + 32, v7);
        v16 = *(_DWORD *)(a1 + 56);
        if ( v16 )
        {
          v17 = v16 - 1;
          v18 = *(_QWORD *)(a1 + 40);
          v19 = 0;
          v20 = v17 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v21 = 1;
          v13 = (__int64 *)(v18 + 8LL * v20);
          v15 = *(_DWORD *)(a1 + 48) + 1;
          result = v32;
          v22 = *v13;
          if ( *v13 != a2 )
          {
            while ( v22 != -8 )
            {
              if ( !v19 && v22 == -16 )
                v19 = v13;
              v20 = v17 & (v21 + v20);
              v13 = (__int64 *)(v18 + 8LL * v20);
              v22 = *v13;
              if ( *v13 == a2 )
                goto LABEL_28;
              ++v21;
            }
            if ( v19 )
              v13 = v19;
          }
          goto LABEL_28;
        }
LABEL_51:
        ++*(_DWORD *)(a1 + 48);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
    }
    v33 = result;
    sub_1E0F6A0(a1 + 32, 2 * v7);
    v25 = *(_DWORD *)(a1 + 56);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 40);
      v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v27 + 8LL * v28);
      v15 = *(_DWORD *)(a1 + 48) + 1;
      result = v33;
      v29 = *v13;
      if ( *v13 != a2 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v31 )
            v31 = v13;
          v28 = v26 & (v30 + v28);
          v13 = (__int64 *)(v27 + 8LL * v28);
          v29 = *v13;
          if ( *v13 == a2 )
            goto LABEL_28;
          ++v30;
        }
        if ( v31 )
          v13 = v31;
      }
      goto LABEL_28;
    }
    goto LABEL_51;
  }
  v34.m128i_i64[0] = a2;
  v23 = *(__m128i **)(a1 + 16);
  v34.m128i_i32[2] = a3 | 0x80000000;
  if ( v23 == *(__m128i **)(a1 + 24) )
  {
    sub_1E0DAF0((const __m128i **)(a1 + 8), v23, &v34);
    v24 = *(__m128i **)(a1 + 16);
  }
  else
  {
    if ( v23 )
    {
      *v23 = _mm_loadu_si128(&v34);
      v23 = *(__m128i **)(a1 + 16);
    }
    v24 = v23 + 1;
    *(_QWORD *)(a1 + 16) = v24;
  }
  return (unsigned int)(((__int64)v24->m128i_i64 - *(_QWORD *)(a1 + 8)) >> 4) - 1;
}
