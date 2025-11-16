// Function: sub_1F818B0
// Address: 0x1f818b0
//
__int64 __fastcall sub_1F818B0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v7; // rsi
  __m128i *v8; // r8
  __m128i *v9; // rcx
  __m128i *v10; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r13
  __int64 v15; // rax
  _BOOL4 v16; // r15d
  __m128i *v17; // rax
  unsigned __int64 v18; // rax
  _BOOL4 v19; // r8d
  __m128i *v20; // rax
  int v21; // eax
  __m128i *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  _BOOL4 v29; // r15d
  __m128i *v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-40h]
  _BOOL4 v34; // [rsp+Ch] [rbp-34h]

  if ( *(_QWORD *)(a1 + 312) )
  {
    v13 = sub_1F817F0(a1 + 272, (unsigned __int64 *)a2);
    v14 = v12;
    v15 = 0;
    if ( v12 )
    {
      v16 = 1;
      if ( !v13 && v12 != a1 + 280 )
      {
        v31 = *(_QWORD *)(v12 + 32);
        v16 = a2->m128i_i64[0] < v31 || a2->m128i_i64[0] == v31 && a2->m128i_i32[2] < *(_DWORD *)(v12 + 40);
      }
      v17 = (__m128i *)sub_22077B0(48);
      v17[2] = _mm_loadu_si128(a2);
      sub_220F040(v16, v17, v14, a1 + 280);
      ++*(_QWORD *)(a1 + 312);
      v15 = 1;
    }
    return (v15 << 32) | 1;
  }
  else
  {
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(__m128i **)a1;
    v9 = (__m128i *)(*(_QWORD *)a1 + 16 * v7);
    if ( *(__m128i **)a1 == v9 )
      goto LABEL_13;
    v10 = *(__m128i **)a1;
    while ( 1 )
    {
      if ( v10->m128i_i64[0] == a2->m128i_i64[0] )
      {
        a6 = a2->m128i_i32[2];
        if ( v10->m128i_i32[2] == a6 )
          break;
      }
      if ( v9 == ++v10 )
        goto LABEL_13;
    }
    if ( v9 == v10 )
    {
LABEL_13:
      v33 = a1 + 280;
      if ( v7 > 0xF )
      {
        while ( 1 )
        {
          v22 = &v8[v7 - 1];
          v23 = sub_1F817F0(a1 + 272, (unsigned __int64 *)v22);
          v25 = v24;
          if ( v24 )
          {
            v19 = 1;
            if ( !v23 && v24 != v33 )
            {
              v18 = *(_QWORD *)(v24 + 32);
              v19 = v22->m128i_i64[0] < v18 || v22->m128i_i64[0] == v18 && v22->m128i_i32[2] < *(_DWORD *)(v24 + 40);
            }
            v34 = v19;
            v20 = (__m128i *)sub_22077B0(48);
            v20[2] = _mm_loadu_si128(v22);
            sub_220F040(v34, v20, v25, v33);
            ++*(_QWORD *)(a1 + 312);
          }
          v21 = *(_DWORD *)(a1 + 8);
          *(_DWORD *)(a1 + 8) = v21 - 1;
          if ( v21 == 1 )
            break;
          v8 = *(__m128i **)a1;
          v7 = (unsigned int)(v21 - 1);
        }
        v26 = sub_1F817F0(a1 + 272, (unsigned __int64 *)a2);
        v28 = v27;
        if ( v27 )
        {
          v29 = 1;
          if ( !v26 && v27 != a1 + 280 )
          {
            v32 = *(_QWORD *)(v27 + 32);
            v29 = a2->m128i_i64[0] < v32 || a2->m128i_i64[0] == v32 && a2->m128i_i32[2] < *(_DWORD *)(v27 + 40);
          }
          v30 = (__m128i *)sub_22077B0(48);
          v30[2] = _mm_loadu_si128(a2);
          sub_220F040(v29, v30, v28, a1 + 280);
          ++*(_QWORD *)(a1 + 312);
        }
        return 0x100000001LL;
      }
      else
      {
        if ( *(_DWORD *)(a1 + 8) >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, (int)v8, a6);
          v9 = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v9 = _mm_loadu_si128(a2);
        ++*(_DWORD *)(a1 + 8);
        return 0x100000001LL;
      }
    }
    else
    {
      return 1;
    }
  }
}
