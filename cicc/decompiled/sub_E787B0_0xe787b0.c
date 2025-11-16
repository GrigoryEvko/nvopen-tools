// Function: sub_E787B0
// Address: 0xe787b0
//
void __fastcall sub_E787B0(__int64 a1, _QWORD *a2, __int64 *a3, unsigned __int64 *a4)
{
  __int64 v5; // r14
  __int64 v7; // rsi
  int v8; // eax
  __int64 v9; // r8
  int v11; // edx
  unsigned int v12; // eax
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // r15
  int v16; // eax
  __int64 v17; // rsi
  unsigned int v18; // r13d
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r12
  __int64 v23; // rdi
  __m128i *v24; // rsi
  __int64 v25; // rax
  int v26; // r10d
  __int64 v27; // [rsp+0h] [rbp-60h]
  unsigned int v28; // [rsp+Ch] [rbp-54h]
  __m128i v29; // [rsp+10h] [rbp-50h] BYREF
  __m128i v30; // [rsp+20h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 2) != 0 )
    return;
  v5 = a2[1];
  v7 = *(_QWORD *)(a2[36] + 8LL);
  v8 = *(_DWORD *)(v5 + 1824);
  v9 = *(_QWORD *)(v5 + 1808);
  if ( !v8 )
    return;
  v11 = v8 - 1;
  v12 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v13 = *(_QWORD *)(v9 + 8LL * v12);
  if ( v7 == v13 )
  {
LABEL_4:
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v25 = *(_QWORD *)(a1 - 8);
      v15 = *(_QWORD *)v25;
      v14 = v25 + 24;
      if ( *(_QWORD *)v25 )
      {
        if ( *(_BYTE *)(v25 + 24) == 95 )
        {
          --v15;
          v14 = v25 + 25;
        }
        goto LABEL_7;
      }
    }
    else
    {
      v14 = 0;
    }
    v15 = 0;
LABEL_7:
    v27 = v14;
    v28 = *(_DWORD *)(v5 + 1796);
    v16 = sub_C8ED90(a3, *a4);
    v17 = *a4;
    v18 = sub_C90410(a3, *a4, v16);
    v22 = sub_E6C430(v5, v17, v19, v20, v21);
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v22, 0);
    v23 = a2[1];
    v29.m128i_i64[1] = v15;
    v29.m128i_i64[0] = v27;
    v24 = *(__m128i **)(v23 + 1856);
    v30.m128i_i64[0] = __PAIR64__(v18, v28);
    v30.m128i_i64[1] = v22;
    if ( v24 == *(__m128i **)(v23 + 1864) )
    {
      sub_E78620((const __m128i **)(v23 + 1848), v24, &v29);
    }
    else
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(&v29);
        v24[1] = _mm_loadu_si128(&v30);
        v24 = *(__m128i **)(v23 + 1856);
      }
      *(_QWORD *)(v23 + 1856) = v24 + 2;
    }
    return;
  }
  v26 = 1;
  while ( v13 != -4096 )
  {
    v12 = v11 & (v26 + v12);
    v13 = *(_QWORD *)(v9 + 8LL * v12);
    if ( v7 == v13 )
      goto LABEL_4;
    ++v26;
  }
}
