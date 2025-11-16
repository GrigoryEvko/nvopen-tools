// Function: sub_13ED650
// Address: 0x13ed650
//
__int64 __fastcall sub_13ED650(__int64 a1, __m128i *a2)
{
  __int64 v2; // r13
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r11d
  __int64 *v8; // r10
  __int64 v9; // rdi
  __int64 v10; // r8
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r8
  unsigned int i; // eax
  __int64 *v14; // r8
  __int64 v15; // r14
  unsigned int v16; // eax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  int v21; // edi
  __int64 v22; // rcx
  int v23; // edi
  int v24; // r11d
  __int64 v25; // rsi
  __int64 v26; // r8
  __int64 *v27; // r9
  unsigned __int64 v28; // r8
  unsigned __int64 v29; // r8
  unsigned int j; // eax
  __int64 v31; // r8
  unsigned int v32; // eax
  __int64 *v33; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 240;
  v5 = *(_DWORD *)(a1 + 264);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 240);
    goto LABEL_25;
  }
  v6 = a2->m128i_i64[1];
  v7 = 1;
  v8 = 0;
  v9 = *(_QWORD *)(a1 + 248);
  v10 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
  v11 = (((v10
         | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
        - 1
        - (v10 << 32)) >> 22)
      ^ ((v10 | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
       - 1
       - (v10 << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  for ( i = (v5 - 1) & (((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - ((_DWORD)v12 << 27))); ; i = (v5 - 1) & v16 )
  {
    v14 = (__int64 *)(v9 + 16LL * i);
    v15 = *v14;
    if ( *v14 == a2->m128i_i64[0] && v14[1] == v6 )
      return 0;
    if ( v15 == -8 )
      break;
    if ( v15 == -16 && v14[1] == -16 && !v8 )
      v8 = (__int64 *)(v9 + 16LL * i);
LABEL_9:
    v16 = v7 + i;
    ++v7;
  }
  if ( v14[1] != -8 )
    goto LABEL_9;
  v18 = *(_DWORD *)(a1 + 256);
  if ( !v8 )
    v8 = v14;
  ++*(_QWORD *)(a1 + 240);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v5 )
  {
LABEL_25:
    sub_13ED3B0(v2, 2 * v5);
    v21 = *(_DWORD *)(a1 + 264);
    if ( !v21 )
    {
      ++*(_DWORD *)(a1 + 256);
      BUG();
    }
    v22 = a2->m128i_i64[1];
    v23 = v21 - 1;
    v24 = 1;
    v26 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
    v27 = 0;
    v28 = (((v26
           | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
          - 1
          - (v26 << 32)) >> 22)
        ^ ((v26
          | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
         - 1
         - (v26 << 32));
    v29 = ((9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13)))) >> 15)
        ^ (9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13))));
    for ( j = v23 & (((v29 - 1 - (v29 << 27)) >> 31) ^ (v29 - 1 - ((_DWORD)v29 << 27))); ; j = v23 & v32 )
    {
      v25 = *(_QWORD *)(a1 + 248);
      v8 = (__int64 *)(v25 + 16LL * j);
      v31 = *v8;
      if ( *v8 == a2->m128i_i64[0] && v8[1] == v22 )
        break;
      if ( v31 == -8 )
      {
        if ( v8[1] == -8 )
        {
          if ( v27 )
            v8 = v27;
          v19 = *(_DWORD *)(a1 + 256) + 1;
          goto LABEL_17;
        }
      }
      else if ( v31 == -16 && v8[1] == -16 && !v27 )
      {
        v27 = (__int64 *)(v25 + 16LL * j);
      }
      v32 = v24 + j;
      ++v24;
    }
    goto LABEL_35;
  }
  if ( v5 - *(_DWORD *)(a1 + 260) - v19 > v5 >> 3 )
    goto LABEL_17;
  sub_13ED3B0(v2, v5);
  sub_13EBF30(v2, a2->m128i_i64, &v33);
  v8 = v33;
LABEL_35:
  v19 = *(_DWORD *)(a1 + 256) + 1;
LABEL_17:
  *(_DWORD *)(a1 + 256) = v19;
  if ( *v8 != -8 || v8[1] != -8 )
    --*(_DWORD *)(a1 + 260);
  *v8 = a2->m128i_i64[0];
  v8[1] = a2->m128i_i64[1];
  v20 = *(unsigned int *)(a1 + 104);
  if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 108) )
  {
    sub_16CD150(a1 + 96, a1 + 112, 0, 16);
    v20 = *(unsigned int *)(a1 + 104);
  }
  *(__m128i *)(*(_QWORD *)(a1 + 96) + 16 * v20) = _mm_loadu_si128(a2);
  ++*(_DWORD *)(a1 + 104);
  return 1;
}
