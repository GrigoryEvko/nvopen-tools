// Function: sub_2EDD470
// Address: 0x2edd470
//
__int64 __fastcall sub_2EDD470(__int64 a1, const __m128i *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r11d
  _QWORD *v7; // r10
  __int64 v8; // rdi
  __int64 v9; // r9
  __int64 result; // rax
  _QWORD *v11; // r8
  __int64 v12; // r13
  int v13; // eax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __m128i si128; // xmm0
  int v18; // edi
  __int64 v19; // rcx
  int v20; // edi
  int v21; // r11d
  __int64 v22; // rsi
  unsigned int i; // eax
  unsigned int v24; // eax
  int v25; // edi
  __int64 v26; // rcx
  int v27; // edi
  int v28; // r11d
  __int64 v29; // rsi
  unsigned int j; // eax
  unsigned int v31; // eax
  __m128i v32[3]; // [rsp+0h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_25:
    sub_2EDD1B0(a1, 2 * v4);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = a2->m128i_i64[1];
      v20 = v18 - 1;
      v21 = 1;
      v9 = 0;
      for ( i = v20
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; i = v20 & v24 )
      {
        v22 = *(_QWORD *)(a1 + 8);
        v7 = (_QWORD *)(v22 + 16LL * i);
        v11 = (_QWORD *)*v7;
        if ( *v7 == a2->m128i_i64[0] && v7[1] == v19 )
          break;
        if ( v11 == (_QWORD *)-4096LL )
        {
          if ( v7[1] == -4096 )
          {
LABEL_48:
            if ( v9 )
              v7 = (_QWORD *)v9;
            v15 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_17;
          }
        }
        else if ( v11 == (_QWORD *)-8192LL && v7[1] == -8192 && !v9 )
        {
          v9 = v22 + 16LL * i;
        }
        v24 = v21 + i;
        ++v21;
      }
      goto LABEL_44;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  v5 = a2->m128i_i64[1];
  v6 = 1;
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v4 - 1;
  for ( result = (unsigned int)v9
               & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                                 | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9)
                                                     ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
                ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; result = (unsigned int)v9 & v13 )
  {
    v11 = (_QWORD *)(v8 + 16LL * (unsigned int)result);
    v12 = *v11;
    if ( *v11 == a2->m128i_i64[0] && v11[1] == v5 )
      return result;
    if ( v12 == -4096 )
      break;
    if ( v12 == -8192 && v11[1] == -8192 && !v7 )
      v7 = (_QWORD *)(v8 + 16LL * (unsigned int)result);
LABEL_9:
    v13 = v6 + result;
    ++v6;
  }
  if ( v11[1] != -4096 )
    goto LABEL_9;
  v14 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v11;
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v4 )
    goto LABEL_25;
  if ( v4 - *(_DWORD *)(a1 + 20) - v15 > v4 >> 3 )
    goto LABEL_17;
  sub_2EDD1B0(a1, v4);
  v25 = *(_DWORD *)(a1 + 24);
  if ( !v25 )
    goto LABEL_53;
  v26 = a2->m128i_i64[1];
  v27 = v25 - 1;
  v28 = 1;
  v9 = 0;
  for ( j = v27
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)
              | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)))); ; j = v27 & v31 )
  {
    v29 = *(_QWORD *)(a1 + 8);
    v7 = (_QWORD *)(v29 + 16LL * j);
    v11 = (_QWORD *)*v7;
    if ( *v7 == a2->m128i_i64[0] && v7[1] == v26 )
      break;
    if ( v11 == (_QWORD *)-4096LL )
    {
      if ( v7[1] == -4096 )
        goto LABEL_48;
    }
    else if ( v11 == (_QWORD *)-8192LL && v7[1] == -8192 && !v9 )
    {
      v9 = v29 + 16LL * j;
    }
    v31 = v28 + j;
    ++v28;
  }
LABEL_44:
  v15 = *(_DWORD *)(a1 + 16) + 1;
LABEL_17:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v7 != -4096 || v7[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = a2->m128i_i64[0];
  v7[1] = a2->m128i_i64[1];
  v16 = *(unsigned int *)(a1 + 40);
  si128 = _mm_loadu_si128(a2);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v32[0] = si128;
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v16 + 1, 0x10u, (__int64)v11, v9);
    v16 = *(unsigned int *)(a1 + 40);
    si128 = _mm_load_si128(v32);
  }
  result = *(_QWORD *)(a1 + 32) + 16 * v16;
  *(__m128i *)result = si128;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
