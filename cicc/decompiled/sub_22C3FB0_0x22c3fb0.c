// Function: sub_22C3FB0
// Address: 0x22c3fb0
//
__int64 __fastcall sub_22C3FB0(__int64 a1, __m128i *a2)
{
  __int64 v2; // r13
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r11d
  __int64 *v8; // r10
  __int64 v9; // rdi
  __int64 v10; // r9
  unsigned int i; // eax
  __int64 *v12; // r8
  __int64 v13; // r14
  unsigned int v14; // eax
  int v16; // eax
  int v17; // edx
  __int64 v18; // rax
  __m128i v19; // xmm0
  __m128i v20; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v21; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 208;
  v5 = *(_DWORD *)(a1 + 232);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 208);
    v21 = 0;
    goto LABEL_25;
  }
  v6 = a2->m128i_i64[1];
  v7 = 1;
  v8 = 0;
  v9 = *(_QWORD *)(a1 + 216);
  v10 = v5 - 1;
  for ( i = v10
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)
              | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))); ; i = v10 & v14 )
  {
    v12 = (__int64 *)(v9 + 16LL * i);
    v13 = *v12;
    if ( *v12 == a2->m128i_i64[0] && v12[1] == v6 )
      return 0;
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v8 )
      v8 = (__int64 *)(v9 + 16LL * i);
LABEL_9:
    v14 = v7 + i;
    ++v7;
  }
  if ( v12[1] != -4096 )
    goto LABEL_9;
  v16 = *(_DWORD *)(a1 + 224);
  if ( !v8 )
    v8 = v12;
  ++*(_QWORD *)(a1 + 208);
  v17 = v16 + 1;
  v21 = v8;
  if ( 4 * (v16 + 1) < 3 * v5 )
  {
    if ( v5 - *(_DWORD *)(a1 + 228) - v17 > v5 >> 3 )
      goto LABEL_17;
    goto LABEL_26;
  }
LABEL_25:
  v5 *= 2;
LABEL_26:
  sub_22C3CF0(v2, v5);
  sub_22C3A10(v2, a2->m128i_i64, &v21);
  v8 = v21;
  v17 = *(_DWORD *)(a1 + 224) + 1;
LABEL_17:
  *(_DWORD *)(a1 + 224) = v17;
  if ( *v8 != -4096 || v8[1] != -4096 )
    --*(_DWORD *)(a1 + 228);
  *v8 = a2->m128i_i64[0];
  v8[1] = a2->m128i_i64[1];
  v18 = *(unsigned int *)(a1 + 72);
  v19 = _mm_loadu_si128(a2);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
  {
    v20 = v19;
    sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v18 + 1, 0x10u, (__int64)v12, v10);
    v18 = *(unsigned int *)(a1 + 72);
    v19 = _mm_load_si128(&v20);
  }
  *(__m128i *)(*(_QWORD *)(a1 + 64) + 16 * v18) = v19;
  ++*(_DWORD *)(a1 + 72);
  return 1;
}
