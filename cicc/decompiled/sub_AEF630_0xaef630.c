// Function: sub_AEF630
// Address: 0xaef630
//
_QWORD *__fastcall sub_AEF630(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __m128i *v5; // r15
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __m128i *v10; // r13
  _QWORD *i; // rdx
  __m128i *v12; // rbx
  __m128i *v13; // r11
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // r9
  __m128i *v17; // r10
  int v18; // r8d
  int v19; // edx
  unsigned int j; // ecx
  __m128i *v21; // rdi
  bool v22; // al
  __m128i *v23; // rdi
  _QWORD *k; // rdx
  __m128i v25; // xmm1
  __m128i *v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+10h] [rbp-70h]
  __m128i *v28; // [rsp+18h] [rbp-68h]
  int v29; // [rsp+20h] [rbp-60h]
  unsigned int v30; // [rsp+24h] [rbp-5Ch]
  __int64 v31; // [rsp+28h] [rbp-58h]
  __m128i *v32; // [rsp+30h] [rbp-50h]
  int v33; // [rsp+30h] [rbp-50h]
  int v34; // [rsp+38h] [rbp-48h]
  __m128i *v35; // [rsp+38h] [rbp-48h]
  _QWORD v36[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__m128i **)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 16 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[(unsigned __int64)v9 / 0x10];
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      v13 = v5;
      while ( 1 )
      {
        while ( v12->m128i_i64[0] == -4096 )
        {
          if ( v12->m128i_i64[1] != -4096 )
            goto LABEL_13;
LABEL_21:
          if ( v10 == ++v12 )
            goto LABEL_22;
        }
        if ( v12->m128i_i64[0] == -8192 && v12->m128i_i64[1] == -8192 )
          goto LABEL_21;
LABEL_13:
        v32 = v13;
        v34 = *(_DWORD *)(a1 + 24);
        if ( !v34 )
        {
          MEMORY[0] = _mm_loadu_si128(v12);
          BUG();
        }
        v14 = *(_QWORD *)(a1 + 8);
        v36[0] = -8192;
        v36[1] = -8192;
        v31 = v14;
        v15 = sub_AEA4A0(v12->m128i_i64, &v12->m128i_i64[1]);
        v16 = v12->m128i_i64[0];
        v17 = 0;
        v13 = v32;
        v18 = 1;
        v19 = v34 - 1;
        for ( j = (v34 - 1) & v15; ; j = v33 & (v29 + v30) )
        {
          v21 = (__m128i *)(v31 + 16LL * j);
          if ( v21->m128i_i64[0] == v16 && v12->m128i_i64[1] == v21->m128i_i64[1] )
            break;
          if ( v21->m128i_i64[0] == -4096 && v21->m128i_i64[1] == -4096 )
          {
            if ( v17 )
              v21 = v17;
            break;
          }
          v27 = v16;
          v29 = v18;
          v28 = v17;
          v30 = j;
          v33 = v19;
          v35 = v13;
          v26 = (__m128i *)(v31 + 16LL * j);
          v22 = sub_AE74A0(v21, v36);
          v13 = v35;
          v19 = v33;
          v16 = v27;
          if ( v28 || (v23 = v26, !v22) )
            v23 = v28;
          v17 = v23;
          v18 = v29 + 1;
        }
        v25 = _mm_loadu_si128(v12++);
        *v21 = v25;
        ++*(_DWORD *)(a1 + 16);
        if ( v10 == v12 )
        {
LABEL_22:
          v5 = v13;
          return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * *(unsigned int *)(a1 + 24)]; k != result; result += 2 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
  }
  return result;
}
