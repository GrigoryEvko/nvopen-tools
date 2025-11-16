// Function: sub_B07440
// Address: 0xb07440
//
_QWORD *__fastcall sub_B07440(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  const __m128i **v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  const __m128i **v9; // r13
  _QWORD *i; // rdx
  const __m128i **j; // rbx
  const __m128i *v12; // rax
  int v13; // r15d
  unsigned __int8 v14; // dl
  __int64 v15; // r10
  const __m128i *v16; // rcx
  __int64 *v17; // rsi
  unsigned __int8 v18; // dl
  __int64 v19; // rcx
  __m128i v20; // xmm0
  int v21; // eax
  int v22; // r15d
  int v23; // eax
  const __m128i *v24; // rcx
  unsigned int v25; // eax
  const __m128i **v26; // rdx
  const __m128i *v27; // rsi
  int v28; // r8d
  const __m128i **v29; // rdi
  __int64 v30; // rdx
  _QWORD *k; // rdx
  __int64 v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  int v34; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 v35; // [rsp+18h] [rbp-68h] BYREF
  __int64 v36; // [rsp+20h] [rbp-60h] BYREF
  __int64 v37; // [rsp+28h] [rbp-58h] BYREF
  __m128i v38; // [rsp+30h] [rbp-50h]
  __int64 v39; // [rsp+40h] [rbp-40h]
  __int64 v40[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(const __m128i ***)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(8LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v32 = 8 * v4;
    v9 = &v5[v4];
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v5; v9 != j; ++j )
    {
      v12 = *j;
      if ( *j != (const __m128i *)-8192LL && v12 != (const __m128i *)-4096LL )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v14 = v12[-1].m128i_u8[0];
        v15 = *(_QWORD *)(a1 + 8);
        v16 = v12 - 1;
        if ( (v14 & 2) != 0 )
          v17 = (__int64 *)v12[-2].m128i_i64[0];
        else
          v17 = &v16->m128i_i64[-((v14 >> 2) & 0xF)];
        v36 = *v17;
        v18 = v12[-1].m128i_u8[0];
        if ( (v18 & 2) != 0 )
          v19 = v12[-2].m128i_i64[0];
        else
          v19 = (__int64)&v16->m128i_i64[-((v18 >> 2) & 0xF)];
        v20 = _mm_loadu_si128(v12 + 1);
        v37 = *(_QWORD *)(v19 + 8);
        v38 = v20;
        v39 = v12[2].m128i_i64[0];
        v40[0] = v12[2].m128i_i64[1];
        if ( (_BYTE)v39 )
        {
          v35 = v38.m128i_i64[1];
          v21 = v38.m128i_i32[0];
        }
        else
        {
          v35 = 0;
          v21 = 0;
        }
        v33 = v15;
        v22 = v13 - 1;
        v34 = v21;
        v23 = sub_AFAA60(&v36, &v37, &v34, &v35, v40);
        v24 = *j;
        v25 = v22 & v23;
        v26 = (const __m128i **)(v33 + 8LL * v25);
        v27 = *v26;
        if ( *j != *v26 )
        {
          v28 = 1;
          v29 = 0;
          while ( v27 != (const __m128i *)-4096LL )
          {
            if ( v29 || v27 != (const __m128i *)-8192LL )
              v26 = v29;
            v25 = v22 & (v28 + v25);
            v27 = *(const __m128i **)(v33 + 8LL * v25);
            if ( v24 == v27 )
            {
              v26 = (const __m128i **)(v33 + 8LL * v25);
              goto LABEL_27;
            }
            ++v28;
            v29 = v26;
            v26 = (const __m128i **)(v33 + 8LL * v25);
          }
          if ( v29 )
            v26 = v29;
        }
LABEL_27:
        *v26 = v24;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v32, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v30]; k != result; ++result )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
