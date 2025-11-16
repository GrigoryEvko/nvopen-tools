// Function: sub_277C980
// Address: 0x277c980
//
__int64 __fastcall sub_277C980(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r14
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __m128i *v10; // r13
  __int64 i; // rdx
  __m128i *v12; // r15
  int v13; // ebx
  int v14; // eax
  __m128i *v15; // rbx
  char v16; // al
  __m128i v17; // xmm0
  __int64 j; // rdx
  int v19; // [rsp+Ch] [rbp-54h]
  __m128i *v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  int v22; // [rsp+20h] [rbp-40h]
  unsigned int v23; // [rsp+24h] [rbp-3Ch]
  __int64 v24; // [rsp+28h] [rbp-38h]
  __int64 v25; // [rsp+28h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
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
  result = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 32LL * v4;
    v10 = (__m128i *)(v5 + v9);
    for ( i = result + 32 * v8; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_BYTE *)(result + 16) = 0;
      }
    }
    if ( v10 != (__m128i *)v5 )
    {
      v12 = (__m128i *)v5;
      do
      {
        while ( v12->m128i_i64[0] == -4096 || v12->m128i_i64[0] == -8192 )
        {
          v12 += 2;
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        v24 = v9;
        if ( !v13 )
        {
          MEMORY[0] = _mm_loadu_si128(v12);
          MEMORY[0x10] = v12[1].m128i_i64[0];
          BUG();
        }
        v21 = *(_QWORD *)(a1 + 8);
        v14 = sub_277C800(v12->m128i_i64);
        v22 = 1;
        v9 = v24;
        v19 = v13 - 1;
        v23 = (v13 - 1) & v14;
        v20 = 0;
        while ( 1 )
        {
          v25 = v9;
          v15 = (__m128i *)(v21 + 32LL * v23);
          v16 = sub_27781D0((__int64)v12, (__int64)v15);
          v9 = v25;
          if ( v16 )
            break;
          if ( v15->m128i_i64[0] == -4096 )
          {
            if ( v20 )
              v15 = v20;
            break;
          }
          if ( !v20 )
          {
            if ( v15->m128i_i64[0] != -8192 )
              v15 = 0;
            v20 = v15;
          }
          v23 = v19 & (v22 + v23);
          ++v22;
        }
        v17 = _mm_loadu_si128(v12);
        v12 += 2;
        *v15 = v17;
        v15[1].m128i_i64[0] = v12[-1].m128i_i64[0];
        v15[1].m128i_i64[1] = v12[-1].m128i_i64[1];
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 32LL * *(unsigned int *)(a1 + 24); j != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_BYTE *)(result + 16) = 0;
      }
    }
  }
  return result;
}
