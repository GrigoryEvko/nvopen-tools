// Function: sub_DB15D0
// Address: 0xdb15d0
//
__int64 __fastcall sub_DB15D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  unsigned int v5; // edi
  __int64 result; // rax
  __int64 v7; // r10
  __int64 v8; // rdx
  __int64 v9; // r11
  const __m128i *v10; // r15
  __int64 i; // rdx
  const __m128i *v12; // rbx
  __int64 v13; // r12
  int v14; // r14d
  __int64 v15; // rdx
  int v16; // r14d
  __int64 v17; // r9
  int v18; // ecx
  __int64 v19; // rdi
  bool v20; // al
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 j; // rdx
  __m128i v24; // xmm0
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  __int64 v27; // [rsp+18h] [rbp-88h]
  __int64 v28; // [rsp+20h] [rbp-80h]
  __int64 v29; // [rsp+28h] [rbp-78h]
  int v30; // [rsp+38h] [rbp-68h]
  int v31; // [rsp+3Ch] [rbp-64h]
  __int64 v32; // [rsp+40h] [rbp-60h]
  const __m128i *v33; // [rsp+48h] [rbp-58h]
  unsigned int v34; // [rsp+48h] [rbp-58h]
  _QWORD v35[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v36; // [rsp+60h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v33 = *(const __m128i **)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(32LL * v5, 8);
  v7 = (__int64)v33;
  *(_QWORD *)(a1 + 8) = result;
  if ( v33 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 32 * v4;
    v10 = &v33[2 * v4];
    for ( i = result + 32 * v8; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_QWORD *)(result + 8) = 0;
        *(_WORD *)(result + 16) = 0;
      }
    }
    if ( v10 != v33 )
    {
      v12 = v33;
      do
      {
        while ( 1 )
        {
          v13 = v12->m128i_i64[0];
          if ( v12->m128i_i64[0] || v12->m128i_i64[1] || v12[1].m128i_i16[0] > 1u )
            break;
          v12 += 2;
          if ( v10 == v12 )
            return sub_C7D6A0(v7, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = _mm_loadu_si128(v12);
          MEMORY[0x10] = v12[1].m128i_i16[0];
          BUG();
        }
        v15 = v12->m128i_i64[1];
        v16 = v14 - 1;
        v35[0] = 0;
        v36 = 1;
        v17 = *(_QWORD *)(a1 + 8);
        v35[1] = 0;
        v18 = v12[1].m128i_u16[0];
        v31 = 1;
        v32 = 0;
        v34 = v16
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned __int64)(unsigned __int16)v18 << 32)
                | (unsigned int)((0xBF58476D1CE4E5B9LL * ((unsigned int)v15 | (unsigned __int64)(v13 << 32))) >> 31)
                ^ (484763065 * (_DWORD)v15))) >> 31)
             ^ (484763065
              * (((0xBF58476D1CE4E5B9LL * ((unsigned int)v15 | (unsigned __int64)(v13 << 32))) >> 31) ^ (484763065 * v15))));
        while ( 1 )
        {
          v19 = v17 + 32LL * v34;
          if ( v13 == *(_QWORD *)v19 && v15 == *(_QWORD *)(v19 + 8) && (_WORD)v18 == *(_WORD *)(v19 + 16) )
            break;
          if ( !*(_QWORD *)v19 && !*(_QWORD *)(v19 + 8) && !*(_WORD *)(v19 + 16) )
          {
            if ( v32 )
              v19 = v32;
            break;
          }
          v30 = v18;
          v26 = v15;
          v27 = v17;
          v28 = v7;
          v29 = v9;
          v25 = v17 + 32LL * v34;
          v20 = sub_D95440(v19, (__int64)v35);
          v9 = v29;
          v7 = v28;
          v17 = v27;
          v15 = v26;
          v18 = v30;
          if ( v32 || (v21 = v25, !v20) )
            v21 = v32;
          v32 = v21;
          v34 = v16 & (v31 + v34);
          ++v31;
        }
        v24 = _mm_loadu_si128(v12);
        v12 += 2;
        *(__m128i *)v19 = v24;
        *(_WORD *)(v19 + 16) = v12[-1].m128i_i16[0];
        *(_QWORD *)(v19 + 24) = v12[-1].m128i_i64[1];
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return sub_C7D6A0(v7, v9, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 32 * v22; j != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_QWORD *)(result + 8) = 0;
        *(_WORD *)(result + 16) = 0;
      }
    }
  }
  return result;
}
