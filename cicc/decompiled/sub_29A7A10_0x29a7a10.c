// Function: sub_29A7A10
// Address: 0x29a7a10
//
__int64 __fastcall sub_29A7A10(__int64 a1, __int64 a2, const __m128i *a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  int v10; // r13d
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r11
  char v14; // cl
  __int64 v15; // rsi
  int v16; // edi
  int v17; // ecx
  int v18; // edx
  __int64 v19; // r8
  int v20; // edi
  __int64 v21; // r9
  unsigned int v22; // esi
  int v23; // r10d
  int v24; // edx
  int v25; // edx
  __int64 v26; // r8
  int v27; // edi
  __int64 v28; // r9
  unsigned int v29; // esi
  int v30; // r10d
  int v31; // edx
  __int64 v32; // [rsp+8h] [rbp-28h]
  __int64 v33; // [rsp+8h] [rbp-28h]

  result = a1;
  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
    goto LABEL_14;
  }
  v8 = a3->m128i_i64[0];
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 1;
  v11 = (v6 - 1) & (((unsigned int)a3->m128i_i64[0] >> 9) ^ ((unsigned int)a3->m128i_i64[0] >> 4));
  v12 = (__int64 *)(v9 + 32LL * v11);
  v13 = *v12;
  if ( v8 == *v12 )
  {
LABEL_3:
    v14 = 0;
    v15 = v9 + 32 * v6;
    goto LABEL_4;
  }
  while ( v13 )
  {
    v11 = (v6 - 1) & (v10 + v11);
    v12 = (__int64 *)(v9 + 32LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
      goto LABEL_3;
    ++v10;
  }
  v16 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a2 = v7 + 1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= (unsigned int)(3 * v6) )
  {
LABEL_14:
    v32 = result;
    sub_29A7810(a2, 2 * v6);
    v18 = *(_DWORD *)(a2 + 24);
    if ( v18 )
    {
      v19 = a3->m128i_i64[0];
      v20 = v18 - 1;
      v21 = *(_QWORD *)(a2 + 8);
      v17 = *(_DWORD *)(a2 + 16) + 1;
      result = v32;
      v22 = (v18 - 1) & (((unsigned int)a3->m128i_i64[0] >> 9) ^ ((unsigned int)a3->m128i_i64[0] >> 4));
      v12 = (__int64 *)(v21 + 32LL * v22);
      if ( *v12 && v19 != *v12 )
      {
        v23 = 1;
        do
        {
          v24 = v23++;
          v22 = v20 & (v24 + v22);
          v12 = (__int64 *)(v21 + 32LL * v22);
        }
        while ( *v12 && v19 != *v12 );
      }
      goto LABEL_10;
    }
    goto LABEL_28;
  }
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v17 <= (unsigned int)v6 >> 3 )
  {
    v33 = result;
    sub_29A7810(a2, v6);
    v25 = *(_DWORD *)(a2 + 24);
    if ( v25 )
    {
      v26 = a3->m128i_i64[0];
      v27 = v25 - 1;
      v28 = *(_QWORD *)(a2 + 8);
      v17 = *(_DWORD *)(a2 + 16) + 1;
      result = v33;
      v29 = (v25 - 1) & (((unsigned int)a3->m128i_i64[0] >> 9) ^ ((unsigned int)a3->m128i_i64[0] >> 4));
      v12 = (__int64 *)(v28 + 32LL * v29);
      if ( *v12 && v26 != *v12 )
      {
        v30 = 1;
        do
        {
          v31 = v30++;
          v29 = v27 & (v31 + v29);
          v12 = (__int64 *)(v28 + 32LL * v29);
        }
        while ( *v12 && v26 != *v12 );
      }
      goto LABEL_10;
    }
LABEL_28:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a2 + 16) = v17;
  if ( *v12 )
    --*(_DWORD *)(a2 + 20);
  *(__m128i *)v12 = _mm_loadu_si128(a3);
  v12[2] = a3[1].m128i_i64[0];
  *((_DWORD *)v12 + 6) = a3[1].m128i_i32[2];
  v14 = 1;
  v7 = *(_QWORD *)a2;
  v15 = *(_QWORD *)(a2 + 8) + 32LL * *(unsigned int *)(a2 + 24);
LABEL_4:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v7;
  *(_QWORD *)(result + 16) = v12;
  *(_QWORD *)(result + 24) = v15;
  *(_BYTE *)(result + 32) = v14;
  return result;
}
