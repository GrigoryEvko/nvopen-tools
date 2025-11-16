// Function: sub_2733BC0
// Address: 0x2733bc0
//
__int64 __fastcall sub_2733BC0(__int64 a1, const __m128i *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // r10
  unsigned __int8 v8; // di
  int v9; // r11d
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  int v13; // eax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __m128i si128; // xmm0
  int v18; // eax
  int v19; // edi
  unsigned __int8 v20; // cl
  __int64 v21; // rsi
  int v22; // r11d
  unsigned int i; // eax
  unsigned int v24; // eax
  int v25; // eax
  int v26; // edx
  unsigned __int8 v27; // si
  __int64 v28; // rdi
  unsigned int j; // eax
  int v30; // eax
  __m128i v31[3]; // [rsp+0h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = v4 - 1;
  v7 = 0;
  v8 = a2->m128i_u8[8];
  v9 = 1;
  for ( result = (unsigned int)v6 & (v8 ^ ((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4));
        ;
        result = (unsigned int)v6 & v13 )
  {
    v11 = v5 + 16LL * (unsigned int)result;
    v12 = *(_QWORD *)v11;
    if ( a2->m128i_i64[0] == *(_QWORD *)v11 && v8 == *(_BYTE *)(v11 + 8) )
      return result;
    if ( !v12 )
    {
      if ( !v7 )
        v7 = v5 + 16LL * (unsigned int)result;
      if ( !*(_BYTE *)(v11 + 8) )
        break;
    }
    v13 = v9 + result;
    ++v9;
  }
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v4 )
  {
LABEL_21:
    sub_27339A0(a1, 2 * v4);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = a2->m128i_u8[8];
      v5 = 0;
      v22 = 1;
      for ( i = (v18 - 1) & (v20 ^ ((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4));
            ;
            i = v19 & v24 )
      {
        v21 = *(_QWORD *)(a1 + 8);
        v7 = v21 + 16LL * i;
        v6 = *(_QWORD *)v7;
        if ( a2->m128i_i64[0] == *(_QWORD *)v7 && v20 == *(_BYTE *)(v7 + 8) )
          break;
        if ( !v6 )
        {
          if ( !*(_BYTE *)(v7 + 8) )
          {
            if ( v5 )
              v7 = v5;
            v15 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_13;
          }
          if ( !v5 )
            v5 = v21 + 16LL * i;
        }
        v24 = v22 + i;
        ++v22;
      }
      goto LABEL_32;
    }
LABEL_49:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v15 > v4 >> 3 )
    goto LABEL_13;
  sub_27339A0(a1, v4);
  v25 = *(_DWORD *)(a1 + 24);
  if ( !v25 )
    goto LABEL_49;
  v26 = v25 - 1;
  v27 = a2->m128i_u8[8];
  v5 = 1;
  for ( j = (v25 - 1) & (v27 ^ ((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4));
        ;
        j = v26 & v30 )
  {
    v28 = *(_QWORD *)(a1 + 8);
    v7 = v28 + 16LL * j;
    v6 = *(_QWORD *)v7;
    if ( a2->m128i_i64[0] == *(_QWORD *)v7 && v27 == *(_BYTE *)(v7 + 8) )
      break;
    if ( !v6 )
    {
      if ( !*(_BYTE *)(v7 + 8) )
      {
        if ( v12 )
          v7 = v12;
        v15 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_13;
      }
      if ( !v12 )
        v12 = v28 + 16LL * j;
    }
    v30 = v5 + j;
    v5 = (unsigned int)(v5 + 1);
  }
LABEL_32:
  v15 = *(_DWORD *)(a1 + 16) + 1;
LABEL_13:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *(_QWORD *)v7 || *(_BYTE *)(v7 + 8) )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v7 = a2->m128i_i64[0];
  *(_WORD *)(v7 + 8) = a2->m128i_i16[4];
  v16 = *(unsigned int *)(a1 + 40);
  si128 = _mm_loadu_si128(a2);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v31[0] = si128;
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v16 + 1, 0x10u, v6, v5);
    v16 = *(unsigned int *)(a1 + 40);
    si128 = _mm_load_si128(v31);
  }
  result = *(_QWORD *)(a1 + 32) + 16 * v16;
  *(__m128i *)result = si128;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
