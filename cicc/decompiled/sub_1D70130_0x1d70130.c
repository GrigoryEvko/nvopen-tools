// Function: sub_1D70130
// Address: 0x1d70130
//
__int64 __fastcall sub_1D70130(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // r13
  unsigned int v5; // eax
  __int64 *v6; // r12
  int v7; // esi
  __int64 v8; // r15
  _QWORD *v9; // r13
  __int64 *v10; // r12
  const __m128i *v11; // rcx
  const __m128i *v12; // rax
  __m128i *v13; // r14
  bool v14; // zf
  _QWORD *v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 *v18; // r8
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *i; // rdx
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  int v25; // esi
  __int64 v26; // rdi
  __int64 *v27; // r14
  __int64 v28; // r11
  int v29; // r13d
  unsigned __int64 v30; // r11
  unsigned __int64 v31; // r11
  unsigned int j; // eax
  __int64 *v33; // r11
  __int64 v34; // r15
  int v35; // esi
  unsigned int v36; // eax
  __int64 *v37; // [rsp+18h] [rbp-B8h] BYREF
  _BYTE v38[176]; // [rsp+20h] [rbp-B0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v6 = *(__int64 **)(a1 + 16);
    v17 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
LABEL_32:
    v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v18 = &v6[2 * v17];
    if ( v14 )
    {
      v19 = *(_QWORD **)(a1 + 16);
      v20 = 2LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v19 = (_QWORD *)(a1 + 16);
      v20 = 16;
    }
    for ( i = &v19[v20]; i != v19; v19 += 2 )
    {
      if ( v19 )
      {
        *v19 = -8;
        v19[1] = -8;
      }
    }
    v22 = v6;
    if ( v18 == v6 )
      return j___libc_free_0(v6);
    while ( 1 )
    {
      v23 = *v22;
      if ( *v22 == -8 )
      {
        if ( v22[1] == -8 )
          goto LABEL_54;
      }
      else if ( v23 == -16 && v22[1] == -16 )
      {
        goto LABEL_54;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v24 = a1 + 16;
        v25 = 7;
      }
      else
      {
        v35 = *(_DWORD *)(a1 + 24);
        v24 = *(_QWORD *)(a1 + 16);
        if ( !v35 )
        {
          MEMORY[0] = *v22;
          BUG();
        }
        v25 = v35 - 1;
      }
      v26 = v22[1];
      v27 = 0;
      v28 = ((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4);
      v29 = 1;
      v30 = (((v28 | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32)) - 1 - (v28 << 32)) >> 22)
          ^ ((v28 | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32)) - 1 - (v28 << 32));
      v31 = ((9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13)))) >> 15)
          ^ (9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13))));
      for ( j = v25 & (((v31 - 1 - (v31 << 27)) >> 31) ^ (v31 - 1 - ((_DWORD)v31 << 27))); ; j = v25 & v36 )
      {
        v33 = (__int64 *)(v24 + 16LL * j);
        v34 = *v33;
        if ( v23 == *v33 && v33[1] == v26 )
          goto LABEL_59;
        if ( v34 == -8 )
          break;
        if ( v34 == -16 && v33[1] == -16 && !v27 )
          v27 = (__int64 *)(v24 + 16LL * j);
LABEL_66:
        v36 = v29 + j;
        ++v29;
      }
      if ( v33[1] != -8 )
        goto LABEL_66;
      if ( v27 )
        v33 = v27;
LABEL_59:
      *v33 = v23;
      v33[1] = v22[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_54:
      v22 += 2;
      if ( v18 == v22 )
        return j___libc_free_0(v6);
    }
  }
  v5 = sub_1454B60(a2 - 1);
  v6 = *(__int64 **)(a1 + 16);
  v7 = v5;
  if ( v5 > 0x40 )
  {
    v8 = 2LL * v5;
    if ( v4 )
      goto LABEL_5;
    v17 = *(unsigned int *)(a1 + 24);
    goto LABEL_64;
  }
  if ( !v4 )
  {
    v17 = *(unsigned int *)(a1 + 24);
    v8 = 128;
    v7 = 64;
LABEL_64:
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
    goto LABEL_32;
  }
  v8 = 128;
  v7 = 64;
LABEL_5:
  v9 = (_QWORD *)(a1 + 16);
  v10 = (__int64 *)v38;
  v11 = (const __m128i *)(a1 + 144);
  v12 = (const __m128i *)(a1 + 16);
  v13 = (__m128i *)v38;
  do
  {
    while ( v12->m128i_i64[0] != -8 )
    {
      if ( v12->m128i_i64[0] != -16 || v12->m128i_i64[1] != -16 )
      {
LABEL_7:
        if ( v13 )
          *v13 = _mm_loadu_si128(v12);
        ++v13;
      }
      if ( ++v12 == v11 )
        goto LABEL_14;
    }
    if ( v12->m128i_i64[1] != -8 )
      goto LABEL_7;
    ++v12;
  }
  while ( v12 != v11 );
LABEL_14:
  *(_BYTE *)(a1 + 8) &= ~1u;
  result = sub_22077B0(v8 * 8);
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  *(_QWORD *)(a1 + 16) = result;
  *(_DWORD *)(a1 + 24) = v7;
  if ( v14 )
  {
    v9 = (_QWORD *)result;
  }
  else
  {
    result = a1 + 16;
    v8 = 16;
  }
  v15 = &v9[v8];
  while ( 1 )
  {
    if ( result )
    {
      *v9 = -8;
      v9[1] = -8;
    }
    v9 += 2;
    if ( v15 == v9 )
      break;
    result = (__int64)v9;
  }
  if ( v13 != (__m128i *)v38 )
  {
    while ( 1 )
    {
      result = *v10;
      if ( *v10 != -8 )
        break;
      if ( v10[1] == -8 )
      {
        v10 += 2;
        if ( v13 == (__m128i *)v10 )
          return result;
      }
      else
      {
LABEL_24:
        sub_1D67E80(a1, v10, &v37);
        v16 = v37;
        *v37 = *v10;
        v16[1] = v10[1];
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
LABEL_25:
        v10 += 2;
        if ( v13 == (__m128i *)v10 )
          return result;
      }
    }
    if ( result == -16 && v10[1] == -16 )
      goto LABEL_25;
    goto LABEL_24;
  }
  return result;
}
