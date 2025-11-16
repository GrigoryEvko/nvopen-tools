// Function: sub_1AF5980
// Address: 0x1af5980
//
__int64 __fastcall sub_1AF5980(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r14
  unsigned __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // r15
  _QWORD *v9; // r13
  __int64 *v10; // rbx
  const __m128i *v11; // rax
  __int64 **v12; // r14
  __m128i v13; // xmm0
  unsigned int v14; // ebx
  bool v15; // zf
  __int64 *v16; // r13
  _QWORD *v17; // rax
  __int64 v18; // rdx
  _QWORD *i; // rdx
  __int64 *v20; // rbx
  __int64 *v21; // rax
  _QWORD *v22; // rdi
  __int64 *v23; // rax
  int v24; // [rsp+Ch] [rbp-104h]
  int v25; // [rsp+Ch] [rbp-104h]
  __int64 *v26; // [rsp+18h] [rbp-F8h] BYREF
  __int64 *v27[30]; // [rsp+20h] [rbp-F0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v14 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
    goto LABEL_18;
  }
  v5 = *(__int64 **)(a1 + 16);
  v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
      | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
      | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
      | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
      | (a2 - 1)
      | ((unsigned __int64)(a2 - 1) >> 1))
     + 1;
  v7 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v8 = 3LL * (unsigned int)v6;
    if ( v4 )
      goto LABEL_5;
    v14 = *(_DWORD *)(a1 + 24);
    goto LABEL_41;
  }
  if ( !v4 )
  {
    v14 = *(_DWORD *)(a1 + 24);
    v8 = 192;
    v7 = 64;
LABEL_41:
    v24 = v7;
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v24;
LABEL_18:
    v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v16 = &v5[3 * v14];
    if ( v15 )
    {
      v17 = *(_QWORD **)(a1 + 16);
      v18 = 3LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v17 = (_QWORD *)(a1 + 16);
      v18 = 24;
    }
    for ( i = &v17[v18]; i != v17; v17 += 3 )
    {
      if ( v17 )
      {
        *v17 = -8;
        v17[1] = -8;
        v17[2] = -8;
      }
    }
    if ( v16 == v5 )
      return j___libc_free_0(v5);
    v20 = v5;
    while ( *v20 == -8 )
    {
      if ( v20[1] == -8 && v20[2] == -8 )
      {
        v20 += 3;
        if ( v16 == v20 )
          return j___libc_free_0(v5);
      }
      else
      {
LABEL_27:
        sub_1AF57C0(a1, v20, v27);
        v21 = v27[0];
        *v27[0] = *v20;
        v21[1] = v20[1];
        v21[2] = v20[2];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_28:
        v20 += 3;
        if ( v16 == v20 )
          return j___libc_free_0(v5);
      }
    }
    if ( *v20 == -16 && v20[1] == -16 && v20[2] == -16 )
      goto LABEL_28;
    goto LABEL_27;
  }
  v8 = 192;
  v7 = 64;
LABEL_5:
  v9 = (_QWORD *)(a1 + 16);
  v10 = (__int64 *)v27;
  v11 = (const __m128i *)(a1 + 16);
  v12 = v27;
  do
  {
    if ( v11->m128i_i64[0] == -8 )
    {
      if ( v11->m128i_i64[1] == -8 && v11[1].m128i_i64[0] == -8 )
        goto LABEL_10;
    }
    else if ( v11->m128i_i64[0] == -16 && v11->m128i_i64[1] == -16 && v11[1].m128i_i64[0] == -16 )
    {
      goto LABEL_10;
    }
    if ( v12 )
    {
      v13 = _mm_loadu_si128(v11);
      v12[2] = (__int64 *)v11[1].m128i_i64[0];
      *(__m128i *)v12 = v13;
    }
    v12 += 3;
LABEL_10:
    v11 = (const __m128i *)((char *)v11 + 24);
  }
  while ( v11 != (const __m128i *)(a1 + 208) );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v25 = v7;
  result = sub_22077B0(v8 * 8);
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  *(_QWORD *)(a1 + 16) = result;
  *(_DWORD *)(a1 + 24) = v25;
  if ( v15 )
  {
    v9 = (_QWORD *)result;
  }
  else
  {
    result = a1 + 16;
    v8 = 24;
  }
  v22 = &v9[v8];
  while ( 1 )
  {
    if ( result )
    {
      *v9 = -8;
      v9[1] = -8;
      v9[2] = -8;
    }
    v9 += 3;
    if ( v22 == v9 )
      break;
    result = (__int64)v9;
  }
  if ( v12 != v27 )
  {
    do
    {
      result = *v10;
      if ( *v10 == -8 )
      {
        if ( v10[1] == -8 && v10[2] == -8 )
          goto LABEL_56;
      }
      else if ( result == -16 && v10[1] == -16 && v10[2] == -16 )
      {
        goto LABEL_56;
      }
      sub_1AF57C0(a1, v10, &v26);
      v23 = v26;
      *v26 = *v10;
      v23[1] = v10[1];
      v23[2] = v10[2];
      result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
      *(_DWORD *)(a1 + 8) = result;
LABEL_56:
      v10 += 3;
    }
    while ( v12 != (__int64 **)v10 );
  }
  return result;
}
