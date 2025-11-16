// Function: sub_2A0C960
// Address: 0x2a0c960
//
__int64 __fastcall sub_2A0C960(__int64 a1, unsigned int a2)
{
  unsigned int v3; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 *v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *i; // rdx
  __int64 *v15; // rbx
  __int64 *v16; // rax
  __int64 result; // rax
  __m128i *v18; // r12
  const __m128i *v19; // rsi
  __int64 *v20; // rbx
  const __m128i *v21; // rax
  __int64 **v22; // r13
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-108h]
  __int64 *v28; // [rsp+18h] [rbp-F8h] BYREF
  __int64 *v29[30]; // [rsp+20h] [rbp-F0h] BYREF

  v3 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
LABEL_25:
    v18 = (__m128i *)(a1 + 16);
    v19 = (const __m128i *)(a1 + 208);
LABEL_26:
    v20 = (__int64 *)v29;
    v21 = v18;
    v22 = v29;
    while ( 1 )
    {
      if ( v21->m128i_i64[0] == -1 )
      {
        if ( v21->m128i_i64[1] != -4096 || v21[1].m128i_i64[0] != -4096 )
          goto LABEL_29;
      }
      else if ( v21->m128i_i64[0] != -2 || v21->m128i_i64[1] != -8192 || v21[1].m128i_i64[0] != -8192 )
      {
LABEL_29:
        if ( v22 )
        {
          v23 = _mm_loadu_si128(v21);
          v22[2] = (__int64 *)v21[1].m128i_i64[0];
          *(__m128i *)v22 = v23;
        }
        v22 += 3;
      }
      v21 = (const __m128i *)((char *)v21 + 24);
      if ( v21 == v19 )
      {
        if ( v3 > 8 )
        {
          *(_BYTE *)(a1 + 8) &= ~1u;
          v26 = sub_C7D670(24LL * v3, 8);
          *(_DWORD *)(a1 + 24) = v3;
          *(_QWORD *)(a1 + 16) = v26;
        }
        v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v24 = 24;
        if ( v10 )
        {
          v18 = *(__m128i **)(a1 + 16);
          v24 = 3LL * *(unsigned int *)(a1 + 24);
        }
        for ( result = (__int64)&v18->m128i_i64[v24]; (__m128i *)result != v18; v18 = (__m128i *)((char *)v18 + 24) )
        {
          if ( v18 )
          {
            v18->m128i_i64[0] = -1;
            v18->m128i_i64[1] = -4096;
            v18[1].m128i_i64[0] = -4096;
          }
        }
        if ( v22 == v29 )
          return result;
        while ( 2 )
        {
          result = *v20;
          if ( *v20 == -1 )
          {
            if ( v20[1] == -4096 && v20[2] == -4096 )
              goto LABEL_53;
          }
          else if ( result == -2 && v20[1] == -8192 && v20[2] == -8192 )
          {
LABEL_53:
            v20 += 3;
            if ( v22 == (__int64 **)v20 )
              return result;
            continue;
          }
          break;
        }
        sub_2A0C820(a1, v20, &v28);
        v25 = v28;
        *v28 = *v20;
        v25[1] = v20[1];
        v25[2] = v20[2];
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
        goto LABEL_53;
      }
    }
  }
  v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v3 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      v8 = 24LL * (unsigned int)v5;
      goto LABEL_5;
    }
    goto LABEL_25;
  }
  if ( v4 )
  {
    v18 = (__m128i *)(a1 + 16);
    v19 = (const __m128i *)(a1 + 208);
    v3 = 64;
    goto LABEL_26;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v3 = 64;
  v8 = 1536;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v3;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v27 = 24LL * v7;
  v11 = (__int64 *)(v6 + v27);
  if ( v10 )
  {
    v12 = *(_QWORD **)(a1 + 16);
    v13 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v12 = (_QWORD *)(a1 + 16);
    v13 = 24;
  }
  for ( i = &v12[v13]; i != v12; v12 += 3 )
  {
    if ( v12 )
    {
      *v12 = -1;
      v12[1] = -4096;
      v12[2] = -4096;
    }
  }
  if ( v11 != (__int64 *)v6 )
  {
    v15 = (__int64 *)v6;
    while ( *v15 == -1 )
    {
      if ( v15[1] == -4096 && v15[2] == -4096 )
      {
        v15 += 3;
        if ( v11 == v15 )
          return sub_C7D6A0(v6, v27, 8);
      }
      else
      {
LABEL_18:
        sub_2A0C820(a1, v15, v29);
        v16 = v29[0];
        *v29[0] = *v15;
        v16[1] = v15[1];
        v16[2] = v15[2];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_19:
        v15 += 3;
        if ( v11 == v15 )
          return sub_C7D6A0(v6, v27, 8);
      }
    }
    if ( *v15 == -2 && v15[1] == -8192 && v15[2] == -8192 )
      goto LABEL_19;
    goto LABEL_18;
  }
  return sub_C7D6A0(v6, v27, 8);
}
