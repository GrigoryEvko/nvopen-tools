// Function: sub_2577460
// Address: 0x2577460
//
__int64 __fastcall sub_2577460(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r12
  char v5; // r14
  unsigned int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r11
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *i; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r9
  int v18; // edi
  __int64 v19; // r8
  int v20; // r14d
  __int64 v21; // r15
  unsigned int j; // eax
  __int64 v23; // rsi
  __int64 v24; // r10
  int v25; // edi
  __int64 result; // rax
  __m128i *v27; // r13
  const __m128i *v28; // rcx
  __int64 *v29; // r12
  const __m128i *v30; // rax
  __int64 *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rax
  unsigned int v35; // eax
  __int64 v36; // [rsp+0h] [rbp-110h]
  __int64 *v37; // [rsp+18h] [rbp-F8h] BYREF
  _BYTE v38[240]; // [rsp+20h] [rbp-F0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
LABEL_6:
      v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v36 = 24LL * v7;
      v11 = v4 + v36;
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
          *v12 = -4096;
          v12[1] = -4096;
        }
      }
      v15 = v4;
      if ( v11 == v4 )
        return sub_C7D6A0(v4, v36, 8);
      while ( 1 )
      {
        v16 = *(_QWORD *)v15;
        if ( *(_QWORD *)v15 == -4096 )
        {
          if ( *(_QWORD *)(v15 + 8) != -4096 )
            goto LABEL_15;
        }
        else if ( v16 != -8192 || *(_QWORD *)(v15 + 8) != -8192 )
        {
LABEL_15:
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v17 = a1 + 16;
            v18 = 7;
          }
          else
          {
            v25 = *(_DWORD *)(a1 + 24);
            v17 = *(_QWORD *)(a1 + 16);
            if ( !v25 )
            {
              MEMORY[0] = *(_QWORD *)v15;
              BUG();
            }
            v18 = v25 - 1;
          }
          v19 = *(_QWORD *)(v15 + 8);
          v20 = 1;
          v21 = 0;
          for ( j = v18
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)
                      | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)))); ; j = v18 & v35 )
          {
            v23 = v17 + 24LL * j;
            v24 = *(_QWORD *)v23;
            if ( v16 == *(_QWORD *)v23 && *(_QWORD *)(v23 + 8) == v19 )
              break;
            if ( v24 == -4096 )
            {
              if ( *(_QWORD *)(v23 + 8) == -4096 )
              {
                if ( v21 )
                  v23 = v21;
                break;
              }
            }
            else if ( v24 == -8192 && *(_QWORD *)(v23 + 8) == -8192 && !v21 )
            {
              v21 = v17 + 24LL * j;
            }
            v35 = v20 + j;
            ++v20;
          }
          *(_QWORD *)v23 = v16;
          *(_QWORD *)(v23 + 8) = *(_QWORD *)(v15 + 8);
          *(_DWORD *)(v23 + 16) = *(_DWORD *)(v15 + 16);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        }
        v15 += 24;
        if ( v11 == v15 )
          return sub_C7D6A0(v4, v36, 8);
      }
    }
    v27 = (__m128i *)(a1 + 16);
    v28 = (const __m128i *)(a1 + 208);
  }
  else
  {
    v6 = sub_AF1560(a2 - 1);
    v2 = v6;
    if ( v6 > 0x40 )
    {
      v27 = (__m128i *)(a1 + 16);
      v28 = (const __m128i *)(a1 + 208);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 24LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 1536;
        v2 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
        goto LABEL_6;
      }
      v27 = (__m128i *)(a1 + 16);
      v28 = (const __m128i *)(a1 + 208);
      v2 = 64;
    }
  }
  v29 = (__int64 *)v38;
  v30 = v27;
  v31 = (__int64 *)v38;
  do
  {
    while ( v30->m128i_i64[0] != -4096 )
    {
      if ( v30->m128i_i64[0] != -8192 || v30->m128i_i64[1] != -8192 )
      {
LABEL_40:
        if ( v31 )
          *(__m128i *)v31 = _mm_loadu_si128(v30);
        v31 += 3;
        *((_DWORD *)v31 - 2) = v30[1].m128i_i32[0];
      }
      v30 = (const __m128i *)((char *)v30 + 24);
      if ( v30 == v28 )
        goto LABEL_47;
    }
    if ( v30->m128i_i64[1] != -4096 )
      goto LABEL_40;
    v30 = (const __m128i *)((char *)v30 + 24);
  }
  while ( v30 != v28 );
LABEL_47:
  if ( v2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(24LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v33 = 24;
  if ( v10 )
  {
    v27 = *(__m128i **)(a1 + 16);
    v33 = 3LL * *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v27->m128i_i64[v33]; (__m128i *)result != v27; v27 = (__m128i *)((char *)v27 + 24) )
  {
    if ( v27 )
    {
      v27->m128i_i64[0] = -4096;
      v27->m128i_i64[1] = -4096;
    }
  }
  if ( v31 != (__int64 *)v38 )
  {
    while ( 1 )
    {
      result = *v29;
      if ( *v29 != -4096 )
        break;
      if ( v29[1] == -4096 )
      {
        v29 += 3;
        if ( v31 == v29 )
          return result;
      }
      else
      {
LABEL_58:
        sub_2568570(a1, v29, &v37);
        v34 = v37;
        *v37 = *v29;
        v34[1] = v29[1];
        *((_DWORD *)v37 + 4) = *((_DWORD *)v29 + 4);
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
LABEL_59:
        v29 += 3;
        if ( v31 == v29 )
          return result;
      }
    }
    if ( result == -8192 && v29[1] == -8192 )
      goto LABEL_59;
    goto LABEL_58;
  }
  return result;
}
