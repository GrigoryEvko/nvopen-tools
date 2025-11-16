// Function: sub_D0AF90
// Address: 0xd0af90
//
__int64 __fastcall sub_D0AF90(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r15
  bool v11; // zf
  __m128i *v12; // r14
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __m128i *v16; // rbx
  __int64 **v17; // rdx
  __m128i *v18; // rax
  __m128i *v19; // rbx
  const __m128i *v20; // rcx
  const __m128i *v21; // rax
  __m128i *v22; // r14
  __m128i v23; // xmm3
  __int64 result; // rax
  __int64 v25; // rax
  __m128i *v26; // rbx
  __m128i *v27; // rax
  __int64 v28; // rax
  __int64 **v29; // [rsp+8h] [rbp-188h]
  __m128i *v30; // [rsp+18h] [rbp-178h] BYREF
  _QWORD v31[46]; // [rsp+20h] [rbp-170h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8);
  if ( a2 <= 8 )
  {
    if ( (v4 & 1) == 0 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) = v4 | 1;
      goto LABEL_8;
    }
LABEL_25:
    v19 = (__m128i *)(a1 + 16);
    v20 = (const __m128i *)(a1 + 336);
LABEL_26:
    v21 = v19;
    v22 = (__m128i *)v31;
    while ( 1 )
    {
      if ( v21->m128i_i64[0] == -4 )
      {
        if ( v21->m128i_i64[1] != -3 || v21[1].m128i_i64[0] != -4 || v21[1].m128i_i64[1] != -3 )
          goto LABEL_29;
      }
      else if ( v21->m128i_i64[0] != -16
             || v21->m128i_i64[1] != -4
             || v21[1].m128i_i64[0] != -16
             || v21[1].m128i_i64[1] != -4 )
      {
LABEL_29:
        if ( v22 )
        {
          v23 = _mm_loadu_si128(v21 + 1);
          *v22 = _mm_loadu_si128(v21);
          v22[1] = v23;
        }
        v22 = (__m128i *)((char *)v22 + 40);
        v22[-1].m128i_i64[1] = v21[2].m128i_i64[0];
      }
      v21 = (const __m128i *)((char *)v21 + 40);
      if ( v21 == v20 )
      {
        if ( v2 > 8 )
        {
          *(_BYTE *)(a1 + 8) &= ~1u;
          v28 = sub_C7D670(40LL * v2, 8);
          *(_DWORD *)(a1 + 24) = v2;
          *(_QWORD *)(a1 + 16) = v28;
        }
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v25 = 40;
        if ( v11 )
        {
          v19 = *(__m128i **)(a1 + 16);
          v25 = 5LL * *(unsigned int *)(a1 + 24);
        }
        for ( result = (__int64)&v19->m128i_i64[v25]; (__m128i *)result != v19; v19 = (__m128i *)((char *)v19 + 40) )
        {
          if ( v19 )
          {
            v19->m128i_i64[0] = -4;
            v19->m128i_i64[1] = -3;
            v19[1].m128i_i64[0] = -4;
            v19[1].m128i_i64[1] = -3;
          }
        }
        if ( v22 == (__m128i *)v31 )
          return result;
        v26 = (__m128i *)v31;
        while ( 2 )
        {
          result = v26->m128i_i64[0];
          if ( v26->m128i_i64[0] == -4 )
          {
            if ( v26->m128i_i64[1] == -3 && v26[1].m128i_i64[0] == -4 && v26[1].m128i_i64[1] == -3 )
              goto LABEL_53;
          }
          else if ( result == -16 && v26->m128i_i64[1] == -4 && v26[1].m128i_i64[0] == -16 && v26[1].m128i_i64[1] == -4 )
          {
LABEL_53:
            v26 = (__m128i *)((char *)v26 + 40);
            if ( v22 == v26 )
              return result;
            continue;
          }
          break;
        }
        sub_D0A160(a1, (unsigned __int64 *)v26, (__int64 **)&v30);
        v27 = v30;
        *v30 = _mm_loadu_si128(v26);
        v27[1] = _mm_loadu_si128(v26 + 1);
        v30[2].m128i_i64[0] = v26[2].m128i_i64[0];
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
  v2 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      v8 = 40LL * (unsigned int)v5;
      goto LABEL_5;
    }
    goto LABEL_25;
  }
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v19 = (__m128i *)(a1 + 16);
    v20 = (const __m128i *)(a1 + 336);
    v2 = 64;
    goto LABEL_26;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v2 = 64;
  v8 = 2560;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v10 = 40LL * v7;
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v12 = (__m128i *)(v6 + v10);
  if ( v11 )
  {
    v13 = *(_QWORD **)(a1 + 16);
    v14 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v13 = (_QWORD *)(a1 + 16);
    v14 = 40;
  }
  for ( i = &v13[v14]; i != v13; v13 += 5 )
  {
    if ( v13 )
    {
      *v13 = -4;
      v13[1] = -3;
      v13[2] = -4;
      v13[3] = -3;
    }
  }
  if ( v12 != (__m128i *)v6 )
  {
    v16 = (__m128i *)v6;
    v17 = (__int64 **)v31;
    do
    {
      if ( v16->m128i_i64[0] == -4 )
      {
        if ( v16->m128i_i64[1] == -3 && v16[1].m128i_i64[0] == -4 && v16[1].m128i_i64[1] == -3 )
          goto LABEL_19;
      }
      else if ( v16->m128i_i64[0] == -16
             && v16->m128i_i64[1] == -4
             && v16[1].m128i_i64[0] == -16
             && v16[1].m128i_i64[1] == -4 )
      {
        goto LABEL_19;
      }
      v29 = v17;
      sub_D0A160(a1, (unsigned __int64 *)v16, v17);
      v18 = (__m128i *)v31[0];
      v17 = v29;
      *(__m128i *)v31[0] = _mm_loadu_si128(v16);
      v18[1] = _mm_loadu_si128(v16 + 1);
      *(_QWORD *)(v31[0] + 32LL) = v16[2].m128i_i64[0];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_19:
      v16 = (__m128i *)((char *)v16 + 40);
    }
    while ( v12 != v16 );
  }
  return sub_C7D6A0(v6, v10, 8);
}
