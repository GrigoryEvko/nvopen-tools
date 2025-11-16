// Function: sub_2FCC2A0
// Address: 0x2fcc2a0
//
_BYTE *__fastcall sub_2FCC2A0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  __int64 v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rax
  __int64 v17; // r10
  int v18; // edi
  int v19; // r15d
  __int64 *v20; // r14
  unsigned int v21; // esi
  __int64 *v22; // rcx
  __int64 v23; // r11
  __int64 v24; // rdx
  int v25; // edi
  _BYTE *result; // rax
  _QWORD *v27; // r14
  _QWORD *v28; // rcx
  _QWORD *v29; // rax
  _QWORD *v30; // r12
  __int64 v31; // rdx
  __m128i v32; // xmm0
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *k; // rdx
  _QWORD *v37; // r8
  int v38; // edi
  int v39; // r13d
  __int64 *v40; // r10
  unsigned int v41; // esi
  __int64 *v42; // rcx
  __int64 v43; // r9
  __int64 v44; // rdx
  int v45; // edi
  _BYTE v46[560]; // [rsp+10h] [rbp-230h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x10 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v27 = (_QWORD *)(a1 + 16);
    v28 = (_QWORD *)(a1 + 528);
  }
  else
  {
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
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v27 = (_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(a1 + 528);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 32LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 2048;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 32LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v4 + v10;
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 4LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 64;
        }
        for ( i = &v13[v14]; i != v13; v13 += 4 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = v4; v12 != j; j += 32 )
        {
          v24 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v24 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = a1 + 16;
              v18 = 15;
            }
            else
            {
              v25 = *(_DWORD *)(a1 + 24);
              v17 = *(_QWORD *)(a1 + 16);
              if ( !v25 )
                goto LABEL_77;
              v18 = v25 - 1;
            }
            v19 = 1;
            v20 = 0;
            v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v22 = (__int64 *)(v17 + 32LL * v21);
            v23 = *v22;
            if ( *v22 != v24 )
            {
              while ( v23 != -4096 )
              {
                if ( v23 == -8192 && !v20 )
                  v20 = v22;
                v21 = v18 & (v19 + v21);
                v22 = (__int64 *)(v17 + 32LL * v21);
                v23 = *v22;
                if ( v24 == *v22 )
                  goto LABEL_16;
                ++v19;
              }
              if ( v20 )
                v22 = v20;
            }
LABEL_16:
            *v22 = v24;
            *(__m128i *)(v22 + 1) = _mm_loadu_si128((const __m128i *)(j + 8));
            v22[3] = *(_QWORD *)(j + 24);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return (_BYTE *)sub_C7D6A0(v4, v10, 8);
      }
      v27 = (_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(a1 + 528);
      v2 = 64;
    }
  }
  v29 = v27;
  v30 = v46;
  do
  {
    v31 = *v29;
    if ( *v29 != -4096 && v31 != -8192 )
    {
      if ( v30 )
        *v30 = v31;
      v32 = _mm_loadu_si128((const __m128i *)(v29 + 1));
      v30 += 4;
      *(v30 - 1) = v29[3];
      *(__m128i *)(v30 - 3) = v32;
    }
    v29 += 4;
  }
  while ( v29 != v28 );
  if ( v2 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(32LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(_QWORD **)(a1 + 16);
    v35 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v27;
    v35 = 64;
  }
  for ( k = &v34[v35]; k != v34; v34 += 4 )
  {
    if ( v34 )
      *v34 = -4096;
  }
  for ( result = v46; v30 != (_QWORD *)result; result += 32 )
  {
    v24 = *(_QWORD *)result;
    if ( *(_QWORD *)result != -4096 && v24 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v37 = v27;
        v38 = 15;
      }
      else
      {
        v45 = *(_DWORD *)(a1 + 24);
        v37 = *(_QWORD **)(a1 + 16);
        if ( !v45 )
        {
LABEL_77:
          MEMORY[0] = v24;
          BUG();
        }
        v38 = v45 - 1;
      }
      v39 = 1;
      v40 = 0;
      v41 = v38 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v42 = &v37[4 * v41];
      v43 = *v42;
      if ( v24 != *v42 )
      {
        while ( v43 != -4096 )
        {
          if ( v43 == -8192 && !v40 )
            v40 = v42;
          v41 = v38 & (v39 + v41);
          v42 = &v37[4 * v41];
          v43 = *v42;
          if ( v24 == *v42 )
            goto LABEL_47;
          ++v39;
        }
        if ( v40 )
          v42 = v40;
      }
LABEL_47:
      *v42 = v24;
      v44 = *((_QWORD *)result + 3);
      *(__m128i *)(v42 + 1) = _mm_loadu_si128((const __m128i *)(result + 8));
      v42[3] = v44;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
