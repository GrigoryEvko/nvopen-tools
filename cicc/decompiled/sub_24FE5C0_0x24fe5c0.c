// Function: sub_24FE5C0
// Address: 0x24fe5c0
//
_BYTE *__fastcall sub_24FE5C0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r10
  bool v11; // zf
  __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rax
  __int64 v17; // rdx
  __int64 v18; // r11
  int v19; // esi
  unsigned int v20; // r9d
  __int64 *v21; // rcx
  __int64 v22; // r13
  __m128i v23; // xmm0
  _BYTE *result; // rax
  int v25; // esi
  _QWORD *v26; // r13
  _QWORD *v27; // rcx
  _QWORD *v28; // rax
  _QWORD *v29; // r12
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *k; // rdx
  __int64 v34; // rdx
  _QWORD *v35; // r9
  int v36; // esi
  unsigned int v37; // r8d
  __int64 *v38; // rcx
  __int64 v39; // r10
  __m128i v40; // xmm1
  __int64 v41; // rdx
  __int64 v42; // rdx
  int v43; // esi
  int v44; // r15d
  __int64 *v45; // r14
  int v46; // ecx
  int v47; // [rsp+4h] [rbp-BCh]
  __int64 *v48; // [rsp+8h] [rbp-B8h]
  _BYTE v49[176]; // [rsp+10h] [rbp-B0h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v26 = (_QWORD *)(a1 + 16);
    v27 = (_QWORD *)(a1 + 144);
  }
  else
  {
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
      v26 = (_QWORD *)(a1 + 16);
      v27 = (_QWORD *)(a1 + 144);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 32LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 2048;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 32LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v6 + v10;
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 4LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 16;
        }
        for ( i = &v13[v14]; i != v13; v13 += 4 )
        {
          if ( v13 )
            *v13 = 0x7FFFFFFFFFFFFFFFLL;
        }
        for ( j = v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_QWORD *)j;
            if ( *(__int64 *)j <= 0x7FFFFFFFFFFFFFFDLL )
              break;
            j += 32;
            if ( v12 == j )
              return (_BYTE *)sub_C7D6A0(v6, v10, 8);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 3;
          }
          else
          {
            v25 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v25 )
              goto LABEL_77;
            v19 = v25 - 1;
          }
          v20 = v19 & (37 * v17);
          v21 = (__int64 *)(v18 + 32LL * v20);
          v22 = *v21;
          if ( v17 != *v21 )
          {
            v47 = 1;
            v48 = 0;
            while ( v22 != 0x7FFFFFFFFFFFFFFFLL )
            {
              if ( v22 == 0x7FFFFFFFFFFFFFFELL )
              {
                if ( v48 )
                  v21 = v48;
                v48 = v21;
              }
              v20 = v19 & (v47 + v20);
              v21 = (__int64 *)(v18 + 32LL * v20);
              v22 = *v21;
              if ( v17 == *v21 )
                goto LABEL_21;
              ++v47;
            }
            if ( v48 )
              v21 = v48;
          }
LABEL_21:
          *v21 = v17;
          v23 = _mm_loadu_si128((const __m128i *)(j + 8));
          j += 32;
          *(__m128i *)(v21 + 1) = v23;
          v21[3] = *(_QWORD *)(j - 8);
        }
        return (_BYTE *)sub_C7D6A0(v6, v10, 8);
      }
      v26 = (_QWORD *)(a1 + 16);
      v27 = (_QWORD *)(a1 + 144);
      v2 = 64;
    }
  }
  v28 = v26;
  v29 = v49;
  do
  {
    if ( (__int64)*v28 <= 0x7FFFFFFFFFFFFFFDLL )
    {
      if ( v29 )
        *v29 = *v28;
      v42 = v28[3];
      v29 += 4;
      *(__m128i *)(v29 - 3) = _mm_loadu_si128((const __m128i *)(v28 + 1));
      *(v29 - 1) = v42;
    }
    v28 += 4;
  }
  while ( v28 != v27 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v30 = sub_C7D670(32LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v30;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v31 = *(_QWORD **)(a1 + 16);
    v32 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v31 = v26;
    v32 = 16;
  }
  for ( k = &v31[v32]; k != v31; v31 += 4 )
  {
    if ( v31 )
      *v31 = 0x7FFFFFFFFFFFFFFFLL;
  }
  for ( result = v49;
        v29 != (_QWORD *)result;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v34 = *(_QWORD *)result;
      if ( *(__int64 *)result <= 0x7FFFFFFFFFFFFFFDLL )
        break;
      result += 32;
      if ( v29 == (_QWORD *)result )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = v26;
      v36 = 3;
    }
    else
    {
      v43 = *(_DWORD *)(a1 + 24);
      v35 = *(_QWORD **)(a1 + 16);
      if ( !v43 )
      {
LABEL_77:
        MEMORY[0] = 0;
        BUG();
      }
      v36 = v43 - 1;
    }
    v37 = v36 & (37 * v34);
    v38 = &v35[4 * v37];
    v39 = *v38;
    if ( v34 != *v38 )
    {
      v44 = 1;
      v45 = 0;
      while ( v39 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v39 == 0x7FFFFFFFFFFFFFFELL && !v45 )
          v45 = v38;
        v46 = v44++;
        v37 = v36 & (v46 + v37);
        v38 = &v35[4 * v37];
        v39 = *v38;
        if ( v34 == *v38 )
          goto LABEL_45;
      }
      if ( v45 )
        v38 = v45;
    }
LABEL_45:
    *v38 = v34;
    v40 = _mm_loadu_si128((const __m128i *)(result + 8));
    result += 32;
    v41 = *((_QWORD *)result - 1);
    *(__m128i *)(v38 + 1) = v40;
    v38[3] = v41;
  }
  return result;
}
