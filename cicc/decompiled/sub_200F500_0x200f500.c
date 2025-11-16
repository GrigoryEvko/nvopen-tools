// Function: sub_200F500
// Address: 0x200f500
//
__int64 __fastcall sub_200F500(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  unsigned __int64 v5; // rax
  int v6; // r15d
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rax
  __m128i *v10; // r13
  __m128i v11; // xmm1
  _DWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  _DWORD *v15; // rdx
  _DWORD *v16; // rdi
  int v17; // edx
  __int64 v18; // r9
  int v19; // edi
  int v20; // r11d
  int *v21; // r10
  unsigned int v22; // esi
  int *v23; // rcx
  int v24; // r8d
  __m128i v25; // xmm2
  int *v26; // r12
  unsigned int v27; // r13d
  bool v28; // zf
  int *v29; // rcx
  _DWORD *v30; // rax
  __int64 v31; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  unsigned int v34; // edx
  __int64 v35; // r10
  int v36; // r8d
  int v37; // r14d
  unsigned int *v38; // r13
  unsigned int v39; // edi
  unsigned int *v40; // rsi
  unsigned int v41; // r11d
  __m128i v42; // xmm0
  int v43; // esi
  __int64 v44; // rax
  int v45; // ecx
  _BYTE v46[240]; // [rsp+10h] [rbp-F0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v26 = *(int **)(a1 + 16);
    v27 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
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
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v7 = 6LL * (unsigned int)v5;
      if ( v4 )
      {
LABEL_5:
        v8 = a1 + 208;
        v9 = a1 + 16;
        v10 = (__m128i *)v46;
        do
        {
          while ( *(_DWORD *)v9 > 0xFFFFFFFD )
          {
            v9 += 24;
            if ( v8 == v9 )
              goto LABEL_11;
          }
          if ( v10 )
            v10->m128i_i32[0] = *(_DWORD *)v9;
          v11 = _mm_loadu_si128((const __m128i *)(v9 + 8));
          v9 += 24;
          v10 = (__m128i *)((char *)v10 + 24);
          v10[-1] = v11;
        }
        while ( v8 != v9 );
LABEL_11:
        *(_BYTE *)(a1 + 8) &= ~1u;
        v12 = (_DWORD *)sub_22077B0(v7 * 4);
        v13 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 16) = v12;
        v14 = v13 & 1;
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 8) = v14;
        if ( (_BYTE)v14 )
        {
          v12 = (_DWORD *)(a1 + 16);
          v7 = 48;
        }
        v15 = v12;
        v16 = &v12[v7];
        while ( 1 )
        {
          if ( v15 )
            *v12 = -1;
          v12 += 6;
          if ( v16 == v12 )
            break;
          v15 = v12;
        }
        for ( result = (__int64)v46;
              v10 != (__m128i *)result;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_DWORD *)result;
            if ( *(_DWORD *)result <= 0xFFFFFFFD )
              break;
            result += 24;
            if ( v10 == (__m128i *)result )
              return result;
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 7;
          }
          else
          {
            v45 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v45 )
              goto LABEL_71;
            v19 = v45 - 1;
          }
          v20 = 1;
          v21 = 0;
          v22 = v19 & (37 * v17);
          v23 = (int *)(v18 + 24LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (int *)(v18 + 24LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_25;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_25:
          v25 = _mm_loadu_si128((const __m128i *)(result + 8));
          *v23 = v17;
          result += 24;
          *(__m128i *)(v23 + 2) = v25;
        }
        return result;
      }
      v26 = *(int **)(a1 + 16);
      v27 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 384;
        v6 = 64;
        goto LABEL_5;
      }
      v26 = *(int **)(a1 + 16);
      v27 = *(_DWORD *)(a1 + 24);
      v6 = 64;
      v7 = 384;
    }
    v44 = sub_22077B0(v7 * 4);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v44;
  }
  v28 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v29 = &v26[6 * v27];
  if ( v28 )
  {
    v30 = *(_DWORD **)(a1 + 16);
    v31 = 6LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v30 = (_DWORD *)(a1 + 16);
    v31 = 48;
  }
  for ( i = &v30[v31]; i != v30; v30 += 6 )
  {
    if ( v30 )
      *v30 = -1;
  }
  for ( j = v26; v29 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v34 = *j;
      if ( (unsigned int)*j <= 0xFFFFFFFD )
        break;
      j += 6;
      if ( v29 == j )
        return j___libc_free_0(v26);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = a1 + 16;
      v36 = 7;
    }
    else
    {
      v43 = *(_DWORD *)(a1 + 24);
      v35 = *(_QWORD *)(a1 + 16);
      if ( !v43 )
      {
LABEL_71:
        MEMORY[0] = 0;
        BUG();
      }
      v36 = v43 - 1;
    }
    v37 = 1;
    v38 = 0;
    v39 = v36 & (37 * v34);
    v40 = (unsigned int *)(v35 + 24LL * v39);
    v41 = *v40;
    if ( v34 != *v40 )
    {
      while ( v41 != -1 )
      {
        if ( v41 == -2 && !v38 )
          v38 = v40;
        v39 = v36 & (v37 + v39);
        v40 = (unsigned int *)(v35 + 24LL * v39);
        v41 = *v40;
        if ( v34 == *v40 )
          goto LABEL_42;
        ++v37;
      }
      if ( v38 )
        v40 = v38;
    }
LABEL_42:
    *v40 = v34;
    v42 = _mm_loadu_si128((const __m128i *)(j + 2));
    j += 6;
    *(__m128i *)(v40 + 2) = v42;
  }
  return j___libc_free_0(v26);
}
