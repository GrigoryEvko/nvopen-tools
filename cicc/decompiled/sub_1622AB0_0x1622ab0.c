// Function: sub_1622AB0
// Address: 0x1622ab0
//
__int64 __fastcall sub_1622AB0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r15
  __int64 v9; // rax
  __m128i *v10; // r12
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  _QWORD *v15; // rdx
  _QWORD *v16; // rdi
  __int64 v17; // r8
  int v18; // edi
  int v19; // r11d
  __int64 *v20; // r10
  unsigned int v21; // ecx
  __int64 *v22; // rsi
  __int64 v23; // r9
  __m128i v24; // xmm1
  __int64 v25; // rdx
  int v26; // ecx
  unsigned int v27; // r13d
  bool v28; // zf
  __int64 *v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v34; // r10
  int v35; // r9d
  int v36; // r14d
  __int64 *v37; // r13
  unsigned int v38; // ecx
  __int64 *v39; // rdi
  __int64 v40; // r11
  int v41; // ecx
  __int64 v42; // rax
  _BYTE v43[144]; // [rsp+10h] [rbp-90h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v27 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
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
      v27 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 192;
        v7 = 64;
LABEL_5:
        v9 = a1 + 16;
        v10 = (__m128i *)v43;
        do
        {
          v11 = *(_QWORD *)v9;
          if ( *(_QWORD *)v9 != -4 && v11 != -8 )
          {
            if ( v10 )
              v10->m128i_i64[0] = v11;
            v10 = (__m128i *)((char *)v10 + 24);
            v10[-1] = _mm_loadu_si128((const __m128i *)(v9 + 8));
          }
          v9 += 24;
        }
        while ( v9 != a1 + 112 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v12 = (_QWORD *)sub_22077B0(v8 * 8);
        v13 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v12;
        v14 = v13 & 1;
        *(_QWORD *)(a1 + 8) = v14;
        if ( (_BYTE)v14 )
        {
          v12 = (_QWORD *)(a1 + 16);
          v8 = 12;
        }
        v15 = v12;
        v16 = &v12[v8];
        while ( 1 )
        {
          if ( v15 )
            *v12 = -4;
          v12 += 3;
          if ( v16 == v12 )
            break;
          v15 = v12;
        }
        result = (__int64)v43;
        if ( v10 != (__m128i *)v43 )
        {
          while ( 1 )
          {
            v25 = *(_QWORD *)result;
            if ( *(_QWORD *)result != -4 && v25 != -8 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v17 = a1 + 16;
                v18 = 3;
              }
              else
              {
                v26 = *(_DWORD *)(a1 + 24);
                v17 = *(_QWORD *)(a1 + 16);
                if ( !v26 )
                  goto LABEL_74;
                v18 = v26 - 1;
              }
              v19 = 1;
              v20 = 0;
              v21 = v18 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v22 = (__int64 *)(v17 + 24LL * v21);
              v23 = *v22;
              if ( v25 != *v22 )
              {
                while ( v23 != -4 )
                {
                  if ( v23 == -8 && !v20 )
                    v20 = v22;
                  v21 = v18 & (v19 + v21);
                  v22 = (__int64 *)(v17 + 24LL * v21);
                  v23 = *v22;
                  if ( v25 == *v22 )
                    goto LABEL_23;
                  ++v19;
                }
                if ( v20 )
                  v22 = v20;
              }
LABEL_23:
              v24 = _mm_loadu_si128((const __m128i *)(result + 8));
              *v22 = v25;
              *(__m128i *)(v22 + 1) = v24;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            }
            result += 24;
            if ( v10 == (__m128i *)result )
              return result;
          }
        }
        return result;
      }
      v27 = *(_DWORD *)(a1 + 24);
      v8 = 192;
      v7 = 64;
    }
    v42 = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v42;
  }
  v28 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v29 = &v5[3 * v27];
  if ( v28 )
  {
    v30 = *(_QWORD **)(a1 + 16);
    v31 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v30 = (_QWORD *)(a1 + 16);
    v31 = 12;
  }
  for ( i = &v30[v31]; i != v30; v30 += 3 )
  {
    if ( v30 )
      *v30 = -4;
  }
  for ( j = v5; v29 != j; j += 3 )
  {
    v25 = *j;
    if ( *j != -8 && v25 != -4 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v34 = a1 + 16;
        v35 = 3;
      }
      else
      {
        v41 = *(_DWORD *)(a1 + 24);
        v34 = *(_QWORD *)(a1 + 16);
        if ( !v41 )
        {
LABEL_74:
          MEMORY[0] = v25;
          BUG();
        }
        v35 = v41 - 1;
      }
      v36 = 1;
      v37 = 0;
      v38 = v35 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v39 = (__int64 *)(v34 + 24LL * v38);
      v40 = *v39;
      if ( *v39 != v25 )
      {
        while ( v40 != -4 )
        {
          if ( !v37 && v40 == -8 )
            v37 = v39;
          v38 = v35 & (v36 + v38);
          v39 = (__int64 *)(v34 + 24LL * v38);
          v40 = *v39;
          if ( v25 == *v39 )
            goto LABEL_43;
          ++v36;
        }
        if ( v37 )
          v39 = v37;
      }
LABEL_43:
      *v39 = v25;
      *(__m128i *)(v39 + 1) = _mm_loadu_si128((const __m128i *)(j + 1));
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v5);
}
