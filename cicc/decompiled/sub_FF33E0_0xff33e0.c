// Function: sub_FF33E0
// Address: 0xff33e0
//
__int64 __fastcall sub_FF33E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r13d
  __int64 v8; // r14
  char v9; // dl
  unsigned __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  bool v16; // zf
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 i; // rdx
  __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rsi
  int v26; // r10d
  unsigned __int64 v27; // r11
  unsigned int j; // eax
  unsigned __int64 v29; // rdi
  __int64 v30; // r9
  __m128i *v31; // r14
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rax
  int v36; // edx
  int v37; // eax
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-138h]
  _BYTE v41[304]; // [rsp+10h] [rbp-130h] BYREF

  v7 = a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( (unsigned int)a2 <= 4 )
  {
    v14 = a1 + 16;
    v15 = a1 + 272;
    if ( !v9 )
    {
      v11 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
LABEL_8:
      v16 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v40 = v11 << 6;
      v17 = v8 + (v11 << 6);
      if ( v16 )
      {
        v18 = *(_QWORD *)(a1 + 16);
        v19 = (unsigned __int64)*(unsigned int *)(a1 + 24) << 6;
      }
      else
      {
        v18 = a1 + 16;
        v19 = 256;
      }
      for ( i = v18 + v19; i != v18; v18 += 64 )
      {
        if ( v18 )
        {
          *(_QWORD *)v18 = -4096;
          *(_DWORD *)(v18 + 8) = 0x7FFFFFFF;
        }
      }
      v21 = v8;
      if ( v17 == v8 )
        return sub_C7D6A0(v8, v40, 8);
      while ( 1 )
      {
        v22 = *(_QWORD *)v21;
        if ( *(_QWORD *)v21 == -4096 )
        {
          if ( *(_DWORD *)(v21 + 8) != 0x7FFFFFFF )
            goto LABEL_17;
        }
        else if ( v22 != -8192 || *(_DWORD *)(v21 + 8) != 0x80000000 )
        {
LABEL_17:
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v23 = a1 + 16;
            v24 = 3;
          }
          else
          {
            v36 = *(_DWORD *)(a1 + 24);
            v23 = *(_QWORD *)(a1 + 16);
            if ( !v36 )
            {
              MEMORY[0] = *(_QWORD *)v21;
              BUG();
            }
            v24 = (unsigned int)(v36 - 1);
          }
          v25 = *(unsigned int *)(v21 + 8);
          v26 = 1;
          v27 = 0;
          for ( j = v24
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v25)
                      | ((unsigned __int64)(((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))) >> 31)
                   ^ (756364221 * v25)); ; j = v24 & v39 )
          {
            v29 = v23 + ((unsigned __int64)j << 6);
            v30 = *(_QWORD *)v29;
            if ( v22 == *(_QWORD *)v29 && *(_DWORD *)(v29 + 8) == (_DWORD)v25 )
              break;
            if ( v30 == -4096 )
            {
              if ( *(_DWORD *)(v29 + 8) == 0x7FFFFFFF )
              {
                if ( v27 )
                  v29 = v27;
                break;
              }
            }
            else if ( v30 == -8192 && *(_DWORD *)(v29 + 8) == 0x80000000 && !v27 )
            {
              v27 = v23 + ((unsigned __int64)j << 6);
            }
            v39 = v26 + j;
            ++v26;
          }
          *(_QWORD *)v29 = v22;
          v37 = *(_DWORD *)(v21 + 8);
          *(_QWORD *)(v29 + 24) = 0x400000000LL;
          *(_DWORD *)(v29 + 8) = v37;
          *(_QWORD *)(v29 + 16) = v29 + 32;
          if ( *(_DWORD *)(v21 + 24) )
          {
            v25 = v21 + 16;
            sub_FEE2C0(v29 + 16, (char **)(v21 + 16), v24, v22, v23, v30);
          }
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v38 = *(_QWORD *)(v21 + 16);
          if ( v38 != v21 + 32 )
            _libc_free(v38, v25);
        }
        v21 += 64;
        if ( v17 == v21 )
          return sub_C7D6A0(v8, v40, 8);
      }
    }
  }
  else
  {
    a4 = ((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16;
    v10 = (a4
         | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
             | (unsigned int)(a2 - 1)
             | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
           | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
           | (unsigned int)(a2 - 1)
           | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
         | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
           | (unsigned int)(a2 - 1)
           | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
         | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
         | (unsigned int)(a2 - 1)
         | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
        + 1;
    v7 = v10;
    if ( (unsigned int)v10 > 0x40 )
    {
      v14 = a1 + 16;
      v15 = a1 + 272;
      if ( !v9 )
      {
        v11 = *(unsigned int *)(a1 + 24);
        v12 = (unsigned __int64)(unsigned int)v10 << 6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v9 )
      {
        v11 = *(unsigned int *)(a1 + 24);
        v7 = 64;
        v12 = 4096;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v13;
        goto LABEL_8;
      }
      v14 = a1 + 16;
      v15 = a1 + 272;
      v7 = 64;
    }
  }
  v31 = (__m128i *)v41;
  do
  {
    if ( *(_QWORD *)v14 == -4096 )
    {
      if ( *(_DWORD *)(v14 + 8) == 0x7FFFFFFF )
        goto LABEL_37;
    }
    else if ( *(_QWORD *)v14 == -8192 && *(_DWORD *)(v14 + 8) == 0x80000000 )
    {
      goto LABEL_37;
    }
    if ( v31 )
      *v31 = _mm_loadu_si128((const __m128i *)v14);
    v32 = *(unsigned int *)(v14 + 24);
    v31[1].m128i_i64[0] = (__int64)v31[2].m128i_i64;
    v31[1].m128i_i64[1] = 0x400000000LL;
    if ( (_DWORD)v32 )
    {
      a2 = v14 + 16;
      sub_FEE2C0((__int64)v31[1].m128i_i64, (char **)(v14 + 16), v32, a4, a5, a6);
    }
    v33 = *(_QWORD *)(v14 + 16);
    v31 += 4;
    if ( v33 != v14 + 32 )
      _libc_free(v33, a2);
LABEL_37:
    v14 += 64;
  }
  while ( v14 != v15 );
  if ( v7 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v34 = sub_C7D670((unsigned __int64)v7 << 6, 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v34;
  }
  return sub_FF3180(a1, (__int64)v41, (__int64)v31);
}
