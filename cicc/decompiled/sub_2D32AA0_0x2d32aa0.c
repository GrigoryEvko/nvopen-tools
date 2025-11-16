// Function: sub_2D32AA0
// Address: 0x2d32aa0
//
void __fastcall sub_2D32AA0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r12d
  char v8; // r13
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  bool v14; // zf
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // r10
  __int64 v24; // r9
  unsigned int j; // eax
  __int64 v26; // r15
  __int64 v27; // r8
  int v28; // ecx
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // rcx
  unsigned int v33; // r15d
  __int64 v34; // r12
  __m128i *v35; // r14
  __int64 v36; // rdx
  unsigned __int64 v37; // rdi
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-1A8h]
  __int64 v42; // [rsp+18h] [rbp-198h]
  _BYTE v43[400]; // [rsp+20h] [rbp-190h] BYREF

  v7 = a2;
  v42 = *(_QWORD *)(a1 + 16);
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v8 )
    {
      v10 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
LABEL_8:
      v13 = 88LL * v10 + v42;
      v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v40 = 88LL * v10;
      if ( v14 )
      {
        v15 = *(_QWORD **)(a1 + 16);
        v16 = 11LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        v15 = (_QWORD *)(a1 + 16);
        v16 = 44;
      }
      for ( i = &v15[v16]; i != v15; v15 += 11 )
      {
        if ( v15 )
        {
          *v15 = -4096;
          v15[1] = -4096;
        }
      }
      v18 = v42;
      if ( v13 == v42 )
      {
LABEL_31:
        sub_C7D6A0(v42, v40, 8);
        return;
      }
      while ( 1 )
      {
        v19 = *(_QWORD *)v18;
        if ( *(_QWORD *)v18 == -4096 )
        {
          if ( *(_QWORD *)(v18 + 8) != -4096 )
            goto LABEL_17;
        }
        else if ( v19 != -8192 || *(_QWORD *)(v18 + 8) != -8192 )
        {
LABEL_17:
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v20 = a1 + 16;
            v21 = 3;
          }
          else
          {
            v28 = *(_DWORD *)(a1 + 24);
            v20 = *(_QWORD *)(a1 + 16);
            if ( !v28 )
            {
              MEMORY[0] = *(_QWORD *)v18;
              BUG();
            }
            v21 = (unsigned int)(v28 - 1);
          }
          v22 = *(_QWORD *)(v18 + 8);
          v23 = 0;
          v24 = 1;
          for ( j = v21
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)
                      | ((unsigned __int64)(((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)))); ; j = v21 & v39 )
          {
            v26 = v20 + 88LL * j;
            v27 = *(_QWORD *)v26;
            if ( v19 == *(_QWORD *)v26 && *(_QWORD *)(v26 + 8) == v22 )
              break;
            if ( v27 == -4096 )
            {
              if ( *(_QWORD *)(v26 + 8) == -4096 )
              {
                if ( v23 )
                  v26 = v23;
                break;
              }
            }
            else if ( v27 == -8192 && *(_QWORD *)(v26 + 8) == -8192 && !v23 )
            {
              v23 = v20 + 88LL * j;
            }
            v39 = v24 + j;
            v24 = (unsigned int)(v24 + 1);
          }
          *(_QWORD *)v26 = v19;
          v29 = *(_QWORD *)(v18 + 8);
          *(_QWORD *)(v26 + 24) = 0x600000000LL;
          *(_QWORD *)(v26 + 8) = v29;
          *(_QWORD *)(v26 + 16) = v26 + 32;
          if ( *(_DWORD *)(v18 + 24) )
            sub_2D23900(v26 + 16, (char **)(v18 + 16), v19, v21, v27, v24);
          *(_DWORD *)(v26 + 80) = *(_DWORD *)(v18 + 80);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v30 = *(_QWORD *)(v18 + 16);
          if ( v30 != v18 + 32 )
            _libc_free(v30);
        }
        v18 += 88;
        if ( v13 == v18 )
          goto LABEL_31;
      }
    }
    v31 = a1 + 16;
    v32 = a1 + 368;
  }
  else
  {
    v9 = sub_AF1560(a2 - 1);
    v7 = v9;
    if ( v9 > 0x40 )
    {
      v31 = a1 + 16;
      v32 = a1 + 368;
      if ( !v8 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        v11 = 88LL * v9;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v8 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        v11 = 5632;
        v7 = 64;
LABEL_5:
        v12 = sub_C7D670(v11, 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v12;
        goto LABEL_8;
      }
      v31 = a1 + 16;
      v7 = 64;
      v32 = a1 + 368;
    }
  }
  v33 = v7;
  v34 = v32;
  v35 = (__m128i *)v43;
  do
  {
    if ( *(_QWORD *)v31 == -4096 )
    {
      if ( *(_QWORD *)(v31 + 8) == -4096 )
        goto LABEL_50;
    }
    else if ( *(_QWORD *)v31 == -8192 && *(_QWORD *)(v31 + 8) == -8192 )
    {
      goto LABEL_50;
    }
    if ( v35 )
      *v35 = _mm_loadu_si128((const __m128i *)v31);
    v35[1].m128i_i64[1] = 0x600000000LL;
    v35[1].m128i_i64[0] = (__int64)v35[2].m128i_i64;
    v36 = *(unsigned int *)(v31 + 24);
    if ( (_DWORD)v36 )
      sub_2D23900((__int64)v35[1].m128i_i64, (char **)(v31 + 16), v36, v32, a5, a6);
    v37 = *(_QWORD *)(v31 + 16);
    v35 = (__m128i *)((char *)v35 + 88);
    v35[-1].m128i_i32[2] = *(_DWORD *)(v31 + 80);
    if ( v37 != v31 + 32 )
      _libc_free(v37);
LABEL_50:
    v31 += 88;
  }
  while ( v31 != v34 );
  if ( v33 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v38 = sub_C7D670(88LL * v33, 8);
    *(_DWORD *)(a1 + 24) = v33;
    *(_QWORD *)(a1 + 16) = v38;
  }
  sub_2D32810(a1, (__int64)v43, (__int64)v35);
}
