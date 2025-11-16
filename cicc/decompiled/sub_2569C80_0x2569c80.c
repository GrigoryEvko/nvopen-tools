// Function: sub_2569C80
// Address: 0x2569c80
//
_QWORD *__fastcall sub_2569C80(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r15
  _QWORD *i; // rcx
  __int64 j; // rbx
  __int64 v10; // rdx
  int v11; // ecx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r9
  unsigned int v15; // eax
  __int64 v16; // r12
  __int64 v17; // r8
  __int64 v18; // rdi
  int v19; // r11d
  __int64 v20; // r10
  __m128i v21; // xmm1
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // esi
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // rdx
  _QWORD *k; // rcx
  __int64 v30; // [rsp+8h] [rbp-48h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(96LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v30 = 96 * v4;
    v7 = 96 * v4 + v3;
    for ( i = &result[12 * *(unsigned int *)(a1 + 24)]; i != result; result += 12 )
    {
      if ( result )
      {
        *result = 0x7FFFFFFFFFFFFFFFLL;
        result[1] = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
    if ( v7 != v3 )
    {
      for ( j = v3; v7 != j; j += 96 )
      {
        v10 = *(_QWORD *)j;
        if ( *(_QWORD *)j == 0x7FFFFFFFFFFFFFFFLL )
        {
          if ( *(_QWORD *)(j + 8) != 0x7FFFFFFFFFFFFFFFLL )
            goto LABEL_12;
        }
        else if ( v10 != 0x7FFFFFFFFFFFFFFELL || *(_QWORD *)(j + 8) != 0x7FFFFFFFFFFFFFFELL )
        {
LABEL_12:
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = _mm_loadu_si128((const __m128i *)j);
            BUG();
          }
          v12 = *(_QWORD *)(j + 8);
          v13 = (unsigned int)(v11 - 1);
          v14 = *(_QWORD *)(a1 + 8);
          v15 = v13
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v12) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))) >> 31)
               ^ (756364221 * v12));
          v16 = v14 + 96LL * v15;
          v17 = *(_QWORD *)(v16 + 8);
          v18 = *(_QWORD *)v16;
          if ( v12 != v17 || v10 != v18 )
          {
            v19 = 1;
            v20 = 0;
            while ( 1 )
            {
              if ( v18 == 0x7FFFFFFFFFFFFFFFLL )
              {
                if ( v17 == 0x7FFFFFFFFFFFFFFFLL )
                {
                  if ( v20 )
                    v16 = v20;
                  break;
                }
              }
              else if ( v17 == 0x7FFFFFFFFFFFFFFELL && v18 == 0x7FFFFFFFFFFFFFFELL && !v20 )
              {
                v20 = v16;
              }
              v10 = (unsigned int)(v19 + 1);
              v15 = v13 & (v19 + v15);
              v16 = v14 + 96LL * v15;
              v17 = *(_QWORD *)(v16 + 8);
              v18 = *(_QWORD *)v16;
              if ( v12 == v17 && *(_QWORD *)j == v18 )
                break;
              ++v19;
            }
          }
          v21 = _mm_loadu_si128((const __m128i *)j);
          *(_QWORD *)(v16 + 16) = v16 + 32;
          *(_QWORD *)(v16 + 24) = 0x400000000LL;
          *(__m128i *)v16 = v21;
          if ( *(_DWORD *)(j + 24) )
            sub_2538950(v16 + 16, (char **)(j + 16), v10, v13, v17, v14);
          v22 = *(_QWORD *)(j + 64);
          v23 = v16 + 56;
          if ( v22 )
          {
            v24 = *(_DWORD *)(j + 56);
            *(_QWORD *)(v16 + 64) = v22;
            *(_DWORD *)(v16 + 56) = v24;
            *(_QWORD *)(v16 + 72) = *(_QWORD *)(j + 72);
            *(_QWORD *)(v16 + 80) = *(_QWORD *)(j + 80);
            *(_QWORD *)(v22 + 8) = v23;
            *(_QWORD *)(v16 + 88) = *(_QWORD *)(j + 88);
            *(_QWORD *)(j + 64) = 0;
            *(_QWORD *)(j + 72) = j + 56;
            *(_QWORD *)(j + 80) = j + 56;
            *(_QWORD *)(j + 88) = 0;
          }
          else
          {
            *(_DWORD *)(v16 + 56) = 0;
            *(_QWORD *)(v16 + 64) = 0;
            *(_QWORD *)(v16 + 72) = v23;
            *(_QWORD *)(v16 + 80) = v23;
            *(_QWORD *)(v16 + 88) = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v25 = *(_QWORD *)(j + 64);
          while ( v25 )
          {
            sub_253AEE0(*(_QWORD *)(v25 + 24));
            v26 = v25;
            v25 = *(_QWORD *)(v25 + 16);
            j_j___libc_free_0(v26);
          }
          v27 = *(_QWORD *)(j + 16);
          if ( v27 != j + 32 )
            _libc_free(v27);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v3, v30, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[12 * v28]; k != result; result += 12 )
    {
      if ( result )
      {
        *result = 0x7FFFFFFFFFFFFFFFLL;
        result[1] = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
  }
  return result;
}
