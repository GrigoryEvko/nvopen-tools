// Function: sub_22BA1C0
// Address: 0x22ba1c0
//
_DWORD *__fastcall sub_22BA1C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r14
  __int64 v5; // rbx
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _DWORD *i; // rdx
  __int64 j; // r14
  int v12; // edx
  int v13; // ecx
  int v14; // edi
  __int64 v15; // r9
  int *v16; // r10
  int v17; // r11d
  unsigned int v18; // esi
  int *v19; // rcx
  int v20; // r8d
  __int64 v21; // rbx
  unsigned __int64 v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rdi
  _DWORD *k; // rdx
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v27 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_DWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 32 * v4;
    v9 = 32 * v4 + v5;
    for ( i = &result[8 * v8]; i != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
    for ( j = v5; v9 != j; j += 32 )
    {
      while ( 1 )
      {
        v12 = *(_DWORD *)j;
        if ( *(_DWORD *)j <= 0xFFFFFFFD )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 0;
          v17 = 1;
          v18 = (v13 - 1) & (37 * v12);
          v19 = (int *)(v15 + 32LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -1 )
            {
              if ( !v16 && v20 == -2 )
                v16 = v19;
              v18 = v14 & (v17 + v18);
              v19 = (int *)(v15 + 32LL * v18);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_14;
              ++v17;
            }
            if ( v16 )
              v19 = v16;
          }
LABEL_14:
          *v19 = v12;
          *((_QWORD *)v19 + 1) = *(_QWORD *)(j + 8);
          *((_QWORD *)v19 + 2) = *(_QWORD *)(j + 16);
          *((_QWORD *)v19 + 3) = *(_QWORD *)(j + 24);
          *(_QWORD *)(j + 16) = 0;
          *(_QWORD *)(j + 8) = 0;
          *(_QWORD *)(j + 24) = 0;
          ++*(_DWORD *)(a1 + 16);
          v21 = *(_QWORD *)(j + 16);
          v22 = *(_QWORD *)(j + 8);
          if ( v21 != v22 )
          {
            do
            {
              v23 = *(unsigned int *)(v22 + 144);
              v24 = *(_QWORD *)(v22 + 128);
              v22 += 152LL;
              sub_C7D6A0(v24, 8 * v23, 4);
              sub_C7D6A0(*(_QWORD *)(v22 - 56), 8LL * *(unsigned int *)(v22 - 40), 4);
              sub_C7D6A0(*(_QWORD *)(v22 - 88), 16LL * *(unsigned int *)(v22 - 72), 8);
              sub_C7D6A0(*(_QWORD *)(v22 - 120), 16LL * *(unsigned int *)(v22 - 104), 8);
            }
            while ( v21 != v22 );
            v22 = *(_QWORD *)(j + 8);
          }
          if ( v22 )
            break;
        }
        j += 32;
        if ( v9 == j )
          return (_DWORD *)sub_C7D6A0(v27, v26, 8);
      }
      j_j___libc_free_0(v22);
    }
    return (_DWORD *)sub_C7D6A0(v27, v26, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[8 * *(unsigned int *)(a1 + 24)]; k != result; result += 8 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
