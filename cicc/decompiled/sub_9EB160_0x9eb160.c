// Function: sub_9EB160
// Address: 0x9eb160
//
_QWORD *__fastcall sub_9EB160(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 *v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 *v9; // r12
  _QWORD *i; // rdx
  __int64 *j; // r15
  __int64 v12; // rdx
  int v13; // edi
  int v14; // edi
  __int64 v15; // r9
  int v16; // r11d
  __int64 *v17; // r10
  unsigned int v18; // esi
  __int64 *v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 *v22; // rdi
  __int64 *v23; // rsi
  int v24; // r8d
  __int64 v25; // r13
  __int64 v26; // r14
  __int64 v27; // rdi
  __int64 v28; // rcx
  _QWORD *k; // rdx
  __int64 v30; // [rsp+0h] [rbp-40h]
  __int64 *v31; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  v31 = v5;
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v30 = 56 * v4;
    v9 = &v5[7 * v4];
    for ( i = &result[7 * v8]; i != result; result += 7 )
    {
      if ( result )
        *result = -1;
    }
    for ( j = v5; v9 != j; j += 7 )
    {
      while ( 1 )
      {
        v12 = *j;
        if ( (unsigned __int64)*j <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = v14 & (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12));
          v19 = (__int64 *)(v15 + 56LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -1 )
            {
              if ( !v17 && v20 == -2 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (__int64 *)(v15 + 56LL * v18);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_14;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_14:
          *v19 = v12;
          v21 = j[3];
          v22 = v19 + 2;
          v23 = j + 2;
          if ( v21 )
          {
            v24 = *((_DWORD *)j + 4);
            v19[3] = v21;
            *((_DWORD *)v19 + 4) = v24;
            v19[4] = j[4];
            v19[5] = j[5];
            *(_QWORD *)(v21 + 8) = v22;
            v19[6] = j[6];
            j[3] = 0;
            j[4] = (__int64)v23;
            j[5] = (__int64)v23;
            j[6] = 0;
          }
          else
          {
            *((_DWORD *)v19 + 4) = 0;
            v19[3] = 0;
            v19[4] = (__int64)v22;
            v19[5] = (__int64)v22;
            v19[6] = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v25 = j[3];
          if ( v25 )
            break;
        }
        j += 7;
        if ( v9 == j )
          return (_QWORD *)sub_C7D6A0(v31, v30, 8);
      }
      do
      {
        v26 = v25;
        sub_9C4EF0(*(_QWORD **)(v25 + 24));
        v27 = *(_QWORD *)(v25 + 32);
        v25 = *(_QWORD *)(v25 + 16);
        if ( v27 != v26 + 48 )
          j_j___libc_free_0(v27, *(_QWORD *)(v26 + 48) + 1LL);
        j_j___libc_free_0(v26, 64);
      }
      while ( v25 );
    }
    return (_QWORD *)sub_C7D6A0(v31, v30, 8);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[7 * v28]; k != result; result += 7 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
