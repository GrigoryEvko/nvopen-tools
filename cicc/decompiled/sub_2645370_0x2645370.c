// Function: sub_2645370
// Address: 0x2645370
//
_QWORD *__fastcall sub_2645370(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // esi
  int v13; // esi
  __int64 v14; // rdi
  int v15; // r10d
  _QWORD *v16; // r9
  unsigned int v17; // ecx
  _QWORD *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r12
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v26 = v3;
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(32LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 32 * v4;
    v8 = (__int64 *)(32 * v4 + v3);
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
    for ( j = (__int64 *)v26; v8 != j; j += 4 )
    {
      while ( 1 )
      {
        v11 = *j;
        if ( (unsigned __int64)*j <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (484763065 * v11));
          v18 = (_QWORD *)(v14 + 32LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -1 )
            {
              if ( !v16 && v19 == -2 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (_QWORD *)(v14 + 32LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_14:
          *v18 = v11;
          v18[1] = j[1];
          v18[2] = j[2];
          v18[3] = j[3];
          j[2] = 0;
          j[1] = 0;
          j[3] = 0;
          ++*(_DWORD *)(a1 + 16);
          v20 = j[2];
          v21 = j[1];
          if ( v20 != v21 )
          {
            do
            {
              sub_C7D6A0(*(_QWORD *)(v21 + 48), 4LL * *(unsigned int *)(v21 + 64), 4);
              v22 = *(_QWORD *)(v21 + 8);
              if ( v22 )
                j_j___libc_free_0(v22);
              v21 += 72LL;
            }
            while ( v20 != v21 );
            v21 = j[1];
          }
          if ( v21 )
            break;
        }
        j += 4;
        if ( v8 == j )
          return (_QWORD *)sub_C7D6A0(v26, v25, 8);
      }
      j_j___libc_free_0(v21);
    }
    return (_QWORD *)sub_C7D6A0(v26, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v23]; k != result; result += 4 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
