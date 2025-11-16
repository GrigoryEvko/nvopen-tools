// Function: sub_EE0D70
// Address: 0xee0d70
//
_QWORD *__fastcall sub_EE0D70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r8
  _QWORD *i; // rdx
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  __int64 v13; // r13
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // rdi
  unsigned __int64 *v17; // r9
  int v18; // r10d
  unsigned int v19; // edx
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // rsi
  const void *v22; // rsi
  __int64 v23; // r9
  __int64 v24; // rdi
  unsigned __int64 v25; // r10
  int v26; // r9d
  __int64 v27; // r10
  size_t v28; // rdx
  _QWORD *j; // rdx
  __int64 v30; // [rsp+0h] [rbp-50h]
  int v31; // [rsp+Ch] [rbp-44h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v33 = 24 * v4;
    v8 = v5 + 24 * v4;
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
        *result = -1;
    }
    v10 = v5 + 24;
    if ( v8 != v5 )
    {
      v32 = v5;
      v11 = v8;
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(v10 - 24);
          v13 = v10;
          if ( v12 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = 0;
            v18 = 1;
            v19 = v15 & (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12));
            v20 = (unsigned __int64 *)(v16 + 24LL * v19);
            v21 = *v20;
            if ( v12 != *v20 )
            {
              while ( v21 != -1 )
              {
                if ( !v17 && v21 == -2 )
                  v17 = v20;
                v8 = (unsigned int)(v18 + 1);
                v19 = v15 & (v18 + v19);
                v20 = (unsigned __int64 *)(v16 + 24LL * v19);
                v21 = *v20;
                if ( v12 == *v20 )
                  goto LABEL_14;
                ++v18;
              }
              if ( v17 )
                v20 = v17;
            }
LABEL_14:
            v22 = v20 + 3;
            *v20 = v12;
            v20[1] = (unsigned __int64)(v20 + 3);
            v20[2] = 0;
            v23 = *(unsigned int *)(v10 - 8);
            if ( (_DWORD)v23 && v20 + 1 != (unsigned __int64 *)(v10 - 16) )
            {
              v25 = *(_QWORD *)(v10 - 16);
              if ( v10 == v25 )
              {
                v30 = *(_QWORD *)(v10 - 16);
                v31 = *(_DWORD *)(v10 - 8);
                sub_C8D5F0((__int64)(v20 + 1), v22, (unsigned int)v23, 0x10u, v8, v23);
                v26 = v31;
                v27 = v30;
                v28 = 16LL * *(unsigned int *)(v10 - 8);
                if ( v28 )
                {
                  v22 = *(const void **)(v10 - 16);
                  memcpy((void *)v20[1], v22, v28);
                  v27 = v30;
                  v26 = v31;
                }
                *((_DWORD *)v20 + 4) = v26;
                *(_DWORD *)(v27 - 8) = 0;
              }
              else
              {
                v20[1] = v25;
                *((_DWORD *)v20 + 4) = *(_DWORD *)(v10 - 8);
                *((_DWORD *)v20 + 5) = *(_DWORD *)(v10 - 4);
                *(_QWORD *)(v10 - 16) = v10;
                *(_DWORD *)(v10 - 4) = 0;
                *(_DWORD *)(v10 - 8) = 0;
              }
            }
            ++*(_DWORD *)(a1 + 16);
            v24 = *(_QWORD *)(v10 - 16);
            if ( v10 != v24 )
              break;
          }
          v10 += 24;
          if ( v11 == v13 )
            goto LABEL_17;
        }
        _libc_free(v24, v22);
        v10 += 24;
      }
      while ( v11 != v13 );
LABEL_17:
      v5 = v32;
    }
    return (_QWORD *)sub_C7D6A0(v5, v33, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[3 * *(unsigned int *)(a1 + 24)]; j != result; result += 3 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
