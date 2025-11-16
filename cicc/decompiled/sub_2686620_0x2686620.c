// Function: sub_2686620
// Address: 0x2686620
//
_QWORD *__fastcall sub_2686620(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  _QWORD *i; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rsi
  int v16; // r10d
  __int64 *v17; // r8
  unsigned int v18; // edx
  __int64 *v19; // r12
  __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  _QWORD *j; // rdx
  __int64 v24; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670((unsigned __int64)v6 << 7, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v24 = v4 << 7;
    v9 = v5 + (v4 << 7);
    for ( i = &result[16 * v8]; i != result; result += 16 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)v11;
          if ( *(_QWORD *)v11 != -8192 && v12 != -4096 )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = *(_QWORD *)v11;
              BUG();
            }
            v14 = v13 - 1;
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 1;
            v17 = 0;
            v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v19 = (__int64 *)(v15 + ((unsigned __int64)v18 << 7));
            v20 = *v19;
            if ( v12 != *v19 )
            {
              while ( v20 != -4096 )
              {
                if ( v20 == -8192 && !v17 )
                  v17 = v19;
                v18 = v14 & (v16 + v18);
                v19 = (__int64 *)(v15 + ((unsigned __int64)v18 << 7));
                v20 = *v19;
                if ( v12 == *v19 )
                  goto LABEL_15;
                ++v16;
              }
              if ( v17 )
                v19 = v17;
            }
LABEL_15:
            *v19 = v12;
            *((_BYTE *)v19 + 8) = *(_BYTE *)(v11 + 8);
            *((_BYTE *)v19 + 9) = *(_BYTE *)(v11 + 9);
            *((_BYTE *)v19 + 10) = *(_BYTE *)(v11 + 10);
            *((_BYTE *)v19 + 11) = *(_BYTE *)(v11 + 11);
            sub_C8CF70((__int64)(v19 + 2), v19 + 6, 2, v11 + 48, v11 + 16);
            sub_C8CF70((__int64)(v19 + 8), v19 + 12, 4, v11 + 96, v11 + 64);
            ++*(_DWORD *)(a1 + 16);
            if ( !*(_BYTE *)(v11 + 92) )
              _libc_free(*(_QWORD *)(v11 + 72));
            if ( !*(_BYTE *)(v11 + 44) )
              break;
          }
          v11 += 128;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, v24, 8);
        }
        v21 = *(_QWORD *)(v11 + 24);
        v11 += 128;
        _libc_free(v21);
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v24, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[16 * v22]; j != result; result += 16 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
