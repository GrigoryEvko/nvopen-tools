// Function: sub_E70840
// Address: 0xe70840
//
_DWORD *__fastcall sub_E70840(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  _DWORD *v9; // r9
  _DWORD *i; // rdx
  _DWORD *v11; // rdx
  int v12; // ecx
  int v13; // r10d
  int v14; // esi
  int v15; // r10d
  __int64 v16; // rdi
  int v17; // r14d
  int *v18; // r11
  unsigned int j; // eax
  int *v20; // r8
  int v21; // r15d
  int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdx
  _DWORD *k; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = (_DWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 16LL * v4;
    v9 = (_DWORD *)(v5 + v26);
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v9 != (_DWORD *)v5 )
    {
      v11 = (_DWORD *)v5;
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 == -1 )
        {
          if ( v11[1] != -1 )
            goto LABEL_12;
          v11 += 4;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, v26, 8);
        }
        else if ( v12 == -2 && v11[1] == -2 )
        {
          v11 += 4;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, v26, 8);
        }
        else
        {
LABEL_12:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v14 = v11[1];
          v15 = v13 - 1;
          v17 = 1;
          v18 = 0;
          for ( j = v15
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v14) | ((unsigned __int64)(unsigned int)(37 * v12) << 32))) >> 31)
                   ^ (756364221 * v14)); ; j = v15 & v23 )
          {
            v16 = *(_QWORD *)(a1 + 8);
            v20 = (int *)(v16 + 16LL * j);
            v21 = *v20;
            if ( v12 == *v20 && v14 == v20[1] )
              break;
            if ( v21 == -1 )
            {
              if ( v20[1] == -1 )
              {
                if ( v18 )
                  v20 = v18;
                break;
              }
            }
            else if ( v21 == -2 && v20[1] == -2 && !v18 )
            {
              v18 = (int *)(v16 + 16LL * j);
            }
            v23 = v17 + j;
            ++v17;
          }
          *v20 = v12;
          v22 = v11[1];
          v11 += 4;
          v20[1] = v22;
          *((_QWORD *)v20 + 1) = *((_QWORD *)v11 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, v26, 8);
        }
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v24]; k != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
  }
  return result;
}
