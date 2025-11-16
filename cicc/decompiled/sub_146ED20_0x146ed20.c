// Function: sub_146ED20
// Address: 0x146ed20
//
_QWORD *__fastcall sub_146ED20(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rdi
  _QWORD *i; // rdx
  __int64 *v10; // rdx
  __int64 v11; // rcx
  int v12; // esi
  __int64 v13; // r8
  int v14; // esi
  __int64 v15; // r9
  __int64 *v16; // r13
  __int64 v17; // r10
  int v18; // r11d
  unsigned __int64 v19; // r10
  unsigned __int64 v20; // r10
  unsigned int j; // eax
  __int64 *v22; // r10
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rdx
  _QWORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
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
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( 1 )
      {
        v11 = *v10;
        if ( *v10 == -8 )
        {
          if ( v10[1] != -8 )
            goto LABEL_12;
          v10 += 2;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else if ( v11 == -16 && v10[1] == -16 )
        {
          v10 += 2;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v13 = v10[1];
          v14 = v12 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 0;
          v17 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
          v18 = 1;
          v19 = (((v17 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                - 1
                - (v17 << 32)) >> 22)
              ^ ((v17 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
               - 1
               - (v17 << 32));
          v20 = ((9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13)))) >> 15)
              ^ (9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13))));
          for ( j = v14 & (((v20 - 1 - (v20 << 27)) >> 31) ^ (v20 - 1 - ((_DWORD)v20 << 27))); ; j = v14 & v25 )
          {
            v22 = (__int64 *)(v15 + 16LL * j);
            v23 = *v22;
            if ( v11 == *v22 && v22[1] == v13 )
              break;
            if ( v23 == -8 )
            {
              if ( v22[1] == -8 )
              {
                if ( v16 )
                  v22 = v16;
                break;
              }
            }
            else if ( v23 == -16 && v22[1] == -16 && !v16 )
            {
              v16 = (__int64 *)(v15 + 16LL * j);
            }
            v25 = v18 + j;
            ++v18;
          }
          *v22 = v11;
          v24 = v10[1];
          v10 += 2;
          v22[1] = v24;
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v26]; k != result; result += 2 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
  }
  return result;
}
