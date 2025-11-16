// Function: sub_1B36650
// Address: 0x1b36650
//
_DWORD *__fastcall sub_1B36650(__int64 a1, int a2)
{
  __int64 v3; // r13
  _DWORD *v4; // r12
  unsigned __int64 v5; // rdi
  _DWORD *result; // rax
  __int64 v7; // rdx
  _DWORD *v8; // r8
  _DWORD *i; // rdx
  _DWORD *v10; // rdx
  int v11; // ecx
  int v12; // esi
  int v13; // edi
  int v14; // esi
  __int64 v15; // r9
  int v16; // r13d
  int *v17; // r11
  unsigned __int64 v18; // r10
  unsigned __int64 v19; // r10
  unsigned int j; // eax
  int *v21; // r10
  int v22; // r14d
  int v23; // eax
  unsigned int v24; // eax
  __int64 v25; // rdx
  _DWORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_DWORD **)(a1 + 8);
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
  result = (_DWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( 1 )
      {
        v11 = *v10;
        if ( *v10 == -1 )
        {
          if ( v10[1] != -1 )
            goto LABEL_12;
          v10 += 4;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else if ( v11 == -2 && v10[1] == -2 )
        {
          v10 += 4;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v13 = v10[1];
          v14 = v12 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = ((((unsigned int)(37 * v13) | ((unsigned __int64)(unsigned int)(37 * v11) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v13) << 32)) >> 22)
              ^ (((unsigned int)(37 * v13) | ((unsigned __int64)(unsigned int)(37 * v11) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v13) << 32));
          v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
              ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
          for ( j = v14 & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; j = v14 & v24 )
          {
            v21 = (int *)(v15 + 16LL * j);
            v22 = *v21;
            if ( v11 == *v21 && v21[1] == v13 )
              break;
            if ( v22 == -1 )
            {
              if ( v21[1] == -1 )
              {
                if ( v17 )
                  v21 = v17;
                break;
              }
            }
            else if ( v22 == -2 && v21[1] == -2 && !v17 )
            {
              v17 = (int *)(v15 + 16LL * j);
            }
            v24 = v16 + j;
            ++v16;
          }
          *v21 = v11;
          v23 = v10[1];
          v10 += 4;
          v21[1] = v23;
          *((_QWORD *)v21 + 1) = *((_QWORD *)v10 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v25]; k != result; result += 4 )
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
