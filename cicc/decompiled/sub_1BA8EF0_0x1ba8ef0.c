// Function: sub_1BA8EF0
// Address: 0x1ba8ef0
//
_QWORD *__fastcall sub_1BA8EF0(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 *v7; // r8
  _QWORD *i; // rdx
  __int64 *v9; // rdx
  __int64 v10; // rcx
  int v11; // esi
  __int64 v12; // rdi
  int v13; // esi
  __int64 v14; // r9
  __int64 *v15; // r14
  int v16; // r13d
  __int64 v17; // r10
  unsigned __int64 v18; // r10
  unsigned __int64 v19; // r10
  unsigned int j; // eax
  __int64 *v21; // r10
  __int64 v22; // r11
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rdx
  _QWORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(24LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[3 * v3];
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      while ( 1 )
      {
        v10 = *v9;
        if ( *v9 == -8 )
        {
          if ( v9[1] != -8 )
            goto LABEL_12;
          v9 += 3;
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else if ( v10 == -16 && v9[1] == -16 )
        {
          v9 += 3;
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *v9;
            BUG();
          }
          v12 = v9[1];
          v13 = v11 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 0;
          v16 = 1;
          v17 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
          v18 = (((v17 | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))
                - 1
                - (v17 << 32)) >> 22)
              ^ ((v17 | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))
               - 1
               - (v17 << 32));
          v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
              ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
          for ( j = v13 & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; j = v13 & v24 )
          {
            v21 = (__int64 *)(v14 + 24LL * j);
            v22 = *v21;
            if ( v10 == *v21 && v21[1] == v12 )
              break;
            if ( v22 == -8 )
            {
              if ( v21[1] == -8 )
              {
                if ( v15 )
                  v21 = v15;
                break;
              }
            }
            else if ( v22 == -16 && v21[1] == -16 && !v15 )
            {
              v15 = (__int64 *)(v14 + 24LL * j);
            }
            v24 = v16 + j;
            ++v16;
          }
          *v21 = v10;
          v23 = v9[1];
          v9 += 3;
          v21[1] = v23;
          v21[2] = *(v9 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v25]; k != result; result += 3 )
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
