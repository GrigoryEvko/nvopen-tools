// Function: sub_1A102D0
// Address: 0x1a102d0
//
__int64 __fastcall sub_1A102D0(__int64 a1, int a2)
{
  __int64 v2; // r13
  __int64 *v4; // r12
  unsigned __int64 v5; // rax
  __int64 result; // rax
  __int64 *v7; // r8
  __int64 i; // rdx
  __int64 *v9; // rdx
  __int64 v10; // rcx
  int v11; // esi
  int v12; // edi
  int v13; // esi
  __int64 v14; // r9
  __int64 *v15; // r14
  int v16; // r13d
  unsigned __int64 v17; // r10
  unsigned __int64 v18; // r10
  unsigned int j; // eax
  __int64 *v20; // r10
  __int64 v21; // r11
  int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 k; // rdx

  v2 = *(unsigned int *)(a1 + 24);
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
  result = sub_22077B0(24LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[3 * v2];
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
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
          if ( *((_DWORD *)v9 + 2) != -1 )
            goto LABEL_12;
          v9 += 3;
          if ( v7 == v9 )
            return j___libc_free_0(v4);
        }
        else if ( v10 == -16 && *((_DWORD *)v9 + 2) == -2 )
        {
          v9 += 3;
          if ( v7 == v9 )
            return j___libc_free_0(v4);
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
          v12 = *((_DWORD *)v9 + 2);
          v13 = v11 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 0;
          v16 = 1;
          v17 = ((((unsigned int)(37 * v12)
                 | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v12) << 32)) >> 22)
              ^ (((unsigned int)(37 * v12)
                | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v12) << 32));
          v18 = ((9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13)))) >> 15)
              ^ (9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13))));
          for ( j = v13 & (((v18 - 1 - (v18 << 27)) >> 31) ^ (v18 - 1 - ((_DWORD)v18 << 27))); ; j = v13 & v23 )
          {
            v20 = (__int64 *)(v14 + 24LL * j);
            v21 = *v20;
            if ( v10 == *v20 && *((_DWORD *)v20 + 2) == v12 )
              break;
            if ( v21 == -8 )
            {
              if ( *((_DWORD *)v20 + 2) == -1 )
              {
                if ( v15 )
                  v20 = v15;
                break;
              }
            }
            else if ( v21 == -16 && *((_DWORD *)v20 + 2) == -2 && !v15 )
            {
              v15 = (__int64 *)(v14 + 24LL * j);
            }
            v23 = v16 + j;
            ++v16;
          }
          *v20 = v10;
          v22 = *((_DWORD *)v9 + 2);
          v9 += 3;
          *((_DWORD *)v20 + 2) = v22;
          v20[2] = *(v9 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v7 == v9 )
            return j___libc_free_0(v4);
        }
      }
    }
    return j___libc_free_0(v4);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v24; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
