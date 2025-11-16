// Function: sub_209BBE0
// Address: 0x209bbe0
//
__int64 __fastcall sub_209BBE0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rax
  __int64 result; // rax
  __int64 *v7; // r8
  __int64 i; // rdx
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // rcx
  int v11; // edx
  int v12; // edi
  int v13; // r9d
  __int64 v14; // r10
  __int64 v15; // r14
  int v16; // ebx
  unsigned int j; // edx
  __int64 v18; // rsi
  unsigned int v19; // edx
  unsigned __int64 v20; // rdx
  __int64 k; // rdx
  int v22; // r11d

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
  result = sub_22077B0(24LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[3 * v3];
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v7 != v4 )
    {
      v9 = (unsigned __int64 *)v4;
      do
      {
        while ( 1 )
        {
          v10 = *v9;
          if ( *v9 || *((_DWORD *)v9 + 2) <= 0xFFFFFFFD )
            break;
          v9 += 3;
          if ( v7 == (__int64 *)v9 )
            return j___libc_free_0(v4);
        }
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = *v9;
          MEMORY[8] = *((_DWORD *)v9 + 2);
          BUG();
        }
        v12 = v11 - 1;
        v13 = *((_DWORD *)v9 + 2);
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 0;
        v16 = 1;
        for ( j = (v11 - 1) & (v13 + ((v10 >> 9) ^ (v10 >> 4))); ; j = v12 & v19 )
        {
          v18 = v14 + 24LL * j;
          if ( v10 == *(_QWORD *)v18 && v13 == *(_DWORD *)(v18 + 8) )
            break;
          if ( !*(_QWORD *)v18 )
          {
            v22 = *(_DWORD *)(v18 + 8);
            if ( v22 == -1 )
            {
              if ( v15 )
                v18 = v15;
              break;
            }
            if ( !v15 && v22 == -2 )
              v15 = v14 + 24LL * j;
          }
          v19 = v16 + j;
          ++v16;
        }
        v20 = *v9;
        v9 += 3;
        *(_QWORD *)v18 = v20;
        *(_DWORD *)(v18 + 8) = *((_DWORD *)v9 - 4);
        *(_QWORD *)(v18 + 16) = *(v9 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v7 != (__int64 *)v9 );
    }
    return j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24LL * *(unsigned int *)(a1 + 24); k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
