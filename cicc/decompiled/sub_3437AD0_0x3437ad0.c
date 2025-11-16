// Function: sub_3437AD0
// Address: 0x3437ad0
//
__int64 __fastcall sub_3437AD0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r9
  __int64 i; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  int v14; // r8d
  int v15; // r10d
  int v16; // r8d
  __int64 v17; // r11
  __int64 v18; // r15
  int v19; // r14d
  unsigned int j; // ebx
  __int64 v21; // rdx
  unsigned int v22; // edx
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 k; // rdx
  int v26; // ecx

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
  result = sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = v5 + v9;
    for ( i = result + 16 * v8; i != result; result += 16 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      do
      {
        while ( 1 )
        {
          v13 = *(_QWORD *)v12;
          if ( *(_QWORD *)v12 || *(_DWORD *)(v12 + 8) <= 0xFFFFFFFD )
            break;
          v12 += 16;
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = *(_QWORD *)v12;
          MEMORY[8] = *(_DWORD *)(v12 + 8);
          BUG();
        }
        v15 = *(_DWORD *)(v12 + 8);
        v16 = v14 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 0;
        v19 = 1;
        for ( j = v16 & (v15 + ((v13 >> 9) ^ (v13 >> 4))); ; j = v16 & v22 )
        {
          v21 = v17 + 16LL * j;
          if ( v13 == *(_QWORD *)v21 && v15 == *(_DWORD *)(v21 + 8) )
            break;
          if ( !*(_QWORD *)v21 )
          {
            v26 = *(_DWORD *)(v21 + 8);
            if ( v26 == -1 )
            {
              if ( v18 )
                v21 = v18;
              break;
            }
            if ( !v18 && v26 == -2 )
              v18 = v17 + 16LL * j;
          }
          v22 = j + v19++;
        }
        v23 = *(_QWORD *)v12;
        v12 += 16;
        *(_QWORD *)v21 = v23;
        *(_DWORD *)(v21 + 8) = *(_DWORD *)(v12 - 8);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 16 * v24; k != result; result += 16 )
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
