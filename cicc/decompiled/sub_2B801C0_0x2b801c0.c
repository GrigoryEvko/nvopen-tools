// Function: sub_2B801C0
// Address: 0x2b801c0
//
__int64 __fastcall sub_2B801C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // r12
  __int64 i; // rdx
  __int64 j; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 k; // rdx
  __int64 v14; // [rsp+8h] [rbp-48h]
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v14 = 40 * v3;
    v7 = v4 + 40 * v3;
    for ( i = result + 40LL * *(unsigned int *)(a1 + 24); i != result; result += 40 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
        *(_QWORD *)(result + 24) = -4096;
        *(_QWORD *)(result + 32) = -4096;
      }
    }
    if ( v7 != v4 )
    {
      for ( j = v4; v7 != j; j += 40 )
      {
        v11 = *(_QWORD *)(j + 32);
        if ( v11 == -4096 )
        {
          if ( *(_QWORD *)(j + 24) == -4096
            && *(_QWORD *)(j + 16) == -4096
            && *(_QWORD *)(j + 8) == -4096
            && *(_DWORD *)j == -1 )
          {
            continue;
          }
        }
        else if ( v11 == -8192
               && *(_QWORD *)(j + 24) == -8192
               && *(_QWORD *)(j + 16) == -8192
               && *(_QWORD *)(j + 8) == -8192
               && *(_DWORD *)j == -2 )
        {
          continue;
        }
        sub_2B44A60(a1, (int *)j, v15);
        v10 = v15[0];
        *(_QWORD *)(v15[0] + 32LL) = *(_QWORD *)(j + 32);
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(j + 24);
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(j + 16);
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(j + 8);
        *(_DWORD *)v10 = *(_DWORD *)j;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return sub_C7D6A0(v4, v14, 8);
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 40 * v12; k != result; result += 40 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
        *(_QWORD *)(result + 24) = -4096;
        *(_QWORD *)(result + 32) = -4096;
      }
    }
  }
  return result;
}
