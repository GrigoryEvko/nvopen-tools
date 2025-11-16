// Function: sub_31C3CB0
// Address: 0x31c3cb0
//
__int64 __fastcall sub_31C3CB0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 i; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 j; // rdx
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v15 = v4;
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
  result = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v5;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v4 + v9;
    for ( i = result + 32 * v8; i != result; result += 32 )
    {
      if ( result )
      {
        *(_DWORD *)result = 0x7FFFFFFF;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
      }
    }
    if ( v10 != v4 )
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v4 + 16);
        if ( v13 != -4096 )
          break;
        if ( *(_QWORD *)(v4 + 8) == -4096 && *(_DWORD *)v4 == 0x7FFFFFFF )
        {
          v4 += 32;
          if ( v10 == v4 )
            return sub_C7D6A0(v15, v9, 8);
        }
        else
        {
LABEL_11:
          sub_31C3810(a1, (int *)v4, v16);
          v12 = v16[0];
          *(_QWORD *)(v16[0] + 16) = *(_QWORD *)(v4 + 16);
          *(_QWORD *)(v12 + 8) = *(_QWORD *)(v4 + 8);
          *(_DWORD *)v12 = *(_DWORD *)v4;
          *(_DWORD *)(v16[0] + 24) = *(_DWORD *)(v4 + 24);
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v4 += 32;
          if ( v10 == v4 )
            return sub_C7D6A0(v15, v9, 8);
        }
      }
      if ( v13 == -8192 && *(_QWORD *)(v4 + 8) == -8192 && *(_DWORD *)v4 == 0x80000000 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v15, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 32LL * *(unsigned int *)(a1 + 24); j != result; result += 32 )
    {
      if ( result )
      {
        *(_DWORD *)result = 0x7FFFFFFF;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
      }
    }
  }
  return result;
}
