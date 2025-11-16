// Function: sub_37B0D60
// Address: 0x37b0d60
//
__int64 __fastcall sub_37B0D60(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 i; // rdx
  __int64 v11; // rbx
  __int64 *v12; // rcx
  __int64 v13; // rax
  int v14; // eax
  __int64 j; // rdx
  __int64 *v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+18h] [rbp-38h] BYREF

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
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 40 * v3;
    v9 = v4 + 40 * v3;
    for ( i = result + 40 * v7; i != result; result += 40 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
        *(_QWORD *)(result + 16) = 0;
        *(_DWORD *)(result + 24) = -1;
      }
    }
    if ( v9 != v4 )
    {
      v11 = v4;
      v12 = &v17;
      do
      {
        if ( !*(_QWORD *)v11 )
        {
          v14 = *(_DWORD *)(v11 + 8);
          if ( v14 == -1 )
          {
            if ( !*(_QWORD *)(v11 + 16) && *(_DWORD *)(v11 + 24) == -1 )
              goto LABEL_11;
          }
          else if ( v14 == -2 && !*(_QWORD *)(v11 + 16) && *(_DWORD *)(v11 + 24) == -2 )
          {
            goto LABEL_11;
          }
        }
        v16 = v12;
        sub_3794400(a1, (__int64 *)v11, v12);
        v13 = v17;
        v12 = v16;
        *(_QWORD *)v17 = *(_QWORD *)v11;
        *(_DWORD *)(v13 + 8) = *(_DWORD *)(v11 + 8);
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v11 + 16);
        *(_DWORD *)(v13 + 24) = *(_DWORD *)(v11 + 24);
        *(_DWORD *)(v17 + 32) = *(_DWORD *)(v11 + 32);
        ++*(_DWORD *)(a1 + 16);
LABEL_11:
        v11 += 40;
      }
      while ( v9 != v11 );
    }
    return sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 40LL * *(unsigned int *)(a1 + 24); j != result; result += 40 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
        *(_QWORD *)(result + 16) = 0;
        *(_DWORD *)(result + 24) = -1;
      }
    }
  }
  return result;
}
