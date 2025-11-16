// Function: sub_15CDC00
// Address: 0x15cdc00
//
__int64 __fastcall sub_15CDC00(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax
  __int64 *v4; // r8
  __int64 *v5; // rdx
  __int64 *v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // r13
  __int64 v10; // r12
  __int64 *v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx

  v1 = *(unsigned int *)(a1 + 308);
  result = 0;
  if ( *(_DWORD *)(a1 + 312) != (_DWORD)v1 )
  {
    v4 = *(__int64 **)(a1 + 296);
    v5 = *(__int64 **)(a1 + 288);
    v6 = &v4[v1];
    if ( v4 != v5 )
      v6 = &v4[*(unsigned int *)(a1 + 304)];
    if ( v4 == v6 )
    {
LABEL_8:
      v10 = a1 + 280;
    }
    else
    {
      v7 = *(__int64 **)(a1 + 296);
      while ( 1 )
      {
        v8 = *v7;
        v9 = v7;
        if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v6 == ++v7 )
          goto LABEL_8;
      }
      v10 = a1 + 280;
      if ( v6 != v7 )
      {
        do
        {
          sub_157F980(v8);
          v11 = v9 + 1;
          if ( v9 + 1 == v6 )
            break;
          while ( 1 )
          {
            v8 = *v11;
            v9 = v11;
            if ( (unsigned __int64)*v11 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v6 == ++v11 )
              goto LABEL_13;
          }
        }
        while ( v11 != v6 );
LABEL_13:
        v4 = *(__int64 **)(a1 + 296);
        v5 = *(__int64 **)(a1 + 288);
      }
    }
    ++*(_QWORD *)(a1 + 280);
    if ( v4 != v5 )
    {
      v12 = 4 * (*(_DWORD *)(a1 + 308) - *(_DWORD *)(a1 + 312));
      v13 = *(unsigned int *)(a1 + 304);
      if ( v12 < 0x20 )
        v12 = 32;
      if ( (unsigned int)v13 > v12 )
      {
        sub_16CC920(v10);
        return 1;
      }
      memset(v4, -1, 8 * v13);
    }
    *(_QWORD *)(a1 + 308) = 0;
    return 1;
  }
  return result;
}
