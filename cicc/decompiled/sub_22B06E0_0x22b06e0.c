// Function: sub_22B06E0
// Address: 0x22b06e0
//
_DWORD *__fastcall sub_22B06E0(__int64 a1, __int64 a2)
{
  _DWORD *result; // rax
  _DWORD *v3; // rbx
  __int64 v4; // r12
  _DWORD *v5; // rdx
  _DWORD *v7; // r8
  int v8; // ecx
  __int64 v9; // r13
  int v10; // esi
  unsigned int v11; // ecx
  int v12; // r11d
  int v13; // r14d

  result = (_DWORD *)*(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v3 = *(_DWORD **)(a1 + 8);
    v4 = *(unsigned int *)(a1 + 24);
    v5 = &v3[v4];
    if ( v3 != v5 )
    {
      result = *(_DWORD **)(a1 + 8);
      while ( 1 )
      {
        v7 = result;
        if ( *result <= 0xFFFFFFFD )
          break;
        if ( v5 == ++result )
          return result;
      }
      if ( v5 != result )
      {
        while ( 1 )
        {
          for ( result = v7 + 1; v5 != result; ++result )
          {
            if ( *result <= 0xFFFFFFFD )
              break;
          }
          v8 = *(_DWORD *)(a2 + 24);
          v9 = *(_QWORD *)(a2 + 8);
          if ( v8 )
          {
            v10 = v8 - 1;
            v11 = (v8 - 1) & (37 * *v7);
            v12 = *(_DWORD *)(v9 + 4LL * (v10 & (unsigned int)(37 * *v7)));
            if ( *v7 == v12 )
              goto LABEL_13;
            v13 = 1;
            while ( v12 != -1 )
            {
              v11 = v10 & (v13 + v11);
              v12 = *(_DWORD *)(v9 + 4LL * v11);
              if ( *v7 == v12 )
                goto LABEL_13;
              ++v13;
            }
          }
          *v7 = -2;
          v3 = *(_DWORD **)(a1 + 8);
          --*(_DWORD *)(a1 + 16);
          v4 = *(unsigned int *)(a1 + 24);
          ++*(_DWORD *)(a1 + 20);
LABEL_13:
          if ( result == &v3[v4] )
            return result;
          v7 = result;
        }
      }
    }
  }
  return result;
}
