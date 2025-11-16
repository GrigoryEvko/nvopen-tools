// Function: sub_22ED510
// Address: 0x22ed510
//
_QWORD *__fastcall sub_22ED510(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v4; // r11
  __int64 v5; // rbx
  _QWORD *v6; // rcx
  __int64 v8; // rdi
  _QWORD *v9; // r8
  int v10; // edx
  __int64 v11; // r12
  int v12; // esi
  unsigned int v13; // edx
  __int64 v14; // r13
  int v15; // r14d

  result = (_QWORD *)*(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD **)(a1 + 8);
    v5 = *(unsigned int *)(a1 + 24);
    v6 = &v4[v5];
    if ( v4 != v6 )
    {
      result = *(_QWORD **)(a1 + 8);
      while ( 1 )
      {
        v8 = *result;
        v9 = result;
        if ( *result != -8192 && v8 != -4096 )
          break;
        if ( v6 == ++result )
          return result;
      }
      if ( v6 != result )
      {
        while ( 1 )
        {
          for ( result = v9 + 1; result != v6; ++result )
          {
            if ( *result != -8192 && *result != -4096 )
              break;
          }
          v10 = *(_DWORD *)(a2 + 24);
          v11 = *(_QWORD *)(a2 + 8);
          if ( v10 )
          {
            v12 = v10 - 1;
            v13 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v14 = *(_QWORD *)(v11 + 8LL * v13);
            if ( v14 == v8 )
              goto LABEL_15;
            v15 = 1;
            while ( v14 != -4096 )
            {
              v13 = v12 & (v15 + v13);
              v14 = *(_QWORD *)(v11 + 8LL * v13);
              if ( v14 == v8 )
                goto LABEL_15;
              ++v15;
            }
          }
          *v9 = -8192;
          v4 = *(_QWORD **)(a1 + 8);
          --*(_DWORD *)(a1 + 16);
          v5 = *(unsigned int *)(a1 + 24);
          ++*(_DWORD *)(a1 + 20);
LABEL_15:
          if ( result == &v4[v5] )
            return result;
          v8 = *result;
          v9 = result;
        }
      }
    }
  }
  return result;
}
