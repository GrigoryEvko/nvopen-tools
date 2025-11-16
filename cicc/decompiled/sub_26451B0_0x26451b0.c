// Function: sub_26451B0
// Address: 0x26451b0
//
_QWORD *__fastcall sub_26451B0(__int64 a1, int a2)
{
  unsigned int v3; // r13d
  __int64 v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 *v9; // r8
  _QWORD *i; // rdx
  __int64 *v11; // rax
  __int64 v12; // rdx
  int v13; // edi
  int v14; // edi
  __int64 v15; // r11
  int v16; // r15d
  _QWORD *v17; // r14
  unsigned int v18; // esi
  _QWORD *v19; // rcx
  __int64 v20; // r10
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *j; // rdx

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(16LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 16LL * v3;
    v9 = (__int64 *)(v4 + v8);
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
    if ( v9 != (__int64 *)v4 )
    {
      v11 = (__int64 *)v4;
      do
      {
        while ( 1 )
        {
          v12 = *v11;
          if ( (unsigned __int64)*v11 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v11 += 2;
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v4, v8, 8);
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = v14 & (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12));
        v19 = (_QWORD *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v17 && v20 == -2 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (_QWORD *)(v15 + 16LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_14;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_14:
        *v19 = v12;
        v21 = v11[1];
        v11 += 2;
        v19[1] = v21;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v22]; j != result; result += 2 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
