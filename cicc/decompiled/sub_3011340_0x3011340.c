// Function: sub_3011340
// Address: 0x3011340
//
_QWORD *__fastcall sub_3011340(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 *v10; // rdi
  _QWORD *i; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // ecx
  int v16; // r8d
  __int64 v17; // r11
  _QWORD *v18; // rbx
  int v19; // r14d
  unsigned int v20; // r9d
  _QWORD *v21; // rcx
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rdx
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = (__int64 *)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v10 != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          v14 = *v12;
          BYTE1(v14) = BYTE1(*v12) & 0xEF;
          if ( v14 != -8192 )
            break;
          v12 += 2;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = *v12;
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 0;
        v19 = 1;
        v20 = (v15 - 1) & (37 * v13);
        v21 = (_QWORD *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( v13 != *v21 )
        {
          while ( v22 != -4096 )
          {
            if ( !v18 && v22 == -8192 )
              v18 = v21;
            v20 = v16 & (v19 + v20);
            v21 = (_QWORD *)(v17 + 16LL * v20);
            v22 = *v21;
            if ( v13 == *v21 )
              goto LABEL_14;
            ++v19;
          }
          if ( v18 )
            v21 = v18;
        }
LABEL_14:
        v23 = *v12;
        v12 += 2;
        *v21 = v23;
        v21[1] = *(v12 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v24]; j != result; result += 2 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
