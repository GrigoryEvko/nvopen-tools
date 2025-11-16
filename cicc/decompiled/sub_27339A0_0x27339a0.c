// Function: sub_27339A0
// Address: 0x27339a0
//
__int64 __fastcall sub_27339A0(__int64 a1, int a2)
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
  __int64 v13; // rdi
  int v14; // r8d
  unsigned __int8 v15; // r10
  int v16; // r8d
  __int64 v17; // r11
  int v18; // r14d
  __int64 v19; // r15
  unsigned int j; // ebx
  __int64 v21; // rdx
  unsigned int v22; // edx
  __int64 v23; // rdx
  __int64 k; // rdx

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
        *(_BYTE *)(result + 8) = 0;
        *(_BYTE *)(result + 9) = 0;
      }
    }
    if ( v10 == v5 )
      return sub_C7D6A0(v5, v9, 8);
    v12 = v5;
    while ( 1 )
    {
      v13 = *(_QWORD *)v12;
      if ( !*(_QWORD *)v12 )
        goto LABEL_22;
      v14 = *(_DWORD *)(a1 + 24);
      if ( !v14 )
      {
        MEMORY[0] = *(_QWORD *)v12;
        MEMORY[8] = *(_WORD *)(v12 + 8);
        BUG();
      }
      v15 = *(_BYTE *)(v12 + 8);
      v16 = v14 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = 1;
      v19 = 0;
      for ( j = v16 & (v15 ^ ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)); ; j = v16 & v22 )
      {
        v21 = v17 + 16LL * j;
        if ( v13 != *(_QWORD *)v21 )
          break;
        if ( v15 == *(_BYTE *)(v21 + 8) )
          goto LABEL_21;
LABEL_25:
        v22 = j + v18++;
      }
      if ( *(_QWORD *)v21 )
        goto LABEL_25;
      if ( *(_BYTE *)(v21 + 8) )
        break;
      if ( v19 )
        v21 = v19;
LABEL_21:
      *(_QWORD *)v21 = *(_QWORD *)v12;
      *(_WORD *)(v21 + 8) = *(_WORD *)(v12 + 8);
      ++*(_DWORD *)(a1 + 16);
LABEL_22:
      v12 += 16;
      if ( v10 == v12 )
        return sub_C7D6A0(v5, v9, 8);
    }
    if ( !v19 )
      v19 = v17 + 16LL * j;
    goto LABEL_25;
  }
  v23 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( k = result + 16 * v23; k != result; result += 16 )
  {
    if ( result )
    {
      *(_QWORD *)result = 0;
      *(_BYTE *)(result + 8) = 0;
      *(_BYTE *)(result + 9) = 0;
    }
  }
  return result;
}
