// Function: sub_2D2D570
// Address: 0x2d2d570
//
_QWORD *__fastcall sub_2D2D570(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 v8; // r13
  _QWORD *i; // rdx
  __int64 j; // r12
  __int64 v11; // rdx
  int v12; // eax
  int v13; // edi
  unsigned __int64 v14; // r8
  int v15; // r11d
  unsigned __int64 v16; // r10
  unsigned int v17; // esi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rax
  _DWORD *v23; // rbx
  _DWORD *v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *k; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h]
  _DWORD *v29; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = 40 * v3;
    v28 = 40 * v3;
    v8 = v4 + 40 * v3;
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    for ( j = v4; v8 != j; j += 40 )
    {
      v11 = *(_QWORD *)j;
      if ( *(_QWORD *)j != -8192 && v11 != -4096 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = v14 + 40LL * v17;
        v19 = *(_QWORD *)v18;
        if ( v11 != *(_QWORD *)v18 )
        {
          while ( v19 != -4096 )
          {
            if ( !v16 && v19 == -8192 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = v14 + 40LL * v17;
            v19 = *(_QWORD *)v18;
            if ( v11 == *(_QWORD *)v18 )
              goto LABEL_13;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_13:
        *(_QWORD *)(v18 + 24) = 0;
        *(_QWORD *)(v18 + 16) = 0;
        *(_DWORD *)(v18 + 32) = 0;
        *(_QWORD *)v18 = v11;
        *(_QWORD *)(v18 + 8) = 1;
        v20 = *(_QWORD *)(j + 16);
        ++*(_QWORD *)(j + 8);
        v21 = *(_QWORD *)(v18 + 16);
        *(_QWORD *)(v18 + 16) = v20;
        LODWORD(v20) = *(_DWORD *)(j + 24);
        *(_QWORD *)(j + 16) = v21;
        LODWORD(v21) = *(_DWORD *)(v18 + 24);
        *(_DWORD *)(v18 + 24) = v20;
        LODWORD(v20) = *(_DWORD *)(j + 28);
        *(_DWORD *)(j + 24) = v21;
        LODWORD(v21) = *(_DWORD *)(v18 + 28);
        *(_DWORD *)(v18 + 28) = v20;
        LODWORD(v20) = *(_DWORD *)(j + 32);
        *(_DWORD *)(j + 28) = v21;
        LODWORD(v21) = *(_DWORD *)(v18 + 32);
        *(_DWORD *)(v18 + 32) = v20;
        *(_DWORD *)(j + 32) = v21;
        ++*(_DWORD *)(a1 + 16);
        v22 = *(unsigned int *)(j + 32);
        if ( (_DWORD)v22 )
        {
          v23 = *(_DWORD **)(j + 16);
          v24 = &v23[54 * v22];
          do
          {
            while ( *v23 > 0xFFFFFFFD || !v23[50] )
            {
              v23 += 54;
              if ( v24 == v23 )
                goto LABEL_19;
            }
            v25 = (__int64)(v23 + 2);
            v29 = v24;
            v23 += 54;
            sub_2D2A3E0(v25, (char *)sub_2D227B0, 0, v7, v14, v19);
            v24 = v29;
          }
          while ( v29 != v23 );
LABEL_19:
          v22 = *(unsigned int *)(j + 32);
        }
        sub_C7D6A0(*(_QWORD *)(j + 16), 216 * v22, 8);
      }
    }
    return (_QWORD *)sub_C7D6A0(v4, v28, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[5 * v26]; k != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
