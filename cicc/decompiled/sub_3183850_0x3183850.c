// Function: sub_3183850
// Address: 0x3183850
//
_QWORD *__fastcall sub_3183850(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rsi
  __int64 *v9; // r14
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // esi
  __int64 v15; // r8
  int v16; // r10d
  unsigned int v17; // edx
  _QWORD *v18; // r9
  _QWORD *v19; // r15
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  _QWORD *j; // rdx
  __int64 v27; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 56 * v4;
    v9 = (__int64 *)(v5 + 56 * v4);
    for ( i = &result[7 * v8]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    if ( v9 != (__int64 *)v5 )
    {
      v11 = (__int64 *)v5;
      do
      {
        v12 = *v11;
        if ( *v11 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *v11;
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v18 = 0;
          v19 = (_QWORD *)(v15 + 56LL * v17);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v18 && v20 == -8192 )
                v18 = v19;
              v17 = v14 & (v16 + v17);
              v19 = (_QWORD *)(v15 + 56LL * v17);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_14;
              ++v16;
            }
            if ( v18 )
              v19 = v18;
          }
LABEL_14:
          *v19 = v12;
          v19[1] = 4;
          v19[2] = 0;
          v21 = v11[3];
          v19[3] = v21;
          if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
            sub_BD6050(v19 + 1, v11[1] & 0xFFFFFFFFFFFFFFF8LL);
          v19[4] = 6;
          v19[5] = 0;
          v22 = v11[6];
          v19[6] = v22;
          if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
            sub_BD6050(v19 + 4, v11[4] & 0xFFFFFFFFFFFFFFF8LL);
          ++*(_DWORD *)(a1 + 16);
          v23 = v11[6];
          if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
            sub_BD60C0(v11 + 4);
          v24 = v11[3];
          if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
            sub_BD60C0(v11 + 1);
        }
        v11 += 7;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v27, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v25]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
