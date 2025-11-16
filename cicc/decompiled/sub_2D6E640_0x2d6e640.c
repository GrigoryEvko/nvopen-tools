// Function: sub_2D6E640
// Address: 0x2d6e640
//
_QWORD *__fastcall sub_2D6E640(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 v10; // rbx
  _QWORD *i; // rcx
  __int64 v12; // r14
  __int64 j; // rax
  __int64 v14; // r15
  int v15; // eax
  int v16; // esi
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // r8
  _QWORD *k; // rdx
  int v22; // r10d
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // [rsp+0h] [rbp-80h]
  __int64 v26; // [rsp+0h] [rbp-80h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  _QWORD v28[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v29; // [rsp+20h] [rbp-60h]
  _QWORD v30[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+40h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v27 = v4;
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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v5;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v4 + v9;
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
    v28[0] = 0;
    v28[1] = 0;
    v29 = -4096;
    v30[0] = 0;
    v30[1] = 0;
    v31 = -8192;
    if ( v10 != v27 )
    {
      v12 = v27;
      for ( j = -4096; ; j = v29 )
      {
        v14 = *(_QWORD *)(v12 + 16);
        if ( v14 != j )
        {
          j = v31;
          if ( v14 != v31 )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
              BUG();
            v16 = v15 - 1;
            v17 = *(_QWORD *)(a1 + 8);
            v18 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = v17 + 32LL * v18;
            v20 = *(_QWORD *)(v19 + 16);
            if ( v14 != v20 )
            {
              v22 = 1;
              v23 = 0;
              while ( v20 != -4096 )
              {
                if ( v20 == -8192 && !v23 )
                  v23 = v19;
                v18 = v16 & (v22 + v18);
                v19 = v17 + 32LL * v18;
                v20 = *(_QWORD *)(v19 + 16);
                if ( v14 == v20 )
                  goto LABEL_15;
                ++v22;
              }
              if ( v23 )
              {
                v24 = *(_QWORD *)(v23 + 16);
                v19 = v23;
              }
              else
              {
                v24 = *(_QWORD *)(v19 + 16);
              }
              if ( v14 != v24 )
              {
                if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
                {
                  v25 = v19;
                  sub_BD60C0((_QWORD *)v19);
                  v19 = v25;
                }
                *(_QWORD *)(v19 + 16) = v14;
                if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
                {
                  v26 = v19;
                  sub_BD73F0(v19);
                  v19 = v26;
                }
              }
            }
LABEL_15:
            *(_DWORD *)(v19 + 24) = *(_DWORD *)(v12 + 24);
            ++*(_DWORD *)(a1 + 16);
            j = *(_QWORD *)(v12 + 16);
          }
        }
        if ( j != -4096 && j != 0 && j != -8192 )
          sub_BD60C0((_QWORD *)v12);
        v12 += 32;
        if ( v10 == v12 )
          break;
      }
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v30);
      if ( v29 != -8192 && v29 != -4096 && v29 != 0 )
        sub_BD60C0(v28);
    }
    return (_QWORD *)sub_C7D6A0(v27, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * *(unsigned int *)(a1 + 24)]; k != result; result += 4 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
  }
  return result;
}
