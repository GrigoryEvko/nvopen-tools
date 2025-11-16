// Function: sub_28F1640
// Address: 0x28f1640
//
_QWORD *__fastcall sub_28F1640(__int64 a1, int a2)
{
  unsigned __int64 v2; // rcx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rbx
  _QWORD *i; // rcx
  _QWORD *v11; // r14
  __int64 j; // rax
  __int64 v13; // r15
  int v14; // eax
  int v15; // ecx
  __int64 v16; // r8
  unsigned int v17; // eax
  _QWORD *v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rdx
  _QWORD *k; // rdx
  int v22; // r10d
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-78h]
  _QWORD v26[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v27; // [rsp+20h] [rbp-60h]
  _QWORD v28[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+40h] [rbp-40h]

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
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 24 * v4;
    v9 = (_QWORD *)(v5 + 24 * v4);
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
    v26[0] = 0;
    v26[1] = 0;
    v27 = -4096;
    v28[0] = 0;
    v28[1] = 0;
    v29 = -8192;
    if ( v9 != (_QWORD *)v5 )
    {
      v11 = (_QWORD *)v5;
      for ( j = -4096; ; j = v27 )
      {
        v13 = v11[2];
        if ( v13 != j )
        {
          j = v29;
          if ( v13 != v29 )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
              BUG();
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = (_QWORD *)(v16 + 24LL * v17);
            v19 = v18[2];
            if ( v13 != v19 )
            {
              v22 = 1;
              v23 = 0;
              while ( v19 != -4096 )
              {
                if ( v19 != -8192 || v23 )
                  v18 = v23;
                v17 = v15 & (v22 + v17);
                v19 = *(_QWORD *)(v16 + 24LL * v17 + 16);
                if ( v13 == v19 )
                  goto LABEL_15;
                ++v22;
                v23 = v18;
                v18 = (_QWORD *)(v16 + 24LL * v17);
              }
              if ( v23 )
              {
                v24 = v23[2];
              }
              else
              {
                v24 = v18[2];
                v23 = v18;
              }
              if ( v13 != v24 )
              {
                if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
                  sub_BD60C0(v23);
                v23[2] = v13;
                if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
                  sub_BD73F0((__int64)v23);
              }
            }
LABEL_15:
            ++*(_DWORD *)(a1 + 16);
            j = v11[2];
          }
        }
        if ( j != -4096 && j != 0 && j != -8192 )
          sub_BD60C0(v11);
        v11 += 3;
        if ( v9 == v11 )
          break;
      }
      if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
        sub_BD60C0(v28);
      if ( v27 != -8192 && v27 != -4096 && v27 != 0 )
        sub_BD60C0(v26);
    }
    return (_QWORD *)sub_C7D6A0(v5, v25, 8);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v20]; k != result; result += 3 )
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
