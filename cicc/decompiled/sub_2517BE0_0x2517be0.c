// Function: sub_2517BE0
// Address: 0x2517be0
//
_QWORD *__fastcall sub_2517BE0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // r15
  _QWORD *i; // rdx
  _QWORD *v12; // rbx
  __int64 j; // rax
  unsigned __int64 v14; // rdx
  int v15; // eax
  int v16; // ecx
  __int64 v17; // r8
  unsigned int v18; // eax
  unsigned __int64 *v19; // r9
  unsigned __int64 v20; // rsi
  __int64 v21; // rdx
  _QWORD *k; // rdx
  int v23; // r10d
  unsigned __int64 *v24; // rdi
  unsigned __int64 v25; // rax
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
    v9 = 24 * v4;
    v10 = (_QWORD *)(v5 + 24 * v4);
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = 4;
        result[1] = 0;
        result[2] = -4096;
      }
    }
    v26[0] = 4;
    v26[1] = 0;
    v27 = -4096;
    v28[0] = 4;
    v28[1] = 0;
    v29 = -8192;
    if ( v10 != (_QWORD *)v5 )
    {
      v12 = (_QWORD *)v5;
      for ( j = -4096; ; j = v27 )
      {
        v14 = v12[2];
        if ( v14 != j )
        {
          j = v29;
          if ( v14 != v29 )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
              BUG();
            v16 = v15 - 1;
            v17 = *(_QWORD *)(a1 + 8);
            v18 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = (unsigned __int64 *)(v17 + 24LL * v18);
            v20 = v19[2];
            if ( v20 != v14 )
            {
              v23 = 1;
              v24 = 0;
              while ( v20 != -4096 )
              {
                if ( v24 || v20 != -8192 )
                  v19 = v24;
                v18 = v16 & (v23 + v18);
                v20 = *(_QWORD *)(v17 + 24LL * v18 + 16);
                if ( v14 == v20 )
                  goto LABEL_15;
                ++v23;
                v24 = v19;
                v19 = (unsigned __int64 *)(v17 + 24LL * v18);
              }
              if ( v24 )
              {
                v25 = v24[2];
              }
              else
              {
                v25 = v19[2];
                v24 = v19;
              }
              if ( v14 != v25 )
              {
                if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
                {
                  sub_BD60C0(v24);
                  v14 = v12[2];
                }
                v24[2] = v14;
                if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
                  sub_BD6050(v24, *v12 & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_15:
            ++*(_DWORD *)(a1 + 16);
            j = v12[2];
          }
        }
        if ( j != -4096 && j != 0 && j != -8192 )
          sub_BD60C0(v12);
        v12 += 3;
        if ( v10 == v12 )
          break;
      }
      if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
        sub_BD60C0(v28);
      if ( v27 != -8192 && v27 != -4096 && v27 != 0 )
        sub_BD60C0(v26);
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v21]; k != result; result += 3 )
    {
      if ( result )
      {
        *result = 4;
        result[1] = 0;
        result[2] = -4096;
      }
    }
  }
  return result;
}
