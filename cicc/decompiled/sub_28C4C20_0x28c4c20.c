// Function: sub_28C4C20
// Address: 0x28c4c20
//
_QWORD *__fastcall sub_28C4C20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _QWORD *i; // rdx
  __int64 v11; // r15
  __int64 v12; // rdx
  int v13; // esi
  int v14; // esi
  __int64 v15; // r9
  int v16; // r11d
  __int64 v17; // r10
  unsigned int v18; // edi
  __int64 v19; // rcx
  __int64 v20; // r14
  __int64 v21; // r8
  unsigned __int64 *v22; // rdx
  unsigned int v23; // r13d
  _QWORD *v24; // r13
  _QWORD *v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rsi
  _QWORD *v29; // r8
  _QWORD *j; // r10
  unsigned __int64 v31; // rsi
  _QWORD *v32; // r14
  _QWORD *v33; // r13
  __int64 v34; // rcx
  __int64 v35; // rdx
  _QWORD *k; // rdx
  _QWORD *v37; // [rsp+8h] [rbp-58h]
  _QWORD *v38; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v41 = v5;
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
  result = (_QWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v40 = 72 * v4;
    v9 = 72 * v4 + v5;
    for ( i = &result[9 * v8]; i != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = v5 + 24;
    if ( v9 != v41 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v11 - 24);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(_QWORD *)(v11 - 24);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = 9LL * v18;
          v20 = v15 + 72LL * v18;
          v21 = *(_QWORD *)v20;
          if ( v12 != *(_QWORD *)v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v17 && v21 == -8192 )
                v17 = v20;
              v18 = v14 & (v16 + v18);
              v19 = 9LL * v18;
              v20 = v15 + 72LL * v18;
              v21 = *(_QWORD *)v20;
              if ( v12 == *(_QWORD *)v20 )
                goto LABEL_13;
              ++v16;
            }
            if ( v17 )
              v20 = v17;
          }
LABEL_13:
          *(_QWORD *)v20 = v12;
          v22 = (unsigned __int64 *)(v20 + 24);
          *(_QWORD *)(v20 + 8) = v20 + 24;
          *(_QWORD *)(v20 + 16) = 0x200000000LL;
          v23 = *(_DWORD *)(v11 - 8);
          if ( v20 + 8 != v11 - 16 && v23 )
          {
            v27 = *(_QWORD *)(v11 - 16);
            if ( v27 == v11 )
            {
              v28 = v23;
              v29 = (_QWORD *)v11;
              if ( v23 > 2 )
              {
                sub_F39130(v20 + 8, v23, (__int64)v22, v19, v11, v15);
                v22 = *(unsigned __int64 **)(v20 + 8);
                v29 = *(_QWORD **)(v11 - 16);
                v28 = *(unsigned int *)(v11 - 8);
              }
              for ( j = &v29[3 * v28]; j != v29; v22 += 3 )
              {
                if ( v22 )
                {
                  *v22 = 6;
                  v22[1] = 0;
                  v31 = v29[2];
                  v22[2] = v31;
                  if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
                  {
                    v37 = j;
                    v38 = v29;
                    v39 = v22;
                    sub_BD6050(v22, *v29 & 0xFFFFFFFFFFFFFFF8LL);
                    j = v37;
                    v29 = v38;
                    v22 = v39;
                  }
                }
                v29 += 3;
              }
              *(_DWORD *)(v20 + 16) = v23;
              v32 = *(_QWORD **)(v11 - 16);
              v33 = &v32[3 * *(unsigned int *)(v11 - 8)];
              while ( v32 != v33 )
              {
                v34 = *(v33 - 1);
                v33 -= 3;
                if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
                  sub_BD60C0(v33);
              }
              *(_DWORD *)(v11 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v20 + 8) = v27;
              *(_DWORD *)(v20 + 16) = *(_DWORD *)(v11 - 8);
              *(_DWORD *)(v20 + 20) = *(_DWORD *)(v11 - 4);
              *(_QWORD *)(v11 - 16) = v11;
              *(_DWORD *)(v11 - 4) = 0;
              *(_DWORD *)(v11 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = *(_QWORD **)(v11 - 16);
          v25 = &v24[3 * *(unsigned int *)(v11 - 8)];
          if ( v24 != v25 )
          {
            do
            {
              v26 = *(v25 - 1);
              v25 -= 3;
              if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
                sub_BD60C0(v25);
            }
            while ( v24 != v25 );
            v25 = *(_QWORD **)(v11 - 16);
          }
          if ( (_QWORD *)v11 != v25 )
            _libc_free((unsigned __int64)v25);
        }
        if ( v9 == v11 + 48 )
          break;
        v11 += 72;
      }
    }
    return (_QWORD *)sub_C7D6A0(v41, v40, 8);
  }
  else
  {
    v35 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[9 * v35]; k != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
