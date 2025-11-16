// Function: sub_2C04490
// Address: 0x2c04490
//
_QWORD *__fastcall sub_2C04490(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  unsigned int v5; // edi
  _QWORD *result; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // r14
  _QWORD *i; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdi
  int v17; // r11d
  __int64 *v18; // r10
  unsigned int v19; // ecx
  __int64 *v20; // r12
  __int64 v21; // rsi
  void *v22; // rdi
  __int64 v23; // rax
  unsigned int v24; // r10d
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r11
  const void *v27; // rsi
  __int64 v28; // r8
  __int64 v29; // rdx
  _QWORD *j; // rdx
  _QWORD *v31; // [rsp+8h] [rbp-48h]
  _QWORD *v32; // [rsp+8h] [rbp-48h]
  unsigned int v33; // [rsp+14h] [rbp-3Ch]
  unsigned int v34; // [rsp+14h] [rbp-3Ch]
  __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v35 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670((unsigned __int64)v5 << 6, 8);
  v7 = v35;
  *(_QWORD *)(a1 + 8) = result;
  if ( v35 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v4 << 6;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (_QWORD *)(v35 + v9);
    for ( i = &result[8 * v8]; i != result; result += 8 )
    {
      if ( result )
        *result = -4096;
    }
    v12 = (_QWORD *)(v35 + 48);
    if ( v10 != (_QWORD *)v35 )
    {
      while ( 1 )
      {
        v13 = *(v12 - 6);
        if ( v13 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(v12 - 6);
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = (__int64 *)(v16 + ((unsigned __int64)v19 << 6));
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (__int64 *)(v16 + ((unsigned __int64)v19 << 6));
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_15;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_15:
          *v20 = v13;
          v22 = v20 + 6;
          *((_DWORD *)v20 + 2) = *((_DWORD *)v12 - 10);
          *((_DWORD *)v20 + 3) = *((_DWORD *)v12 - 9);
          *((_DWORD *)v20 + 4) = *((_DWORD *)v12 - 8);
          *((_DWORD *)v20 + 5) = *((_DWORD *)v12 - 7);
          v23 = *(v12 - 3);
          v20[4] = (__int64)(v20 + 6);
          v20[3] = v23;
          v20[5] = 0x400000000LL;
          v24 = *((_DWORD *)v12 - 2);
          if ( v24 && v20 + 4 != v12 - 2 )
          {
            v26 = (_QWORD *)*(v12 - 2);
            if ( v12 == v26 )
            {
              v27 = v12;
              v28 = 4LL * v24;
              if ( v24 <= 4 )
                goto LABEL_23;
              v32 = (_QWORD *)*(v12 - 2);
              v34 = *((_DWORD *)v12 - 2);
              v38 = v7;
              sub_C8D5F0((__int64)(v20 + 4), v20 + 6, v24, 4u, v28, v7);
              v22 = (void *)v20[4];
              v27 = (const void *)*(v12 - 2);
              v7 = v38;
              v24 = v34;
              v26 = v32;
              v28 = 4LL * *((unsigned int *)v12 - 2);
              if ( v28 )
              {
LABEL_23:
                v31 = v26;
                v33 = v24;
                v37 = v7;
                memcpy(v22, v27, v28);
                v7 = v37;
                *((_DWORD *)v20 + 10) = v33;
                *((_DWORD *)v31 - 2) = 0;
              }
              else
              {
                *((_DWORD *)v20 + 10) = v34;
                *((_DWORD *)v32 - 2) = 0;
              }
            }
            else
            {
              v20[4] = (__int64)v26;
              *((_DWORD *)v20 + 10) = *((_DWORD *)v12 - 2);
              *((_DWORD *)v20 + 11) = *((_DWORD *)v12 - 1);
              *(v12 - 2) = v12;
              *((_DWORD *)v12 - 1) = 0;
              *((_DWORD *)v12 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v25 = *(v12 - 2);
          if ( (_QWORD *)v25 != v12 )
          {
            v36 = v7;
            _libc_free(v25);
            v7 = v36;
          }
        }
        if ( v10 == v12 + 2 )
          break;
        v12 += 8;
      }
    }
    return (_QWORD *)sub_C7D6A0(v7, v9, 8);
  }
  else
  {
    v29 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v29]; j != result; result += 8 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
