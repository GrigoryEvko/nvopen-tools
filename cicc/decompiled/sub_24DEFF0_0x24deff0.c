// Function: sub_24DEFF0
// Address: 0x24deff0
//
_QWORD *__fastcall sub_24DEFF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r13
  _QWORD *v10; // r14
  _QWORD *i; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdi
  int v17; // r11d
  unsigned int v18; // ecx
  __int64 *v19; // r10
  __int64 *v20; // r15
  __int64 v21; // rsi
  void *v22; // rdi
  unsigned int v23; // r10d
  unsigned __int64 v24; // rdi
  _QWORD *v25; // r11
  const void *v26; // rsi
  __int64 v27; // r8
  _QWORD *j; // rdx
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  _QWORD *v30; // [rsp+8h] [rbp-48h]
  unsigned int v31; // [rsp+14h] [rbp-3Ch]
  unsigned int v32; // [rsp+14h] [rbp-3Ch]
  __int64 v33; // [rsp+18h] [rbp-38h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v33 = *(_QWORD *)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(56LL * v5, 8);
  v7 = v33;
  *(_QWORD *)(a1 + 8) = result;
  if ( v33 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 56 * v4;
    v10 = (_QWORD *)(v33 + 56 * v4);
    for ( i = &result[7 * v8]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v12 = (_QWORD *)(v33 + 24);
    if ( v10 != (_QWORD *)v33 )
    {
      while ( 1 )
      {
        v13 = *(v12 - 3);
        if ( v13 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(v12 - 3);
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v19 = 0;
          v20 = (__int64 *)(v16 + 56LL * v18);
          v21 = *v20;
          if ( v13 != *v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v19 && v21 == -8192 )
                v19 = v20;
              v18 = v15 & (v17 + v18);
              v20 = (__int64 *)(v16 + 56LL * v18);
              v21 = *v20;
              if ( v13 == *v20 )
                goto LABEL_15;
              ++v17;
            }
            if ( v19 )
              v20 = v19;
          }
LABEL_15:
          *v20 = v13;
          v22 = v20 + 3;
          v20[1] = (__int64)(v20 + 3);
          v20[2] = 0x400000000LL;
          v23 = *((_DWORD *)v12 - 2);
          if ( v23 && v20 + 1 != v12 - 2 )
          {
            v25 = (_QWORD *)*(v12 - 2);
            if ( v12 == v25 )
            {
              v26 = v12;
              v27 = 8LL * v23;
              if ( v23 <= 4 )
                goto LABEL_23;
              v30 = (_QWORD *)*(v12 - 2);
              v32 = *((_DWORD *)v12 - 2);
              v36 = v7;
              sub_C8D5F0((__int64)(v20 + 1), v20 + 3, v23, 8u, v27, v7);
              v22 = (void *)v20[1];
              v26 = (const void *)*(v12 - 2);
              v7 = v36;
              v23 = v32;
              v25 = v30;
              v27 = 8LL * *((unsigned int *)v12 - 2);
              if ( v27 )
              {
LABEL_23:
                v29 = v25;
                v31 = v23;
                v35 = v7;
                memcpy(v22, v26, v27);
                v7 = v35;
                *((_DWORD *)v20 + 4) = v31;
                *((_DWORD *)v29 - 2) = 0;
              }
              else
              {
                *((_DWORD *)v20 + 4) = v32;
                *((_DWORD *)v30 - 2) = 0;
              }
            }
            else
            {
              v20[1] = (__int64)v25;
              *((_DWORD *)v20 + 4) = *((_DWORD *)v12 - 2);
              *((_DWORD *)v20 + 5) = *((_DWORD *)v12 - 1);
              *(v12 - 2) = v12;
              *((_DWORD *)v12 - 1) = 0;
              *((_DWORD *)v12 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = *(v12 - 2);
          if ( (_QWORD *)v24 != v12 )
          {
            v34 = v7;
            _libc_free(v24);
            v7 = v34;
          }
        }
        if ( v10 == v12 + 4 )
          break;
        v12 += 7;
      }
    }
    return (_QWORD *)sub_C7D6A0(v7, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
