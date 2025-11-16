// Function: sub_B998D0
// Address: 0xb998d0
//
_QWORD *__fastcall sub_B998D0(__int64 a1, int a2)
{
  __int64 v3; // r13
  unsigned int v4; // eax
  _QWORD *result; // rax
  _QWORD *v6; // r9
  __int64 v7; // rdx
  __int64 v8; // r13
  _QWORD *v9; // r14
  _QWORD *i; // rdx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdi
  int v16; // r11d
  __int64 v17; // r10
  unsigned int v18; // ecx
  __int64 v19; // r12
  const void *v20; // rsi
  void *v21; // rdi
  unsigned int v22; // r10d
  _QWORD *v23; // rdi
  _QWORD *v24; // r11
  size_t v25; // r8
  __int64 v26; // rdx
  _QWORD *j; // rdx
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  unsigned int v30; // [rsp+14h] [rbp-3Ch]
  unsigned int v31; // [rsp+14h] [rbp-3Ch]
  _QWORD *v32; // [rsp+18h] [rbp-38h]
  _QWORD *v33; // [rsp+18h] [rbp-38h]
  _QWORD *v34; // [rsp+18h] [rbp-38h]
  _QWORD *v35; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v32 = *(_QWORD **)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_C7D670(32LL * v4, 8);
  v6 = v32;
  *(_QWORD *)(a1 + 8) = result;
  if ( v32 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = 4 * v3;
    *(_QWORD *)(a1 + 16) = 0;
    v9 = &v32[v8];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = v32 + 3;
    if ( v9 != v32 )
    {
      while ( 1 )
      {
        v12 = *(v11 - 3);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(v11 - 3);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = v15 + 32LL * v18;
          v20 = *(const void **)v19;
          if ( v12 != *(_QWORD *)v19 )
          {
            while ( v20 != (const void *)-4096LL )
            {
              if ( !v17 && v20 == (const void *)-8192LL )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = v15 + 32LL * v18;
              v20 = *(const void **)v19;
              if ( v12 == *(_QWORD *)v19 )
                goto LABEL_15;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_15:
          *(_QWORD *)v19 = v12;
          v21 = (void *)(v19 + 24);
          *(_QWORD *)(v19 + 8) = v19 + 24;
          *(_QWORD *)(v19 + 16) = 0x100000000LL;
          v22 = *((_DWORD *)v11 - 2);
          if ( v22 && (_QWORD *)(v19 + 8) != v11 - 2 )
          {
            v24 = (_QWORD *)*(v11 - 2);
            if ( v11 == v24 )
            {
              v20 = v11;
              v25 = 8;
              if ( v22 == 1 )
                goto LABEL_23;
              v29 = (_QWORD *)*(v11 - 2);
              v31 = *((_DWORD *)v11 - 2);
              v35 = v6;
              sub_C8D5F0(v19 + 8, v19 + 24, v22, 8);
              v21 = *(void **)(v19 + 8);
              v20 = (const void *)*(v11 - 2);
              v6 = v35;
              v22 = v31;
              v24 = v29;
              v25 = 8LL * *((unsigned int *)v11 - 2);
              if ( v25 )
              {
LABEL_23:
                v28 = v24;
                v30 = v22;
                v34 = v6;
                memcpy(v21, v20, v25);
                v6 = v34;
                *(_DWORD *)(v19 + 16) = v30;
                *((_DWORD *)v28 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v19 + 16) = v31;
                *((_DWORD *)v29 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v19 + 8) = v24;
              *(_DWORD *)(v19 + 16) = *((_DWORD *)v11 - 2);
              *(_DWORD *)(v19 + 20) = *((_DWORD *)v11 - 1);
              *(v11 - 2) = v11;
              *((_DWORD *)v11 - 1) = 0;
              *((_DWORD *)v11 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v23 = (_QWORD *)*(v11 - 2);
          if ( v11 != v23 )
          {
            v33 = v6;
            _libc_free(v23, v20);
            v6 = v33;
          }
        }
        if ( v9 == v11 + 1 )
          break;
        v11 += 4;
      }
    }
    return (_QWORD *)sub_C7D6A0(v6, v8 * 8, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v26]; j != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
