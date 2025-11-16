// Function: sub_FFA970
// Address: 0xffa970
//
_QWORD *__fastcall sub_FFA970(__int64 a1, int a2)
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
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 v20; // r12
  const void *v21; // rsi
  void *v22; // rdi
  unsigned int v23; // r10d
  _QWORD *v24; // rdi
  _QWORD *v25; // r11
  size_t v26; // r8
  __int64 v27; // rdx
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
  result = (_QWORD *)sub_C7D670(32LL * v5, 8);
  v7 = v33;
  *(_QWORD *)(a1 + 8) = result;
  if ( v33 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (_QWORD *)(v33 + v9);
    for ( i = &result[4 * v8]; i != result; result += 4 )
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
          v18 = 0;
          v19 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = v16 + 32LL * v19;
          v21 = *(const void **)v20;
          if ( v13 != *(_QWORD *)v20 )
          {
            while ( v21 != (const void *)-4096LL )
            {
              if ( !v18 && v21 == (const void *)-8192LL )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = v16 + 32LL * v19;
              v21 = *(const void **)v20;
              if ( v13 == *(_QWORD *)v20 )
                goto LABEL_15;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_15:
          *(_QWORD *)v20 = v13;
          v22 = (void *)(v20 + 24);
          *(_QWORD *)(v20 + 8) = v20 + 24;
          *(_QWORD *)(v20 + 16) = 0x100000000LL;
          v23 = *((_DWORD *)v12 - 2);
          if ( v23 && (_QWORD *)(v20 + 8) != v12 - 2 )
          {
            v25 = (_QWORD *)*(v12 - 2);
            if ( v12 == v25 )
            {
              v21 = v12;
              v26 = 8;
              if ( v23 == 1 )
                goto LABEL_23;
              v30 = (_QWORD *)*(v12 - 2);
              v32 = *((_DWORD *)v12 - 2);
              v36 = v7;
              sub_C8D5F0(v20 + 8, (const void *)(v20 + 24), v23, 8u, 8, v7);
              v22 = *(void **)(v20 + 8);
              v21 = (const void *)*(v12 - 2);
              v7 = v36;
              v23 = v32;
              v25 = v30;
              v26 = 8LL * *((unsigned int *)v12 - 2);
              if ( v26 )
              {
LABEL_23:
                v29 = v25;
                v31 = v23;
                v35 = v7;
                memcpy(v22, v21, v26);
                v7 = v35;
                *(_DWORD *)(v20 + 16) = v31;
                *((_DWORD *)v29 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v20 + 16) = v32;
                *((_DWORD *)v30 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v20 + 8) = v25;
              *(_DWORD *)(v20 + 16) = *((_DWORD *)v12 - 2);
              *(_DWORD *)(v20 + 20) = *((_DWORD *)v12 - 1);
              *(v12 - 2) = v12;
              *((_DWORD *)v12 - 1) = 0;
              *((_DWORD *)v12 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = (_QWORD *)*(v12 - 2);
          if ( v12 != v24 )
          {
            v34 = v7;
            _libc_free(v24, v21);
            v7 = v34;
          }
        }
        if ( v10 == v12 + 1 )
          break;
        v12 += 4;
      }
    }
    return (_QWORD *)sub_C7D6A0(v7, v9, 8);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v27]; j != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
