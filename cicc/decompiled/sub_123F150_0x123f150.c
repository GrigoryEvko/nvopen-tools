// Function: sub_123F150
// Address: 0x123f150
//
_QWORD *__fastcall sub_123F150(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r14
  _QWORD *i; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  int v15; // r11d
  __int64 v16; // r10
  unsigned int v17; // ecx
  __int64 v18; // r15
  const void *v19; // rsi
  void *v20; // rdi
  unsigned int v21; // r10d
  _QWORD *v22; // rdi
  _QWORD *v23; // r11
  __int64 v24; // r8
  _QWORD *j; // rdx
  _QWORD *v26; // [rsp+8h] [rbp-48h]
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  unsigned int v28; // [rsp+14h] [rbp-3Ch]
  unsigned int v29; // [rsp+14h] [rbp-3Ch]
  __int64 v30; // [rsp+18h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v30 = 40 * v4;
    v8 = v5 + 40 * v4;
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (_QWORD *)(v5 + 24);
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 3);
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 3);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = v14 + 40LL * v17;
          v19 = *(const void **)v18;
          if ( v11 != *(_QWORD *)v18 )
          {
            while ( v19 != (const void *)-4096LL )
            {
              if ( !v16 && v19 == (const void *)-8192LL )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = v14 + 40LL * v17;
              v19 = *(const void **)v18;
              if ( v11 == *(_QWORD *)v18 )
                goto LABEL_15;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_15:
          *(_QWORD *)v18 = v11;
          v20 = (void *)(v18 + 24);
          *(_QWORD *)(v18 + 8) = v18 + 24;
          *(_QWORD *)(v18 + 16) = 0x200000000LL;
          v21 = *((_DWORD *)v10 - 2);
          if ( v21 && (_QWORD *)(v18 + 8) != v10 - 2 )
          {
            v23 = (_QWORD *)*(v10 - 2);
            if ( v10 == v23 )
            {
              v19 = v10;
              v24 = 8LL * v21;
              if ( v21 <= 2 )
                goto LABEL_23;
              v27 = (_QWORD *)*(v10 - 2);
              v29 = *((_DWORD *)v10 - 2);
              sub_C8D5F0(v18 + 8, (const void *)(v18 + 24), v21, 8u, v24, v18 + 8);
              v20 = *(void **)(v18 + 8);
              v19 = (const void *)*(v10 - 2);
              v21 = v29;
              v23 = v27;
              v24 = 8LL * *((unsigned int *)v10 - 2);
              if ( v24 )
              {
LABEL_23:
                v26 = v23;
                v28 = v21;
                memcpy(v20, v19, v24);
                *(_DWORD *)(v18 + 16) = v28;
                *((_DWORD *)v26 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v18 + 16) = v29;
                *((_DWORD *)v27 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v18 + 8) = v23;
              *(_DWORD *)(v18 + 16) = *((_DWORD *)v10 - 2);
              *(_DWORD *)(v18 + 20) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v22 = (_QWORD *)*(v10 - 2);
          if ( v22 != v10 )
            _libc_free(v22, v19);
        }
        if ( (_QWORD *)v8 == v10 + 2 )
          break;
        v10 += 5;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v30, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
