// Function: sub_297EA40
// Address: 0x297ea40
//
_QWORD *__fastcall sub_297EA40(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r12
  _QWORD *i; // rdx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdi
  int v17; // r11d
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 v20; // r8
  __int64 v21; // rsi
  void *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r10d
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rax
  const void *v28; // rsi
  size_t v29; // rdx
  __int64 v30; // rdx
  _QWORD *j; // rdx
  __int64 v32; // [rsp+0h] [rbp-40h]
  __int64 v33; // [rsp+0h] [rbp-40h]
  unsigned int v34; // [rsp+Ch] [rbp-34h]
  unsigned int v35; // [rsp+Ch] [rbp-34h]

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
  result = (_QWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 72 * v4;
    v10 = v5 + 72 * v4;
    for ( i = &result[9 * v8]; i != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
    v12 = (_QWORD *)(v5 + 56);
    if ( v10 != v5 )
    {
      while ( 1 )
      {
        v13 = *(v12 - 7);
        if ( v13 != -8192 && v13 != -4096 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = *(v12 - 7);
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = v16 + 72LL * v19;
          v21 = *(_QWORD *)v20;
          if ( v13 != *(_QWORD *)v20 )
          {
            while ( v21 != -4096 )
            {
              if ( !v18 && v21 == -8192 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = v16 + 72LL * v19;
              v21 = *(_QWORD *)v20;
              if ( v13 == *(_QWORD *)v20 )
                goto LABEL_15;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_15:
          *(_QWORD *)(v20 + 24) = 0;
          v22 = (void *)(v20 + 56);
          *(_QWORD *)(v20 + 16) = 0;
          *(_DWORD *)(v20 + 32) = 0;
          *(_QWORD *)v20 = v13;
          *(_QWORD *)(v20 + 8) = 1;
          v23 = *(v12 - 5);
          ++*(v12 - 6);
          v24 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(v20 + 16) = v23;
          LODWORD(v23) = *((_DWORD *)v12 - 8);
          *(v12 - 5) = v24;
          LODWORD(v24) = *(_DWORD *)(v20 + 24);
          *(_DWORD *)(v20 + 24) = v23;
          LODWORD(v23) = *((_DWORD *)v12 - 7);
          *((_DWORD *)v12 - 8) = v24;
          LODWORD(v24) = *(_DWORD *)(v20 + 28);
          *(_DWORD *)(v20 + 28) = v23;
          LODWORD(v23) = *((_DWORD *)v12 - 6);
          *((_DWORD *)v12 - 7) = v24;
          LODWORD(v24) = *(_DWORD *)(v20 + 32);
          *(_DWORD *)(v20 + 32) = v23;
          *((_DWORD *)v12 - 6) = v24;
          *(_QWORD *)(v20 + 40) = v20 + 56;
          *(_QWORD *)(v20 + 48) = 0x200000000LL;
          v25 = *((_DWORD *)v12 - 2);
          if ( v25 && (_QWORD *)(v20 + 40) != v12 - 2 )
          {
            v27 = (_QWORD *)*(v12 - 2);
            if ( v27 == v12 )
            {
              v28 = v12;
              v29 = 8LL * v25;
              if ( v25 <= 2 )
                goto LABEL_24;
              v33 = v20;
              v35 = *((_DWORD *)v12 - 2);
              sub_C8D5F0(v20 + 40, (const void *)(v20 + 56), v25, 8u, v20, v25);
              v20 = v33;
              v28 = (const void *)*(v12 - 2);
              v25 = v35;
              v29 = 8LL * *((unsigned int *)v12 - 2);
              v22 = *(void **)(v33 + 40);
              if ( v29 )
              {
LABEL_24:
                v32 = v20;
                v34 = v25;
                memcpy(v22, v28, v29);
                *(_DWORD *)(v32 + 48) = v34;
                *((_DWORD *)v12 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v33 + 48) = v35;
                *((_DWORD *)v12 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v20 + 40) = v27;
              *(_DWORD *)(v20 + 48) = *((_DWORD *)v12 - 2);
              *(_DWORD *)(v20 + 52) = *((_DWORD *)v12 - 1);
              *(v12 - 2) = v12;
              *((_DWORD *)v12 - 1) = 0;
              *((_DWORD *)v12 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v26 = *(v12 - 2);
          if ( (_QWORD *)v26 != v12 )
            _libc_free(v26);
          sub_C7D6A0(*(v12 - 5), 8LL * *((unsigned int *)v12 - 6), 8);
        }
        if ( (_QWORD *)v10 == v12 + 2 )
          break;
        v12 += 9;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[9 * v30]; j != result; result += 9 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
