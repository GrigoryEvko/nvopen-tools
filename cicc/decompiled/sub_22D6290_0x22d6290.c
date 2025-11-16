// Function: sub_22D6290
// Address: 0x22d6290
//
_DWORD *__fastcall sub_22D6290(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  _DWORD *i; // rdx
  _DWORD *v11; // rbx
  char *v12; // rax
  unsigned int v13; // eax
  int v14; // edx
  int v15; // edx
  __int64 v16; // rdi
  int v17; // r11d
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 v20; // r8
  int v21; // esi
  void *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r10d
  unsigned __int64 v26; // rdi
  _DWORD *v27; // rax
  const void *v28; // rsi
  size_t v29; // rdx
  __int64 v30; // rdx
  _DWORD *j; // rdx
  unsigned int v32; // [rsp+4h] [rbp-3Ch]
  unsigned int v33; // [rsp+4h] [rbp-3Ch]
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 88 * v4;
    v9 = v5 + 88 * v4;
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
    v11 = (_DWORD *)(v5 + 56);
    if ( v9 != v5 )
    {
      while ( 1 )
      {
        v13 = *(v11 - 14);
        if ( v13 > 0xFFFFFFFD )
        {
          v12 = (char *)(v11 + 22);
          if ( (_DWORD *)v9 == v11 + 8 )
            return (_DWORD *)sub_C7D6A0(v5, v8, 8);
        }
        else
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = v15 & (37 * v13);
          v20 = v16 + 88LL * v19;
          v21 = *(_DWORD *)v20;
          if ( v13 != *(_DWORD *)v20 )
          {
            while ( v21 != -1 )
            {
              if ( !v18 && v21 == -2 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = v16 + 88LL * v19;
              v21 = *(_DWORD *)v20;
              if ( v13 == *(_DWORD *)v20 )
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
          *(_DWORD *)v20 = v13;
          *(_QWORD *)(v20 + 8) = 1;
          v23 = *((_QWORD *)v11 - 5);
          ++*((_QWORD *)v11 - 6);
          v24 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(v20 + 16) = v23;
          LODWORD(v23) = *(v11 - 8);
          *((_QWORD *)v11 - 5) = v24;
          LODWORD(v24) = *(_DWORD *)(v20 + 24);
          *(_DWORD *)(v20 + 24) = v23;
          LODWORD(v23) = *(v11 - 7);
          *(v11 - 8) = v24;
          LODWORD(v24) = *(_DWORD *)(v20 + 28);
          *(_DWORD *)(v20 + 28) = v23;
          LODWORD(v23) = *(v11 - 6);
          *(v11 - 7) = v24;
          LODWORD(v24) = *(_DWORD *)(v20 + 32);
          *(_DWORD *)(v20 + 32) = v23;
          *(v11 - 6) = v24;
          *(_QWORD *)(v20 + 40) = v20 + 56;
          *(_QWORD *)(v20 + 48) = 0x400000000LL;
          v25 = *(v11 - 2);
          if ( v25 && (_DWORD *)(v20 + 40) != v11 - 4 )
          {
            v27 = (_DWORD *)*((_QWORD *)v11 - 2);
            if ( v27 == v11 )
            {
              v28 = v11;
              v29 = 8LL * v25;
              if ( v25 <= 4 )
                goto LABEL_23;
              v33 = *(v11 - 2);
              v35 = v20;
              sub_C8D5F0(v20 + 40, (const void *)(v20 + 56), v25, 8u, v20, v25);
              v20 = v35;
              v28 = (const void *)*((_QWORD *)v11 - 2);
              v25 = v33;
              v29 = 8LL * (unsigned int)*(v11 - 2);
              v22 = *(void **)(v35 + 40);
              if ( v29 )
              {
LABEL_23:
                v32 = v25;
                v34 = v20;
                memcpy(v22, v28, v29);
                *(_DWORD *)(v34 + 48) = v32;
                *(v11 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v35 + 48) = v33;
                *(v11 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v20 + 40) = v27;
              *(_DWORD *)(v20 + 48) = *(v11 - 2);
              *(_DWORD *)(v20 + 52) = *(v11 - 1);
              *((_QWORD *)v11 - 2) = v11;
              *(v11 - 1) = 0;
              *(v11 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v26 = *((_QWORD *)v11 - 2);
          if ( (_DWORD *)v26 != v11 )
            _libc_free(v26);
          sub_C7D6A0(*((_QWORD *)v11 - 5), 8LL * (unsigned int)*(v11 - 6), 8);
          v12 = (char *)(v11 + 22);
          if ( (_DWORD *)v9 == v11 + 8 )
            return (_DWORD *)sub_C7D6A0(v5, v8, 8);
        }
        v11 = v12;
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v8, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v30]; j != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
