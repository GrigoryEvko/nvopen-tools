// Function: sub_3158A80
// Address: 0x3158a80
//
_QWORD *__fastcall sub_3158A80(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  _QWORD *i; // rdx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdi
  int v16; // r11d
  __int64 v17; // r10
  unsigned int v18; // ecx
  __int64 v19; // r8
  __int64 v20; // rsi
  void *v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // r10d
  unsigned __int64 v25; // rdi
  _QWORD *v26; // rax
  const void *v27; // rsi
  size_t v28; // rdx
  __int64 v29; // rdx
  _QWORD *j; // rdx
  __int64 v31; // [rsp+0h] [rbp-40h]
  __int64 v32; // [rsp+0h] [rbp-40h]
  unsigned int v33; // [rsp+Ch] [rbp-34h]
  unsigned int v34; // [rsp+Ch] [rbp-34h]

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
  result = (_QWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 88 * v4;
    v9 = v5 + 88 * v4;
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = (_QWORD *)(v5 + 56);
    if ( v9 != v5 )
    {
      while ( 1 )
      {
        v12 = *(v11 - 7);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(v11 - 7);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = v15 + 88LL * v18;
          v20 = *(_QWORD *)v19;
          if ( v12 != *(_QWORD *)v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = v15 + 88LL * v18;
              v20 = *(_QWORD *)v19;
              if ( v12 == *(_QWORD *)v19 )
                goto LABEL_15;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_15:
          *(_QWORD *)(v19 + 24) = 0;
          v21 = (void *)(v19 + 56);
          *(_QWORD *)(v19 + 16) = 0;
          *(_DWORD *)(v19 + 32) = 0;
          *(_QWORD *)v19 = v12;
          *(_QWORD *)(v19 + 8) = 1;
          v22 = *(v11 - 5);
          ++*(v11 - 6);
          v23 = *(_QWORD *)(v19 + 16);
          *(_QWORD *)(v19 + 16) = v22;
          LODWORD(v22) = *((_DWORD *)v11 - 8);
          *(v11 - 5) = v23;
          LODWORD(v23) = *(_DWORD *)(v19 + 24);
          *(_DWORD *)(v19 + 24) = v22;
          LODWORD(v22) = *((_DWORD *)v11 - 7);
          *((_DWORD *)v11 - 8) = v23;
          LODWORD(v23) = *(_DWORD *)(v19 + 28);
          *(_DWORD *)(v19 + 28) = v22;
          LODWORD(v22) = *((_DWORD *)v11 - 6);
          *((_DWORD *)v11 - 7) = v23;
          LODWORD(v23) = *(_DWORD *)(v19 + 32);
          *(_DWORD *)(v19 + 32) = v22;
          *((_DWORD *)v11 - 6) = v23;
          *(_QWORD *)(v19 + 40) = v19 + 56;
          *(_QWORD *)(v19 + 48) = 0x400000000LL;
          v24 = *((_DWORD *)v11 - 2);
          if ( v24 && (_QWORD *)(v19 + 40) != v11 - 2 )
          {
            v26 = (_QWORD *)*(v11 - 2);
            if ( v26 == v11 )
            {
              v27 = v11;
              v28 = 8LL * v24;
              if ( v24 <= 4 )
                goto LABEL_24;
              v32 = v19;
              v34 = *((_DWORD *)v11 - 2);
              sub_C8D5F0(v19 + 40, (const void *)(v19 + 56), v24, 8u, v19, v24);
              v19 = v32;
              v27 = (const void *)*(v11 - 2);
              v24 = v34;
              v28 = 8LL * *((unsigned int *)v11 - 2);
              v21 = *(void **)(v32 + 40);
              if ( v28 )
              {
LABEL_24:
                v31 = v19;
                v33 = v24;
                memcpy(v21, v27, v28);
                *(_DWORD *)(v31 + 48) = v33;
                *((_DWORD *)v11 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v32 + 48) = v34;
                *((_DWORD *)v11 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v19 + 40) = v26;
              *(_DWORD *)(v19 + 48) = *((_DWORD *)v11 - 2);
              *(_DWORD *)(v19 + 52) = *((_DWORD *)v11 - 1);
              *(v11 - 2) = v11;
              *((_DWORD *)v11 - 1) = 0;
              *((_DWORD *)v11 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v25 = *(v11 - 2);
          if ( (_QWORD *)v25 != v11 )
            _libc_free(v25);
          sub_C7D6A0(*(v11 - 5), 8LL * *((unsigned int *)v11 - 6), 8);
        }
        if ( (_QWORD *)v9 == v11 + 4 )
          break;
        v11 += 11;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v8, 8);
  }
  else
  {
    v29 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[11 * v29]; j != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
