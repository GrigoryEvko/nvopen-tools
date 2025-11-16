// Function: sub_34F6DC0
// Address: 0x34f6dc0
//
_DWORD *__fastcall sub_34F6DC0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 v10; // r12
  _DWORD *i; // rdx
  _DWORD *v12; // rbx
  char *v13; // rax
  unsigned int v14; // eax
  int v15; // edx
  int v16; // edx
  __int64 v17; // rdi
  int v18; // r11d
  __int64 v19; // r10
  unsigned int v20; // ecx
  __int64 v21; // r8
  int v22; // esi
  int v23; // eax
  void *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned int v27; // r10d
  unsigned __int64 v28; // rdi
  _DWORD *v29; // rax
  const void *v30; // rsi
  size_t v31; // rdx
  __int64 v32; // rcx
  _DWORD *j; // rdx
  unsigned int v34; // [rsp+4h] [rbp-3Ch]
  unsigned int v35; // [rsp+4h] [rbp-3Ch]
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]

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
  result = (_DWORD *)sub_C7D670(120LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 120 * v4;
    v10 = v5 + 120 * v4;
    for ( i = &result[30 * v8]; i != result; result += 30 )
    {
      if ( result )
        *result = -1;
    }
    v12 = (_DWORD *)(v5 + 56);
    if ( v10 != v5 )
    {
      while ( 1 )
      {
        v14 = *(v12 - 14);
        if ( v14 > 0xFFFFFFFD )
        {
          v13 = (char *)(v12 + 30);
          if ( (_DWORD *)v10 == v12 + 16 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
          {
            MEMORY[0] = *(v12 - 14);
            BUG();
          }
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = 1;
          v19 = 0;
          v20 = v16 & (37 * v14);
          v21 = v17 + 120LL * v20;
          v22 = *(_DWORD *)v21;
          if ( v14 != *(_DWORD *)v21 )
          {
            while ( v22 != -1 )
            {
              if ( !v19 && v22 == -2 )
                v19 = v21;
              v20 = v16 & (v18 + v20);
              v21 = v17 + 120LL * v20;
              v22 = *(_DWORD *)v21;
              if ( v14 == *(_DWORD *)v21 )
                goto LABEL_15;
              ++v18;
            }
            if ( v19 )
              v21 = v19;
          }
LABEL_15:
          v23 = *(v12 - 14);
          *(_QWORD *)(v21 + 24) = 0;
          v24 = (void *)(v21 + 56);
          *(_QWORD *)(v21 + 16) = 0;
          *(_DWORD *)(v21 + 32) = 0;
          *(_DWORD *)v21 = v23;
          *(_QWORD *)(v21 + 8) = 1;
          v25 = *((_QWORD *)v12 - 5);
          ++*((_QWORD *)v12 - 6);
          v26 = *(_QWORD *)(v21 + 16);
          *(_QWORD *)(v21 + 16) = v25;
          LODWORD(v25) = *(v12 - 8);
          *((_QWORD *)v12 - 5) = v26;
          LODWORD(v26) = *(_DWORD *)(v21 + 24);
          *(_DWORD *)(v21 + 24) = v25;
          LODWORD(v25) = *(v12 - 7);
          *(v12 - 8) = v26;
          LODWORD(v26) = *(_DWORD *)(v21 + 28);
          *(_DWORD *)(v21 + 28) = v25;
          LODWORD(v25) = *(v12 - 6);
          *(v12 - 7) = v26;
          LODWORD(v26) = *(_DWORD *)(v21 + 32);
          *(_DWORD *)(v21 + 32) = v25;
          *(v12 - 6) = v26;
          *(_QWORD *)(v21 + 40) = v21 + 56;
          *(_QWORD *)(v21 + 48) = 0x1000000000LL;
          v27 = *(v12 - 2);
          if ( v27 && (_DWORD *)(v21 + 40) != v12 - 4 )
          {
            v29 = (_DWORD *)*((_QWORD *)v12 - 2);
            if ( v29 == v12 )
            {
              v30 = v12;
              v31 = 4LL * v27;
              if ( v27 <= 0x10 )
                goto LABEL_23;
              v35 = *(v12 - 2);
              v37 = v21;
              sub_C8D5F0(v21 + 40, (const void *)(v21 + 56), v27, 4u, v21, v27);
              v21 = v37;
              v30 = (const void *)*((_QWORD *)v12 - 2);
              v27 = v35;
              v31 = 4LL * (unsigned int)*(v12 - 2);
              v24 = *(void **)(v37 + 40);
              if ( v31 )
              {
LABEL_23:
                v34 = v27;
                v36 = v21;
                memcpy(v24, v30, v31);
                *(_DWORD *)(v36 + 48) = v34;
                *(v12 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v37 + 48) = v35;
                *(v12 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v21 + 40) = v29;
              *(_DWORD *)(v21 + 48) = *(v12 - 2);
              *(_DWORD *)(v21 + 52) = *(v12 - 1);
              *((_QWORD *)v12 - 2) = v12;
              *(v12 - 1) = 0;
              *(v12 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v28 = *((_QWORD *)v12 - 2);
          if ( (_DWORD *)v28 != v12 )
            _libc_free(v28);
          sub_C7D6A0(*((_QWORD *)v12 - 5), 4LL * (unsigned int)*(v12 - 6), 4);
          v13 = (char *)(v12 + 30);
          if ( (_DWORD *)v10 == v12 + 16 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v12 = v13;
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v32 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[30 * v32]; j != result; result += 30 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
