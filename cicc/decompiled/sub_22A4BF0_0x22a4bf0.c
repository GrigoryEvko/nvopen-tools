// Function: sub_22A4BF0
// Address: 0x22a4bf0
//
_QWORD *__fastcall sub_22A4BF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r13
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r14
  int v15; // edx
  int v16; // edx
  __int64 v17; // rdi
  int v18; // r11d
  unsigned int v19; // ecx
  __int64 *v20; // r10
  __int64 *v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r10d
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  int v28; // r10d
  size_t v29; // rdx
  __int64 v30; // rcx
  _QWORD *j; // rdx
  int v32; // [rsp+Ch] [rbp-44h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v34 = v5;
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v33 = 56 * v4;
    v10 = 56 * v4 + v5;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v12 = v5 + 56;
    if ( v10 != v5 )
    {
      do
      {
        v13 = *(_QWORD *)(v12 - 56);
        v14 = v12;
        if ( v13 != -8192 && v13 != -4096 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
          {
            MEMORY[0] = *(_QWORD *)(v12 - 56);
            BUG();
          }
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = 1;
          v19 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v20 = 0;
          v21 = (__int64 *)(v17 + 56LL * v19);
          v22 = *v21;
          if ( v13 != *v21 )
          {
            while ( v22 != -4096 )
            {
              if ( !v20 && v22 == -8192 )
                v20 = v21;
              v8 = (unsigned int)(v18 + 1);
              v19 = v16 & (v18 + v19);
              v21 = (__int64 *)(v17 + 56LL * v19);
              v22 = *v21;
              if ( v13 == *v21 )
                goto LABEL_13;
              ++v18;
            }
            if ( v20 )
              v21 = v20;
          }
LABEL_13:
          v21[3] = 0;
          v21[2] = 0;
          *((_DWORD *)v21 + 8) = 0;
          *v21 = v13;
          v21[1] = 1;
          v23 = *(_QWORD *)(v12 - 40);
          ++*(_QWORD *)(v12 - 48);
          v24 = v21[2];
          v21[2] = v23;
          LODWORD(v23) = *(_DWORD *)(v12 - 32);
          *(_QWORD *)(v12 - 40) = v24;
          LODWORD(v24) = *((_DWORD *)v21 + 6);
          *((_DWORD *)v21 + 6) = v23;
          LODWORD(v23) = *(_DWORD *)(v12 - 28);
          *(_DWORD *)(v12 - 32) = v24;
          LODWORD(v24) = *((_DWORD *)v21 + 7);
          *((_DWORD *)v21 + 7) = v23;
          LODWORD(v23) = *(_DWORD *)(v12 - 24);
          *(_DWORD *)(v12 - 28) = v24;
          LODWORD(v24) = *((_DWORD *)v21 + 8);
          *((_DWORD *)v21 + 8) = v23;
          *(_DWORD *)(v12 - 24) = v24;
          v21[5] = (__int64)(v21 + 7);
          v21[6] = 0;
          v25 = *(_DWORD *)(v12 - 8);
          if ( v25 && v21 + 5 != (__int64 *)(v12 - 16) )
          {
            v27 = *(_QWORD *)(v12 - 16);
            if ( v27 == v12 )
            {
              v32 = *(_DWORD *)(v12 - 8);
              sub_C8D5F0((__int64)(v21 + 5), v21 + 7, v25, 8u, v8, v9);
              v28 = v32;
              v29 = 8LL * *(unsigned int *)(v12 - 8);
              if ( v29 )
              {
                memcpy((void *)v21[5], *(const void **)(v12 - 16), v29);
                v28 = v32;
              }
              *((_DWORD *)v21 + 12) = v28;
              *(_DWORD *)(v12 - 8) = 0;
            }
            else
            {
              v21[5] = v27;
              *((_DWORD *)v21 + 12) = *(_DWORD *)(v12 - 8);
              *((_DWORD *)v21 + 13) = *(_DWORD *)(v12 - 4);
              *(_QWORD *)(v12 - 16) = v12;
              *(_DWORD *)(v12 - 4) = 0;
              *(_DWORD *)(v12 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v26 = *(_QWORD *)(v12 - 16);
          if ( v26 != v12 )
            _libc_free(v26);
          sub_C7D6A0(*(_QWORD *)(v12 - 40), 8LL * *(unsigned int *)(v12 - 24), 8);
        }
        v12 += 56;
      }
      while ( v10 != v14 );
    }
    return (_QWORD *)sub_C7D6A0(v34, v33, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v30]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
