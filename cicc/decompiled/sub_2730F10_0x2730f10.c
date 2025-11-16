// Function: sub_2730F10
// Address: 0x2730f10
//
_QWORD *__fastcall sub_2730F10(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // r14
  _QWORD *v11; // r12
  _QWORD *i; // rdx
  _QWORD *v13; // rbx
  __int64 v14; // rax
  int v15; // edx
  int v16; // ecx
  __int64 v17; // rsi
  int v18; // r10d
  __int64 *v19; // r9
  unsigned int v20; // edx
  __int64 *v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned __int64 v26; // rdi
  _QWORD *v27; // r10
  int v28; // r9d
  __int64 v29; // r10
  size_t v30; // rdx
  __int64 v31; // rdx
  _QWORD *j; // rdx
  __int64 v33; // [rsp+8h] [rbp-48h]
  int v34; // [rsp+14h] [rbp-3Ch]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v35 = v4;
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
  result = (_QWORD *)sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v9 = *(unsigned int *)(a1 + 24);
    v10 = v5 << 6;
    *(_QWORD *)(a1 + 16) = 0;
    v11 = (_QWORD *)(v4 + v10);
    for ( i = &result[8 * v9]; i != result; result += 8 )
    {
      if ( result )
        *result = -4096;
    }
    v13 = (_QWORD *)(v4 + 56);
    if ( v11 != (_QWORD *)v35 )
    {
      while ( 1 )
      {
        v14 = *(v13 - 7);
        if ( v14 != -8192 && v14 != -4096 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
          {
            MEMORY[0] = *(v13 - 7);
            BUG();
          }
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = 1;
          v19 = 0;
          v20 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v21 = (__int64 *)(v17 + ((unsigned __int64)v20 << 6));
          v22 = *v21;
          if ( v14 != *v21 )
          {
            while ( v22 != -4096 )
            {
              if ( !v19 && v22 == -8192 )
                v19 = v21;
              v20 = v16 & (v18 + v20);
              v21 = (__int64 *)(v17 + ((unsigned __int64)v20 << 6));
              v22 = *v21;
              if ( v14 == *v21 )
                goto LABEL_15;
              ++v18;
            }
            if ( v19 )
              v21 = v19;
          }
LABEL_15:
          v21[3] = 0;
          v21[2] = 0;
          *((_DWORD *)v21 + 8) = 0;
          *v21 = v14;
          v21[1] = 1;
          v23 = *(v13 - 5);
          ++*(v13 - 6);
          v24 = v21[2];
          v21[2] = v23;
          LODWORD(v23) = *((_DWORD *)v13 - 8);
          *(v13 - 5) = v24;
          LODWORD(v24) = *((_DWORD *)v21 + 6);
          *((_DWORD *)v21 + 6) = v23;
          LODWORD(v23) = *((_DWORD *)v13 - 7);
          *((_DWORD *)v13 - 8) = v24;
          LODWORD(v24) = *((_DWORD *)v21 + 7);
          *((_DWORD *)v21 + 7) = v23;
          LODWORD(v23) = *((_DWORD *)v13 - 6);
          *((_DWORD *)v13 - 7) = v24;
          LODWORD(v24) = *((_DWORD *)v21 + 8);
          *((_DWORD *)v21 + 8) = v23;
          *((_DWORD *)v13 - 6) = v24;
          v21[5] = (__int64)(v21 + 7);
          v21[6] = 0;
          v25 = *((unsigned int *)v13 - 2);
          if ( (_DWORD)v25 && v21 + 5 != v13 - 2 )
          {
            v27 = (_QWORD *)*(v13 - 2);
            if ( v13 == v27 )
            {
              v33 = *(v13 - 2);
              v34 = *((_DWORD *)v13 - 2);
              sub_C8D5F0((__int64)(v21 + 5), v21 + 7, (unsigned int)v25, 8u, v8, v25);
              v28 = v34;
              v29 = v33;
              v30 = 8LL * *((unsigned int *)v13 - 2);
              if ( v30 )
              {
                memcpy((void *)v21[5], (const void *)*(v13 - 2), v30);
                v29 = v33;
                v28 = v34;
              }
              *((_DWORD *)v21 + 12) = v28;
              *(_DWORD *)(v29 - 8) = 0;
            }
            else
            {
              v21[5] = (__int64)v27;
              *((_DWORD *)v21 + 12) = *((_DWORD *)v13 - 2);
              *((_DWORD *)v21 + 13) = *((_DWORD *)v13 - 1);
              *(v13 - 2) = v13;
              *((_DWORD *)v13 - 1) = 0;
              *((_DWORD *)v13 - 2) = 0;
            }
          }
          v21[7] = *v13;
          ++*(_DWORD *)(a1 + 16);
          v26 = *(v13 - 2);
          if ( v13 != (_QWORD *)v26 )
            _libc_free(v26);
          sub_C7D6A0(*(v13 - 5), 8LL * *((unsigned int *)v13 - 6), 8);
        }
        if ( v11 == v13 + 1 )
          break;
        v13 += 8;
      }
    }
    return (_QWORD *)sub_C7D6A0(v35, v10, 8);
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v31]; j != result; result += 8 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
