// Function: sub_25D8AF0
// Address: 0x25d8af0
//
_QWORD *__fastcall sub_25D8AF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  __int64 v10; // r15
  __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // ecx
  __int64 v18; // r9
  __int64 *v19; // r13
  __int64 v20; // rsi
  void *v21; // rdi
  unsigned int v22; // r10d
  __int64 v23; // rax
  __int64 *v24; // rdx
  int v25; // esi
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  const void *v30; // rsi
  size_t v31; // rdx
  __int64 v32; // rcx
  _QWORD *j; // rdx
  unsigned int v34; // [rsp+4h] [rbp-3Ch]
  unsigned int v35; // [rsp+4h] [rbp-3Ch]
  __int64 v36; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(136LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v36 = 136 * v4;
    v8 = v5 + 136 * v4;
    for ( i = &result[17 * *(unsigned int *)(a1 + 24)]; i != result; result += 17 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = v5 + 24;
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v10 - 24);
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(_QWORD *)(v10 - 24);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = v17;
          v19 = (__int64 *)(v14 + 136LL * v17);
          v20 = *v19;
          if ( v11 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v16 && v20 == -8192 )
                v16 = v19;
              v17 = v13 & (v15 + v17);
              v18 = v17;
              v19 = (__int64 *)(v14 + 136LL * v17);
              v20 = *v19;
              if ( v11 == *v19 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v19 = v16;
          }
LABEL_13:
          *v19 = v11;
          v21 = v19 + 3;
          v19[2] = 0x400000000LL;
          v19[1] = (__int64)(v19 + 3);
          v22 = *(_DWORD *)(v10 - 8);
          if ( v19 + 1 != (__int64 *)(v10 - 16) && v22 )
          {
            v29 = *(_QWORD *)(v10 - 16);
            if ( v29 == v10 )
            {
              v30 = (const void *)v10;
              v31 = 16LL * v22;
              if ( v22 <= 4
                || (v35 = *(_DWORD *)(v10 - 8),
                    sub_C8D5F0((__int64)(v19 + 1), v19 + 3, v22, 0x10u, v22, v18),
                    v21 = (void *)v19[1],
                    v30 = *(const void **)(v10 - 16),
                    v22 = v35,
                    (v31 = 16LL * *(unsigned int *)(v10 - 8)) != 0) )
              {
                v34 = v22;
                memcpy(v21, v30, v31);
                *((_DWORD *)v19 + 4) = v34;
                *(_DWORD *)(v10 - 8) = 0;
              }
              else
              {
                *((_DWORD *)v19 + 4) = v35;
                *(_DWORD *)(v10 - 8) = 0;
              }
            }
            else
            {
              v19[1] = v29;
              *((_DWORD *)v19 + 4) = *(_DWORD *)(v10 - 8);
              *((_DWORD *)v19 + 5) = *(_DWORD *)(v10 - 4);
              *(_QWORD *)(v10 - 16) = v10;
              *(_DWORD *)(v10 - 4) = 0;
              *(_DWORD *)(v10 - 8) = 0;
            }
          }
          v23 = *(_QWORD *)(v10 + 80);
          v24 = v19 + 12;
          if ( v23 )
          {
            v25 = *(_DWORD *)(v10 + 72);
            v19[13] = v23;
            *((_DWORD *)v19 + 24) = v25;
            v19[14] = *(_QWORD *)(v10 + 88);
            v19[15] = *(_QWORD *)(v10 + 96);
            *(_QWORD *)(v23 + 8) = v24;
            v19[16] = *(_QWORD *)(v10 + 104);
            *(_QWORD *)(v10 + 80) = 0;
            *(_QWORD *)(v10 + 88) = v10 + 72;
            *(_QWORD *)(v10 + 96) = v10 + 72;
            *(_QWORD *)(v10 + 104) = 0;
          }
          else
          {
            *((_DWORD *)v19 + 24) = 0;
            v19[13] = 0;
            v19[14] = (__int64)v24;
            v19[15] = (__int64)v24;
            v19[16] = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v26 = *(_QWORD *)(v10 + 80);
          while ( v26 )
          {
            sub_25D6BE0(*(_QWORD *)(v26 + 24));
            v27 = v26;
            v26 = *(_QWORD *)(v26 + 16);
            j_j___libc_free_0(v27);
          }
          v28 = *(_QWORD *)(v10 - 16);
          if ( v28 != v10 )
            _libc_free(v28);
        }
        if ( v8 == v10 + 112 )
          break;
        v10 += 136;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v36, 8);
  }
  else
  {
    v32 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[17 * v32]; j != result; result += 17 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
