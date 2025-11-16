// Function: sub_3245760
// Address: 0x3245760
//
_QWORD *__fastcall sub_3245760(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  _QWORD *v10; // r13
  __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  int v15; // r11d
  __int64 v16; // r10
  unsigned int v17; // ecx
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  char *v22; // rdx
  int v23; // esi
  void *v24; // rdi
  unsigned int v25; // r10d
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  _DWORD *v29; // r15
  const void *v30; // rsi
  __int64 v31; // r8
  __int64 v32; // rcx
  _QWORD *j; // rdx
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  unsigned int v36; // [rsp+14h] [rbp-3Ch]
  unsigned int v37; // [rsp+14h] [rbp-3Ch]
  __int64 v38; // [rsp+18h] [rbp-38h]

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
    v38 = 136 * v4;
    v8 = v5 + 136 * v4;
    for ( i = &result[17 * *(unsigned int *)(a1 + 24)]; i != result; result += 17 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (_QWORD *)(v5 + 72);
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 9);
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 9);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = v14 + 136LL * v17;
          v19 = *(_QWORD *)v18;
          if ( v11 != *(_QWORD *)v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = v14 + 136LL * v17;
              v19 = *(_QWORD *)v18;
              if ( v11 == *(_QWORD *)v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_13:
          *(_QWORD *)v18 = v11;
          v20 = *(v10 - 6);
          v21 = v18 + 16;
          v22 = (char *)(v10 - 7);
          if ( v20 )
          {
            v23 = *((_DWORD *)v10 - 14);
            *(_QWORD *)(v18 + 24) = v20;
            *(_DWORD *)(v18 + 16) = v23;
            *(_QWORD *)(v18 + 32) = *(v10 - 5);
            *(_QWORD *)(v18 + 40) = *(v10 - 4);
            *(_QWORD *)(v20 + 8) = v21;
            *(_QWORD *)(v18 + 48) = *(v10 - 3);
            *(v10 - 6) = 0;
            *(v10 - 5) = v22;
            *(v10 - 4) = v22;
            *(v10 - 3) = 0;
          }
          else
          {
            *(_DWORD *)(v18 + 16) = 0;
            *(_QWORD *)(v18 + 24) = 0;
            *(_QWORD *)(v18 + 32) = v21;
            *(_QWORD *)(v18 + 40) = v21;
            *(_QWORD *)(v18 + 48) = 0;
          }
          v24 = (void *)(v18 + 72);
          *(_QWORD *)(v18 + 56) = v18 + 72;
          *(_QWORD *)(v18 + 64) = 0x800000000LL;
          v25 = *((_DWORD *)v10 - 2);
          if ( v25 && (_QWORD *)(v18 + 56) != v10 - 2 )
          {
            v29 = (_DWORD *)*(v10 - 2);
            if ( v10 == (_QWORD *)v29 )
            {
              v30 = v10;
              v31 = 8LL * v25;
              if ( v25 <= 8 )
                goto LABEL_26;
              v35 = v18;
              v37 = *((_DWORD *)v10 - 2);
              sub_C8D5F0(v18 + 56, (const void *)(v18 + 72), v25, 8u, v31, v18);
              v18 = v35;
              v30 = (const void *)*(v10 - 2);
              v25 = v37;
              v24 = *(void **)(v35 + 56);
              v31 = 8LL * *((unsigned int *)v10 - 2);
              if ( v31 )
              {
LABEL_26:
                v34 = v18;
                v36 = v25;
                memcpy(v24, v30, v31);
                *(_DWORD *)(v34 + 64) = v36;
                *(v29 - 2) = 0;
              }
              else
              {
                *(_DWORD *)(v35 + 64) = v37;
                *(v29 - 2) = 0;
              }
            }
            else
            {
              *(_QWORD *)(v18 + 56) = v29;
              *(_DWORD *)(v18 + 64) = *((_DWORD *)v10 - 2);
              *(_DWORD *)(v18 + 68) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v26 = *(v10 - 2);
          if ( (_QWORD *)v26 != v10 )
            _libc_free(v26);
          v27 = *(v10 - 6);
          while ( v27 )
          {
            sub_32449F0(*(_QWORD *)(v27 + 24));
            v28 = v27;
            v27 = *(_QWORD *)(v27 + 16);
            j_j___libc_free_0(v28);
          }
        }
        if ( (_QWORD *)v8 == v10 + 8 )
          break;
        v10 += 17;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v38, 8);
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
