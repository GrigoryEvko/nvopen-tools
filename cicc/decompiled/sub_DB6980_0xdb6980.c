// Function: sub_DB6980
// Address: 0xdb6980
//
_QWORD *__fastcall sub_DB6980(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  unsigned __int64 v9; // rdx
  int v10; // ecx
  __int64 v11; // rcx
  unsigned __int64 *v12; // r8
  int v13; // r10d
  __int64 v14; // rsi
  __int64 v15; // r9
  unsigned __int64 *v16; // rbx
  unsigned __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *j; // rdx
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v25 = v5;
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
  result = (_QWORD *)sub_C7D670(168LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v24 = 168 * v4;
    v26 = 168 * v4 + v5;
    for ( i = &result[21 * *(unsigned int *)(a1 + 24)]; i != result; result += 21 )
    {
      if ( result )
        *result = -4096;
    }
    for ( ; v26 != v5; v5 += 168 )
    {
      v9 = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -8192 && v9 != -4096 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        if ( !v10 )
        {
          MEMORY[0] = *(_QWORD *)v5;
          BUG();
        }
        v11 = (unsigned int)(v10 - 1);
        v12 = *(unsigned __int64 **)(a1 + 8);
        v13 = 1;
        v14 = (unsigned int)v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v15 = 0;
        v16 = &v12[21 * v14];
        v17 = *v16;
        if ( v9 != *v16 )
        {
          while ( v17 != -4096 )
          {
            if ( v17 == -8192 && !v15 )
              v15 = (__int64)v16;
            v14 = (unsigned int)v11 & (v13 + (_DWORD)v14);
            v16 = &v12[21 * (unsigned int)v14];
            v17 = *v16;
            if ( v9 == *v16 )
              goto LABEL_13;
            ++v13;
          }
          if ( v15 )
            v16 = (unsigned __int64 *)v15;
        }
LABEL_13:
        *v16 = v9;
        v16[1] = (unsigned __int64)(v16 + 3);
        v16[2] = 0x100000000LL;
        if ( *(_DWORD *)(v5 + 16) )
        {
          v14 = v5 + 8;
          sub_D9EE30((__int64)(v16 + 1), v5 + 8, (__int64)(v16 + 3), v11, v12, v15);
        }
        v16[17] = *(_QWORD *)(v5 + 136);
        *((_BYTE *)v16 + 144) = *(_BYTE *)(v5 + 144);
        v16[19] = *(_QWORD *)(v5 + 152);
        *((_BYTE *)v16 + 160) = *(_BYTE *)(v5 + 160);
        ++*(_DWORD *)(a1 + 16);
        v18 = *(_QWORD *)(v5 + 8);
        v19 = v18 + 112LL * *(unsigned int *)(v5 + 16);
        if ( v18 != v19 )
        {
          do
          {
            v19 -= 112;
            v20 = *(_QWORD *)(v19 + 64);
            if ( v20 != v19 + 80 )
              _libc_free(v20, v14);
            if ( *(_BYTE *)(v19 + 32) )
              *(_QWORD *)(v19 + 24) = 0;
            v21 = *(_QWORD *)(v19 + 24);
            *(_QWORD *)v19 = &unk_49DB368;
            LOBYTE(v14) = v21 != 0;
            if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
              sub_BD60C0((_QWORD *)(v19 + 8));
          }
          while ( v18 != v19 );
          v19 = *(_QWORD *)(v5 + 8);
        }
        if ( v19 != v5 + 24 )
          _libc_free(v19, v14);
      }
    }
    return (_QWORD *)sub_C7D6A0(v25, v24, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[21 * v22]; j != result; result += 21 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
