// Function: sub_2EEEEA0
// Address: 0x2eeeea0
//
_QWORD *__fastcall sub_2EEEEA0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r13
  _QWORD *i; // rdx
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rsi
  int v16; // r10d
  __int64 *v17; // r8
  unsigned int v18; // edx
  __int64 *v19; // r12
  __int64 v20; // rdi
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *j; // rdx
  __int64 v32; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v32 = v4;
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
  result = (_QWORD *)sub_C7D670(368LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 368 * v5;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v4 + v9;
    for ( i = &result[46 * v8]; i != result; result += 46 )
    {
      if ( result )
        *result = -4096;
    }
    for ( ; v10 != v4; v4 += 368 )
    {
      v12 = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -8192 && v12 != -4096 )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = *(_QWORD *)v4;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v19 = (__int64 *)(v15 + 368LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v17 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (__int64 *)(v15 + 368LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_16;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_16:
        *v19 = v12;
        v21 = *(_BYTE *)(v4 + 8);
        v19[4] = 0;
        v19[3] = 0;
        *((_DWORD *)v19 + 10) = 0;
        *((_BYTE *)v19 + 8) = v21;
        v19[2] = 1;
        v22 = *(_QWORD *)(v4 + 24);
        ++*(_QWORD *)(v4 + 16);
        v23 = v19[3];
        v19[3] = v22;
        LODWORD(v22) = *(_DWORD *)(v4 + 32);
        *(_QWORD *)(v4 + 24) = v23;
        LODWORD(v23) = *((_DWORD *)v19 + 8);
        *((_DWORD *)v19 + 8) = v22;
        LODWORD(v22) = *(_DWORD *)(v4 + 36);
        *(_DWORD *)(v4 + 32) = v23;
        LODWORD(v23) = *((_DWORD *)v19 + 9);
        *((_DWORD *)v19 + 9) = v22;
        LODWORD(v22) = *(_DWORD *)(v4 + 40);
        *(_DWORD *)(v4 + 36) = v23;
        LODWORD(v23) = *((_DWORD *)v19 + 10);
        *((_DWORD *)v19 + 10) = v22;
        *(_DWORD *)(v4 + 40) = v23;
        v19[7] = 0;
        v19[6] = 1;
        v19[8] = 0;
        *((_DWORD *)v19 + 18) = 0;
        ++*(_QWORD *)(v4 + 48);
        v24 = v19[7];
        v19[7] = *(_QWORD *)(v4 + 56);
        LODWORD(v22) = *(_DWORD *)(v4 + 64);
        *(_QWORD *)(v4 + 56) = v24;
        LODWORD(v24) = *((_DWORD *)v19 + 16);
        *((_DWORD *)v19 + 16) = v22;
        LODWORD(v22) = *(_DWORD *)(v4 + 68);
        *(_DWORD *)(v4 + 64) = v24;
        LODWORD(v24) = *((_DWORD *)v19 + 17);
        *((_DWORD *)v19 + 17) = v22;
        LODWORD(v22) = *(_DWORD *)(v4 + 72);
        *(_DWORD *)(v4 + 68) = v24;
        LODWORD(v24) = *((_DWORD *)v19 + 18);
        *((_DWORD *)v19 + 18) = v22;
        *(_DWORD *)(v4 + 72) = v24;
        v19[12] = 0;
        v19[11] = 0;
        v19[10] = 1;
        *((_DWORD *)v19 + 26) = 0;
        v25 = *(_QWORD *)(v4 + 88);
        ++*(_QWORD *)(v4 + 80);
        v26 = v19[11];
        v19[11] = v25;
        LODWORD(v25) = *(_DWORD *)(v4 + 96);
        *(_QWORD *)(v4 + 88) = v26;
        LODWORD(v26) = *((_DWORD *)v19 + 24);
        *((_DWORD *)v19 + 24) = v25;
        LODWORD(v25) = *(_DWORD *)(v4 + 100);
        *(_DWORD *)(v4 + 96) = v26;
        LODWORD(v26) = *((_DWORD *)v19 + 25);
        *((_DWORD *)v19 + 25) = v25;
        *(_DWORD *)(v4 + 100) = v26;
        LODWORD(v26) = *((_DWORD *)v19 + 26);
        *((_DWORD *)v19 + 26) = *(_DWORD *)(v4 + 104);
        *(_DWORD *)(v4 + 104) = v26;
        v19[16] = 0;
        v19[15] = 0;
        *((_DWORD *)v19 + 34) = 0;
        v19[14] = 1;
        v27 = *(_QWORD *)(v4 + 120);
        ++*(_QWORD *)(v4 + 112);
        v28 = v19[15];
        v19[15] = v27;
        LODWORD(v27) = *(_DWORD *)(v4 + 128);
        *(_QWORD *)(v4 + 120) = v28;
        LODWORD(v28) = *((_DWORD *)v19 + 32);
        *((_DWORD *)v19 + 32) = v27;
        LODWORD(v27) = *(_DWORD *)(v4 + 132);
        *(_DWORD *)(v4 + 128) = v28;
        LODWORD(v28) = *((_DWORD *)v19 + 33);
        *((_DWORD *)v19 + 33) = v27;
        LODWORD(v27) = *(_DWORD *)(v4 + 136);
        *(_DWORD *)(v4 + 132) = v28;
        LODWORD(v28) = *((_DWORD *)v19 + 34);
        *((_DWORD *)v19 + 34) = v27;
        *(_DWORD *)(v4 + 136) = v28;
        v19[18] = 1;
        v19[19] = 0;
        v19[20] = 0;
        *((_DWORD *)v19 + 42) = 0;
        ++*(_QWORD *)(v4 + 144);
        v29 = v19[19];
        v19[19] = *(_QWORD *)(v4 + 152);
        LODWORD(v27) = *(_DWORD *)(v4 + 160);
        *(_QWORD *)(v4 + 152) = v29;
        LODWORD(v29) = *((_DWORD *)v19 + 40);
        *((_DWORD *)v19 + 40) = v27;
        LODWORD(v27) = *(_DWORD *)(v4 + 164);
        *(_DWORD *)(v4 + 160) = v29;
        LODWORD(v29) = *((_DWORD *)v19 + 41);
        *((_DWORD *)v19 + 41) = v27;
        LODWORD(v27) = *(_DWORD *)(v4 + 168);
        *(_DWORD *)(v4 + 164) = v29;
        LODWORD(v29) = *((_DWORD *)v19 + 42);
        *((_DWORD *)v19 + 42) = v27;
        *(_DWORD *)(v4 + 168) = v29;
        sub_C8CF70((__int64)(v19 + 22), v19 + 26, 8, v4 + 208, v4 + 176);
        sub_C8CF70((__int64)(v19 + 34), v19 + 38, 8, v4 + 304, v4 + 272);
        ++*(_DWORD *)(a1 + 16);
        if ( !*(_BYTE *)(v4 + 300) )
          _libc_free(*(_QWORD *)(v4 + 280));
        if ( !*(_BYTE *)(v4 + 204) )
          _libc_free(*(_QWORD *)(v4 + 184));
        sub_C7D6A0(*(_QWORD *)(v4 + 152), 4LL * *(unsigned int *)(v4 + 168), 4);
        sub_C7D6A0(*(_QWORD *)(v4 + 120), 4LL * *(unsigned int *)(v4 + 136), 4);
        sub_C7D6A0(*(_QWORD *)(v4 + 88), 4LL * *(unsigned int *)(v4 + 104), 4);
        sub_C7D6A0(*(_QWORD *)(v4 + 56), 4LL * *(unsigned int *)(v4 + 72), 4);
        sub_C7D6A0(*(_QWORD *)(v4 + 24), 16LL * *(unsigned int *)(v4 + 40), 8);
      }
    }
    return (_QWORD *)sub_C7D6A0(v32, v9, 8);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[46 * v30]; j != result; result += 46 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
