// Function: sub_1E850A0
// Address: 0x1e850a0
//
_QWORD *__fastcall sub_1E850A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r15
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *v10; // rbx
  __int64 v11; // rdx
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rsi
  int v15; // r9d
  __int64 *v16; // r8
  unsigned int v17; // eax
  __int64 *v18; // r12
  __int64 v19; // rdi
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(384LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[48 * v3];
    for ( i = &result[48 * v7]; i != result; result += 48 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        v11 = *v10;
        if ( *v10 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 384LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( v19 == -16 && !v16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 384LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_14:
          *v18 = v11;
          v20 = *((_BYTE *)v10 + 8);
          v18[4] = 0;
          v18[3] = 0;
          *((_DWORD *)v18 + 10) = 0;
          *((_BYTE *)v18 + 8) = v20;
          v18[2] = 1;
          v21 = v10[3];
          ++v10[2];
          v22 = v18[3];
          v18[3] = v21;
          LODWORD(v21) = *((_DWORD *)v10 + 8);
          v10[3] = v22;
          LODWORD(v22) = *((_DWORD *)v18 + 8);
          *((_DWORD *)v18 + 8) = v21;
          LODWORD(v21) = *((_DWORD *)v10 + 9);
          *((_DWORD *)v10 + 8) = v22;
          LODWORD(v22) = *((_DWORD *)v18 + 9);
          *((_DWORD *)v18 + 9) = v21;
          LODWORD(v21) = *((_DWORD *)v10 + 10);
          *((_DWORD *)v10 + 9) = v22;
          LODWORD(v22) = *((_DWORD *)v18 + 10);
          *((_DWORD *)v18 + 10) = v21;
          *((_DWORD *)v10 + 10) = v22;
          v18[7] = 0;
          v18[6] = 1;
          v18[8] = 0;
          *((_DWORD *)v18 + 18) = 0;
          ++v10[6];
          v23 = v18[7];
          v18[7] = v10[7];
          LODWORD(v21) = *((_DWORD *)v10 + 16);
          v10[7] = v23;
          LODWORD(v23) = *((_DWORD *)v18 + 16);
          *((_DWORD *)v18 + 16) = v21;
          LODWORD(v21) = *((_DWORD *)v10 + 17);
          *((_DWORD *)v10 + 16) = v23;
          LODWORD(v23) = *((_DWORD *)v18 + 17);
          *((_DWORD *)v18 + 17) = v21;
          LODWORD(v21) = *((_DWORD *)v10 + 18);
          *((_DWORD *)v10 + 17) = v23;
          LODWORD(v23) = *((_DWORD *)v18 + 18);
          *((_DWORD *)v18 + 18) = v21;
          *((_DWORD *)v10 + 18) = v23;
          v18[12] = 0;
          v18[11] = 0;
          v18[10] = 1;
          *((_DWORD *)v18 + 26) = 0;
          v24 = v10[11];
          ++v10[10];
          v25 = v18[11];
          v18[11] = v24;
          LODWORD(v24) = *((_DWORD *)v10 + 24);
          v10[11] = v25;
          LODWORD(v25) = *((_DWORD *)v18 + 24);
          *((_DWORD *)v18 + 24) = v24;
          LODWORD(v24) = *((_DWORD *)v10 + 25);
          *((_DWORD *)v10 + 24) = v25;
          LODWORD(v25) = *((_DWORD *)v18 + 25);
          *((_DWORD *)v18 + 25) = v24;
          *((_DWORD *)v10 + 25) = v25;
          LODWORD(v25) = *((_DWORD *)v18 + 26);
          *((_DWORD *)v18 + 26) = *((_DWORD *)v10 + 26);
          *((_DWORD *)v10 + 26) = v25;
          v18[16] = 0;
          v18[15] = 0;
          *((_DWORD *)v18 + 34) = 0;
          v18[14] = 1;
          v26 = v10[15];
          ++v10[14];
          v27 = v18[15];
          v18[15] = v26;
          LODWORD(v26) = *((_DWORD *)v10 + 32);
          v10[15] = v27;
          LODWORD(v27) = *((_DWORD *)v18 + 32);
          *((_DWORD *)v18 + 32) = v26;
          LODWORD(v26) = *((_DWORD *)v10 + 33);
          *((_DWORD *)v10 + 32) = v27;
          LODWORD(v27) = *((_DWORD *)v18 + 33);
          *((_DWORD *)v18 + 33) = v26;
          LODWORD(v26) = *((_DWORD *)v10 + 34);
          *((_DWORD *)v10 + 33) = v27;
          LODWORD(v27) = *((_DWORD *)v18 + 34);
          *((_DWORD *)v18 + 34) = v26;
          *((_DWORD *)v10 + 34) = v27;
          v18[18] = 1;
          v18[19] = 0;
          v18[20] = 0;
          *((_DWORD *)v18 + 42) = 0;
          ++v10[18];
          v28 = v18[19];
          v18[19] = v10[19];
          LODWORD(v26) = *((_DWORD *)v10 + 40);
          v10[19] = v28;
          LODWORD(v28) = *((_DWORD *)v18 + 40);
          *((_DWORD *)v18 + 40) = v26;
          LODWORD(v26) = *((_DWORD *)v10 + 41);
          *((_DWORD *)v10 + 40) = v28;
          LODWORD(v28) = *((_DWORD *)v18 + 41);
          *((_DWORD *)v18 + 41) = v26;
          LODWORD(v26) = *((_DWORD *)v10 + 42);
          *((_DWORD *)v10 + 41) = v28;
          LODWORD(v28) = *((_DWORD *)v18 + 42);
          *((_DWORD *)v18 + 42) = v26;
          *((_DWORD *)v10 + 42) = v28;
          sub_16CCEE0(v18 + 22, (__int64)(v18 + 27), 8, (__int64)(v10 + 22));
          sub_16CCEE0(v18 + 35, (__int64)(v18 + 40), 8, (__int64)(v10 + 35));
          ++*(_DWORD *)(a1 + 16);
          v29 = v10[37];
          if ( v29 != v10[36] )
            _libc_free(v29);
          v30 = v10[24];
          if ( v30 != v10[23] )
            _libc_free(v30);
          j___libc_free_0(v10[19]);
          j___libc_free_0(v10[15]);
          j___libc_free_0(v10[11]);
          j___libc_free_0(v10[7]);
          j___libc_free_0(v10[3]);
        }
        v10 += 48;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[48 * v31]; j != result; result += 48 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
