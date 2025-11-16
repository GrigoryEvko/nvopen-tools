// Function: sub_270E700
// Address: 0x270e700
//
_QWORD *__fastcall sub_270E700(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  __int64 v6; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 v14; // r8
  unsigned int v15; // eax
  __int64 v16; // r12
  __int64 v17; // rdi
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // r12
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  __int64 v29; // r12
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // rdi

  v5 = a2;
  v6 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[24 * v6]; i != result; result += 24 )
  {
    if ( result )
      *result = -4096;
  }
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v9 = *v5;
      if ( *v5 != -4096 && v9 != -8192 )
        break;
LABEL_39:
      v5 += 24;
      if ( a3 == v5 )
        return result;
    }
    v10 = *(_DWORD *)(a1 + 24);
    if ( !v10 )
    {
      MEMORY[0] = *v5;
      BUG();
    }
    v11 = (unsigned int)(v10 - 1);
    v12 = *(_QWORD *)(a1 + 8);
    v13 = 1;
    v14 = 0;
    v15 = v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v16 = v12 + 192LL * v15;
    v17 = *(_QWORD *)v16;
    if ( v9 != *(_QWORD *)v16 )
    {
      while ( v17 != -4096 )
      {
        if ( v17 == -8192 && !v14 )
          v14 = v16;
        v15 = v11 & (v13 + v15);
        v16 = v12 + 192LL * v15;
        v17 = *(_QWORD *)v16;
        if ( v9 == *(_QWORD *)v16 )
          goto LABEL_10;
        v13 = (unsigned int)(v13 + 1);
      }
      if ( v14 )
        v16 = v14;
    }
LABEL_10:
    *(_QWORD *)v16 = v9;
    *(_DWORD *)(v16 + 8) = *((_DWORD *)v5 + 2);
    v18 = *((_DWORD *)v5 + 3);
    *(_QWORD *)(v16 + 32) = 0;
    *(_QWORD *)(v16 + 24) = 0;
    *(_DWORD *)(v16 + 40) = 0;
    *(_DWORD *)(v16 + 12) = v18;
    *(_QWORD *)(v16 + 16) = 1;
    v19 = v5[3];
    ++v5[2];
    v20 = *(_QWORD *)(v16 + 24);
    *(_QWORD *)(v16 + 24) = v19;
    LODWORD(v19) = *((_DWORD *)v5 + 8);
    v5[3] = v20;
    LODWORD(v20) = *(_DWORD *)(v16 + 32);
    *(_DWORD *)(v16 + 32) = v19;
    LODWORD(v19) = *((_DWORD *)v5 + 9);
    *((_DWORD *)v5 + 8) = v20;
    LODWORD(v20) = *(_DWORD *)(v16 + 36);
    *(_DWORD *)(v16 + 36) = v19;
    LODWORD(v19) = *((_DWORD *)v5 + 10);
    *((_DWORD *)v5 + 9) = v20;
    LODWORD(v20) = *(_DWORD *)(v16 + 40);
    *(_DWORD *)(v16 + 40) = v19;
    *((_DWORD *)v5 + 10) = v20;
    *(_QWORD *)(v16 + 48) = v5[6];
    *(_QWORD *)(v16 + 56) = v5[7];
    *(_QWORD *)(v16 + 64) = v5[8];
    v5[8] = 0;
    v5[7] = 0;
    v5[6] = 0;
    *(_QWORD *)(v16 + 88) = 0;
    *(_QWORD *)(v16 + 80) = 0;
    *(_DWORD *)(v16 + 96) = 0;
    *(_QWORD *)(v16 + 72) = 1;
    v21 = v5[10];
    ++v5[9];
    v22 = *(_QWORD *)(v16 + 80);
    *(_QWORD *)(v16 + 80) = v21;
    LODWORD(v21) = *((_DWORD *)v5 + 22);
    v5[10] = v22;
    LODWORD(v22) = *(_DWORD *)(v16 + 88);
    *(_DWORD *)(v16 + 88) = v21;
    LODWORD(v21) = *((_DWORD *)v5 + 23);
    *((_DWORD *)v5 + 22) = v22;
    LODWORD(v22) = *(_DWORD *)(v16 + 92);
    *(_DWORD *)(v16 + 92) = v21;
    LODWORD(v21) = *((_DWORD *)v5 + 24);
    *((_DWORD *)v5 + 23) = v22;
    LODWORD(v22) = *(_DWORD *)(v16 + 96);
    *(_DWORD *)(v16 + 96) = v21;
    *((_DWORD *)v5 + 24) = v22;
    *(_QWORD *)(v16 + 104) = v5[13];
    *(_QWORD *)(v16 + 112) = v5[14];
    *(_QWORD *)(v16 + 120) = v5[15];
    v5[15] = 0;
    v5[14] = 0;
    v5[13] = 0;
    *(_QWORD *)(v16 + 128) = v16 + 144;
    *(_QWORD *)(v16 + 136) = 0x200000000LL;
    v23 = *((unsigned int *)v5 + 34);
    if ( (_DWORD)v23 )
      sub_270E5A0(v16 + 128, (char **)v5 + 16, v23, v11, v14, v13);
    *(_QWORD *)(v16 + 160) = v16 + 176;
    *(_QWORD *)(v16 + 168) = 0x200000000LL;
    if ( *((_DWORD *)v5 + 42) )
      sub_270E5A0(v16 + 160, (char **)v5 + 20, v23, v11, v14, v13);
    ++*(_DWORD *)(a1 + 16);
    v24 = v5[20];
    if ( (__int64 *)v24 != v5 + 22 )
      _libc_free(v24);
    v25 = v5[16];
    if ( (__int64 *)v25 != v5 + 18 )
      _libc_free(v25);
    v26 = v5[14];
    v27 = v5[13];
    if ( v26 == v27 )
    {
LABEL_26:
      if ( v27 )
        j_j___libc_free_0(v27);
      sub_C7D6A0(v5[10], 16LL * *((unsigned int *)v5 + 24), 8);
      v29 = v5[7];
      v30 = v5[6];
      if ( v29 == v30 )
      {
LABEL_36:
        if ( v30 )
          j_j___libc_free_0(v30);
        result = (_QWORD *)sub_C7D6A0(v5[3], 16LL * *((unsigned int *)v5 + 10), 8);
        goto LABEL_39;
      }
      while ( 1 )
      {
        if ( *(_BYTE *)(v30 + 108) )
        {
          if ( *(_BYTE *)(v30 + 60) )
            goto LABEL_31;
LABEL_34:
          v31 = *(_QWORD *)(v30 + 40);
          v30 += 136LL;
          _libc_free(v31);
          if ( v29 == v30 )
          {
LABEL_35:
            v30 = v5[6];
            goto LABEL_36;
          }
        }
        else
        {
          _libc_free(*(_QWORD *)(v30 + 88));
          if ( !*(_BYTE *)(v30 + 60) )
            goto LABEL_34;
LABEL_31:
          v30 += 136LL;
          if ( v29 == v30 )
            goto LABEL_35;
        }
      }
    }
    while ( 1 )
    {
      if ( *(_BYTE *)(v27 + 108) )
      {
        if ( *(_BYTE *)(v27 + 60) )
          goto LABEL_21;
LABEL_24:
        v28 = *(_QWORD *)(v27 + 40);
        v27 += 136LL;
        _libc_free(v28);
        if ( v26 == v27 )
        {
LABEL_25:
          v27 = v5[13];
          goto LABEL_26;
        }
      }
      else
      {
        _libc_free(*(_QWORD *)(v27 + 88));
        if ( !*(_BYTE *)(v27 + 60) )
          goto LABEL_24;
LABEL_21:
        v27 += 136LL;
        if ( v26 == v27 )
          goto LABEL_25;
      }
    }
  }
  return result;
}
