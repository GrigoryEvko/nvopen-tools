// Function: sub_D97270
// Address: 0xd97270
//
__int64 __fastcall sub_D97270(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // ecx
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // rsi
  int v9; // edx
  unsigned int v10; // eax
  __int64 *v11; // r13
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rsi
  int v17; // edx
  unsigned int v18; // eax
  __int64 *v19; // r13
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // edx
  __int64 *v25; // r13
  __int64 v26; // rcx
  __int64 v27; // rdi
  int v28; // edi
  int v29; // edi
  int v30; // edi

  v3 = *(unsigned __int16 *)(a2 + 28);
  result = a3 & v3;
  if ( a3 != (_DWORD)result )
  {
    if ( (a3 & 6) != 0 )
      LOWORD(a3) = a3 | 1;
    *(_WORD *)(a2 + 28) = v3 | a3;
    v7 = *(_DWORD *)(a1 + 992);
    v8 = *(_QWORD *)(a1 + 976);
    if ( v7 )
    {
      v9 = v7 - 1;
      v10 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (__int64 *)(v8 + 40LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
      {
LABEL_6:
        if ( *((_DWORD *)v11 + 8) > 0x40u )
        {
          v13 = v11[3];
          if ( v13 )
            j_j___libc_free_0_0(v13);
        }
        if ( *((_DWORD *)v11 + 4) > 0x40u )
        {
          v14 = v11[1];
          if ( v14 )
            j_j___libc_free_0_0(v14);
        }
        *v11 = -8192;
        --*(_DWORD *)(a1 + 984);
        ++*(_DWORD *)(a1 + 988);
      }
      else
      {
        v30 = 1;
        while ( v12 != -4096 )
        {
          v10 = v9 & (v30 + v10);
          v11 = (__int64 *)(v8 + 40LL * v10);
          v12 = *v11;
          if ( a2 == *v11 )
            goto LABEL_6;
          ++v30;
        }
      }
    }
    v15 = *(_DWORD *)(a1 + 1024);
    v16 = *(_QWORD *)(a1 + 1008);
    if ( v15 )
    {
      v17 = v15 - 1;
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (__int64 *)(v16 + 40LL * v18);
      v20 = *v19;
      if ( a2 == *v19 )
      {
LABEL_15:
        if ( *((_DWORD *)v19 + 8) > 0x40u )
        {
          v21 = v19[3];
          if ( v21 )
            j_j___libc_free_0_0(v21);
        }
        if ( *((_DWORD *)v19 + 4) > 0x40u )
        {
          v22 = v19[1];
          if ( v22 )
            j_j___libc_free_0_0(v22);
        }
        *v19 = -8192;
        --*(_DWORD *)(a1 + 1016);
        ++*(_DWORD *)(a1 + 1020);
      }
      else
      {
        v29 = 1;
        while ( v20 != -4096 )
        {
          v18 = v17 & (v29 + v18);
          v19 = (__int64 *)(v16 + 40LL * v18);
          v20 = *v19;
          if ( a2 == *v19 )
            goto LABEL_15;
          ++v29;
        }
      }
    }
    result = *(unsigned int *)(a1 + 640);
    v23 = *(_QWORD *)(a1 + 624);
    if ( (_DWORD)result )
    {
      v24 = result - 1;
      result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = (__int64 *)(v23 + 24 * result);
      v26 = *v25;
      if ( *v25 == a2 )
      {
LABEL_24:
        if ( *((_DWORD *)v25 + 4) > 0x40u )
        {
          v27 = v25[1];
          if ( v27 )
            result = j_j___libc_free_0_0(v27);
        }
        *v25 = -8192;
        --*(_DWORD *)(a1 + 632);
        ++*(_DWORD *)(a1 + 636);
      }
      else
      {
        v28 = 1;
        while ( v26 != -4096 )
        {
          result = v24 & (unsigned int)(v28 + result);
          v25 = (__int64 *)(v23 + 24LL * (unsigned int)result);
          v26 = *v25;
          if ( a2 == *v25 )
            goto LABEL_24;
          ++v28;
        }
      }
    }
  }
  return result;
}
