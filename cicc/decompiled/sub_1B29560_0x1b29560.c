// Function: sub_1B29560
// Address: 0x1b29560
//
char __fastcall sub_1B29560(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r15
  int v14; // edx
  __int64 v15; // rax
  unsigned int v16; // esi
  __int64 v17; // r9
  int v18; // r11d
  __int64 *v19; // rcx
  unsigned int v20; // r8d
  _QWORD *v21; // rax
  __int64 v22; // rdi
  int v23; // r9d
  int v24; // eax
  int v25; // edi
  int v26; // eax
  int v27; // esi
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // r8
  int v31; // r10d
  __int64 *v32; // r9
  int v33; // eax
  int v34; // eax
  __int64 v35; // r8
  int v36; // r10d
  unsigned int v37; // edx
  __int64 v38; // rsi
  _QWORD *v39; // [rsp+8h] [rbp-38h]
  _QWORD *v40; // [rsp+8h] [rbp-38h]
  unsigned int v41; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a2 + 40);
  if ( (_DWORD)v6 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v9 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v7 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v6) )
      {
        v12 = v10[1];
        return sub_143B490(v12, a2, a3);
      }
    }
    else
    {
      v14 = 1;
      while ( v11 != -8 )
      {
        v23 = v14 + 1;
        v9 = (v6 - 1) & (v14 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
          goto LABEL_3;
        v14 = v23;
      }
    }
  }
  v15 = sub_22077B0(552);
  v12 = v15;
  if ( v15 )
    sub_143ACA0(v15, v7);
  v16 = *(_DWORD *)(a1 + 24);
  if ( !v16 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_33;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v18 = 1;
  v19 = 0;
  v20 = (v16 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v21 = (_QWORD *)(v17 + 16LL * v20);
  v22 = *v21;
  if ( v7 == *v21 )
  {
LABEL_12:
    if ( v12 )
    {
      if ( (*(_BYTE *)(v12 + 8) & 1) == 0 )
      {
        v40 = v21;
        j___libc_free_0(*(_QWORD *)(v12 + 16));
        v21 = v40;
      }
      v39 = v21;
      j_j___libc_free_0(v12, 552);
      v21 = v39;
    }
    v12 = v21[1];
    return sub_143B490(v12, a2, a3);
  }
  while ( v22 != -8 )
  {
    if ( !v19 && v22 == -16 )
      v19 = v21;
    v20 = (v16 - 1) & (v18 + v20);
    v21 = (_QWORD *)(v17 + 16LL * v20);
    v22 = *v21;
    if ( v7 == *v21 )
      goto LABEL_12;
    ++v18;
  }
  if ( !v19 )
    v19 = v21;
  v24 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v16 )
  {
LABEL_33:
    sub_1B29350(a1, 2 * v16);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = (v26 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v25 = *(_DWORD *)(a1 + 16) + 1;
      v19 = (__int64 *)(v28 + 16LL * v29);
      v30 = *v19;
      if ( v7 == *v19 )
        goto LABEL_29;
      v31 = 1;
      v32 = 0;
      while ( v30 != -8 )
      {
        if ( v30 == -16 && !v32 )
          v32 = v19;
        v29 = v27 & (v31 + v29);
        v19 = (__int64 *)(v28 + 16LL * v29);
        v30 = *v19;
        if ( v7 == *v19 )
          goto LABEL_29;
        ++v31;
      }
LABEL_37:
      if ( v32 )
        v19 = v32;
      goto LABEL_29;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v16 - *(_DWORD *)(a1 + 20) - v25 <= v16 >> 3 )
  {
    v41 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
    sub_1B29350(a1, v16);
    v33 = *(_DWORD *)(a1 + 24);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 8);
      v36 = 1;
      v32 = 0;
      v37 = v34 & v41;
      v25 = *(_DWORD *)(a1 + 16) + 1;
      v19 = (__int64 *)(v35 + 16LL * (v34 & v41));
      v38 = *v19;
      if ( v7 == *v19 )
        goto LABEL_29;
      while ( v38 != -8 )
      {
        if ( v38 == -16 && !v32 )
          v32 = v19;
        v37 = v34 & (v36 + v37);
        v19 = (__int64 *)(v35 + 16LL * v37);
        v38 = *v19;
        if ( v7 == *v19 )
          goto LABEL_29;
        ++v36;
      }
      goto LABEL_37;
    }
    goto LABEL_53;
  }
LABEL_29:
  *(_DWORD *)(a1 + 16) = v25;
  if ( *v19 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v19 = v7;
  v19[1] = v12;
  return sub_143B490(v12, a2, a3);
}
