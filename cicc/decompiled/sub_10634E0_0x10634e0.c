// Function: sub_10634E0
// Address: 0x10634e0
//
unsigned __int64 __fastcall sub_10634E0(__int64 a1, char *a2)
{
  __int64 v2; // r10
  unsigned int v4; // esi
  char *v5; // rdx
  __int64 v6; // r8
  char **v7; // r11
  int v8; // r12d
  unsigned __int64 result; // rax
  char **v10; // rdi
  char *v11; // rcx
  char **v12; // r13
  int v13; // eax
  int v14; // ecx
  _BYTE *v15; // rsi
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  char *v19; // rdi
  int v20; // r10d
  char **v21; // r9
  int v22; // eax
  int v23; // esi
  int v24; // r10d
  __int64 v25; // r8
  char *v26; // rdi
  char *v27; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 888;
  v27 = a2;
  v4 = *(_DWORD *)(a1 + 912);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 888);
    goto LABEL_21;
  }
  v5 = v27;
  v6 = *(_QWORD *)(a1 + 896);
  v7 = 0;
  v8 = 1;
  result = (v4 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v10 = (char **)(v6 + 8 * result);
  v11 = *v10;
  if ( *v10 == v27 )
    return result;
  while ( v11 != (char *)-4096LL )
  {
    if ( v11 != (char *)-8192LL || v7 )
      v10 = v7;
    result = (v4 - 1) & (v8 + (_DWORD)result);
    v12 = (char **)(v6 + 8LL * (unsigned int)result);
    v11 = *v12;
    if ( v27 == *v12 )
      return result;
    ++v8;
    v7 = v10;
    v10 = (char **)(v6 + 8LL * (unsigned int)result);
  }
  v13 = *(_DWORD *)(a1 + 904);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)(a1 + 888);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_21:
    sub_E49260(v2, 2 * v4);
    v16 = *(_DWORD *)(a1 + 912);
    if ( v16 )
    {
      v5 = v27;
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 896);
      result = (v16 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v7 = (char **)(v18 + 8 * result);
      v19 = *v7;
      v14 = *(_DWORD *)(a1 + 904) + 1;
      if ( *v7 == v27 )
        goto LABEL_13;
      v20 = 1;
      v21 = 0;
      while ( v19 != (char *)-4096LL )
      {
        if ( v19 == (char *)-8192LL && !v21 )
          v21 = v7;
        result = v17 & (unsigned int)(v20 + result);
        v7 = (char **)(v18 + 8LL * (unsigned int)result);
        v19 = *v7;
        if ( v27 == *v7 )
          goto LABEL_13;
        ++v20;
      }
LABEL_25:
      if ( v21 )
        v7 = v21;
      goto LABEL_13;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 904);
    BUG();
  }
  result = v4 - *(_DWORD *)(a1 + 908) - v14;
  if ( (unsigned int)result <= v4 >> 3 )
  {
    sub_E49260(v2, v4);
    v22 = *(_DWORD *)(a1 + 912);
    if ( v22 )
    {
      v5 = v27;
      v23 = v22 - 1;
      v24 = 1;
      v21 = 0;
      v25 = *(_QWORD *)(a1 + 896);
      result = (v22 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v7 = (char **)(v25 + 8 * result);
      v26 = *v7;
      v14 = *(_DWORD *)(a1 + 904) + 1;
      if ( *v7 == v27 )
        goto LABEL_13;
      while ( v26 != (char *)-4096LL )
      {
        if ( !v21 && v26 == (char *)-8192LL )
          v21 = v7;
        result = v23 & (unsigned int)(v24 + result);
        v7 = (char **)(v25 + 8LL * (unsigned int)result);
        v26 = *v7;
        if ( v27 == *v7 )
          goto LABEL_13;
        ++v24;
      }
      goto LABEL_25;
    }
    goto LABEL_42;
  }
LABEL_13:
  *(_DWORD *)(a1 + 904) = v14;
  if ( *v7 != (char *)-4096LL )
    --*(_DWORD *)(a1 + 908);
  *v7 = v5;
  v15 = *(_BYTE **)(a1 + 928);
  if ( v15 == *(_BYTE **)(a1 + 936) )
    return (unsigned __int64)sub_1061C80(a1 + 920, v15, &v27);
  if ( v15 )
  {
    result = (unsigned __int64)v27;
    *(_QWORD *)v15 = v27;
    v15 = *(_BYTE **)(a1 + 928);
  }
  *(_QWORD *)(a1 + 928) = v15 + 8;
  return result;
}
