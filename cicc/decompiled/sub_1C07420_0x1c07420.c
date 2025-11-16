// Function: sub_1C07420
// Address: 0x1c07420
//
char __fastcall sub_1C07420(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v7; // rsi
  int v8; // r9d
  __int64 v9; // rcx
  int v10; // r14d
  unsigned int v11; // ebx
  unsigned int v12; // r8d
  unsigned int v13; // r11d
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r10
  int v17; // r15d
  __int64 *v18; // r14
  __int64 *v19; // r10
  int v20; // r15d
  int v21; // eax
  int v22; // ecx
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  unsigned int v26; // ebx
  __int64 v27; // rsi
  __int64 *v28; // r8
  int v29; // r9d
  int v30; // eax
  int v31; // eax
  __int64 v32; // rdi
  unsigned int v33; // ebx
  int v34; // r9d
  __int64 v35; // rsi
  __int64 *v36; // r11

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8LL);
  v5 = *(_QWORD *)(v4 + 104);
  if ( a3 )
    return sub_1C05930(*(_QWORD *)(v4 + 104), a2, a3);
  v7 = *(unsigned int *)(v5 + 64);
  if ( !(_DWORD)v7 )
    return 0;
  v8 = v7 - 1;
  v9 = *(_QWORD *)(v5 + 48);
  v10 = 1;
  v11 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v12 = (v7 - 1) & v11;
  v13 = v12;
  v14 = (__int64 *)(v9 + 16LL * v12);
  v15 = *v14;
  v16 = *v14;
  if ( a2 == *v14 )
  {
    if ( v14 != (__int64 *)(v9 + 16 * v7) )
    {
      a3 = v14[1];
      return *(_DWORD *)(a3 + 12) == 0;
    }
    return 0;
  }
  while ( 1 )
  {
    if ( v16 == -8 )
      return 0;
    v17 = v10 + 1;
    v13 = v8 & (v10 + v13);
    v18 = (__int64 *)(v9 + 16LL * v13);
    v16 = *v18;
    if ( a2 == *v18 )
      break;
    v10 = v17;
  }
  v19 = 0;
  v20 = 1;
  if ( v18 == (__int64 *)(v9 + 16LL * (unsigned int)v7) )
    return 0;
  while ( v15 != -8 )
  {
    if ( v19 || v15 != -16 )
      v14 = v19;
    v12 = v8 & (v20 + v12);
    v36 = (__int64 *)(v9 + 16LL * v12);
    v15 = *v36;
    if ( a2 == *v36 )
    {
      a3 = v36[1];
      return *(_DWORD *)(a3 + 12) == 0;
    }
    ++v20;
    v19 = v14;
    v14 = (__int64 *)(v9 + 16LL * v12);
  }
  if ( !v19 )
    v19 = v14;
  v21 = *(_DWORD *)(v5 + 56);
  ++*(_QWORD *)(v5 + 40);
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= (unsigned int)(3 * v7) )
  {
    sub_1C04E30(v5 + 40, 2 * v7);
    v23 = *(_DWORD *)(v5 + 64);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v5 + 48);
      v26 = v24 & v11;
      a3 = 0;
      v22 = *(_DWORD *)(v5 + 56) + 1;
      v19 = (__int64 *)(v25 + 16LL * v26);
      v27 = *v19;
      if ( a2 == *v19 )
        goto LABEL_18;
      v28 = 0;
      v29 = 1;
      while ( v27 != -8 )
      {
        if ( v27 == -16 && !v28 )
          v28 = v19;
        v26 = v24 & (v29 + v26);
        v19 = (__int64 *)(v25 + 16LL * v26);
        v27 = *v19;
        if ( a2 == *v19 )
          goto LABEL_18;
        ++v29;
      }
      goto LABEL_25;
    }
LABEL_47:
    ++*(_DWORD *)(v5 + 56);
    BUG();
  }
  if ( (int)v7 - *(_DWORD *)(v5 + 60) - v22 > (unsigned int)v7 >> 3 )
    goto LABEL_18;
  sub_1C04E30(v5 + 40, v7);
  v30 = *(_DWORD *)(v5 + 64);
  if ( !v30 )
    goto LABEL_47;
  v31 = v30 - 1;
  v32 = *(_QWORD *)(v5 + 48);
  v28 = 0;
  v33 = v31 & v11;
  a3 = 0;
  v34 = 1;
  v22 = *(_DWORD *)(v5 + 56) + 1;
  v19 = (__int64 *)(v32 + 16LL * v33);
  v35 = *v19;
  if ( a2 == *v19 )
    goto LABEL_18;
  while ( v35 != -8 )
  {
    if ( !v28 && v35 == -16 )
      v28 = v19;
    v33 = v31 & (v34 + v33);
    v19 = (__int64 *)(v32 + 16LL * v33);
    v35 = *v19;
    if ( a2 == *v19 )
      goto LABEL_18;
    ++v34;
  }
LABEL_25:
  if ( v28 )
    v19 = v28;
LABEL_18:
  *(_DWORD *)(v5 + 56) = v22;
  if ( *v19 != -8 )
    --*(_DWORD *)(v5 + 60);
  *v19 = a2;
  v19[1] = 0;
  return *(_DWORD *)(a3 + 12) == 0;
}
