// Function: sub_1C04FF0
// Address: 0x1c04ff0
//
__int64 __fastcall sub_1C04FF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  int v6; // r8d
  int v7; // r10d
  unsigned int v8; // r13d
  unsigned int v9; // edi
  unsigned int v10; // r11d
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // rax
  int v16; // r14d
  __int64 *v17; // r10
  int v18; // r14d
  _QWORD *v19; // r9
  __int64 v20; // rdi
  int v21; // eax
  int v22; // edx
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  unsigned int v26; // r13d
  __int64 v27; // rcx
  int v28; // r8d
  _QWORD *v29; // rdi
  int v30; // eax
  int v31; // eax
  __int64 v32; // rsi
  int v33; // r8d
  unsigned int v34; // r13d
  __int64 v35; // rcx
  __int64 *v36; // r10

  v3 = *(unsigned int *)(a1 + 64);
  if ( !(_DWORD)v3 )
    return 7;
  v4 = *(_QWORD *)(a1 + 48);
  v6 = v3 - 1;
  v7 = 1;
  v8 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v9 = (v3 - 1) & v8;
  v10 = v9;
  v11 = (_QWORD *)(v4 + 16LL * v9);
  v12 = *v11;
  v13 = *v11;
  if ( *v11 == a2 )
  {
    if ( v11 != (_QWORD *)(v4 + 16 * v3) )
    {
      v14 = v11[1];
      return *(unsigned int *)(v14 + 16);
    }
    return 7;
  }
  while ( 1 )
  {
    if ( v13 == -8 )
      return 7;
    v16 = v7 + 1;
    v10 = v6 & (v10 + v7);
    v17 = (__int64 *)(v4 + 16LL * v10);
    v13 = *v17;
    if ( *v17 == a2 )
      break;
    v7 = v16;
  }
  v18 = 1;
  v19 = 0;
  if ( v17 == (__int64 *)(v4 + 16LL * (unsigned int)v3) )
    return 7;
  while ( v12 != -8 )
  {
    if ( v19 || v12 != -16 )
      v11 = v19;
    v9 = v6 & (v18 + v9);
    v36 = (__int64 *)(v4 + 16LL * v9);
    v12 = *v36;
    if ( *v36 == a2 )
    {
      v14 = v36[1];
      return *(unsigned int *)(v14 + 16);
    }
    ++v18;
    v19 = v11;
    v11 = (_QWORD *)(v4 + 16LL * v9);
  }
  v20 = a1 + 40;
  if ( !v19 )
    v19 = v11;
  v21 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= (unsigned int)(3 * v3) )
  {
    sub_1C04E30(v20, 2 * v3);
    v23 = *(_DWORD *)(a1 + 64);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 48);
      v26 = v24 & v8;
      v22 = *(_DWORD *)(a1 + 56) + 1;
      v19 = (_QWORD *)(v25 + 16LL * v26);
      v27 = *v19;
      if ( *v19 == a2 )
        goto LABEL_16;
      v28 = 1;
      v29 = 0;
      while ( v27 != -8 )
      {
        if ( v27 == -16 && !v29 )
          v29 = v19;
        v26 = v24 & (v28 + v26);
        v19 = (_QWORD *)(v25 + 16LL * v26);
        v27 = *v19;
        if ( *v19 == a2 )
          goto LABEL_16;
        ++v28;
      }
      goto LABEL_23;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
  if ( (int)v3 - *(_DWORD *)(a1 + 60) - v22 > (unsigned int)v3 >> 3 )
    goto LABEL_16;
  sub_1C04E30(v20, v3);
  v30 = *(_DWORD *)(a1 + 64);
  if ( !v30 )
    goto LABEL_45;
  v31 = v30 - 1;
  v32 = *(_QWORD *)(a1 + 48);
  v33 = 1;
  v34 = v31 & v8;
  v29 = 0;
  v22 = *(_DWORD *)(a1 + 56) + 1;
  v19 = (_QWORD *)(v32 + 16LL * v34);
  v35 = *v19;
  if ( *v19 == a2 )
    goto LABEL_16;
  while ( v35 != -8 )
  {
    if ( !v29 && v35 == -16 )
      v29 = v19;
    v34 = v31 & (v33 + v34);
    v19 = (_QWORD *)(v32 + 16LL * v34);
    v35 = *v19;
    if ( *v19 == a2 )
      goto LABEL_16;
    ++v33;
  }
LABEL_23:
  if ( v29 )
    v19 = v29;
LABEL_16:
  *(_DWORD *)(a1 + 56) = v22;
  if ( *v19 != -8 )
    --*(_DWORD *)(a1 + 60);
  *v19 = a2;
  v14 = 0;
  v19[1] = 0;
  return *(unsigned int *)(v14 + 16);
}
