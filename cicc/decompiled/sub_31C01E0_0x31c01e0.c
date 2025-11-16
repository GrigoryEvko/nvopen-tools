// Function: sub_31C01E0
// Address: 0x31c01e0
//
__int64 __fastcall sub_31C01E0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v6; // r8d
  __int64 v7; // rcx
  __int64 v8; // r10
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r11
  __int64 v12; // rax
  __int64 v13; // rcx
  bool v14; // bl
  __int64 v15; // rcx
  __int64 *v16; // rsi
  int v17; // r8d
  __int64 *v18; // r9
  __int64 v19; // r10
  char v20; // r11
  __int64 *v21; // rdx
  __int64 *v22; // r12
  int v23; // r13d
  __int64 v24; // rdi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 result; // rax
  int v31; // eax
  int v32; // r14d
  __int64 *v33; // rdx
  int v34; // eax
  __int64 *v35; // rdx
  int v36; // ebx

  v6 = *(_DWORD *)(a1 + 64);
  v7 = *a2;
  v8 = *(_QWORD *)(a1 + 48);
  if ( !v6 )
    goto LABEL_35;
  v9 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v7 != *v10 )
  {
    v34 = 1;
    while ( v11 != -4096 )
    {
      v36 = v34 + 1;
      v9 = (v6 - 1) & (v34 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
        goto LABEL_3;
      v34 = v36;
    }
LABEL_35:
    v16 = (__int64 *)sub_31BFEB0((__int64)a2, a3, 1);
    v22 = v35;
    if ( v35 != v16 )
    {
      v20 = 1;
      v15 = 0;
      v14 = 0;
      goto LABEL_7;
    }
    return 0;
  }
LABEL_3:
  if ( (__int64 *)(v8 + 16LL * v6) == v10 )
    goto LABEL_35;
  v12 = v10[1];
  if ( !v12 )
    goto LABEL_35;
  v13 = *(_QWORD *)(v12 + 32);
  if ( v13 )
  {
    v14 = *(_DWORD *)(v13 + 8) != 1;
    v16 = (__int64 *)sub_31BFEB0((__int64)a2, a3, 1);
    v22 = v21;
    if ( v16 == v21 )
      return (unsigned int)v14 + 2;
    goto LABEL_7;
  }
  v16 = (__int64 *)sub_31BFEB0((__int64)a2, a3, 1);
  v22 = v33;
  if ( v33 == v16 )
    return 0;
  v14 = 0;
LABEL_7:
  v23 = v17 - 1;
  do
  {
    while ( 1 )
    {
      v24 = *v16;
      if ( !v17 )
        goto LABEL_17;
      v25 = v23 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v26 = (__int64 *)(v19 + 16LL * v25);
      v27 = *v26;
      if ( v24 == *v26 )
        break;
      v31 = 1;
      while ( v27 != -4096 )
      {
        v32 = v31 + 1;
        v25 = v23 & (v31 + v25);
        v26 = (__int64 *)(v19 + 16LL * v25);
        v27 = *v26;
        if ( v24 == *v26 )
          goto LABEL_14;
        v31 = v32;
      }
LABEL_17:
      if ( v15 )
      {
        if ( *(_DWORD *)(v15 + 8) != 1 )
          return 1;
        goto LABEL_19;
      }
LABEL_11:
      if ( v22 == ++v16 )
        goto LABEL_20;
    }
LABEL_14:
    if ( v18 == v26 )
      goto LABEL_17;
    v28 = v26[1];
    if ( !v28 )
      goto LABEL_17;
    v29 = *(_QWORD *)(v28 + 32);
    if ( !v29 )
      goto LABEL_17;
    if ( *(_DWORD *)(v29 + 8) != 1 )
    {
      if ( v29 != v15 )
        return 1;
      v20 = 0;
      goto LABEL_11;
    }
    v20 = 0;
    if ( v29 != v15 && v15 && *(_DWORD *)(v15 + 8) != 1 )
      return 1;
LABEL_19:
    ++v16;
    v14 = 0;
  }
  while ( v22 != v16 );
LABEL_20:
  result = 0;
  if ( v20 )
    return result;
  return (unsigned int)v14 + 2;
}
