// Function: sub_1C055E0
// Address: 0x1c055e0
//
__int64 __fastcall sub_1C055E0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // r9
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rdx
  int v14; // r11d
  unsigned int v15; // r14d
  unsigned int v16; // edi
  unsigned int v17; // r10d
  __int64 *v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rdx
  const char *v21; // rsi
  int v23; // r15d
  __int64 **v24; // r11
  int v25; // r15d
  __int64 v26; // rdi
  int v27; // eax
  int v28; // edx
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  unsigned int v32; // r14d
  __int64 *v33; // rdi
  int v34; // eax
  int v35; // eax
  __int64 v36; // rsi
  unsigned int v37; // r14d
  __int64 **v38; // r10
  const char *v39; // [rsp+0h] [rbp-50h] BYREF
  __int64 v40; // [rsp+8h] [rbp-48h]
  char v41[64]; // [rsp+10h] [rbp-40h] BYREF

  sub_223E0D0(a2, "\"", 1);
  v5 = (char *)sub_1649960(a3);
  if ( v5 )
  {
    v39 = v41;
    sub_1C04B10((__int64 *)&v39, v5, (__int64)&v5[v6]);
    v7 = sub_223E0D0(a2, v39, v40);
  }
  else
  {
    v40 = 0;
    v39 = v41;
    v41[0] = 0;
    v7 = sub_223E0D0(a2, v41, 0);
  }
  sub_223E0D0(v7, "\" is ", 5);
  if ( v39 != v41 )
    j_j___libc_free_0(v39, *(_QWORD *)v41 + 1LL);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 160) + 8LL) + 104LL);
  v12 = *(unsigned int *)(v11 + 64);
  if ( !(_DWORD)v12 )
  {
LABEL_10:
    v20 = 16;
    v21 = "not convergent.\n";
    return sub_223E0D0(a2, v21, v20, v8, v9, v10);
  }
  v9 = (unsigned int)(v12 - 1);
  v13 = *(_QWORD *)(v11 + 48);
  v14 = 1;
  v15 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v16 = v9 & v15;
  v17 = v9 & v15;
  v18 = (__int64 *)(v13 + 16LL * ((unsigned int)v9 & v15));
  v8 = (__int64 *)*v18;
  v10 = (__int64 *)*v18;
  if ( a3 == *v18 )
  {
    if ( v18 != (__int64 *)(v13 + 16 * v12) )
    {
      v19 = (__int64 *)v18[1];
      goto LABEL_9;
    }
    goto LABEL_10;
  }
  while ( 1 )
  {
    if ( v10 == (__int64 *)-8LL )
      goto LABEL_10;
    v23 = v14 + 1;
    v17 = v9 & (v14 + v17);
    v24 = (__int64 **)(v13 + 16LL * v17);
    v10 = *v24;
    if ( (__int64 *)a3 == *v24 )
      break;
    v14 = v23;
  }
  v25 = 1;
  v10 = 0;
  if ( v24 == (__int64 **)(v13 + 16LL * (unsigned int)v12) )
    goto LABEL_10;
  while ( v8 != (__int64 *)-8LL )
  {
    if ( v8 != (__int64 *)-16LL || v10 )
      v18 = v10;
    v10 = (__int64 *)(unsigned int)(v25 + 1);
    v16 = v9 & (v25 + v16);
    v38 = (__int64 **)(v13 + 16LL * v16);
    v8 = *v38;
    if ( (__int64 *)a3 == *v38 )
    {
      v19 = v38[1];
      goto LABEL_9;
    }
    ++v25;
    v10 = v18;
    v18 = (__int64 *)(v13 + 16LL * v16);
  }
  v9 = (unsigned int)(2 * v12);
  v26 = v11 + 40;
  if ( !v10 )
    v10 = v18;
  v27 = *(_DWORD *)(v11 + 56);
  ++*(_QWORD *)(v11 + 40);
  v28 = v27 + 1;
  if ( 4 * (v27 + 1) >= (unsigned int)(3 * v12) )
  {
    sub_1C04E30(v26, v9);
    v29 = *(_DWORD *)(v11 + 64);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v11 + 48);
      v32 = v30 & v15;
      v28 = *(_DWORD *)(v11 + 56) + 1;
      v10 = (__int64 *)(v31 + 16LL * v32);
      v8 = (__int64 *)*v10;
      if ( a3 == *v10 )
        goto LABEL_22;
      v9 = 1;
      v33 = 0;
      while ( v8 != (__int64 *)-8LL )
      {
        if ( !v33 && v8 == (__int64 *)-16LL )
          v33 = v10;
        v32 = v30 & (v9 + v32);
        v10 = (__int64 *)(v31 + 16LL * v32);
        v8 = (__int64 *)*v10;
        if ( a3 == *v10 )
          goto LABEL_22;
        v9 = (unsigned int)(v9 + 1);
      }
LABEL_29:
      if ( v33 )
        v10 = v33;
      goto LABEL_22;
    }
LABEL_51:
    ++*(_DWORD *)(v11 + 56);
    BUG();
  }
  v8 = (__int64 *)((unsigned int)v12 >> 3);
  if ( (int)v12 - *(_DWORD *)(v11 + 60) - v28 <= (unsigned int)v8 )
  {
    sub_1C04E30(v26, v12);
    v34 = *(_DWORD *)(v11 + 64);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v11 + 48);
      v33 = 0;
      v37 = v35 & v15;
      v9 = 1;
      v28 = *(_DWORD *)(v11 + 56) + 1;
      v10 = (__int64 *)(v36 + 16LL * v37);
      v8 = (__int64 *)*v10;
      if ( a3 == *v10 )
        goto LABEL_22;
      while ( v8 != (__int64 *)-8LL )
      {
        if ( !v33 && v8 == (__int64 *)-16LL )
          v33 = v10;
        v37 = v35 & (v9 + v37);
        v10 = (__int64 *)(v36 + 16LL * v37);
        v8 = (__int64 *)*v10;
        if ( a3 == *v10 )
          goto LABEL_22;
        v9 = (unsigned int)(v9 + 1);
      }
      goto LABEL_29;
    }
    goto LABEL_51;
  }
LABEL_22:
  *(_DWORD *)(v11 + 56) = v28;
  if ( *v10 != -8 )
    --*(_DWORD *)(v11 + 60);
  *v10 = a3;
  v19 = 0;
  v10[1] = 0;
LABEL_9:
  v20 = 12;
  v21 = "convergent.\n";
  if ( *((_DWORD *)v19 + 3) )
    goto LABEL_10;
  return sub_223E0D0(a2, v21, v20, v8, v9, v10);
}
