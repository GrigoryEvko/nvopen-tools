// Function: sub_1063100
// Address: 0x1063100
//
__int64 __fastcall sub_1063100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 *v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // r8
  __int64 *v10; // rax
  int v11; // edx
  __int64 *v12; // rsi
  __int64 *v13; // rdx
  __int64 ***v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 *v22; // r14
  unsigned int v23; // esi
  __int64 v24; // r8
  _QWORD *v25; // rcx
  int v26; // r11d
  unsigned int v27; // edx
  _QWORD *v28; // rax
  __int64 v29; // r10
  __int64 *v30; // rax
  int v32; // eax
  int v33; // edx
  __int64 v34; // rax
  int v35; // eax
  int v36; // eax
  __int64 v37; // r9
  unsigned int v38; // esi
  __int64 v39; // r8
  int v40; // r11d
  _QWORD *v41; // r10
  int v42; // eax
  int v43; // eax
  __int64 v44; // r9
  int v45; // r11d
  unsigned int v46; // esi
  __int64 v47; // r8
  __int64 *v48; // [rsp+10h] [rbp-60h] BYREF
  __int64 v49; // [rsp+18h] [rbp-58h]
  _QWORD v50[10]; // [rsp+20h] [rbp-50h] BYREF

  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(__int64 **)(a2 - 32);
    v8 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v8 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    v7 = (__int64 *)(a2 - 8LL * ((v6 >> 2) & 0xF) - 16);
  }
  v48 = v50;
  v9 = v8;
  v49 = 0x400000000LL;
  if ( v8 > 4 )
  {
    sub_C8D5F0((__int64)&v48, v50, v8, 8u, v9 * 8, a6);
    v12 = v48;
    v11 = v49;
    v9 = v8;
    v10 = &v48[(unsigned int)v49];
  }
  else
  {
    v10 = v50;
    v11 = 0;
    v12 = v50;
  }
  if ( v9 * 8 )
  {
    v13 = &v10[v9];
    do
    {
      if ( v10 )
        *v10 = *v7;
      ++v10;
      ++v7;
    }
    while ( v10 != v13 );
    v12 = v48;
    v11 = v49;
  }
  v14 = *(__int64 ****)a1;
  LODWORD(v49) = v11 + v8;
  v15 = sub_B9C770(**v14, v12, (__int64 *)(unsigned int)(v11 + v8), 1, 1);
  if ( v48 != v50 )
    _libc_free(v48, v12);
  v16 = **(_QWORD **)(a1 + 8);
  v17 = *(_BYTE *)(v16 - 16);
  if ( (v17 & 2) != 0 )
    v18 = *(__int64 **)(v16 - 32);
  else
    v18 = (__int64 *)(v16 - 8LL * ((v17 >> 2) & 0xF) - 16);
  v48 = (__int64 *)*v18;
  v19 = **(_QWORD **)(a1 + 16);
  v50[0] = v15;
  v49 = v19;
  v20 = sub_B9C770(***(__int64 ****)a1, (__int64 *)&v48, (__int64 *)3, 1, 1);
  sub_B970B0(**(_QWORD **)(a1 + 24), **(_DWORD **)(a1 + 32), v20);
  v21 = *(_QWORD *)(a1 + 40);
  v22 = *(__int64 **)(a1 + 16);
  v23 = *(_DWORD *)(v21 + 24);
  if ( !v23 )
  {
    ++*(_QWORD *)v21;
    goto LABEL_36;
  }
  v24 = *(_QWORD *)(v21 + 8);
  v25 = 0;
  v26 = 1;
  v27 = (v23 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
  v28 = (_QWORD *)(v24 + 24LL * v27);
  v29 = *v28;
  if ( *v22 == *v28 )
  {
LABEL_17:
    v30 = v28 + 1;
    goto LABEL_18;
  }
  while ( v29 != -4096 )
  {
    if ( !v25 && v29 == -8192 )
      v25 = v28;
    v27 = (v23 - 1) & (v26 + v27);
    v28 = (_QWORD *)(v24 + 24LL * v27);
    v29 = *v28;
    if ( *v22 == *v28 )
      goto LABEL_17;
    ++v26;
  }
  if ( !v25 )
    v25 = v28;
  v32 = *(_DWORD *)(v21 + 16);
  ++*(_QWORD *)v21;
  v33 = v32 + 1;
  if ( 4 * (v32 + 1) >= 3 * v23 )
  {
LABEL_36:
    sub_1062CA0(v21, 2 * v23);
    v35 = *(_DWORD *)(v21 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(v21 + 8);
      v38 = v36 & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
      v25 = (_QWORD *)(v37 + 24LL * v38);
      v39 = *v25;
      v33 = *(_DWORD *)(v21 + 16) + 1;
      if ( *v25 == *v22 )
        goto LABEL_32;
      v40 = 1;
      v41 = 0;
      while ( v39 != -4096 )
      {
        if ( !v41 && v39 == -8192 )
          v41 = v25;
        v38 = v36 & (v40 + v38);
        v25 = (_QWORD *)(v37 + 24LL * v38);
        v39 = *v25;
        if ( *v22 == *v25 )
          goto LABEL_32;
        ++v40;
      }
LABEL_40:
      if ( v41 )
        v25 = v41;
      goto LABEL_32;
    }
LABEL_56:
    ++*(_DWORD *)(v21 + 16);
    BUG();
  }
  if ( v23 - *(_DWORD *)(v21 + 20) - v33 <= v23 >> 3 )
  {
    sub_1062CA0(v21, v23);
    v42 = *(_DWORD *)(v21 + 24);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(v21 + 8);
      v45 = 1;
      v41 = 0;
      v46 = v43 & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
      v25 = (_QWORD *)(v44 + 24LL * v46);
      v47 = *v25;
      v33 = *(_DWORD *)(v21 + 16) + 1;
      if ( *v25 == *v22 )
        goto LABEL_32;
      while ( v47 != -4096 )
      {
        if ( !v41 && v47 == -8192 )
          v41 = v25;
        v46 = v43 & (v45 + v46);
        v25 = (_QWORD *)(v44 + 24LL * v46);
        v47 = *v25;
        if ( *v22 == *v25 )
          goto LABEL_32;
        ++v45;
      }
      goto LABEL_40;
    }
    goto LABEL_56;
  }
LABEL_32:
  *(_DWORD *)(v21 + 16) = v33;
  if ( *v25 != -4096 )
    --*(_DWORD *)(v21 + 20);
  v34 = *v22;
  v25[1] = 0;
  *((_DWORD *)v25 + 4) = 0;
  *v25 = v34;
  v30 = v25 + 1;
LABEL_18:
  *v30 = v20;
  return v15;
}
