// Function: sub_1A73B00
// Address: 0x1a73b00
//
__int64 __fastcall sub_1A73B00(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned __int64 v6; // r14
  __int64 v7; // rcx
  _QWORD *v8; // rax
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r15
  __int64 *v17; // rax
  __int64 *v18; // rbx
  int v19; // eax
  _BYTE *v20; // rsi
  __int64 v21; // r10
  unsigned int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r14
  __int64 v28; // rdi
  int v30; // eax
  int v31; // esi
  __int64 v32; // rdi
  unsigned int v33; // eax
  int v34; // ecx
  __int64 *v35; // rdx
  __int64 v36; // r8
  int v37; // r10d
  __int64 *v38; // r9
  int v39; // edx
  __int64 *v40; // rax
  int v41; // r11d
  int v42; // eax
  int v43; // r9d
  int v44; // eax
  int v45; // eax
  __int64 v46; // rdi
  __int64 *v47; // r8
  unsigned int v48; // r15d
  int v49; // r9d
  __int64 v50; // rsi
  __int64 v51; // [rsp+8h] [rbp-58h]
  _QWORD v52[2]; // [rsp+10h] [rbp-50h] BYREF
  char v53; // [rsp+20h] [rbp-40h]
  char v54; // [rsp+21h] [rbp-3Fh]

  v4 = sub_15E0530(*(_QWORD *)(a1 + 192));
  v5 = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)v5 )
    v6 = **(_QWORD **)(*(_QWORD *)(a1 + 232) + 8 * v5 - 8) & 0xFFFFFFFFFFFFFFF8LL;
  else
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 32LL);
  v7 = *(_QWORD *)(a1 + 192);
  v54 = 1;
  v52[0] = "Flow";
  v51 = v7;
  v53 = 3;
  v8 = (_QWORD *)sub_22077B0(64);
  v9 = (__int64)v8;
  if ( v8 )
    sub_157FB60(v8, v4, (__int64)v52, v51, v6);
  v10 = *(_QWORD *)(a1 + 216);
  v11 = *(unsigned int *)(v10 + 48);
  if ( !(_DWORD)v11 )
    goto LABEL_35;
  v12 = *(_QWORD *)(v10 + 32);
  v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( a2 != *v14 )
  {
    v39 = 1;
    while ( v15 != -8 )
    {
      v43 = v39 + 1;
      v13 = (v11 - 1) & (v39 + v13);
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_7;
      v39 = v43;
    }
    goto LABEL_35;
  }
LABEL_7:
  if ( v14 == (__int64 *)(v12 + 16 * v11) )
  {
LABEL_35:
    *(_BYTE *)(v10 + 72) = 0;
    v40 = (__int64 *)sub_22077B0(56);
    v18 = v40;
    if ( !v40 )
    {
      v52[0] = 0;
      BUG();
    }
    *v40 = v9;
    v16 = 0;
    v19 = 0;
    v18[1] = 0;
    goto LABEL_11;
  }
  v16 = v14[1];
  *(_BYTE *)(v10 + 72) = 0;
  v17 = (__int64 *)sub_22077B0(56);
  v18 = v17;
  if ( !v17 )
    goto LABEL_12;
  *v17 = v9;
  v17[1] = v16;
  if ( v16 )
    v19 = *(_DWORD *)(v16 + 16) + 1;
  else
    v19 = 0;
LABEL_11:
  *((_DWORD *)v18 + 4) = v19;
  v18[3] = 0;
  v18[4] = 0;
  v18[5] = 0;
  v18[6] = -1;
LABEL_12:
  v52[0] = v18;
  v20 = *(_BYTE **)(v16 + 32);
  if ( v20 == *(_BYTE **)(v16 + 40) )
  {
    sub_15CE310(v16 + 24, v20, v52);
    v22 = *(_DWORD *)(v10 + 48);
    v21 = v10 + 24;
    if ( v22 )
      goto LABEL_16;
LABEL_24:
    ++*(_QWORD *)(v10 + 24);
    goto LABEL_25;
  }
  if ( v20 )
  {
    *(_QWORD *)v20 = v18;
    v20 = *(_BYTE **)(v16 + 32);
  }
  v21 = v10 + 24;
  *(_QWORD *)(v16 + 32) = v20 + 8;
  v22 = *(_DWORD *)(v10 + 48);
  if ( !v22 )
    goto LABEL_24;
LABEL_16:
  v23 = *(_QWORD *)(v10 + 32);
  v24 = (v22 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( v9 == *v25 )
  {
LABEL_17:
    v27 = v25[1];
    v25[1] = (__int64)v18;
    if ( v27 )
    {
      v28 = *(_QWORD *)(v27 + 24);
      if ( v28 )
        j_j___libc_free_0(v28, *(_QWORD *)(v27 + 40) - v28);
      j_j___libc_free_0(v27, 56);
    }
    goto LABEL_21;
  }
  v41 = 1;
  v35 = 0;
  while ( v26 != -8 )
  {
    if ( !v35 && v26 == -16 )
      v35 = v25;
    v24 = (v22 - 1) & (v41 + v24);
    v25 = (__int64 *)(v23 + 16LL * v24);
    v26 = *v25;
    if ( v9 == *v25 )
      goto LABEL_17;
    ++v41;
  }
  if ( !v35 )
    v35 = v25;
  v42 = *(_DWORD *)(v10 + 40);
  ++*(_QWORD *)(v10 + 24);
  v34 = v42 + 1;
  if ( 4 * (v42 + 1) >= 3 * v22 )
  {
LABEL_25:
    sub_15CFCF0(v21, 2 * v22);
    v30 = *(_DWORD *)(v10 + 48);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(v10 + 32);
      v33 = (v30 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v34 = *(_DWORD *)(v10 + 40) + 1;
      v35 = (__int64 *)(v32 + 16LL * v33);
      v36 = *v35;
      if ( *v35 != v9 )
      {
        v37 = 1;
        v38 = 0;
        while ( v36 != -8 )
        {
          if ( !v38 && v36 == -16 )
            v38 = v35;
          v33 = v31 & (v37 + v33);
          v35 = (__int64 *)(v32 + 16LL * v33);
          v36 = *v35;
          if ( v9 == *v35 )
            goto LABEL_43;
          ++v37;
        }
        if ( v38 )
          v35 = v38;
      }
      goto LABEL_43;
    }
    goto LABEL_70;
  }
  if ( v22 - *(_DWORD *)(v10 + 44) - v34 <= v22 >> 3 )
  {
    sub_15CFCF0(v21, v22);
    v44 = *(_DWORD *)(v10 + 48);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(v10 + 32);
      v47 = 0;
      v48 = v45 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v49 = 1;
      v34 = *(_DWORD *)(v10 + 40) + 1;
      v35 = (__int64 *)(v46 + 16LL * v48);
      v50 = *v35;
      if ( v9 != *v35 )
      {
        while ( v50 != -8 )
        {
          if ( !v47 && v50 == -16 )
            v47 = v35;
          v48 = v45 & (v49 + v48);
          v35 = (__int64 *)(v46 + 16LL * v48);
          v50 = *v35;
          if ( v9 == *v35 )
            goto LABEL_43;
          ++v49;
        }
        if ( v47 )
          v35 = v47;
      }
      goto LABEL_43;
    }
LABEL_70:
    ++*(_DWORD *)(v10 + 40);
    BUG();
  }
LABEL_43:
  *(_DWORD *)(v10 + 40) = v34;
  if ( *v35 != -8 )
    --*(_DWORD *)(v10 + 44);
  *v35 = v9;
  v35[1] = (__int64)v18;
LABEL_21:
  sub_1448350(*(_QWORD *)(*(_QWORD *)(a1 + 200) + 16LL), v9, *(_QWORD *)(a1 + 200));
  return v9;
}
