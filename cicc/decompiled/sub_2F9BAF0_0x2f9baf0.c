// Function: sub_2F9BAF0
// Address: 0x2f9baf0
//
__int64 __fastcall sub_2F9BAF0(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  char v8; // dl
  _BYTE *v9; // rdi
  int v10; // eax
  __int64 v11; // r12
  char v12; // si
  __int64 v13; // rdi
  int v14; // r8d
  unsigned int v15; // edx
  __int64 *v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // r13d
  __int64 v23; // rdx
  __int64 v24; // rdi
  char v25; // dl
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rcx
  __int64 v36; // rcx
  __int64 v37; // rax
  _BYTE *v38; // rdi
  char v39; // r9
  __int64 v40; // r8
  int v41; // ecx
  unsigned int v42; // esi
  _QWORD *v43; // r10
  _BYTE *v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rcx
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rcx
  int v53; // ecx
  int v54; // r9d
  __int64 v55; // r10
  int v56; // r10d
  int v57; // r11d
  __int64 v58; // [rsp+8h] [rbp-38h]

  v8 = *(_BYTE *)(a1 + 8);
  v9 = *(_BYTE **)a1;
  v10 = (unsigned __int8)*v9;
  if ( a2 )
  {
    if ( !v8 )
    {
LABEL_3:
      if ( (_BYTE)v10 == 86 )
      {
        v11 = *((_QWORD *)v9 - 8);
        goto LABEL_5;
      }
      if ( (unsigned int)(v10 - 42) <= 0x11 )
        goto LABEL_25;
      goto LABEL_96;
    }
  }
  else if ( v8 )
  {
    goto LABEL_3;
  }
  if ( (_BYTE)v10 == 86 )
  {
    v11 = *((_QWORD *)v9 - 4);
LABEL_5:
    if ( v11 )
      goto LABEL_6;
    goto LABEL_25;
  }
  if ( (unsigned int)(v10 - 42) > 0x11 )
LABEL_96:
    BUG();
  v11 = *(_QWORD *)&v9[32 * (1 - *(_DWORD *)(a1 + 12)) - 64];
  if ( v11 )
  {
LABEL_6:
    if ( *(_BYTE *)v11 <= 0x1Cu )
      return v11;
    v12 = *(_BYTE *)(a3 + 8) & 1;
    if ( v12 )
    {
      v13 = a3 + 16;
      v14 = 1;
    }
    else
    {
      v20 = *(unsigned int *)(a3 + 24);
      v13 = *(_QWORD *)(a3 + 16);
      if ( !(_DWORD)v20 )
        goto LABEL_74;
      v14 = v20 - 1;
    }
    v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = (__int64 *)(v13 + 24LL * v15);
    v17 = *v16;
    if ( v11 == *v16 )
    {
LABEL_10:
      v18 = 48;
      if ( !v12 )
        v18 = 24LL * *(unsigned int *)(a3 + 24);
      if ( v16 != (__int64 *)(v13 + v18) )
      {
        v11 = v16[2];
        if ( a2 )
          return v16[1];
      }
      return v11;
    }
    v53 = 1;
    while ( v17 != -4096 )
    {
      v54 = v53 + 1;
      v15 = v14 & (v53 + v15);
      v16 = (__int64 *)(v13 + 24LL * v15);
      v17 = *v16;
      if ( v11 == *v16 )
        goto LABEL_10;
      v53 = v54;
    }
    if ( v12 )
    {
      v52 = 48;
      goto LABEL_75;
    }
    v20 = *(unsigned int *)(a3 + 24);
LABEL_74:
    v52 = 24 * v20;
LABEL_75:
    v16 = (__int64 *)(v13 + v52);
    goto LABEL_10;
  }
LABEL_25:
  v21 = sub_B47F80(v9);
  v22 = *(_DWORD *)(a1 + 12);
  v11 = v21;
  if ( (*(_BYTE *)(v21 + 7) & 0x40) != 0 )
    v23 = *(_QWORD *)(v21 - 8);
  else
    v23 = v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF);
  v24 = *(_QWORD *)(v21 + 8);
  v58 = 32LL * v22;
  v25 = **(_BYTE **)(v23 + v58);
  if ( v25 == 55 || v25 == 68 )
  {
    v31 = sub_AD64C0(v24, 1, 0);
    if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
      v32 = *(_QWORD *)(v11 - 8);
    else
      v32 = v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
    v33 = v58 + v32;
    if ( *(_QWORD *)v33 )
    {
      v34 = *(_QWORD *)(v33 + 8);
      **(_QWORD **)(v33 + 16) = v34;
      if ( v34 )
        *(_QWORD *)(v34 + 16) = *(_QWORD *)(v33 + 16);
    }
    *(_QWORD *)v33 = v31;
    if ( v31 )
    {
      v35 = *(_QWORD *)(v31 + 16);
      *(_QWORD *)(v33 + 8) = v35;
      if ( v35 )
        *(_QWORD *)(v35 + 16) = v33 + 8;
      *(_QWORD *)(v33 + 16) = v31 + 16;
      *(_QWORD *)(v31 + 16) = v33;
    }
  }
  else
  {
    v26 = sub_AD64C0(v24, -1, 0);
    if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
      v27 = *(_QWORD *)(v11 - 8);
    else
      v27 = v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
    v28 = v27 + v58;
    if ( *(_QWORD *)(v27 + 32LL * v22) )
    {
      v29 = *(_QWORD *)(v28 + 8);
      **(_QWORD **)(v28 + 16) = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = *(_QWORD *)(v28 + 16);
    }
    *(_QWORD *)v28 = v26;
    if ( v26 )
    {
      v30 = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v28 + 8) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = v28 + 8;
      *(_QWORD *)(v28 + 16) = v26 + 16;
      *(_QWORD *)(v26 + 16) = v28;
    }
  }
  if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
    v36 = *(_QWORD *)(v11 - 8);
  else
    v36 = v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
  v37 = v36 + 32LL * (1 - v22);
  v38 = *(_BYTE **)v37;
  if ( **(_BYTE **)v37 <= 0x1Cu )
    goto LABEL_65;
  v39 = *(_BYTE *)(a3 + 8) & 1;
  if ( v39 )
  {
    v40 = a3 + 16;
    v41 = 1;
  }
  else
  {
    v51 = *(unsigned int *)(a3 + 24);
    v40 = *(_QWORD *)(a3 + 16);
    if ( !(_DWORD)v51 )
      goto LABEL_86;
    v41 = v51 - 1;
  }
  v42 = v41 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
  v43 = (_QWORD *)(v40 + 24LL * v42);
  v44 = (_BYTE *)*v43;
  if ( v38 != (_BYTE *)*v43 )
  {
    v56 = 1;
    while ( v44 != (_BYTE *)-4096LL )
    {
      v57 = v56 + 1;
      v42 = v41 & (v56 + v42);
      v43 = (_QWORD *)(v40 + 24LL * v42);
      v44 = (_BYTE *)*v43;
      if ( v38 == (_BYTE *)*v43 )
        goto LABEL_54;
      v56 = v57;
    }
    if ( v39 )
    {
      v55 = 48;
      goto LABEL_87;
    }
    v51 = *(unsigned int *)(a3 + 24);
LABEL_86:
    v55 = 24 * v51;
LABEL_87:
    v43 = (_QWORD *)(v40 + v55);
  }
LABEL_54:
  v45 = 48;
  if ( !v39 )
    v45 = 24LL * *(unsigned int *)(a3 + 24);
  if ( v43 != (_QWORD *)(v40 + v45) )
  {
    v46 = *(_QWORD *)(v37 + 8);
    v47 = v43[2];
    if ( a2 )
      v47 = v43[1];
    **(_QWORD **)(v37 + 16) = v46;
    if ( v46 )
      *(_QWORD *)(v46 + 16) = *(_QWORD *)(v37 + 16);
    *(_QWORD *)v37 = v47;
    if ( v47 )
    {
      v48 = *(_QWORD *)(v47 + 16);
      *(_QWORD *)(v37 + 8) = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = v37 + 8;
      *(_QWORD *)(v37 + 16) = v47 + 16;
      *(_QWORD *)(v47 + 16) = v37;
    }
  }
LABEL_65:
  v49 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v49 == a4 + 48 )
  {
    v50 = 0;
  }
  else
  {
    if ( !v49 )
      BUG();
    v50 = v49 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v49 - 24) - 30 >= 0xB )
      v50 = 0;
  }
  sub_B44220((_QWORD *)v11, v50 + 24, 0);
  return v11;
}
