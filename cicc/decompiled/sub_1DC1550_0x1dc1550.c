// Function: sub_1DC1550
// Address: 0x1dc1550
//
unsigned __int64 __fastcall sub_1DC1550(__int64 a1, unsigned __int64 a2, char a3)
{
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r9
  unsigned int v10; // edi
  __int64 *v11; // rcx
  __int64 v12; // r11
  __int64 v13; // r8
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r8
  unsigned int v22; // r9d
  __int64 *v23; // rcx
  __int64 v24; // r11
  __int64 v25; // rax
  int v26; // r14d
  unsigned int v27; // ecx
  int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // rax
  unsigned int v32; // esi
  unsigned __int64 result; // rax
  __int64 v34; // rdi
  unsigned int v35; // ecx
  unsigned __int64 *v36; // rdx
  unsigned __int64 v37; // r14
  int v38; // ecx
  int v39; // r13d
  __int64 v40; // rcx
  int v41; // ecx
  int v42; // r13d
  int v43; // r15d
  unsigned __int64 *v44; // r10
  int v45; // edi
  int v46; // ecx
  int v47; // r9d
  int v48; // r9d
  __int64 v49; // r11
  unsigned int v50; // edx
  unsigned __int64 v51; // r8
  int v52; // edi
  unsigned __int64 *v53; // rsi
  int v54; // r8d
  int v55; // r8d
  __int64 v56; // r9
  unsigned __int64 *v57; // rdi
  int v58; // edx
  __int64 v59; // r13
  unsigned __int64 v60; // rsi
  __int64 *v61; // [rsp+0h] [rbp-40h]
  unsigned int v62; // [rsp+8h] [rbp-38h]
  unsigned __int64 v63; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 24);
  if ( !a3 )
  {
    v17 = *(_QWORD *)(v5 + 32);
    v18 = a2;
    while ( v17 != v18 )
    {
      v18 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v18 )
LABEL_82:
        BUG();
      v19 = *(_QWORD *)v18;
      if ( (*(_QWORD *)v18 & 4) == 0 && (*(_BYTE *)(v18 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v18 = v19 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
            break;
          v19 = *(_QWORD *)v18;
        }
      }
      v20 = *(unsigned int *)(a1 + 384);
      if ( (_DWORD)v20 )
      {
        v21 = *(_QWORD *)(a1 + 368);
        v22 = (v20 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( v18 == *v23 )
        {
LABEL_24:
          if ( v23 != (__int64 *)(v21 + 16 * v20) )
          {
            v25 = v23[1];
            goto LABEL_27;
          }
        }
        else
        {
          v38 = 1;
          while ( v24 != -8 )
          {
            v39 = v38 + 1;
            v40 = ((_DWORD)v20 - 1) & (v22 + v38);
            v22 = v40;
            v23 = (__int64 *)(v21 + 16 * v40);
            v24 = *v23;
            if ( v18 == *v23 )
              goto LABEL_24;
            v38 = v39;
          }
        }
      }
    }
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 16LL * *(unsigned int *)(v5 + 48));
LABEL_27:
    v16 = v25 & 0xFFFFFFFFFFFFFFF8LL;
    v14 = *(_QWORD *)(v16 + 8);
    v15 = v14;
    goto LABEL_28;
  }
  v6 = a2;
  v7 = v5 + 24;
  if ( (*(_BYTE *)a2 & 4) == 0 )
  {
LABEL_13:
    while ( (*(_BYTE *)(v6 + 46) & 8) != 0 )
      v6 = *(_QWORD *)(v6 + 8);
  }
  v6 = *(_QWORD *)(v6 + 8);
  if ( v7 == v6 )
  {
LABEL_10:
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 16LL * *(unsigned int *)(v5 + 48) + 8);
    goto LABEL_11;
  }
  while ( 1 )
  {
    v8 = *(unsigned int *)(a1 + 384);
    if ( !(_DWORD)v8 )
      goto LABEL_7;
    v9 = *(_QWORD *)(a1 + 368);
    v10 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
      break;
    v41 = 1;
    while ( v12 != -8 )
    {
      v42 = v41 + 1;
      v10 = (v8 - 1) & (v41 + v10);
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == v6 )
        goto LABEL_6;
      v41 = v42;
    }
LABEL_7:
    if ( !v6 )
      goto LABEL_82;
    if ( (*(_BYTE *)v6 & 4) == 0 )
      goto LABEL_13;
    v6 = *(_QWORD *)(v6 + 8);
    if ( v7 == v6 )
      goto LABEL_10;
  }
LABEL_6:
  if ( v11 == (__int64 *)(v9 + 16 * v8) )
    goto LABEL_7;
  v13 = v11[1];
LABEL_11:
  v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = v14;
  v16 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_28:
  v26 = *(_DWORD *)(v16 + 24);
  v61 = (__int64 *)v14;
  v27 = ((unsigned int)(*(_DWORD *)(v14 + 24) - v26) >> 1) & 0xFFFFFFFC;
  v28 = v27 + v26;
  v62 = v27;
  v29 = sub_145CBF0((__int64 *)(a1 + 232), 32, 8);
  *(_QWORD *)(v29 + 8) = 0;
  v30 = v29;
  *(_QWORD *)v29 = 0;
  *(_QWORD *)(v29 + 16) = a2;
  *(_DWORD *)(v29 + 24) = v28;
  v31 = *v61;
  *(_QWORD *)(v30 + 8) = v15;
  v31 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v30 = v31;
  *(_QWORD *)(v31 + 8) = v30;
  *v61 = v30 | *v61 & 7;
  if ( !v62 )
    sub_1F107E0(a1, v30);
  v32 = *(_DWORD *)(a1 + 384);
  result = v30 & 0xFFFFFFFFFFFFFFF9LL;
  if ( !v32 )
  {
    ++*(_QWORD *)(a1 + 360);
    goto LABEL_53;
  }
  v34 = *(_QWORD *)(a1 + 368);
  v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v36 = (unsigned __int64 *)(v34 + 16LL * v35);
  v37 = *v36;
  if ( a2 == *v36 )
    return result;
  v43 = 1;
  v44 = 0;
  while ( v37 != -8 )
  {
    if ( !v44 && v37 == -16 )
      v44 = v36;
    v35 = (v32 - 1) & (v43 + v35);
    v36 = (unsigned __int64 *)(v34 + 16LL * v35);
    v37 = *v36;
    if ( a2 == *v36 )
      return result;
    ++v43;
  }
  v45 = *(_DWORD *)(a1 + 376);
  if ( !v44 )
    v44 = v36;
  ++*(_QWORD *)(a1 + 360);
  v46 = v45 + 1;
  if ( 4 * (v45 + 1) >= 3 * v32 )
  {
LABEL_53:
    sub_1DC1390(a1 + 360, 2 * v32);
    v47 = *(_DWORD *)(a1 + 384);
    if ( v47 )
    {
      v48 = v47 - 1;
      v49 = *(_QWORD *)(a1 + 368);
      v46 = *(_DWORD *)(a1 + 376) + 1;
      result = v30 & 0xFFFFFFFFFFFFFFF9LL;
      v50 = v48 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v44 = (unsigned __int64 *)(v49 + 16LL * v50);
      v51 = *v44;
      if ( a2 != *v44 )
      {
        v52 = 1;
        v53 = 0;
        while ( v51 != -8 )
        {
          if ( !v53 && v51 == -16 )
            v53 = v44;
          v50 = v48 & (v52 + v50);
          v44 = (unsigned __int64 *)(v49 + 16LL * v50);
          v51 = *v44;
          if ( a2 == *v44 )
            goto LABEL_49;
          ++v52;
        }
        if ( v53 )
          v44 = v53;
      }
      goto LABEL_49;
    }
    goto LABEL_81;
  }
  if ( v32 - *(_DWORD *)(a1 + 380) - v46 <= v32 >> 3 )
  {
    v63 = v30 & 0xFFFFFFFFFFFFFFF9LL;
    sub_1DC1390(a1 + 360, v32);
    v54 = *(_DWORD *)(a1 + 384);
    if ( v54 )
    {
      v55 = v54 - 1;
      v56 = *(_QWORD *)(a1 + 368);
      v57 = 0;
      v58 = 1;
      LODWORD(v59) = v55 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v46 = *(_DWORD *)(a1 + 376) + 1;
      result = v63;
      v44 = (unsigned __int64 *)(v56 + 16LL * (unsigned int)v59);
      v60 = *v44;
      if ( a2 != *v44 )
      {
        while ( v60 != -8 )
        {
          if ( !v57 && v60 == -16 )
            v57 = v44;
          v59 = v55 & (unsigned int)(v59 + v58);
          v44 = (unsigned __int64 *)(v56 + 16 * v59);
          v60 = *v44;
          if ( a2 == *v44 )
            goto LABEL_49;
          ++v58;
        }
        if ( v57 )
          v44 = v57;
      }
      goto LABEL_49;
    }
LABEL_81:
    ++*(_DWORD *)(a1 + 376);
    BUG();
  }
LABEL_49:
  *(_DWORD *)(a1 + 376) = v46;
  if ( *v44 != -8 )
    --*(_DWORD *)(a1 + 380);
  *v44 = a2;
  v44[1] = result;
  return result;
}
