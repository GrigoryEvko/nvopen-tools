// Function: sub_193EB70
// Address: 0x193eb70
//
unsigned __int64 __fastcall sub_193EB70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r12d
  __int64 v6; // rbx
  unsigned __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // r13
  char v10; // r14
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // r10d
  unsigned int v19; // r8d
  __int64 *v20; // rsi
  __int64 v21; // r11
  __int64 *v22; // rdx
  unsigned int v23; // esi
  __int64 *v24; // rax
  __int64 v25; // r11
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r8
  int v29; // ecx
  __int64 v30; // rsi
  __int64 v31; // rdi
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r10
  __int64 v35; // rax
  __int64 v36; // r10
  __int64 v37; // r9
  unsigned int v38; // edi
  __int64 *v39; // rdx
  __int64 v40; // rbx
  __int64 *v41; // r9
  int v42; // r10d
  __int64 v43; // rdi
  __int64 v44; // rax
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r11
  int v49; // eax
  int v50; // eax
  int v51; // esi
  int v52; // ecx
  int v53; // edx
  int v54; // r12d
  int v55; // eax
  int v56; // ebx
  int v57; // ecx
  int v58; // r11d
  __int64 v61; // [rsp+18h] [rbp-38h]

  v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v5 )
  {
    v6 = 0;
    v7 = 0;
    v8 = 24LL * v5;
    v9 = 8LL * v5;
    v10 = *(_BYTE *)(a1 + 23) & 0x40;
    v61 = a1 - v8;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = v61;
        if ( v10 )
          v12 = *(_QWORD *)(a1 - 8);
        if ( a2 == *(_QWORD *)(v12 + 3 * v6) )
          break;
        v6 += 8;
        if ( v9 == v6 )
          goto LABEL_26;
      }
      v13 = *(_QWORD *)(v6 + v12 + 24LL * *(unsigned int *)(a1 + 56) + 8);
      if ( !v7 )
        goto LABEL_25;
      v14 = *(_QWORD *)(v7 + 40);
      v15 = *(_QWORD *)(*(_QWORD *)(v14 + 56) + 80LL);
      if ( v15 )
        v15 -= 24;
      if ( v14 != v15 && v13 != v15 )
        break;
LABEL_24:
      v13 = v15;
LABEL_25:
      v6 += 8;
      v7 = sub_157EBA0(v13);
      if ( v9 == v6 )
        goto LABEL_26;
    }
    v16 = *(unsigned int *)(a3 + 48);
    v17 = *(_QWORD *)(a3 + 32);
    if ( !(_DWORD)v16 )
    {
LABEL_43:
      v15 = 0;
      goto LABEL_24;
    }
    v18 = v16 - 1;
    v19 = (v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v20 = (__int64 *)(v17 + 16LL * v19);
    v21 = *v20;
    if ( v14 == *v20 )
    {
LABEL_14:
      v22 = (__int64 *)(v17 + 16 * v16);
      if ( v22 != v20 )
      {
        v15 = v20[1];
        goto LABEL_16;
      }
    }
    else
    {
      v51 = 1;
      while ( v21 != -8 )
      {
        v57 = v51 + 1;
        v19 = v18 & (v51 + v19);
        v20 = (__int64 *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( v14 == *v20 )
          goto LABEL_14;
        v51 = v57;
      }
      v22 = (__int64 *)(v17 + 16 * v16);
    }
    v15 = 0;
LABEL_16:
    v23 = v18 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v24 = (__int64 *)(v17 + 16LL * v23);
    v25 = *v24;
    if ( v13 == *v24 )
    {
LABEL_17:
      if ( v22 != v24 )
      {
        v26 = v24[1];
        if ( v15 )
        {
          if ( v26 )
          {
            while ( v26 != v15 )
            {
              if ( *(_DWORD *)(v15 + 16) < *(_DWORD *)(v26 + 16) )
              {
                v27 = v15;
                v15 = v26;
                v26 = v27;
              }
              v15 = *(_QWORD *)(v15 + 8);
              if ( !v15 )
                goto LABEL_24;
            }
            v15 = *(_QWORD *)v15;
            goto LABEL_24;
          }
        }
      }
    }
    else
    {
      v49 = 1;
      while ( v25 != -8 )
      {
        v52 = v49 + 1;
        v23 = v18 & (v49 + v23);
        v24 = (__int64 *)(v17 + 16LL * v23);
        v25 = *v24;
        if ( v13 == *v24 )
          goto LABEL_17;
        v49 = v52;
      }
    }
    goto LABEL_43;
  }
  v7 = 0;
LABEL_26:
  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    return v7;
  v28 = 0;
  v29 = *(_DWORD *)(a4 + 24);
  v30 = *(_QWORD *)(a4 + 8);
  if ( v29 )
  {
    v31 = *(_QWORD *)(a2 + 40);
    v32 = (v29 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v33 = (__int64 *)(v30 + 16LL * v32);
    v34 = *v33;
    if ( v31 == *v33 )
    {
LABEL_29:
      v28 = v33[1];
    }
    else
    {
      v55 = 1;
      while ( v34 != -8 )
      {
        v58 = v55 + 1;
        v32 = (v29 - 1) & (v55 + v32);
        v33 = (__int64 *)(v30 + 16LL * v32);
        v34 = *v33;
        if ( v31 == *v33 )
          goto LABEL_29;
        v55 = v58;
      }
      v28 = 0;
    }
  }
  v35 = *(unsigned int *)(a3 + 48);
  if ( !(_DWORD)v35 )
LABEL_68:
    BUG();
  v36 = *(_QWORD *)(v7 + 40);
  v37 = *(_QWORD *)(a3 + 32);
  v38 = (v35 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
  v39 = (__int64 *)(v37 + 16LL * v38);
  v40 = *v39;
  if ( v36 != *v39 )
  {
    v53 = 1;
    while ( v40 != -8 )
    {
      v54 = v53 + 1;
      v38 = (v35 - 1) & (v53 + v38);
      v39 = (__int64 *)(v37 + 16LL * v38);
      v40 = *v39;
      if ( v36 == *v39 )
        goto LABEL_32;
      v53 = v54;
    }
    goto LABEL_68;
  }
LABEL_32:
  if ( v39 == (__int64 *)(v37 + 16 * v35) )
    goto LABEL_68;
  v41 = (__int64 *)v39[1];
  v42 = v29 - 1;
  while ( 1 )
  {
    v43 = *v41;
    v44 = 0;
    if ( v29 )
    {
      v45 = v42 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v46 = (__int64 *)(v30 + 16LL * v45);
      v47 = *v46;
      if ( v43 == *v46 )
      {
LABEL_37:
        v44 = v46[1];
      }
      else
      {
        v50 = 1;
        while ( v47 != -8 )
        {
          v56 = v50 + 1;
          v45 = v42 & (v50 + v45);
          v46 = (__int64 *)(v30 + 16LL * v45);
          v47 = *v46;
          if ( v43 == *v46 )
            goto LABEL_37;
          v50 = v56;
        }
        v44 = 0;
      }
    }
    if ( v28 == v44 )
      return sub_157EBA0(v43);
    v41 = (__int64 *)v41[1];
  }
}
