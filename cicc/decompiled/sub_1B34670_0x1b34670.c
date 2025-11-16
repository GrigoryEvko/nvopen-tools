// Function: sub_1B34670
// Address: 0x1b34670
//
__int64 __fastcall sub_1B34670(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  int v6; // edi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r11
  __int64 v14; // r14
  int v15; // r15d
  int v16; // eax
  __int64 v17; // r8
  int v18; // eax
  unsigned int v19; // ecx
  __int64 v20; // rdi
  int v21; // r8d
  unsigned int v22; // r9d
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rbx
  char v27; // al
  __int64 v28; // rax
  unsigned int v29; // ecx
  __int64 v30; // r8
  int v31; // eax
  int v32; // r9d
  __int64 v33; // rax
  int v34; // r9d
  int v35; // eax
  __int64 v36; // rax
  int v37; // edx
  int v38; // edx
  int v39; // r11d
  int v40; // r11d
  __int64 v41; // rdi
  unsigned int v42; // ecx
  __int64 v43; // rsi
  __int64 *v44; // r9
  int v45; // r11d
  int v46; // r11d
  __int64 v47; // rdi
  unsigned int v48; // ecx
  __int64 v49; // rsi
  int v50; // r9d
  int v51; // [rsp+8h] [rbp-48h]
  unsigned int v52; // [rsp+8h] [rbp-48h]
  __int64 *v53; // [rsp+10h] [rbp-40h]
  int v54; // [rsp+10h] [rbp-40h]
  int v55; // [rsp+10h] [rbp-40h]
  __int64 v56; // [rsp+18h] [rbp-38h]

  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v5 + 16LL * (unsigned int)v4) )
        return *((unsigned int *)v8 + 2);
    }
    else
    {
      v35 = 1;
      while ( v9 != -8 )
      {
        v50 = v35 + 1;
        v7 = v6 & (v35 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v35 = v50;
      }
    }
    v36 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)(v36 + 48);
    v13 = v36 + 40;
    if ( v36 + 40 == v12 )
    {
LABEL_24:
      v29 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v5 + 16LL * v29);
      v30 = *v8;
      if ( a2 != *v8 )
      {
        v31 = 1;
        while ( v30 != -8 )
        {
          v32 = v31 + 1;
          v29 = v6 & (v31 + v29);
          v8 = (__int64 *)(v5 + 16LL * v29);
          v30 = *v8;
          if ( a2 == *v8 )
            return *((unsigned int *)v8 + 2);
          v31 = v32;
        }
        goto LABEL_36;
      }
      return *((unsigned int *)v8 + 2);
    }
  }
  else
  {
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)(v11 + 48);
    v13 = v11 + 40;
    if ( v11 + 40 == v12 )
      goto LABEL_35;
  }
  v56 = a2;
  v14 = v13;
  v15 = 0;
  do
  {
    while ( 1 )
    {
      v25 = *(_QWORD *)(a1 + 32);
      v26 = v12 - 24;
      if ( !v12 )
        v26 = 0;
      if ( !v25 )
        break;
      v16 = *(_DWORD *)(v25 + 24);
      if ( v16 )
      {
        v17 = *(_QWORD *)(v25 + 8);
        v18 = v16 - 1;
        v19 = v18 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v20 = *(_QWORD *)(v17 + 8LL * v19);
        if ( v26 != v20 )
        {
          v34 = 1;
          while ( v20 != -8 )
          {
            v19 = v18 & (v34 + v19);
            v20 = *(_QWORD *)(v17 + 8LL * v19);
            if ( v26 == v20 )
              goto LABEL_9;
            ++v34;
          }
          goto LABEL_12;
        }
        goto LABEL_9;
      }
LABEL_12:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == v14 )
        goto LABEL_22;
    }
    v27 = *(_BYTE *)(v26 + 16);
    if ( v27 == 54 )
    {
      if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
        v33 = *(_QWORD *)(v26 - 8);
      else
        v33 = v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF);
      if ( *(_BYTE *)(*(_QWORD *)v33 + 16LL) != 53 )
        goto LABEL_12;
LABEL_9:
      v21 = v15 + 1;
      if ( (_DWORD)v4 )
      {
        v22 = (v4 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v23 = (__int64 *)(v5 + 16LL * v22);
        v24 = *v23;
        if ( v26 == *v23 )
        {
LABEL_11:
          *((_DWORD *)v23 + 2) = v15;
          v5 = *(_QWORD *)(a1 + 8);
          v15 = v21;
          v4 = *(unsigned int *)(a1 + 24);
          goto LABEL_12;
        }
        v51 = 1;
        v53 = 0;
        while ( v24 != -8 )
        {
          if ( v24 == -16 )
          {
            if ( v53 )
              v23 = v53;
            v53 = v23;
          }
          v22 = (v4 - 1) & (v51 + v22);
          v23 = (__int64 *)(v5 + 16LL * v22);
          v24 = *v23;
          if ( v26 == *v23 )
            goto LABEL_11;
          ++v51;
        }
        if ( v53 )
          v23 = v53;
        v37 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v38 = v37 + 1;
        if ( 4 * v38 < (unsigned int)(3 * v4) )
        {
          if ( (int)v4 - (v38 + *(_DWORD *)(a1 + 20)) > (unsigned int)v4 >> 3 )
            goto LABEL_51;
          v52 = ((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4);
          sub_1541C50(a1, v4);
          v45 = *(_DWORD *)(a1 + 24);
          if ( !v45 )
          {
LABEL_84:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v46 = v45 - 1;
          v47 = *(_QWORD *)(a1 + 8);
          v21 = v15 + 1;
          v48 = v46 & v52;
          v38 = *(_DWORD *)(a1 + 16) + 1;
          v23 = (__int64 *)(v47 + 16LL * (v46 & v52));
          v49 = *v23;
          if ( v26 == *v23 )
            goto LABEL_51;
          v55 = 1;
          v44 = 0;
          while ( v49 != -8 )
          {
            if ( !v44 && v49 == -16 )
              v44 = v23;
            v48 = v46 & (v55 + v48);
            v23 = (__int64 *)(v47 + 16LL * v48);
            v49 = *v23;
            if ( v26 == *v23 )
              goto LABEL_51;
            ++v55;
          }
          goto LABEL_65;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      sub_1541C50(a1, 2 * v4);
      v39 = *(_DWORD *)(a1 + 24);
      if ( !v39 )
        goto LABEL_84;
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 8);
      v21 = v15 + 1;
      v42 = v40 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v38 = *(_DWORD *)(a1 + 16) + 1;
      v23 = (__int64 *)(v41 + 16LL * v42);
      v43 = *v23;
      if ( v26 == *v23 )
        goto LABEL_51;
      v54 = 1;
      v44 = 0;
      while ( v43 != -8 )
      {
        if ( v43 == -16 && !v44 )
          v44 = v23;
        v42 = v40 & (v54 + v42);
        v23 = (__int64 *)(v41 + 16LL * v42);
        v43 = *v23;
        if ( v26 == *v23 )
          goto LABEL_51;
        ++v54;
      }
LABEL_65:
      if ( v44 )
        v23 = v44;
LABEL_51:
      *(_DWORD *)(a1 + 16) = v38;
      if ( *v23 != -8 )
        --*(_DWORD *)(a1 + 20);
      *v23 = v26;
      *((_DWORD *)v23 + 2) = 0;
      goto LABEL_11;
    }
    if ( v27 != 55 )
      goto LABEL_12;
    if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
      v28 = *(_QWORD *)(v26 - 8);
    else
      v28 = v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF);
    if ( *(_BYTE *)(*(_QWORD *)(v28 + 24) + 16LL) == 53 )
      goto LABEL_9;
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 != v14 );
LABEL_22:
  a2 = v56;
  if ( (_DWORD)v4 )
  {
    v6 = v4 - 1;
    goto LABEL_24;
  }
LABEL_35:
  v4 = 0;
LABEL_36:
  v8 = (__int64 *)(v5 + 16 * v4);
  return *((unsigned int *)v8 + 2);
}
