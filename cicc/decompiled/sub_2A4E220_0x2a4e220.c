// Function: sub_2A4E220
// Address: 0x2a4e220
//
__int64 __fastcall sub_2A4E220(__int64 a1, __int64 a2)
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
  int v16; // ecx
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // eax
  __int64 v20; // r8
  int v21; // r11d
  _QWORD *v22; // rax
  __int64 v23; // r8
  _DWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rcx
  unsigned int v28; // ecx
  __int64 v29; // r8
  int v30; // eax
  int v31; // r9d
  _BYTE **v32; // rcx
  int v33; // r9d
  int v34; // eax
  __int64 v35; // rax
  _QWORD *v36; // rdi
  int v37; // eax
  int v38; // eax
  int v39; // r9d
  int v40; // r9d
  __int64 v41; // rsi
  unsigned int v42; // edx
  __int64 v43; // rcx
  _QWORD *v44; // r8
  int v45; // r9d
  int v46; // r9d
  __int64 v47; // rsi
  unsigned int v48; // ecx
  __int64 v49; // rdx
  int v50; // r9d
  __int64 v51; // rax
  int v52; // [rsp+8h] [rbp-48h]
  unsigned int v53; // [rsp+8h] [rbp-48h]
  __int64 v54; // [rsp+10h] [rbp-40h]
  unsigned int v55; // [rsp+1Ch] [rbp-34h]
  int v56; // [rsp+1Ch] [rbp-34h]
  int v57; // [rsp+1Ch] [rbp-34h]

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
      v51 = *(_QWORD *)(a2 + 40);
      v12 = *(_QWORD *)(v51 + 56);
      v13 = v51 + 48;
      if ( v12 == v51 + 48 )
        goto LABEL_25;
    }
    else
    {
      v34 = 1;
      while ( v9 != -4096 )
      {
        v50 = v34 + 1;
        v7 = v6 & (v34 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v34 = v50;
      }
      v35 = *(_QWORD *)(a2 + 40);
      v12 = *(_QWORD *)(v35 + 56);
      v13 = v35 + 48;
      if ( v35 + 48 == v12 )
      {
LABEL_25:
        v28 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v8 = (__int64 *)(v5 + 16LL * v28);
        v29 = *v8;
        if ( a2 != *v8 )
        {
          v30 = 1;
          while ( v29 != -4096 )
          {
            v31 = v30 + 1;
            v28 = v6 & (v30 + v28);
            v8 = (__int64 *)(v5 + 16LL * v28);
            v29 = *v8;
            if ( a2 == *v8 )
              return *((unsigned int *)v8 + 2);
            v30 = v31;
          }
          goto LABEL_54;
        }
        return *((unsigned int *)v8 + 2);
      }
    }
  }
  else
  {
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)(v11 + 56);
    v13 = v11 + 48;
    if ( v12 == v11 + 48 )
      goto LABEL_53;
  }
  v54 = a2;
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
      v17 = *(_QWORD *)(v25 + 8);
      if ( v16 )
      {
        v18 = v16 - 1;
        v19 = v18 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v20 = *(_QWORD *)(v17 + 8LL * v19);
        if ( v26 != v20 )
        {
          v33 = 1;
          while ( v20 != -4096 )
          {
            v19 = v18 & (v33 + v19);
            v20 = *(_QWORD *)(v17 + 8LL * v19);
            if ( v26 == v20 )
              goto LABEL_9;
            ++v33;
          }
          goto LABEL_13;
        }
        goto LABEL_9;
      }
LABEL_13:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == v14 )
        goto LABEL_23;
    }
    if ( *(_BYTE *)v26 == 61 )
    {
      if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
        v32 = *(_BYTE ***)(v26 - 8);
      else
        v32 = (_BYTE **)(v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF));
      if ( **v32 != 60 )
        goto LABEL_13;
LABEL_9:
      v21 = v15 + 1;
      if ( (_DWORD)v4 )
      {
        v55 = (v4 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v22 = (_QWORD *)(v5 + 16LL * v55);
        v23 = *v22;
        if ( v26 == *v22 )
        {
LABEL_11:
          v24 = v22 + 1;
LABEL_12:
          *v24 = v15;
          v5 = *(_QWORD *)(a1 + 8);
          v15 = v21;
          v4 = *(unsigned int *)(a1 + 24);
          goto LABEL_13;
        }
        v52 = 1;
        v36 = 0;
        while ( v23 != -4096 )
        {
          if ( v23 == -8192 && !v36 )
            v36 = v22;
          v55 = (v4 - 1) & (v55 + v52);
          v22 = (_QWORD *)(v5 + 16LL * v55);
          v23 = *v22;
          if ( v26 == *v22 )
            goto LABEL_11;
          ++v52;
        }
        if ( !v36 )
          v36 = v22;
        v37 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v38 = v37 + 1;
        if ( 4 * v38 < (unsigned int)(3 * v4) )
        {
          if ( (int)v4 - (v38 + *(_DWORD *)(a1 + 20)) > (unsigned int)v4 >> 3 )
            goto LABEL_50;
          v53 = ((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4);
          sub_A41E30(a1, v4);
          v45 = *(_DWORD *)(a1 + 24);
          if ( !v45 )
          {
LABEL_86:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v46 = v45 - 1;
          v47 = *(_QWORD *)(a1 + 8);
          v21 = v15 + 1;
          v48 = v46 & v53;
          v38 = *(_DWORD *)(a1 + 16) + 1;
          v36 = (_QWORD *)(v47 + 16LL * (v46 & v53));
          v49 = *v36;
          if ( v26 == *v36 )
            goto LABEL_50;
          v57 = 1;
          v44 = 0;
          while ( v49 != -4096 )
          {
            if ( v49 == -8192 && !v44 )
              v44 = v36;
            v48 = v46 & (v57 + v48);
            v36 = (_QWORD *)(v47 + 16LL * v48);
            v49 = *v36;
            if ( v26 == *v36 )
              goto LABEL_50;
            ++v57;
          }
          goto LABEL_65;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      sub_A41E30(a1, 2 * v4);
      v39 = *(_DWORD *)(a1 + 24);
      if ( !v39 )
        goto LABEL_86;
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 8);
      v21 = v15 + 1;
      v42 = v40 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v38 = *(_DWORD *)(a1 + 16) + 1;
      v36 = (_QWORD *)(v41 + 16LL * v42);
      v43 = *v36;
      if ( v26 == *v36 )
        goto LABEL_50;
      v56 = 1;
      v44 = 0;
      while ( v43 != -4096 )
      {
        if ( !v44 && v43 == -8192 )
          v44 = v36;
        v42 = v40 & (v56 + v42);
        v36 = (_QWORD *)(v41 + 16LL * v42);
        v43 = *v36;
        if ( v26 == *v36 )
          goto LABEL_50;
        ++v56;
      }
LABEL_65:
      if ( v44 )
        v36 = v44;
LABEL_50:
      *(_DWORD *)(a1 + 16) = v38;
      if ( *v36 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v36 = v26;
      v24 = v36 + 1;
      *((_DWORD *)v36 + 2) = 0;
      goto LABEL_12;
    }
    if ( *(_BYTE *)v26 != 62 )
      goto LABEL_13;
    if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
      v27 = *(_QWORD *)(v26 - 8);
    else
      v27 = v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
    if ( **(_BYTE **)(v27 + 32) == 60 )
      goto LABEL_9;
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 != v14 );
LABEL_23:
  a2 = v54;
  if ( (_DWORD)v4 )
  {
    v6 = v4 - 1;
    goto LABEL_25;
  }
LABEL_53:
  v4 = 0;
LABEL_54:
  v8 = (__int64 *)(v5 + 16 * v4);
  return *((unsigned int *)v8 + 2);
}
