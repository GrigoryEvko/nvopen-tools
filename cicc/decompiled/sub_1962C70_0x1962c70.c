// Function: sub_1962C70
// Address: 0x1962c70
//
void __fastcall sub_1962C70(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v3; // r14
  __int64 v6; // rbx
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // eax
  _QWORD *v10; // r9
  __int64 v11; // r8
  __int64 v12; // r15
  char v13; // dl
  unsigned int v14; // esi
  unsigned int v15; // eax
  _QWORD *v16; // rdi
  int v17; // ecx
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // r15
  __int64 v21; // rax
  int v22; // r10d
  int v23; // edx
  __int64 v24; // rsi
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rcx
  int v28; // edx
  __int64 v29; // rsi
  int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rcx
  int v33; // r9d
  _QWORD *v34; // r8
  int v35; // r9d

  if ( a2 != a3 )
  {
    v3 = (const void *)(a1 + 96);
    v6 = a2;
    while ( 1 )
    {
      v12 = sub_1648700(v6)[5];
      v13 = *(_BYTE *)(a1 + 8) & 1;
      if ( v13 )
      {
        v7 = a1 + 16;
        v8 = 7;
      }
      else
      {
        v14 = *(_DWORD *)(a1 + 24);
        v7 = *(_QWORD *)(a1 + 16);
        if ( !v14 )
        {
          v15 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v16 = 0;
          v17 = (v15 >> 1) + 1;
          goto LABEL_12;
        }
        v8 = v14 - 1;
      }
      v9 = v8 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = (_QWORD *)(v7 + 8LL * v9);
      v11 = *v10;
      if ( v12 != *v10 )
        break;
      do
LABEL_5:
        v6 = *(_QWORD *)(v6 + 8);
      while ( v6 && (unsigned __int8)(*((_BYTE *)sub_1648700(v6) + 16) - 25) > 9u );
      if ( a3 == v6 )
        return;
    }
    v22 = 1;
    v16 = 0;
    while ( v11 != -8 )
    {
      if ( v16 || v11 != -16 )
        v10 = v16;
      v9 = v8 & (v22 + v9);
      v11 = *(_QWORD *)(v7 + 8LL * v9);
      if ( v12 == v11 )
        goto LABEL_5;
      ++v22;
      v16 = v10;
      v10 = (_QWORD *)(v7 + 8LL * v9);
    }
    v15 = *(_DWORD *)(a1 + 8);
    if ( !v16 )
      v16 = v10;
    ++*(_QWORD *)a1;
    v17 = (v15 >> 1) + 1;
    if ( v13 )
    {
      v14 = 8;
      if ( (unsigned int)(4 * v17) >= 0x18 )
      {
LABEL_26:
        sub_19628C0(a1, 2 * v14);
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v24 = a1 + 16;
          v25 = 7;
        }
        else
        {
          v23 = *(_DWORD *)(a1 + 24);
          v24 = *(_QWORD *)(a1 + 16);
          if ( !v23 )
            goto LABEL_60;
          v25 = v23 - 1;
        }
        v26 = v25 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v16 = (_QWORD *)(v24 + 8LL * v26);
        v27 = *v16;
        if ( v12 == *v16 )
          goto LABEL_30;
        v35 = 1;
        v34 = 0;
        while ( v27 != -8 )
        {
          if ( v27 == -16 && !v34 )
            v34 = v16;
          v26 = v25 & (v35 + v26);
          v16 = (_QWORD *)(v24 + 8LL * v26);
          v27 = *v16;
          if ( v12 == *v16 )
            goto LABEL_30;
          ++v35;
        }
        goto LABEL_37;
      }
    }
    else
    {
      v14 = *(_DWORD *)(a1 + 24);
LABEL_12:
      if ( 4 * v17 >= 3 * v14 )
        goto LABEL_26;
    }
    if ( v14 - *(_DWORD *)(a1 + 12) - v17 > v14 >> 3 )
    {
LABEL_14:
      *(_DWORD *)(a1 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
      if ( *v16 != -8 )
        --*(_DWORD *)(a1 + 12);
      *v16 = v12;
      v20 = sub_1648700(v6)[5];
      v21 = *(unsigned int *)(a1 + 88);
      if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 92) )
      {
        sub_16CD150(a1 + 80, v3, 0, 8, v18, v19);
        v21 = *(unsigned int *)(a1 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v21) = v20;
      ++*(_DWORD *)(a1 + 88);
      goto LABEL_5;
    }
    sub_19628C0(a1, v14);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v29 = a1 + 16;
      v30 = 7;
    }
    else
    {
      v28 = *(_DWORD *)(a1 + 24);
      v29 = *(_QWORD *)(a1 + 16);
      if ( !v28 )
      {
LABEL_60:
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        BUG();
      }
      v30 = v28 - 1;
    }
    v31 = v30 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v16 = (_QWORD *)(v29 + 8LL * v31);
    v32 = *v16;
    if ( v12 == *v16 )
    {
LABEL_30:
      v15 = *(_DWORD *)(a1 + 8);
      goto LABEL_14;
    }
    v33 = 1;
    v34 = 0;
    while ( v32 != -8 )
    {
      if ( v32 == -16 && !v34 )
        v34 = v16;
      v31 = v30 & (v33 + v31);
      v16 = (_QWORD *)(v29 + 8LL * v31);
      v32 = *v16;
      if ( v12 == *v16 )
        goto LABEL_30;
      ++v33;
    }
LABEL_37:
    if ( v34 )
      v16 = v34;
    goto LABEL_30;
  }
}
