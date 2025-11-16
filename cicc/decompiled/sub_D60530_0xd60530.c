// Function: sub_D60530
// Address: 0xd60530
//
__int64 __fastcall sub_D60530(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v8; // edx
  __int64 v9; // rdi
  int v10; // esi
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rbx
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned int v18; // esi
  unsigned int v19; // eax
  _QWORD *v20; // rbx
  int v21; // ecx
  unsigned int v22; // edi
  unsigned int v23; // eax
  unsigned int v24; // eax
  int v25; // r9d
  __int64 v26; // rsi
  int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // eax
  __int64 v33; // rdx
  int v34; // r8d
  _QWORD *v35; // rdi
  int v36; // edx
  int v37; // ecx
  int v38; // r8d

  v8 = *(_BYTE *)(a3 + 8) & 1;
  if ( v8 )
  {
    v9 = a3 + 16;
    v10 = 7;
  }
  else
  {
    v18 = *(_DWORD *)(a3 + 24);
    v9 = *(_QWORD *)(a3 + 16);
    if ( !v18 )
    {
      v19 = *(_DWORD *)(a3 + 8);
      ++*(_QWORD *)a3;
      v20 = 0;
      v21 = (v19 >> 1) + 1;
LABEL_14:
      v22 = 3 * v18;
      goto LABEL_15;
    }
    v10 = v18 - 1;
  }
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v9 + 40LL * v11;
  v13 = *(_QWORD *)v12;
  if ( a2 == *(_QWORD *)v12 )
  {
LABEL_4:
    v14 = v12 + 8;
    if ( *(_DWORD *)(v12 + 16) > 0x40u )
      goto LABEL_5;
    goto LABEL_20;
  }
  v25 = 1;
  v20 = 0;
  while ( v13 != -4096 )
  {
    if ( !v20 && v13 == -8192 )
      v20 = (_QWORD *)v12;
    v11 = v10 & (v25 + v11);
    v12 = v9 + 40LL * v11;
    v13 = *(_QWORD *)v12;
    if ( a2 == *(_QWORD *)v12 )
      goto LABEL_4;
    ++v25;
  }
  v19 = *(_DWORD *)(a3 + 8);
  v22 = 24;
  v18 = 8;
  if ( !v20 )
    v20 = (_QWORD *)v12;
  ++*(_QWORD *)a3;
  v21 = (v19 >> 1) + 1;
  if ( !(_BYTE)v8 )
  {
    v18 = *(_DWORD *)(a3 + 24);
    goto LABEL_14;
  }
LABEL_15:
  if ( 4 * v21 >= v22 )
  {
    sub_D60340(a3, 2 * v18);
    if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
    {
      v26 = a3 + 16;
      v27 = 7;
    }
    else
    {
      v36 = *(_DWORD *)(a3 + 24);
      v26 = *(_QWORD *)(a3 + 16);
      if ( !v36 )
        goto LABEL_64;
      v27 = v36 - 1;
    }
    v28 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v20 = (_QWORD *)(v26 + 40LL * v28);
    v29 = *v20;
    if ( a2 != *v20 )
    {
      v38 = 1;
      v35 = 0;
      while ( v29 != -4096 )
      {
        if ( !v35 && v29 == -8192 )
          v35 = v20;
        v28 = v27 & (v38 + v28);
        v20 = (_QWORD *)(v26 + 40LL * v28);
        v29 = *v20;
        if ( a2 == *v20 )
          goto LABEL_35;
        ++v38;
      }
      goto LABEL_41;
    }
LABEL_35:
    v19 = *(_DWORD *)(a3 + 8);
    goto LABEL_17;
  }
  if ( v18 - *(_DWORD *)(a3 + 12) - v21 <= v18 >> 3 )
  {
    sub_D60340(a3, v18);
    if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
    {
      v30 = a3 + 16;
      v31 = 7;
      goto LABEL_38;
    }
    v37 = *(_DWORD *)(a3 + 24);
    v30 = *(_QWORD *)(a3 + 16);
    if ( v37 )
    {
      v31 = v37 - 1;
LABEL_38:
      v32 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (_QWORD *)(v30 + 40LL * v32);
      v33 = *v20;
      if ( a2 != *v20 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v35 )
            v35 = v20;
          v32 = v31 & (v34 + v32);
          v20 = (_QWORD *)(v30 + 40LL * v32);
          v33 = *v20;
          if ( a2 == *v20 )
            goto LABEL_35;
          ++v34;
        }
LABEL_41:
        if ( v35 )
          v20 = v35;
        goto LABEL_35;
      }
      goto LABEL_35;
    }
LABEL_64:
    *(_DWORD *)(a3 + 8) = (2 * (*(_DWORD *)(a3 + 8) >> 1) + 2) | *(_DWORD *)(a3 + 8) & 1;
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a3 + 8) = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *v20 != -4096 )
    --*(_DWORD *)(a3 + 12);
  *v20 = a2;
  v14 = (__int64)(v20 + 1);
  *(_DWORD *)(v14 + 8) = 1;
  *(_QWORD *)v14 = 0;
  *(_DWORD *)(v14 + 24) = 1;
  *(_QWORD *)(v14 + 16) = 0;
LABEL_20:
  if ( *(_DWORD *)(a4 + 8) <= 0x40u )
  {
    *(_QWORD *)v14 = *(_QWORD *)a4;
    *(_DWORD *)(v14 + 8) = *(_DWORD *)(a4 + 8);
    goto LABEL_6;
  }
LABEL_5:
  sub_C43990(v14, a4);
LABEL_6:
  if ( *(_DWORD *)(v14 + 24) > 0x40u || *(_DWORD *)(a4 + 24) > 0x40u )
  {
    sub_C43990(v14 + 16, a4 + 16);
    v15 = *(_DWORD *)(v14 + 8);
    *(_DWORD *)(a1 + 8) = v15;
    if ( v15 <= 0x40 )
      goto LABEL_9;
LABEL_24:
    sub_C43780(a1, (const void **)v14);
    v24 = *(_DWORD *)(v14 + 24);
    *(_DWORD *)(a1 + 24) = v24;
    if ( v24 <= 0x40 )
      goto LABEL_10;
LABEL_25:
    sub_C43780(a1 + 16, (const void **)(v14 + 16));
    return a1;
  }
  *(_QWORD *)(v14 + 16) = *(_QWORD *)(a4 + 16);
  *(_DWORD *)(v14 + 24) = *(_DWORD *)(a4 + 24);
  v23 = *(_DWORD *)(v14 + 8);
  *(_DWORD *)(a1 + 8) = v23;
  if ( v23 > 0x40 )
    goto LABEL_24;
LABEL_9:
  *(_QWORD *)a1 = *(_QWORD *)v14;
  v16 = *(_DWORD *)(v14 + 24);
  *(_DWORD *)(a1 + 24) = v16;
  if ( v16 > 0x40 )
    goto LABEL_25;
LABEL_10:
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v14 + 16);
  return a1;
}
