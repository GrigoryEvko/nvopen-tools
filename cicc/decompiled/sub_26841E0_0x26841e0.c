// Function: sub_26841E0
// Address: 0x26841e0
//
__int64 __fastcall sub_26841E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // rdi
  int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // rdi
  unsigned int v21; // eax
  int v22; // ecx
  _QWORD *v23; // rax
  unsigned int v24; // esi
  int v25; // r10d
  __int64 v26; // rdi
  int v27; // ecx
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // rdi
  int v31; // ecx
  unsigned int v32; // eax
  __int64 v33; // rsi
  int v34; // ecx
  int v35; // ecx

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 != 85 || a2 != v2 - 32 )
    goto LABEL_73;
  v3 = *a1;
  if ( *(char *)(v2 + 7) < 0 )
  {
    v4 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
    v6 = v4 + v5;
    v7 = 0;
    if ( *(char *)(v2 + 7) < 0 )
      v7 = sub_BD2BC0(v2);
    if ( (unsigned int)((v6 - v7) >> 4) )
      goto LABEL_73;
  }
  if ( v3 )
  {
    v8 = *(_QWORD *)(v3 + 120);
    if ( !v8
      || (v9 = *(_QWORD *)(v2 - 32)) == 0
      || *(_BYTE *)v9
      || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v2 + 80)
      || v8 != v9 )
    {
LABEL_73:
      BUG();
    }
  }
  v10 = a1[1];
  v11 = *(_QWORD *)(v2 + 40);
  v12 = *(_BYTE *)(v10 + 8) & 1;
  if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
  {
    v13 = v10 + 16;
    v14 = 3;
  }
  else
  {
    v24 = *(_DWORD *)(v10 + 24);
    v13 = *(_QWORD *)(v10 + 16);
    if ( !v24 )
    {
      v21 = *(_DWORD *)(v10 + 8);
      ++*(_QWORD *)v10;
      v16 = 0;
      v22 = (v21 >> 1) + 1;
      goto LABEL_20;
    }
    v14 = v24 - 1;
  }
  v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v16 = 9LL * v15;
  v17 = v13 + 72LL * v15;
  v18 = *(_QWORD *)v17;
  if ( v11 != *(_QWORD *)v17 )
  {
    v25 = 1;
    v16 = 0;
    while ( v18 != -4096 )
    {
      if ( v18 == -8192 && !v16 )
        v16 = v17;
      v15 = v14 & (v25 + v15);
      v17 = v13 + 72LL * v15;
      v18 = *(_QWORD *)v17;
      if ( v11 == *(_QWORD *)v17 )
        goto LABEL_16;
      ++v25;
    }
    v21 = *(_DWORD *)(v10 + 8);
    v18 = 12;
    v24 = 4;
    if ( !v16 )
      v16 = v17;
    ++*(_QWORD *)v10;
    v22 = (v21 >> 1) + 1;
    if ( (_BYTE)v12 )
    {
LABEL_21:
      if ( (unsigned int)v18 > 4 * v22 )
      {
        if ( v24 - *(_DWORD *)(v10 + 12) - v22 > v24 >> 3 )
        {
LABEL_23:
          *(_DWORD *)(v10 + 8) = (2 * (v21 >> 1) + 2) | v21 & 1;
          if ( *(_QWORD *)v16 != -4096 )
            --*(_DWORD *)(v10 + 12);
          *(_QWORD *)v16 = v11;
          v19 = v16 + 8;
          *(_QWORD *)(v16 + 8) = 0;
          *(_QWORD *)(v16 + 16) = v16 + 40;
          *(_QWORD *)(v16 + 24) = 4;
          *(_DWORD *)(v16 + 32) = 0;
          *(_BYTE *)(v16 + 36) = 1;
          goto LABEL_26;
        }
        sub_2683CA0(v10, v24);
        if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
        {
          v30 = v10 + 16;
          v31 = 3;
          goto LABEL_45;
        }
        v35 = *(_DWORD *)(v10 + 24);
        v30 = *(_QWORD *)(v10 + 16);
        if ( v35 )
        {
          v31 = v35 - 1;
LABEL_45:
          v32 = v31 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v16 = v30 + 72LL * v32;
          v33 = *(_QWORD *)v16;
          if ( v11 != *(_QWORD *)v16 )
          {
            v18 = 1;
            v12 = 0;
            while ( v33 != -4096 )
            {
              if ( !v12 && v33 == -8192 )
                v12 = v16;
              v32 = v31 & (v18 + v32);
              v16 = v30 + 72LL * v32;
              v33 = *(_QWORD *)v16;
              if ( v11 == *(_QWORD *)v16 )
                goto LABEL_42;
              v18 = (unsigned int)(v18 + 1);
            }
LABEL_48:
            if ( v12 )
              v16 = v12;
            goto LABEL_42;
          }
          goto LABEL_42;
        }
LABEL_72:
        *(_DWORD *)(v10 + 8) = (2 * (*(_DWORD *)(v10 + 8) >> 1) + 2) | *(_DWORD *)(v10 + 8) & 1;
        BUG();
      }
      sub_2683CA0(v10, 2 * v24);
      if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
      {
        v26 = v10 + 16;
        v27 = 3;
      }
      else
      {
        v34 = *(_DWORD *)(v10 + 24);
        v26 = *(_QWORD *)(v10 + 16);
        if ( !v34 )
          goto LABEL_72;
        v27 = v34 - 1;
      }
      v28 = v27 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = v26 + 72LL * v28;
      v29 = *(_QWORD *)v16;
      if ( v11 != *(_QWORD *)v16 )
      {
        v18 = 1;
        v12 = 0;
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v12 )
            v12 = v16;
          v28 = v27 & (v18 + v28);
          v16 = v26 + 72LL * v28;
          v29 = *(_QWORD *)v16;
          if ( v11 == *(_QWORD *)v16 )
            goto LABEL_42;
          v18 = (unsigned int)(v18 + 1);
        }
        goto LABEL_48;
      }
LABEL_42:
      v21 = *(_DWORD *)(v10 + 8);
      goto LABEL_23;
    }
    v24 = *(_DWORD *)(v10 + 24);
LABEL_20:
    v18 = 3 * v24;
    goto LABEL_21;
  }
LABEL_16:
  v19 = v17 + 8;
  if ( !*(_BYTE *)(v17 + 36) )
    goto LABEL_17;
LABEL_26:
  v23 = *(_QWORD **)(v19 + 8);
  v17 = *(unsigned int *)(v19 + 20);
  v16 = (__int64)&v23[v17];
  if ( v23 != (_QWORD *)v16 )
  {
    while ( v2 != *v23 )
    {
      if ( (_QWORD *)v16 == ++v23 )
        goto LABEL_29;
    }
    return 0;
  }
LABEL_29:
  if ( (unsigned int)v17 >= *(_DWORD *)(v19 + 16) )
  {
LABEL_17:
    sub_C8CC70(v19, v2, v16, v17, v12, v18);
    return 0;
  }
  *(_DWORD *)(v19 + 20) = v17 + 1;
  *(_QWORD *)v16 = v2;
  ++*(_QWORD *)v19;
  return 0;
}
