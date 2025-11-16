// Function: sub_D60910
// Address: 0xd60910
//
__int64 __fastcall sub_D60910(__int64 a1, __int64 a2, __int64 a3)
{
  char v6; // cl
  __int64 v7; // rdi
  int v8; // esi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r8
  _QWORD *v12; // rbx
  __int64 v13; // rdi
  bool v14; // cc
  unsigned int v15; // eax
  unsigned int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned int v21; // eax
  _QWORD *v22; // rbx
  int v23; // edx
  unsigned int v24; // edi
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

  v6 = *(_BYTE *)(a3 + 8) & 1;
  if ( v6 )
  {
    v7 = a3 + 16;
    v8 = 7;
  }
  else
  {
    v17 = *(_DWORD *)(a3 + 24);
    v7 = *(_QWORD *)(a3 + 16);
    if ( !v17 )
    {
      v21 = *(_DWORD *)(a3 + 8);
      ++*(_QWORD *)a3;
      v22 = 0;
      v23 = (v21 >> 1) + 1;
LABEL_19:
      v24 = 3 * v17;
      goto LABEL_20;
    }
    v8 = v17 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = v7 + 40LL * v9;
  v11 = *(_QWORD *)v10;
  if ( a2 == *(_QWORD *)v10 )
  {
LABEL_4:
    v12 = (_QWORD *)(v10 + 8);
    if ( *(_DWORD *)(v10 + 16) > 0x40u )
    {
      v13 = *(_QWORD *)(v10 + 8);
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
    goto LABEL_7;
  }
  v25 = 1;
  v22 = 0;
  while ( v11 != -4096 )
  {
    if ( !v22 && v11 == -8192 )
      v22 = (_QWORD *)v10;
    v9 = v8 & (v25 + v9);
    v10 = v7 + 40LL * v9;
    v11 = *(_QWORD *)v10;
    if ( a2 == *(_QWORD *)v10 )
      goto LABEL_4;
    ++v25;
  }
  v21 = *(_DWORD *)(a3 + 8);
  v24 = 24;
  v17 = 8;
  if ( !v22 )
    v22 = (_QWORD *)v10;
  ++*(_QWORD *)a3;
  v23 = (v21 >> 1) + 1;
  if ( !v6 )
  {
    v17 = *(_DWORD *)(a3 + 24);
    goto LABEL_19;
  }
LABEL_20:
  if ( 4 * v23 >= v24 )
  {
    sub_D60340(a3, 2 * v17);
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
        goto LABEL_63;
      v27 = v36 - 1;
    }
    v28 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v22 = (_QWORD *)(v26 + 40LL * v28);
    v29 = *v22;
    if ( a2 != *v22 )
    {
      v38 = 1;
      v35 = 0;
      while ( v29 != -4096 )
      {
        if ( !v35 && v29 == -8192 )
          v35 = v22;
        v28 = v27 & (v38 + v28);
        v22 = (_QWORD *)(v26 + 40LL * v28);
        v29 = *v22;
        if ( a2 == *v22 )
          goto LABEL_34;
        ++v38;
      }
      goto LABEL_40;
    }
LABEL_34:
    v21 = *(_DWORD *)(a3 + 8);
    goto LABEL_22;
  }
  if ( v17 - *(_DWORD *)(a3 + 12) - v23 <= v17 >> 3 )
  {
    sub_D60340(a3, v17);
    if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
    {
      v30 = a3 + 16;
      v31 = 7;
      goto LABEL_37;
    }
    v37 = *(_DWORD *)(a3 + 24);
    v30 = *(_QWORD *)(a3 + 16);
    if ( v37 )
    {
      v31 = v37 - 1;
LABEL_37:
      v32 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = (_QWORD *)(v30 + 40LL * v32);
      v33 = *v22;
      if ( a2 != *v22 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v22;
          v32 = v31 & (v34 + v32);
          v22 = (_QWORD *)(v30 + 40LL * v32);
          v33 = *v22;
          if ( a2 == *v22 )
            goto LABEL_34;
          ++v34;
        }
LABEL_40:
        if ( v35 )
          v22 = v35;
        goto LABEL_34;
      }
      goto LABEL_34;
    }
LABEL_63:
    *(_DWORD *)(a3 + 8) = (2 * (*(_DWORD *)(a3 + 8) >> 1) + 2) | *(_DWORD *)(a3 + 8) & 1;
    BUG();
  }
LABEL_22:
  *(_DWORD *)(a3 + 8) = (2 * (v21 >> 1) + 2) | v21 & 1;
  if ( *v22 != -4096 )
    --*(_DWORD *)(a3 + 12);
  *v22 = a2;
  v12 = v22 + 1;
  *((_DWORD *)v12 + 2) = 1;
  *v12 = 0;
  *((_DWORD *)v12 + 6) = 1;
  v12[2] = 0;
LABEL_7:
  v14 = *((_DWORD *)v12 + 6) <= 0x40u;
  *v12 = 0;
  *((_DWORD *)v12 + 2) = 1;
  if ( v14 )
  {
    v12[2] = 0;
    *((_DWORD *)v12 + 6) = 1;
    *(_DWORD *)(a1 + 8) = 1;
  }
  else
  {
    v18 = v12[2];
    if ( v18 )
    {
      j_j___libc_free_0_0(v18);
      v19 = *((_DWORD *)v12 + 2);
      v12[2] = 0;
      *((_DWORD *)v12 + 6) = 1;
      *(_DWORD *)(a1 + 8) = v19;
      if ( v19 > 0x40 )
      {
        sub_C43780(a1, (const void **)v12);
        v20 = *((_DWORD *)v12 + 6);
        *(_DWORD *)(a1 + 24) = v20;
        if ( v20 <= 0x40 )
          goto LABEL_10;
        goto LABEL_16;
      }
    }
    else
    {
      *((_DWORD *)v12 + 6) = 1;
      *(_DWORD *)(a1 + 8) = 1;
    }
  }
  *(_QWORD *)a1 = *v12;
  v15 = *((_DWORD *)v12 + 6);
  *(_DWORD *)(a1 + 24) = v15;
  if ( v15 <= 0x40 )
  {
LABEL_10:
    *(_QWORD *)(a1 + 16) = v12[2];
    return a1;
  }
LABEL_16:
  sub_C43780(a1 + 16, (const void **)v12 + 2);
  return a1;
}
