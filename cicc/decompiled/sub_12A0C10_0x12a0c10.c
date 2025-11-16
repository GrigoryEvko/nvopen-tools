// Function: sub_12A0C10
// Address: 0x12a0c10
//
__int64 __fastcall sub_12A0C10(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r13
  int v11; // edx
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  _QWORD *v16; // r14
  __int64 v17; // rdx
  _QWORD *v18; // r12
  int v19; // r9d
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // edi
  int v23; // ecx
  int v24; // eax
  int v25; // edx
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // r9d
  _QWORD *v30; // r8
  int v31; // edx
  int v32; // edx
  __int64 v33; // rdi
  int v34; // r9d
  unsigned int v35; // eax
  __int64 v36; // rsi
  unsigned int v37; // [rsp+Ch] [rbp-34h]

  v4 = *(unsigned int *)(a1 + 632);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 616);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
      {
        v9 = v7[1];
        if ( v9 )
          return v9;
      }
    }
    else
    {
      v11 = 1;
      while ( v8 != -8 )
      {
        v19 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v11 = v19;
      }
    }
  }
  v12 = sub_12A2130(a1, a2);
  v13 = *(_DWORD *)(a1 + 632);
  v9 = v12;
  if ( !v13 )
  {
    ++*(_QWORD *)(a1 + 608);
    goto LABEL_26;
  }
  v14 = *(_QWORD *)(a1 + 616);
  v15 = (v13 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v16 = (_QWORD *)(v14 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != a2 )
  {
    v20 = 1;
    v21 = 0;
    while ( v17 != -8 )
    {
      if ( !v21 && v17 == -16 )
        v21 = v16;
      v15 = (v13 - 1) & (v20 + v15);
      v16 = (_QWORD *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a2 )
        goto LABEL_10;
      ++v20;
    }
    v22 = *(_DWORD *)(a1 + 624);
    if ( v21 )
      v16 = v21;
    ++*(_QWORD *)(a1 + 608);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(a1 + 628) - v23 > v13 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(a1 + 624) = v23;
        if ( *v16 != -8 )
          --*(_DWORD *)(a1 + 628);
        *v16 = a2;
        v18 = v16 + 1;
        v16[1] = 0;
        goto LABEL_12;
      }
      v37 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_12A0A00(a1 + 608, v13);
      v31 = *(_DWORD *)(a1 + 632);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 616);
        v30 = 0;
        v34 = 1;
        v35 = v32 & v37;
        v23 = *(_DWORD *)(a1 + 624) + 1;
        v16 = (_QWORD *)(v33 + 16LL * (v32 & v37));
        v36 = *v16;
        if ( *v16 == a2 )
          goto LABEL_22;
        while ( v36 != -8 )
        {
          if ( !v30 && v36 == -16 )
            v30 = v16;
          v35 = v32 & (v34 + v35);
          v16 = (_QWORD *)(v33 + 16LL * v35);
          v36 = *v16;
          if ( *v16 == a2 )
            goto LABEL_22;
          ++v34;
        }
        goto LABEL_30;
      }
      goto LABEL_51;
    }
LABEL_26:
    sub_12A0A00(a1 + 608, 2 * v13);
    v24 = *(_DWORD *)(a1 + 632);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 616);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 624) + 1;
      v16 = (_QWORD *)(v26 + 16LL * v27);
      v28 = *v16;
      if ( *v16 == a2 )
        goto LABEL_22;
      v29 = 1;
      v30 = 0;
      while ( v28 != -8 )
      {
        if ( v28 == -16 && !v30 )
          v30 = v16;
        v27 = v25 & (v29 + v27);
        v16 = (_QWORD *)(v26 + 16LL * v27);
        v28 = *v16;
        if ( *v16 == a2 )
          goto LABEL_22;
        ++v29;
      }
LABEL_30:
      if ( v30 )
        v16 = v30;
      goto LABEL_22;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 624);
    BUG();
  }
LABEL_10:
  v18 = v16 + 1;
  if ( v16[1] )
    sub_161E7C0(v16 + 1);
LABEL_12:
  v16[1] = v9;
  if ( v9 )
    sub_1623A60(v18, v9, 2);
  return v9;
}
