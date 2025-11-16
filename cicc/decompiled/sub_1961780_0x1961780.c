// Function: sub_1961780
// Address: 0x1961780
//
__int64 __fastcall sub_1961780(__int64 a1, __int64 a2)
{
  unsigned int v4; // r14d
  unsigned int v5; // ecx
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v10; // r15
  int v11; // r12d
  unsigned int v12; // r12d
  __int64 v13; // rdi
  unsigned int v14; // r8d
  __int64 *v15; // rdx
  __int64 v16; // rsi
  int v17; // eax
  int v18; // r9d
  int v19; // r9d
  __int64 v20; // r10
  unsigned int v21; // eax
  int v22; // esi
  __int64 v23; // r8
  int v24; // r10d
  __int64 *v25; // r9
  int v26; // ecx
  int v27; // r9d
  int v28; // r9d
  __int64 v29; // r10
  __int64 *v30; // r8
  __int64 v31; // rax
  int v32; // ecx
  __int64 v33; // rdi
  int v34; // edi
  int v35; // edi
  __int64 *v36; // rcx
  __int64 v37; // [rsp+8h] [rbp-38h]
  unsigned int v38; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 56);
  v37 = *(_QWORD *)(a1 + 40);
  if ( !v4 )
  {
    v10 = *(_QWORD *)(a2 + 8);
    if ( v10 )
      goto LABEL_6;
    v13 = a1 + 32;
    v12 = 0;
    goto LABEL_23;
  }
  v5 = v4 - 1;
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = *(_QWORD *)(a1 + 40) + 16LL * v6;
  v8 = *(_QWORD *)v7;
  if ( *(_QWORD *)v7 == a2 )
  {
LABEL_3:
    if ( v7 != v37 + 16LL * v4 )
      return *(unsigned int *)(v7 + 8);
  }
  else
  {
    v17 = 1;
    while ( v8 != -8 )
    {
      v34 = v17 + 1;
      v6 = v5 & (v17 + v6);
      v7 = v37 + 16LL * v6;
      v8 = *(_QWORD *)v7;
      if ( *(_QWORD *)v7 == a2 )
        goto LABEL_3;
      v17 = v34;
    }
  }
  v10 = *(_QWORD *)(a2 + 8);
  v13 = a1 + 32;
  v12 = 0;
  if ( v10 )
  {
LABEL_6:
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v10) + 16) - 25) > 9u )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
      {
        v12 = 0;
        goto LABEL_12;
      }
    }
    v11 = 0;
    while ( 1 )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v10) + 16) - 25) <= 9u )
      {
        v10 = *(_QWORD *)(v10 + 8);
        ++v11;
        if ( !v10 )
          goto LABEL_11;
      }
    }
LABEL_11:
    v12 = v11 + 1;
LABEL_12:
    v13 = a1 + 32;
    if ( v4 )
    {
      v5 = v4 - 1;
      goto LABEL_14;
    }
LABEL_23:
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_24;
  }
LABEL_14:
  v14 = v5 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v15 = (__int64 *)(v37 + 16LL * v14);
  v16 = *v15;
  if ( *v15 != a2 )
  {
    v24 = 1;
    v25 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v25 )
        v25 = v15;
      v14 = v5 & (v24 + v14);
      v15 = (__int64 *)(v37 + 16LL * v14);
      v16 = *v15;
      if ( *v15 == a2 )
        goto LABEL_15;
      ++v24;
    }
    v26 = *(_DWORD *)(a1 + 48);
    if ( v25 )
      v15 = v25;
    ++*(_QWORD *)(a1 + 32);
    v22 = v26 + 1;
    if ( 4 * (v26 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 52) - v22 > v4 >> 3 )
      {
LABEL_26:
        *(_DWORD *)(a1 + 48) = v22;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 52);
        *v15 = a2;
        *((_DWORD *)v15 + 2) = 0;
        goto LABEL_15;
      }
      v38 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_13FEAC0(v13, v4);
      v27 = *(_DWORD *)(a1 + 56);
      if ( v27 )
      {
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 40);
        v30 = 0;
        LODWORD(v31) = v28 & v38;
        v22 = *(_DWORD *)(a1 + 48) + 1;
        v32 = 1;
        v15 = (__int64 *)(v29 + 16LL * (v28 & v38));
        v33 = *v15;
        if ( *v15 != a2 )
        {
          while ( v33 != -8 )
          {
            if ( !v30 && v33 == -16 )
              v30 = v15;
            v31 = v28 & (unsigned int)(v31 + v32);
            v15 = (__int64 *)(v29 + 16 * v31);
            v33 = *v15;
            if ( *v15 == a2 )
              goto LABEL_26;
            ++v32;
          }
          if ( v30 )
            v15 = v30;
        }
        goto LABEL_26;
      }
LABEL_63:
      ++*(_DWORD *)(a1 + 48);
      BUG();
    }
LABEL_24:
    sub_13FEAC0(v13, 2 * v4);
    v18 = *(_DWORD *)(a1 + 56);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 40);
      v21 = v19 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(a1 + 48) + 1;
      v15 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v15;
      if ( *v15 != a2 )
      {
        v35 = 1;
        v36 = 0;
        while ( v23 != -8 )
        {
          if ( !v36 && v23 == -16 )
            v36 = v15;
          v21 = v19 & (v35 + v21);
          v15 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v15;
          if ( *v15 == a2 )
            goto LABEL_26;
          ++v35;
        }
        if ( v36 )
          v15 = v36;
      }
      goto LABEL_26;
    }
    goto LABEL_63;
  }
LABEL_15:
  *((_DWORD *)v15 + 2) = v12;
  return v12;
}
