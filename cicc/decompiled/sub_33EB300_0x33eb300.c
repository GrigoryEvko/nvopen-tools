// Function: sub_33EB300
// Address: 0x33eb300
//
__int64 __fastcall sub_33EB300(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  const void *v3; // rbx
  size_t v4; // r15
  size_t v5; // r12
  size_t v6; // rdx
  const void *v7; // r13
  int v8; // eax
  signed __int64 v9; // rax
  int v10; // eax
  int v11; // eax
  signed __int64 v12; // rax
  signed __int64 v13; // rax
  __int64 v14; // r12
  size_t v15; // r12
  __int64 v16; // r14
  int v17; // eax
  const void *v18; // r10
  signed __int64 v19; // rax
  int v20; // eax
  size_t v21; // r15
  size_t v22; // r13
  signed __int64 v24; // rax
  int v25; // eax
  const void *v26; // r8
  signed __int64 v27; // rax
  signed __int64 v28; // rax
  size_t v29; // r13
  size_t v30; // r14
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  size_t n; // [rsp+28h] [rbp-38h]
  size_t na; // [rsp+28h] [rbp-38h]
  size_t nb; // [rsp+28h] [rbp-38h]
  const void *nc; // [rsp+28h] [rbp-38h]
  const void *nd; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v34 = a1 + 8;
  if ( !v2 )
    return v34;
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  while ( 1 )
  {
    v5 = *(_QWORD *)(v2 + 40);
    v6 = v4;
    v7 = *(const void **)(v2 + 32);
    if ( v5 <= v4 )
      v6 = *(_QWORD *)(v2 + 40);
    if ( !v6 )
    {
      v9 = v5 - v4;
      if ( (__int64)(v5 - v4) >= 0x80000000LL )
        goto LABEL_51;
      goto LABEL_8;
    }
    n = v6;
    v8 = memcmp(*(const void **)(v2 + 32), v3, v6);
    v6 = n;
    if ( !v8 )
      break;
    if ( v8 < 0 )
      goto LABEL_45;
LABEL_11:
    na = v6;
    v10 = memcmp(v3, v7, v6);
    v6 = na;
    if ( !v10 )
      goto LABEL_51;
    if ( v10 < 0 )
      goto LABEL_49;
LABEL_13:
    if ( *(_DWORD *)(v2 + 64) >= *(_DWORD *)(a2 + 32) )
    {
      if ( v6 )
        goto LABEL_15;
      v12 = v4 - v5;
      if ( (__int64)(v4 - v5) < 0x80000000LL )
        goto LABEL_17;
      goto LABEL_21;
    }
LABEL_45:
    v2 = *(_QWORD *)(v2 + 24);
LABEL_46:
    if ( !v2 )
      return v34;
  }
  v9 = v5 - v4;
  if ( (__int64)(v5 - v4) >= 0x80000000LL )
    goto LABEL_11;
LABEL_8:
  if ( v9 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v9 < 0 )
    goto LABEL_45;
  if ( v6 )
    goto LABEL_11;
LABEL_51:
  v24 = v4 - v5;
  if ( (__int64)(v4 - v5) >= 0x80000000LL || v24 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v24 >= 0 )
    goto LABEL_13;
  if ( !v6 )
  {
    v12 = v4 - v5;
    goto LABEL_17;
  }
LABEL_15:
  nb = v6;
  v11 = memcmp(v3, v7, v6);
  v6 = nb;
  if ( !v11 )
  {
    v12 = v4 - v5;
    if ( (__int64)(v4 - v5) >= 0x80000000LL )
      goto LABEL_20;
LABEL_17:
    if ( v12 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v12 >= 0 )
    {
      if ( v6 )
        goto LABEL_20;
LABEL_21:
      v13 = v5 - v4;
      if ( (__int64)(v5 - v4) < 0x80000000LL )
      {
        if ( v13 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_24;
        goto LABEL_23;
      }
      goto LABEL_48;
    }
    goto LABEL_49;
  }
  if ( v11 < 0 )
    goto LABEL_49;
LABEL_20:
  LODWORD(v13) = memcmp(v7, v3, v6);
  if ( !(_DWORD)v13 )
    goto LABEL_21;
LABEL_23:
  if ( (int)v13 < 0 )
    goto LABEL_24;
LABEL_48:
  if ( *(_DWORD *)(a2 + 32) < *(_DWORD *)(v2 + 64) )
  {
LABEL_49:
    v34 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    goto LABEL_46;
  }
LABEL_24:
  v14 = *(_QWORD *)(v2 + 16);
  if ( !*(_QWORD *)(v2 + 24) )
    goto LABEL_61;
  v32 = *(_QWORD *)(v2 + 16);
  v15 = v4;
  v31 = v2;
  v16 = *(_QWORD *)(v2 + 24);
  while ( 2 )
  {
    v21 = *(_QWORD *)(v16 + 40);
    v18 = *(const void **)(v16 + 32);
    v22 = v21;
    if ( v15 <= v21 )
      v22 = v15;
    if ( v22 )
    {
      nc = *(const void **)(v16 + 32);
      v17 = memcmp(v3, nc, v22);
      v18 = nc;
      if ( v17 )
      {
        if ( v17 < 0 )
          goto LABEL_37;
      }
      else
      {
        v19 = v15 - v21;
        if ( (__int64)(v15 - v21) < 0x80000000LL )
          goto LABEL_28;
      }
LABEL_31:
      v20 = memcmp(v18, v3, v22);
      if ( !v20 )
        goto LABEL_32;
LABEL_35:
      if ( v20 >= 0 )
        goto LABEL_36;
LABEL_79:
      v16 = *(_QWORD *)(v16 + 24);
      goto LABEL_38;
    }
    v19 = v15 - v21;
    if ( (__int64)(v15 - v21) < 0x80000000LL )
    {
LABEL_28:
      if ( v19 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v19 < 0 )
        goto LABEL_37;
      if ( !v22 )
        goto LABEL_32;
      goto LABEL_31;
    }
LABEL_32:
    if ( (__int64)(v21 - v15) < 0x80000000LL )
    {
      if ( (__int64)(v21 - v15) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v20 = v21 - v15;
        goto LABEL_35;
      }
      goto LABEL_79;
    }
LABEL_36:
    if ( *(_DWORD *)(a2 + 32) >= *(_DWORD *)(v16 + 64) )
      goto LABEL_79;
LABEL_37:
    v16 = *(_QWORD *)(v16 + 16);
LABEL_38:
    if ( v16 )
      continue;
    break;
  }
  v4 = v15;
  v2 = v31;
  v14 = v32;
LABEL_61:
  if ( !v14 )
    return v2;
  v33 = v2;
  while ( 2 )
  {
    while ( 2 )
    {
      v29 = *(_QWORD *)(v14 + 40);
      v26 = *(const void **)(v14 + 32);
      v30 = v29;
      if ( v4 <= v29 )
        v30 = v4;
      if ( v30 )
      {
        nd = *(const void **)(v14 + 32);
        v25 = memcmp(nd, v3, v30);
        v26 = nd;
        if ( v25 )
        {
          if ( v25 < 0 )
            goto LABEL_81;
        }
        else
        {
          v27 = v29 - v4;
          if ( (__int64)(v29 - v4) < 0x80000000LL )
            goto LABEL_65;
        }
LABEL_68:
        LODWORD(v28) = memcmp(v3, v26, v30);
        if ( !(_DWORD)v28 )
          goto LABEL_69;
LABEL_71:
        if ( (int)v28 < 0 )
        {
LABEL_73:
          v33 = v14;
          v14 = *(_QWORD *)(v14 + 16);
          if ( !v14 )
            return v33;
          continue;
        }
LABEL_72:
        if ( *(_DWORD *)(v14 + 64) < *(_DWORD *)(a2 + 32) )
          goto LABEL_81;
        goto LABEL_73;
      }
      break;
    }
    v27 = v29 - v4;
    if ( (__int64)(v29 - v4) >= 0x80000000LL )
    {
LABEL_69:
      v28 = v4 - v29;
      if ( (__int64)(v4 - v29) < 0x80000000LL )
      {
        if ( v28 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_71;
        goto LABEL_73;
      }
      goto LABEL_72;
    }
LABEL_65:
    if ( v27 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v27 >= 0 )
    {
      if ( !v30 )
        goto LABEL_69;
      goto LABEL_68;
    }
LABEL_81:
    v14 = *(_QWORD *)(v14 + 24);
    if ( v14 )
      continue;
    return v33;
  }
}
