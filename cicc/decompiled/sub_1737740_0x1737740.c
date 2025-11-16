// Function: sub_1737740
// Address: 0x1737740
//
__int64 __fastcall sub_1737740(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  char v4; // al
  __int64 v5; // rdx
  char v6; // cl
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rdx
  char v17; // dl
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // r10
  __int64 v31; // rcx

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 50 )
  {
    if ( v4 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 == 52 )
    {
      v26 = *(_QWORD *)(v5 - 48);
      v27 = *(_QWORD *)(v5 - 24);
      v28 = **a1;
      if ( v26 != v28 || !v27 )
      {
        if ( v28 != v27 || !v26 )
          goto LABEL_10;
        *a1[1] = v26;
LABEL_51:
        *a1[2] = v5;
        v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        if ( v7 )
          goto LABEL_28;
LABEL_11:
        v8 = *(_BYTE *)(v7 + 16);
        if ( v8 == 52 )
        {
          v11 = *(_QWORD *)(v7 - 48);
          v9 = *(_QWORD *)(v7 - 24);
          v29 = **a1;
          if ( v11 == v29 && v9 )
          {
LABEL_17:
            *a1[1] = v9;
            goto LABEL_18;
          }
          if ( !v11 || v29 != v9 )
            return 0;
        }
        else
        {
          if ( v8 != 5 || *(_WORD *)(v7 + 18) != 28 )
            return 0;
          v9 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
          v10 = **a1;
          v11 = *(_QWORD *)(v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)));
          if ( v9 != v10 || !v11 )
          {
            if ( !v9 || v10 != v11 )
              return 0;
            goto LABEL_17;
          }
        }
        *a1[1] = v11;
LABEL_18:
        *a1[2] = v7;
        v7 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( !v7 )
          return 0;
        goto LABEL_28;
      }
    }
    else
    {
      if ( v6 != 5 || *(_WORD *)(v5 + 18) != 28 )
        goto LABEL_10;
      v27 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
      v30 = **a1;
      v31 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
      if ( v27 == v30 && v31 )
      {
        *a1[1] = v31;
        goto LABEL_51;
      }
      if ( !v27 || v30 != v31 )
      {
LABEL_10:
        v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        goto LABEL_11;
      }
    }
    *a1[1] = v27;
    goto LABEL_51;
  }
  v12 = *(_QWORD *)(a2 - 48);
  v13 = *(_BYTE *)(v12 + 16);
  if ( v13 == 52 )
  {
    v16 = *(_QWORD *)(v12 - 48);
    v24 = *(_QWORD *)(v12 - 24);
    v25 = **a1;
    if ( v16 == v25 && v24 )
    {
      *a1[1] = v24;
LABEL_27:
      *a1[2] = v12;
      v7 = *(_QWORD *)(a2 - 24);
      if ( v7 )
        goto LABEL_28;
      goto LABEL_30;
    }
    if ( !v16 || v24 != v25 )
      goto LABEL_29;
LABEL_46:
    *a1[1] = v16;
    goto LABEL_27;
  }
  if ( v13 != 5 || *(_WORD *)(v12 + 18) != 28 )
    goto LABEL_29;
  v14 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
  v15 = **a1;
  v16 = *(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
  if ( v14 == v15 && v16 )
    goto LABEL_46;
  if ( v15 == v16 && v14 )
  {
    *a1[1] = v14;
    goto LABEL_27;
  }
LABEL_29:
  v7 = *(_QWORD *)(a2 - 24);
LABEL_30:
  v17 = *(_BYTE *)(v7 + 16);
  if ( v17 == 52 )
  {
    v21 = *(_QWORD *)(v7 - 48);
    v22 = *(_QWORD *)(v7 - 24);
    v23 = **a1;
    if ( v21 == v23 && v22 )
    {
      *a1[1] = v22;
    }
    else
    {
      if ( v23 != v22 || !v21 )
        return 0;
      *a1[1] = v21;
    }
  }
  else
  {
    if ( v17 != 5 || *(_WORD *)(v7 + 18) != 28 )
      return 0;
    v18 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    v19 = **a1;
    v20 = *(_QWORD *)(v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)));
    if ( v18 == v19 && v20 )
    {
      *a1[1] = v20;
    }
    else
    {
      if ( v19 != v20 || !v18 )
        return 0;
      *a1[1] = v18;
    }
  }
  *a1[2] = v7;
  v7 = *(_QWORD *)(a2 - 48);
  if ( !v7 )
    return 0;
LABEL_28:
  *a1[3] = v7;
  return 1;
}
