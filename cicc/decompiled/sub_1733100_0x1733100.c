// Function: sub_1733100
// Address: 0x1733100
//
char __fastcall sub_1733100(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rax
  char v7; // dl
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  char v17; // dl
  char v18; // dl
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rdx
  __int64 v24; // rax
  char v25; // dl
  __int16 v26; // dx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  char v41; // dl
  __int16 v42; // dx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 52 )
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v15 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v16 = *(_QWORD *)(a2 - 24 * v15);
    v17 = *(_BYTE *)(v16 + 16);
    if ( v17 == 50 )
    {
      v39 = *(_QWORD *)(v16 - 48);
      if ( !v39 )
      {
LABEL_20:
        v6 = *(_QWORD *)(a2 + 24 * (1 - v15));
        v18 = *(_BYTE *)(v6 + 16);
        if ( v18 == 50 )
        {
LABEL_21:
          v19 = *(_QWORD *)(v6 - 48);
          if ( !v19 )
            return 0;
          **a1 = v19;
          v20 = *(_QWORD *)(v6 - 24);
          if ( !v20 )
            return 0;
LABEL_23:
          *a1[1] = v20;
          v21 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          v22 = *(_BYTE *)(v21 + 16);
          if ( v22 != 51 )
          {
            if ( v22 != 5 || *(_WORD *)(v21 + 18) != 27 )
              return 0;
            v12 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
            v13 = *a1[2];
            v14 = *(_QWORD *)(v21 + 24 * (1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF)));
            if ( v12 != v13 )
              goto LABEL_15;
            goto LABEL_27;
          }
          v36 = *(_QWORD *)(v21 - 48);
          v44 = *(_QWORD *)(v21 - 24);
          v45 = *a1[2];
          if ( v36 == v45 && *a1[3] == v44 )
            return 1;
          if ( v45 == v44 )
            return *a1[3] == v36;
          return 0;
        }
        if ( v18 != 5 )
          return 0;
        v42 = *(_WORD *)(v6 + 18);
        goto LABEL_62;
      }
      **a1 = v39;
      v40 = *(_QWORD *)(v16 - 24);
      if ( !v40 )
      {
LABEL_75:
        v15 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        goto LABEL_20;
      }
    }
    else
    {
      if ( v17 != 5 )
        goto LABEL_20;
      if ( *(_WORD *)(v16 + 18) != 26 )
        goto LABEL_20;
      v46 = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
      if ( !v46 )
        goto LABEL_20;
      **a1 = v46;
      v40 = *(_QWORD *)(v16 + 24 * (1LL - (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)));
      if ( !v40 )
        goto LABEL_75;
    }
    *a1[1] = v40;
    v6 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v41 = *(_BYTE *)(v6 + 16);
    if ( v41 != 51 )
    {
      if ( v41 != 5 )
      {
        if ( v41 != 50 )
          return 0;
        goto LABEL_21;
      }
      v42 = *(_WORD *)(v6 + 18);
      if ( v42 != 27 )
      {
LABEL_62:
        if ( v42 != 26 )
          return 0;
        v43 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( !v43 )
          return 0;
        **a1 = v43;
        v20 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        if ( !v20 )
          return 0;
        goto LABEL_23;
      }
      v32 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
      v33 = *(_QWORD *)(v6 - 24 * v32);
      v34 = *a1[2];
LABEL_48:
      v35 = *(_QWORD *)(v6 + 24 * (1 - v32));
      return v33 == v34 && *a1[3] == v35 || v34 == v35 && v33 == *a1[3];
    }
LABEL_38:
    v28 = *(_QWORD *)(v6 - 48);
    v29 = *(_QWORD *)(v6 - 24);
    v30 = *a1[2];
    return v28 == v30 && *a1[3] == v29 || v30 == v29 && v28 == *a1[3];
  }
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 50 )
  {
    if ( v5 != 5 )
      goto LABEL_8;
    if ( *(_WORD *)(v4 + 18) != 26 )
      goto LABEL_8;
    v31 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    if ( !v31 )
      goto LABEL_8;
    **a1 = v31;
    v24 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( !v24 )
      goto LABEL_8;
LABEL_31:
    *a1[1] = v24;
    v6 = *(_QWORD *)(a2 - 24);
    v25 = *(_BYTE *)(v6 + 16);
    if ( v25 == 51 )
      goto LABEL_38;
    if ( v25 != 5 )
    {
      if ( v25 != 50 )
        return 0;
      goto LABEL_9;
    }
    v26 = *(_WORD *)(v6 + 18);
    if ( v26 == 27 )
    {
      v32 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
      v33 = *(_QWORD *)(v6 - 24 * v32);
      v34 = *a1[2];
      goto LABEL_48;
    }
LABEL_34:
    if ( v26 != 26 )
      return 0;
    v27 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    if ( !v27 )
      return 0;
    **a1 = v27;
    v9 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
    if ( !v9 )
      return 0;
    goto LABEL_11;
  }
  v23 = *(_QWORD *)(v4 - 48);
  if ( v23 )
  {
    **a1 = v23;
    v24 = *(_QWORD *)(v4 - 24);
    if ( v24 )
      goto LABEL_31;
  }
LABEL_8:
  v6 = *(_QWORD *)(a2 - 24);
  v7 = *(_BYTE *)(v6 + 16);
  if ( v7 != 50 )
  {
    if ( v7 != 5 )
      return 0;
    v26 = *(_WORD *)(v6 + 18);
    goto LABEL_34;
  }
LABEL_9:
  v8 = *(_QWORD *)(v6 - 48);
  if ( !v8 )
    return 0;
  **a1 = v8;
  v9 = *(_QWORD *)(v6 - 24);
  if ( !v9 )
    return 0;
LABEL_11:
  *a1[1] = v9;
  v10 = *(_QWORD *)(a2 - 48);
  v11 = *(_BYTE *)(v10 + 16);
  if ( v11 != 51 )
  {
    if ( v11 != 5 || *(_WORD *)(v10 + 18) != 27 )
      return 0;
    v12 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    v13 = *a1[2];
    v14 = *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
    if ( v12 != v13 )
    {
LABEL_15:
      if ( v13 == v14 )
        return *a1[3] == v12;
      return 0;
    }
LABEL_27:
    if ( *a1[3] == v14 )
      return 1;
    goto LABEL_15;
  }
  v36 = *(_QWORD *)(v10 - 48);
  v37 = *(_QWORD *)(v10 - 24);
  v38 = *a1[2];
  if ( v36 == v38 && *a1[3] == v37 )
    return 1;
  if ( v38 != v37 )
    return 0;
  return *a1[3] == v36;
}
