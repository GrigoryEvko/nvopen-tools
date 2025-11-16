// Function: sub_173A670
// Address: 0x173a670
//
__int64 __fastcall sub_173A670(__int64 **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rax
  char v7; // dl
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rdx
  char v12; // cl
  __int64 v13; // r9
  __int64 v14; // r10
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  char v20; // dl
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // rcx
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rcx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 50 )
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v11 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v12 = *(_BYTE *)(v11 + 16);
    if ( v12 == 52 )
    {
      v29 = *(_QWORD *)(v11 - 48);
      v13 = *(_QWORD *)(v11 - 24);
      v30 = **a1;
      if ( v29 == v30 && v13 )
        goto LABEL_22;
      if ( v30 == v13 && v29 )
      {
        *a1[1] = v29;
        goto LABEL_23;
      }
    }
    else if ( v12 == 5 && *(_WORD *)(v11 + 18) == 28 )
    {
      v13 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      v14 = **a1;
      v15 = *(_QWORD *)(v11 + 24 * (1LL - (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)));
      if ( v13 == v14 && v15 )
      {
        *a1[1] = v15;
LABEL_23:
        *a1[2] = v11;
        v16 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        if ( v16 )
        {
LABEL_24:
          *a1[3] = v16;
          return 1;
        }
LABEL_32:
        v20 = *(_BYTE *)(v16 + 16);
        if ( v20 == 52 )
        {
          v23 = *(_QWORD *)(v16 - 48);
          v21 = *(_QWORD *)(v16 - 24);
          v28 = **a1;
          if ( v23 == v28 && v21 )
          {
LABEL_38:
            *a1[1] = v21;
            goto LABEL_39;
          }
          if ( !v23 || v28 != v21 )
            return 0;
        }
        else
        {
          if ( v20 != 5 || *(_WORD *)(v16 + 18) != 28 )
            return 0;
          v21 = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
          v22 = **a1;
          v23 = *(_QWORD *)(v16 + 24 * (1LL - (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)));
          if ( v21 != v22 || !v23 )
          {
            if ( !v21 || v22 != v23 )
              return 0;
            goto LABEL_38;
          }
        }
        *a1[1] = v23;
LABEL_39:
        *a1[2] = v16;
        v16 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( v16 )
          goto LABEL_24;
        return 0;
      }
      if ( v13 && v14 == v15 )
      {
LABEL_22:
        *a1[1] = v13;
        goto LABEL_23;
      }
    }
    v16 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    goto LABEL_32;
  }
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 52 )
  {
    if ( v5 != 5 || *(_WORD *)(v4 + 18) != 28 )
      goto LABEL_8;
    v25 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    v27 = **a1;
    v24 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( v25 != v27 || !v24 )
    {
      if ( !v25 || v27 != v24 )
      {
LABEL_8:
        v6 = *(_QWORD *)(a2 - 24);
        goto LABEL_9;
      }
      goto LABEL_48;
    }
LABEL_44:
    *a1[1] = v24;
    goto LABEL_49;
  }
  v24 = *(_QWORD *)(v4 - 48);
  v25 = *(_QWORD *)(v4 - 24);
  v26 = **a1;
  if ( v24 != v26 || !v25 )
  {
    if ( !v24 || v26 != v25 )
      goto LABEL_8;
    goto LABEL_44;
  }
LABEL_48:
  *a1[1] = v25;
LABEL_49:
  *a1[2] = v4;
  v6 = *(_QWORD *)(a2 - 24);
  if ( !v6 )
  {
LABEL_9:
    v7 = *(_BYTE *)(v6 + 16);
    if ( v7 == 52 )
    {
      v17 = *(_QWORD *)(v6 - 48);
      v18 = *(_QWORD *)(v6 - 24);
      v19 = **a1;
      if ( v17 == v19 && v18 )
      {
        *a1[1] = v18;
      }
      else
      {
        if ( v19 != v18 || !v17 )
          return 0;
        *a1[1] = v17;
      }
    }
    else
    {
      if ( v7 != 5 || *(_WORD *)(v6 + 18) != 28 )
        return 0;
      v8 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      v9 = **a1;
      v10 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
      if ( v8 == v9 && v10 )
      {
        *a1[1] = v10;
      }
      else
      {
        if ( v9 != v10 || !v8 )
          return 0;
        *a1[1] = v8;
      }
    }
    *a1[2] = v6;
    v16 = *(_QWORD *)(a2 - 48);
    if ( v16 )
      goto LABEL_24;
    return 0;
  }
  *a1[3] = v6;
  return 1;
}
