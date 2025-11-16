// Function: sub_111F1A0
// Address: 0x111f1a0
//
__int64 __fastcall sub_111F1A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rax
  _BYTE *v8; // rax
  char v9; // cl
  _BYTE *v10; // rax
  char v11; // dl
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  _BYTE *v16; // rdx
  char v17; // cl
  _BYTE *v18; // rax
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rcx
  __int64 v35; // rcx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 || *(_QWORD *)(v3 + 8) || *(_BYTE *)v2 != 57 )
    goto LABEL_3;
  v7 = *(_QWORD *)(v2 - 64);
  if ( !v7 )
    goto LABEL_12;
  *(_QWORD *)a1[1] = v7;
  v8 = *(_BYTE **)(v2 - 32);
  v9 = *v8;
  if ( *v8 == 42 )
  {
    v23 = *((_QWORD *)v8 - 8);
    if ( v23 )
    {
      *(_QWORD *)a1[2] = v23;
      v24 = *((_QWORD *)v8 - 4);
      if ( v24 == *(_QWORD *)a1[3] )
        goto LABEL_19;
      if ( !v24 )
      {
LABEL_44:
        v9 = *v8;
        goto LABEL_10;
      }
    }
    else
    {
      v24 = *((_QWORD *)v8 - 4);
      if ( !v24 )
      {
LABEL_13:
        if ( !v8 )
          goto LABEL_3;
        *(_QWORD *)a1[1] = v8;
        v10 = *(_BYTE **)(v2 - 64);
        v11 = *v10;
        if ( *v10 != 42 )
        {
LABEL_15:
          if ( v11 != 59 )
          {
LABEL_16:
            if ( v11 == 44 )
            {
              v12 = *((_QWORD *)v10 - 8);
              if ( v12 )
              {
                *(_QWORD *)a1[6] = v12;
                if ( *((_QWORD *)v10 - 4) == *(_QWORD *)a1[7] )
                  goto LABEL_19;
              }
            }
LABEL_3:
            v4 = *(_QWORD *)(a2 - 32);
            goto LABEL_4;
          }
          v37 = *((_QWORD *)v10 - 8);
          if ( v37 )
          {
            *(_QWORD *)a1[4] = v37;
            v38 = *((_QWORD *)v10 - 4);
            if ( v38 == *(_QWORD *)a1[5] )
              goto LABEL_19;
            if ( !v38 )
            {
LABEL_80:
              v11 = *v10;
              goto LABEL_16;
            }
          }
          else
          {
            v38 = *((_QWORD *)v10 - 4);
            if ( !v38 )
              goto LABEL_3;
          }
          *(_QWORD *)a1[4] = v38;
          if ( *((_QWORD *)v10 - 8) == *(_QWORD *)a1[5] )
            goto LABEL_19;
          goto LABEL_80;
        }
        v29 = *((_QWORD *)v10 - 8);
        if ( v29 )
        {
          *(_QWORD *)a1[2] = v29;
          v30 = *((_QWORD *)v10 - 4);
          if ( v30 == *(_QWORD *)a1[3] )
            goto LABEL_19;
          if ( !v30 )
          {
LABEL_60:
            v11 = *v10;
            goto LABEL_15;
          }
        }
        else
        {
          v30 = *((_QWORD *)v10 - 4);
          if ( !v30 )
            goto LABEL_3;
        }
        *(_QWORD *)a1[2] = v30;
        if ( *((_QWORD *)v10 - 8) == *(_QWORD *)a1[3] )
          goto LABEL_19;
        goto LABEL_60;
      }
    }
    *(_QWORD *)a1[2] = v24;
    if ( *((_QWORD *)v8 - 8) == *(_QWORD *)a1[3] )
      goto LABEL_19;
    goto LABEL_44;
  }
LABEL_10:
  if ( v9 == 59 )
  {
    v33 = *((_QWORD *)v8 - 8);
    if ( v33 )
    {
      *(_QWORD *)a1[4] = v33;
      v34 = *((_QWORD *)v8 - 4);
      if ( v34 == *(_QWORD *)a1[5] )
        goto LABEL_19;
      if ( !v34 )
      {
LABEL_70:
        v9 = *v8;
        goto LABEL_11;
      }
    }
    else
    {
      v34 = *((_QWORD *)v8 - 4);
      if ( !v34 )
      {
LABEL_12:
        v8 = *(_BYTE **)(v2 - 32);
        goto LABEL_13;
      }
    }
    *(_QWORD *)a1[4] = v34;
    if ( *((_QWORD *)v8 - 8) == *(_QWORD *)a1[5] )
      goto LABEL_19;
    goto LABEL_70;
  }
LABEL_11:
  if ( v9 != 44 )
    goto LABEL_12;
  v27 = *((_QWORD *)v8 - 8);
  if ( !v27 )
    goto LABEL_12;
  *(_QWORD *)a1[6] = v27;
  if ( *((_QWORD *)v8 - 4) != *(_QWORD *)a1[7] )
    goto LABEL_12;
LABEL_19:
  v4 = *(_QWORD *)(a2 - 32);
  if ( *(_QWORD *)a1[8] == v4 )
  {
    if ( *a1 )
    {
      v13 = sub_B53900(a2);
      v14 = *a1;
      *(_DWORD *)v14 = v13;
      *(_BYTE *)(v14 + 4) = BYTE4(v13);
    }
    return 1;
  }
LABEL_4:
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) || *(_BYTE *)v4 != 57 )
    return 0;
  v15 = *(_QWORD *)(v4 - 64);
  if ( !v15 )
    goto LABEL_29;
  *(_QWORD *)a1[1] = v15;
  v16 = *(_BYTE **)(v4 - 32);
  v17 = *v16;
  if ( *v16 != 42 )
  {
LABEL_27:
    if ( v17 != 59 )
    {
LABEL_28:
      if ( v17 == 44 )
      {
        v28 = *((_QWORD *)v16 - 8);
        if ( v28 )
        {
          *(_QWORD *)a1[6] = v28;
          if ( *((_QWORD *)v16 - 4) == *(_QWORD *)a1[7] )
            goto LABEL_36;
        }
      }
      goto LABEL_29;
    }
    v35 = *((_QWORD *)v16 - 8);
    if ( v35 )
    {
      *(_QWORD *)a1[4] = v35;
      v36 = *((_QWORD *)v16 - 4);
      if ( v36 == *(_QWORD *)a1[5] )
        goto LABEL_36;
      if ( !v36 )
      {
LABEL_75:
        v17 = *v16;
        goto LABEL_28;
      }
LABEL_74:
      *(_QWORD *)a1[4] = v36;
      if ( *((_QWORD *)v16 - 8) == *(_QWORD *)a1[5] )
        goto LABEL_36;
      goto LABEL_75;
    }
    v36 = *((_QWORD *)v16 - 4);
    if ( v36 )
      goto LABEL_74;
LABEL_29:
    v16 = *(_BYTE **)(v4 - 32);
    goto LABEL_30;
  }
  v25 = *((_QWORD *)v16 - 8);
  if ( v25 )
  {
    *(_QWORD *)a1[2] = v25;
    v26 = *((_QWORD *)v16 - 4);
    if ( v26 == *(_QWORD *)a1[3] )
      goto LABEL_36;
    if ( !v26 )
    {
LABEL_49:
      v17 = *v16;
      goto LABEL_27;
    }
LABEL_48:
    *(_QWORD *)a1[2] = v26;
    if ( *((_QWORD *)v16 - 8) == *(_QWORD *)a1[3] )
      goto LABEL_36;
    goto LABEL_49;
  }
  v26 = *((_QWORD *)v16 - 4);
  if ( v26 )
    goto LABEL_48;
LABEL_30:
  if ( !v16 )
    return 0;
  *(_QWORD *)a1[1] = v16;
  v18 = *(_BYTE **)(v4 - 64);
  v19 = *v18;
  if ( *v18 == 42 )
  {
    v31 = *((_QWORD *)v18 - 8);
    if ( v31 )
    {
      *(_QWORD *)a1[2] = v31;
      v32 = *((_QWORD *)v18 - 4);
      if ( v32 == *(_QWORD *)a1[3] )
        goto LABEL_36;
      if ( !v32 )
      {
LABEL_65:
        v19 = *v18;
        goto LABEL_32;
      }
    }
    else
    {
      v32 = *((_QWORD *)v18 - 4);
      if ( !v32 )
        return 0;
    }
    *(_QWORD *)a1[2] = v32;
    if ( *((_QWORD *)v18 - 8) == *(_QWORD *)a1[3] )
      goto LABEL_36;
    goto LABEL_65;
  }
LABEL_32:
  if ( v19 == 59 )
  {
    v39 = *((_QWORD *)v18 - 8);
    if ( v39 )
    {
      *(_QWORD *)a1[4] = v39;
      v40 = *((_QWORD *)v18 - 4);
      if ( v40 == *(_QWORD *)a1[5] )
        goto LABEL_36;
      if ( !v40 )
      {
LABEL_85:
        v19 = *v18;
        goto LABEL_33;
      }
    }
    else
    {
      v40 = *((_QWORD *)v18 - 4);
      if ( !v40 )
        return 0;
    }
    *(_QWORD *)a1[4] = v40;
    if ( *((_QWORD *)v18 - 8) == *(_QWORD *)a1[5] )
      goto LABEL_36;
    goto LABEL_85;
  }
LABEL_33:
  if ( v19 != 44 )
    return 0;
  v20 = *((_QWORD *)v18 - 8);
  if ( !v20 )
    return 0;
  *(_QWORD *)a1[6] = v20;
  if ( *((_QWORD *)v18 - 4) != *(_QWORD *)a1[7] )
    return 0;
LABEL_36:
  if ( *(_QWORD *)a1[8] != *(_QWORD *)(a2 - 64) )
    return 0;
  if ( !*a1 )
    return 1;
  v21 = sub_B53960(a2);
  v22 = *a1;
  *(_DWORD *)v22 = v21;
  *(_BYTE *)(v22 + 4) = BYTE4(v21);
  return 1;
}
