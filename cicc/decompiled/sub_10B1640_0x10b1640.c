// Function: sub_10B1640
// Address: 0x10b1640
//
bool __fastcall sub_10B1640(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v10; // rcx
  unsigned int v11; // r13d
  int v12; // eax
  bool v13; // al
  __int64 *v14; // rax
  _BYTE *v15; // r13
  _BYTE *v16; // r15
  __int64 v17; // rcx
  unsigned int v18; // r15d
  int v19; // eax
  bool v20; // al
  __int64 *v21; // rax
  bool v22; // al
  unsigned __int8 *v23; // rdx
  __int64 v24; // r13
  _BYTE *v25; // rax
  unsigned int v26; // r13d
  int v27; // eax
  __int64 v28; // r15
  __int64 v29; // rdx
  _BYTE *v30; // rax
  unsigned int v31; // r15d
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  bool v35; // r13
  __int64 v36; // rax
  unsigned int v37; // r13d
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  bool v41; // r15
  __int64 v42; // rax
  unsigned int v43; // r15d
  int v44; // eax
  __int64 v45; // [rsp-48h] [rbp-48h]
  __int64 v46; // [rsp-48h] [rbp-48h]
  __int64 v47; // [rsp-40h] [rbp-40h]
  __int64 v48; // [rsp-40h] [rbp-40h]
  __int64 v49; // [rsp-40h] [rbp-40h]
  __int64 v50; // [rsp-40h] [rbp-40h]
  int v51; // [rsp-40h] [rbp-40h]
  int v52; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return 0;
  v3 = *(_QWORD *)(a2 - 64);
  if ( !v3 )
    goto LABEL_14;
  v4 = *(_QWORD **)a1;
  **(_QWORD **)a1 = v3;
  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 || *(_QWORD *)(v6 + 8) || *(_BYTE *)v5 != 58 )
    goto LABEL_4;
  v16 = *(_BYTE **)(v5 - 64);
  if ( *v16 != 44 )
    goto LABEL_20;
  v10 = *((_QWORD *)v16 - 8);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
    {
      v13 = *(_QWORD *)(v10 + 24) == 0;
    }
    else
    {
      v47 = *((_QWORD *)v16 - 8);
      v12 = sub_C444A0(v10 + 24);
      v10 = v47;
      v13 = v11 == v12;
    }
    goto LABEL_9;
  }
  v24 = *(_QWORD *)(v10 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 && *(_BYTE *)v10 <= 0x15u )
  {
    v49 = *((_QWORD *)v16 - 8);
    v25 = sub_AD7630(v49, 0, (__int64)v4);
    v10 = v49;
    if ( !v25 || *v25 != 17 )
    {
      if ( *(_BYTE *)(v24 + 8) == 17 )
      {
        v33 = *(_DWORD *)(v24 + 32);
        v34 = 0;
        v35 = 0;
        v51 = v33;
        while ( v51 != (_DWORD)v34 )
        {
          v45 = v10;
          v36 = sub_AD69F0((unsigned __int8 *)v10, v34);
          v10 = v45;
          if ( !v36 )
            goto LABEL_37;
          if ( *(_BYTE *)v36 != 13 )
          {
            if ( *(_BYTE *)v36 != 17 )
              goto LABEL_37;
            v37 = *(_DWORD *)(v36 + 32);
            if ( v37 <= 0x40 )
            {
              v35 = *(_QWORD *)(v36 + 24) == 0;
            }
            else
            {
              v38 = sub_C444A0(v36 + 24);
              v10 = v45;
              v35 = v37 == v38;
            }
            if ( !v35 )
              goto LABEL_37;
          }
          v34 = (unsigned int)(v34 + 1);
        }
        if ( v35 )
        {
LABEL_10:
          v14 = *(__int64 **)(a1 + 8);
          if ( v14 )
            *v14 = v10;
          v15 = *(_BYTE **)(v5 - 32);
          if ( *((_QWORD *)v16 - 4) == **(_QWORD **)(a1 + 16) && **(_BYTE ***)(a1 + 24) == v15 )
            return 1;
LABEL_13:
          if ( *v15 != 44 )
            goto LABEL_14;
          goto LABEL_21;
        }
      }
LABEL_37:
      v15 = *(_BYTE **)(v5 - 32);
      goto LABEL_13;
    }
    v26 = *((_DWORD *)v25 + 8);
    if ( v26 <= 0x40 )
    {
      v13 = *((_QWORD *)v25 + 3) == 0;
    }
    else
    {
      v27 = sub_C444A0((__int64)(v25 + 24));
      v10 = v49;
      v13 = v26 == v27;
    }
LABEL_9:
    if ( v13 )
      goto LABEL_10;
    goto LABEL_37;
  }
LABEL_20:
  v15 = *(_BYTE **)(v5 - 32);
  if ( *v15 != 44 )
    goto LABEL_4;
LABEL_21:
  v17 = *((_QWORD *)v15 - 8);
  if ( *(_BYTE *)v17 == 17 )
  {
    v18 = *(_DWORD *)(v17 + 32);
    if ( v18 <= 0x40 )
    {
      v20 = *(_QWORD *)(v17 + 24) == 0;
    }
    else
    {
      v48 = *((_QWORD *)v15 - 8);
      v19 = sub_C444A0(v17 + 24);
      v17 = v48;
      v20 = v18 == v19;
    }
    if ( !v20 )
      goto LABEL_14;
  }
  else
  {
    v28 = *(_QWORD *)(v17 + 8);
    v29 = (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17;
    if ( (unsigned int)v29 > 1 || *(_BYTE *)v17 > 0x15u )
      goto LABEL_14;
    v50 = *((_QWORD *)v15 - 8);
    v30 = sub_AD7630(v50, 0, v29);
    v17 = v50;
    if ( v30 && *v30 == 17 )
    {
      v31 = *((_DWORD *)v30 + 8);
      if ( v31 <= 0x40 )
      {
        if ( *((_QWORD *)v30 + 3) )
          goto LABEL_14;
      }
      else
      {
        v32 = sub_C444A0((__int64)(v30 + 24));
        v17 = v50;
        if ( v31 != v32 )
          goto LABEL_14;
      }
    }
    else
    {
      if ( *(_BYTE *)(v28 + 8) != 17 )
        goto LABEL_14;
      v39 = *(_DWORD *)(v28 + 32);
      v40 = 0;
      v41 = 0;
      v52 = v39;
      while ( v52 != (_DWORD)v40 )
      {
        v46 = v17;
        v42 = sub_AD69F0((unsigned __int8 *)v17, v40);
        if ( !v42 )
          goto LABEL_14;
        v17 = v46;
        if ( *(_BYTE *)v42 != 13 )
        {
          if ( *(_BYTE *)v42 != 17 )
            goto LABEL_14;
          v43 = *(_DWORD *)(v42 + 32);
          if ( v43 <= 0x40 )
          {
            v41 = *(_QWORD *)(v42 + 24) == 0;
          }
          else
          {
            v44 = sub_C444A0(v42 + 24);
            v17 = v46;
            v41 = v43 == v44;
          }
          if ( !v41 )
            goto LABEL_14;
        }
        v40 = (unsigned int)(v40 + 1);
      }
      if ( !v41 )
        goto LABEL_14;
    }
  }
  v21 = *(__int64 **)(a1 + 8);
  if ( v21 )
    *v21 = v17;
  if ( *((_QWORD *)v15 - 4) == **(_QWORD **)(a1 + 16) && *(_QWORD *)(v5 - 64) == **(_QWORD **)(a1 + 24) )
    return 1;
LABEL_14:
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 )
    return 0;
LABEL_4:
  **(_QWORD **)a1 = v5;
  v7 = *(_QWORD *)(a2 - 64);
  v8 = *(_QWORD *)(v7 + 16);
  if ( !v8 || *(_QWORD *)(v8 + 8) || *(_BYTE *)v7 != 58 )
    return 0;
  v22 = sub_10B14D0((__int64 **)(a1 + 8), 15, *(unsigned __int8 **)(v7 - 64));
  v23 = *(unsigned __int8 **)(v7 - 32);
  if ( v22 && v23 == **(unsigned __int8 ***)(a1 + 24) )
    return 1;
  if ( !sub_10B14D0((__int64 **)(a1 + 8), 15, v23) )
    return 0;
  return **(_QWORD **)(a1 + 24) == *(_QWORD *)(v7 - 64);
}
