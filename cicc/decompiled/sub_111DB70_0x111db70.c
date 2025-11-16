// Function: sub_111DB70
// Address: 0x111db70
//
unsigned __int8 *__fastcall sub_111DB70(_QWORD *a1, unsigned __int64 a2, __int64 a3, unsigned __int8 *a4, __int64 a5)
{
  _BYTE *v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int16 v22; // ax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // r12d
  unsigned __int16 v26; // ax
  __int64 v27; // rax
  unsigned int **v28; // r12
  const char *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  unsigned int **v32; // r12
  const char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // edx
  __int64 v43; // rax
  _BYTE *v45; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+20h] [rbp-80h]
  char v47; // [rsp+38h] [rbp-68h] BYREF
  const char *v48; // [rsp+40h] [rbp-60h] BYREF
  __int64 v49; // [rsp+48h] [rbp-58h]
  __int16 v50; // [rsp+60h] [rbp-40h]

  v7 = *(_BYTE **)(a3 - 64);
  v8 = sub_1016CC0(a2, v7, a4, a1 + 12);
  if ( !v8 )
  {
    v26 = sub_9A13D0(*(_QWORD *)(a3 - 96), a2, (__int64)v7, a4, a1[11], 1u, 0);
    v46 = 0;
    if ( !HIBYTE(v26) )
      goto LABEL_5;
    v27 = sub_AD64C0(*(_QWORD *)(a5 + 8), (unsigned __int8)v26, 0);
    v46 = v27;
    if ( !v27 )
      goto LABEL_5;
    v8 = v27;
  }
  v9 = 0;
  if ( *(_BYTE *)v8 == 17 )
    v9 = v8;
  v46 = v9;
LABEL_5:
  v45 = *(_BYTE **)(a3 - 32);
  v10 = sub_1016CC0(a2, v45, a4, a1 + 12);
  if ( v10 )
    goto LABEL_6;
  v22 = sub_9A13D0(*(_QWORD *)(a3 - 96), a2, (__int64)v45, a4, a1[11], 0, 0);
  if ( HIBYTE(v22) )
  {
    v23 = sub_AD64C0(*(_QWORD *)(a5 + 8), (unsigned __int8)v22, 0);
    if ( v23 )
    {
      v10 = v23;
LABEL_6:
      v11 = 0;
      if ( *(_BYTE *)v10 == 17 )
        v11 = v10;
      v46 = v11;
      if ( v8 )
        goto LABEL_9;
      v36 = *(_QWORD *)(a3 - 64);
      if ( *(_BYTE *)v36 != 85 )
        goto LABEL_34;
      v41 = *(_QWORD *)(v36 - 32);
      if ( !v41 )
        goto LABEL_34;
      goto LABEL_65;
    }
  }
  if ( v8 )
    goto LABEL_34;
  v36 = *(_QWORD *)(a3 - 64);
  if ( *(_BYTE *)v36 == 85 )
  {
    v41 = *(_QWORD *)(v36 - 32);
    if ( v41 )
    {
      v10 = 0;
LABEL_65:
      if ( !*(_BYTE *)v41 && *(_QWORD *)(v41 + 24) == *(_QWORD *)(v36 + 80) && (*(_BYTE *)(v41 + 33) & 0x20) != 0 )
      {
        v42 = *(_DWORD *)(v41 + 36);
        if ( v42 == 313 || v42 == 362 )
        {
          v43 = *(_QWORD *)(v36 + 16);
          if ( v43 )
          {
            if ( !*(_QWORD *)(v43 + 8) )
            {
              LOBYTE(v49) = 0;
              v48 = &v47;
              if ( (unsigned __int8)sub_991580((__int64)&v48, (__int64)a4) )
                goto LABEL_34;
            }
          }
        }
      }
      if ( v10 )
        goto LABEL_34;
    }
  }
  v37 = *(_QWORD *)(a3 - 32);
  if ( *(_BYTE *)v37 != 85 )
    return 0;
  v38 = *(_QWORD *)(v37 - 32);
  if ( !v38 || *(_BYTE *)v38 || *(_QWORD *)(v38 + 24) != *(_QWORD *)(v37 + 80) || (*(_BYTE *)(v38 + 33) & 0x20) == 0 )
    return 0;
  v39 = sub_987FE0(*(_QWORD *)(a3 - 32));
  if ( v39 != 313 )
  {
    v14 = 0;
    if ( v39 != 362 )
      return v14;
  }
  v40 = *(_QWORD *)(v37 + 16);
  if ( !v40 )
    return 0;
  v14 = *(unsigned __int8 **)(v40 + 8);
  if ( v14 )
    return 0;
  LOBYTE(v49) = 0;
  v48 = &v47;
  if ( !(unsigned __int8)sub_991580((__int64)&v48, (__int64)a4) )
    return v14;
  v10 = 0;
LABEL_34:
  v24 = *(_QWORD *)(a3 + 16);
  if ( v24 && !*(_QWORD *)(v24 + 8) )
  {
    if ( v8 )
      goto LABEL_47;
    goto LABEL_46;
  }
  if ( !v46 )
    return 0;
  v25 = *(_DWORD *)(v46 + 32);
  if ( v25 <= 0x40 )
  {
    if ( !*(_QWORD *)(v46 + 24) )
      return 0;
  }
  else if ( v25 == (unsigned int)sub_C444A0(v46 + 24) )
  {
    return 0;
  }
  if ( v8 )
  {
    if ( (unsigned __int8)sub_11176F0((__int64)a1, a3, a5, 2u) )
      goto LABEL_47;
    return 0;
  }
  if ( !(unsigned __int8)sub_11176F0((__int64)a1, a3, a5, 1u) )
    return 0;
LABEL_46:
  v28 = (unsigned int **)a1[4];
  v29 = sub_BD5D20(a5);
  v50 = 261;
  v49 = v30;
  v31 = *(_QWORD *)(a3 - 64);
  v48 = v29;
  v8 = sub_92B530(v28, a2, v31, a4, (__int64)&v48);
LABEL_47:
  if ( !v10 )
  {
    v32 = (unsigned int **)a1[4];
    v33 = sub_BD5D20(a5);
    v49 = v34;
    v35 = *(_QWORD *)(a3 - 32);
    v50 = 261;
    v48 = v33;
    v10 = sub_92B530(v32, a2, v35, a4, (__int64)&v48);
  }
LABEL_9:
  v12 = *(_QWORD *)(a3 - 96);
  v50 = 257;
  v13 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v14 = v13;
  if ( v13 )
  {
    sub_B44260((__int64)v13, *(_QWORD *)(v8 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v14 - 12) )
    {
      v15 = *((_QWORD *)v14 - 11);
      **((_QWORD **)v14 - 10) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *((_QWORD *)v14 - 10);
    }
    *((_QWORD *)v14 - 12) = v12;
    if ( v12 )
    {
      v16 = *(_QWORD *)(v12 + 16);
      *((_QWORD *)v14 - 11) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = v14 - 88;
      *((_QWORD *)v14 - 10) = v12 + 16;
      *(_QWORD *)(v12 + 16) = v14 - 96;
    }
    if ( *((_QWORD *)v14 - 8) )
    {
      v17 = *((_QWORD *)v14 - 7);
      **((_QWORD **)v14 - 6) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *((_QWORD *)v14 - 6);
    }
    *((_QWORD *)v14 - 8) = v8;
    v18 = *(_QWORD *)(v8 + 16);
    *((_QWORD *)v14 - 7) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = v14 - 56;
    *((_QWORD *)v14 - 6) = v8 + 16;
    *(_QWORD *)(v8 + 16) = v14 - 64;
    if ( *((_QWORD *)v14 - 4) )
    {
      v19 = *((_QWORD *)v14 - 3);
      **((_QWORD **)v14 - 2) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *((_QWORD *)v14 - 2);
    }
    *((_QWORD *)v14 - 4) = v10;
    if ( v10 )
    {
      v20 = *(_QWORD *)(v10 + 16);
      *((_QWORD *)v14 - 3) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = v14 - 24;
      *((_QWORD *)v14 - 2) = v10 + 16;
      *(_QWORD *)(v10 + 16) = v14 - 32;
    }
    sub_BD6B50(v14, &v48);
  }
  return v14;
}
