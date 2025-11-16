// Function: sub_644920
// Address: 0x644920
//
void __fastcall sub_644920(_QWORD *a1, int a2)
{
  int v2; // ecx
  __int64 *i; // rax
  int v5; // ebx
  char v6; // bl
  __int64 v7; // r14
  char v8; // al
  __int64 v9; // rax
  char v10; // si
  char v11; // al
  __int64 v12; // rax
  unsigned __int8 v13; // r12
  char v14; // bl
  char v15; // al
  int v16; // r12d
  _BOOL4 v17; // r13d
  __int64 *v18; // rdx
  __int64 *v19; // rsi
  char v20; // al
  char v21; // dl
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int8 v27; // al
  char v28; // al
  bool v29; // zf
  __int64 v30; // rdi
  char v31; // al
  __int64 v32; // rax
  __int16 v33; // bx
  unsigned __int8 v34; // r12
  _BOOL4 v35; // ecx
  _BOOL4 v36; // r9d
  int v37; // r8d
  __int64 v38; // rdx
  __int64 v39; // rax
  char v40; // bl
  unsigned __int8 v41; // al
  char v42; // bl
  __int64 v43; // rax
  int v44; // [rsp+4h] [rbp-4Ch]
  int v45; // [rsp+8h] [rbp-48h]
  int v46; // [rsp+Ch] [rbp-44h]
  int v47; // [rsp+Ch] [rbp-44h]
  char v48; // [rsp+1Bh] [rbp-35h] BYREF
  _DWORD v49[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = a2;
  v48 = 0;
  i = (__int64 *)a1[25];
  if ( !unk_4D04598 )
    goto LABEL_3;
  v5 = unk_4D04630;
  if ( unk_4D04630 || (*((_BYTE *)a1 + 127) & 0x10) != 0 )
    goto LABEL_3;
  if ( dword_4F04C44 == -1 )
  {
    v38 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v38 + 6) & 6) == 0 && *(_BYTE *)(v38 + 4) != 12 )
    {
      v6 = 1;
      if ( !i && !a1[23] )
        goto LABEL_66;
      goto LABEL_4;
    }
  }
  if ( (a1[1] & 8) != 0 )
  {
LABEL_3:
    v6 = 0;
    if ( !i && !a1[23] )
      return;
LABEL_4:
    if ( (a1[1] & 8) != 0 && !a2 && (!HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4) )
    {
      v18 = (__int64 *)a1[23];
      if ( v18 )
      {
        v19 = 0;
        do
        {
          while ( 1 )
          {
            v20 = *((_BYTE *)v18 + 9);
            if ( v20 == 1 || v20 == 4 )
              break;
            v18 = (__int64 *)*v18;
            if ( !v18 )
              goto LABEL_30;
          }
          *((_BYTE *)v18 + 8) = 0;
          if ( !v19 )
            v19 = v18;
          v18 = (__int64 *)*v18;
        }
        while ( v18 );
LABEL_30:
        for ( i = (__int64 *)a1[25]; i; i = (__int64 *)*i )
        {
LABEL_34:
          while ( 1 )
          {
            v21 = *((_BYTE *)i + 9);
            if ( v21 == 1 || v21 == 4 )
              break;
            i = (__int64 *)*i;
            if ( !i )
              goto LABEL_38;
          }
          *((_BYTE *)i + 8) = 0;
          if ( !v19 )
            v19 = i;
        }
LABEL_38:
        if ( v19 )
        {
          sub_6851C0(1885, v19[5]);
          return;
        }
      }
      else if ( i )
      {
        v19 = 0;
        goto LABEL_34;
      }
    }
    v7 = *a1;
    if ( !*a1 )
    {
      if ( !v6 )
      {
        sub_644830((__int64)a1, 0, 0, v2);
        return;
      }
      v10 = 0;
      goto LABEL_56;
    }
    v8 = *(_BYTE *)(v7 + 80);
    if ( v8 == 20 )
    {
      v22 = *(_QWORD *)(v7 + 88);
      v48 = 11;
      v10 = 11;
      v7 = *(_QWORD *)(v22 + 176);
      if ( v6 )
        goto LABEL_45;
      goto LABEL_10;
    }
    if ( v8 == 21 )
    {
      v9 = *(_QWORD *)(v7 + 88);
      v48 = 7;
      v10 = 7;
      v7 = *(_QWORD *)(v9 + 192);
      if ( !v6 )
        goto LABEL_10;
      goto LABEL_20;
    }
LABEL_53:
    v47 = v2;
    v26 = sub_87D510(v7, &v48);
    v10 = v48;
    v2 = v47;
    v7 = v26;
    if ( v6 )
    {
      if ( v48 != 11 )
      {
        if ( v48 != 7 )
        {
LABEL_56:
          v44 = 0;
          v16 = 0;
          v17 = 0;
          v5 = 0;
          v46 = 0;
          v45 = 0;
          goto LABEL_46;
        }
LABEL_20:
        v13 = *(_BYTE *)(v7 + 156);
        v14 = v13 >> 1;
        v15 = v13 >> 2;
        v16 = v13 & 1;
        v45 = v15 & 1;
        v46 = *(_BYTE *)(v7 + 157) & 1;
        v44 = v14 & 1;
        v10 = 7;
        v17 = 0;
        v5 = 0;
        *(_WORD *)(v7 + 156) = *(_WORD *)(v7 + 156) & 0xFEF8 | (dword_4D045EC != 0);
        goto LABEL_46;
      }
LABEL_45:
      v23 = *(_BYTE *)(v7 + 198);
      v44 = 0;
      v10 = 11;
      v46 = 0;
      v45 = 0;
      v17 = (v23 & 8) != 0;
      v5 = (v23 & 0x20) != 0;
      v16 = (v23 & 0x10) != 0;
      *(_BYTE *)(v7 + 198) = (16 * (dword_4D045EC != 0)) | v23 & 0xC7;
      goto LABEL_46;
    }
LABEL_10:
    sub_644830((__int64)a1, v10, v7, v2);
    if ( !v7 )
      return;
    goto LABEL_49;
  }
  if ( !i && !a1[23] )
  {
LABEL_66:
    v30 = *a1;
    if ( !*a1 )
      return;
    v31 = *(_BYTE *)(v30 + 80);
    if ( v31 == 20 )
    {
      v39 = *(_QWORD *)(v30 + 88);
      v48 = 11;
      v7 = *(_QWORD *)(v39 + 176);
    }
    else
    {
      if ( v31 == 21 )
      {
        v32 = *(_QWORD *)(v30 + 88);
        v48 = 7;
        v7 = *(_QWORD *)(v32 + 192);
LABEL_70:
        v33 = *(_WORD *)(v7 + 156);
        v34 = *(_BYTE *)(v7 + 156) & 1;
        v35 = (*(_BYTE *)(v7 + 156) & 4) != 0;
        v36 = (*(_BYTE *)(v7 + 156) & 2) != 0;
        v37 = *(_BYTE *)(v7 + 157) & 1;
        *(_WORD *)(v7 + 156) = v33 & 0xFEF8 | (dword_4D045EC != 0);
        sub_6421F0((_BYTE *)v7, a1, v34, v35, v37, v36);
        *(_BYTE *)(v7 + 156) |= v34;
        *(_WORD *)(v7 + 156) = *(_WORD *)(v7 + 156) & 0xFEF9 | (v33 | *(_WORD *)(v7 + 156)) & 0x106;
        goto LABEL_49;
      }
      v7 = sub_87D510(v30, &v48);
      if ( v48 != 11 )
      {
        if ( v48 == 7 )
          goto LABEL_70;
LABEL_48:
        if ( !v7 )
          return;
        goto LABEL_49;
      }
    }
    v40 = *(_BYTE *)(v7 + 198);
    *(_BYTE *)(v7 + 198) = v40 & 0xC7 | (16 * (dword_4D045EC != 0));
    v41 = sub_641DE0(v7, (v40 & 0x20) != 0, (v40 & 8) != 0, (v40 & 0x10) != 0, v49);
    if ( v41 != 3 )
      sub_6853B0(v41, v49[0], a1 + 6, *a1);
    v42 = *(_BYTE *)(v7 + 198) & 0xF7 | (8 * ((*(_BYTE *)(v7 + 198) & 8) != 0 || (v40 & 8) != 0)) | v40 & 0x30;
    *(_BYTE *)(v7 + 198) = v42;
    v29 = (v42 & 0x38) == 0;
    goto LABEL_63;
  }
  v7 = *a1;
  if ( *a1 )
  {
    v11 = *(_BYTE *)(v7 + 80);
    if ( v11 == 20 )
    {
      v43 = *(_QWORD *)(v7 + 88);
      v48 = 11;
      v7 = *(_QWORD *)(v43 + 176);
      goto LABEL_45;
    }
    v6 = 1;
    if ( v11 == 21 )
    {
      v12 = *(_QWORD *)(v7 + 88);
      v48 = 7;
      v7 = *(_QWORD *)(v12 + 192);
      goto LABEL_20;
    }
    goto LABEL_53;
  }
  v44 = 0;
  v10 = 0;
  v16 = 0;
  v17 = 0;
  v46 = 0;
  v45 = 0;
LABEL_46:
  sub_644830((__int64)a1, v10, v7, v2);
  if ( v48 == 11 )
  {
    v27 = sub_641DE0(v7, v5, v17, v16, v49);
    if ( v27 != 3 )
      sub_6853B0(v27, v49[0], a1 + 6, *a1);
    v28 = *(_BYTE *)(v7 + 198) & 0xC7
        | ((32 * (v5 | ((*(_BYTE *)(v7 + 198) & 0x20) != 0)))
         | (8 * ((*(_BYTE *)(v7 + 198) & 8) != 0 || v17))
         | (16 * (v16 | ((*(_BYTE *)(v7 + 198) & 0x10) != 0))))
        & 0x38;
    *(_BYTE *)(v7 + 198) = v28;
    v29 = (v28 & 0x38) == 0;
LABEL_63:
    *(_BYTE *)(v7 + 197) = (!v29 << 7) | *(_BYTE *)(v7 + 197) & 0x7F;
    goto LABEL_49;
  }
  if ( v48 != 7 )
    goto LABEL_48;
  sub_6421F0((_BYTE *)v7, a1, v16, v45, v46, v44);
  *(_WORD *)(v7 + 156) = *(_WORD *)(v7 + 156) & 0xFEF8
                       | (4 * (((*(_BYTE *)(v7 + 156) & 4) != 0) | (unsigned __int16)v45))
                       | (2 * (((*(_BYTE *)(v7 + 156) & 2) != 0) | (unsigned __int16)v44))
                       | *(_BYTE *)(v7 + 156) & 1
                       | v16
                       | ((*(_BYTE *)(v7 + 157) & 1 | (unsigned __int16)v46) << 8);
LABEL_49:
  if ( v48 == 11 )
  {
    v24 = *(_QWORD *)(v7 + 256);
    if ( v24 )
    {
      v25 = *(_QWORD *)(v24 + 24);
      if ( v25 )
      {
        *(_BYTE *)(v25 + 198) = *(_BYTE *)(v7 + 198) & 0x20 | *(_BYTE *)(v25 + 198) & 0xDF;
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 256) + 24LL) + 198LL) = *(_BYTE *)(v7 + 198) & 8
                                                                      | *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 256)
                                                                                             + 24LL)
                                                                                 + 198LL)
                                                                      & 0xF7;
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 256) + 24LL) + 198LL) = *(_BYTE *)(v7 + 198) & 0x10
                                                                      | *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 256)
                                                                                             + 24LL)
                                                                                 + 198LL)
                                                                      & 0xEF;
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 256) + 24LL) + 197LL) = *(_BYTE *)(v7 + 197) & 0x80
                                                                      | *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 256)
                                                                                             + 24LL)
                                                                                 + 197LL)
                                                                      & 0x7F;
      }
    }
  }
}
