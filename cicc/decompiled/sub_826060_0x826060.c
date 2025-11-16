// Function: sub_826060
// Address: 0x826060
//
void __fastcall sub_826060(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rax
  char v4; // dl
  char v5; // al
  const char *v6; // rcx
  __int64 v7; // rax
  const char *v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  const char *v11; // rcx
  __int64 **v12; // r12
  char v13; // cl
  __int64 v14; // rax
  int v15; // r12d
  int v16; // r14d
  __int64 *v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  char v23; // al
  int v24; // [rsp-3Ch] [rbp-3Ch]

  if ( !a1 || (*(_BYTE *)(a1 + 193) & 0x10) != 0 )
    return;
  v3 = *(_QWORD *)(a1 + 152);
  v4 = *(_BYTE *)(a1 + 198);
  if ( v3 )
  {
    if ( (v4 & 0x18) == 0x10 && (v4 & 0x30) != 0x10 )
    {
      v5 = *(_BYTE *)(a1 + 199);
      goto LABEL_7;
    }
    while ( *(_BYTE *)(v3 + 140) == 12 )
      v3 = *(_QWORD *)(v3 + 160);
    v12 = **(__int64 ****)(v3 + 168);
    if ( v12 )
    {
      do
      {
        while ( ((_BYTE)v12[4] & 2) == 0 )
        {
          v12 = (__int64 **)*v12;
          if ( !v12 )
            goto LABEL_41;
        }
        sub_684AA0(7u, 0xE76u, a2);
        v12 = (__int64 **)*v12;
      }
      while ( v12 );
LABEL_41:
      v4 = *(_BYTE *)(a1 + 198);
    }
  }
  v5 = *(_BYTE *)(a1 + 199);
  if ( (v5 & 0x10) != 0 && (v4 & 0x30) == 0x10 )
  {
    if ( (*(_BYTE *)(a1 + 196) & 0x40) != 0 )
      sub_684AA0(7u, 0xE80u, a2);
    if ( *(char *)(a1 + 192) < 0 && (*(_BYTE *)(a1 + 202) & 1) != 0 )
      sub_684AA0(7u, 0xE81u, a2);
    v5 = *(_BYTE *)(a1 + 199);
    v4 = *(_BYTE *)(a1 + 198);
  }
LABEL_7:
  if ( (v5 & 4) == 0 )
    goto LABEL_50;
  if ( (v4 & 0x20) != 0 )
  {
    v6 = "__global__";
    goto LABEL_49;
  }
  v6 = "__host__";
  if ( (v4 & 0x18) != 0x10 )
  {
LABEL_49:
    sub_6849F0(7u, 0xE4Du, dword_4F07508, (__int64)v6);
    v4 = *(_BYTE *)(a1 + 198);
LABEL_50:
    v7 = *(_QWORD *)(a1 + 328);
    v13 = v4 & 0x20;
    if ( !v7 )
      goto LABEL_112;
    if ( v13 )
    {
LABEL_17:
      v9 = *(int *)(v7 + 20);
      goto LABEL_18;
    }
    if ( *(_QWORD *)v7 )
      goto LABEL_12;
LABEL_56:
    if ( !*(_QWORD *)(v7 + 8) )
      goto LABEL_13;
    goto LABEL_12;
  }
  v7 = *(_QWORD *)(a1 + 328);
  if ( !v7 )
    goto LABEL_30;
  if ( !*(_QWORD *)v7 )
    goto LABEL_56;
LABEL_12:
  sub_6849F0(7u, 0xDCEu, dword_4F07508, (__int64)"__launch_bounds__");
  v7 = *(_QWORD *)(a1 + 328);
LABEL_13:
  if ( (*(_BYTE *)(a1 + 199) & 0x20) != 0 || (v9 = *(int *)(v7 + 20), (int)v9 >= 0) )
  {
    v8 = "__block_size__";
    if ( *(int *)(v7 + 40) <= 0 )
      v8 = "__cluster_dims__";
    sub_6849F0(7u, 0xDCEu, dword_4F07508, (__int64)v8);
    v7 = *(_QWORD *)(a1 + 328);
    if ( !v7 )
      goto LABEL_111;
    goto LABEL_17;
  }
LABEL_18:
  if ( (int)v9 > 0 )
  {
    v10 = *(int *)(v7 + 16);
    if ( (int)v10 > 0 && v10 < *(int *)(v7 + 28) * (__int64)*(int *)(v7 + 24) * v9 )
    {
      v11 = "__block_size__";
      if ( *(int *)(v7 + 40) <= 0 )
        v11 = "__cluster_dims__";
      sub_6849F0(7u, 0xE7Bu, dword_4F07508, (__int64)v11);
      v7 = *(_QWORD *)(a1 + 328);
      if ( !v7 )
        goto LABEL_111;
    }
  }
  if ( *(int *)(v7 + 32) >= 0 && (*(_BYTE *)(a1 + 198) & 0x20) == 0 )
  {
    sub_6849F0(7u, 0xE83u, dword_4F07508, (__int64)"__maxnreg__");
    v7 = *(_QWORD *)(a1 + 328);
    if ( !v7 )
    {
LABEL_111:
      v4 = *(_BYTE *)(a1 + 198);
      v13 = v4 & 0x20;
LABEL_112:
      if ( !v13 )
      {
        if ( !unk_4F072F3 )
          return;
        goto LABEL_31;
      }
LABEL_81:
      sub_684AA0(4u, 0xE6Fu, &dword_4F063F8);
      v4 = *(_BYTE *)(a1 + 198);
      if ( !unk_4F072F3 )
        goto LABEL_32;
      goto LABEL_31;
    }
  }
  if ( !*(_QWORD *)v7 || *(int *)(v7 + 32) < 0 )
  {
    v4 = *(_BYTE *)(a1 + 198);
    if ( (v4 & 0x20) == 0 )
      goto LABEL_30;
LABEL_58:
    if ( *(_QWORD *)v7 || *(_QWORD *)(v7 + 8) )
    {
      if ( !unk_4F072F3 )
        goto LABEL_60;
      goto LABEL_31;
    }
    goto LABEL_81;
  }
  sub_6849F0(7u, 0xE87u, dword_4F07508, (__int64)"__launch_bounds__ and __maxnreg__");
  v4 = *(_BYTE *)(a1 + 198);
  if ( (v4 & 0x20) != 0 )
  {
    v7 = *(_QWORD *)(a1 + 328);
    if ( !v7 )
      goto LABEL_81;
    goto LABEL_58;
  }
LABEL_30:
  if ( !unk_4F072F3 )
    return;
LABEL_31:
  if ( (v4 & 0x10) != 0 )
  {
    if ( (unsigned int)sub_826000(*(const char **)(a1 + 8)) )
      sub_6849F0(7u, 0xE4Au, a2, *(_QWORD *)(a1 + 8));
    v4 = *(_BYTE *)(a1 + 198);
  }
LABEL_32:
  if ( (v4 & 0x20) == 0 )
    return;
LABEL_60:
  v14 = *(_QWORD *)(a1 + 152);
  if ( !v14 )
    return;
  while ( *(_BYTE *)(v14 + 140) == 12 )
    v14 = *(_QWORD *)(v14 + 160);
  v24 = 0;
  v15 = 0;
  v16 = 0;
  v17 = **(__int64 ***)(v14 + 168);
  if ( v17 )
  {
    while ( 1 )
    {
      if ( (v17[4] & 2) == 0 )
        goto LABEL_76;
      *(_BYTE *)(a1 + 199) |= 8u;
      if ( (*(_BYTE *)(a1 + 195) & 8) != 0 )
        goto LABEL_89;
      v18 = v17[2];
      if ( !v18 )
        goto LABEL_76;
      if ( (*(_BYTE *)(v18 + 140) & 0xFB) != 8 )
        break;
      if ( (sub_8D4C10(v18, dword_4F077C4 != 2) & 1) == 0 && (*((_BYTE *)v17 + 33) & 8) == 0 )
        goto LABEL_88;
LABEL_89:
      v19 = v17[2];
      if ( !v19 || !(unsigned int)sub_8D2FB0(v19) )
        goto LABEL_76;
LABEL_91:
      sub_6851C0(0xE71u, a2);
LABEL_76:
      if ( !v16 )
      {
        v16 = sub_8D3C40(v17[1]);
        if ( v16 )
        {
          v16 = 1;
          sub_6851C0(0xDF4u, a2);
        }
      }
      if ( !v24 && (*(_BYTE *)(a1 + 195) & 8) == 0 && (unsigned int)sub_8D3110(v17[1]) )
      {
        sub_6851C0(0xDC0u, a2);
        v24 = 1;
      }
      if ( !v15 )
      {
        v20 = v17[2];
        if ( v20 )
        {
          while ( 1 )
          {
            v21 = *(_BYTE *)(v20 + 140);
            if ( v21 != 12 )
              break;
            if ( (*(_BYTE *)(v20 + 143) & 0x10) != 0 )
              goto LABEL_102;
            v20 = *(_QWORD *)(v20 + 160);
            if ( !v20 )
              goto LABEL_69;
          }
          if ( !dword_4D0455C )
          {
            v15 = 0;
            goto LABEL_69;
          }
          if ( v21 == 8 )
          {
            v22 = *(_QWORD *)(v20 + 160);
            if ( v22 )
            {
              if ( (unsigned __int8)(*(_BYTE *)(v22 + 140) - 9) <= 2u
                && (*(_BYTE *)(*(_QWORD *)(v22 + 168) + 110LL) & 0x40) != 0 )
              {
LABEL_102:
                v15 = 1;
                sub_684AA0(7u, 0xE04u, a2);
                v17 = (__int64 *)*v17;
                if ( v17 )
                  continue;
                goto LABEL_103;
              }
            }
          }
        }
      }
LABEL_69:
      v17 = (__int64 *)*v17;
      if ( !v17 )
        goto LABEL_103;
    }
    if ( (*((_BYTE *)v17 + 33) & 8) != 0 )
    {
      if ( !(unsigned int)sub_8D2FB0(v18) )
        goto LABEL_76;
      goto LABEL_91;
    }
LABEL_88:
    sub_6851C0(0xE70u, a2);
    goto LABEL_89;
  }
LABEL_103:
  v23 = *(_BYTE *)(a1 + 193);
  if ( (v23 & 4) != 0 )
  {
    sub_6851C0(0xE7Fu, a2);
  }
  else if ( (v23 & 3) != 0 )
  {
    sub_6851C0(0xDF7u, a2);
  }
  if ( (unsigned int)sub_825FC0(*(_QWORD *)(a1 + 40)) )
    sub_6851C0(0xDFDu, a2);
}
