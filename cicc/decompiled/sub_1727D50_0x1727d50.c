// Function: sub_1727D50
// Address: 0x1727d50
//
__int64 __fastcall sub_1727D50(__int64 a1, __int64 a2)
{
  unsigned int v3; // r12d
  unsigned __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  bool v12; // al
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // r15d
  bool v17; // al
  __int64 v18; // rax
  unsigned int v19; // ebx
  int v20; // eax
  bool v21; // al
  int v22; // eax
  bool v23; // al
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  unsigned int v27; // r15d
  __int64 v28; // rax
  unsigned int v29; // esi
  unsigned int v30; // r15d
  __int64 v31; // rax
  char v32; // cl
  unsigned int v33; // esi
  bool v34; // al
  __int64 v35; // rax
  unsigned int v36; // r8d
  bool v37; // al
  __int64 v38; // rax
  int v39; // eax
  bool v40; // al
  __int64 v41; // rax
  int v42; // ecx
  int v43; // eax
  bool v44; // al
  int v45; // [rsp+0h] [rbp-40h]
  unsigned int v46; // [rsp+0h] [rbp-40h]
  unsigned int v47; // [rsp+0h] [rbp-40h]
  int v48; // [rsp+4h] [rbp-3Ch]
  int v49; // [rsp+4h] [rbp-3Ch]
  int v50; // [rsp+4h] [rbp-3Ch]
  int v51; // [rsp+4h] [rbp-3Ch]
  unsigned int v52; // [rsp+8h] [rbp-38h]
  unsigned int v53; // [rsp+8h] [rbp-38h]
  unsigned int v54; // [rsp+8h] [rbp-38h]
  unsigned int v55; // [rsp+8h] [rbp-38h]
  unsigned int v56; // [rsp+8h] [rbp-38h]
  int v57; // [rsp+8h] [rbp-38h]
  unsigned int v58; // [rsp+8h] [rbp-38h]
  int v59; // [rsp+8h] [rbp-38h]
  int v60; // [rsp+Ch] [rbp-34h]

  v60 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !v60 )
    return 1;
  v3 = 0;
  while ( 1 )
  {
    v4 = v3;
    v5 = sub_15A0A60(a1, v3);
    v6 = sub_15A0A60(a2, v3);
    v9 = v6;
    if ( !v5 || !v6 )
      return 0;
    if ( sub_1593BB0(v5, v3, v7, v8) )
      goto LABEL_6;
    if ( *(_BYTE *)(v5 + 16) == 13 )
    {
      v11 = *(unsigned int *)(v5 + 32);
      if ( (unsigned int)v11 <= 0x40 )
      {
        v21 = *(_QWORD *)(v5 + 24) == 0;
      }
      else
      {
        v53 = *(_DWORD *)(v5 + 32);
        v20 = sub_16A57B0(v5 + 24);
        v11 = v53;
        v21 = v53 == v20;
      }
      goto LABEL_28;
    }
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
      goto LABEL_15;
    v24 = sub_15A1020((_BYTE *)v5, v3, v10, v11);
    if ( v24 && *(_BYTE *)(v24 + 16) == 13 )
    {
      v11 = *(unsigned int *)(v24 + 32);
      if ( (unsigned int)v11 <= 0x40 )
      {
        v21 = *(_QWORD *)(v24 + 24) == 0;
      }
      else
      {
        v55 = *(_DWORD *)(v24 + 32);
        v25 = sub_16A57B0(v24 + 24);
        v11 = v55;
        v21 = v55 == v25;
      }
LABEL_28:
      if ( !v21 )
        goto LABEL_15;
      goto LABEL_6;
    }
    v50 = *(_QWORD *)(*(_QWORD *)v5 + 32LL);
    if ( v50 )
    {
      LODWORD(v11) = 0;
      do
      {
        v4 = (unsigned int)v11;
        v58 = v11;
        v38 = sub_15A0A60(v5, v11);
        if ( !v38 )
          goto LABEL_15;
        v4 = *(unsigned __int8 *)(v38 + 16);
        v11 = v58;
        if ( (_BYTE)v4 != 9 )
        {
          if ( (_BYTE)v4 != 13 )
            goto LABEL_15;
          v4 = *(unsigned int *)(v38 + 32);
          if ( (unsigned int)v4 <= 0x40 )
          {
            v40 = *(_QWORD *)(v38 + 24) == 0;
          }
          else
          {
            v46 = *(_DWORD *)(v38 + 32);
            v39 = sub_16A57B0(v38 + 24);
            v4 = v46;
            v11 = v58;
            v40 = v46 == v39;
          }
          if ( !v40 )
            goto LABEL_15;
        }
        v11 = (unsigned int)(v11 + 1);
      }
      while ( v50 != (_DWORD)v11 );
    }
LABEL_6:
    if ( *(_BYTE *)(v9 + 16) == 13 )
    {
      v4 = *(unsigned int *)(v9 + 32);
      if ( (unsigned int)v4 <= 0x40 )
      {
        v11 = (unsigned int)(64 - v4);
        v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) == *(_QWORD *)(v9 + 24);
      }
      else
      {
        v52 = *(_DWORD *)(v9 + 32);
        v4 = v52;
        v12 = v52 == (unsigned int)sub_16A58F0(v9 + 24);
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
        goto LABEL_15;
      v28 = sub_15A1020((_BYTE *)v9, v4, v10, v11);
      if ( !v28 || *(_BYTE *)(v28 + 16) != 13 )
      {
        v49 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
        if ( !v49 )
          goto LABEL_10;
        v4 = 0;
        while ( 1 )
        {
          v35 = sub_15A0A60(v9, v4);
          if ( !v35 )
            goto LABEL_15;
          v11 = *(unsigned __int8 *)(v35 + 16);
          v4 = (unsigned int)v4;
          if ( (_BYTE)v11 != 9 )
          {
            if ( (_BYTE)v11 != 13 )
              goto LABEL_15;
            v36 = *(_DWORD *)(v35 + 32);
            if ( v36 <= 0x40 )
            {
              v11 = 64 - v36;
              v37 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) == *(_QWORD *)(v35 + 24);
            }
            else
            {
              v45 = *(_DWORD *)(v35 + 32);
              v4 = (unsigned int)v4;
              v37 = v45 == (unsigned int)sub_16A58F0(v35 + 24);
            }
            if ( !v37 )
              goto LABEL_15;
          }
          v4 = (unsigned int)(v4 + 1);
          if ( v49 == (_DWORD)v4 )
            goto LABEL_10;
        }
      }
      v29 = *(_DWORD *)(v28 + 32);
      if ( v29 <= 0x40 )
      {
        v11 = 64 - v29;
        v4 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v29);
        v12 = v4 == *(_QWORD *)(v28 + 24);
      }
      else
      {
        v56 = *(_DWORD *)(v28 + 32);
        v4 = v56;
        v12 = v56 == (unsigned int)sub_16A58F0(v28 + 24);
      }
    }
    if ( v12 )
      goto LABEL_10;
LABEL_15:
    if ( sub_1593BB0(v9, v4, v10, v11) )
      goto LABEL_16;
    if ( *(_BYTE *)(v9 + 16) == 13 )
    {
      v15 = *(unsigned int *)(v9 + 32);
      if ( (unsigned int)v15 <= 0x40 )
      {
        v23 = *(_QWORD *)(v9 + 24) == 0;
      }
      else
      {
        v54 = *(_DWORD *)(v9 + 32);
        v22 = sub_16A57B0(v9 + 24);
        v15 = v54;
        v23 = v54 == v22;
      }
      goto LABEL_34;
    }
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
      return 0;
    v26 = sub_15A1020((_BYTE *)v9, v4, v14, v15);
    if ( v26 && *(_BYTE *)(v26 + 16) == 13 )
    {
      v27 = *(_DWORD *)(v26 + 32);
      if ( v27 <= 0x40 )
        v23 = *(_QWORD *)(v26 + 24) == 0;
      else
        v23 = v27 == (unsigned int)sub_16A57B0(v26 + 24);
LABEL_34:
      if ( !v23 )
        return 0;
      goto LABEL_16;
    }
    v51 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
    if ( v51 )
    {
      LODWORD(v15) = 0;
      do
      {
        v59 = v15;
        v41 = sub_15A0A60(v9, v15);
        if ( !v41 )
          return 0;
        v4 = *(unsigned __int8 *)(v41 + 16);
        v42 = v59;
        if ( (_BYTE)v4 != 9 )
        {
          if ( (_BYTE)v4 != 13 )
            return 0;
          v4 = *(unsigned int *)(v41 + 32);
          if ( (unsigned int)v4 <= 0x40 )
          {
            v44 = *(_QWORD *)(v41 + 24) == 0;
          }
          else
          {
            v47 = *(_DWORD *)(v41 + 32);
            v43 = sub_16A57B0(v41 + 24);
            v4 = v47;
            v42 = v59;
            v44 = v47 == v43;
          }
          if ( !v44 )
            return 0;
        }
        v15 = (unsigned int)(v42 + 1);
      }
      while ( v51 != (_DWORD)v15 );
    }
LABEL_16:
    if ( *(_BYTE *)(v5 + 16) == 13 )
    {
      v16 = *(_DWORD *)(v5 + 32);
      if ( v16 <= 0x40 )
        v17 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) == *(_QWORD *)(v5 + 24);
      else
        v17 = v16 == (unsigned int)sub_16A58F0(v5 + 24);
      goto LABEL_19;
    }
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
      return 0;
    v18 = sub_15A1020((_BYTE *)v5, v4, v14, v15);
    if ( v18 && *(_BYTE *)(v18 + 16) == 13 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
        v17 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v19) == *(_QWORD *)(v18 + 24);
      else
        v17 = v19 == (unsigned int)sub_16A58F0(v18 + 24);
LABEL_19:
      if ( !v17 )
        return 0;
      goto LABEL_10;
    }
    v57 = *(_QWORD *)(*(_QWORD *)v5 + 32LL);
    if ( v57 )
    {
      v30 = 0;
      do
      {
        v31 = sub_15A0A60(v5, v30);
        if ( !v31 )
          return 0;
        v32 = *(_BYTE *)(v31 + 16);
        if ( v32 != 9 )
        {
          if ( v32 != 13 )
            return 0;
          v33 = *(_DWORD *)(v31 + 32);
          if ( v33 <= 0x40 )
          {
            v34 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) == *(_QWORD *)(v31 + 24);
          }
          else
          {
            v48 = *(_DWORD *)(v31 + 32);
            v34 = v48 == (unsigned int)sub_16A58F0(v31 + 24);
          }
          if ( !v34 )
            return 0;
        }
      }
      while ( v57 != ++v30 );
    }
LABEL_10:
    if ( v60 == ++v3 )
      return 1;
  }
}
