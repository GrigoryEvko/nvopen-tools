// Function: sub_1791020
// Address: 0x1791020
//
__int64 __fastcall sub_1791020(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  _BYTE *v4; // r13
  _QWORD *v5; // rbx
  char v6; // al
  char v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // eax
  unsigned int v11; // r13d
  unsigned __int8 v12; // al
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int8 v16; // al
  unsigned int v17; // r14d
  bool v18; // al
  _QWORD *v19; // rax
  __int64 v20; // r15
  unsigned __int8 v21; // al
  unsigned int v22; // r14d
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned int v28; // r14d
  bool v29; // al
  int v30; // r14d
  unsigned int v31; // r15d
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // r14d
  __int64 v35; // rax
  unsigned int v36; // r14d
  __int64 v37; // rax
  unsigned int v38; // r14d
  bool v39; // al
  unsigned int v40; // r14d
  __int64 v41; // rax
  char v42; // cl
  unsigned int v43; // r8d
  int v44; // r14d
  unsigned int v45; // edx
  __int64 v46; // rax
  char v47; // cl
  unsigned int v48; // edx
  unsigned int v49; // r8d
  int v50; // eax
  bool v51; // al
  int v52; // r14d
  unsigned int v53; // edx
  __int64 v54; // rax
  unsigned int v55; // r8d
  int v56; // eax
  bool v57; // al
  int v58; // [rsp+8h] [rbp-48h]
  int v59; // [rsp+8h] [rbp-48h]
  int v60; // [rsp+8h] [rbp-48h]
  int v61; // [rsp+Ch] [rbp-44h]
  unsigned int v62; // [rsp+Ch] [rbp-44h]
  unsigned int v63; // [rsp+Ch] [rbp-44h]
  _QWORD *v64; // [rsp+10h] [rbp-40h] BYREF

  v4 = (_BYTE *)a3;
  v5 = (_QWORD *)a2;
  v6 = *(_BYTE *)(a1 + 16);
  v64 = (_QWORD *)a2;
  if ( v6 != 52 )
  {
    if ( v6 != 5 || *(_WORD *)(a1 + 18) != 28 )
      goto LABEL_4;
    v24 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v25 = 4 * v24;
    v26 = *(_QWORD *)(a1 - 24 * v24);
    if ( v26 )
    {
      *(_QWORD *)a2 = v26;
      v25 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      v27 = *(_QWORD *)(a1 + 24 * (1 - v25));
      if ( *(_BYTE *)(v27 + 16) == 13 )
      {
        v28 = *(_DWORD *)(v27 + 32);
        if ( v28 <= 0x40 )
        {
          a4 = 64 - v28;
          v29 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v28) == *(_QWORD *)(v27 + 24);
        }
        else
        {
          v29 = v28 == (unsigned int)sub_16A58F0(v27 + 24);
        }
        goto LABEL_34;
      }
      if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) == 16 )
      {
        v35 = sub_15A1020(*(_BYTE **)(a1 + 24 * (1 - v25)), a2, v25, a4);
        if ( v35 && *(_BYTE *)(v35 + 16) == 13 )
        {
          v36 = *(_DWORD *)(v35 + 32);
          if ( v36 <= 0x40 )
          {
            a4 = 64 - v36;
            v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36);
            v29 = v25 == *(_QWORD *)(v35 + 24);
          }
          else
          {
            v29 = v36 == (unsigned int)sub_16A58F0(v35 + 24);
          }
LABEL_34:
          if ( v29 )
            goto LABEL_18;
          goto LABEL_35;
        }
        v52 = *(_QWORD *)(*(_QWORD *)v27 + 32LL);
        if ( !v52 )
          goto LABEL_18;
        v53 = 0;
        while ( 1 )
        {
          v63 = v53;
          v54 = sub_15A0A60(v27, v53);
          if ( !v54 )
            break;
          a4 = *(unsigned __int8 *)(v54 + 16);
          v25 = v63;
          if ( (_BYTE)a4 != 9 )
          {
            if ( (_BYTE)a4 != 13 )
              break;
            v55 = *(_DWORD *)(v54 + 32);
            if ( v55 <= 0x40 )
            {
              a4 = 64 - v55;
              v57 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v55) == *(_QWORD *)(v54 + 24);
            }
            else
            {
              v60 = *(_DWORD *)(v54 + 32);
              v56 = sub_16A58F0(v54 + 24);
              v25 = v63;
              v57 = v60 == v56;
            }
            if ( !v57 )
              break;
          }
          v53 = v25 + 1;
          if ( v52 == v53 )
            goto LABEL_18;
        }
      }
    }
LABEL_35:
    v23 = sub_1790680(&v64, a1, v25, a4);
    goto LABEL_28;
  }
  v14 = *(_QWORD *)(a1 - 48);
  if ( !v14 )
    goto LABEL_20;
  *(_QWORD *)a2 = v14;
  v15 = *(_QWORD *)(a1 - 24);
  v16 = *(_BYTE *)(v15 + 16);
  if ( v16 == 13 )
  {
    v17 = *(_DWORD *)(v15 + 32);
    if ( v17 <= 0x40 )
      v18 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) == *(_QWORD *)(v15 + 24);
    else
      v18 = v17 == (unsigned int)sub_16A58F0(v15 + 24);
    if ( v18 )
      goto LABEL_18;
    goto LABEL_24;
  }
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 && v16 <= 0x10u )
  {
    v37 = sub_15A1020(*(_BYTE **)(a1 - 24), a2, a3, *(_QWORD *)v15);
    if ( v37 && *(_BYTE *)(v37 + 16) == 13 )
    {
      v38 = *(_DWORD *)(v37 + 32);
      if ( v38 <= 0x40 )
      {
        a3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38);
        v39 = a3 == *(_QWORD *)(v37 + 24);
      }
      else
      {
        v39 = v38 == (unsigned int)sub_16A58F0(v37 + 24);
      }
      if ( v39 )
        goto LABEL_18;
      v15 = *(_QWORD *)(a1 - 24);
      if ( !v15 )
        goto LABEL_4;
      goto LABEL_21;
    }
    v40 = 0;
    v61 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
    if ( !v61 )
      goto LABEL_18;
    while ( 1 )
    {
      a2 = v40;
      v41 = sub_15A0A60(v15, v40);
      if ( !v41 )
        break;
      v42 = *(_BYTE *)(v41 + 16);
      if ( v42 != 9 )
      {
        if ( v42 != 13 )
          break;
        v43 = *(_DWORD *)(v41 + 32);
        if ( v43 <= 0x40 )
        {
          if ( *(_QWORD *)(v41 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v43) )
            break;
        }
        else
        {
          v58 = *(_DWORD *)(v41 + 32);
          if ( v58 != (unsigned int)sub_16A58F0(v41 + 24) )
            break;
        }
      }
      if ( v61 == ++v40 )
        goto LABEL_18;
    }
LABEL_20:
    v15 = *(_QWORD *)(a1 - 24);
    if ( !v15 )
      goto LABEL_4;
LABEL_21:
    v19 = v64;
    goto LABEL_25;
  }
LABEL_24:
  v19 = (_QWORD *)a2;
LABEL_25:
  *v19 = v15;
  v20 = *(_QWORD *)(a1 - 48);
  v21 = *(_BYTE *)(v20 + 16);
  if ( v21 == 13 )
  {
    v22 = *(_DWORD *)(v20 + 32);
    if ( v22 <= 0x40 )
      v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) == *(_QWORD *)(v20 + 24);
    else
      v23 = v22 == (unsigned int)sub_16A58F0(v20 + 24);
    goto LABEL_28;
  }
  if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 16 && v21 <= 0x10u )
  {
    v33 = sub_15A1020(*(_BYTE **)(a1 - 48), a2, a3, *(_QWORD *)v20);
    if ( v33 && *(_BYTE *)(v33 + 16) == 13 )
    {
      v34 = *(_DWORD *)(v33 + 32);
      if ( v34 <= 0x40 )
        v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v34) == *(_QWORD *)(v33 + 24);
      else
        v23 = v34 == (unsigned int)sub_16A58F0(v33 + 24);
LABEL_28:
      if ( !v23 )
        goto LABEL_4;
LABEL_18:
      *v4 |= (unsigned __int8)sub_1648D00(a1, 3) ^ 1;
      return 1;
    }
    v44 = *(_QWORD *)(*(_QWORD *)v20 + 32LL);
    if ( !v44 )
      goto LABEL_18;
    v45 = 0;
    while ( 1 )
    {
      v62 = v45;
      v46 = sub_15A0A60(v20, v45);
      if ( !v46 )
        break;
      v47 = *(_BYTE *)(v46 + 16);
      v48 = v62;
      if ( v47 != 9 )
      {
        if ( v47 != 13 )
          break;
        v49 = *(_DWORD *)(v46 + 32);
        if ( v49 <= 0x40 )
        {
          v51 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49) == *(_QWORD *)(v46 + 24);
        }
        else
        {
          v59 = *(_DWORD *)(v46 + 32);
          v50 = sub_16A58F0(v46 + 24);
          v48 = v62;
          v51 = v59 == v50;
        }
        if ( !v51 )
          break;
      }
      v45 = v48 + 1;
      if ( v44 == v45 )
        goto LABEL_18;
    }
  }
LABEL_4:
  v7 = sub_1648D00(a1, 3);
  LOBYTE(v10) = sub_15FB730(a1, 3, v8, v9);
  v11 = v10;
  if ( (_BYTE)v10 )
    goto LABEL_11;
  v12 = *(_BYTE *)(a1 + 16);
  if ( v12 == 13 )
    goto LABEL_11;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 && v12 <= 0x10u )
  {
    v30 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( v30 )
    {
      v31 = 0;
      do
      {
        v32 = sub_15A0A60(a1, v31);
        if ( !v32 || (*(_BYTE *)(v32 + 16) & 0xFB) != 9 )
          return v11;
      }
      while ( v30 != ++v31 );
    }
LABEL_11:
    *v5 = 0;
    return 1;
  }
  if ( v12 > 0x17u
    && ((unsigned __int8)(v12 - 75) <= 1u
     || (unsigned int)v12 - 35 <= 0x11
     && ((v12 - 35) & 0xFD) == 0
     && (*(_BYTE *)(*(_QWORD *)(a1 - 48) + 16LL) <= 0x10u || *(_BYTE *)(*(_QWORD *)(a1 - 24) + 16LL) <= 0x10u))
    && !v7 )
  {
    goto LABEL_11;
  }
  return v11;
}
