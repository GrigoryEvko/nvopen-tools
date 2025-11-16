// Function: sub_14CBC40
// Address: 0x14cbc40
//
char __fastcall sub_14CBC40(__int64 *a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rax
  bool v10; // zf
  __int64 *v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned int v14; // eax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r14
  unsigned int v18; // eax
  __int64 v19; // rbx
  __int64 v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  unsigned int v24; // esi
  int v25; // eax
  bool v26; // al
  __int64 v27; // rbx
  unsigned int v28; // r12d
  __int16 v29; // dx
  __int64 v30; // rax
  __int64 v31; // r15
  unsigned int v32; // ebx
  bool v33; // al
  __int64 v34; // rdx
  unsigned int v35; // ebx
  __int64 *v36; // rdx
  unsigned int v37; // ebx
  __int64 v38; // rax
  unsigned int v39; // ebx
  unsigned int v40; // ebx
  char v41; // cl
  unsigned int v42; // r15d
  unsigned int v43; // r12d
  char v44; // cl
  unsigned int v45; // r15d
  unsigned int v46; // ebx
  __int64 v47; // rax
  char v48; // cl
  unsigned int v49; // r8d
  bool v50; // al
  int v52; // [rsp+4h] [rbp-6Ch]
  int v53; // [rsp+4h] [rbp-6Ch]
  int v54; // [rsp+4h] [rbp-6Ch]
  __int64 v55; // [rsp+8h] [rbp-68h]
  __int64 v56; // [rsp+8h] [rbp-68h]
  __int64 v57; // [rsp+8h] [rbp-68h]
  int v58; // [rsp+8h] [rbp-68h]
  int v59; // [rsp+8h] [rbp-68h]
  __int64 v60; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v61; // [rsp+20h] [rbp-50h] BYREF
  __int64 v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+30h] [rbp-40h]
  int v64; // [rsp+38h] [rbp-38h]

  LOBYTE(v5) = *(_BYTE *)(a2 + 16);
  if ( (_BYTE)v5 == 17 )
  {
    v61 = 6;
    v17 = *a1;
    v62 = 0;
    v63 = a2;
    if ( a2 != -16 && a2 != -8 )
      sub_164C220(&v61);
    v64 = a3;
    v18 = *(_DWORD *)(v17 + 8);
    if ( v18 >= *(_DWORD *)(v17 + 12) )
    {
      sub_14CB640(v17, 0);
      v18 = *(_DWORD *)(v17 + 8);
    }
    v19 = *(_QWORD *)v17 + 32LL * v18;
    if ( v19 )
    {
      *(_QWORD *)v19 = 6;
      *(_QWORD *)(v19 + 8) = 0;
      v20 = v63;
      v10 = v63 == -8;
      *(_QWORD *)(v19 + 16) = v63;
      if ( v20 != 0 && !v10 && v20 != -16 )
        sub_1649AC0(v19, v61 & 0xFFFFFFFFFFFFFFF8LL);
      *(_DWORD *)(v19 + 24) = v64;
      v18 = *(_DWORD *)(v17 + 8);
    }
    *(_DWORD *)(v17 + 8) = v18 + 1;
LABEL_46:
    LOBYTE(v5) = v63;
    if ( v63 != -8 && v63 != 0 && v63 != -16 )
      LOBYTE(v5) = sub_1649B30(&v61);
    return v5;
  }
  if ( (unsigned __int8)v5 <= 0x17u )
    return v5;
  v61 = 6;
  v6 = *a1;
  v62 = 0;
  v63 = a2;
  if ( a2 != -8 && a2 != -16 )
    sub_164C220(&v61);
  v64 = a3;
  v7 = *(_DWORD *)(v6 + 8);
  if ( v7 >= *(_DWORD *)(v6 + 12) )
  {
    sub_14CB640(v6, 0);
    v7 = *(_DWORD *)(v6 + 8);
  }
  v8 = *(_QWORD *)v6 + 32LL * v7;
  if ( v8 )
  {
    *(_QWORD *)v8 = 6;
    *(_QWORD *)(v8 + 8) = 0;
    v9 = v63;
    v10 = v63 == 0;
    *(_QWORD *)(v8 + 16) = v63;
    if ( v9 != -8 && !v10 && v9 != -16 )
      sub_1649AC0(v8, v61 & 0xFFFFFFFFFFFFFFF8LL);
    *(_DWORD *)(v8 + 24) = v64;
    v7 = *(_DWORD *)(v6 + 8);
  }
  *(_DWORD *)(v6 + 8) = v7 + 1;
  if ( v63 != 0 && v63 != -8 && v63 != -16 )
    sub_1649B30(&v61);
  LOBYTE(v5) = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)v5 <= 0x17u )
  {
    v29 = *(_WORD *)(a2 + 18);
    if ( v29 != 47 )
      goto LABEL_61;
  }
  else if ( (_BYTE)v5 != 71 )
  {
    goto LABEL_50;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v11 = *(__int64 **)(a2 - 8);
  else
    v11 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v12 = *v11;
  if ( *v11 )
    goto LABEL_21;
  if ( (unsigned __int8)v5 > 0x17u )
  {
LABEL_50:
    if ( (_BYTE)v5 != 69 )
      goto LABEL_51;
    goto LABEL_84;
  }
  v29 = *(_WORD *)(a2 + 18);
LABEL_61:
  if ( v29 != 45 )
  {
    v61 = (unsigned __int64)&v60;
    goto LABEL_63;
  }
LABEL_84:
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v36 = *(__int64 **)(a2 - 8);
  else
    v36 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v12 = *v36;
  if ( *v36 )
  {
LABEL_21:
    v60 = v12;
    goto LABEL_22;
  }
LABEL_51:
  v21 = &v60;
  v61 = (unsigned __int64)&v60;
  if ( (_BYTE)v5 != 52 )
  {
LABEL_63:
    if ( (_BYTE)v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return v5;
    v30 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( !*(_QWORD *)(a2 - 24 * v30) )
      goto LABEL_70;
    v60 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v31 = *(_QWORD *)(a2 + 24 * (1 - v30));
    if ( *(_BYTE *)(v31 + 16) == 13 )
    {
      v32 = *(_DWORD *)(v31 + 32);
      if ( v32 <= 0x40 )
        v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) == *(_QWORD *)(v31 + 24);
      else
        v33 = v32 == (unsigned int)sub_16A58F0(v31 + 24);
      goto LABEL_69;
    }
    if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) != 16 )
    {
LABEL_70:
      LOBYTE(v5) = sub_14CA6E0((_QWORD **)&v61, a2);
      goto LABEL_71;
    }
    v38 = sub_15A1020(v31);
    if ( v38 && *(_BYTE *)(v38 + 16) == 13 )
    {
      v39 = *(_DWORD *)(v38 + 32);
      if ( v39 <= 0x40 )
        v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39) == *(_QWORD *)(v38 + 24);
      else
        v33 = v39 == (unsigned int)sub_16A58F0(v38 + 24);
LABEL_69:
      v12 = v60;
      if ( v33 )
        goto LABEL_22;
      goto LABEL_70;
    }
    v46 = 0;
    v59 = *(_DWORD *)(*(_QWORD *)v31 + 32LL);
    while ( v59 != v46 )
    {
      v47 = sub_15A0A60(v31, v46);
      if ( !v47 )
        goto LABEL_70;
      v48 = *(_BYTE *)(v47 + 16);
      if ( v48 != 9 )
      {
        if ( v48 != 13 )
          goto LABEL_70;
        v49 = *(_DWORD *)(v47 + 32);
        if ( v49 <= 0x40 )
        {
          v50 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49) == *(_QWORD *)(v47 + 24);
        }
        else
        {
          v54 = *(_DWORD *)(v47 + 32);
          v50 = v54 == (unsigned int)sub_16A58F0(v47 + 24);
        }
        if ( !v50 )
          goto LABEL_70;
      }
      ++v46;
    }
LABEL_72:
    v12 = v60;
LABEL_22:
    LOBYTE(v5) = *(_BYTE *)(v12 + 16);
    if ( (unsigned __int8)v5 <= 0x17u && (_BYTE)v5 != 17 )
      return v5;
    v61 = 6;
    v13 = *a1;
    v62 = 0;
    v63 = v12;
    if ( v12 != -8 && v12 != -16 )
      sub_164C220(&v61);
    v64 = a3;
    v14 = *(_DWORD *)(v13 + 8);
    if ( v14 >= *(_DWORD *)(v13 + 12) )
    {
      sub_14CB640(v13, 0);
      v14 = *(_DWORD *)(v13 + 8);
    }
    v15 = *(_QWORD *)v13 + 32LL * v14;
    if ( v15 )
    {
      *(_QWORD *)v15 = 6;
      *(_QWORD *)(v15 + 8) = 0;
      v16 = v63;
      v10 = v63 == 0;
      *(_QWORD *)(v15 + 16) = v63;
      if ( v16 != -8 && !v10 && v16 != -16 )
        sub_1649AC0(v15, v61 & 0xFFFFFFFFFFFFFFF8LL);
      *(_DWORD *)(v15 + 24) = v64;
      v14 = *(_DWORD *)(v13 + 8);
    }
    *(_DWORD *)(v13 + 8) = v14 + 1;
    goto LABEL_46;
  }
  v12 = *(_QWORD *)(a2 - 48);
  v22 = *(_QWORD *)(a2 - 24);
  if ( !v12 )
  {
LABEL_82:
    if ( !v22 )
      return v5;
    v21 = (_QWORD *)v61;
    goto LABEL_57;
  }
  v23 = *(_BYTE *)(v22 + 16);
  v60 = *(_QWORD *)(a2 - 48);
  if ( v23 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) != 16 || v23 > 0x10u )
      goto LABEL_57;
    v56 = v22;
    v5 = sub_15A1020(v22);
    v34 = v56;
    if ( v5 && *(_BYTE *)(v5 + 16) == 13 )
    {
      v35 = *(_DWORD *)(v5 + 32);
      if ( v35 <= 0x40 )
        LOBYTE(v5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v35) == *(_QWORD *)(v5 + 24);
      else
        LOBYTE(v5) = v35 == (unsigned int)sub_16A58F0(v5 + 24);
      if ( (_BYTE)v5 )
        goto LABEL_72;
    }
    else
    {
      v53 = *(_QWORD *)(*(_QWORD *)v56 + 32LL);
      if ( !v53 )
        goto LABEL_72;
      v40 = 0;
      while ( 1 )
      {
        v57 = v34;
        v5 = sub_15A0A60(v34, v40);
        v34 = v57;
        if ( !v5 )
          break;
        v41 = *(_BYTE *)(v5 + 16);
        if ( v41 != 9 )
        {
          if ( v41 != 13 )
            break;
          v42 = *(_DWORD *)(v5 + 32);
          if ( v42 <= 0x40 )
          {
            LOBYTE(v5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v42) == *(_QWORD *)(v5 + 24);
          }
          else
          {
            LODWORD(v5) = sub_16A58F0(v5 + 24);
            v34 = v57;
            LOBYTE(v5) = v42 == (_DWORD)v5;
          }
          if ( !(_BYTE)v5 )
            break;
        }
        if ( v53 == ++v40 )
          goto LABEL_72;
      }
    }
    v22 = *(_QWORD *)(a2 - 24);
    goto LABEL_82;
  }
  v24 = *(_DWORD *)(v22 + 32);
  if ( v24 <= 0x40 )
  {
    v26 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == *(_QWORD *)(v22 + 24);
  }
  else
  {
    v52 = *(_DWORD *)(v22 + 32);
    v55 = v22;
    v25 = sub_16A58F0(v22 + 24);
    v22 = v55;
    v26 = v52 == v25;
  }
  if ( v26 )
    goto LABEL_22;
LABEL_57:
  *v21 = v22;
  v27 = *(_QWORD *)(a2 - 48);
  LOBYTE(v5) = *(_BYTE *)(v27 + 16);
  if ( (_BYTE)v5 == 13 )
  {
    v28 = *(_DWORD *)(v27 + 32);
    if ( v28 <= 0x40 )
      LOBYTE(v5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v28) == *(_QWORD *)(v27 + 24);
    else
      LOBYTE(v5) = v28 == (unsigned int)sub_16A58F0(v27 + 24);
    goto LABEL_71;
  }
  if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 16 || (unsigned __int8)v5 > 0x10u )
    return v5;
  v5 = sub_15A1020(*(_QWORD *)(a2 - 48));
  if ( v5 && *(_BYTE *)(v5 + 16) == 13 )
  {
    v37 = *(_DWORD *)(v5 + 32);
    if ( v37 <= 0x40 )
      LOBYTE(v5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) == *(_QWORD *)(v5 + 24);
    else
      LOBYTE(v5) = v37 == (unsigned int)sub_16A58F0(v5 + 24);
LABEL_71:
    if ( !(_BYTE)v5 )
      return v5;
    goto LABEL_72;
  }
  v43 = 0;
  v58 = *(_QWORD *)(*(_QWORD *)v27 + 32LL);
  if ( !v58 )
    goto LABEL_72;
  while ( 1 )
  {
    v5 = sub_15A0A60(v27, v43);
    if ( !v5 )
      return v5;
    v44 = *(_BYTE *)(v5 + 16);
    if ( v44 != 9 )
    {
      if ( v44 != 13 )
        return v5;
      v45 = *(_DWORD *)(v5 + 32);
      if ( v45 <= 0x40 )
      {
        if ( *(_QWORD *)(v5 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v45) )
          return v5;
      }
      else
      {
        LODWORD(v5) = sub_16A58F0(v5 + 24);
        if ( v45 != (_DWORD)v5 )
          return v5;
      }
    }
    if ( v58 == ++v43 )
      goto LABEL_72;
  }
}
