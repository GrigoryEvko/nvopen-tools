// Function: sub_14CC460
// Address: 0x14cc460
//
char __fastcall sub_14CC460(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r13
  unsigned __int8 v12; // al
  unsigned int v13; // r14d
  __int64 v14; // rax
  unsigned int v15; // r13d
  bool v16; // al
  __int64 v17; // r15
  unsigned __int8 v18; // al
  unsigned int v19; // r13d
  char v20; // al
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned int v23; // r13d
  bool v24; // al
  __int64 v25; // rax
  unsigned int v26; // r13d
  __int64 v27; // rax
  unsigned int v28; // r13d
  int v29; // r14d
  unsigned int v30; // r15d
  __int64 v31; // rax
  char v32; // cl
  unsigned int v33; // esi
  bool v34; // al
  int v35; // r13d
  unsigned int v36; // r14d
  __int64 v37; // rax
  char v38; // cl
  unsigned int v39; // esi
  bool v40; // al
  int v41; // r13d
  unsigned int v42; // r14d
  __int64 v43; // rax
  char v44; // cl
  unsigned int v45; // esi
  bool v46; // al
  int v48; // [rsp+Ch] [rbp-54h]
  int v49; // [rsp+Ch] [rbp-54h]
  int v50; // [rsp+Ch] [rbp-54h]
  __int64 v51; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v52; // [rsp+20h] [rbp-40h] BYREF

  v2 = &v51;
  v3 = a2;
  LODWORD(v4) = *(unsigned __int8 *)(a2 + 16);
  v52 = &v51;
  if ( (_BYTE)v4 != 52 )
  {
    if ( (_BYTE)v4 != 5 )
    {
      if ( (unsigned __int8)v4 <= 0x17u )
        return v4;
      goto LABEL_9;
    }
    v5 = *(unsigned __int16 *)(a2 + 18);
    if ( (_WORD)v5 != 28 )
      goto LABEL_4;
    v21 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( *(_QWORD *)(a2 - 24 * v21) )
    {
      v51 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v22 = *(_QWORD *)(a2 + 24 * (1 - v21));
      if ( *(_BYTE *)(v22 + 16) == 13 )
      {
        v23 = *(_DWORD *)(v22 + 32);
        if ( v23 <= 0x40 )
          v24 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23) == *(_QWORD *)(v22 + 24);
        else
          v24 = v23 == (unsigned int)sub_16A58F0(v22 + 24);
        goto LABEL_57;
      }
      if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) == 16 )
      {
        v27 = sub_15A1020(v22);
        if ( v27 && *(_BYTE *)(v27 + 16) == 13 )
        {
          v28 = *(_DWORD *)(v27 + 32);
          if ( v28 <= 0x40 )
            v24 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v28) == *(_QWORD *)(v27 + 24);
          else
            v24 = v28 == (unsigned int)sub_16A58F0(v27 + 24);
LABEL_57:
          if ( v24 )
            goto LABEL_27;
          goto LABEL_58;
        }
        v41 = *(_QWORD *)(*(_QWORD *)v22 + 32LL);
        if ( !v41 )
          goto LABEL_27;
        v42 = 0;
        while ( 1 )
        {
          v43 = sub_15A0A60(v22, v42);
          if ( !v43 )
            break;
          v44 = *(_BYTE *)(v43 + 16);
          if ( v44 != 9 )
          {
            if ( v44 != 13 )
              break;
            v45 = *(_DWORD *)(v43 + 32);
            if ( v45 <= 0x40 )
            {
              v46 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v45) == *(_QWORD *)(v43 + 24);
            }
            else
            {
              v50 = *(_DWORD *)(v43 + 32);
              v46 = v50 == (unsigned int)sub_16A58F0(v43 + 24);
            }
            if ( !v46 )
              break;
          }
          if ( v41 == ++v42 )
            goto LABEL_27;
        }
      }
    }
LABEL_58:
    v20 = sub_14CA6E0(&v52, v3);
    goto LABEL_45;
  }
  v11 = *(_QWORD *)(a2 - 24);
  if ( !*(_QWORD *)(a2 - 48) )
    goto LABEL_34;
  v51 = *(_QWORD *)(a2 - 48);
  v12 = *(_BYTE *)(v11 + 16);
  if ( v12 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 || v12 > 0x10u )
      goto LABEL_42;
    v14 = sub_15A1020(v11);
    if ( v14 && *(_BYTE *)(v14 + 16) == 13 )
    {
      v15 = *(_DWORD *)(v14 + 32);
      if ( v15 <= 0x40 )
        v16 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == *(_QWORD *)(v14 + 24);
      else
        v16 = v15 == (unsigned int)sub_16A58F0(v14 + 24);
      if ( v16 )
        goto LABEL_27;
    }
    else
    {
      v29 = *(_QWORD *)(*(_QWORD *)v11 + 32LL);
      if ( !v29 )
        goto LABEL_27;
      v30 = 0;
      while ( 1 )
      {
        v31 = sub_15A0A60(v11, v30);
        if ( !v31 )
          break;
        v32 = *(_BYTE *)(v31 + 16);
        if ( v32 != 9 )
        {
          if ( v32 != 13 )
            break;
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
            break;
        }
        if ( v29 == ++v30 )
          goto LABEL_27;
      }
    }
    v11 = *(_QWORD *)(v3 - 24);
    goto LABEL_34;
  }
  v13 = *(_DWORD *)(v11 + 32);
  if ( v13 > 0x40 )
  {
    if ( v13 == (unsigned int)sub_16A58F0(v11 + 24) )
      goto LABEL_27;
LABEL_34:
    if ( !v11 )
      goto LABEL_35;
    v2 = v52;
    goto LABEL_42;
  }
  if ( *(_QWORD *)(v11 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) )
    goto LABEL_27;
LABEL_42:
  *v2 = v11;
  v17 = *(_QWORD *)(v3 - 48);
  v18 = *(_BYTE *)(v17 + 16);
  if ( v18 == 13 )
  {
    v19 = *(_DWORD *)(v17 + 32);
    if ( v19 <= 0x40 )
      v20 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v19) == *(_QWORD *)(v17 + 24);
    else
      v20 = v19 == (unsigned int)sub_16A58F0(v17 + 24);
LABEL_45:
    if ( !v20 )
      goto LABEL_35;
LABEL_27:
    sub_14CBC40(*a1, v51, -1);
    v3 = v51;
    goto LABEL_35;
  }
  if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) == 16 && v18 <= 0x10u )
  {
    v25 = sub_15A1020(*(_QWORD *)(v3 - 48));
    if ( v25 && *(_BYTE *)(v25 + 16) == 13 )
    {
      v26 = *(_DWORD *)(v25 + 32);
      if ( v26 <= 0x40 )
      {
        if ( *(_QWORD *)(v25 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) )
          goto LABEL_27;
      }
      else if ( v26 == (unsigned int)sub_16A58F0(v25 + 24) )
      {
        goto LABEL_27;
      }
    }
    else
    {
      v35 = *(_QWORD *)(*(_QWORD *)v17 + 32LL);
      if ( !v35 )
        goto LABEL_27;
      v36 = 0;
      while ( 1 )
      {
        v37 = sub_15A0A60(v17, v36);
        if ( !v37 )
          break;
        v38 = *(_BYTE *)(v37 + 16);
        if ( v38 != 9 )
        {
          if ( v38 != 13 )
            break;
          v39 = *(_DWORD *)(v37 + 32);
          if ( v39 <= 0x40 )
          {
            v40 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39) == *(_QWORD *)(v37 + 24);
          }
          else
          {
            v49 = *(_DWORD *)(v37 + 32);
            v40 = v49 == (unsigned int)sub_16A58F0(v37 + 24);
          }
          if ( !v40 )
            break;
        }
        if ( v35 == ++v36 )
          goto LABEL_27;
      }
    }
  }
LABEL_35:
  LODWORD(v4) = *(unsigned __int8 *)(v3 + 16);
  if ( (unsigned __int8)v4 > 0x17u )
  {
LABEL_9:
    if ( (unsigned int)(v4 - 50) <= 2 )
    {
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
      {
        v4 = **(_QWORD **)(v3 - 8);
        if ( !v4 )
          return v4;
        v51 = **(_QWORD **)(v3 - 8);
        v8 = *(__int64 **)(v3 - 8);
      }
      else
      {
        v8 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
        v4 = *v8;
        if ( !*v8 )
          return v4;
        v51 = *v8;
      }
      v7 = v8[3];
      if ( !v7 )
        return v4;
      v6 = v51;
      goto LABEL_15;
    }
    LODWORD(v4) = v4 - 47;
    if ( (unsigned int)v4 > 2 )
      return v4;
    if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    {
      v4 = **(_QWORD **)(v3 - 8);
      if ( !v4 )
        return v4;
      v51 = **(_QWORD **)(v3 - 8);
      v9 = *(__int64 **)(v3 - 8);
    }
    else
    {
      v9 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
      v4 = *v9;
      if ( !*v9 )
        return v4;
      v51 = *v9;
    }
    v4 = v9[3];
    if ( *(_BYTE *)(v4 + 16) != 13 )
      return v4;
    v10 = v51;
LABEL_22:
    LOBYTE(v4) = sub_14CBC40(*a1, v10, -1);
    return v4;
  }
  if ( (_BYTE)v4 != 5 )
    return v4;
  v5 = *(unsigned __int16 *)(v3 + 18);
LABEL_4:
  if ( (unsigned int)(v5 - 26) <= 2 )
  {
    v4 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
    v6 = *(_QWORD *)(v3 - 24 * v4);
    if ( !v6 )
      return v4;
    v51 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
    v4 = 3 * (1 - v4);
    v7 = *(_QWORD *)(v3 + 8 * v4);
    if ( !v7 )
      return v4;
LABEL_15:
    sub_14CBC40(*a1, v6, -1);
    LOBYTE(v4) = sub_14CBC40(*a1, v7, -1);
    return v4;
  }
  LODWORD(v4) = v5 - 23;
  if ( (unsigned int)v4 <= 2 )
  {
    v4 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
    v10 = *(_QWORD *)(v3 - 24 * v4);
    if ( v10 )
    {
      v51 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
      v4 = *(_QWORD *)(v3 + 24 * (1 - v4));
      if ( *(_BYTE *)(v4 + 16) == 13 )
        goto LABEL_22;
    }
  }
  return v4;
}
