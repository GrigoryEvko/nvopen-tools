// Function: sub_1A6DEE0
// Address: 0x1a6dee0
//
bool __fastcall sub_1A6DEE0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rbx
  unsigned __int8 v10; // al
  unsigned int v11; // r12d
  __int64 v12; // rcx
  __int64 v13; // rax
  _BYTE *v14; // r12
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  int v17; // r12d
  unsigned int v18; // r14d
  __int64 v19; // rax
  char v20; // dl
  unsigned int v21; // r15d
  unsigned __int8 v23; // al
  unsigned int v24; // r14d
  bool v25; // al
  unsigned int v26; // ebx
  __int64 v27; // rax
  unsigned int v28; // r12d
  bool v29; // al
  unsigned int v30; // r13d
  bool v31; // al
  __int64 v32; // rax
  unsigned int v33; // r12d
  unsigned int v34; // r15d
  int v35; // r12d
  __int64 v36; // rax
  unsigned int v37; // esi
  bool v38; // al
  int v39; // r12d
  unsigned int v40; // r14d
  __int64 v41; // rax
  char v42; // dl
  unsigned int v43; // r15d
  int v45; // r13d
  unsigned int v46; // r15d
  __int64 v47; // rax
  char v48; // cl
  unsigned int v49; // esi
  bool v50; // al
  unsigned int v51; // [rsp+Ch] [rbp-34h]
  unsigned int v52; // [rsp+Ch] [rbp-34h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 != 52 )
  {
    if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v12 = *a1;
    v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v14 = *(_BYTE **)(a2 + 24 * (1 - v13));
    v15 = 4 * v13;
    if ( *a1 != *(_QWORD *)(v4 - 24 * v13) )
    {
LABEL_11:
      if ( v14 != (_BYTE *)v12 )
        return 0;
      v9 = *(_QWORD *)(v4 - 24 * v13);
      if ( *(_BYTE *)(v9 + 16) != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
        {
          v16 = sub_15A1020((_BYTE *)v9, v15, 4 * v13, v12);
          if ( !v16 || *(_BYTE *)(v16 + 16) != 13 )
          {
            v17 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
            if ( v17 )
            {
              v18 = 0;
              while ( 1 )
              {
                v19 = sub_15A0A60(v9, v18);
                if ( !v19 )
                  break;
                v20 = *(_BYTE *)(v19 + 16);
                if ( v20 != 9 )
                {
                  if ( v20 != 13 )
                    break;
                  v21 = *(_DWORD *)(v19 + 32);
                  if ( !(v21 <= 0x40
                       ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21) == *(_QWORD *)(v19 + 24)
                       : v21 == (unsigned int)sub_16A58F0(v19 + 24)) )
                    break;
                }
                if ( v17 == ++v18 )
                  return 1;
              }
              return 0;
            }
            return 1;
          }
          goto LABEL_35;
        }
        return 0;
      }
      goto LABEL_8;
    }
    if ( v14[16] == 13 )
    {
      v30 = *((_DWORD *)v14 + 8);
      if ( v30 <= 0x40 )
        v31 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v30) == *((_QWORD *)v14 + 3);
      else
        v31 = v30 == (unsigned int)sub_16A58F0((__int64)(v14 + 24));
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
        goto LABEL_11;
      v32 = sub_15A1020(v14, v15, *(_QWORD *)v14, v12);
      if ( !v32 || *(_BYTE *)(v32 + 16) != 13 )
      {
        v45 = *(_QWORD *)(*(_QWORD *)v14 + 32LL);
        if ( v45 )
        {
          v46 = 0;
          while ( 1 )
          {
            v15 = v46;
            v47 = sub_15A0A60((__int64)v14, v46);
            if ( !v47 )
              break;
            v48 = *(_BYTE *)(v47 + 16);
            if ( v48 != 9 )
            {
              if ( v48 != 13 )
                goto LABEL_51;
              v49 = *(_DWORD *)(v47 + 32);
              if ( v49 <= 0x40 )
              {
                v15 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49);
                v50 = v15 == *(_QWORD *)(v47 + 24);
              }
              else
              {
                v52 = *(_DWORD *)(v47 + 32);
                v15 = v52;
                v50 = v52 == (unsigned int)sub_16A58F0(v47 + 24);
              }
              if ( !v50 )
                goto LABEL_51;
            }
            if ( v45 == ++v46 )
              return 1;
          }
          v12 = *a1;
          v13 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
          v14 = *(_BYTE **)(v4 + 24 * (1 - v13));
          goto LABEL_11;
        }
        return 1;
      }
      v33 = *(_DWORD *)(v32 + 32);
      if ( v33 <= 0x40 )
        v31 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) == *(_QWORD *)(v32 + 24);
      else
        v31 = v33 == (unsigned int)sub_16A58F0(v32 + 24);
    }
    if ( !v31 )
    {
LABEL_51:
      v12 = *a1;
      v13 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
      v14 = *(_BYTE **)(v4 + 24 * (1 - v13));
      goto LABEL_11;
    }
    return 1;
  }
  v7 = *a1;
  v8 = *(_QWORD *)(a2 - 24);
  if ( *a1 == *(_QWORD *)(a2 - 48) )
  {
    v23 = *(_BYTE *)(v8 + 16);
    if ( v23 == 13 )
    {
      v24 = *(_DWORD *)(v8 + 32);
      if ( v24 <= 0x40 )
      {
        a4 = 64 - v24;
        v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == *(_QWORD *)(v8 + 24);
      }
      else
      {
        v25 = v24 == (unsigned int)sub_16A58F0(v8 + 24);
      }
      if ( v25 )
        return 1;
    }
    else if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 && v23 <= 0x10u )
    {
      v27 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, *(_QWORD *)v8, a4);
      if ( v27 && *(_BYTE *)(v27 + 16) == 13 )
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
        if ( v29 )
          return 1;
      }
      else
      {
        v34 = 0;
        v35 = *(_QWORD *)(*(_QWORD *)v8 + 32LL);
        if ( !v35 )
          return 1;
        while ( 1 )
        {
          a2 = v34;
          v36 = sub_15A0A60(v8, v34);
          if ( !v36 )
            break;
          a4 = *(unsigned __int8 *)(v36 + 16);
          if ( (_BYTE)a4 != 9 )
          {
            if ( (_BYTE)a4 != 13 )
              break;
            v37 = *(_DWORD *)(v36 + 32);
            if ( v37 <= 0x40 )
            {
              a4 = 64 - v37;
              a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37);
              v38 = a2 == *(_QWORD *)(v36 + 24);
            }
            else
            {
              v51 = *(_DWORD *)(v36 + 32);
              a2 = v51;
              v38 = v51 == (unsigned int)sub_16A58F0(v36 + 24);
            }
            if ( !v38 )
              break;
          }
          if ( v35 == ++v34 )
            return 1;
        }
      }
      v7 = *a1;
      v8 = *(_QWORD *)(v4 - 24);
    }
  }
  if ( v8 != v7 )
    return 0;
  v9 = *(_QWORD *)(v4 - 48);
  v10 = *(_BYTE *)(v9 + 16);
  if ( v10 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 || v10 > 0x10u )
      return 0;
    v16 = sub_15A1020((_BYTE *)v9, a2, *(_QWORD *)v9, a4);
    if ( v16 && *(_BYTE *)(v16 + 16) == 13 )
    {
LABEL_35:
      v26 = *(_DWORD *)(v16 + 32);
      if ( v26 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) == *(_QWORD *)(v16 + 24);
      else
        return v26 == (unsigned int)sub_16A58F0(v16 + 24);
    }
    v39 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
    if ( v39 )
    {
      v40 = 0;
      while ( 1 )
      {
        v41 = sub_15A0A60(v9, v40);
        if ( !v41 )
          break;
        v42 = *(_BYTE *)(v41 + 16);
        if ( v42 != 9 )
        {
          if ( v42 != 13 )
            break;
          v43 = *(_DWORD *)(v41 + 32);
          if ( !(v43 <= 0x40
               ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v43) == *(_QWORD *)(v41 + 24)
               : v43 == (unsigned int)sub_16A58F0(v41 + 24)) )
            break;
        }
        if ( v39 == ++v40 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
LABEL_8:
  v11 = *(_DWORD *)(v9 + 32);
  if ( v11 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *(_QWORD *)(v9 + 24);
  else
    return v11 == (unsigned int)sub_16A58F0(v9 + 24);
}
