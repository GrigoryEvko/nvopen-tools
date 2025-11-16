// Function: sub_173F330
// Address: 0x173f330
//
bool __fastcall sub_173F330(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 v7; // rax
  char v8; // dl
  _BYTE *v9; // r12
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r12
  char v19; // al
  unsigned int v20; // r14d
  bool v21; // al
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  int v27; // r12d
  unsigned int v28; // r14d
  __int64 v29; // rax
  char v30; // dl
  unsigned int v31; // r15d
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // ebx
  __int64 v36; // rax
  unsigned __int8 v37; // al
  unsigned int v38; // r12d
  __int64 v39; // rax
  unsigned int v40; // r12d
  unsigned int v41; // ebx
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned int v46; // r12d
  int v47; // r14d
  unsigned int v48; // r15d
  __int64 v49; // rax
  unsigned int v50; // esi
  bool v51; // al
  int v52; // r12d
  unsigned int v53; // r14d
  __int64 v54; // rax
  char v55; // dl
  unsigned int v56; // r15d
  int v58; // r14d
  unsigned int v59; // r15d
  __int64 v60; // rax
  unsigned int v61; // esi
  bool v62; // al
  unsigned int v63; // [rsp+Ch] [rbp-34h]
  unsigned int v64; // [rsp+Ch] [rbp-34h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 != 52 )
  {
    if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v14 = *(_QWORD *)(a2 - 24 * v13);
    v15 = *(unsigned __int8 *)(v14 + 16);
    if ( (_BYTE)v15 == 51 )
    {
      v15 = *(_QWORD *)(v14 - 48);
      if ( v15 )
      {
        *(_QWORD *)*a1 = v15;
        v17 = *(_QWORD *)(v14 - 24);
        if ( !v17 )
          goto LABEL_24;
LABEL_20:
        *(_QWORD *)a1[1] = v17;
        v18 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
        v19 = *(_BYTE *)(v18 + 16);
        if ( v19 == 13 )
        {
          v20 = *(_DWORD *)(v18 + 32);
          if ( v20 <= 0x40 )
          {
            v15 = 64 - v20;
            v21 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20) == *(_QWORD *)(v18 + 24);
          }
          else
          {
            v21 = v20 == (unsigned int)sub_16A58F0(v18 + 24);
          }
        }
        else
        {
          if ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 16 )
            goto LABEL_26;
          v45 = sub_15A1020((_BYTE *)v18, a2, *(_QWORD *)v18, v15);
          if ( !v45 || *(_BYTE *)(v45 + 16) != 13 )
          {
            v58 = *(_QWORD *)(*(_QWORD *)v18 + 32LL);
            if ( v58 )
            {
              v59 = 0;
              while ( 1 )
              {
                a2 = v59;
                v60 = sub_15A0A60(v18, v59);
                if ( !v60 )
                  break;
                v15 = *(unsigned __int8 *)(v60 + 16);
                if ( (_BYTE)v15 != 9 )
                {
                  if ( (_BYTE)v15 != 13 )
                    goto LABEL_24;
                  v61 = *(_DWORD *)(v60 + 32);
                  if ( v61 <= 0x40 )
                  {
                    v15 = 64 - v61;
                    a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v61);
                    v62 = a2 == *(_QWORD *)(v60 + 24);
                  }
                  else
                  {
                    v64 = *(_DWORD *)(v60 + 32);
                    a2 = v64;
                    v62 = v64 == (unsigned int)sub_16A58F0(v60 + 24);
                  }
                  if ( !v62 )
                    goto LABEL_24;
                }
                if ( v58 == ++v59 )
                  return 1;
              }
              v18 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
              v19 = *(_BYTE *)(v18 + 16);
              goto LABEL_26;
            }
            return 1;
          }
          v46 = *(_DWORD *)(v45 + 32);
          if ( v46 <= 0x40 )
          {
            v15 = 64 - v46;
            v21 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v46) == *(_QWORD *)(v45 + 24);
          }
          else
          {
            v21 = v46 == (unsigned int)sub_16A58F0(v45 + 24);
          }
        }
        if ( !v21 )
        {
LABEL_24:
          v18 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
          v19 = *(_BYTE *)(v18 + 16);
LABEL_26:
          if ( v19 == 51 )
          {
            v44 = *(_QWORD *)(v18 - 48);
            if ( !v44 )
              return 0;
            *(_QWORD *)*a1 = v44;
            v23 = *(_QWORD *)(v18 - 24);
            if ( !v23 )
              return 0;
          }
          else
          {
            if ( v19 != 5 )
              return 0;
            if ( *(_WORD *)(v18 + 18) != 27 )
              return 0;
            v22 = *(_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
            if ( !v22 )
              return 0;
            *(_QWORD *)*a1 = v22;
            v23 = *(_QWORD *)(v18 + 24 * (1LL - (*(_DWORD *)(v18 + 20) & 0xFFFFFFF)));
            if ( !v23 )
              return 0;
          }
          *(_QWORD *)a1[1] = v23;
          v24 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
          v25 = *(_QWORD *)(v4 - 24 * v24);
          if ( *(_BYTE *)(v25 + 16) != 13 )
          {
            if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) == 16 )
            {
              v26 = sub_15A1020((_BYTE *)v25, a2, 4 * v24, v15);
              if ( !v26 || *(_BYTE *)(v26 + 16) != 13 )
              {
                v27 = *(_QWORD *)(*(_QWORD *)v25 + 32LL);
                if ( v27 )
                {
                  v28 = 0;
                  while ( 1 )
                  {
                    v29 = sub_15A0A60(v25, v28);
                    if ( !v29 )
                      break;
                    v30 = *(_BYTE *)(v29 + 16);
                    if ( v30 != 9 )
                    {
                      if ( v30 != 13 )
                        break;
                      v31 = *(_DWORD *)(v29 + 32);
                      if ( !(v31 <= 0x40
                           ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v31) == *(_QWORD *)(v29 + 24)
                           : v31 == (unsigned int)sub_16A58F0(v29 + 24)) )
                        break;
                    }
                    if ( v27 == ++v28 )
                      return 1;
                  }
                  return 0;
                }
                return 1;
              }
              goto LABEL_66;
            }
            return 0;
          }
          goto LABEL_53;
        }
        return 1;
      }
    }
    else if ( (_BYTE)v15 == 5 && *(_WORD *)(v14 + 18) == 27 )
    {
      v16 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
      a2 = 4 * v16;
      v15 = *(_QWORD *)(v14 - 24 * v16);
      if ( v15 )
      {
        a2 = 1;
        *(_QWORD *)*a1 = v15;
        v15 = 1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        v17 = *(_QWORD *)(v14 + 24 * v15);
        if ( !v17 )
        {
          a2 = 1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
          v18 = *(_QWORD *)(v4 + 24 * a2);
          v19 = *(_BYTE *)(v18 + 16);
          goto LABEL_26;
        }
        goto LABEL_20;
      }
    }
    v18 = *(_QWORD *)(v4 + 24 * (1 - v13));
    v19 = *(_BYTE *)(v18 + 16);
    goto LABEL_26;
  }
  v7 = *(_QWORD *)(a2 - 48);
  v8 = *(_BYTE *)(v7 + 16);
  if ( v8 == 51 )
  {
    v33 = *(_QWORD *)(v7 - 48);
    if ( !v33 )
      goto LABEL_8;
    a4 = *a1;
    *(_QWORD *)*a1 = v33;
    v34 = *(_QWORD *)(v7 - 24);
    if ( !v34 )
      goto LABEL_8;
  }
  else
  {
    if ( v8 != 5 )
      goto LABEL_8;
    if ( *(_WORD *)(v7 + 18) != 27 )
      goto LABEL_8;
    v42 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    a4 = 4 * v42;
    v43 = *(_QWORD *)(v7 - 24 * v42);
    if ( !v43 )
      goto LABEL_8;
    *(_QWORD *)*a1 = v43;
    a4 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    v34 = *(_QWORD *)(v7 + 24 * (1 - a4));
    if ( !v34 )
      goto LABEL_8;
  }
  *(_QWORD *)a1[1] = v34;
  v9 = *(_BYTE **)(a2 - 24);
  v10 = v9[16];
  if ( v10 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 || v10 > 0x10u )
      goto LABEL_9;
    v39 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, *(_QWORD *)v9, a4);
    if ( v39 && *(_BYTE *)(v39 + 16) == 13 )
    {
      v40 = *(_DWORD *)(v39 + 32);
      if ( v40 <= 0x40 )
      {
        a4 = 64 - v40;
        if ( *(_QWORD *)(v39 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v40) )
          return 1;
      }
      else if ( v40 == (unsigned int)sub_16A58F0(v39 + 24) )
      {
        return 1;
      }
    }
    else
    {
      v47 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
      if ( !v47 )
        return 1;
      v48 = 0;
      while ( 1 )
      {
        a2 = v48;
        v49 = sub_15A0A60((__int64)v9, v48);
        if ( !v49 )
          break;
        a4 = *(unsigned __int8 *)(v49 + 16);
        if ( (_BYTE)a4 != 9 )
        {
          if ( (_BYTE)a4 != 13 )
            break;
          v50 = *(_DWORD *)(v49 + 32);
          if ( v50 <= 0x40 )
          {
            a4 = 64 - v50;
            a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v50);
            v51 = a2 == *(_QWORD *)(v49 + 24);
          }
          else
          {
            v63 = *(_DWORD *)(v49 + 32);
            a2 = v63;
            v51 = v63 == (unsigned int)sub_16A58F0(v49 + 24);
          }
          if ( !v51 )
            break;
        }
        if ( v47 == ++v48 )
          return 1;
      }
    }
LABEL_8:
    v9 = *(_BYTE **)(v4 - 24);
    v10 = v9[16];
LABEL_9:
    if ( v10 == 51 )
    {
      v36 = *((_QWORD *)v9 - 6);
      if ( !v36 )
        return 0;
      *(_QWORD *)*a1 = v36;
      v12 = *((_QWORD *)v9 - 3);
      if ( !v12 )
        return 0;
    }
    else
    {
      if ( v10 != 5 )
        return 0;
      if ( *((_WORD *)v9 + 9) != 27 )
        return 0;
      v11 = *(_QWORD *)&v9[-24 * (*((_DWORD *)v9 + 5) & 0xFFFFFFF)];
      if ( !v11 )
        return 0;
      *(_QWORD *)*a1 = v11;
      v12 = *(_QWORD *)&v9[24 * (1LL - (*((_DWORD *)v9 + 5) & 0xFFFFFFF))];
      if ( !v12 )
        return 0;
    }
    *(_QWORD *)a1[1] = v12;
    v25 = *(_QWORD *)(v4 - 48);
    v37 = *(_BYTE *)(v25 + 16);
    if ( v37 == 13 )
    {
LABEL_53:
      v38 = *(_DWORD *)(v25 + 32);
      if ( v38 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) == *(_QWORD *)(v25 + 24);
      else
        return v38 == (unsigned int)sub_16A58F0(v25 + 24);
    }
    if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) != 16 || v37 > 0x10u )
      return 0;
    v26 = sub_15A1020((_BYTE *)v25, a2, *(_QWORD *)v25, a4);
    if ( v26 && *(_BYTE *)(v26 + 16) == 13 )
    {
LABEL_66:
      v41 = *(_DWORD *)(v26 + 32);
      if ( v41 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v41) == *(_QWORD *)(v26 + 24);
      else
        return v41 == (unsigned int)sub_16A58F0(v26 + 24);
    }
    v52 = *(_QWORD *)(*(_QWORD *)v25 + 32LL);
    if ( v52 )
    {
      v53 = 0;
      while ( 1 )
      {
        v54 = sub_15A0A60(v25, v53);
        if ( !v54 )
          break;
        v55 = *(_BYTE *)(v54 + 16);
        if ( v55 != 9 )
        {
          if ( v55 != 13 )
            break;
          v56 = *(_DWORD *)(v54 + 32);
          if ( !(v56 <= 0x40
               ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v56) == *(_QWORD *)(v54 + 24)
               : v56 == (unsigned int)sub_16A58F0(v54 + 24)) )
            break;
        }
        if ( v52 == ++v53 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  v35 = *((_DWORD *)v9 + 8);
  if ( v35 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v35) == *((_QWORD *)v9 + 3);
  else
    return v35 == (unsigned int)sub_16A58F0((__int64)(v9 + 24));
}
