// Function: sub_171DA10
// Address: 0x171da10
//
bool __fastcall sub_171DA10(_QWORD **a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 v7; // rax
  __int64 *v8; // r12
  unsigned __int8 v9; // al
  unsigned int v10; // r14d
  bool v11; // al
  __int64 v12; // rax
  unsigned int v13; // r12d
  unsigned int *v14; // rbx
  unsigned __int8 v15; // al
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r15
  unsigned int v22; // r12d
  bool v23; // al
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // r12d
  unsigned int v27; // r14d
  __int64 v28; // rax
  char v29; // dl
  unsigned int v30; // r15d
  unsigned int v31; // ebx
  __int64 v32; // rax
  unsigned int v33; // r12d
  int v34; // r14d
  unsigned int v35; // r15d
  __int64 v36; // rax
  unsigned int v37; // esi
  bool v38; // al
  int v39; // r12d
  unsigned int v40; // r14d
  __int64 v41; // rax
  char v42; // dl
  unsigned int v43; // r15d
  int v45; // r12d
  unsigned int v46; // r14d
  __int64 v47; // rax
  unsigned int v48; // esi
  bool v49; // al
  unsigned int v50; // [rsp+Ch] [rbp-34h]
  unsigned int v51; // [rsp+Ch] [rbp-34h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 52 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    if ( v7 )
    {
      **a1 = v7;
      v8 = *(__int64 **)(a2 - 24);
      v9 = *((_BYTE *)v8 + 16);
      if ( v9 == 13 )
      {
        v10 = *((_DWORD *)v8 + 8);
        if ( v10 <= 0x40 )
        {
          a4 = 64 - v10;
          v11 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == v8[3];
        }
        else
        {
          v11 = v10 == (unsigned int)sub_16A58F0((__int64)(v8 + 3));
        }
        if ( v11 )
          return 1;
        goto LABEL_18;
      }
      if ( *(_BYTE *)(*v8 + 8) != 16 || v9 > 0x10u )
      {
LABEL_18:
        **a1 = v8;
        v14 = *(unsigned int **)(v4 - 48);
        v15 = *((_BYTE *)v14 + 16);
        if ( v15 == 13 )
          goto LABEL_19;
        if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 || v15 > 0x10u )
          return 0;
        v25 = sub_15A1020(v14, a2, *(_QWORD *)v14, a4);
        if ( v25 && *(_BYTE *)(v25 + 16) == 13 )
        {
LABEL_45:
          v31 = *(_DWORD *)(v25 + 32);
          if ( v31 <= 0x40 )
            return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v31) == *(_QWORD *)(v25 + 24);
          else
            return v31 == (unsigned int)sub_16A58F0(v25 + 24);
        }
        v39 = *(_QWORD *)(*(_QWORD *)v14 + 32LL);
        if ( v39 )
        {
          v40 = 0;
          while ( 1 )
          {
            v41 = sub_15A0A60((__int64)v14, v40);
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
      v12 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, *v8, a4);
      if ( v12 && *(_BYTE *)(v12 + 16) == 13 )
      {
        v13 = *(_DWORD *)(v12 + 32);
        if ( v13 <= 0x40 )
        {
          a4 = 64 - v13;
          if ( *(_QWORD *)(v12 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) )
            return 1;
        }
        else if ( v13 == (unsigned int)sub_16A58F0(v12 + 24) )
        {
          return 1;
        }
      }
      else
      {
        v34 = *(_QWORD *)(*v8 + 32);
        if ( !v34 )
          return 1;
        v35 = 0;
        while ( 1 )
        {
          a2 = v35;
          v36 = sub_15A0A60((__int64)v8, v35);
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
              v50 = *(_DWORD *)(v36 + 32);
              a2 = v50;
              v38 = v50 == (unsigned int)sub_16A58F0(v36 + 24);
            }
            if ( !v38 )
              break;
          }
          if ( v34 == ++v35 )
            return 1;
        }
      }
    }
    v8 = *(__int64 **)(v4 - 24);
    if ( !v8 )
      return 0;
    goto LABEL_18;
  }
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v17 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v18 = 4 * v17;
  v19 = *(_QWORD *)(a2 - 24 * v17);
  if ( v19 )
  {
    **a1 = v19;
    v20 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v21 = *(_QWORD *)(a2 + 24 * (1 - v20));
    if ( *(_BYTE *)(v21 + 16) == 13 )
    {
      v22 = *(_DWORD *)(v21 + 32);
      if ( v22 <= 0x40 )
      {
        v18 = 64 - v22;
        v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) == *(_QWORD *)(v21 + 24);
      }
      else
      {
        v23 = v22 == (unsigned int)sub_16A58F0(v21 + 24);
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) != 16 )
        goto LABEL_28;
      v32 = sub_15A1020(*(_BYTE **)(a2 + 24 * (1 - v20)), a2, v20, v18);
      if ( !v32 || *(_BYTE *)(v32 + 16) != 13 )
      {
        v45 = *(_QWORD *)(*(_QWORD *)v21 + 32LL);
        if ( v45 )
        {
          v46 = 0;
          while ( 1 )
          {
            a2 = v46;
            v47 = sub_15A0A60(v21, v46);
            if ( !v47 )
              break;
            v18 = *(unsigned __int8 *)(v47 + 16);
            if ( (_BYTE)v18 != 9 )
            {
              if ( (_BYTE)v18 != 13 )
                break;
              v48 = *(_DWORD *)(v47 + 32);
              if ( v48 <= 0x40 )
              {
                v18 = 64 - v48;
                a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v48);
                v49 = a2 == *(_QWORD *)(v47 + 24);
              }
              else
              {
                v51 = *(_DWORD *)(v47 + 32);
                a2 = v51;
                v49 = v51 == (unsigned int)sub_16A58F0(v47 + 24);
              }
              if ( !v49 )
                break;
            }
            if ( v45 == ++v46 )
              return 1;
          }
          goto LABEL_26;
        }
        return 1;
      }
      v33 = *(_DWORD *)(v32 + 32);
      if ( v33 <= 0x40 )
      {
        v18 = 64 - v33;
        v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) == *(_QWORD *)(v32 + 24);
      }
      else
      {
        v23 = v33 == (unsigned int)sub_16A58F0(v32 + 24);
      }
    }
    if ( !v23 )
    {
LABEL_26:
      v17 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
      goto LABEL_27;
    }
    return 1;
  }
LABEL_27:
  v21 = *(_QWORD *)(v4 + 24 * (1 - v17));
  if ( !v21 )
    return 0;
LABEL_28:
  **a1 = v21;
  v24 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
  v14 = *(unsigned int **)(v4 - 24 * v24);
  if ( *((_BYTE *)v14 + 16) != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
    {
      v25 = sub_15A1020(v14, a2, 4 * v24, v18);
      if ( !v25 || *(_BYTE *)(v25 + 16) != 13 )
      {
        v26 = *(_QWORD *)(*(_QWORD *)v14 + 32LL);
        if ( v26 )
        {
          v27 = 0;
          while ( 1 )
          {
            v28 = sub_15A0A60((__int64)v14, v27);
            if ( !v28 )
              break;
            v29 = *(_BYTE *)(v28 + 16);
            if ( v29 != 9 )
            {
              if ( v29 != 13 )
                return 0;
              v30 = *(_DWORD *)(v28 + 32);
              if ( v30 <= 0x40 )
              {
                if ( *(_QWORD *)(v28 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v30) )
                  return 0;
              }
              else if ( v30 != (unsigned int)sub_16A58F0(v28 + 24) )
              {
                return 0;
              }
            }
            if ( v26 == ++v27 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
      goto LABEL_45;
    }
    return 0;
  }
LABEL_19:
  v16 = v14[8];
  if ( v16 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) == *((_QWORD *)v14 + 3);
  else
    return v16 == (unsigned int)sub_16A58F0((__int64)(v14 + 6));
}
