// Function: sub_1769F00
// Address: 0x1769f00
//
__int64 __fastcall sub_1769F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  char v5; // al
  _BYTE *v7; // r13
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // r13
  char v14; // al
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // r15d
  __int64 v20; // rax
  unsigned __int8 v21; // al
  __int64 v22; // rax
  unsigned int v23; // r15d
  __int64 v24; // rax
  int v25; // r14d
  unsigned int v26; // r15d
  __int64 v27; // rax
  int v28; // r14d
  unsigned int v29; // r15d
  __int64 v30; // rax
  int v31; // r14d
  int v32; // r14d
  unsigned int v33; // r15d
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // r14d
  __int64 v37; // rax
  int v38; // r14d
  unsigned int v39; // r15d
  __int64 v40; // rax
  char v41; // cl
  __int64 v42; // rax
  int v43; // r14d
  unsigned int v44; // r15d
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned int v47; // r14d
  __int64 v48; // rax
  char v49; // cl
  unsigned int v50; // r15d
  __int64 v51; // rax
  int v52; // eax
  unsigned int v53; // [rsp+8h] [rbp-48h]
  int v54; // [rsp+8h] [rbp-48h]
  __int64 v55; // [rsp+8h] [rbp-48h]
  __int64 v56; // [rsp+10h] [rbp-40h]
  __int64 v57; // [rsp+10h] [rbp-40h]
  unsigned int v58; // [rsp+10h] [rbp-40h]
  int v59; // [rsp+10h] [rbp-40h]
  int v60; // [rsp+18h] [rbp-38h]
  int v61; // [rsp+18h] [rbp-38h]
  int v62; // [rsp+18h] [rbp-38h]
  __int64 v63; // [rsp+18h] [rbp-38h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 != 50 )
  {
    if ( v5 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v12 = 4 * v11;
    v13 = *(_BYTE **)(a2 - 24 * v11);
    v14 = v13[16];
    if ( v14 == 48 )
    {
      if ( (unsigned __int8)sub_1757CC0(*((_BYTE **)v13 - 6), a2, v12, a4) )
        goto LABEL_62;
    }
    else
    {
      if ( v14 != 5 )
      {
LABEL_58:
        if ( v14 != 13 )
        {
LABEL_18:
          if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) != 16 )
          {
LABEL_19:
            v15 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
            v10 = *(_QWORD *)(v4 + 24 * (1 - v15));
            goto LABEL_20;
          }
          v37 = sub_15A1020(v13, a2, v12, a4);
          if ( v37 && *(_BYTE *)(v37 + 16) == 13 )
          {
            if ( !sub_13CFFB0(v37 + 24) )
            {
LABEL_101:
              v15 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
              v10 = *(_QWORD *)(v4 + 24 * (1 - v15));
              goto LABEL_20;
            }
          }
          else
          {
            v43 = *(_QWORD *)(*(_QWORD *)v13 + 32LL);
            if ( v43 )
            {
              v44 = 0;
              while ( 1 )
              {
                a2 = v44;
                v45 = sub_15A0A60((__int64)v13, v44);
                if ( !v45 )
                  goto LABEL_101;
                a4 = *(unsigned __int8 *)(v45 + 16);
                if ( (_BYTE)a4 != 9 )
                {
                  if ( (_BYTE)a4 != 13 )
                    goto LABEL_19;
                  a2 = *(unsigned int *)(v45 + 32);
                  if ( (unsigned int)a2 > 0x40 )
                  {
                    v58 = *(_DWORD *)(v45 + 32);
                    v63 = v45 + 24;
                    v52 = sub_16A58F0(v45 + 24);
                    a2 = v58;
                    if ( !v52 )
                      goto LABEL_101;
                    a2 = v58;
                    a4 = (unsigned int)sub_16A57B0(v63) + v52;
                    if ( v58 != (_DWORD)a4 )
                      goto LABEL_101;
                  }
                  else
                  {
                    v46 = *(_QWORD *)(v45 + 24);
                    if ( !v46 )
                      goto LABEL_101;
                    a4 = v46 + 1;
                    if ( (v46 & (v46 + 1)) != 0 )
                      goto LABEL_101;
                  }
                }
                if ( v43 == ++v44 )
                  goto LABEL_62;
              }
            }
          }
          goto LABEL_62;
        }
        v26 = *((_DWORD *)v13 + 8);
        if ( v26 > 0x40 )
        {
          v31 = sub_16A58F0((__int64)(v13 + 24));
          if ( !v31 || v26 != (unsigned int)sub_16A57B0((__int64)(v13 + 24)) + v31 )
            goto LABEL_19;
        }
        else
        {
          v27 = *((_QWORD *)v13 + 3);
          if ( !v27 || (v27 & (v27 + 1)) != 0 )
            goto LABEL_19;
        }
LABEL_62:
        **(_QWORD **)(a1 + 8) = v13;
        v15 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
        v10 = *(_QWORD *)(v4 + 24 * (1 - v15));
        if ( !v10 )
        {
LABEL_20:
          v16 = *(_BYTE *)(v10 + 16);
          if ( v16 == 48 )
          {
            if ( (unsigned __int8)sub_1757CC0(*(_BYTE **)(v10 - 48), a2, v15, a4) )
              goto LABEL_27;
          }
          else
          {
            if ( v16 != 5 )
              goto LABEL_73;
            if ( *(_WORD *)(v10 + 18) != 24 )
              goto LABEL_23;
            if ( sub_1757E30(
                   *(_BYTE **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)),
                   a2,
                   4LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF),
                   a4) )
            {
              goto LABEL_27;
            }
          }
          v16 = *(_BYTE *)(v10 + 16);
LABEL_73:
          if ( v16 == 13 )
          {
            v29 = *(_DWORD *)(v10 + 32);
            if ( v29 > 0x40 )
            {
              v36 = sub_16A58F0(v10 + 24);
              if ( !v36 || v29 != (unsigned int)sub_16A57B0(v10 + 24) + v36 )
                return 0;
            }
            else
            {
              v30 = *(_QWORD *)(v10 + 24);
              if ( !v30 || (v30 & (v30 + 1)) != 0 )
                return 0;
            }
            goto LABEL_27;
          }
LABEL_23:
          if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
            return 0;
          v17 = sub_15A1020((_BYTE *)v10, a2, v15, a4);
          if ( v17 && *(_BYTE *)(v17 + 16) == 13 )
          {
            if ( sub_13CFFB0(v17 + 24) )
              goto LABEL_27;
            return 0;
          }
          v62 = *(_QWORD *)(*(_QWORD *)v10 + 32LL);
          if ( v62 )
          {
            v47 = 0;
            do
            {
              v48 = sub_15A0A60(v10, v47);
              if ( !v48 )
                return 0;
              v49 = *(_BYTE *)(v48 + 16);
              if ( v49 != 9 )
              {
                if ( v49 != 13 )
                  return 0;
                v50 = *(_DWORD *)(v48 + 32);
                if ( v50 > 0x40 )
                {
                  v55 = v48 + 24;
                  v59 = sub_16A58F0(v48 + 24);
                  if ( !v59 || v50 != (unsigned int)sub_16A57B0(v55) + v59 )
                    return 0;
                }
                else
                {
                  v51 = *(_QWORD *)(v48 + 24);
                  if ( !v51 || (v51 & (v51 + 1)) != 0 )
                    return 0;
                }
              }
            }
            while ( v62 != ++v47 );
          }
LABEL_27:
          **(_QWORD **)(a1 + 8) = v10;
          v18 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
          if ( !v18 )
            return 0;
          goto LABEL_45;
        }
LABEL_14:
        **(_QWORD **)(a1 + 16) = v10;
        return 1;
      }
      if ( *((_WORD *)v13 + 9) != 24 )
        goto LABEL_18;
      if ( sub_1757E30(
             *(_BYTE **)&v13[-24 * (*((_DWORD *)v13 + 5) & 0xFFFFFFF)],
             a2,
             4LL * (*((_DWORD *)v13 + 5) & 0xFFFFFFF),
             a4) )
      {
        goto LABEL_62;
      }
    }
    v14 = v13[16];
    goto LABEL_58;
  }
  v7 = *(_BYTE **)(a2 - 48);
  v8 = v7[16];
  if ( v8 == 48 )
  {
    if ( (unsigned __int8)sub_1757CC0(*((_BYTE **)v7 - 6), a2, a3, a4) )
      goto LABEL_13;
  }
  else
  {
    if ( v8 != 5 )
      goto LABEL_31;
    if ( *((_WORD *)v7 + 9) != 24 )
      goto LABEL_8;
    if ( sub_1757E30(
           *(_BYTE **)&v7[-24 * (*((_DWORD *)v7 + 5) & 0xFFFFFFF)],
           a2,
           4LL * (*((_DWORD *)v7 + 5) & 0xFFFFFFF),
           a4) )
    {
      goto LABEL_13;
    }
  }
  v8 = v7[16];
LABEL_31:
  if ( v8 == 13 )
  {
    v19 = *((_DWORD *)v7 + 8);
    if ( v19 > 0x40 )
    {
      v25 = sub_16A58F0((__int64)(v7 + 24));
      if ( !v25 || v19 != (unsigned int)sub_16A57B0((__int64)(v7 + 24)) + v25 )
        goto LABEL_35;
    }
    else
    {
      v20 = *((_QWORD *)v7 + 3);
      if ( !v20 )
        goto LABEL_35;
      a3 = v20 + 1;
      if ( (v20 & (v20 + 1)) != 0 )
        goto LABEL_35;
    }
    goto LABEL_13;
  }
LABEL_8:
  a3 = *(_QWORD *)v7;
  if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 || v8 > 0x10u )
    goto LABEL_35;
  v9 = sub_15A1020(v7, a2, a3, a4);
  if ( v9 && *(_BYTE *)(v9 + 16) == 13 )
  {
    if ( sub_13CFFB0(v9 + 24) )
      goto LABEL_13;
LABEL_35:
    v10 = *(_QWORD *)(v4 - 24);
    goto LABEL_36;
  }
  v32 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
  if ( v32 )
  {
    v33 = 0;
    do
    {
      a2 = v33;
      v34 = sub_15A0A60((__int64)v7, v33);
      if ( !v34 )
        goto LABEL_35;
      a4 = *(unsigned __int8 *)(v34 + 16);
      if ( (_BYTE)a4 != 9 )
      {
        if ( (_BYTE)a4 != 13 )
          goto LABEL_35;
        a2 = *(unsigned int *)(v34 + 32);
        if ( (unsigned int)a2 > 0x40 )
        {
          v53 = *(_DWORD *)(v34 + 32);
          v56 = v34 + 24;
          v60 = sub_16A58F0(v34 + 24);
          if ( !v60 )
            goto LABEL_35;
          a2 = v53;
          a4 = (unsigned int)sub_16A57B0(v56) + v60;
          if ( v53 != (_DWORD)a4 )
            goto LABEL_35;
        }
        else
        {
          v35 = *(_QWORD *)(v34 + 24);
          if ( !v35 )
            goto LABEL_35;
          a4 = v35 + 1;
          if ( (v35 & (v35 + 1)) != 0 )
            goto LABEL_35;
        }
      }
    }
    while ( v32 != ++v33 );
  }
LABEL_13:
  **(_QWORD **)(a1 + 8) = v7;
  v10 = *(_QWORD *)(v4 - 24);
  if ( v10 )
    goto LABEL_14;
LABEL_36:
  v21 = *(_BYTE *)(v10 + 16);
  if ( v21 == 48 )
  {
    if ( (unsigned __int8)sub_1757CC0(*(_BYTE **)(v10 - 48), a2, a3, a4) )
      goto LABEL_44;
  }
  else
  {
    if ( v21 != 5 )
      goto LABEL_48;
    if ( *(_WORD *)(v10 + 18) != 24 )
      goto LABEL_39;
    if ( sub_1757E30(
           *(_BYTE **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)),
           a2,
           4LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF),
           a4) )
    {
      goto LABEL_44;
    }
  }
  v21 = *(_BYTE *)(v10 + 16);
LABEL_48:
  if ( v21 == 13 )
  {
    v23 = *(_DWORD *)(v10 + 32);
    if ( v23 > 0x40 )
    {
      v28 = sub_16A58F0(v10 + 24);
      if ( !v28 || v23 != (unsigned int)sub_16A57B0(v10 + 24) + v28 )
        return 0;
    }
    else
    {
      v24 = *(_QWORD *)(v10 + 24);
      if ( !v24 || (v24 & (v24 + 1)) != 0 )
        return 0;
    }
    goto LABEL_44;
  }
LABEL_39:
  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 || v21 > 0x10u )
    return 0;
  v22 = sub_15A1020((_BYTE *)v10, a2, *(_QWORD *)v10, a4);
  if ( v22 && *(_BYTE *)(v22 + 16) == 13 )
  {
    if ( !sub_13CFFB0(v22 + 24) )
      return 0;
  }
  else
  {
    v38 = *(_QWORD *)(*(_QWORD *)v10 + 32LL);
    if ( v38 )
    {
      v39 = 0;
      do
      {
        v40 = sub_15A0A60(v10, v39);
        if ( !v40 )
          return 0;
        v41 = *(_BYTE *)(v40 + 16);
        if ( v41 != 9 )
        {
          if ( v41 != 13 )
            return 0;
          if ( *(_DWORD *)(v40 + 32) > 0x40u )
          {
            v54 = *(_DWORD *)(v40 + 32);
            v57 = v40 + 24;
            v61 = sub_16A58F0(v40 + 24);
            if ( !v61 || v54 != (unsigned int)sub_16A57B0(v57) + v61 )
              return 0;
          }
          else
          {
            v42 = *(_QWORD *)(v40 + 24);
            if ( !v42 || (v42 & (v42 + 1)) != 0 )
              return 0;
          }
        }
      }
      while ( v38 != ++v39 );
    }
  }
LABEL_44:
  **(_QWORD **)(a1 + 8) = v10;
  v18 = *(_QWORD *)(v4 - 48);
  if ( !v18 )
    return 0;
LABEL_45:
  **(_QWORD **)(a1 + 16) = v18;
  return 1;
}
