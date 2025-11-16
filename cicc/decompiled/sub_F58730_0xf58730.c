// Function: sub_F58730
// Address: 0xf58730
//
char __fastcall sub_F58730(unsigned __int8 *a1, unsigned int a2)
{
  char v3; // dl
  unsigned __int8 *v4; // rcx
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  int v8; // eax
  int v9; // edi
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rbx
  unsigned __int8 *v15; // r13
  unsigned __int8 **v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int8 **v19; // r12
  __int64 v20; // r15
  signed __int64 v21; // rax
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int8 v25; // al
  unsigned __int64 v26; // rcx
  bool v27; // zf
  unsigned __int8 **v28; // r14
  __int64 v29; // rdx
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rax
  unsigned __int8 v32; // dl
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r14
  int v37; // r14d
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  char v41; // cl
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax

  v3 = a1[7];
  if ( (v3 & 0x40) != 0 )
    v4 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v4 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v5 = a2;
  v6 = 4LL * a2;
  v7 = *(_QWORD *)&v4[v6 * 8];
  if ( *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) == 9 )
  {
LABEL_11:
    LOBYTE(v10) = 0;
    return v10;
  }
  if ( *(_BYTE *)v7 > 0x15u )
    goto LABEL_7;
  v8 = *a1;
  v9 = v8 - 29;
  switch ( v8 )
  {
    case ' ':
    case ']':
      LOBYTE(v10) = a2 == 0;
      return v10;
    case '"':
    case 'U':
      if ( **((_BYTE **)a1 - 4) == 25 )
        goto LABEL_11;
      if ( v3 >= 0 )
        goto LABEL_20;
      v11 = sub_BD2BC0((__int64)a1);
      v13 = v11 + v12;
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( !(unsigned int)(v13 >> 4) )
          goto LABEL_19;
        goto LABEL_85;
      }
      if ( !(unsigned int)((v13 - sub_BD2BC0((__int64)a1)) >> 4) )
        goto LABEL_19;
      if ( (a1[7] & 0x80u) == 0 )
LABEL_85:
        BUG();
      if ( a2 >= *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8) )
      {
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v43 = sub_BD2BC0((__int64)a1);
        if ( a2 < *(_DWORD *)(v43 + v44 - 4) )
        {
          LOBYTE(v10) = 0;
          return v10;
        }
      }
LABEL_19:
      v9 = *a1 - 29;
LABEL_20:
      if ( v9 == 11 )
      {
        v14 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
      }
      else
      {
        v14 = 0;
        if ( v9 != 56 )
        {
          v14 = 64;
          if ( v9 != 5 )
            BUG();
        }
      }
      if ( (a1[7] & 0x80u) == 0 )
        goto LABEL_73;
      v34 = sub_BD2BC0((__int64)a1);
      v36 = v34 + v35;
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v36 >> 4) )
LABEL_95:
          BUG();
LABEL_73:
        v40 = 0;
        goto LABEL_67;
      }
      if ( !(unsigned int)((v36 - sub_BD2BC0((__int64)a1)) >> 4) )
        goto LABEL_73;
      if ( (a1[7] & 0x80u) == 0 )
        goto LABEL_95;
      v37 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v38 = sub_BD2BC0((__int64)a1);
      v40 = 32LL * (unsigned int)(*(_DWORD *)(v38 + v39 - 4) - v37);
LABEL_67:
      v41 = *a1;
      if ( a2 < (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v14 - v40) >> 5) )
      {
        if ( v41 == 85 )
        {
          v45 = *((_QWORD *)a1 - 4);
          if ( v45 )
          {
            if ( !*(_BYTE *)v45 )
            {
              v46 = *((_QWORD *)a1 + 10);
              if ( *(_QWORD *)(v45 + 24) == v46 && (*(_BYTE *)(v45 + 33) & 0x20) != 0 && a2 >= *(_DWORD *)(v46 + 12) - 1 )
              {
                LOBYTE(v10) = (unsigned int)sub_B49240((__int64)a1) == 158;
                return v10;
              }
            }
          }
        }
        if ( (unsigned int)sub_B49240((__int64)a1) != 183 )
          return (unsigned int)sub_B49B80((__int64)a1, a2, 14) ^ 1;
        goto LABEL_11;
      }
      LOBYTE(v10) = 1;
      if ( v41 == 85 )
      {
        v42 = *((_QWORD *)a1 - 4);
        if ( v42 )
        {
          if ( !*(_BYTE *)v42 && *(_QWORD *)(v42 + 24) == *((_QWORD *)a1 + 10) )
            LOBYTE(v10) = ((*(_BYTE *)(v42 + 33) >> 5) ^ 1) & 1;
        }
      }
      return v10;
    case '<':
      return !sub_B4D040((__int64)a1);
    case '?':
      if ( !a2 )
        goto LABEL_7;
      if ( (v3 & 0x40) != 0 )
        v15 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v15 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v16 = (unsigned __int8 **)(v15 + 32);
      v17 = sub_BB5290((__int64)a1);
      v19 = v16;
      v20 = v17 & 0xFFFFFFFFFFFFFFF9LL | 4;
      v21 = v20;
      break;
    case '\\':
      LOBYTE(v10) = a2 != 2;
      return v10;
    case '^':
      LOBYTE(v10) = a2 <= 1;
      return v10;
    default:
      goto LABEL_7;
  }
  while ( 1 )
  {
    v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    v23 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v21 )
      goto LABEL_35;
    v24 = (v21 >> 1) & 3;
    if ( v24 != 2 )
      break;
    if ( !v22 )
      goto LABEL_35;
LABEL_32:
    v25 = *(_BYTE *)(v23 + 8);
    if ( v25 == 16 )
    {
      v21 = *(_QWORD *)(v23 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else if ( (unsigned int)v25 - 17 > 1 )
    {
      v26 = v23 & 0xFFFFFFFFFFFFFFF9LL;
      v27 = v25 == 15;
      v21 = 0;
      if ( v27 )
        v21 = v26;
    }
    else
    {
      v21 = v23 & 0xFFFFFFFFFFFFFFF9LL | 2;
    }
    v19 += 4;
    if ( !--v5 )
    {
      v28 = &v16[v6];
      if ( v28 != v16 )
      {
        while ( 1 )
        {
          v29 = (v20 >> 1) & 3;
          if ( ((v20 >> 1) & 3) == 0 )
            goto LABEL_11;
          v30 = v20 & 0xFFFFFFFFFFFFFFF8LL;
          v31 = v20 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v20 )
            goto LABEL_53;
          if ( v29 != 2 )
            break;
          if ( !v30 )
            goto LABEL_53;
LABEL_50:
          v32 = *(_BYTE *)(v31 + 8);
          if ( v32 == 16 )
          {
            v20 = *(_QWORD *)(v31 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
          }
          else
          {
            v33 = v31 & 0xFFFFFFFFFFFFFFF9LL;
            if ( (unsigned int)v32 - 17 > 1 )
            {
              if ( v32 != 15 )
                v33 = 0;
              v20 = v33;
            }
            else
            {
              v20 = v33 | 2;
            }
          }
          v16 += 4;
          if ( v28 == v16 )
            goto LABEL_7;
        }
        if ( (_DWORD)v29 == 1 && v30 )
        {
          v31 = *(_QWORD *)(v30 + 24);
          goto LABEL_50;
        }
LABEL_53:
        v31 = sub_BCBAE0(v30, *v16, v29);
        goto LABEL_50;
      }
LABEL_7:
      LOBYTE(v10) = 1;
      return v10;
    }
  }
  if ( v24 == 1 && v22 )
  {
    v23 = *(_QWORD *)(v22 + 24);
    goto LABEL_32;
  }
LABEL_35:
  v23 = sub_BCBAE0(v22, *v19, v18);
  goto LABEL_32;
}
