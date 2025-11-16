// Function: sub_668230
// Address: 0x668230
//
char __fastcall sub_668230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, _QWORD *a6, _DWORD *a7)
{
  __int16 v10; // r12
  int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rsi
  bool v15; // cf
  bool v16; // zf
  bool v17; // zf
  unsigned int v18; // eax
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // r13
  int v24; // eax
  _QWORD *v25; // r9
  unsigned __int8 v26; // bl
  _QWORD *v28; // [rsp+0h] [rbp-50h]
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  _QWORD *v30; // [rsp+8h] [rbp-48h]
  _QWORD *v31; // [rsp+8h] [rbp-48h]
  _QWORD *v32; // [rsp+8h] [rbp-48h]
  _QWORD *v33; // [rsp+8h] [rbp-48h]
  _QWORD v34[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = a1;
  v11 = a2;
  if ( (_WORD)a1 == 77 )
  {
    v34[0] = *(_QWORD *)(a3 + 104);
    if ( (a2 & 0x100000) == 0 )
      goto LABEL_3;
LABEL_26:
    LOBYTE(v13) = sub_684B30(1570, v34);
    return v13;
  }
  if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 )
  {
    if ( qword_4D04A00 )
    {
      v15 = *(_QWORD *)(qword_4D04A00 + 16) < 0xDu;
      v16 = *(_QWORD *)(qword_4D04A00 + 16) == 13;
      if ( *(_QWORD *)(qword_4D04A00 + 16) == 13 )
      {
        a2 = *(_QWORD *)(qword_4D04A00 + 8);
        v19 = 13;
        a1 = (__int64)"_Thread_local";
        do
        {
          if ( !v19 )
            break;
          v15 = *(_BYTE *)a2 < *(_BYTE *)a1;
          v16 = *(_BYTE *)a2++ == *(_BYTE *)a1++;
          --v19;
        }
        while ( v16 );
        if ( (!v15 && !v16) == v15 )
        {
          v28 = a6;
          a1 = dword_4F063F8;
          v20 = sub_729F80(dword_4F063F8);
          a6 = v28;
          if ( !v20 )
          {
            sub_684AA0(4 - ((unsigned int)(dword_4D04964 == 0) - 1), 3294, &dword_4F063F8);
            a2 = 1;
            a1 = 3294;
            sub_67D850(3294, 1, 0);
            a6 = v28;
          }
        }
      }
    }
  }
  v29 = a6;
  v34[0] = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, &dword_4F063F8);
  a6 = v29;
  if ( (v11 & 0x100000) != 0 )
    goto LABEL_26;
LABEL_3:
  if ( (v11 & 1) == 0 )
  {
    if ( dword_4F077C4 == 2 && v10 == 103 )
      v12 = 935;
    else
      v12 = 80;
LABEL_7:
    sub_6851C0(v12, v34);
LABEL_8:
    LOBYTE(v13) = (_BYTE)a7;
    *a7 = 1;
    return v13;
  }
  v13 = *a6;
  if ( (*a6 & 0x81) != 0 && (unsigned __int16)(v10 - 193) > 1u )
  {
    v14 = v34;
    if ( v10 == 77 )
    {
      v21 = *(unsigned int *)(a3 + 260);
      v22 = LODWORD(v34[0]);
      if ( (_DWORD)v21 == LODWORD(v34[0]) )
      {
        v21 = *(unsigned __int16 *)(a3 + 264);
        v22 = WORD2(v34[0]);
      }
      v23 = (_QWORD *)(a3 + 260);
      v14 = v34;
      if ( v21 - v22 > 0 )
        v14 = v23;
    }
    sub_6851C0(81, v14);
    *a7 = 1;
    LOBYTE(v13) = (_BYTE)a7;
    return v13;
  }
  if ( (v11 & 8) == 0 || v10 == 95 )
  {
    if ( v10 == 174 )
    {
      if ( (v11 & 4) == 0 || (v13 & 8) != 0 )
      {
        sub_6851C0(719, v34);
        *a7 = 1;
        LOBYTE(v13) = (_BYTE)a7;
      }
      else
      {
        if ( unk_4D04324 )
        {
          v31 = a6;
          sub_684AB0(v34, 881);
          a6 = v31;
          v13 = *v31;
        }
        LOBYTE(v13) = v13 | 0x80;
        *a6 = v13;
        *(_QWORD *)(a3 + 8) |= 0x1000uLL;
        if ( (*(_BYTE *)a6 & 1) == 0 )
        {
          LOBYTE(v13) = v34[0];
          *(_QWORD *)(a3 + 260) = v34[0];
        }
      }
      return v13;
    }
    if ( (unsigned __int16)(v10 - 193) <= 1u )
    {
      if ( (v11 & 0x8000) == 0 )
      {
        if ( (v13 & 8) != 0 )
        {
          sub_6851C0(2501, v34);
          *a7 = 1;
          LOBYTE(v13) = (_BYTE)a7;
        }
        else if ( (v13 & 0x2000) != 0 )
        {
          LOBYTE(v13) = sub_6851C0(240, dword_4F07508);
        }
        else
        {
          BYTE1(v13) |= 0x20u;
          *a6 = v13;
          *(_QWORD *)(a3 + 8) |= 0x400000uLL;
          if ( (*(_BYTE *)a6 & 1) == 0 )
            *(_QWORD *)(a3 + 260) = v34[0];
          LOBYTE(v13) = sub_643E40((__int64)sub_667550, a3, 1);
        }
        return v13;
      }
LABEL_106:
      sub_6851C0(80, v34);
      *a7 = 1;
      LOBYTE(v13) = (_BYTE)a7;
      return v13;
    }
  }
  else if ( dword_4F077C4 != 2 || v10 != 77 )
  {
    if ( v10 == 103 )
    {
      if ( (v11 & 0x1000) != 0 )
      {
        *(_BYTE *)(a3 + 268) = 4;
        *a6 |= 1uLL;
      }
      else
      {
        sub_6851C0(935, v34);
        *a7 = 1;
        LOBYTE(v13) = (_BYTE)a7;
      }
    }
    else
    {
      sub_684AA0(8, 85, v34);
      *a7 = 1;
      LOBYTE(v13) = (_BYTE)a7;
    }
    return v13;
  }
  if ( (v13 & 8) != 0 )
  {
    sub_6851C0(784, v34);
    *a7 = 1;
    LOBYTE(v13) = (_BYTE)a7;
    return v13;
  }
  if ( (v11 & 0x20000) != 0 && v10 != 100 && v10 != 88 )
  {
    v17 = v10 == 103;
    v12 = 935;
    v18 = 80;
    goto LABEL_34;
  }
  if ( (v11 & 4) != 0 && v10 != 100 && v10 != 103 )
  {
    sub_6851C0(328, v34);
    *a7 = 1;
    LOBYTE(v13) = (_BYTE)a7;
    return v13;
  }
  if ( (v11 & 0x4000) != 0 && v10 != 103 && !dword_4F077BC )
    goto LABEL_106;
  if ( (v11 & 0x200) != 0 && v10 != 88 && v10 != 100 )
  {
    v17 = v10 == 103;
    v12 = 935;
    v18 = 481;
LABEL_34:
    if ( !v17 )
      v12 = v18;
    goto LABEL_7;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( (v11 & 0x8000) == 0 )
      goto LABEL_96;
    goto LABEL_110;
  }
  a1 = (unsigned int)dword_4F04C64;
  if ( dword_4F04C64 )
  {
    if ( (v11 & 0x8000) == 0 )
      goto LABEL_81;
LABEL_110:
    if ( v10 != 77 && v10 != 95 )
      goto LABEL_112;
    goto LABEL_116;
  }
  if ( v10 == 77 )
  {
    if ( dword_4F077C0 )
    {
      LOBYTE(v13) = sub_684B30(1159, v34);
      return v13;
    }
    goto LABEL_123;
  }
  if ( v10 != 95 )
  {
    if ( (v11 & 0x8000) == 0 )
    {
LABEL_81:
      if ( unk_4F07778 > 199900 && v10 != 77 && v10 != 95 )
      {
        if ( unk_4D03B90 > 0 && (*(_BYTE *)(unk_4D03B98 + 176LL * unk_4D03B90 + 4) & 0x10) != 0 )
        {
          sub_6851C0(1144, v34);
          *a7 = 1;
          LOBYTE(v13) = (_BYTE)a7;
          return v13;
        }
        if ( dword_4F077C4 == 1 || *a7 | a5 )
          goto LABEL_87;
LABEL_126:
        v33 = a6;
        a1 = 4 - ((unsigned int)(dword_4D04964 == 0) - 1);
        sub_684AA0(a1, 82, v34);
        a6 = v33;
LABEL_97:
        if ( v10 == 95 )
        {
          a1 = LODWORD(v34[0]);
          v30 = a6;
          v24 = sub_729F80(LODWORD(v34[0]));
          v25 = v30;
          if ( !v24 )
          {
            if ( unk_4D04770 )
            {
              sub_6851C0(2798, v34);
              sub_67D850(2798, 1, 0);
              *a7 = 1;
              LOBYTE(v13) = (_BYTE)a7;
              return v13;
            }
            if ( unk_4D04774 )
            {
              a1 = 2799;
              sub_67D850(2799, 1, 0);
              v25 = v30;
            }
          }
          v13 = v34[0];
          *v25 |= 1uLL;
          *(_QWORD *)(a3 + 260) = v13;
          if ( !a4 )
          {
LABEL_103:
            *(_BYTE *)(a3 + 268) = 5;
            return v13;
          }
          goto LABEL_88;
        }
LABEL_87:
        v13 = v34[0];
        *a6 |= 1uLL;
        *(_QWORD *)(a3 + 260) = v13;
        if ( !a4 )
        {
LABEL_89:
          switch ( v10 )
          {
            case 'M':
              *(_BYTE *)(a3 + 268) = 3;
              return v13;
            case 'X':
              *(_BYTE *)(a3 + 268) = 1;
              return v13;
            case '_':
              goto LABEL_103;
            case 'd':
              *(_BYTE *)(a3 + 268) = 2;
              return v13;
            case 'g':
              *(_BYTE *)(a3 + 268) = 4;
              return v13;
            default:
              sub_721090(a1);
          }
        }
LABEL_88:
        *(_QWORD *)(a4 + 8) = v13;
        goto LABEL_89;
      }
      goto LABEL_95;
    }
LABEL_112:
    if ( v10 == 103 )
      sub_684AA0(8, 935, v34);
    else
      sub_684AA0(8, 80, v34);
    goto LABEL_8;
  }
  if ( !dword_4F077C0 )
  {
LABEL_123:
    sub_6851C0(149, v34);
    *a7 = 1;
    LOBYTE(v13) = (_BYTE)a7;
    return v13;
  }
  if ( (v11 & 0x8000) == 0 )
  {
LABEL_95:
    if ( dword_4F077C4 == 1 )
      goto LABEL_97;
LABEL_96:
    if ( *a7 | a5 )
      goto LABEL_97;
    goto LABEL_126;
  }
LABEL_116:
  if ( dword_4D04964 )
  {
    v26 = byte_4F07472[0];
    if ( byte_4F07472[0] != 3 )
    {
      v32 = a6;
      sub_684AA0(byte_4F07472[0], 80, v34);
      if ( v26 > 5u )
        goto LABEL_8;
      a6 = v32;
      v13 = *v32;
    }
  }
  v13 |= 1uLL;
  *a6 = v13;
  return v13;
}
