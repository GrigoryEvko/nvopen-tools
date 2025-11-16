// Function: sub_8DD8B0
// Address: 0x8dd8b0
//
__int64 __fastcall sub_8DD8B0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  __int64 v6; // r13
  char v7; // al
  _BOOL4 v8; // r15d
  unsigned int v9; // r14d
  __int64 v11; // rbx
  __int64 v12; // r9
  char v13; // dl
  __int64 v14; // rax
  __int64 v15; // rdx
  bool v16; // si
  unsigned __int64 v17; // rcx
  char i; // dl
  unsigned __int64 v19; // rcx
  int v20; // r14d
  unsigned int v21; // eax
  int v22; // eax
  unsigned __int64 v23; // rsi
  bool v24; // cl
  unsigned __int64 v25; // rcx
  bool v26; // di
  __int64 v27; // rcx
  char v28; // al
  __int64 v29; // rdx
  char v30; // al
  unsigned __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // rdx
  __int64 v34; // rcx
  char v35; // dl
  char v36; // bl
  __int64 v37; // rsi
  __int64 v38; // rsi
  unsigned int v39; // edx
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r15
  unsigned int v44; // ebx
  __int64 v45; // r14
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  char v54; // cl
  char v55; // dl
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // rdi
  char v61; // al
  __int64 v62; // rdx
  _QWORD *v63; // rbx
  _QWORD *v64; // rax
  __int64 v65; // rdx
  _BOOL4 v66; // [rsp+0h] [rbp-60h]
  __int16 v67; // [rsp+4h] [rbp-5Ch]
  _QWORD *v68; // [rsp+8h] [rbp-58h]
  unsigned int v69; // [rsp+10h] [rbp-50h]
  int v70; // [rsp+14h] [rbp-4Ch]
  int v71; // [rsp+18h] [rbp-48h]
  int v72; // [rsp+1Ch] [rbp-44h]
  int v73; // [rsp+1Ch] [rbp-44h]
  __int64 v74; // [rsp+20h] [rbp-40h] BYREF
  __int64 v75[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a3 & 0xFFFF7FFF;
  v75[0] = a1;
  v74 = a2;
  if ( (a3 & 4) != 0 )
    v5 = a3 & 0xFF7F6FFF | 0x801000;
  v67 = a3;
  v68 = (_QWORD *)a4;
  if ( a2 == a1 )
    return 1;
  v6 = a1;
  v69 = a3 & 1;
  v7 = *(_BYTE *)(a1 + 140);
  v72 = a3 & 2;
  if ( v7 == 12 )
  {
    if ( *(_QWORD *)(a1 + 8) )
      goto LABEL_12;
    v31 = *(unsigned __int8 *)(a1 + 184);
    if ( (unsigned __int8)v31 <= 0xCu )
    {
      v32 = 6338;
      if ( _bittest64(&v32, v31) )
        goto LABEL_12;
    }
    if ( *(_BYTE *)(a2 + 140) != 12 )
      goto LABEL_13;
  }
  else if ( *(_BYTE *)(a2 + 140) != 12 )
  {
    v8 = 0;
    if ( (a3 & 1) == 0 )
    {
LABEL_106:
      v35 = *(_BYTE *)(a2 + 140);
      if ( v35 != v7 && (v7 != 9 || v35 != 10) && (v35 != 9 || v7 != 10) )
        return 0;
      if ( dword_4F07588 && *qword_4D03FD0 )
      {
        if ( (unsigned int)sub_8D1330(v75, &v74, (v5 >> 8) & 1) )
          return (unsigned int)sub_8DD8B0(v75[0], v74, v5, v68);
        v6 = v75[0];
        v7 = *(_BYTE *)(v75[0] + 140);
      }
      v71 = v5 & 0x10;
      if ( v72 )
        v5 &= ~2u;
      v36 = v5 & 4;
      if ( (v5 & 4) != 0 )
      {
        v5 &= ~4u;
        v36 = 1;
      }
      switch ( v7 )
      {
        case 0:
          return 0;
        case 1:
        case 19:
        case 20:
        case 21:
          goto LABEL_126;
        case 2:
          v54 = *(_BYTE *)(v6 + 161) & 8;
          if ( dword_4F077C4 == 2 || (v5 & 0x100) != 0 )
          {
            if ( v54 )
              return 0;
            v37 = v74;
            if ( (*(_BYTE *)(v74 + 161) & 8) != 0 )
              return 0;
          }
          else
          {
            v37 = v74;
            if ( v54 && (*(_BYTE *)(v74 + 161) & 8) != 0 && (!dword_4F077C0 || qword_4F077A8 > 0x76BFu) )
              return 0;
          }
          if ( *(_BYTE *)(v6 + 160) != *(_BYTE *)(v37 + 160) )
            return 0;
          v55 = *(_BYTE *)(v37 + 161) ^ *(_BYTE *)(v6 + 161);
          if ( (v55 & 0x40) != 0 || v55 < 0 || ((*(_BYTE *)(v37 + 162) ^ *(_BYTE *)(v6 + 162)) & 7) != 0 )
            return 0;
          if ( !HIDWORD(qword_4F077B4) )
            return 1;
          v9 = 1;
          goto LABEL_130;
        case 3:
        case 4:
        case 5:
          v9 = *(_BYTE *)(v74 + 160) == *(_BYTE *)(v6 + 160);
          goto LABEL_140;
        case 6:
          if ( ((*(_BYTE *)(v74 + 168) ^ *(_BYTE *)(v6 + 168)) & 3) != 0 )
            return 0;
          v60 = *(_QWORD *)(v6 + 160);
          if ( v71 && dword_4F06978 )
          {
            v61 = *(_BYTE *)(v60 + 140);
            if ( v61 == 12 )
            {
              v62 = *(_QWORD *)(v6 + 160);
              do
              {
                v62 = *(_QWORD *)(v62 + 160);
                v61 = *(_BYTE *)(v62 + 140);
              }
              while ( v61 == 12 );
            }
            if ( v61 != 7 )
              v5 |= 0x400000u;
          }
          v9 = sub_8DD8B0(v60, *(_QWORD *)(v74 + 160), v5, v68);
          goto LABEL_140;
        case 7:
          v73 = v5 & 8;
          if ( (v5 & 8) != 0 )
          {
            v73 = 1;
            v5 &= ~8u;
          }
          if ( (v5 & 0x100000) != 0 )
          {
            v66 = 1;
            v5 &= ~0x100000u;
          }
          else
          {
            v66 = dword_4F06978 == 0;
          }
          v43 = *(_QWORD *)(v6 + 168);
          v44 = v5;
          v45 = *(_QWORD *)(v74 + 168);
          if ( (v5 & 0x400) != 0 )
          {
            BYTE1(v44) = BYTE1(v5) & 0xFB;
            v5 = v44 | 2;
          }
          if ( dword_4F0774C )
            v5 |= 0x20000u;
          v70 = v44 & 0x40000;
          if ( (v44 & 0x40000) != 0 )
          {
            v70 = 1;
            v44 &= ~0x40000u;
          }
          if ( dword_4F077C4 != 2 && unk_4F07778 > 201709 )
            v5 |= 2u;
          if ( !(unsigned int)sub_8DD8B0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v74 + 160), v5, v68)
            || !(unsigned int)sub_8DE890(v75[0], v74, v44, v68)
            || ((*(_BYTE *)(v45 + 19) ^ *(_BYTE *)(v43 + 19)) & 0xC0) != 0 )
          {
            return 0;
          }
          if ( (v44 & 0x80u) != 0 )
            goto LABEL_217;
          if ( ((*(_BYTE *)(v45 + 18) ^ *(_BYTE *)(v43 + 18)) & 0x7F) != 0 )
            return 0;
          v49 = *(_QWORD *)(v43 + 40);
          v50 = *(_QWORD *)(v45 + 40);
          if ( v49 )
          {
            if ( !v50
              || !(unsigned int)sub_8DD8B0(v49, v50, v44, v68)
              && (!dword_4D0443C || (v67 & 0x8000) == 0 || !sub_8D5CE0(*(_QWORD *)(v43 + 40), *(_QWORD *)(v45 + 40))) )
            {
              return 0;
            }
          }
          else if ( v50 )
          {
            return 0;
          }
LABEL_217:
          if ( !v66
            && (sub_8DADD0(v75[0], v74, v46, v47, v48)
             || (!v71 || !dword_4F06978 || (v44 & 0x400000) != 0) && sub_8DADD0(v74, v75[0], v51, v52, v53))
            || v70 && ((*(_BYTE *)(v43 + 20) & 2) != 0 || (*(_BYTE *)(v45 + 20) & 2) != 0) && !sub_8D73A0(v75[0], v74) )
          {
            return 0;
          }
          if ( v73 )
          {
            if ( !HIDWORD(qword_4F077B4) )
              return 1;
            if ( (v44 & 0x4000) == 0 || !v68 )
              goto LABEL_149;
            if ( sub_8D7310(v75[0], v74) )
              goto LABEL_227;
          }
          else
          {
            if ( !sub_8D7260((*(_BYTE *)(v43 + 17) >> 4) & 7, (*(_BYTE *)(v45 + 17) >> 4) & 7, v71 != 0) )
              return 0;
            if ( !HIDWORD(qword_4F077B4) )
              return 1;
            if ( sub_8D7310(v75[0], v74) )
            {
LABEL_227:
              if ( !HIDWORD(qword_4F077B4) )
                return 1;
              goto LABEL_149;
            }
            if ( (v44 & 0x4000) == 0 || !v68 )
              return 0;
          }
          v63 = sub_8784C0();
          v64 = sub_8784C0();
          v65 = v75[0];
          *v63 = v64;
          v63[1] = v65;
          v64[1] = v74;
          *v64 = *v68;
          *v68 = v63;
          if ( !HIDWORD(qword_4F077B4) )
            return 1;
LABEL_149:
          v6 = v75[0];
LABEL_127:
          v9 = 1;
LABEL_128:
          v7 = *(_BYTE *)(v6 + 140);
          if ( v7 != 7 )
          {
            v37 = v74;
LABEL_130:
            if ( *(_DWORD *)(v6 + 136) != *(_DWORD *)(v37 + 136)
              && v7 != 8
              && (*(_BYTE *)(v6 + 141) & 0x20) == 0
              && (*(_BYTE *)(v37 + 141) & 0x20) == 0
              && (!dword_4F077BC || !(unsigned int)sub_8DD3B0(v6) && !(unsigned int)sub_8DD3B0(v74))
              && (!v69 || !(unsigned int)sub_8D97B0(v75[0]) && !(unsigned int)sub_8D97B0(v74)) )
            {
              return 0;
            }
          }
          return v9;
        case 8:
          if ( v72 && (dword_4F077C4 == 2 || (v5 & 0x80000) != 0) )
            v5 |= 2u;
          if ( !(unsigned int)sub_8DD8B0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v74 + 160), v5, v68) )
            return 0;
          if ( (unsigned int)sub_8D1590(v75[0], v74)
            || dword_4D047EC && ((*(_BYTE *)(v75[0] + 169) & 2) != 0 || (*(_BYTE *)(v74 + 169) & 2) != 0) )
          {
            goto LABEL_148;
          }
          if ( dword_4F077C4 == 2 && (v36 & 1) == 0 )
          {
            if ( unk_4F07778 <= 202001 && !dword_4F077BC || !v71 )
              return 0;
          }
          else if ( (*(_WORD *)(v75[0] + 168) & 0x180) == 0 && !*(_QWORD *)(v75[0] + 176) )
          {
            goto LABEL_148;
          }
          if ( (*(_WORD *)(v74 + 168) & 0x180) == 0 && !*(_QWORD *)(v74 + 176) )
            goto LABEL_148;
          return 0;
        case 9:
        case 10:
        case 11:
          if ( dword_4F077C4 != 2 )
            return 0;
          if ( (v5 & 0x1000000) != 0 )
          {
            if ( (*(_BYTE *)(v6 + 177) & 0x20) != 0 )
              return 0;
            v38 = v74;
            if ( (*(_BYTE *)(v74 + 177) & 0x20) != 0 )
              return 0;
          }
          else
          {
            v38 = v74;
          }
          if ( !(unsigned int)sub_8DA820(v6, v38, v69, 0, (v5 >> 13) & 1, (v5 >> 12) & 1, 1) )
            return 0;
          goto LABEL_148;
        case 13:
          if ( (v5 & 0x40) != 0 )
          {
            v40 = sub_8D4870(v74);
            v41 = sub_8D4870(v75[0]);
            v42 = v5;
            LOBYTE(v42) = v5 | 0x80;
            if ( !(unsigned int)sub_8DD8B0(v41, v40, v42, v68) )
              return 0;
          }
          else
          {
            v56 = sub_8D4870(v74);
            v57 = sub_8D4870(v75[0]);
            if ( !(unsigned int)sub_8DD8B0(v57, v56, v5, v68) )
              return 0;
            v58 = sub_8D4890(v74);
            v59 = sub_8D4890(v75[0]);
            if ( !(unsigned int)sub_8DD8B0(v59, v58, v5, v68) )
              return 0;
          }
LABEL_148:
          if ( !HIDWORD(qword_4F077B4) )
            return 1;
          goto LABEL_149;
        case 14:
          if ( (v5 & 0x1000000) != 0 )
            return 0;
          v39 = (v5 >> 13) & 8;
          if ( (v5 & 0x1000) != 0 )
            BYTE1(v39) |= 1u;
          if ( (v5 & 0x800000) != 0 )
            BYTE1(v39) |= 0x40u;
          v9 = sub_8D97D0(v6, v74, v39, a4, a5);
LABEL_140:
          if ( !HIDWORD(qword_4F077B4) )
            return v9;
          if ( !v9 )
            return 0;
          v6 = v75[0];
          goto LABEL_128;
        case 15:
          if ( (unsigned int)sub_8D97D0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v74 + 160), v5, a4, a5) )
          {
            v6 = v75[0];
            if ( *(_BYTE *)(v75[0] + 177) == *(_BYTE *)(v74 + 177)
              && *(_QWORD *)(v75[0] + 128) == *(_QWORD *)(v74 + 128) )
            {
              goto LABEL_126;
            }
          }
          return 0;
        case 16:
          if ( !(unsigned int)sub_8D97D0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v74 + 160), v5, a4, a5) )
            return 0;
          v6 = v75[0];
          if ( *(_BYTE *)(v75[0] + 168) != *(_BYTE *)(v74 + 168) )
            return 0;
LABEL_126:
          if ( HIDWORD(qword_4F077B4) )
            goto LABEL_127;
          return 1;
        default:
          sub_721090();
      }
    }
    goto LABEL_7;
  }
  if ( *(_QWORD *)(a2 + 8)
    || (v33 = *(unsigned __int8 *)(a2 + 184), (unsigned __int8)v33 <= 0xCu) && (v34 = 6338, _bittest64(&v34, v33)) )
  {
LABEL_12:
    v5 &= ~0x4000u;
  }
LABEL_13:
  v11 = 6338;
LABEL_14:
  if ( !v72 )
    goto LABEL_58;
  while ( 2 )
  {
    v12 = v74;
    v8 = 0;
    v13 = *(_BYTE *)(v74 + 140);
LABEL_16:
    if ( (v5 & 0x20000) == 0 )
    {
      if ( v7 == 12 )
        goto LABEL_44;
      goto LABEL_20;
    }
    if ( v7 != 12 )
    {
      if ( v13 != 12 )
        goto LABEL_55;
      if ( v7 == 14 )
      {
        if ( !*(_BYTE *)(a1 + 160) )
        {
          v29 = *(_QWORD *)(a1 + 168);
          v9 = 0;
          v14 = a1;
          if ( *(_DWORD *)(v29 + 28) != -1 )
            goto LABEL_23;
          v30 = *(_BYTE *)(v12 + 184);
          v9 = v30 == 2;
          if ( *(_DWORD *)(v29 + 24) == 1 )
            v9 = v30 == 3;
          if ( !v9 )
          {
            v14 = a1;
            goto LABEL_23;
          }
          return v9;
        }
LABEL_21:
        v14 = a1;
        goto LABEL_22;
      }
LABEL_20:
      if ( v13 != 12 )
        goto LABEL_55;
      goto LABEL_21;
    }
    v14 = a1;
    if ( v13 == 14 && !*(_BYTE *)(v12 + 160) )
    {
      v27 = *(_QWORD *)(v12 + 168);
      if ( *(_DWORD *)(v27 + 28) == -1 )
      {
        v28 = *(_BYTE *)(a1 + 184);
        v9 = v28 == 2;
        if ( *(_DWORD *)(v27 + 24) == 1 )
          v9 = v28 == 3;
        if ( v9 )
          return v9;
LABEL_44:
        v14 = a1;
        goto LABEL_45;
      }
    }
    do
    {
LABEL_45:
      if ( (*(_BYTE *)(v14 + 186) & 8) != 0 )
      {
        v19 = *(unsigned __int8 *)(v14 + 184);
        if ( (unsigned __int8)v19 <= 0xAu )
        {
          a5 = ((0x71DuLL >> v19) & 1) == 0;
        }
        else
        {
          if ( (unsigned __int8)v19 > 0xCu )
            goto LABEL_48;
          a5 = 1;
        }
        if ( !_bittest64(&v11, v19) && (_BYTE)a5 )
        {
LABEL_48:
          if ( v13 == 12 )
          {
            v9 = 1;
            goto LABEL_23;
          }
          v15 = v12;
LABEL_64:
          v74 = v15;
          if ( v8 || *(_BYTE *)(v14 + 140) != 12 || (*(_BYTE *)(v14 + 186) & 8) == 0 )
            return 0;
          v23 = *(unsigned __int8 *)(v14 + 184);
          if ( (unsigned __int8)v23 <= 0xAu )
          {
            v24 = ((0x71DuLL >> v23) & 1) == 0;
          }
          else
          {
            v24 = 1;
            if ( (unsigned __int8)v23 > 0xCu )
              goto LABEL_71;
          }
          if ( _bittest64(&v11, v23) || !v24 )
            return 0;
LABEL_71:
          if ( *(_BYTE *)(v15 + 140) != 12 || (*(_BYTE *)(v15 + 186) & 8) == 0 )
            return 0;
          v25 = *(unsigned __int8 *)(v15 + 184);
          if ( (unsigned __int8)v25 <= 0xAu )
          {
            v26 = ((0x71DuLL >> v25) & 1) == 0;
          }
          else
          {
            v26 = 1;
            if ( (unsigned __int8)v25 > 0xCu )
              goto LABEL_77;
          }
          if ( _bittest64(&v11, v25) || !v26 )
            return 0;
LABEL_77:
          if ( (_BYTE)v23 != (_BYTE)v25 )
            return 0;
          a1 = *(_QWORD *)(v14 + 160);
          v12 = *(_QWORD *)(v15 + 160);
          v7 = *(_BYTE *)(a1 + 140);
          v75[0] = a1;
          v74 = v12;
          if ( v7 != 12 )
          {
            i = *(_BYTE *)(v12 + 140);
            if ( i != 12 )
              goto LABEL_38;
          }
          goto LABEL_14;
        }
      }
      v14 = *(_QWORD *)(v14 + 160);
    }
    while ( *(_BYTE *)(v14 + 140) == 12 );
    if ( v13 != 12 )
      goto LABEL_55;
LABEL_22:
    v9 = 0;
LABEL_23:
    v15 = v12;
    a5 = 1821;
    do
    {
      if ( (*(_BYTE *)(v15 + 186) & 8) != 0 )
      {
        v17 = *(unsigned __int8 *)(v15 + 184);
        if ( (unsigned __int8)v17 > 0xAu )
        {
          if ( (unsigned __int8)v17 > 0xCu )
            goto LABEL_64;
          v16 = 1;
        }
        else
        {
          v16 = ((0x71DuLL >> v17) & 1) == 0;
        }
        if ( !_bittest64(&v11, v17) && v16 )
          goto LABEL_64;
      }
      v15 = *(_QWORD *)(v15 + 160);
    }
    while ( *(_BYTE *)(v15 + 140) == 12 );
    if ( v9 )
      goto LABEL_64;
LABEL_55:
    if ( (v5 & 0x800) != 0 )
    {
      if ( (unsigned int)sub_8D46E0(v75, &v74) )
      {
        a1 = v75[0];
        v7 = *(_BYTE *)(v75[0] + 140);
        if ( v72 )
          continue;
LABEL_58:
        v20 = 0;
        if ( (v7 & 0xFB) == 8 )
          v20 = sub_8D4C10(a1, dword_4F077C4 != 2) & 0xFFFFFF8F;
        v12 = v74;
        v21 = 0;
        v13 = *(_BYTE *)(v74 + 140);
        if ( (v13 & 0xFB) == 8 )
        {
          v22 = sub_8D4C10(v74, dword_4F077C4 != 2);
          v12 = v74;
          v21 = v22 & 0xFFFFFF8F;
          v13 = *(_BYTE *)(v74 + 140);
        }
        a1 = v75[0];
        v8 = v20 != v21;
        v7 = *(_BYTE *)(v75[0] + 140);
        goto LABEL_16;
      }
      v12 = v74;
      a1 = v75[0];
    }
    break;
  }
  if ( (v5 & 0x1000) == 0 || dword_4F077C4 != 2 || (a5 = dword_4F07588) == 0 )
  {
LABEL_34:
    for ( i = *(_BYTE *)(v12 + 140); *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
      ;
    for ( v75[0] = a1; i == 12; i = *(_BYTE *)(v12 + 140) )
      v12 = *(_QWORD *)(v12 + 160);
LABEL_38:
    v74 = v12;
    if ( dword_4F077BC )
    {
      if ( *(_BYTE *)(v6 + 140) != i && i == 14 )
      {
        v9 = sub_8DBA70(v6, v12);
        if ( v9 )
          return v9;
      }
    }
    a4 = v69;
    if ( !v69 )
    {
LABEL_41:
      if ( v8 )
        return 0;
      v6 = v75[0];
      a2 = v74;
      v7 = *(_BYTE *)(v75[0] + 140);
      if ( v74 == v75[0] )
        return v7 != 14 || (v5 & 0x1000000) == 0;
      goto LABEL_106;
    }
    v7 = *(_BYTE *)(v75[0] + 140);
LABEL_7:
    if ( !v7 || !*(_BYTE *)(v74 + 140) )
      return 1;
    goto LABEL_41;
  }
  if ( !sub_8D1DD0(a1, v12, 0) )
  {
    a1 = v75[0];
    v12 = v74;
    goto LABEL_34;
  }
  return 0;
}
