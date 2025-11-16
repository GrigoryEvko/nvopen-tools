// Function: sub_7D5120
// Address: 0x7d5120
//
__int64 __fastcall sub_7D5120(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5)
{
  __int64 v6; // r14
  int v7; // ecx
  unsigned int v8; // r12d
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r15
  int v12; // ebx
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // eax
  int v16; // edi
  int v17; // eax
  __int64 v18; // r10
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // r9d
  __int64 v23; // r10
  _BOOL4 v24; // eax
  __int64 v25; // rdi
  int v26; // ecx
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // rbx
  unsigned int v30; // r13d
  signed int v31; // r12d
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned int v41; // ebx
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 i; // rax
  __int64 j; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  char v50; // al
  __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int8 v56; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+10h] [rbp-60h]
  unsigned int v59; // [rsp+20h] [rbp-50h]
  __int64 v60; // [rsp+20h] [rbp-50h]
  __int64 v61; // [rsp+20h] [rbp-50h]
  __int64 v62; // [rsp+20h] [rbp-50h]
  __int64 v63; // [rsp+20h] [rbp-50h]
  __int64 v64; // [rsp+28h] [rbp-48h]
  __int64 v65[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a2;
  v64 = a1;
  while ( 2 )
  {
    v59 = dword_4F077BC;
    if ( dword_4F077BC )
      v59 = *(_DWORD *)(v6 + 100) != 0;
    if ( (int)a3 <= a4 )
      return 0;
    v7 = 0;
    v8 = a3;
    while ( 1 )
    {
      v9 = 776LL * (int)v8;
      v10 = qword_4F04C68[0];
      v11 = qword_4F04C68[0] + v9;
      v12 = *(unsigned __int8 *)(qword_4F04C68[0] + v9 + 4);
      if ( v7 )
      {
        if ( *(_DWORD *)(v6 + 52) )
        {
          a5 = *(unsigned int *)(v6 + 36);
          if ( (_DWORD)a5 )
          {
            if ( (_BYTE)v12 == 9 )
              goto LABEL_39;
          }
        }
        if ( (*(_BYTE *)(v11 + 11) & 0x10) != 0 )
          goto LABEL_39;
        *(_DWORD *)(v6 + 52) = 0;
      }
      else if ( (*(_BYTE *)(v11 + 11) & 0x10) != 0 )
      {
        goto LABEL_39;
      }
      v13 = *(unsigned int *)(v6 + 24);
      *(_DWORD *)(v6 + 76) = 0;
      if ( !(_DWORD)v13 )
      {
        a1 = *(unsigned int *)(v6 + 32);
        if ( !(_DWORD)a1 )
        {
          a2 = *(unsigned int *)(v6 + 44);
          if ( !(_DWORD)a2 )
          {
            v14 = v59;
            if ( v59 )
              goto LABEL_17;
            v14 = (unsigned int)(v12 - 6);
            if ( (unsigned __int8)(v12 - 6) > 1u )
              goto LABEL_17;
          }
        }
      }
      *(_DWORD *)(v6 + 136) = 0;
      v14 = unk_4F04C48;
      if ( unk_4F04C48 != -1 )
      {
        a2 = 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v10 + a2 + 6) & 6) == 0 )
        {
          v14 = 776LL * unk_4F04C48;
          v20 = *(_QWORD *)(v10 + v14 + 360);
          if ( v20 )
          {
            if ( *(_BYTE *)(v20 + 80) == 3 )
LABEL_17:
              *(_DWORD *)(v6 + 136) = 1;
          }
        }
      }
      v15 = *(_DWORD *)(v11 + 424);
      if ( v15 )
        *(_DWORD *)(v6 + 128) = v15;
      if ( (unsigned __int8)(v12 - 4) <= 1u )
      {
        v21 = *(_QWORD *)(v11 + 24);
        v14 = v11 + 32;
        if ( !v21 )
          v21 = v11 + 32;
        if ( (*(_BYTE *)(v21 + 144) & 1) == 0 )
        {
          LOBYTE(v12) = 3;
          if ( (*(_BYTE *)(v11 + 10) & 4) != 0 )
            goto LABEL_58;
          LOBYTE(a5) = 3;
          goto LABEL_143;
        }
        if ( (_BYTE)v12 == 4 )
        {
          v16 = 4;
LABEL_25:
          a2 = v8;
          a1 = sub_7D1970(v16, v8, (__int64 *)v64, (_DWORD *)v6);
          if ( v8 != -1 )
          {
LABEL_26:
            v11 = v9 + qword_4F04C68[0];
            goto LABEL_27;
          }
          goto LABEL_135;
        }
        if ( (_BYTE)v12 == 5 )
          goto LABEL_127;
      }
      else
      {
        if ( (_BYTE)v12 == 7 )
        {
          if ( !(_DWORD)v13 || *(_DWORD *)(v6 + 28) )
          {
            v16 = 7;
            goto LABEL_25;
          }
          goto LABEL_48;
        }
        if ( (((_BYTE)v12 - 5) & 0xFB) == 0 )
        {
LABEL_127:
          a2 = v8;
          v47 = sub_7D1970((unsigned __int8)v12, v8, (__int64 *)v64, (_DWORD *)v6);
          a1 = v47;
          if ( v8 == -1 )
          {
            if ( v47 || (_BYTE)v12 != 9 )
            {
LABEL_135:
              v11 = 0;
              goto LABEL_27;
            }
            v11 = 0;
            a2 = 0xFFFFFFFFLL;
            a1 = sub_7D1590(9, -1, (__int64 *)v64, v6);
          }
          else
          {
            if ( v47 || (_BYTE)v12 != 9 )
              goto LABEL_26;
            a2 = v8;
            LOBYTE(v12) = 9;
            a1 = sub_7D1590(9, v8, (__int64 *)v64, v6);
            v11 = v9 + qword_4F04C68[0];
          }
LABEL_27:
          if ( !*(_DWORD *)(v6 + 76) )
            goto LABEL_28;
          goto LABEL_62;
        }
        if ( !(_BYTE)v12 )
        {
          LOBYTE(a5) = *(_BYTE *)(v11 + 10) & 4;
          if ( (_BYTE)a5 )
          {
            v16 = 0;
            goto LABEL_25;
          }
          goto LABEL_143;
        }
        if ( (_BYTE)v12 == 13 )
          goto LABEL_170;
      }
      if ( (_BYTE)v12 != 6 )
      {
        if ( (*(_BYTE *)(v11 + 10) & 4) != 0 )
        {
          if ( (_BYTE)v12 == 17 || (_BYTE)v12 == 2 )
          {
LABEL_60:
            a2 = v8;
            v11 = 0;
            a1 = sub_7D1970((unsigned __int8)v12, v8, (__int64 *)v64, (_DWORD *)v6);
            if ( v8 == -1 )
            {
              if ( !*(_DWORD *)(v6 + 76) )
                goto LABEL_28;
LABEL_62:
              v22 = *(_DWORD *)(v6 + 36);
              v23 = *(_QWORD *)(v6 + 112);
              v24 = 1;
              v65[0] = 0;
              v25 = *(_QWORD *)(v11 + 208);
              if ( !v22 && !*(_DWORD *)(v6 + 40) )
                v24 = (*(_BYTE *)(v25 + 177) & 0x20) != 0;
              v26 = 1;
              if ( !*(_DWORD *)(v6 + 80) )
                v26 = *(_BYTE *)(v11 + 14) & 1;
              if ( (unsigned int)sub_886B00(
                                   v25,
                                   v64,
                                   *(_DWORD *)(v6 + 120),
                                   v26,
                                   *(_DWORD *)(v6 + 84),
                                   *(_DWORD *)(v6 + 16),
                                   *(_DWORD *)(v6 + 20),
                                   v24,
                                   *(_DWORD *)(v6 + 96),
                                   v23,
                                   (__int64)v65,
                                   0) )
              {
                a1 = v65[0];
                if ( !v65[0] )
                {
                  *(_DWORD *)(v6 + 48) = 1;
                  return 0;
                }
                v27 = *(unsigned __int8 *)(v65[0] + 80);
                v13 = v65[0];
                v28 = *(_BYTE *)(v65[0] + 80);
                if ( (_BYTE)v27 == 16 )
                {
                  v13 = **(_QWORD **)(v65[0] + 88);
                  v28 = *(_BYTE *)(v13 + 80);
                }
                if ( v28 == 24 )
                  v13 = *(_QWORD *)(v13 + 88);
                v14 = (__int64)dword_4F04BA0;
                a2 = *(unsigned int *)(v6 + 124);
                if ( dword_4F04BA0[v27] != (_DWORD)a2
                  || *(char *)(v13 + 83) < 0
                  && unk_4F04C48 != -1
                  && (a2 = (__int64)qword_4F04C68,
                      v51 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368),
                      v13 == v51)
                  && v51
                  || (a2 = v13, v57 = v13, !sub_7CF0D0(v65[0], v13, (_DWORD *)v6))
                  || dword_4F077BC
                  && qword_4F077A8 <= 0x9E33u
                  && !*(_DWORD *)v6
                  && (v13 = v57, *(_BYTE *)(v57 + 80) == 3)
                  && *(_BYTE *)(v57 + 104)
                  && (v49 = *(_QWORD *)(v57 + 88), (*(_BYTE *)(v49 + 177) & 0x10) != 0)
                  && *(_QWORD *)(*(_QWORD *)(v49 + 168) + 168LL) )
                {
                  if ( *(_DWORD *)(v6 + 48) )
                    return 0;
LABEL_75:
                  if ( (*(_BYTE *)(v64 + 16) & 0x10) == 0 )
                  {
LABEL_29:
                    v17 = *(_DWORD *)(v6 + 24);
                    goto LABEL_30;
                  }
                  a2 = *(_QWORD *)(v11 + 208);
                  a1 = (__int64)sub_7D2920(v64, a2);
LABEL_28:
                  if ( a1 )
                    return a1;
                  goto LABEL_29;
                }
              }
              else
              {
                if ( *(_DWORD *)(v6 + 64) )
                {
                  v14 = *(_QWORD *)(v11 + 208);
                  for ( i = v14; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                    ;
                  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 180LL) & 1) != 0 )
                  {
                    *(_QWORD *)(v6 + 104) = v14;
                    *(_DWORD *)(v6 + 68) = 1;
                  }
                }
                a5 = dword_4D047C0;
                if ( dword_4D047C0 && !*(_DWORD *)(v6 + 80) )
                {
                  for ( j = *(_QWORD *)(v11 + 208); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                    ;
                  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 180LL) & 2) != 0 )
                    *(_DWORD *)(v6 + 72) = 1;
                }
              }
              a2 = *(unsigned int *)(v6 + 48);
              a1 = v65[0];
              if ( (_DWORD)a2 || v65[0] )
                return a1;
              goto LABEL_75;
            }
            goto LABEL_26;
          }
LABEL_58:
          if ( (_BYTE)v12 == 15 || (_BYTE)v12 == 8 )
            goto LABEL_60;
        }
        LOBYTE(a5) = v12;
        goto LABEL_143;
      }
      a5 = 6;
      if ( *(_DWORD *)(v6 + 56) )
      {
LABEL_170:
        if ( !(_DWORD)v13 )
          goto LABEL_31;
        if ( !HIDWORD(qword_4D0495C) )
          goto LABEL_36;
LABEL_49:
        if ( (_BYTE)v12 != 17 )
          goto LABEL_36;
        goto LABEL_50;
      }
LABEL_143:
      a2 = v8;
      v56 = a5;
      v11 = 0;
      v48 = sub_7D1590(a5, v8, (__int64 *)v64, v6);
      a5 = v56;
      a1 = v48;
      if ( v8 != -1 )
        v11 = v9 + qword_4F04C68[0];
      if ( !v48 )
        break;
      if ( !*(_DWORD *)(v6 + 24)
        || *(_BYTE *)(v48 + 80) != 7
        || (*(_BYTE *)(*(_QWORD *)(v48 + 88) + 89LL) & 1) == 0
        || (a2 = (__int64)qword_4F04C68,
            v50 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4),
            ((v50 - 15) & 0xFD) != 0)
        && v50 != 2
        || dword_4F077C4 == 2 && dword_4D04964 )
      {
        if ( !*(_DWORD *)(v6 + 76) )
          return a1;
        goto LABEL_148;
      }
      v7 = 1;
LABEL_39:
      v8 = *(_DWORD *)(v11 + 552);
      if ( a4 >= (int)v8 )
        return 0;
    }
    v14 = *(unsigned int *)(v6 + 76);
    if ( (_DWORD)v14 )
    {
LABEL_148:
      LOBYTE(v12) = v56;
      goto LABEL_62;
    }
    v17 = *(_DWORD *)(v6 + 24);
    LOBYTE(v12) = v56;
LABEL_30:
    if ( v17 )
    {
      if ( (unsigned __int8)(v12 - 3) <= 1u )
        return 0;
LABEL_48:
      if ( !HIDWORD(qword_4D0495C) )
        goto LABEL_36;
      goto LABEL_49;
    }
LABEL_31:
    v13 = *(unsigned int *)(v6 + 32);
    if ( !(_DWORD)v13 || (*(_BYTE *)(v64 + 18) & 1) != 0 )
      goto LABEL_48;
    if ( (unsigned __int8)(v12 - 3) <= 1u )
    {
      if ( !*(_DWORD *)(v6 + 4) && !*(_DWORD *)(v6 + 28) || !dword_4F077BC || qword_4F077A8 > 0x9C3Fu )
        return 0;
      goto LABEL_37;
    }
    if ( HIDWORD(qword_4D0495C) )
    {
      if ( (_BYTE)v12 != 17 )
        goto LABEL_36;
LABEL_50:
      if ( (*(_BYTE *)(*(_QWORD *)(v11 + 216) + 89LL) & 4) == 0 )
        *(_DWORD *)(v6 + 60) = 1;
LABEL_37:
      v7 = 1;
      if ( v59 )
      {
        v7 = v59;
        v59 = (unsigned __int8)(v12 - 6) > 1u;
      }
      goto LABEL_39;
    }
    if ( !(_DWORD)qword_4D0495C && ((_BYTE)v12 == 17 || (_BYTE)v12 == 2) )
      return 0;
LABEL_36:
    if ( (_BYTE)v12 != 9 || (*(_BYTE *)(v11 + 7) & 0x20) != 0 )
      goto LABEL_37;
    if ( v8 == -1 )
      BUG();
    v29 = v9 + qword_4F04C68[0];
    a3 = *(_DWORD *)(v9 + qword_4F04C68[0] + 560);
    v30 = *(_DWORD *)(v9 + qword_4F04C68[0] + 552);
    v31 = *(_DWORD *)(v29 + 556);
    if ( unk_4D047BC )
      *(_DWORD *)(v6 + 132) = sub_7D3BE0(a1, a2, v14, v13, a5);
    if ( dword_4D047C8 )
    {
      v52 = 0;
      if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
        v52 = sub_7D3BE0(a1, qword_4F04C68, 0, v13, a5);
      *(_DWORD *)(v6 + 128) = v52;
    }
    else if ( (*(_BYTE *)(v29 + 11) & 0x20) != 0 )
    {
      *(_DWORD *)(v6 + 128) = sub_7D3BE0(a1, a2, v14, v13, a5);
    }
    a2 = v6;
    v32 = sub_7D5120(v64, v6, v30, a3);
    a1 = v32;
    if ( !v32 )
    {
      if ( dword_4D047C0 | dword_4D047C8 || !unk_4D047C4 )
      {
        v33 = *(_DWORD *)(v6 + 48);
        if ( v33 )
          return a1;
        a4 = -1;
        if ( *(_DWORD *)(v6 + 32) )
        {
          v18 = 0;
          v34 = 0;
          goto LABEL_91;
        }
        continue;
      }
      v18 = 0;
      if ( (int)a3 < v31 )
      {
        v54 = sub_7D5120(v64, v6, (unsigned int)v31, a3);
        v18 = a1;
        v34 = v54;
        goto LABEL_206;
      }
      goto LABEL_194;
    }
    break;
  }
  v18 = v32;
  if ( (*(_BYTE *)(v32 + 81) & 0x10) != 0 || dword_4D047C0 | dword_4D047C8 || !unk_4D047C4 )
  {
    if ( *(_DWORD *)(v6 + 48) || !*(_DWORD *)(v6 + 32) )
      return v18;
    v33 = 1;
    goto LABEL_90;
  }
  if ( (int)a3 >= v31 )
  {
LABEL_194:
    v33 = *(_DWORD *)(v6 + 48);
    if ( v33 )
      return v18;
    if ( !*(_DWORD *)(v6 + 32) )
    {
      v61 = v18;
      v53 = sub_7D5120(v64, v6, a3, 0xFFFFFFFFLL);
      v18 = v61;
      if ( !v61 )
        return v53;
LABEL_197:
      v34 = v53;
      goto LABEL_95;
    }
LABEL_90:
    v34 = 0;
    goto LABEL_91;
  }
  v63 = v32;
  v55 = sub_7D5120(v64, v6, (unsigned int)v31, a3);
  v18 = v63;
  v34 = v55;
  if ( v55 )
    goto LABEL_96;
LABEL_206:
  v33 = *(_DWORD *)(v6 + 48);
  if ( !v33 )
  {
    if ( !*(_DWORD *)(v6 + 32) )
      goto LABEL_208;
LABEL_91:
    if ( (*(_DWORD *)(v6 + 4) && dword_4F077BC && qword_4F077A8 <= 0x9C3Fu
       || (int)a3 >= dword_4F04C34
       || (*(_BYTE *)(v64 + 18) & 1) != 0)
      && !v33 )
    {
LABEL_208:
      v62 = v18;
      v53 = sub_7D5120(v64, v6, a3, 0xFFFFFFFFLL);
      v18 = v62;
      if ( !v62 )
      {
        v18 = v53;
        goto LABEL_95;
      }
      goto LABEL_197;
    }
  }
LABEL_95:
  if ( !v34 )
    return v18;
LABEL_96:
  v35 = *(unsigned __int8 *)(v34 + 80);
  v36 = v34;
  if ( (_BYTE)v35 == 16 )
  {
    v36 = **(_QWORD **)(v34 + 88);
    v35 = *(unsigned __int8 *)(v36 + 80);
  }
  if ( (_BYTE)v35 == 24 )
    v35 = *(unsigned __int8 *)(*(_QWORD *)(v36 + 88) + 80LL);
  if ( (unsigned __int8)v35 <= 0x14u )
  {
    v37 = 1182720;
    if ( _bittest64(&v37, v35) )
    {
      if ( v18 )
      {
        v38 = *(unsigned __int8 *)(v18 + 80);
        v39 = v18;
        if ( (_BYTE)v38 == 16 )
        {
          v39 = **(_QWORD **)(v18 + 88);
          v38 = *(unsigned __int8 *)(v39 + 80);
        }
        if ( (_BYTE)v38 == 24 )
          v38 = *(unsigned __int8 *)(*(_QWORD *)(v39 + 88) + 80LL);
        if ( (unsigned __int8)v38 <= 0x14u )
        {
          v40 = 1182720;
          if ( _bittest64(&v40, v38) )
          {
            if ( v34 != v18 )
            {
              v41 = *(_DWORD *)(v6 + 120);
              v60 = v18;
              BYTE1(v41) |= 1u;
              v42 = *(_QWORD *)v64;
              LODWORD(v65[0]) = 0;
              v43 = sub_7CF9D0(v42, v41, 0, 0);
              v44 = sub_7D09E0(v43, v34, v64, 0, 0, v41, v65);
              return sub_7D09E0(v44, v60, v64, 0, 0, v41, v65);
            }
          }
        }
      }
      else
      {
        return v34;
      }
    }
  }
  return v18;
}
