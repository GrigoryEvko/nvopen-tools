// Function: sub_8E1010
// Address: 0x8e1010
//
__int64 __fastcall sub_8E1010(
        __int64 a1,
        int a2,
        int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        int a10,
        int a11,
        __int64 a12,
        int a13)
{
  __int64 v18; // rbx
  int v19; // ecx
  int v20; // r9d
  __int64 v21; // rcx
  int v22; // eax
  char v23; // dl
  char i; // al
  int v25; // eax
  unsigned int v26; // r14d
  _BOOL4 v28; // eax
  __int64 v29; // rcx
  int v30; // r10d
  __int64 v31; // r8
  __int64 v32; // r11
  char v33; // cl
  char v34; // al
  char v35; // al
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // r8
  unsigned __int8 v40; // dl
  char v41; // dl
  _BOOL4 v42; // eax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r11
  int v46; // edx
  __int64 v47; // rdi
  _QWORD *v48; // rax
  __int64 v49; // rcx
  __int64 v50; // r8
  int v51; // eax
  char v52; // dl
  char v53; // al
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r14
  int v57; // eax
  char v58; // dl
  char v59; // al
  __int64 v60; // rcx
  _BOOL4 v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rax
  unsigned __int64 v64; // rsi
  unsigned __int64 v65; // rcx
  unsigned __int64 v66; // rdx
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  int v69; // eax
  unsigned __int64 v70; // r14
  _BOOL4 v71; // eax
  unsigned __int64 v72; // rax
  int v73; // eax
  char v74; // al
  __int64 v75; // rsi
  __int64 v76; // rsi
  int v77; // eax
  _BOOL4 v78; // eax
  _BOOL4 v79; // eax
  _BOOL4 v80; // eax
  __int64 v81; // [rsp+8h] [rbp-48h]
  unsigned int v82; // [rsp+10h] [rbp-40h]
  __int64 v83; // [rsp+10h] [rbp-40h]
  unsigned int v84; // [rsp+10h] [rbp-40h]
  __int64 v85; // [rsp+10h] [rbp-40h]
  __int64 v86; // [rsp+10h] [rbp-40h]
  int v87; // [rsp+18h] [rbp-38h]
  int v88; // [rsp+18h] [rbp-38h]
  __int64 v89; // [rsp+18h] [rbp-38h]
  unsigned int v90; // [rsp+18h] [rbp-38h]
  __int64 v91; // [rsp+18h] [rbp-38h]
  __int64 v92; // [rsp+18h] [rbp-38h]
  __int64 v93; // [rsp+18h] [rbp-38h]
  int v94; // [rsp+18h] [rbp-38h]

  v18 = a7;
  *(_OWORD *)a12 = 0;
  v19 = *(unsigned __int8 *)(a12 + 12);
  v20 = dword_4D04964;
  *(_QWORD *)(a12 + 16) = 0;
  v21 = v19 | 0x20u;
  *(_BYTE *)(a12 + 12) = v21;
  if ( v20 )
  {
    v22 = 1;
    if ( byte_4F07472[0] != 8 )
      v22 = a10;
    a10 = v22;
  }
  while ( 1 )
  {
    v23 = *(_BYTE *)(a1 + 140);
    if ( v23 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  for ( i = *(_BYTE *)(a7 + 140); i == 12; i = *(_BYTE *)(v18 + 140) )
    v18 = *(_QWORD *)(v18 + 160);
  if ( (*(_BYTE *)(v18 + 141) & 0x20) != 0 )
  {
    if ( (dword_4F077C4 == 2 && unk_4F07778 > 202001 || dword_4F077BC)
      && (unsigned int)sub_8D23E0(v18)
      && sub_8D3410(a1) )
    {
      v36 = sub_8D40F0(v18);
      v37 = sub_8D40F0(a1);
      if ( v36 != v37 && !(unsigned int)sub_8D97D0(v36, v37, 0, v38, v39) )
        goto LABEL_59;
      goto LABEL_26;
    }
LABEL_13:
    LOBYTE(v25) = *(_BYTE *)(a1 + 140);
    goto LABEL_14;
  }
  if ( i != 2 || (*(_BYTE *)(v18 + 162) & 4) == 0 )
  {
    if ( qword_4F077B4 )
    {
      if ( v23 == 15 )
      {
        if ( i != 15 )
          goto LABEL_13;
        v90 = a5;
        if ( v18 == a1 )
          goto LABEL_49;
        if ( !(unsigned int)sub_8D97D0(a1, v18, 0, v21, a5) )
        {
          if ( (_DWORD)qword_4F077B4 && *(_QWORD *)(a1 + 128) == *(_QWORD *)(v18 + 128) )
            goto LABEL_26;
          v84 = v90;
          v91 = sub_8D4620(a1);
          v54 = sub_8D4620(v18);
          v55 = *(_QWORD *)(a1 + 160);
          v56 = *(_QWORD *)(v18 + 160);
          if ( v91 != v54
            || !v84
            || v55 != v56
            && (v92 = *(_QWORD *)(a1 + 160),
                v57 = sub_8D97D0(v92, *(_QWORD *)(v18 + 160), 0x20u, v55, v84),
                v55 = v92,
                !v57) )
          {
            if ( (*(_BYTE *)(a1 + 176) & 1) == 0 || *(_QWORD *)(a1 + 128) != *(_QWORD *)(v18 + 128) )
            {
              if ( unk_4D04240 )
              {
                while ( 1 )
                {
                  v58 = *(_BYTE *)(v55 + 140);
                  if ( v58 != 12 )
                    break;
                  v55 = *(_QWORD *)(v55 + 160);
                }
                while ( 1 )
                {
                  v59 = *(_BYTE *)(v56 + 140);
                  if ( v59 != 12 )
                    break;
                  v56 = *(_QWORD *)(v56 + 160);
                }
                if ( v58 == v59 && *(_QWORD *)(a1 + 128) == *(_QWORD *)(v18 + 128) )
                {
                  *(_DWORD *)(a12 + 8) = 1757;
                  return 1;
                }
              }
              goto LABEL_13;
            }
          }
        }
        goto LABEL_48;
      }
      if ( (_DWORD)qword_4F077B4 && i == 15 && *(_BYTE *)(v18 + 177) == 1 && (unsigned __int8)(v23 - 2) <= 3u )
      {
        *(_BYTE *)(a12 + 12) |= 0x40u;
        v26 = 1;
        goto LABEL_50;
      }
    }
LABEL_23:
    v81 = a6;
    v82 = a5;
    v87 = a3;
    v28 = sub_8D2BF0(v18);
    v30 = v87;
    v31 = v82;
    v32 = v81;
    if ( v28 )
    {
      if ( v18 == a1 || (unsigned int)sub_8D97D0(v18, a1, 0, v29, v82) )
        goto LABEL_26;
      v34 = *(_BYTE *)(v18 + 140);
      v30 = v87;
      v31 = v82;
      v32 = v81;
      if ( (unsigned __int8)(v34 - 2) <= 3u )
      {
LABEL_72:
        v83 = v32;
        v88 = v31;
        if ( !(unsigned int)sub_8D97D0(a1, v18, 0, v29, v31) )
        {
          LOBYTE(v25) = *(_BYTE *)(a1 + 140);
          if ( dword_4F077C4 == 2 && *(_BYTE *)(v18 + 140) == 2 )
          {
            v33 = *(_BYTE *)(v18 + 161);
            if ( (v33 & 8) != 0 )
            {
              if ( HIDWORD(qword_4D0495C) && (_BYTE)v25 == 2 )
              {
                if ( (v33 & 0x10) == 0 )
                {
                  *(_DWORD *)(a12 + 8) = 188;
                  return 1;
                }
                if ( !unk_4D041E4 || (v33 & 0x14) == 0 )
                  return sub_8D3D40(a1);
LABEL_242:
                if ( (*(_BYTE *)(a1 + 161) & 0x10) != 0 || !a8 || v88 )
                  return sub_8D3D40(a1);
                goto LABEL_48;
              }
              if ( unk_4D041E4 && (v33 & 0x14) != 0 )
              {
                if ( (_BYTE)v25 != 2 )
                {
                  if ( (unsigned __int8)(v25 - 3) > 2u )
                    goto LABEL_14;
                  if ( v88 || !a8 )
                    return sub_8D3D40(a1);
                  goto LABEL_48;
                }
                goto LABEL_242;
              }
LABEL_14:
              if ( !(_BYTE)v25 )
                return 1;
              return sub_8D3D40(a1);
            }
          }
          if ( (_BYTE)v25 == 2 )
          {
            if ( (*(_BYTE *)(a1 + 161) & 0x10) == 0 )
            {
LABEL_76:
              v42 = sub_8D2A90(a1);
              v45 = v83;
              if ( !v42
                || (v61 = sub_8D2A90(v18), v45 = v83, !v61)
                || ((v62 = *(unsigned __int8 *)(a1 + 160), v63 = *(unsigned __int8 *)(v18 + 160), (_BYTE)v62 == 14)
                 || (unsigned __int8)v62 <= 8u)
                && ((_BYTE)v63 == 14 || (unsigned __int8)v63 <= 8u) )
              {
                if ( !unk_4D04200 || a2 )
                  goto LABEL_79;
                if ( *(_BYTE *)(v18 + 140) == 2 && (*(_BYTE *)(v18 + 162) & 4) != 0 )
                {
                  v26 = 1;
                  v46 = dword_4F077C4;
                  if ( dword_4F077C4 == 2 )
                    goto LABEL_82;
                  goto LABEL_177;
                }
                v93 = v45;
                v69 = sub_8D97D0(a1, v18, 0, v43, v44);
                v45 = v93;
                if ( v69 )
                  goto LABEL_79;
                v70 = sub_8D2A90(v18)
                    ? dword_4D04120[*(unsigned __int8 *)(v18 + 160)]
                    : *(_QWORD *)(v18 + 128) * dword_4F06BA0;
                v71 = sub_8D2A90(a1);
                v45 = v93;
                v72 = v71 ? dword_4D04120[*(unsigned __int8 *)(a1 + 160)] : *(_QWORD *)(a1 + 128) * dword_4F06BA0;
                if ( v72 <= v70 )
                {
                  v78 = sub_8D2A90(a1);
                  v45 = v93;
                  if ( !v78 )
                    goto LABEL_79;
                  v79 = sub_8D2A90(v18);
                  v45 = v93;
                  if ( v79 )
                    goto LABEL_79;
                }
              }
              else
              {
                v64 = dword_4D04120[v62];
                v65 = dword_4D04120[v63];
                v66 = (int)dword_4D04020[v62];
                v67 = (int)dword_4D04020[v63];
                if ( v64 <= v65 && v66 <= v67 )
                {
                  if ( v64 == v65 && v66 == v67 )
                  {
                    *(_BYTE *)(a12 + 14) |= 2u;
                    v26 = 1;
                    goto LABEL_80;
                  }
LABEL_79:
                  v26 = 1;
LABEL_80:
                  v46 = dword_4F077C4;
                  if ( dword_4F077C4 == 2 || *(_BYTE *)(v18 + 140) != 2 )
                    goto LABEL_82;
LABEL_177:
                  if ( (*(_BYTE *)(v18 + 161) & 8) != 0 )
                  {
                    v68 = v18;
LABEL_179:
                    if ( *(_BYTE *)(a1 + 140) != 2 )
                    {
LABEL_180:
                      *(_QWORD *)(a12 + 8) = *(_QWORD *)(a12 + 8) & 0xFFFFEFFF00000000LL | 0x1000000000BCLL;
                      goto LABEL_82;
                    }
                    v60 = *(unsigned __int8 *)(a1 + 161);
                    if ( (v60 & 8) != 0 )
                    {
                      v75 = a1;
                      if ( a1 == v68 )
                        goto LABEL_159;
                    }
                    else
                    {
                      v75 = *(_QWORD *)(a1 + 168);
                      if ( v75 == v68 )
                        goto LABEL_231;
                      if ( !v75 )
                        goto LABEL_180;
                    }
                    if ( dword_4F07588 )
                    {
                      v76 = *(_QWORD *)(v75 + 32);
                      if ( *(_QWORD *)(v68 + 32) == v76 )
                      {
                        if ( v76 )
                        {
LABEL_158:
                          v47 = a1;
                          if ( (v60 & 8) == 0 )
                            goto LABEL_83;
LABEL_159:
                          v47 = a1;
                          if ( *(_BYTE *)(v18 + 140) != 2 || !unk_4D04000 && (*(_BYTE *)(v18 + 161) & 8) != 0 )
                          {
LABEL_83:
                            v89 = v45;
                            v48 = sub_8D6740(v47);
                            v45 = v89;
                            if ( v48 == (_QWORD *)v18
                              || (v51 = sub_8DED30((__int64)v48, v18, 1, v49, v50), v45 = v89, v51) )
                            {
                              *(_BYTE *)(a12 + 12) |= 0x40u;
                            }
                            else if ( dword_4F077BC )
                            {
                              v85 = v89;
                              v94 = sub_8D2B50(a1);
                              v73 = sub_8D2B50(v18);
                              v45 = v85;
                              if ( v94 != v73 )
                              {
                                if ( qword_4F077A8 <= 0x9D6Bu )
                                {
                                  v40 = *(_BYTE *)(a1 + 140);
                                  LOBYTE(v25) = v40;
                                  if ( dword_4F077C4 == 2 )
                                    goto LABEL_14;
LABEL_194:
                                  if ( unk_4F07778 <= 199900 )
                                    goto LABEL_14;
                                  v74 = *(_BYTE *)(v18 + 140);
                                  if ( v40 == 4 )
                                  {
                                    if ( (unsigned __int8)(v74 - 4) <= 1u )
                                    {
                                      LOBYTE(v25) = 4;
                                      goto LABEL_14;
                                    }
LABEL_199:
                                    *(_QWORD *)(a12 + 8) = *(_QWORD *)(a12 + 8) & 0xFFFFEFFF00000000LL
                                                         | 0x100000000418LL;
                                    LOBYTE(v25) = *(_BYTE *)(a1 + 140);
                                    goto LABEL_14;
                                  }
                                  if ( v74 != 4 || v40 == 5 )
                                    goto LABEL_60;
                                  if ( !a2 )
                                    goto LABEL_199;
                                  v26 = sub_72A2A0(v45);
                                  if ( v26 )
                                    goto LABEL_13;
                                  goto LABEL_189;
                                }
                                v80 = sub_8D2B50(v18);
                                v45 = v85;
                                if ( !v80 )
                                {
                                  v40 = *(_BYTE *)(a1 + 140);
                                  v25 = v40;
                                  if ( dword_4F077C4 == 2 )
                                    goto LABEL_14;
                                  goto LABEL_194;
                                }
                                *(_BYTE *)(a12 + 13) |= 0x40u;
                              }
                            }
                            goto LABEL_86;
                          }
                          LOBYTE(v60) = v60 & 4;
                          if ( (_BYTE)v60 )
                          {
                            v47 = *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8LL);
                            if ( v46 != 2 )
                            {
                              if ( !v47 )
                                goto LABEL_87;
                              goto LABEL_83;
                            }
                            if ( unk_4F07778 <= 201401 )
                              goto LABEL_229;
                            if ( dword_4F077BC )
                            {
                              if ( !(_DWORD)qword_4F077B4 )
                              {
                                if ( qword_4F077A8 > 0x1869Fu )
                                {
LABEL_222:
                                  if ( v47 == v18
                                    || (v86 = v45, v77 = sub_8D97D0(v47, v18, 0, v60, v44), v45 = v86, v77) )
                                  {
                                    *(_BYTE *)(a12 + 12) |= 0xC0u;
                                  }
                                  else if ( v47 )
                                  {
                                    goto LABEL_83;
                                  }
LABEL_86:
                                  if ( dword_4F077C4 == 2 )
                                    goto LABEL_90;
LABEL_87:
                                  if ( unk_4F07778 <= 199900 )
                                    goto LABEL_90;
                                  v52 = *(_BYTE *)(a1 + 140);
                                  v53 = *(_BYTE *)(v18 + 140);
                                  if ( v52 == 4 )
                                  {
                                    if ( (unsigned __int8)(v53 - 4) <= 1u )
                                      goto LABEL_90;
                                    goto LABEL_189;
                                  }
                                  if ( v53 == 4 && v52 != 5 )
                                  {
                                    if ( a2 && (unsigned int)sub_72A2A0(v45) )
                                    {
                                      if ( v26 )
                                        return v26;
                                      goto LABEL_13;
                                    }
LABEL_189:
                                    *(_QWORD *)(a12 + 8) = *(_QWORD *)(a12 + 8) & 0xFFFFEFFF00000000LL
                                                         | 0x100000000418LL;
                                  }
LABEL_90:
                                  if ( !v26 )
                                  {
LABEL_59:
                                    v40 = *(_BYTE *)(a1 + 140);
LABEL_60:
                                    LOBYTE(v25) = v40;
                                    goto LABEL_14;
                                  }
                                  goto LABEL_50;
                                }
LABEL_229:
                                if ( !v47 )
                                  goto LABEL_90;
                                goto LABEL_83;
                              }
                            }
                            else if ( !(_DWORD)qword_4F077B4 )
                            {
                              goto LABEL_222;
                            }
                            if ( qword_4F077A0 > 0x1869Fu )
                              goto LABEL_222;
                            goto LABEL_229;
                          }
LABEL_231:
                          v47 = a1;
                          goto LABEL_83;
                        }
                      }
                    }
                    goto LABEL_180;
                  }
                  v68 = *(_QWORD *)(v18 + 168);
                  if ( v68 )
                    goto LABEL_179;
LABEL_82:
                  v47 = a1;
                  if ( *(_BYTE *)(a1 + 140) != 2 )
                    goto LABEL_83;
                  v60 = *(unsigned __int8 *)(a1 + 161);
                  goto LABEL_158;
                }
                v26 = HIDWORD(qword_4F077B4);
                if ( !HIDWORD(qword_4F077B4) )
                {
                  *(_DWORD *)(a12 + 8) = 2463;
                  goto LABEL_80;
                }
                if ( a2 )
                  goto LABEL_79;
              }
              *(_DWORD *)(a12 + 8) = 2463;
              goto LABEL_79;
            }
            if ( dword_4F077C4 == 1 )
              return sub_8D3D40(a1);
          }
          else
          {
            if ( (unsigned __int8)(v25 - 3) <= 2u )
              goto LABEL_76;
            if ( dword_4F077C4 == 1 )
            {
LABEL_144:
              if ( (_BYTE)v25 == 6 )
              {
                if ( (*(_BYTE *)(a1 + 168) & 1) != 0 || *(_BYTE *)(v18 + 140) != 2 )
                  return sub_8D3D40(a1);
                v26 = 1;
                *(_DWORD *)(a12 + 8) = a11;
                goto LABEL_50;
              }
              goto LABEL_14;
            }
          }
          if ( !(dword_4F077C0 | unk_4D0436C) )
            goto LABEL_14;
          goto LABEL_144;
        }
LABEL_48:
        LOBYTE(v21) = *(_BYTE *)(a12 + 12);
        goto LABEL_49;
      }
    }
    else
    {
      v34 = *(_BYTE *)(v18 + 140);
      if ( (unsigned __int8)(v34 - 2) <= 3u )
      {
        if ( v18 == a1 )
          goto LABEL_48;
        goto LABEL_72;
      }
    }
    switch ( v34 )
    {
      case 6:
        if ( (*(_BYTE *)(v18 + 168) & 1) == 0 )
        {
          v26 = sub_8DFA20(a1, a2, v30, a4, v32, v18, a9, a10, a11, a12, a13);
          goto LABEL_90;
        }
        break;
      case 13:
        v26 = sub_8E0CC0(a1, a2, a4, v32, v18, a9, a12);
        goto LABEL_90;
      case 19:
        LOBYTE(v25) = *(_BYTE *)(a1 + 140);
        if ( (_BYTE)v25 == 19 )
        {
          *(_BYTE *)(a12 + 12) &= ~0x20u;
          if ( *(_BYTE *)(a1 + 140) != 2 )
            goto LABEL_26;
        }
        else
        {
          if ( !a2 || (*(_BYTE *)(v18 + 141) & 0x20) != 0 )
            goto LABEL_14;
          if ( !sub_712690(v32) )
            goto LABEL_13;
          *(_BYTE *)(a12 + 12) = (32 * (*(_BYTE *)(a1 + 140) != 19)) | *(_BYTE *)(a12 + 12) & 0xDF;
          if ( *(_BYTE *)(a1 + 140) != 2 )
            return 1;
        }
        if ( unk_4D04000 || (*(_BYTE *)(a1 + 161) & 8) == 0 )
        {
          *(_BYTE *)(a12 + 12) |= 0x18u;
          v26 = 1;
          goto LABEL_50;
        }
LABEL_26:
        v26 = 1;
        goto LABEL_50;
      case 20:
        LOBYTE(v25) = *(_BYTE *)(a1 + 140);
        if ( (_BYTE)v25 != 20 )
          goto LABEL_14;
        goto LABEL_48;
      case 0:
        goto LABEL_26;
    }
    if ( !sub_8D3D40(v18) )
      goto LABEL_13;
    goto LABEL_26;
  }
  if ( v23 == 2 )
  {
    if ( (*(_BYTE *)(a1 + 161) & 0x18) != 8 )
    {
      if ( (*(_BYTE *)(a1 + 162) & 4) == 0 )
      {
        if ( (*(_BYTE *)(a1 + 161) & 0x10) != 0 )
          goto LABEL_13;
        goto LABEL_26;
      }
LABEL_49:
      v26 = 1;
      *(_BYTE *)(a12 + 12) = v21 & 0xDF;
      goto LABEL_50;
    }
    goto LABEL_23;
  }
  if ( (unsigned __int8)(v23 - 3) <= 2u )
    goto LABEL_26;
  if ( v23 == 6 )
  {
    if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
      goto LABEL_13;
  }
  else if ( v23 != 13 && ((_DWORD)a5 || v23 != 19) )
  {
    goto LABEL_13;
  }
  *(_BYTE *)(a12 + 13) |= 1u;
  v26 = 1;
LABEL_50:
  if ( !a2 && !*(_DWORD *)(a12 + 8) )
  {
    while ( 1 )
    {
      v35 = *(_BYTE *)(a1 + 140);
      if ( v35 != 12 )
        break;
      a1 = *(_QWORD *)(a1 + 160);
    }
    while ( 1 )
    {
      v41 = *(_BYTE *)(v18 + 140);
      if ( v41 != 12 )
        break;
      v18 = *(_QWORD *)(v18 + 160);
    }
    if ( v35 == 2 )
    {
      if ( unk_4D04000 )
      {
        if ( v41 != 2 )
          return v26;
      }
      else if ( (*(_BYTE *)(a1 + 161) & 8) != 0 || v41 != 2 || (*(_BYTE *)(v18 + 161) & 8) != 0 )
      {
        return v26;
      }
      if ( dword_4F06BA0 * *(_QWORD *)(a1 + 128) == 64
        && *(_QWORD *)(v18 + 128) * (unsigned __int64)dword_4F06BA0 <= 0x3F )
      {
        *(_QWORD *)(a12 + 8) = *(_QWORD *)(a12 + 8) & 0xFFFFEFFF00000000LL | 0x10000000055DLL;
      }
    }
  }
  return v26;
}
