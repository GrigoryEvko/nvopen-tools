// Function: sub_827710
// Address: 0x827710
//
__int64 __fastcall sub_827710(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // dl
  __int64 v6; // r14
  __int64 v7; // r13
  unsigned __int8 v8; // al
  unsigned int v9; // r9d
  char v10; // al
  unsigned int v12; // r9d
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // rcx
  char v19; // dl
  int v20; // r12d
  char v21; // al
  _BOOL4 v22; // edx
  char v23; // dl
  _BOOL4 v24; // edx
  __int64 i; // rcx
  __int64 j; // rdx
  bool v27; // si
  bool v28; // al
  _BOOL4 v29; // edi
  _BOOL4 v30; // r9d
  __int64 v31; // r14
  char k; // cl
  __int64 v33; // r13
  char m; // al
  int v35; // edx
  int v36; // eax
  int v37; // eax
  __int64 v39; // rax
  __int64 v40; // [rsp-80h] [rbp-80h]
  __int64 v41; // [rsp-78h] [rbp-78h]
  _BOOL4 v42; // [rsp-78h] [rbp-78h]
  _BOOL4 v43; // [rsp-78h] [rbp-78h]
  __int64 v44; // [rsp-70h] [rbp-70h]
  int v45; // [rsp-68h] [rbp-68h]
  bool v46; // [rsp-61h] [rbp-61h]
  __int64 v47; // [rsp-60h] [rbp-60h]
  unsigned __int64 v48; // [rsp-58h] [rbp-58h]
  int v49; // [rsp-58h] [rbp-58h]
  unsigned __int64 v50; // [rsp-50h] [rbp-50h]
  int v51; // [rsp-50h] [rbp-50h]
  _DWORD v52[15]; // [rsp-3Ch] [rbp-3Ch] BYREF

  v2 = *(_BYTE *)(a1 + 85);
  v3 = v2 ^ *(_BYTE *)(a2 + 85);
  if ( (v3 & 8) == 0 )
  {
    v6 = *(_QWORD *)(a1 + 32);
    v7 = *(_QWORD *)(a2 + 32);
    if ( !v6 || !v7 )
    {
LABEL_11:
      if ( v3 < 0 )
      {
        if ( v2 >= 0 )
          return (unsigned int)-1;
        return 1;
      }
      if ( dword_4F077C4 != 2
        || unk_4F07778 <= 201401
        || (*(_BYTE *)(a1 + 86) & 1) == 0
        || (*(_BYTE *)(a2 + 86) & 1) == 0 )
      {
        goto LABEL_13;
      }
      v13 = sub_8D4840(*(_QWORD *)(a1 + 32));
      v14 = sub_8D4840(*(_QWORD *)(a2 + 32));
      v15 = sub_8D40F0(v13);
      v16 = sub_8D40F0(v14);
      v17 = *(_QWORD *)(a1 + 88);
      v18 = *(_QWORD *)(a2 + 88);
      if ( v15 != v16 )
      {
        v48 = *(_QWORD *)(a2 + 88);
        v50 = *(_QWORD *)(a1 + 88);
        if ( !(unsigned int)sub_8D97D0(v15, v16, 0, v18, v17) )
        {
LABEL_13:
          v10 = *(_BYTE *)(a1 + 13);
          v9 = 0;
          if ( v10 == *(_BYTE *)(a2 + 13) )
            return v9;
          if ( v10 )
            return (unsigned int)-1;
          return 1;
        }
        v17 = v50;
        v18 = v48;
      }
      if ( v17 != v18 )
      {
        if ( v17 >= v18 || !v17 )
          return (unsigned int)-1;
        return 1;
      }
      if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 || dword_4F077BC )
      {
        v39 = *(_QWORD *)(v14 + 176);
        if ( !*(_QWORD *)(v13 + 176) )
        {
          if ( v39 )
            return (unsigned int)-1;
          return 1;
        }
        if ( !v39 )
          return 1;
      }
      goto LABEL_13;
    }
    v8 = *(_BYTE *)(a1 + 86);
    if ( ((v8 ^ *(_BYTE *)(a2 + 86)) & 2) != 0 )
    {
      if ( (v8 & 2) == 0 )
        return (unsigned int)-1;
      return 1;
    }
    if ( HIDWORD(qword_4D0495C) )
    {
LABEL_33:
      if ( !*(_DWORD *)(a1 + 8) || *(_BYTE *)(a1 + 15) && *(_BYTE *)(a2 + 15) )
      {
        v19 = *(_BYTE *)(a2 + 84) & 6;
        if ( (*(_BYTE *)(a1 + 84) & 6) != 0 )
        {
          if ( !v19 )
            return (unsigned int)-1;
LABEL_36:
          if ( (*(_BYTE *)(a1 + 84) & 6) == 0 )
            return 1;
          goto LABEL_10;
        }
        if ( v19 )
          goto LABEL_36;
      }
LABEL_10:
      v2 = *(_BYTE *)(a1 + 85);
      v3 = v2 ^ *(_BYTE *)(a2 + 85);
      goto LABEL_11;
    }
    if ( (_DWORD)qword_4D0495C )
    {
      if ( !(unsigned int)sub_8D32B0(v6) || !(unsigned int)sub_8D32B0(v7) )
        goto LABEL_10;
      goto LABEL_33;
    }
    v20 = sub_8D32E0(v6);
    v51 = sub_8D32E0(v7);
    v21 = *(_BYTE *)(a1 + 84);
    if ( (v21 & 0x20) != 0 || (!v20 ? (v22 = (v21 & 2) != 0) : (v22 = (v21 & 4) != 0), v22) )
    {
      v23 = *(_BYTE *)(a2 + 84);
      if ( (v23 & 0x20) != 0 )
        goto LABEL_51;
      if ( v51 )
        goto LABEL_48;
    }
    else
    {
      v23 = *(_BYTE *)(a2 + 84);
      if ( (v23 & 0x20) != 0 )
        goto LABEL_51;
      if ( v51 )
      {
        if ( (v23 & 4) != 0 )
          return 1;
LABEL_48:
        v24 = (v23 & 4) != 0;
        goto LABEL_49;
      }
      if ( (v23 & 2) != 0 )
        return 1;
    }
    v24 = (v23 & 2) != 0;
LABEL_49:
    if ( (*(_BYTE *)(a1 + 84) & 0x20) == 0 && !v24 && (v20 ? (v21 & 4) != 0 : (v21 & 2) != 0) )
      return (unsigned int)-1;
LABEL_51:
    v46 = v51 != 0 && v20 != 0;
    if ( dword_4D04474 && !((unsigned int)qword_4F077B4 | dword_4F077BC) && v51 != 0 && v20 != 0 )
    {
      v9 = sub_826DB0(a1, a2);
      if ( v9 )
        return v9;
      v47 = sub_8D46C0(v6);
      if ( (*(_BYTE *)(v47 + 140) & 0xFB) != 8 )
      {
        v45 = 0;
LABEL_96:
        v49 = 0;
        v44 = sub_8D46C0(v7);
        if ( (*(_BYTE *)(v44 + 140) & 0xFB) == 8 )
          v49 = sub_8D4C10(v44, dword_4F077C4 != 2);
LABEL_56:
        for ( i = v6; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        for ( j = v7; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v40 = j;
        v41 = i;
        if ( (unsigned int)sub_8D2F30(i, j) )
        {
          v27 = (*(_BYTE *)(a1 + 84) & 2) != 0;
          v28 = (*(_BYTE *)(a2 + 84) & 2) != 0;
          v29 = v27;
          v30 = v28;
          if ( dword_4F06978 && !v28 && !v27 )
          {
            v31 = *(_QWORD *)(v41 + 160);
            for ( k = *(_BYTE *)(v31 + 140); k == 12; k = *(_BYTE *)(v31 + 140) )
              v31 = *(_QWORD *)(v31 + 160);
            v33 = *(_QWORD *)(v40 + 160);
            for ( m = *(_BYTE *)(v33 + 140); m == 12; m = *(_BYTE *)(v33 + 140) )
              v33 = *(_QWORD *)(v33 + 160);
            if ( m == 7 && k == 7 )
            {
              if ( (unsigned int)sub_8DADD0(v31, v33) )
                return (unsigned int)-1;
              if ( (unsigned int)sub_8DADD0(v33, v31) )
                return 1;
            }
LABEL_69:
            if ( dword_4D04474 && (unsigned int)qword_4F077B4 | dword_4F077BC )
            {
              if ( v46 )
              {
LABEL_115:
                v9 = sub_826DB0(a1, a2);
                if ( v9 )
                  return v9;
                goto LABEL_73;
              }
            }
            else if ( v46 )
            {
              goto LABEL_73;
            }
LABEL_71:
            if ( !unk_4D04308 || !(v20 | v51) )
              goto LABEL_10;
LABEL_73:
            if ( v49 == v45 || v44 != v47 && !(unsigned int)sub_8DED30(v47, v44, 3) )
              goto LABEL_10;
            v35 = v49 & ~v45;
            v36 = v45 & ~v49;
            if ( v36 )
            {
              if ( !v35 )
                return (unsigned int)-1;
            }
            else if ( !v35 )
            {
              goto LABEL_10;
            }
            if ( !v36 )
              return 1;
            goto LABEL_10;
          }
        }
        else
        {
          if ( !v46 )
            goto LABEL_71;
          v7 = v44;
          v6 = v47;
          if ( !(unsigned int)sub_8D2F30(v47, v44) )
          {
            if ( !dword_4D04474 || !(dword_4F077BC | (unsigned int)qword_4F077B4) )
              goto LABEL_73;
            goto LABEL_115;
          }
          v30 = (*(_BYTE *)(a2 + 84) & 4) != 0;
          v29 = (*(_BYTE *)(a1 + 84) & 4) != 0;
        }
        if ( v29 )
        {
          v43 = v30;
          v37 = sub_8DEFB0(v7, v6, 0, v52);
          v30 = v43;
          if ( v37 )
          {
            if ( v52[0] )
              return (unsigned int)-1;
          }
        }
        if ( v30 )
        {
          v42 = v30;
          if ( (unsigned int)sub_8DEFB0(v6, v7, 0, v52) )
          {
            if ( v52[0] )
              return v42;
          }
        }
        goto LABEL_69;
      }
    }
    else
    {
      v47 = v6;
      v45 = 0;
      if ( !v20 || (v47 = sub_8D46C0(v6), (*(_BYTE *)(v47 + 140) & 0xFB) != 8) )
      {
LABEL_55:
        v44 = v7;
        v49 = 0;
        if ( !v51 )
          goto LABEL_56;
        goto LABEL_96;
      }
    }
    v45 = sub_8D4C10(v47, dword_4F077C4 != 2);
    goto LABEL_55;
  }
  v12 = 1;
  if ( (v2 & 8) != 0 )
    return (unsigned int)-1;
  return v12;
}
