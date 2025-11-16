// Function: sub_8DFA20
// Address: 0x8dfa20
//
__int64 __fastcall sub_8DFA20(
        __int64 a1,
        int a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BOOL4 a7,
        int a8,
        int a9,
        __int64 a10,
        int a11)
{
  __int64 v11; // r14
  int v13; // esi
  int v14; // eax
  __int64 v15; // r15
  __int64 i; // r12
  _BOOL4 v17; // r10d
  char v18; // al
  unsigned int v19; // r10d
  _BOOL4 v20; // eax
  __int64 v21; // rcx
  __int64 v22; // r8
  char v23; // al
  __int64 v25; // rax
  __int64 v26; // r8
  int v27; // edx
  _BOOL4 v28; // r10d
  char v29; // si
  __int64 v30; // r9
  char v31; // al
  __int64 v32; // rcx
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  int v38; // r14d
  int v39; // eax
  int v40; // r13d
  char v41; // al
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  char v45; // dl
  _BOOL4 v46; // eax
  int v47; // eax
  __int64 v48; // rax
  int v49; // eax
  _BOOL4 v50; // eax
  __int64 v51; // rax
  _BOOL4 v52; // eax
  int v53; // eax
  int v54; // eax
  int v55; // eax
  _BOOL4 v56; // eax
  _BOOL4 v57; // eax
  int v58; // eax
  _BOOL4 v59; // eax
  _QWORD *v60; // rax
  __int64 v61; // rax
  _BOOL4 v62; // eax
  int v63; // eax
  _BOOL4 v64; // eax
  unsigned int v65; // [rsp+0h] [rbp-70h]
  unsigned int v66; // [rsp+8h] [rbp-68h]
  unsigned int v67; // [rsp+8h] [rbp-68h]
  __int64 v68; // [rsp+8h] [rbp-68h]
  unsigned int v69; // [rsp+8h] [rbp-68h]
  unsigned int v70; // [rsp+8h] [rbp-68h]
  unsigned int v71; // [rsp+8h] [rbp-68h]
  unsigned int v72; // [rsp+8h] [rbp-68h]
  unsigned int v73; // [rsp+8h] [rbp-68h]
  unsigned int v74; // [rsp+8h] [rbp-68h]
  unsigned int v75; // [rsp+8h] [rbp-68h]
  unsigned int v76; // [rsp+8h] [rbp-68h]
  unsigned int v77; // [rsp+10h] [rbp-60h]
  unsigned int v78; // [rsp+10h] [rbp-60h]
  __int64 v79; // [rsp+10h] [rbp-60h]
  __int64 v80; // [rsp+10h] [rbp-60h]
  __int64 v81; // [rsp+10h] [rbp-60h]
  __int64 v82; // [rsp+10h] [rbp-60h]
  __int64 v83; // [rsp+10h] [rbp-60h]
  __int64 v84; // [rsp+10h] [rbp-60h]
  __int64 v85; // [rsp+10h] [rbp-60h]
  __int64 v86; // [rsp+10h] [rbp-60h]
  __int64 v87; // [rsp+10h] [rbp-60h]
  __int64 v88; // [rsp+10h] [rbp-60h]
  _BOOL4 v89; // [rsp+18h] [rbp-58h]
  _BOOL4 v92; // [rsp+20h] [rbp-50h]
  __int64 v93; // [rsp+20h] [rbp-50h]
  __int64 v94; // [rsp+20h] [rbp-50h]
  __int64 v96; // [rsp+28h] [rbp-48h]
  unsigned int v97; // [rsp+28h] [rbp-48h]
  int v98; // [rsp+38h] [rbp-38h] BYREF
  _DWORD v99[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v11 = a6;
  *(_OWORD *)a10 = 0;
  v13 = dword_4D04964;
  *(_BYTE *)(a10 + 12) |= 0x20u;
  *(_QWORD *)(a10 + 16) = 0;
  if ( v13 )
  {
    v14 = 1;
    if ( byte_4F07472[0] != 8 )
      v14 = a8;
    a8 = v14;
  }
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  if ( *(_BYTE *)(a6 + 140) == 12 )
  {
    do
      v11 = *(_QWORD *)(v11 + 160);
    while ( *(_BYTE *)(v11 + 140) == 12 );
  }
  v15 = sub_8D46C0(v11);
  for ( i = v15; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v17 = sub_8D3D40(a1);
  if ( v17 )
    return 1;
  if ( a2 )
  {
    v20 = sub_712690(a5);
    v17 = 0;
    if ( v20 )
    {
      v23 = *(_BYTE *)(a1 + 140);
      if ( dword_4F077C4 != 2 && v23 == 6 )
      {
        if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
          goto LABEL_57;
        if ( v11 == a1 || (unsigned int)sub_8DED30(a1, v11, 1, v21, v22) )
          return 1;
        v23 = *(_BYTE *)(a1 + 140);
      }
      if ( v23 == 19 )
        return 1;
LABEL_57:
      *(_BYTE *)(a10 + 12) |= 0x18u;
      return 1;
    }
  }
  v18 = *(_BYTE *)(a1 + 140);
  if ( v18 == 19 )
    return 1;
  if ( v18 != 6 )
  {
    if ( dword_4F077C4 != 1 && !(dword_4F077C0 | unk_4D0436C) || a8 || v18 != 2 )
      return v18 == 0;
LABEL_73:
    v19 = 1;
    *(_DWORD *)(a10 + 8) = a9;
    return v19;
  }
  v92 = v17;
  if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
    return v18 == 0;
  v77 = (a11 & 0x1000000) == 0 ? 19 : 3;
  v25 = sub_8D46C0(a1);
  v27 = v77;
  v28 = v92;
  v29 = *(_BYTE *)(v25 + 140);
  v96 = v25;
  v30 = v25;
  if ( v29 == 12 )
  {
    do
    {
      v30 = *(_QWORD *)(v30 + 160);
      v31 = *(_BYTE *)(v30 + 140);
    }
    while ( v31 == 12 );
    v32 = (__int64)&dword_4F06978;
    if ( !dword_4F06978 || v31 != 7 || *(_BYTE *)(i + 140) != 7 )
    {
LABEL_31:
      v33 = v96;
      do
      {
        v33 = *(_QWORD *)(v33 + 160);
        v29 = *(_BYTE *)(v33 + 140);
      }
      while ( v29 == 12 );
      goto LABEL_33;
    }
LABEL_78:
    v89 = v92;
    v94 = v30;
    v46 = sub_8DADD0(v30, i, v77, v32, v26);
    v30 = v94;
    v27 = (a11 & 0x1000000) == 0 ? 19 : 3;
    v28 = v89;
    if ( v46
      || (a11 & 0x1000000) != 0
      && (v57 = sub_8DADD0(i, v94, v77, v32, v26), v30 = v94, v27 = (a11 & 0x1000000) == 0 ? 19 : 3, v28 = v89, v57) )
    {
      v27 |= 0x100000u;
    }
    v29 = *(_BYTE *)(v96 + 140);
    if ( v29 != 12 )
      goto LABEL_33;
    goto LABEL_31;
  }
  v30 = v25;
  v32 = (unsigned int)dword_4F06978;
  if ( dword_4F06978 && v29 == 7 )
  {
    if ( *(_BYTE *)(i + 140) != 7 )
      goto LABEL_35;
    goto LABEL_78;
  }
LABEL_33:
  if ( v29 != 7 )
    v27 |= 0x400000u;
LABEL_35:
  v78 = v28;
  v93 = v30;
  v34 = sub_8DED30(v30, i, v27, v32, v26);
  v37 = v93;
  v19 = v78;
  if ( !v34 )
  {
    v45 = *(_BYTE *)(i + 140);
    if ( !v45 )
      goto LABEL_75;
    v41 = *(_BYTE *)(v93 + 140);
    if ( !v41 )
      goto LABEL_75;
    if ( dword_4F077C4 != 2 )
    {
      if ( v45 != 1 )
      {
LABEL_65:
        if ( v41 != 1 )
        {
          if ( !a3 || a8 )
          {
            if ( dword_4F077C4 != 2 )
            {
              if ( a8 )
                goto LABEL_70;
              goto LABEL_112;
            }
LABEL_129:
            if ( (unsigned __int8)(v41 - 9) > 1u || (unsigned __int8)(*(_BYTE *)(i + 140) - 9) > 1u )
              goto LABEL_130;
            goto LABEL_136;
          }
LABEL_110:
          v71 = v19;
          v83 = v37;
          v52 = sub_8D29E0(v37);
          v37 = v83;
          v19 = v71;
          if ( !v52 || (v64 = sub_8D29E0(i), v37 = v83, v19 = v71, !v64) )
          {
            if ( dword_4F077C4 != 2
              || (unsigned __int8)(*(_BYTE *)(v37 + 140) - 9) > 1u
              || (unsigned __int8)(*(_BYTE *)(i + 140) - 9) > 1u )
            {
              goto LABEL_112;
            }
LABEL_136:
            v75 = v19;
            v87 = v37;
            v59 = sub_8D23B0(v37);
            v37 = v87;
            v19 = v75;
            if ( v59 )
            {
              sub_8AE000(v87);
              v19 = v75;
              v37 = v87;
            }
            if ( (*(_BYTE *)(v37 + 141) & 0x20) == 0
              || (_DWORD)qword_4F077B4
              || dword_4F077BC
              && (qword_4F077A8 <= 0x15F8Fu || qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0) )
            {
              v76 = v19;
              v88 = v37;
              v60 = sub_8D5CE0(v37, i);
              v37 = v88;
              v19 = v76;
              if ( v60 )
              {
                *(_QWORD *)a10 = v60;
                goto LABEL_75;
              }
            }
LABEL_130:
            if ( a8 )
            {
              if ( dword_4F077C4 == 2 )
              {
                v74 = v19;
                v86 = v37;
                v58 = sub_8DF7B0(v96, v15, &v98, v99, 0);
                v37 = v86;
                v19 = v74;
                if ( !v58 )
                {
                  if ( !qword_4D0495C )
                    return v19;
LABEL_71:
                  v67 = v19;
                  v80 = v37;
                  if ( sub_8DEFB0(v96, v15, a7, &v98) )
                  {
                    *(_BYTE *)(a10 + 12) = *(_BYTE *)(a10 + 12) & 0xDD | (2 * (v98 & 1));
                    if ( qword_4D0495C )
                      return 1;
                    goto LABEL_73;
                  }
                  v19 = v67;
                  if ( dword_4F077C4 == 2 || a8 )
                    return v19;
                  v63 = sub_8DED40(i, v80, v42, v43, v44);
                  v37 = v80;
                  v19 = v67;
                  if ( !v63
                    && (dword_4F077C4 == 2
                     || (*(_BYTE *)(i + 140) != 7 || *(_BYTE *)(v80 + 140) != 7)
                     && dword_4F077C4 != 1
                     && !(dword_4F077C0 | unk_4D0436C)) )
                  {
                    return v19;
                  }
                  goto LABEL_122;
                }
LABEL_113:
                v19 = 1;
                *(_BYTE *)(a10 + 12) = *(_BYTE *)(a10 + 12) & 0xDD | (2 * (v98 & 1));
                *(_DWORD *)(a10 + 8) = v99[0];
                return v19;
              }
LABEL_70:
              if ( !qword_4D0495C )
                return v19;
              goto LABEL_71;
            }
LABEL_112:
            v72 = v19;
            v84 = v37;
            v53 = sub_8DF7B0(v96, v15, &v98, v99, 0);
            v37 = v84;
            v19 = v72;
            if ( !v53 )
              goto LABEL_71;
            goto LABEL_113;
          }
          if ( !dword_4D04964 )
          {
LABEL_75:
            if ( a7 )
              return 1;
            goto LABEL_38;
          }
LABEL_122:
          *(_DWORD *)(a10 + 8) = a9;
          goto LABEL_75;
        }
        v73 = v19;
        v85 = v37;
        v56 = sub_8D2530(i);
        v37 = v85;
        v19 = v73;
        if ( v56 || (*(_BYTE *)(i + 141) & 0x20) != 0 )
          goto LABEL_75;
        if ( !a8 )
          goto LABEL_122;
        if ( dword_4F077C4 != 2 )
          goto LABEL_70;
LABEL_128:
        v41 = *(_BYTE *)(v37 + 140);
        goto LABEL_129;
      }
LABEL_98:
      v70 = v19;
      v82 = v37;
      v50 = sub_8D2530(v37);
      v37 = v82;
      v19 = v70;
      if ( !v50 )
      {
        if ( dword_4F077C4 == 2 )
        {
          if ( *(_BYTE *)(v82 + 140) != 7 || a8 )
            return v19;
          while ( *(_BYTE *)(a1 + 140) == 12 )
            a1 = *(_QWORD *)(a1 + 160);
          while ( *(_BYTE *)(v11 + 140) == 12 )
            v11 = *(_QWORD *)(v11 + 160);
          if ( *(_QWORD *)(v11 + 128) < *(_QWORD *)(a1 + 128) )
            return v19;
          *(_BYTE *)(a10 + 12) |= 0x10u;
          *(_DWORD *)(a10 + 8) = a9;
          goto LABEL_75;
        }
        if ( (*(_BYTE *)(v82 + 141) & 0x20) == 0 && (*(_BYTE *)(v82 + 140) != 7 || a8) )
          return v19;
      }
      *(_BYTE *)(a10 + 12) |= 0x10u;
      goto LABEL_75;
    }
    if ( dword_4F04C44 != -1
      || (v51 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v51 + 6) & 6) != 0)
      || *(_BYTE *)(v51 + 4) == 12 )
    {
      v54 = sub_8DBE70(i);
      v37 = v93;
      v19 = v78;
      if ( v54 )
        goto LABEL_75;
      v55 = sub_8DBE70(v93);
      v37 = v93;
      v19 = v78;
      if ( v55 )
        goto LABEL_75;
      if ( *(_BYTE *)(i + 140) == 1 )
        goto LABEL_98;
      if ( dword_4F077C4 != 2 )
      {
        v41 = *(_BYTE *)(v93 + 140);
        goto LABEL_65;
      }
    }
    else if ( v45 == 1 )
    {
      goto LABEL_98;
    }
    if ( !a8 && a3 )
      goto LABEL_110;
    goto LABEL_128;
  }
  *(_BYTE *)(a10 + 12) &= ~0x20u;
  if ( a7 )
    return 1;
  if ( *(_BYTE *)(i + 140) == 7 )
  {
    v49 = sub_8DBCE0(v93, i, a7, v35, v36);
    v37 = v93;
    v19 = v78;
    if ( !v49 )
    {
      *(_BYTE *)(a10 + 13) |= 4u;
      if ( dword_4F06978 )
        return v19;
    }
  }
LABEL_38:
  if ( (*(_BYTE *)(v15 + 140) & 0xFB) != 8 )
  {
    if ( (*(_BYTE *)(v96 + 140) & 0xFB) != 8 )
      return 1;
    v38 = 0;
    goto LABEL_41;
  }
  v65 = v19;
  v68 = v37;
  v47 = sub_8D4C10(v15, dword_4F077C4 != 2);
  v37 = v68;
  v19 = v65;
  v38 = v47;
  if ( (*(_BYTE *)(v96 + 140) & 0xFB) == 8 )
  {
LABEL_41:
    v66 = v19;
    v79 = v37;
    v39 = sub_8D4C10(v96, dword_4F077C4 != 2);
    v37 = v79;
    v19 = v66;
    v40 = v39;
    goto LABEL_42;
  }
  v40 = 0;
LABEL_42:
  if ( v40 == v38 )
    return 1;
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F04C44 != -1
      || (v48 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v48 + 6) & 6) != 0)
      || *(_BYTE *)(v48 + 4) == 12 )
    {
      v69 = v19;
      v81 = v37;
      if ( sub_8D3D40(v15) || sub_8D3D40(v96) )
        return 1;
      v37 = v81;
      v19 = v69;
    }
  }
  if ( (v40 & ~v38) == 0 )
  {
    *(_BYTE *)(a10 + 12) |= 2u;
    return 1;
  }
  if ( a3 )
  {
    if ( dword_4D047E0 )
    {
      if ( unk_4F0771C )
      {
        if ( (v38 | 1) == v40 )
        {
          if ( i == v37 || dword_4F07588 && (v61 = *(_QWORD *)(i + 32), *(_QWORD *)(v37 + 32) == v61) && v61 )
          {
            *(_BYTE *)(a10 + 13) |= 8u;
            if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774)
              || dword_4F077BC && qword_4F077A8 > 0x9D07u )
            {
              v62 = sub_8D29E0(i);
              v19 = 1;
              *(_DWORD *)(a10 + 8) = !v62 + 2464;
              return v19;
            }
            return 1;
          }
        }
      }
    }
  }
  if ( HIDWORD(qword_4D0495C) && *(_BYTE *)(i + 140) == 1 && *(_BYTE *)(v37 + 140) == 1 )
    return 1;
  if ( dword_4F077C0 )
    goto LABEL_73;
  if ( (_DWORD)qword_4F077B4 && (v40 & 8) != 0 && (v40 & ~(v38 | 8)) == 0 )
  {
    v97 = v19;
    if ( !sub_8D2600(i) )
      return v97;
    goto LABEL_73;
  }
  return v19;
}
