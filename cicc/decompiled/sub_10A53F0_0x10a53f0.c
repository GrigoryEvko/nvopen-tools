// Function: sub_10A53F0
// Address: 0x10a53f0
//
__int64 __fastcall sub_10A53F0(__int64 a1, char *a2)
{
  char *v3; // rax
  char *v4; // rdx
  unsigned __int8 v5; // cl
  __int64 v6; // r14
  _BYTE *v7; // r13
  char *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  char v12; // al
  _BYTE *v13; // rdx
  _BYTE *v14; // rax
  char v15; // al
  __int64 *v16; // rdx
  _BYTE *v17; // r8
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  char *v20; // rdx
  char *v21; // rdx
  __int64 v22; // rax
  char *v23; // rcx
  __int64 v24; // rdx
  bool v25; // bl
  _BYTE *v26; // r8
  _BYTE *v27; // rax
  bool v28; // al
  __int64 v29; // rdx
  __int64 v30; // r8
  _BYTE *v31; // rax
  char v32; // dl
  char v33; // cl
  __int64 v34; // rbx
  unsigned __int8 v35; // cl
  _BYTE *v36; // rax
  unsigned int v37; // edx
  bool v38; // al
  const char *v39; // rax
  _QWORD *v40; // rdx
  unsigned __int8 *v41; // r13
  __int64 *v42; // r12
  __int64 v43; // rbx
  __int64 v44; // r12
  __int64 v45; // rdx
  unsigned int v46; // esi
  __int64 v47; // rsi
  _BYTE *v48; // rbx
  unsigned __int8 v49; // cl
  _BYTE *v51; // rcx
  unsigned int v52; // ebx
  bool v53; // al
  _BYTE *v54; // rax
  _BYTE *v55; // rax
  unsigned __int8 *v56; // r8
  bool v57; // dl
  unsigned int v58; // ebx
  bool v59; // dl
  __int64 v60; // rdx
  __int64 v61; // rdx
  _BYTE *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rdx
  _BYTE *v67; // rax
  unsigned int v68; // ebx
  unsigned int v69; // ecx
  _BYTE *v70; // rax
  unsigned int v71; // ecx
  bool v72; // al
  unsigned int v73; // ecx
  __int64 v74; // rax
  unsigned int v75; // ecx
  int v76; // eax
  int v77; // [rsp+4h] [rbp-CCh]
  unsigned __int8 *v78; // [rsp+8h] [rbp-C8h]
  bool v79; // [rsp+8h] [rbp-C8h]
  int v80; // [rsp+8h] [rbp-C8h]
  int v81; // [rsp+8h] [rbp-C8h]
  int v82; // [rsp+8h] [rbp-C8h]
  bool v83; // [rsp+10h] [rbp-C0h]
  bool v84; // [rsp+10h] [rbp-C0h]
  int v85; // [rsp+10h] [rbp-C0h]
  __int64 v86; // [rsp+18h] [rbp-B8h]
  __int64 v87; // [rsp+18h] [rbp-B8h]
  _BYTE *v88; // [rsp+18h] [rbp-B8h]
  int v89; // [rsp+18h] [rbp-B8h]
  __int64 v90; // [rsp+18h] [rbp-B8h]
  int v91; // [rsp+18h] [rbp-B8h]
  __int64 v92; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v93; // [rsp+18h] [rbp-B8h]
  __int64 v94; // [rsp+18h] [rbp-B8h]
  unsigned int v95; // [rsp+18h] [rbp-B8h]
  unsigned int v96; // [rsp+18h] [rbp-B8h]
  __int64 v97; // [rsp+20h] [rbp-B0h]
  __int64 v98; // [rsp+28h] [rbp-A8h]
  char v99; // [rsp+47h] [rbp-89h] BYREF
  _BYTE *v100; // [rsp+48h] [rbp-88h] BYREF
  __int64 v101; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v102; // [rsp+58h] [rbp-78h] BYREF
  _BYTE *v103; // [rsp+60h] [rbp-70h] BYREF
  int v104; // [rsp+68h] [rbp-68h] BYREF
  char v105; // [rsp+6Ch] [rbp-64h]
  char *v106; // [rsp+70h] [rbp-60h] BYREF
  _QWORD *v107; // [rsp+78h] [rbp-58h]
  __int64 *v108; // [rsp+80h] [rbp-50h] BYREF
  char *v109; // [rsp+88h] [rbp-48h] BYREF
  _QWORD *v110; // [rsp+90h] [rbp-40h]
  _QWORD *v111; // [rsp+98h] [rbp-38h]

  v3 = (char *)*((_QWORD *)a2 - 8);
  v4 = (char *)*((_QWORD *)a2 - 4);
  v5 = *v3;
  if ( (unsigned __int8)*v3 <= 0x1Cu )
    goto LABEL_60;
  if ( v5 == 67 )
  {
    v48 = (_BYTE *)*((_QWORD *)v3 - 4);
    v97 = (__int64)v48;
    if ( *v48 != 55 )
      goto LABEL_60;
    v6 = *((_QWORD *)v48 - 8);
    if ( !v6 )
      goto LABEL_60;
    v7 = (_BYTE *)*((_QWORD *)v48 - 4);
    if ( *v7 <= 0x1Cu )
      goto LABEL_60;
  }
  else
  {
    if ( v5 != 55 )
      goto LABEL_60;
    v6 = *((_QWORD *)v3 - 8);
    if ( !v6 )
      goto LABEL_60;
    v7 = (_BYTE *)*((_QWORD *)v3 - 4);
    if ( *v7 <= 0x1Cu )
      goto LABEL_60;
    v97 = *((_QWORD *)a2 - 8);
  }
  if ( v4 )
  {
    v8 = (char *)*((_QWORD *)a2 - 4);
    goto LABEL_9;
  }
LABEL_60:
  v49 = *v4;
  if ( (unsigned __int8)*v4 <= 0x1Cu )
    return 0;
  if ( v49 == 67 )
  {
    v51 = (_BYTE *)*((_QWORD *)v4 - 4);
    v97 = (__int64)v51;
    if ( *v51 != 55 )
      return 0;
    v6 = *((_QWORD *)v51 - 8);
    if ( !v6 )
      return 0;
    v7 = (_BYTE *)*((_QWORD *)v51 - 4);
    if ( *v7 <= 0x1Cu )
      return 0;
  }
  else
  {
    if ( v49 != 55 )
      return 0;
    v6 = *((_QWORD *)v4 - 8);
    if ( !v6 )
      return 0;
    v7 = (_BYTE *)*((_QWORD *)v4 - 4);
    if ( *v7 <= 0x1Cu )
      return 0;
    v97 = *((_QWORD *)a2 - 4);
  }
  v8 = (char *)*((_QWORD *)a2 - 8);
  if ( *a2 == 44 && v3 != v4 )
    return 0;
LABEL_9:
  v9 = *(_QWORD *)(v6 + 8);
  v98 = *((_QWORD *)a2 + 1);
  if ( v98 != v9 )
  {
    v10 = *((_QWORD *)v3 + 2);
    if ( !v10 || *(_QWORD *)(v10 + 8) )
    {
      v11 = *((_QWORD *)v4 + 2);
      if ( !v11 || *(_QWORD *)(v11 + 8) )
        return 0;
    }
  }
  v106 = (char *)(unsigned int)sub_BCB060(v9);
  v107 = &v100;
  v108 = (__int64 *)&v100;
  v109 = v106;
  v110 = &v100;
  v111 = &v100;
  v12 = *v7;
  if ( *v7 != 68 )
    goto LABEL_17;
  v13 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( *v13 != 44 )
    return 0;
  v86 = *((_QWORD *)v7 - 4);
  if ( sub_F17ED0(&v106, *((_QWORD *)v13 - 8)) )
  {
    v54 = *(_BYTE **)(v86 - 32);
    if ( *v54 == 68 && (v63 = *((_QWORD *)v54 - 4)) != 0 )
      *v107 = v63;
    else
      *v108 = (__int64)v54;
    goto LABEL_21;
  }
  v12 = *v7;
LABEL_17:
  if ( v12 != 44 || !sub_F17ED0(&v109, *((_QWORD *)v7 - 8)) )
    return 0;
  v14 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( *v14 == 68 && (v60 = *((_QWORD *)v14 - 4)) != 0 )
    *v110 = v60;
  else
    *v111 = v14;
LABEL_21:
  v15 = *v8;
  if ( *a2 == 44 )
  {
    if ( v15 != 68 )
      goto LABEL_23;
  }
  else if ( v15 != 69 )
  {
    goto LABEL_23;
  }
  if ( *((_QWORD *)v8 - 4) )
    v8 = (char *)*((_QWORD *)v8 - 4);
LABEL_23:
  v104 = 42;
  v106 = (char *)&v104;
  v108 = &v101;
  v110 = &v102;
  v105 = 0;
  v107 = (_QWORD *)v6;
  LOBYTE(v109) = 0;
  v111 = &v103;
  if ( *v8 != 86 )
    return 0;
  v16 = (v8[7] & 0x40) != 0 ? (__int64 *)*((_QWORD *)v8 - 1) : (__int64 *)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
  v17 = (_BYTE *)*v16;
  if ( *(_BYTE *)*v16 != 82 )
    return 0;
  v18 = *((_QWORD *)v17 - 8);
  if ( v6 != v18 )
    return 0;
  if ( !v18 )
    return 0;
  v87 = *v16;
  if ( !(unsigned __int8)sub_991580((__int64)&v108, *((_QWORD *)v17 - 4)) )
    return 0;
  if ( v106 )
  {
    v19 = sub_B53900(v87);
    v20 = v106;
    *(_DWORD *)v106 = v19;
    v20[4] = BYTE4(v19);
  }
  v21 = (v8[7] & 0x40) != 0 ? (char *)*((_QWORD *)v8 - 1) : &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
  v22 = *((_QWORD *)v21 + 4);
  if ( !v22 )
    return 0;
  *v110 = v22;
  v23 = (v8[7] & 0x40) != 0 ? (char *)*((_QWORD *)v8 - 1) : &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
  v24 = *((_QWORD *)v23 + 8);
  if ( !v24 )
    return 0;
  *v111 = v24;
  v25 = sub_9893F0(v104, v101, &v99);
  if ( !v25 )
    return 0;
  v26 = v103;
  if ( !v99 )
  {
    v27 = v102;
    v102 = v103;
    v103 = v27;
    v26 = v27;
  }
  if ( *v26 > 0x15u )
    return 0;
  v88 = v26;
  v28 = sub_AC30F0((__int64)v26);
  v30 = (__int64)v88;
  if ( !v28 )
  {
    if ( *v88 == 17 )
    {
      v52 = *((_DWORD *)v88 + 8);
      if ( v52 <= 0x40 )
        v53 = *((_QWORD *)v88 + 3) == 0;
      else
        v53 = v52 == (unsigned int)sub_C444A0((__int64)(v88 + 24));
    }
    else
    {
      v90 = *((_QWORD *)v88 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v90 + 8) - 17 > 1 )
        return 0;
      v78 = (unsigned __int8 *)v30;
      v55 = sub_AD7630(v30, 0, v29);
      v56 = v78;
      v57 = 0;
      if ( !v55 || *v55 != 17 )
      {
        if ( *(_BYTE *)(v90 + 8) == 17 )
        {
          v77 = *(_DWORD *)(v90 + 32);
          if ( v77 )
          {
            v64 = 0;
            while ( 1 )
            {
              v79 = v57;
              v93 = v56;
              v65 = sub_AD69F0(v56, v64);
              if ( !v65 )
                break;
              v56 = v93;
              v57 = v79;
              if ( *(_BYTE *)v65 != 13 )
              {
                if ( *(_BYTE *)v65 != 17 )
                  return 0;
                if ( *(_DWORD *)(v65 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v65 + 24) )
                    return 0;
                  v57 = v25;
                }
                else
                {
                  v80 = *(_DWORD *)(v65 + 32);
                  if ( v80 != (unsigned int)sub_C444A0(v65 + 24) )
                    return 0;
                  v56 = v93;
                  v57 = v25;
                }
              }
              v64 = (unsigned int)(v64 + 1);
              if ( v77 == (_DWORD)v64 )
              {
                if ( v57 )
                  goto LABEL_43;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v58 = *((_DWORD *)v55 + 8);
      if ( v58 <= 0x40 )
        v53 = *((_QWORD *)v55 + 3) == 0;
      else
        v53 = v58 == (unsigned int)sub_C444A0((__int64)(v55 + 24));
    }
    if ( !v53 )
      return 0;
  }
LABEL_43:
  v31 = v102;
  v32 = *a2;
  v33 = *v102;
  if ( *a2 == 44 )
  {
    if ( v33 != 68 )
      goto LABEL_45;
  }
  else if ( v33 != 69 )
  {
    goto LABEL_45;
  }
  if ( *((_QWORD *)v102 - 4) )
    v31 = (_BYTE *)*((_QWORD *)v102 - 4);
LABEL_45:
  v102 = v31;
  if ( *v31 != 54 )
    return 0;
  v34 = *((_QWORD *)v31 - 8);
  v35 = *(_BYTE *)v34;
  if ( *(_BYTE *)v34 > 0x15u )
    return 0;
  v36 = (_BYTE *)*((_QWORD *)v31 - 4);
  if ( (*v36 != 68 || v100 != *((_BYTE **)v36 - 4)) && v100 != v36 )
    return 0;
  if ( v32 == 44 )
  {
    if ( v35 == 17 )
    {
      if ( *(_DWORD *)(v34 + 32) <= 0x40u )
      {
        v38 = *(_QWORD *)(v34 + 24) == 1;
        goto LABEL_54;
      }
      v91 = *(_DWORD *)(v34 + 32);
      v59 = v91 - 1 == (unsigned int)sub_C444A0(v34 + 24);
    }
    else
    {
      v66 = *(_QWORD *)(v34 + 8);
      v94 = v66;
      if ( (unsigned int)*(unsigned __int8 *)(v66 + 8) - 17 > 1 )
        return 0;
      v67 = sub_AD7630(v34, 0, v66);
      if ( v67 && *v67 == 17 )
      {
        v68 = *((_DWORD *)v67 + 8);
        if ( v68 <= 0x40 )
          v38 = *((_QWORD *)v67 + 3) == 1;
        else
          v38 = v68 - 1 == (unsigned int)sub_C444A0((__int64)(v67 + 24));
        goto LABEL_54;
      }
      if ( *(_BYTE *)(v94 + 8) != 17 )
        return 0;
      v59 = 0;
      v82 = *(_DWORD *)(v94 + 32);
      if ( v82 )
      {
        v73 = 0;
        while ( 1 )
        {
          v84 = v59;
          v96 = v73;
          v74 = sub_AD69F0((unsigned __int8 *)v34, v73);
          if ( !v74 )
            return 0;
          v75 = v96;
          v59 = v84;
          if ( *(_BYTE *)v74 != 13 )
          {
            if ( *(_BYTE *)v74 != 17 )
              return 0;
            if ( *(_DWORD *)(v74 + 32) <= 0x40u )
            {
              v59 = *(_QWORD *)(v74 + 24) == 1;
            }
            else
            {
              v85 = *(_DWORD *)(v74 + 32);
              v76 = sub_C444A0(v74 + 24);
              v75 = v96;
              v59 = v85 - 1 == v76;
            }
            if ( !v59 )
              return 0;
          }
          v73 = v75 + 1;
          if ( v82 == v73 )
          {
            v38 = v59;
            goto LABEL_54;
          }
        }
      }
    }
LABEL_105:
    v38 = v59;
    goto LABEL_54;
  }
  if ( v35 != 17 )
  {
    v61 = *(_QWORD *)(v34 + 8);
    v92 = v61;
    if ( (unsigned int)*(unsigned __int8 *)(v61 + 8) - 17 > 1 )
      return 0;
    v62 = sub_AD7630(v34, 0, v61);
    if ( v62 && *v62 == 17 )
    {
      v38 = sub_986760((__int64)(v62 + 24));
      goto LABEL_54;
    }
    if ( *(_BYTE *)(v92 + 8) != 17 )
      return 0;
    v59 = 0;
    v81 = *(_DWORD *)(v92 + 32);
    if ( v81 )
    {
      v69 = 0;
      while ( 1 )
      {
        v83 = v59;
        v95 = v69;
        v70 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v34, v69);
        if ( !v70 )
          return 0;
        v71 = v95;
        v59 = v83;
        if ( *v70 != 13 )
        {
          if ( *v70 != 17 )
            return 0;
          v72 = sub_986760((__int64)(v70 + 24));
          v71 = v95;
          v59 = v72;
          if ( !v72 )
            return 0;
        }
        v69 = v71 + 1;
        if ( v81 == v69 )
        {
          v38 = v59;
          goto LABEL_54;
        }
      }
    }
    goto LABEL_105;
  }
  v37 = *(_DWORD *)(v34 + 32);
  if ( !v37 )
    goto LABEL_55;
  if ( v37 <= 0x40 )
  {
    v38 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) == *(_QWORD *)(v34 + 24);
  }
  else
  {
    v89 = *(_DWORD *)(v34 + 32);
    v38 = v89 == (unsigned int)sub_C445E0(v34 + 24);
  }
LABEL_54:
  if ( !v38 )
    return 0;
LABEL_55:
  v39 = sub_BD5D20(v97);
  LOWORD(v110) = 773;
  v106 = (char *)v39;
  v107 = v40;
  v108 = (__int64 *)".sext";
  v41 = (unsigned __int8 *)sub_B504D0(27, v6, (__int64)v7, (__int64)&v106, 0, 0);
  sub_B45260(v41, v97, 1);
  if ( v98 != v9 )
  {
    LOWORD(v110) = 257;
    v42 = *(__int64 **)(a1 + 32);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, char **, __int64, __int64))(*(_QWORD *)v42[11] + 16LL))(
      v42[11],
      v41,
      &v106,
      v42[7],
      v42[8]);
    v43 = *v42;
    v44 = *v42 + 16LL * *((unsigned int *)v42 + 2);
    while ( v44 != v43 )
    {
      v45 = *(_QWORD *)(v43 + 8);
      v46 = *(_DWORD *)v43;
      v43 += 16;
      sub_B99FD0((__int64)v41, v46, v45);
    }
    v47 = *((_QWORD *)a2 + 1);
    LOWORD(v110) = 257;
    return sub_B52120((__int64)v41, v47, (__int64)&v106, 0, 0);
  }
  return (__int64)v41;
}
