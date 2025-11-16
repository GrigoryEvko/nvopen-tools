// Function: sub_1009F30
// Address: 0x1009f30
//
unsigned __int8 *__fastcall sub_1009F30(_BYTE *a1, _BYTE *a2, char a3, __int64 *a4, char a5, char a6)
{
  bool v8; // bl
  __int64 v9; // r12
  _BYTE *v11; // rax
  char v12; // cl
  _BYTE *v13; // r14
  _BYTE *v14; // rdx
  unsigned __int8 v15; // al
  _BYTE *v16; // rsi
  _BYTE *v17; // r15
  _BYTE *v18; // r13
  _BYTE *v19; // rax
  __int64 v20; // rdi
  void *v21; // rax
  _BYTE *v22; // rdx
  __int64 v23; // rdi
  bool v24; // al
  __int64 v25; // r14
  void *v26; // rax
  __int64 v27; // rdi
  _BYTE *v28; // rdx
  bool v29; // al
  void *v30; // rax
  _BYTE *v31; // rax
  bool v32; // al
  __int64 v33; // r15
  __int64 v34; // rdx
  _BYTE *v35; // rax
  void *v36; // rax
  _BYTE *v37; // r14
  char v38; // al
  _BYTE *v39; // rax
  _BYTE *v40; // r15
  void *v41; // rax
  _BYTE *v42; // r15
  char v43; // al
  _BYTE *v44; // rax
  void *v45; // rax
  _BYTE *v46; // rsi
  __int64 v47; // r15
  __int64 v48; // rdx
  void **v49; // rax
  bool v50; // cl
  void **v51; // r14
  void **v52; // r14
  unsigned int v53; // r15d
  void **v54; // rax
  void **v55; // rdx
  char v56; // al
  void *v57; // rax
  _BYTE *v58; // rdx
  unsigned int v59; // r15d
  void **v60; // rax
  void **v61; // rsi
  char v62; // al
  void *v63; // rax
  _BYTE *v64; // rsi
  bool v65; // r9
  unsigned int v66; // r8d
  void **v67; // rax
  void **v68; // rsi
  char v69; // al
  unsigned int v70; // r8d
  void *v71; // rax
  _BYTE *v72; // rsi
  unsigned int v73; // r14d
  void **v74; // rax
  void **v75; // r15
  char v76; // al
  _BYTE *v77; // r15
  _BYTE *v78; // [rsp+0h] [rbp-80h]
  int v79; // [rsp+8h] [rbp-78h]
  _BYTE *v80; // [rsp+10h] [rbp-70h]
  bool v81; // [rsp+10h] [rbp-70h]
  char v82; // [rsp+10h] [rbp-70h]
  _BYTE *v83; // [rsp+18h] [rbp-68h]
  char v84; // [rsp+18h] [rbp-68h]
  char v85; // [rsp+18h] [rbp-68h]
  _BYTE *v86; // [rsp+18h] [rbp-68h]
  bool v87; // [rsp+18h] [rbp-68h]
  char v88; // [rsp+18h] [rbp-68h]
  _BYTE *v89; // [rsp+18h] [rbp-68h]
  bool v90; // [rsp+18h] [rbp-68h]
  _BYTE *v91; // [rsp+20h] [rbp-60h]
  int v92; // [rsp+20h] [rbp-60h]
  char v93; // [rsp+20h] [rbp-60h]
  unsigned int v94; // [rsp+20h] [rbp-60h]
  bool v95; // [rsp+20h] [rbp-60h]
  char v96; // [rsp+28h] [rbp-58h]
  char v97; // [rsp+28h] [rbp-58h]
  _BYTE *v98; // [rsp+28h] [rbp-58h]
  __int64 v99; // [rsp+28h] [rbp-58h]
  __int64 v100; // [rsp+28h] [rbp-58h]
  char v101; // [rsp+28h] [rbp-58h]
  bool v102; // [rsp+28h] [rbp-58h]
  char v103; // [rsp+28h] [rbp-58h]
  void **v104; // [rsp+28h] [rbp-58h]
  int v105; // [rsp+28h] [rbp-58h]
  int v106; // [rsp+28h] [rbp-58h]
  _BYTE *v107; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v108; // [rsp+38h] [rbp-48h] BYREF
  __int64 v109; // [rsp+40h] [rbp-40h] BYREF
  _BYTE *v110; // [rsp+48h] [rbp-38h]

  v108 = a1;
  v107 = a2;
  v8 = a6 == 1 && a5 == 0;
  if ( !v8 )
  {
    v109 = (__int64)v108;
    v110 = v107;
    return sub_1003820(&v109, 2, a3, (__int64)a4, a5, a6);
  }
  v9 = sub_FFE3E0(0x15u, &v108, &v107, a4);
  if ( v9 )
    return (unsigned __int8 *)v9;
  v109 = (__int64)v108;
  v110 = v107;
  v9 = (__int64)sub_1003820(&v109, 2, a3, (__int64)a4, 0, 1);
  if ( v9 )
    return (unsigned __int8 *)v9;
  v109 = 0x3FF0000000000000LL;
  v12 = sub_1009690((double *)&v109, (__int64)v107);
  if ( v12 )
    return v108;
  if ( (a3 & 2) == 0 )
    return (unsigned __int8 *)v9;
  v13 = v108;
  if ( (a3 & 8) != 0 )
  {
    if ( *v108 == 18 )
    {
      v30 = sub_C33340();
      v12 = 0;
      if ( *((void **)v13 + 3) == v30 )
        v31 = (_BYTE *)*((_QWORD *)v13 + 4);
      else
        v31 = v13 + 24;
      v32 = (v31[20] & 7) == 3;
    }
    else
    {
      v33 = *((_QWORD *)v108 + 1);
      v34 = (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17;
      if ( (unsigned int)v34 > 1 || *v108 > 0x15u )
        goto LABEL_10;
      v35 = sub_AD7630((__int64)v108, 0, v34);
      v12 = 0;
      if ( !v35 || (v98 = v35, *v35 != 18) )
      {
        if ( *(_BYTE *)(v33 + 8) == 17 )
        {
          v92 = *(_DWORD *)(v33 + 32);
          if ( v92 )
          {
            v87 = 0;
            v53 = 0;
            while ( 1 )
            {
              v103 = v12;
              v54 = (void **)sub_AD69F0(v13, v53);
              v12 = v103;
              v55 = v54;
              if ( !v54 )
                break;
              v56 = *(_BYTE *)v54;
              v104 = v55;
              if ( v56 != 13 )
              {
                if ( v56 != 18 )
                  break;
                v88 = v12;
                v57 = sub_C33340();
                v12 = v88;
                v58 = v104[3] == v57 ? v104[4] : v104 + 3;
                if ( (v58[20] & 7) != 3 )
                  break;
                v87 = v8;
              }
              if ( v92 == ++v53 )
              {
                v13 = v108;
                if ( v87 )
                  return sub_AD9290(*((_QWORD *)v13 + 1), 0);
                goto LABEL_10;
              }
            }
          }
        }
        v13 = v108;
        goto LABEL_10;
      }
      v36 = sub_C33340();
      v12 = 0;
      v37 = v98 + 24;
      if ( *((void **)v98 + 3) == v36 )
        v37 = (_BYTE *)*((_QWORD *)v98 + 4);
      v38 = v37[20];
      v13 = v108;
      v32 = (v38 & 7) == 3;
    }
    if ( v32 )
      return sub_AD9290(*((_QWORD *)v13 + 1), 0);
  }
LABEL_10:
  v14 = v107;
  if ( v107 == v13 )
    return sub_AD8DD0(*((_QWORD *)v13 + 1), 1.0);
  v15 = *v13;
  if ( (a3 & 1) != 0 && v15 == 47 )
  {
    v11 = (_BYTE *)*((_QWORD *)v13 - 8);
    v16 = (_BYTE *)*((_QWORD *)v13 - 4);
    if ( !v11 || v107 != v16 )
    {
      if ( v16 && v107 == v11 )
        return (unsigned __int8 *)*((_QWORD *)v13 - 4);
      v17 = v13;
      goto LABEL_18;
    }
    return v11;
  }
  v17 = v13;
  if ( v15 != 45 )
    goto LABEL_18;
  v20 = *((_QWORD *)v13 - 8);
  if ( *(_BYTE *)v20 == 18 )
  {
    v83 = v107;
    v96 = v12;
    v21 = sub_C33340();
    v12 = v96;
    v22 = v83;
    if ( *(void **)(v20 + 24) == v21 )
      v23 = *(_QWORD *)(v20 + 32);
    else
      v23 = v20 + 24;
    v17 = v13;
    v24 = (*(_BYTE *)(v23 + 20) & 7) == 3;
  }
  else
  {
    v99 = *(_QWORD *)(v20 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v99 + 8) - 17 > 1 || *(_BYTE *)v20 > 0x15u )
      goto LABEL_18;
    v80 = v107;
    v84 = v12;
    v39 = sub_AD7630(v20, 0, (__int64)v107);
    v12 = v84;
    v22 = v80;
    v40 = v39;
    if ( !v39 || *v39 != 18 )
    {
      if ( *(_BYTE *)(v99 + 8) == 17 )
      {
        v79 = *(_DWORD *)(v99 + 32);
        if ( v79 )
        {
          v81 = 0;
          v59 = 0;
          while ( 1 )
          {
            v89 = v22;
            v93 = v12;
            v60 = (void **)sub_AD69F0((unsigned __int8 *)v20, v59);
            v12 = v93;
            v22 = v89;
            v61 = v60;
            if ( !v60 )
              break;
            v62 = *(_BYTE *)v60;
            if ( v62 != 13 )
            {
              if ( v62 != 18 )
                break;
              v63 = sub_C33340();
              v12 = v93;
              v22 = v89;
              v64 = v61[3] == v63 ? v61[4] : v61 + 3;
              if ( (v64[20] & 7) != 3 )
                break;
              v81 = v8;
            }
            if ( v79 == ++v59 )
            {
              v17 = v108;
              if ( v81 )
                goto LABEL_32;
              goto LABEL_33;
            }
          }
        }
      }
      v17 = v108;
      v14 = v107;
      goto LABEL_18;
    }
    v41 = sub_C33340();
    v12 = v84;
    v22 = v80;
    if ( *((void **)v40 + 3) == v41 )
      v42 = (_BYTE *)*((_QWORD *)v40 + 4);
    else
      v42 = v40 + 24;
    v43 = v42[20];
    v17 = v108;
    v24 = (v43 & 7) == 3;
  }
  if ( v24 )
  {
LABEL_32:
    if ( v22 == *((_BYTE **)v13 - 4) )
      return sub_AD8DD0(*((_QWORD *)v17 + 1), -1.0);
  }
LABEL_33:
  v14 = v107;
LABEL_18:
  if ( *v14 != 45 )
    goto LABEL_19;
  v25 = *((_QWORD *)v14 - 8);
  v91 = v14;
  if ( *(_BYTE *)v25 == 18 )
  {
    v97 = v12;
    v26 = sub_C33340();
    v12 = v97;
    v27 = v25 + 24;
    v28 = v91;
    if ( *(void **)(v25 + 24) == v26 )
      v27 = *(_QWORD *)(v25 + 32);
    v29 = (*(_BYTE *)(v27 + 20) & 7) == 3;
  }
  else
  {
    v100 = *(_QWORD *)(v25 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v100 + 8) - 17 > 1 || *(_BYTE *)v25 > 0x15u )
      goto LABEL_19;
    v85 = v12;
    v44 = sub_AD7630(v25, 0, (__int64)v14);
    v12 = v85;
    v28 = v91;
    if ( !v44 || (v86 = v44, *v44 != 18) )
    {
      if ( *(_BYTE *)(v100 + 8) == 17 )
      {
        v105 = *(_DWORD *)(v100 + 32);
        if ( v105 )
        {
          v65 = 0;
          v66 = 0;
          while ( 1 )
          {
            v82 = v12;
            v78 = v28;
            v90 = v65;
            v94 = v66;
            v67 = (void **)sub_AD69F0((unsigned __int8 *)v25, v66);
            v12 = v82;
            v68 = v67;
            if ( !v67 )
              break;
            v69 = *(_BYTE *)v67;
            v70 = v94;
            v65 = v90;
            v28 = v78;
            if ( v69 != 13 )
            {
              if ( v69 != 18 )
                goto LABEL_19;
              v71 = sub_C33340();
              v12 = v82;
              v70 = v94;
              v28 = v78;
              v72 = v68[3] == v71 ? v68[4] : v68 + 3;
              if ( (v72[20] & 7) != 3 )
                goto LABEL_19;
              v65 = v8;
            }
            v66 = v70 + 1;
            if ( v105 == v66 )
            {
              if ( v65 )
                goto LABEL_39;
              goto LABEL_19;
            }
          }
        }
      }
      goto LABEL_19;
    }
    v101 = v12;
    v45 = sub_C33340();
    v12 = v101;
    v28 = v91;
    if ( *((void **)v86 + 3) == v45 )
      v46 = (_BYTE *)*((_QWORD *)v86 + 4);
    else
      v46 = v86 + 24;
    v29 = (v46[20] & 7) == 3;
  }
  if ( v29 )
  {
LABEL_39:
    if ( *((_BYTE **)v28 - 4) == v17 )
    {
      v17 = v108;
      return sub_AD8DD0(*((_QWORD *)v17 + 1), -1.0);
    }
  }
LABEL_19:
  if ( (a3 & 4) != 0 )
  {
    v18 = v107;
    if ( *v107 == 18 )
    {
      if ( *((void **)v18 + 3) == sub_C33340() )
        v19 = (_BYTE *)*((_QWORD *)v18 + 4);
      else
        v19 = v18 + 24;
      if ( (v19[20] & 7) == 3 )
        return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)v18 + 1));
    }
    else
    {
      v47 = *((_QWORD *)v107 + 1);
      v102 = v12;
      v48 = (unsigned int)*(unsigned __int8 *)(v47 + 8) - 17;
      if ( (unsigned int)v48 <= 1 && *v107 <= 0x15u )
      {
        v49 = (void **)sub_AD7630((__int64)v107, 0, v48);
        v50 = v102;
        v51 = v49;
        if ( v49 && *(_BYTE *)v49 == 18 )
        {
          if ( v49[3] == sub_C33340() )
            v52 = (void **)v51[4];
          else
            v52 = v51 + 3;
          if ( (*((_BYTE *)v52 + 20) & 7) == 3 )
          {
LABEL_79:
            v18 = v107;
            return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)v18 + 1));
          }
        }
        else if ( *(_BYTE *)(v47 + 8) == 17 )
        {
          v106 = *(_DWORD *)(v47 + 32);
          if ( v106 )
          {
            v73 = 0;
            while ( 1 )
            {
              v95 = v50;
              v74 = (void **)sub_AD69F0(v18, v73);
              v75 = v74;
              if ( !v74 )
                break;
              v76 = *(_BYTE *)v74;
              v50 = v95;
              if ( v76 != 13 )
              {
                if ( v76 != 18 )
                  return (unsigned __int8 *)v9;
                v77 = v75[3] == sub_C33340() ? v75[4] : v75 + 3;
                if ( (v77[20] & 7) != 3 )
                  return (unsigned __int8 *)v9;
                v50 = v8;
              }
              if ( v106 == ++v73 )
              {
                if ( v50 )
                  goto LABEL_79;
                return (unsigned __int8 *)v9;
              }
            }
          }
        }
      }
    }
  }
  return (unsigned __int8 *)v9;
}
