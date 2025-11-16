// Function: sub_13D86C0
// Address: 0x13d86c0
//
_QWORD *__fastcall sub_13D86C0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, int a5)
{
  unsigned int v5; // r11d
  __int64 v6; // r15
  unsigned int v7; // r14d
  __int64 v9; // rbx
  _QWORD *v10; // r13
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // r10
  unsigned int v14; // r11d
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned int v17; // r11d
  __int64 v18; // r8
  char v19; // al
  _QWORD *result; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // ebx
  int v25; // eax
  bool v26; // al
  unsigned __int8 v27; // al
  unsigned int v28; // ebx
  __int64 v29; // rdi
  int v30; // eax
  bool v31; // al
  unsigned __int8 v32; // al
  unsigned int v33; // ebx
  bool v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r8
  unsigned int v40; // ebx
  __int64 v41; // rax
  unsigned int v42; // ebx
  int v43; // eax
  unsigned int v44; // ebx
  __int64 v45; // rax
  char v46; // dl
  int v47; // eax
  unsigned int v48; // ebx
  __int64 v49; // rax
  char v50; // dl
  int v51; // eax
  bool v52; // al
  unsigned int v53; // ebx
  __int64 v54; // rax
  char v55; // dl
  int v56; // eax
  bool v57; // al
  __int64 v58; // [rsp-50h] [rbp-50h]
  __int64 v59; // [rsp-50h] [rbp-50h]
  unsigned int v60; // [rsp-48h] [rbp-48h]
  unsigned int v61; // [rsp-48h] [rbp-48h]
  unsigned int v62; // [rsp-48h] [rbp-48h]
  unsigned int v63; // [rsp-48h] [rbp-48h]
  int v64; // [rsp-48h] [rbp-48h]
  int v65; // [rsp-48h] [rbp-48h]
  int v66; // [rsp-48h] [rbp-48h]
  unsigned int v67; // [rsp-44h] [rbp-44h]
  unsigned int v68; // [rsp-44h] [rbp-44h]
  unsigned int v69; // [rsp-44h] [rbp-44h]
  int v70; // [rsp-44h] [rbp-44h]
  int v71; // [rsp-44h] [rbp-44h]
  int v72; // [rsp-44h] [rbp-44h]
  __int64 v73; // [rsp-40h] [rbp-40h]
  __int64 v74; // [rsp-40h] [rbp-40h]
  __int64 v75; // [rsp-40h] [rbp-40h]
  __int64 v76; // [rsp-40h] [rbp-40h]
  __int64 v77; // [rsp-40h] [rbp-40h]
  __int64 v78; // [rsp-40h] [rbp-40h]
  __int64 v79; // [rsp-40h] [rbp-40h]
  __int64 v80; // [rsp-40h] [rbp-40h]
  __int64 v81; // [rsp-40h] [rbp-40h]

  if ( !a5 )
    return 0;
  v5 = a1;
  v6 = a3;
  v7 = a5 - 1;
  v9 = a2;
  if ( *(_BYTE *)(a2 + 16) == 79 )
  {
    v6 = a2;
    v9 = a3;
  }
  else
  {
    v5 = sub_15FF5D0(a1);
  }
  v10 = *(_QWORD **)(v6 - 72);
  v11 = *(_QWORD *)(v6 - 48);
  v73 = *(_QWORD *)(v6 - 24);
  v67 = v5 - 32;
  if ( v5 - 32 > 9 )
  {
    v58 = *(_QWORD *)(v6 - 48);
    v60 = v5;
    v12 = sub_13D8D60(v5, v11, v9, 0, a4, v7);
    v13 = v58;
    v14 = v60;
    v15 = v12;
    if ( v10 != (_QWORD *)v12 )
      goto LABEL_6;
LABEL_18:
    v62 = v14;
    v22 = sub_15A0600(*v10);
    v14 = v62;
    v15 = v22;
    if ( v67 <= 9 )
      goto LABEL_8;
LABEL_19:
    v69 = v14;
    v23 = sub_13D8D60(v14, v73, v9, 0, a4, v7);
    v17 = v69;
    v18 = v23;
    if ( v10 == (_QWORD *)v23 )
      goto LABEL_20;
    goto LABEL_9;
  }
  v59 = *(_QWORD *)(v6 - 48);
  v61 = v5;
  v21 = sub_13D9330(v5, v11, v9, a4, v7);
  v14 = v61;
  v13 = v59;
  v15 = v21;
  if ( v10 == (_QWORD *)v21 )
    goto LABEL_18;
LABEL_6:
  if ( !v15 )
  {
    v63 = v14;
    if ( !sub_13CB8F0((__int64)v10, v14, v13, v9) )
      return 0;
    v36 = sub_15A0600(*v10);
    v14 = v63;
    v15 = v36;
  }
  if ( v67 > 9 )
    goto LABEL_19;
LABEL_8:
  v68 = v14;
  v16 = sub_13D9330(v14, v73, v9, a4, v7);
  v17 = v68;
  v18 = v16;
  if ( v10 == (_QWORD *)v16 )
    goto LABEL_20;
LABEL_9:
  if ( v18 )
  {
    if ( v15 != v18 )
      goto LABEL_11;
    return (_QWORD *)v15;
  }
  if ( !sub_13CB8F0((__int64)v10, v17, v73, v9) )
    return 0;
LABEL_20:
  v18 = sub_15A0640(*v10);
  if ( v15 == v18 )
    return (_QWORD *)v15;
LABEL_11:
  if ( (*(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16) != (*(_BYTE *)(*v10 + 8LL) == 16) )
    return 0;
  if ( *(_BYTE *)(v18 + 16) <= 0x10u )
  {
    v74 = v18;
    v19 = sub_1593BB0(v18);
    v18 = v74;
    if ( !v19 )
    {
      if ( *(_BYTE *)(v74 + 16) == 13 )
      {
        v24 = *(_DWORD *)(v74 + 32);
        if ( v24 <= 0x40 )
        {
          v26 = *(_QWORD *)(v74 + 24) == 0;
        }
        else
        {
          v25 = sub_16A57B0(v74 + 24);
          v18 = v74;
          v26 = v24 == v25;
        }
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v74 + 8LL) != 16 )
          goto LABEL_26;
        v41 = sub_15A1020(v74);
        v18 = v74;
        if ( !v41 || *(_BYTE *)(v41 + 16) != 13 )
        {
          v72 = *(_QWORD *)(*(_QWORD *)v74 + 32LL);
          if ( v72 )
          {
            v53 = 0;
            while ( 1 )
            {
              v81 = v18;
              v54 = sub_15A0A60(v18, v53);
              v18 = v81;
              if ( !v54 )
                goto LABEL_26;
              v55 = *(_BYTE *)(v54 + 16);
              if ( v55 != 9 )
              {
                if ( v55 != 13 )
                  goto LABEL_26;
                if ( *(_DWORD *)(v54 + 32) <= 0x40u )
                {
                  v57 = *(_QWORD *)(v54 + 24) == 0;
                }
                else
                {
                  v66 = *(_DWORD *)(v54 + 32);
                  v56 = sub_16A57B0(v54 + 24);
                  v18 = v81;
                  v57 = v66 == v56;
                }
                if ( !v57 )
                  goto LABEL_26;
              }
              if ( v72 == ++v53 )
                goto LABEL_14;
            }
          }
          goto LABEL_14;
        }
        v42 = *(_DWORD *)(v41 + 32);
        if ( v42 <= 0x40 )
        {
          v26 = *(_QWORD *)(v41 + 24) == 0;
        }
        else
        {
          v43 = sub_16A57B0(v41 + 24);
          v18 = v74;
          v26 = v42 == v43;
        }
      }
      if ( !v26 )
        goto LABEL_26;
    }
LABEL_14:
    v75 = v18;
    result = (_QWORD *)sub_13DF820(v10, v15, a4, v7);
    v18 = v75;
    if ( result )
      return result;
  }
LABEL_26:
  v27 = *(_BYTE *)(v15 + 16);
  if ( v27 == 13 )
  {
    v28 = *(_DWORD *)(v15 + 32);
    if ( v28 > 0x40 )
    {
      v76 = v18;
      v29 = v15 + 24;
LABEL_29:
      v30 = sub_16A57B0(v29);
      v18 = v76;
      v31 = v28 - 1 == v30;
      goto LABEL_30;
    }
    v31 = *(_QWORD *)(v15 + 24) == 1;
    goto LABEL_30;
  }
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 || v27 > 0x10u )
  {
LABEL_32:
    v32 = *(_BYTE *)(v18 + 16);
    if ( v32 == 13 )
    {
      v33 = *(_DWORD *)(v18 + 32);
      if ( v33 <= 0x40 )
        v34 = *(_QWORD *)(v18 + 24) == 1;
      else
        v34 = v33 - 1 == (unsigned int)sub_16A57B0(v18 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 16 || v32 > 0x10u )
        return 0;
      v78 = v18;
      v38 = sub_15A1020(v18);
      v39 = v78;
      if ( !v38 || *(_BYTE *)(v38 + 16) != 13 )
      {
        v71 = *(_QWORD *)(*(_QWORD *)v78 + 32LL);
        if ( !v71 )
        {
LABEL_36:
          if ( sub_13CD190(v15) )
          {
            v35 = sub_15A04A0(*v10);
            return (_QWORD *)sub_13DE280(v10, v35, a4, v7);
          }
          return 0;
        }
        v48 = 0;
        while ( 1 )
        {
          v80 = v39;
          v49 = sub_15A0A60(v39, v48);
          if ( !v49 )
            return 0;
          v50 = *(_BYTE *)(v49 + 16);
          v39 = v80;
          if ( v50 != 9 )
          {
            if ( v50 != 13 )
              return 0;
            if ( *(_DWORD *)(v49 + 32) <= 0x40u )
            {
              v52 = *(_QWORD *)(v49 + 24) == 1;
            }
            else
            {
              v65 = *(_DWORD *)(v49 + 32);
              v51 = sub_16A57B0(v49 + 24);
              v39 = v80;
              v52 = v65 - 1 == v51;
            }
            if ( !v52 )
              return 0;
          }
          if ( v71 == ++v48 )
            goto LABEL_36;
        }
      }
      v40 = *(_DWORD *)(v38 + 32);
      if ( v40 <= 0x40 )
        v34 = *(_QWORD *)(v38 + 24) == 1;
      else
        v34 = v40 - 1 == (unsigned int)sub_16A57B0(v38 + 24);
    }
    if ( v34 )
      goto LABEL_36;
    return 0;
  }
  v76 = v18;
  v37 = sub_15A1020(v15);
  v18 = v76;
  if ( !v37 || *(_BYTE *)(v37 + 16) != 13 )
  {
    v70 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
    if ( !v70 )
      goto LABEL_31;
    v44 = 0;
    while ( 1 )
    {
      v79 = v18;
      v45 = sub_15A0A60(v15, v44);
      v18 = v79;
      if ( !v45 )
        goto LABEL_32;
      v46 = *(_BYTE *)(v45 + 16);
      if ( v46 != 9 )
      {
        if ( v46 != 13 )
          goto LABEL_32;
        if ( *(_DWORD *)(v45 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v45 + 24) != 1 )
            goto LABEL_32;
        }
        else
        {
          v64 = *(_DWORD *)(v45 + 32);
          v47 = sub_16A57B0(v45 + 24);
          v18 = v79;
          if ( v47 != v64 - 1 )
            goto LABEL_32;
        }
      }
      if ( v70 == ++v44 )
        goto LABEL_31;
    }
  }
  v28 = *(_DWORD *)(v37 + 32);
  if ( v28 > 0x40 )
  {
    v29 = v37 + 24;
    goto LABEL_29;
  }
  v31 = *(_QWORD *)(v37 + 24) == 1;
LABEL_30:
  if ( !v31 )
    goto LABEL_32;
LABEL_31:
  v77 = v18;
  result = sub_13D7FB0((__int64)v10, v18, a4, v7);
  v18 = v77;
  if ( !result )
    goto LABEL_32;
  return result;
}
