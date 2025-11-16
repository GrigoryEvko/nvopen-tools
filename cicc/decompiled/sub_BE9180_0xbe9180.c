// Function: sub_BE9180
// Address: 0xbe9180
//
void __fastcall sub_BE9180(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int16 v4; // cx
  __int64 v5; // rax
  const char *v6; // r13
  unsigned __int8 v7; // al
  const char *v8; // rax
  unsigned __int8 *v9; // rdx
  __int64 v10; // rdi
  const char *v11; // rax
  unsigned __int8 *v12; // r14
  unsigned __int8 v13; // al
  const char *v14; // r13
  __int64 v15; // rax
  char v16; // al
  char v17; // al
  __int64 v18; // rdx
  int v19; // ecx
  __int64 v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 *i; // rdx
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // r14
  bool v27; // zf
  __int64 v28; // r15
  __int64 *v29; // rax
  __int64 v30; // rcx
  __int64 *v31; // rdx
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // r13
  __int64 v36; // r9
  _BYTE *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  void *v40; // rdx
  __int64 v41; // rax
  _WORD *v42; // rdx
  __int64 v43; // rdi
  void *v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax
  _WORD *v47; // rdx
  __int64 v48; // r13
  _BYTE *v49; // rax
  __int64 v50; // rax
  char v51; // dl
  __int64 v52; // r13
  char v53; // dl
  __int64 v54; // r13
  __int64 v55; // r12
  __int64 v56; // r8
  _BYTE *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdi
  void *v60; // rdx
  __int64 v61; // rax
  _WORD *v62; // rdx
  void *v63; // rdx
  const char *v64; // rax
  __int64 v65; // r13
  _BYTE *v66; // rax
  _BYTE *v67; // rsi
  __int64 v68; // rdi
  _BYTE *v69; // rax
  bool v70; // al
  char v71; // cl
  _BYTE *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  void *v75; // rdx
  __int64 v76; // rax
  _WORD *v77; // rdx
  __int64 v78; // rdx
  __int64 v79; // [rsp+8h] [rbp-C8h]
  __int64 v80; // [rsp+20h] [rbp-B0h]
  __int64 v81; // [rsp+20h] [rbp-B0h]
  _BYTE *v82; // [rsp+28h] [rbp-A8h]
  _BYTE *v83[4]; // [rsp+30h] [rbp-A0h] BYREF
  char v84; // [rsp+50h] [rbp-80h]
  char v85; // [rsp+51h] [rbp-7Fh]
  char *v86; // [rsp+60h] [rbp-70h] BYREF
  __int64 v87; // [rsp+68h] [rbp-68h]
  char v88[16]; // [rsp+70h] [rbp-60h] BYREF
  char v89; // [rsp+80h] [rbp-50h]
  char v90; // [rsp+81h] [rbp-4Fh]

  v2 = a2;
  if ( sub_B2FC80(a2) && (*(_BYTE *)(a2 + 32) & 0xF) != 9 && (*(_BYTE *)(a2 + 32) & 0xF) != 0 )
  {
    v48 = *(_QWORD *)a1;
    v90 = 1;
    v86 = "Global is external, but doesn't have external or weak linkage!";
    v89 = 3;
    if ( v48 )
    {
      sub_CA0E80(&v86, v48);
      v49 = *(_BYTE **)(v48 + 32);
      if ( (unsigned __int64)v49 >= *(_QWORD *)(v48 + 24) )
      {
        sub_CB5D20(v48, 10);
      }
      else
      {
        *(_QWORD *)(v48 + 32) = v49 + 1;
        *v49 = 10;
      }
      v50 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 152) = 1;
      if ( !v50 )
        return;
LABEL_71:
      sub_BDBD80(a1, (_BYTE *)a2);
      return;
    }
    goto LABEL_112;
  }
  if ( (unsigned __int8)(*(_BYTE *)a2 - 2) <= 1u || !*(_BYTE *)a2 )
  {
    v4 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
    if ( v4 && (unsigned __int64)(1LL << ((unsigned __int8)v4 - 1)) > 0x100000000LL )
    {
      v90 = 1;
      v86 = "huge alignment values are unsupported";
      v89 = 3;
      sub_BDBF70((__int64 *)a1, (__int64)&v86);
      if ( !*(_QWORD *)a1 )
        return;
      goto LABEL_71;
    }
    if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    {
      a2 = 22;
      v5 = sub_B91C10(v2, 22);
      v6 = (const char *)v5;
      if ( !v5 )
      {
LABEL_23:
        if ( (*(_BYTE *)(v2 + 7) & 0x20) != 0 )
        {
          a2 = 21;
          v14 = (const char *)sub_B91C10(v2, 21);
          if ( v14 )
          {
            v15 = sub_AE4450(*(_QWORD *)(a1 + 136), *(_QWORD *)(v2 + 8));
            a2 = v2;
            sub_BE8600((__int64 *)a1, (_BYTE *)v2, v14, v15, 1);
          }
        }
        goto LABEL_26;
      }
      v7 = *(_BYTE *)(v5 - 16);
      if ( (v7 & 2) != 0 )
      {
        if ( *((_DWORD *)v6 - 6) == 1 )
        {
          v8 = (const char *)*((_QWORD *)v6 - 4);
          goto LABEL_12;
        }
      }
      else if ( ((*((_WORD *)v6 - 8) >> 6) & 0xF) == 1 )
      {
        v8 = &v6[-8 * ((v7 >> 2) & 0xF) - 16];
LABEL_12:
        v9 = *(unsigned __int8 **)v8;
        if ( *(_QWORD *)v8 )
        {
          if ( (unsigned int)*v9 - 1 > 1 )
          {
            v90 = 1;
            v11 = "associated metadata must be ValueAsMetadata";
          }
          else
          {
            v10 = *((_QWORD *)v9 + 17);
            if ( *(_BYTE *)(*(_QWORD *)(v10 + 8) + 8LL) != 14 )
            {
              v90 = 1;
              v11 = "associated value must be pointer typed";
              goto LABEL_16;
            }
            v12 = sub_BD3BE0((unsigned __int8 *)v10, 22);
            v13 = *v12;
            if ( v13 != 0 && (unsigned __int8)(v13 - 2) > 1u && v13 > 0x15u )
            {
              v90 = 1;
              v86 = "associated metadata must point to a GlobalObject";
              v89 = 3;
              sub_BDBF70((__int64 *)a1, (__int64)&v86);
              if ( *(_QWORD *)a1 )
              {
                sub_BDBD80(a1, (_BYTE *)v2);
                sub_BDBD80(a1, v12);
              }
              return;
            }
            if ( (unsigned __int8 *)v2 != v12 )
              goto LABEL_23;
            v90 = 1;
            v11 = "global values should not associate to themselves";
          }
        }
        else
        {
          v90 = 1;
          v11 = "associated metadata must have a global value";
        }
LABEL_16:
        v86 = (char *)v11;
        v89 = 3;
        sub_BDBF70((__int64 *)a1, (__int64)&v86);
        if ( *(_QWORD *)a1 )
        {
          sub_BDBD80(a1, (_BYTE *)v2);
          sub_BD9900((__int64 *)a1, v6);
        }
        return;
      }
      v90 = 1;
      v11 = "associated metadata must have one operand";
      goto LABEL_16;
    }
  }
LABEL_26:
  v16 = *(_BYTE *)(v2 + 32) & 0xF;
  if ( v16 == 6 )
  {
    v27 = *(_BYTE *)v2 == 3;
    v83[0] = (_BYTE *)v2;
    if ( !v27 )
    {
      v90 = 1;
      v64 = "Only global variables can have appending linkage!";
      goto LABEL_121;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v2 + 24) + 8LL) != 16 )
    {
      v90 = 1;
      v86 = "Only global arrays can have appending linkage!";
      v89 = 3;
      sub_BE1030((_BYTE *)a1, (__int64)&v86, v83);
      return;
    }
  }
  else if ( v16 == 1 )
  {
    goto LABEL_29;
  }
  if ( !sub_B2FC80(v2) )
    goto LABEL_30;
LABEL_29:
  if ( sub_B326A0(v2) )
  {
    v65 = *(_QWORD *)a1;
    v90 = 1;
    v86 = "Declaration may not be in a Comdat!";
    v89 = 3;
    if ( v65 )
    {
      sub_CA0E80(&v86, v65);
      v66 = *(_BYTE **)(v65 + 32);
      if ( (unsigned __int64)v66 >= *(_QWORD *)(v65 + 24) )
      {
        sub_CB5D20(v65, 10);
      }
      else
      {
        *(_QWORD *)(v65 + 32) = v66 + 1;
        *v66 = 10;
      }
      v67 = *(_BYTE **)a1;
      *(_BYTE *)(a1 + 152) = 1;
      if ( !v67 )
        return;
      if ( *(_BYTE *)v2 <= 0x1Cu )
      {
        sub_A5C020((_BYTE *)v2, (__int64)v67, 1, a1 + 16);
        v68 = *(_QWORD *)a1;
        v69 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
        if ( (unsigned __int64)v69 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          goto LABEL_111;
      }
      else
      {
        sub_A693B0(v2, v67, a1 + 16, 0);
        v68 = *(_QWORD *)a1;
        v69 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
        if ( (unsigned __int64)v69 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
LABEL_111:
          *(_QWORD *)(v68 + 32) = v69 + 1;
          *v69 = 10;
          return;
        }
      }
      sub_CB5D20(v68, 10);
      return;
    }
LABEL_112:
    *(_BYTE *)(a1 + 152) = 1;
    return;
  }
LABEL_30:
  v17 = *(_BYTE *)(v2 + 33) & 3;
  if ( v17 == 2 )
  {
    v18 = *(unsigned __int8 *)(v2 + 32);
    if ( (*(_BYTE *)(v2 + 32) & 0x30) != 0x10 )
      goto LABEL_33;
    v83[0] = (_BYTE *)v2;
    v64 = "dllexport GlobalValue must have default or protected visibility";
    v90 = 1;
LABEL_121:
    v86 = (char *)v64;
    v89 = 3;
    sub_BE0F30((_BYTE *)a1, (__int64)&v86, v83);
    return;
  }
  if ( v17 != 1 )
  {
    v18 = *(unsigned __int8 *)(v2 + 32);
LABEL_33:
    LOBYTE(v19) = v18 & 0xF;
    goto LABEL_34;
  }
  if ( (*(_BYTE *)(v2 + 32) & 0x30) != 0 )
  {
    v83[0] = (_BYTE *)v2;
    v64 = "dllimport GlobalValue must have default visibility";
    v90 = 1;
    goto LABEL_121;
  }
  if ( (*(_BYTE *)(v2 + 33) & 0x40) != 0 )
  {
    v83[0] = (_BYTE *)v2;
    v64 = "GlobalValue with DLLImport Storage is dso_local!";
    v90 = 1;
    goto LABEL_121;
  }
  v70 = sub_B2FC80(v2);
  v18 = *(unsigned __int8 *)(v2 + 32);
  v71 = *(_BYTE *)(v2 + 32);
  if ( !v70 )
  {
    LOBYTE(v19) = v71 & 0xF;
LABEL_119:
    if ( (_BYTE)v19 == 1 )
      goto LABEL_102;
    v83[0] = (_BYTE *)v2;
    v64 = "Global is marked as dllimport, but not external";
    v90 = 1;
    goto LABEL_121;
  }
  v19 = v71 & 0xF;
  if ( v19 && (_BYTE)v19 != 9 )
    goto LABEL_119;
LABEL_34:
  if ( (unsigned int)(unsigned __int8)v19 - 7 <= 1 )
    goto LABEL_35;
LABEL_102:
  v18 &= 0x30u;
  if ( (_DWORD)v18 && (_BYTE)v19 != 9 )
  {
LABEL_35:
    if ( (*(_BYTE *)(v2 + 33) & 0x40) == 0 )
    {
      v83[0] = (_BYTE *)v2;
      v64 = "GlobalValue with local linkage or non-default visibility must be dso_local!";
      v90 = 1;
      goto LABEL_121;
    }
  }
  if ( (*(_BYTE *)(v2 + 34) & 1) != 0 && (*(_BYTE *)sub_B31490(v2, a2, v18) & 4) != 0 )
  {
    sub_B32650((_BYTE *)v2, a2);
    if ( v78 )
    {
      v83[0] = (_BYTE *)v2;
      v64 = "tagged GlobalValue must not be in section.";
      v90 = 1;
      goto LABEL_121;
    }
  }
  v20 = a1 + 1568;
  if ( !*(_BYTE *)(a1 + 1596) )
    goto LABEL_74;
  v21 = *(__int64 **)(a1 + 1576);
  v22 = *(unsigned int *)(a1 + 1588);
  for ( i = &v21[v22]; i != v21; ++v21 )
  {
    if ( v2 == *v21 )
      return;
  }
  if ( (unsigned int)v22 < *(_DWORD *)(a1 + 1584) )
  {
    *(_DWORD *)(a1 + 1588) = v22 + 1;
    *i = v2;
    ++*(_QWORD *)(a1 + 1568);
  }
  else
  {
LABEL_74:
    sub_C8CC70(a1 + 1568, v2);
    if ( !v51 )
      return;
  }
  v24 = *(_QWORD *)(v2 + 16);
  v25 = (__int64)v88;
  v26 = a1;
  v86 = v88;
  v87 = 0x600000000LL;
  sub_BD9420((__int64 *)&v86, v88, v24, 0);
  v82 = (_BYTE *)v2;
  while ( (_DWORD)v87 )
  {
    v27 = *(_BYTE *)(v26 + 1596) == 0;
    v28 = *(_QWORD *)&v86[8 * (unsigned int)v87 - 8];
    LODWORD(v87) = v87 - 1;
    if ( v27 )
    {
LABEL_78:
      v25 = v28;
      sub_C8CC70(v20, v28);
      if ( v53 )
      {
        v32 = *(_BYTE *)v28;
        if ( *(_BYTE *)v28 <= 0x1Cu )
          goto LABEL_80;
LABEL_51:
        v33 = *(_QWORD *)(v28 + 40);
        v34 = *(_QWORD *)(v26 + 8);
        if ( v33 && (v35 = *(_QWORD *)(v33 + 72)) != 0 )
        {
          v80 = *(_QWORD *)(v35 + 40);
          if ( v80 != v34 )
          {
            v36 = *(_QWORD *)v26;
            v85 = 1;
            v83[0] = "Global is referenced in a different module!";
            v84 = 3;
            if ( !v36 )
              goto LABEL_77;
            v25 = v36;
            v79 = v36;
            sub_CA0E80(v83, v36);
            v37 = *(_BYTE **)(v79 + 32);
            if ( (unsigned __int64)v37 >= *(_QWORD *)(v79 + 24) )
            {
              v25 = 10;
              sub_CB5D20(v79, 10);
            }
            else
            {
              *(_QWORD *)(v79 + 32) = v37 + 1;
              *v37 = 10;
            }
            v38 = *(_QWORD *)v26;
            *(_BYTE *)(v26 + 152) = 1;
            if ( v38 )
            {
              sub_BDBD80(v26, v82);
              v39 = *(_QWORD *)v26;
              v40 = *(void **)(*(_QWORD *)v26 + 32LL);
              if ( *(_QWORD *)(*(_QWORD *)v26 + 24LL) - (_QWORD)v40 <= 0xDu )
              {
                v39 = sub_CB6200(v39, "; ModuleID = '", 14);
              }
              else
              {
                qmemcpy(v40, "; ModuleID = '", 14);
                *(_QWORD *)(v39 + 32) += 14LL;
              }
              v41 = sub_CB6200(v39, *(_QWORD *)(v34 + 168), *(_QWORD *)(v34 + 176));
              v42 = *(_WORD **)(v41 + 32);
              if ( *(_QWORD *)(v41 + 24) - (_QWORD)v42 <= 1u )
              {
                sub_CB6200(v41, "'\n", 2);
              }
              else
              {
                *v42 = 2599;
                *(_QWORD *)(v41 + 32) += 2LL;
              }
              sub_BDBD80(v26, (_BYTE *)v28);
              sub_BDBD80(v26, (_BYTE *)v35);
              v43 = *(_QWORD *)v26;
              v44 = *(void **)(*(_QWORD *)v26 + 32LL);
              if ( *(_QWORD *)(*(_QWORD *)v26 + 24LL) - (_QWORD)v44 <= 0xDu )
              {
                v43 = sub_CB6200(v43, "; ModuleID = '", 14);
              }
              else
              {
                qmemcpy(v44, "; ModuleID = '", 14);
                *(_QWORD *)(v43 + 32) += 14LL;
              }
              v45 = *(_QWORD *)(v80 + 176);
              v25 = *(_QWORD *)(v80 + 168);
              goto LABEL_65;
            }
          }
        }
        else
        {
          v52 = *(_QWORD *)v26;
          v85 = 1;
          v83[0] = "Global is referenced by parentless instruction!";
          v84 = 3;
          if ( !v52 )
            goto LABEL_77;
          v25 = v52;
          sub_CA0E80(v83, v52);
          v72 = *(_BYTE **)(v52 + 32);
          if ( (unsigned __int64)v72 >= *(_QWORD *)(v52 + 24) )
          {
            v25 = 10;
            sub_CB5D20(v52, 10);
          }
          else
          {
            *(_QWORD *)(v52 + 32) = v72 + 1;
            *v72 = 10;
          }
          v73 = *(_QWORD *)v26;
          *(_BYTE *)(v26 + 152) = 1;
          if ( v73 )
          {
            sub_BDBD80(v26, v82);
            v74 = *(_QWORD *)v26;
            v75 = *(void **)(*(_QWORD *)v26 + 32LL);
            if ( *(_QWORD *)(*(_QWORD *)v26 + 24LL) - (_QWORD)v75 <= 0xDu )
            {
              v74 = sub_CB6200(v74, "; ModuleID = '", 14);
            }
            else
            {
              qmemcpy(v75, "; ModuleID = '", 14);
              *(_QWORD *)(v74 + 32) += 14LL;
            }
            v76 = sub_CB6200(v74, *(_QWORD *)(v34 + 168), *(_QWORD *)(v34 + 176));
            v77 = *(_WORD **)(v76 + 32);
            if ( *(_QWORD *)(v76 + 24) - (_QWORD)v77 <= 1u )
            {
              sub_CB6200(v76, "'\n", 2);
            }
            else
            {
              *v77 = 2599;
              *(_QWORD *)(v76 + 32) += 2LL;
            }
            v25 = v28;
            sub_BDBD80(v26, (_BYTE *)v28);
          }
        }
      }
    }
    else
    {
      v29 = *(__int64 **)(v26 + 1576);
      v30 = *(unsigned int *)(v26 + 1588);
      v31 = &v29[v30];
      if ( v29 == v31 )
      {
LABEL_49:
        if ( (unsigned int)v30 >= *(_DWORD *)(v26 + 1584) )
          goto LABEL_78;
        *(_DWORD *)(v26 + 1588) = v30 + 1;
        *v31 = v28;
        ++*(_QWORD *)(v26 + 1568);
        v32 = *(_BYTE *)v28;
        if ( *(_BYTE *)v28 > 0x1Cu )
          goto LABEL_51;
LABEL_80:
        if ( v32 )
        {
          v25 = (__int64)&v86[8 * (unsigned int)v87];
          sub_BD9420((__int64 *)&v86, (char *)v25, *(_QWORD *)(v28 + 16), 0);
        }
        else
        {
          v54 = *(_QWORD *)(v28 + 40);
          v55 = *(_QWORD *)(v26 + 8);
          if ( v54 != v55 )
          {
            v56 = *(_QWORD *)v26;
            v85 = 1;
            v83[0] = "Global is used by function in a different module";
            v84 = 3;
            if ( v56 )
            {
              v25 = v56;
              v81 = v56;
              sub_CA0E80(v83, v56);
              v57 = *(_BYTE **)(v81 + 32);
              if ( (unsigned __int64)v57 >= *(_QWORD *)(v81 + 24) )
              {
                v25 = 10;
                sub_CB5D20(v81, 10);
              }
              else
              {
                *(_QWORD *)(v81 + 32) = v57 + 1;
                *v57 = 10;
              }
              v58 = *(_QWORD *)v26;
              *(_BYTE *)(v26 + 152) = 1;
              if ( v58 )
              {
                sub_BDBD80(v26, v82);
                v59 = *(_QWORD *)v26;
                v60 = *(void **)(*(_QWORD *)v26 + 32LL);
                if ( *(_QWORD *)(*(_QWORD *)v26 + 24LL) - (_QWORD)v60 <= 0xDu )
                {
                  v59 = sub_CB6200(v59, "; ModuleID = '", 14);
                }
                else
                {
                  qmemcpy(v60, "; ModuleID = '", 14);
                  *(_QWORD *)(v59 + 32) += 14LL;
                }
                v61 = sub_CB6200(v59, *(_QWORD *)(v55 + 168), *(_QWORD *)(v55 + 176));
                v62 = *(_WORD **)(v61 + 32);
                if ( *(_QWORD *)(v61 + 24) - (_QWORD)v62 <= 1u )
                {
                  sub_CB6200(v61, "'\n", 2);
                }
                else
                {
                  *v62 = 2599;
                  *(_QWORD *)(v61 + 32) += 2LL;
                }
                sub_BDBD80(v26, (_BYTE *)v28);
                v43 = *(_QWORD *)v26;
                v63 = *(void **)(*(_QWORD *)v26 + 32LL);
                if ( *(_QWORD *)(*(_QWORD *)v26 + 24LL) - (_QWORD)v63 <= 0xDu )
                {
                  v43 = sub_CB6200(v43, "; ModuleID = '", 14);
                }
                else
                {
                  qmemcpy(v63, "; ModuleID = '", 14);
                  *(_QWORD *)(v43 + 32) += 14LL;
                }
                v45 = *(_QWORD *)(v54 + 176);
                v25 = *(_QWORD *)(v54 + 168);
LABEL_65:
                v46 = sub_CB6200(v43, v25, v45);
                v47 = *(_WORD **)(v46 + 32);
                if ( *(_QWORD *)(v46 + 24) - (_QWORD)v47 <= 1u )
                {
                  v25 = (__int64)"'\n";
                  sub_CB6200(v46, "'\n", 2);
                }
                else
                {
                  *v47 = 2599;
                  *(_QWORD *)(v46 + 32) += 2LL;
                }
              }
            }
            else
            {
LABEL_77:
              *(_BYTE *)(v26 + 152) = 1;
            }
          }
        }
      }
      else
      {
        while ( v28 != *v29 )
        {
          if ( v31 == ++v29 )
            goto LABEL_49;
        }
      }
    }
  }
  if ( v86 != v88 )
    _libc_free(v86, v25);
}
