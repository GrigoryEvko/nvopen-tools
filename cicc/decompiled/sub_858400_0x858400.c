// Function: sub_858400
// Address: 0x858400
//
_DWORD *__fastcall sub_858400(unsigned __int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // r9
  bool v4; // cf
  bool v5; // zf
  __int64 v6; // rcx
  const char *v7; // rdi
  _QWORD **v8; // rdx
  __int64 v9; // rsi
  char v10; // al
  bool v11; // cf
  bool v12; // zf
  __int64 v13; // rcx
  char *v14; // rdi
  unsigned int *v15; // rsi
  int v16; // r13d
  bool v17; // cf
  bool v18; // zf
  __int64 v19; // rcx
  char *v20; // rdi
  char v21; // al
  bool v22; // cf
  bool v23; // zf
  _DWORD *result; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // r14
  const char *v33; // rdi
  char v34; // al
  bool v35; // cf
  bool v36; // zf
  char *v37; // rdi
  char v38; // al
  bool v39; // cf
  bool v40; // zf
  __int64 v41; // rcx
  unsigned __int64 v42; // rdi
  bool v43; // cf
  bool v44; // zf
  __int64 v45; // rcx
  const char *v46; // rdi
  const char *v47; // rsi
  char v48; // al
  bool v49; // cf
  bool v50; // zf
  __int64 v51; // rcx
  const char *v52; // rdi
  const char *v53; // rsi
  char v54; // al
  bool v55; // cf
  bool v56; // zf
  __int64 v57; // rcx
  const char *v58; // rdi
  const char *v59; // rsi
  __int64 v60; // r8
  __int64 v61; // r9
  bool v62; // cf
  bool v63; // zf
  __int64 v64; // rcx
  char *v65; // rdi
  unsigned int *v66; // rsi
  char v67; // al
  bool v68; // cf
  bool v69; // zf
  __int64 v70; // rcx
  char *v71; // rdi
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  _QWORD *v80; // r14
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  unsigned __int16 v89; // ax
  int v90; // r13d
  _QWORD *v91; // rax
  _QWORD *v92; // r15
  _BYTE *v93; // rax
  _QWORD *v94; // rax
  _QWORD *v95; // rax
  unsigned __int8 v96; // al
  unsigned __int64 v97; // rdi
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rdx
  __int64 v103; // rcx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // [rsp+8h] [rbp-48h]
  _BYTE *v107; // [rsp+10h] [rbp-40h]
  _QWORD *v108; // [rsp+18h] [rbp-38h]

  sub_7C9660(a1);
  v4 = word_4F06418[0] == 0;
  v5 = word_4F06418[0] == 1;
  if ( word_4F06418[0] != 1 )
    goto LABEL_18;
  v6 = 14;
  v7 = "system_header";
  v8 = *(_QWORD ***)(qword_4D04A00 + 8);
  v9 = (__int64)v8;
  do
  {
    if ( !v6 )
      break;
    v4 = *(_BYTE *)v9 < *v7;
    v5 = *(_BYTE *)v9++ == *v7++;
    --v6;
  }
  while ( v5 );
  v10 = (!v4 && !v5) - v4;
  v11 = 0;
  v12 = v10 == 0;
  if ( !v10 )
  {
    *(_BYTE *)(a1 + 96) = 1;
    v29 = qword_4F064B0;
    v30 = qword_4F064B0[7];
    v31 = *(unsigned __int8 *)(v30 + 72);
    if ( (v31 & 0x40) == 0 )
    {
      v31 &= 4u;
      if ( (_DWORD)v31 )
      {
        v32 = qword_4F064B0[8];
        if ( v30 != v32 )
        {
          sub_729A00(v30, unk_4F06468 - 1);
          v29 = qword_4F064B0;
        }
        v30 = v32;
        v9 = unk_4F06468;
        sub_729880(
          v32,
          unk_4F06468,
          *((_DWORD *)v29 + 10),
          v29[1],
          0,
          0,
          v29 + 7,
          (*(_BYTE *)(v32 + 72) & 4) != 0,
          (*(_BYTE *)(v32 + 72) & 8) != 0,
          (*(_BYTE *)(v32 + 72) & 0x10) != 0,
          (*(_BYTE *)(v32 + 72) & 0x20) != 0,
          (*(_BYTE *)(v32 + 72) & 2) != 0,
          1);
        *((_BYTE *)qword_4F064B0 + 88) |= 2u;
      }
    }
    sub_7B8B50(v30, (unsigned int *)v9, v31, v6, v2, v3);
    if ( word_4F06418[0] != 9 )
    {
      v9 = (__int64)dword_4F07508;
      LOBYTE(v16) = 1;
      sub_684B30(0xEu, dword_4F07508);
      goto LABEL_15;
    }
LABEL_14:
    LOBYTE(v16) = 1;
    goto LABEL_15;
  }
  v13 = 11;
  v14 = "visibility";
  v15 = *(unsigned int **)(qword_4D04A00 + 8);
  do
  {
    if ( !v13 )
      break;
    v11 = *(_BYTE *)v15 < (unsigned __int8)*v14;
    v12 = *(_BYTE *)v15 == (unsigned __int8)*v14;
    v15 = (unsigned int *)((char *)v15 + 1);
    ++v14;
    --v13;
  }
  while ( v12 );
  v16 = (char)((!v11 && !v12) - v11);
  v17 = 0;
  v18 = v16 == 0;
  if ( !v16 )
  {
    sub_7B8B50((unsigned __int64)v14, v15, (__int64)v8, v13, v2, v3);
    v62 = word_4F06418[0] == 0;
    v63 = word_4F06418[0] == 1;
    if ( word_4F06418[0] != 1 )
    {
LABEL_53:
      v9 = (__int64)dword_4F07508;
      sub_684B30(0x68Cu, dword_4F07508);
      goto LABEL_15;
    }
    v64 = 5;
    v65 = "push";
    v66 = *(unsigned int **)(qword_4D04A00 + 8);
    do
    {
      if ( !v64 )
        break;
      v62 = *(_BYTE *)v66 < (unsigned __int8)*v65;
      v63 = *(_BYTE *)v66 == (unsigned __int8)*v65;
      v66 = (unsigned int *)((char *)v66 + 1);
      ++v65;
      --v64;
    }
    while ( v63 );
    v67 = (!v62 && !v63) - v62;
    v68 = 0;
    v69 = v67 == 0;
    if ( v67 )
    {
      v70 = 4;
      v71 = "pop";
      v9 = *(_QWORD *)(qword_4D04A00 + 8);
      do
      {
        if ( !v70 )
          break;
        v68 = *(_BYTE *)v9 < (unsigned __int8)*v71;
        v69 = *(_BYTE *)v9++ == (unsigned __int8)*v71++;
        --v70;
      }
      while ( v69 );
      if ( (!v68 && !v69) != v68 )
        goto LABEL_53;
      sub_5D09C0(0);
      sub_7B8B50(0, (unsigned int *)v9, v72, v73, v74, v75);
      *(_BYTE *)(a1 + 96) = 3;
LABEL_63:
      if ( word_4F06418[0] != 9 )
      {
        v9 = (__int64)dword_4F07508;
        sub_684B30(0xEu, dword_4F07508);
      }
      goto LABEL_15;
    }
    sub_7B8B50((unsigned __int64)v65, v66, *(_QWORD *)(qword_4D04A00 + 8), v64, v60, v61);
    if ( word_4F06418[0] != 27 )
    {
      v9 = (__int64)dword_4F07508;
      sub_684B30(0x7Du, dword_4F07508);
      goto LABEL_15;
    }
    sub_7B8B50((unsigned __int64)v65, v66, v76, v77, v78, v79);
    if ( word_4F06418[0] == 83 )
    {
      v96 = 4;
    }
    else if ( word_4F06418[0] == 158 )
    {
      v96 = 2;
    }
    else if ( word_4F06418[0] != 1 || (v96 = sub_5D0A30(*(const char **)(qword_4D04A00 + 8))) == 0 )
    {
      v9 = (__int64)dword_4F07508;
      sub_684B30(0x68Du, dword_4F07508);
      sub_7B8B50(0x68Du, dword_4F07508, v81, v82, v83, v84);
      if ( word_4F06418[0] == 28 )
      {
        sub_7B8B50(0x68Du, dword_4F07508, v85, v86, v87, v88);
        LOBYTE(v16) = 0;
        goto LABEL_15;
      }
      goto LABEL_73;
    }
    *(_BYTE *)(a1 + 97) = v96;
    v9 = 0;
    v97 = v96;
    *(_BYTE *)(a1 + 96) = 2;
    sub_5D0960(v96, 0);
    sub_7B8B50(v97, 0, v98, v99, v100, v101);
    if ( word_4F06418[0] == 28 )
    {
      sub_7B8B50(v97, 0, v102, v103, v104, v105);
      goto LABEL_63;
    }
LABEL_73:
    v9 = (__int64)dword_4F07508;
    LOBYTE(v16) = 0;
    sub_684B30(0x12u, dword_4F07508);
    goto LABEL_15;
  }
  v19 = 11;
  v20 = "diagnostic";
  v9 = *(_QWORD *)(qword_4D04A00 + 8);
  do
  {
    if ( !v19 )
      break;
    v17 = *(_BYTE *)v9 < (unsigned __int8)*v20;
    v18 = *(_BYTE *)v9++ == (unsigned __int8)*v20++;
    --v19;
  }
  while ( v18 );
  v21 = (!v17 && !v18) - v17;
  v22 = 0;
  v23 = v21 == 0;
  if ( !v21 )
    goto LABEL_14;
  v19 = 6;
  v33 = "ivdep";
  v9 = *(_QWORD *)(qword_4D04A00 + 8);
  do
  {
    if ( !v19 )
      break;
    v22 = *(_BYTE *)v9 < *v33;
    v23 = *(_BYTE *)v9++ == *v33++;
    --v19;
  }
  while ( v23 );
  v34 = (!v22 && !v23) - v22;
  v35 = 0;
  v36 = v34 == 0;
  if ( !v34 )
    goto LABEL_14;
  v19 = 8;
  v37 = "warning";
  v9 = *(_QWORD *)(qword_4D04A00 + 8);
  do
  {
    if ( !v19 )
      break;
    v35 = *(_BYTE *)v9 < (unsigned __int8)*v37;
    v36 = *(_BYTE *)v9++ == (unsigned __int8)*v37++;
    --v19;
  }
  while ( v36 );
  v38 = (!v35 && !v36) - v35;
  v39 = 0;
  v40 = v38 == 0;
  if ( !v38 )
    goto LABEL_14;
  v41 = 7;
  v42 = (unsigned __int64)"target";
  v9 = *(_QWORD *)(qword_4D04A00 + 8);
  do
  {
    if ( !v41 )
      break;
    v39 = *(_BYTE *)v9 < *(_BYTE *)v42;
    v40 = *(_BYTE *)v9++ == *(_BYTE *)v42++;
    --v41;
  }
  while ( v40 );
  if ( (!v39 && !v40) == v39 )
  {
    *(_BYTE *)(a1 + 96) = 4;
    sub_7B8B50(v42, (unsigned int *)v9, (__int64)v8, v41, v2, v3);
    v80 = (_QWORD *)qword_4F04C50;
    if ( qword_4F04C50 )
    {
      v9 = (__int64)&dword_4F063F8;
      LOBYTE(v16) = 1;
      sub_6851C0(0xA3Bu, &dword_4F063F8);
    }
    else
    {
      v89 = word_4F06418[0];
      if ( word_4F06418[0] == 9 )
        goto LABEL_95;
      v108 = 0;
      v90 = 0;
      do
      {
        switch ( v89 )
        {
          case 0x1Bu:
            if ( v90 )
              goto LABEL_95;
            v90 = 1;
            break;
          case 0x1Cu:
            if ( !v90 )
              goto LABEL_94;
            v90 = 0;
            break;
          case 7u:
            if ( (unk_4F063A8 & 7) != 0 )
            {
              v9 = (__int64)&dword_4F063F8;
              LOBYTE(v16) = 1;
              sub_6851C0(0xA38u, &dword_4F063F8);
              goto LABEL_15;
            }
            v91 = sub_7276D0();
            *((_BYTE *)v91 + 10) = 1;
            v92 = v91;
            v91[3] = *(_QWORD *)&dword_4F063F8;
            v91[4] = qword_4F063F0;
            v106 = *(_QWORD *)word_4F063B0;
            v93 = sub_724830(*(_QWORD *)word_4F063B0 + 2LL);
            *v93 = 34;
            v42 = (unsigned __int64)(v93 + 1);
            v107 = v93;
            v9 = (__int64)qword_4F063B8;
            strcpy(v93 + 1, qword_4F063B8);
            v19 = (__int64)v107;
            v2 = v106;
            v107[v106] = 34;
            v107[v106 + 1] = 0;
            v92[5] = v107;
            if ( v80 )
            {
              v94 = v108;
              v108 = v92;
              *v94 = v92;
            }
            else
            {
              if ( !qword_4D03CB8 || (v80 = (_QWORD *)qword_4D03CB8[1]) == 0 )
              {
                v9 = (__int64)"GCC-target";
                v80 = sub_727670();
                v42 = dword_4F073B8[0];
                v80[2] = sub_724840(dword_4F073B8[0], "GCC-target");
              }
              *((_BYTE *)v80 + 8) = 61;
              v80[4] = v92;
              v108 = v92;
            }
            break;
          default:
            if ( v89 != 67 || !v80 )
              goto LABEL_95;
            break;
        }
        sub_7B8B50(v42, (unsigned int *)v9, (__int64)v8, v19, v2, v3);
        v89 = word_4F06418[0];
      }
      while ( word_4F06418[0] != 9 );
      if ( v90 )
      {
LABEL_94:
        v9 = (__int64)&dword_4F063F8;
        LOBYTE(v16) = 1;
        sub_6851C0(0xA39u, &dword_4F063F8);
        goto LABEL_15;
      }
      if ( !v80 )
      {
LABEL_95:
        v9 = (__int64)&dword_4F063F8;
        LOBYTE(v16) = 1;
        sub_6851C0(0x40Eu, &dword_4F063F8);
      }
      else
      {
        v95 = qword_4D03CB8;
        if ( !qword_4D03CB8 )
        {
          v95 = (_QWORD *)qword_4F5FC00;
          if ( qword_4F5FC00 )
          {
            v8 = *(_QWORD ***)qword_4F5FC00;
            qword_4F5FC00 = *(_QWORD *)qword_4F5FC00;
          }
          else
          {
            v95 = (_QWORD *)sub_823970(16);
          }
          *v95 = 0;
          v95[1] = 0;
          qword_4D03CB8 = v95;
        }
        v95[1] = v80;
        LOBYTE(v16) = 1;
      }
    }
    goto LABEL_15;
  }
  v43 = qword_4F077A8 < 0x9DCFu;
  v44 = qword_4F077A8 == 40399;
  if ( qword_4F077A8 <= 0x9DCFu )
    goto LABEL_18;
  v45 = 13;
  v46 = "push_options";
  v47 = *(const char **)(qword_4D04A00 + 8);
  do
  {
    if ( !v45 )
      break;
    v43 = *v47 < (unsigned int)*v46;
    v44 = *v47++ == *v46++;
    --v45;
  }
  while ( v44 );
  v48 = (!v43 && !v44) - v43;
  v49 = 0;
  v50 = v48 == 0;
  if ( v48 )
  {
    v51 = 12;
    v52 = "pop_options";
    v53 = *(const char **)(qword_4D04A00 + 8);
    do
    {
      if ( !v51 )
        break;
      v49 = *v53 < (unsigned int)*v52;
      v50 = *v53++ == *v52++;
      --v51;
    }
    while ( v50 );
    v54 = (!v49 && !v50) - v49;
    v55 = 0;
    v56 = v54 == 0;
    if ( v54 )
    {
      v57 = 14;
      v58 = "reset_options";
      v59 = *(const char **)(qword_4D04A00 + 8);
      do
      {
        if ( !v57 )
          break;
        v55 = *v59 < (unsigned int)*v58;
        v56 = *v59++ == *v58++;
        --v57;
      }
      while ( v56 );
      if ( (!v55 && !v56) == v55 )
      {
        v9 = 7;
        LOBYTE(v16) = 1;
        sub_856670(a1, (unsigned int *)7, v8, v57, v2, v3);
        goto LABEL_15;
      }
LABEL_18:
      sub_684B30(0x68Bu, dword_4F07508);
      sub_7C96B0(1u, dword_4F07508, v25, v26, v27, v28);
      return sub_8543B0((_QWORD *)a1, 0, 0);
    }
    v9 = 6;
    LOBYTE(v16) = 1;
    sub_856670(a1, (unsigned int *)6, v8, v51, v2, v3);
  }
  else
  {
    v9 = 5;
    LOBYTE(v16) = 1;
    sub_856670(a1, (unsigned int *)5, v8, v45, v2, v3);
  }
LABEL_15:
  sub_7C96B0(1u, (unsigned int *)v9, (__int64)v8, v19, v2, v3);
  sub_8543B0((_QWORD *)a1, 0, 0);
  result = *(_DWORD **)(a1 + 88);
  if ( result )
  {
    *((_BYTE *)result + 9) = v16;
    *((_WORD *)result + 28) = *(_WORD *)(a1 + 96);
  }
  return result;
}
