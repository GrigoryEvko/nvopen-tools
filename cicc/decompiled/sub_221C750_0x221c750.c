// Function: sub_221C750
// Address: 0x221c750
//
char *__fastcall sub_221C750(
        __int64 a1,
        char *a2,
        __int64 a3,
        char *a4,
        int a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  int v12; // r12d
  __int64 v13; // r9
  void *v14; // rsp
  bool v15; // r12
  char v16; // dl
  char v17; // cl
  char v19; // r10
  __int64 v20; // r12
  unsigned __int64 v21; // r14
  __int64 v22; // r13
  char v23; // bl
  __int64 *v24; // r15
  __int64 *v25; // rax
  unsigned __int64 v26; // r15
  __int64 *v27; // r14
  __int64 *v28; // r12
  size_t v29; // rax
  unsigned __int64 v30; // r13
  size_t v31; // r12
  size_t v32; // rax
  unsigned __int64 v33; // r13
  __int64 *v34; // r12
  unsigned __int64 v35; // rax
  char v36; // r13
  char v37; // dl
  char *v38; // rdx
  __int64 *v39; // rcx
  unsigned __int64 i; // r13
  char v41; // r12
  _BYTE *v42; // rax
  unsigned __int64 v43; // rt0
  unsigned __int64 v44; // rax
  const char *v45; // rbx
  char *v46; // r12
  size_t v47; // rax
  char *v48; // rcx
  const char *v49; // r8
  char *v50; // r14
  char *v51; // rsi
  char v52; // bl
  char v53; // r15
  char v54; // dl
  char v55; // r15
  _BYTE *v56; // rax
  char v57; // dl
  __int64 v58; // rax
  bool v59; // zf
  __int64 v60; // rax
  int v61; // eax
  bool v62; // zf
  char *v63; // rax
  bool v64; // zf
  char *v65; // rax
  char *v66; // rax
  __int64 v67; // rax
  int v68; // eax
  bool v69; // zf
  char *v70; // rax
  __int64 v71; // rax
  int v72; // eax
  bool v73; // zf
  char *v74; // rax
  int v75; // eax
  bool v76; // zf
  char *v77; // rax
  __int64 v78; // rax
  int v79; // ecx
  __int64 v80; // rax
  int v81; // eax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rax
  int v85; // eax
  __int64 v86; // [rsp+0h] [rbp-70h] BYREF
  _DWORD *v87; // [rsp+8h] [rbp-68h]
  __int64 v88; // [rsp+10h] [rbp-60h]
  char *v89; // [rsp+18h] [rbp-58h]
  __int64 *v90; // [rsp+20h] [rbp-50h]
  const char *v91; // [rsp+28h] [rbp-48h]
  char *v92; // [rsp+30h] [rbp-40h]
  __int64 *v93; // [rsp+38h] [rbp-38h]

  v12 = a3;
  v89 = a4;
  v88 = a3;
  v87 = a6;
  LODWORD(v90) = a3;
  v92 = a2;
  v13 = sub_222F790(a9 + 208);
  v14 = alloca(4 * a8 + 8);
  v15 = v12 == -1;
  if ( v15 && a2 != 0 )
  {
    if ( *((_QWORD *)v92 + 2) >= *((_QWORD *)v92 + 3) )
    {
      v71 = *(_QWORD *)v92;
      LOBYTE(v91) = v15 && v92 != 0;
      v93 = (__int64 *)v13;
      v72 = (*(__int64 (__fastcall **)(char *))(v71 + 72))(v92);
      v16 = (char)v91;
      v13 = (__int64)v93;
      v73 = v72 == -1;
      if ( v72 != -1 )
        v16 = 0;
      v74 = 0;
      if ( !v73 )
        v74 = v92;
      v92 = v74;
    }
    else
    {
      v16 = 0;
    }
  }
  else
  {
    v16 = v15;
  }
  LOBYTE(v91) = a5 == -1;
  if ( ((unsigned __int8)v91 & (a4 != 0)) != 0 )
  {
    if ( *((_QWORD *)a4 + 2) >= *((_QWORD *)a4 + 3) )
    {
      v67 = *(_QWORD *)a4;
      HIBYTE(v86) = v16;
      LOBYTE(v89) = (unsigned __int8)v91 & (a4 != 0);
      v93 = (__int64 *)v13;
      v68 = (*(__int64 (__fastcall **)(char *))(v67 + 72))(a4);
      v17 = (char)v89;
      v13 = (__int64)v93;
      v69 = v68 == -1;
      v16 = HIBYTE(v86);
      if ( v68 != -1 )
        v17 = 0;
      v70 = 0;
      if ( !v69 )
        v70 = a4;
      v89 = v70;
    }
    else
    {
      v17 = 0;
    }
  }
  else
  {
    v17 = (char)v91;
  }
  if ( v17 == v16 )
    goto LABEL_6;
  if ( v92 && v15 )
  {
    v66 = (char *)*((_QWORD *)v92 + 2);
    if ( (unsigned __int64)v66 >= *((_QWORD *)v92 + 3) )
    {
      v93 = (__int64 *)v13;
      v75 = (*(__int64 (__fastcall **)(char *))(*(_QWORD *)v92 + 72LL))(v92);
      v13 = (__int64)v93;
      v76 = v75 == -1;
      v19 = v75;
      if ( v75 == -1 )
        v19 = -1;
      v77 = 0;
      if ( !v76 )
        v77 = v92;
      v92 = v77;
    }
    else
    {
      v19 = *v66;
    }
  }
  else
  {
    v19 = (char)v90;
  }
  if ( !a8 )
  {
LABEL_6:
    *a10 |= 4u;
    return v92;
  }
  v93 = &v86;
  v20 = 0;
  v21 = 0;
  v22 = v13;
  v23 = v19;
  do
  {
    while ( **(_BYTE **)(a7 + 8 * v20) != v23
         && (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v22 + 16LL))(v22) != v23 )
    {
      if ( a8 == ++v20 )
        goto LABEL_17;
    }
    *((_DWORD *)v93 + v21++) = v20++;
  }
  while ( a8 != v20 );
LABEL_17:
  v24 = v93;
  v93 = 0;
  if ( v21 <= 1 )
    goto LABEL_40;
  v25 = v24;
  v26 = v21;
  v27 = v25;
  do
  {
    v28 = (__int64 *)*(int *)v27;
    v29 = strlen(*(const char **)(a7 + 8LL * (_QWORD)v28));
    v90 = v28;
    v30 = 1;
    v31 = v29;
    do
    {
      v32 = strlen(*(const char **)(a7 + 8LL * *((int *)v27 + v30)));
      if ( v31 > v32 )
        v31 = v32;
      ++v30;
    }
    while ( v30 < v26 );
    v33 = v31;
    v34 = v90;
    v35 = *((_QWORD *)v92 + 2);
    if ( v35 >= *((_QWORD *)v92 + 3) )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v92 + 80LL))(v92);
    else
      *((_QWORD *)v92 + 2) = v35 + 1;
    v93 = (__int64 *)((char *)v93 + 1);
    if ( (unsigned __int64)v93 >= v33 )
      goto LABEL_58;
    v36 = 0;
    if ( *((_QWORD *)v92 + 2) >= *((_QWORD *)v92 + 3) )
    {
      v64 = (*(unsigned int (__fastcall **)(char *))(*(_QWORD *)v92 + 72LL))(v92) == -1;
      v65 = 0;
      if ( !v64 )
        v65 = v92;
      v36 = v64;
      v92 = v65;
    }
    if ( ((unsigned __int8)v91 & (v89 != 0)) != 0 )
    {
      if ( *((_QWORD *)v89 + 2) < *((_QWORD *)v89 + 3) )
      {
        if ( !v36 )
          goto LABEL_58;
        goto LABEL_30;
      }
      v60 = *(_QWORD *)v89;
      LOBYTE(v90) = (unsigned __int8)v91 & (v89 != 0);
      v61 = (*(__int64 (__fastcall **)(char *))(v60 + 72))(v89);
      v37 = (char)v90;
      v62 = v61 == -1;
      if ( v61 != -1 )
        v37 = 0;
      v63 = 0;
      if ( !v62 )
        v63 = v89;
      v89 = v63;
    }
    else
    {
      v37 = (char)v91;
    }
    if ( v36 == v37 )
      goto LABEL_58;
LABEL_30:
    v38 = v92;
    v39 = v93;
    for ( i = 0; ; v34 = (__int64 *)*((int *)v27 + i) )
    {
      v41 = *((_BYTE *)v39 + *(_QWORD *)(a7 + 8LL * (_QWORD)v34));
      LOBYTE(v42) = -1;
      if ( v38 )
      {
        v42 = (_BYTE *)*((_QWORD *)v38 + 2);
        if ( (unsigned __int64)v42 >= *((_QWORD *)v38 + 3) )
        {
          v58 = *(_QWORD *)v38;
          v90 = v39;
          v92 = v38;
          LODWORD(v42) = (*(__int64 (__fastcall **)(char *))(v58 + 72))(v38);
          v38 = v92;
          v59 = (_DWORD)v42 == -1;
          if ( (_DWORD)v42 == -1 )
            LOBYTE(v42) = -1;
          if ( v59 )
            v38 = 0;
          v39 = v90;
        }
        else
        {
          LOBYTE(v42) = *v42;
        }
      }
      if ( v41 == (_BYTE)v42 )
        break;
      --v26;
      *((_DWORD *)v27 + i) = *((_DWORD *)v27 + v26);
      if ( v26 <= i )
        goto LABEL_38;
LABEL_32:
      ;
    }
    if ( v26 > ++i )
      goto LABEL_32;
LABEL_38:
    v92 = v38;
  }
  while ( v26 > 1 );
  LODWORD(v90) = -1;
  v43 = v26;
  v24 = v27;
  v21 = v43;
LABEL_40:
  if ( v21 != 1 )
    goto LABEL_6;
  v44 = *((_QWORD *)v92 + 2);
  if ( v44 >= *((_QWORD *)v92 + 3) )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v92 + 80LL))(v92);
  else
    *((_QWORD *)v92 + 2) = v44 + 1;
  v45 = *(const char **)(a7 + 8LL * *(int *)v24);
  LODWORD(v90) = *(_DWORD *)v24;
  v46 = (char *)v93 + 1;
  v47 = strlen(v45);
  v48 = (char *)v47;
  if ( (unsigned __int64)v46 < v47 )
  {
    v49 = v45;
    v50 = v92;
    v51 = v89;
    v52 = (char)v91;
    while ( 1 )
    {
      v57 = 0;
      if ( *((_QWORD *)v50 + 2) >= *((_QWORD *)v50 + 3) )
      {
        v84 = *(_QWORD *)v50;
        v91 = v48;
        v92 = v51;
        v93 = (__int64 *)v49;
        v85 = (*(__int64 (__fastcall **)(char *, char *, _QWORD))(v84 + 72))(v50, v51, 0);
        v48 = (char *)v91;
        v51 = v92;
        v49 = (const char *)v93;
        if ( v85 == -1 )
          v50 = 0;
        v57 = v85 == -1;
      }
      v53 = v52 & (v51 != 0);
      if ( v53 )
      {
        if ( *((_QWORD *)v51 + 2) >= *((_QWORD *)v51 + 3) )
        {
          v82 = *(_QWORD *)v51;
          v89 = v48;
          v91 = v49;
          LOBYTE(v92) = v57;
          v93 = (__int64 *)v51;
          v83 = (*(__int64 (__fastcall **)(char *))(v82 + 72))(v51);
          v51 = (char *)v93;
          v57 = (char)v92;
          v49 = v91;
          v48 = v89;
          if ( v83 == -1 )
            v51 = 0;
          else
            v53 = 0;
        }
        else
        {
          v53 = 0;
        }
      }
      else
      {
        v53 = v52;
      }
      if ( v53 == v57 )
      {
LABEL_96:
        v92 = v50;
        LODWORD(v90) = -1;
        goto LABEL_6;
      }
      v54 = v46[(_QWORD)v49];
      v55 = -1;
      if ( v50 )
      {
        v56 = (_BYTE *)*((_QWORD *)v50 + 2);
        if ( (unsigned __int64)v56 < *((_QWORD *)v50 + 3) )
        {
          if ( v54 != *v56 )
            goto LABEL_96;
          goto LABEL_50;
        }
        v80 = *(_QWORD *)v50;
        v89 = v48;
        v91 = v51;
        v92 = (char *)v49;
        LOBYTE(v93) = v54;
        v81 = (*(__int64 (__fastcall **)(char *))(v80 + 72))(v50);
        v48 = v89;
        v51 = (char *)v91;
        v49 = v92;
        v54 = (char)v93;
        if ( v81 == -1 )
          v50 = 0;
        else
          v55 = v81;
      }
      if ( v54 != v55 )
        goto LABEL_96;
      v56 = (_BYTE *)*((_QWORD *)v50 + 2);
      if ( (unsigned __int64)v56 >= *((_QWORD *)v50 + 3) )
      {
        v78 = *(_QWORD *)v50;
        v91 = v48;
        v92 = v51;
        v93 = (__int64 *)v49;
        (*(void (__fastcall **)(char *))(v78 + 80))(v50);
        v48 = (char *)v91;
        v51 = v92;
        v49 = (const char *)v93;
        goto LABEL_51;
      }
LABEL_50:
      *((_QWORD *)v50 + 2) = v56 + 1;
LABEL_51:
      if ( ++v46 == v48 )
      {
        v92 = v50;
        goto LABEL_99;
      }
    }
  }
  if ( v46 != (char *)v47 )
  {
LABEL_58:
    LODWORD(v90) = -1;
    goto LABEL_6;
  }
LABEL_99:
  v79 = (int)v90;
  LODWORD(v90) = -1;
  *v87 = v79;
  return v92;
}
