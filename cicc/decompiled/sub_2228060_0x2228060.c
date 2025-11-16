// Function: sub_2228060
// Address: 0x2228060
//
_QWORD *__fastcall sub_2228060(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  int v11; // r13d
  _QWORD *v12; // r12
  __int64 v13; // r9
  void *v14; // rsp
  bool v15; // r13
  char v16; // dl
  char v17; // si
  int v19; // r10d
  __int64 v20; // r13
  unsigned __int64 v21; // r14
  __int64 v22; // r12
  int v23; // ebx
  int *v24; // r15
  int *v25; // rax
  unsigned __int64 v26; // r15
  int *v27; // r14
  _QWORD *v28; // r13
  size_t v29; // rax
  unsigned __int64 v30; // r12
  size_t v31; // r13
  size_t v32; // rax
  unsigned __int64 v33; // r12
  _QWORD *v34; // r13
  unsigned __int64 v35; // rax
  int *v36; // rax
  int v37; // eax
  bool v38; // zf
  _QWORD *v39; // rax
  char v40; // r12
  char v41; // dl
  _QWORD *v42; // rdx
  unsigned __int64 v43; // r12
  __int64 v44; // rsi
  int v45; // r13d
  int v46; // eax
  int *v47; // rax
  unsigned __int64 v48; // rt0
  unsigned __int64 v49; // rax
  const wchar_t *v50; // rbx
  size_t v51; // r14
  size_t v52; // rax
  size_t v53; // r13
  const wchar_t *v54; // r13
  _QWORD *v55; // rcx
  char v56; // bl
  _QWORD *v57; // rdx
  unsigned __int64 v58; // rax
  int *v59; // rax
  int v60; // eax
  bool v61; // zf
  char v62; // si
  char v63; // r15
  wchar_t v64; // r15d
  int v65; // eax
  int *v66; // rax
  int *v67; // rax
  int v68; // eax
  bool v69; // zf
  _QWORD *v70; // rax
  __int64 v71; // rax
  int *v72; // rax
  int v73; // eax
  int *v74; // rax
  int v75; // eax
  bool v76; // zf
  _QWORD *v77; // rax
  int *v78; // rax
  int v79; // eax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  int *v83; // rax
  int v84; // eax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // [rsp+0h] [rbp-70h] BYREF
  _DWORD *v90; // [rsp+8h] [rbp-68h]
  __int64 v91; // [rsp+10h] [rbp-60h]
  _QWORD *v92; // [rsp+18h] [rbp-58h]
  __int64 v93; // [rsp+20h] [rbp-50h]
  _QWORD *v94; // [rsp+28h] [rbp-48h]
  _QWORD *v95; // [rsp+30h] [rbp-40h]
  unsigned __int64 v96; // [rsp+38h] [rbp-38h]

  v11 = a3;
  v12 = a2;
  v91 = a3;
  v96 = (unsigned __int64)a5;
  v90 = a6;
  v94 = a2;
  LODWORD(v95) = a3;
  v92 = a4;
  v13 = sub_2243120(a9 + 208);
  v14 = alloca(4 * a8 + 8);
  v15 = v11 == -1;
  v16 = v15 && a2 != 0;
  if ( v16 )
  {
    v72 = (int *)v94[2];
    if ( (unsigned __int64)v72 >= v94[3] )
    {
      v81 = *v94;
      HIBYTE(v89) = v15 && v94 != 0;
      v93 = v13;
      v73 = (*(__int64 (__fastcall **)(_QWORD *))(v81 + 72))(v94);
      v16 = HIBYTE(v89);
      v13 = v93;
      a2 = v94;
    }
    else
    {
      v73 = *v72;
    }
    v12 = 0;
    if ( v73 != -1 )
    {
      v12 = a2;
      v16 = 0;
    }
  }
  else
  {
    v16 = v15;
  }
  LOBYTE(v93) = (_DWORD)v96 == -1;
  v17 = v93 & (a4 != 0);
  if ( v17 )
  {
    v74 = (int *)a4[2];
    if ( (unsigned __int64)v74 >= a4[3] )
    {
      v80 = *a4;
      LOBYTE(v92) = v16;
      LOBYTE(v94) = v93 & (a4 != 0);
      v96 = v13;
      v75 = (*(__int64 (__fastcall **)(_QWORD *))(v80 + 72))(a4);
      v16 = (char)v92;
      v17 = (char)v94;
      v13 = v96;
    }
    else
    {
      v75 = *v74;
    }
    v76 = v75 == -1;
    v77 = 0;
    if ( !v76 )
      v77 = a4;
    v92 = v77;
    if ( !v76 )
      v17 = 0;
  }
  else
  {
    v17 = v93;
  }
  if ( v17 == v16 )
    goto LABEL_6;
  if ( v12 && v15 )
  {
    v78 = (int *)v12[2];
    if ( (unsigned __int64)v78 >= v12[3] )
    {
      v82 = *v12;
      v96 = v13;
      v79 = (*(__int64 (__fastcall **)(_QWORD *))(v82 + 72))(v12);
      v13 = v96;
      v19 = v79;
    }
    else
    {
      v19 = *v78;
      v79 = *v78;
    }
    if ( v79 == -1 )
      v12 = 0;
  }
  else
  {
    v19 = (int)v95;
  }
  if ( !a8 )
    goto LABEL_6;
  v94 = v12;
  v20 = 0;
  v21 = 0;
  v22 = v13;
  v96 = (unsigned __int64)&v89;
  v23 = v19;
  do
  {
    while ( **(_DWORD **)(a7 + 8 * v20) != v23
         && (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v22 + 48LL))(v22) != v23 )
    {
      if ( a8 == ++v20 )
        goto LABEL_17;
    }
    *(_DWORD *)(v96 + 4 * v21++) = v20++;
  }
  while ( a8 != v20 );
LABEL_17:
  v12 = v94;
  v24 = (int *)v96;
  v96 = 0;
  if ( v21 <= 1 )
    goto LABEL_45;
  v95 = v94;
  v25 = v24;
  v26 = v21;
  v27 = v25;
  do
  {
    v28 = (_QWORD *)*v27;
    v29 = wcslen(*(const wchar_t **)(a7 + 8LL * (_QWORD)v28));
    v94 = v28;
    v30 = 1;
    v31 = v29;
    do
    {
      v32 = wcslen(*(const wchar_t **)(a7 + 8LL * v27[v30]));
      if ( v31 > v32 )
        v31 = v32;
      ++v30;
    }
    while ( v30 < v26 );
    v33 = v31;
    v34 = v94;
    v35 = v95[2];
    if ( v35 >= v95[3] )
      (*(void (__fastcall **)(_QWORD *))(*v95 + 80LL))(v95);
    else
      v95[2] = v35 + 4;
    if ( ++v96 >= v33 )
      goto LABEL_67;
    v36 = (int *)v95[2];
    if ( (unsigned __int64)v36 >= v95[3] )
      v37 = (*(__int64 (__fastcall **)(_QWORD *))(*v95 + 72LL))(v95);
    else
      v37 = *v36;
    v38 = v37 == -1;
    v39 = 0;
    if ( !v38 )
      v39 = v95;
    v40 = v38;
    v95 = v39;
    v41 = v93 & (v92 != 0);
    if ( v41 )
    {
      v67 = (int *)v92[2];
      if ( (unsigned __int64)v67 >= v92[3] )
      {
        LOBYTE(v94) = v93 & (v92 != 0);
        v68 = (*(__int64 (__fastcall **)(_QWORD *))(*v92 + 72LL))(v92);
        v41 = (char)v94;
      }
      else
      {
        v68 = *v67;
      }
      v69 = v68 == -1;
      v70 = 0;
      if ( !v69 )
        v70 = v92;
      v92 = v70;
      if ( !v69 )
        v41 = 0;
    }
    else
    {
      v41 = v93;
    }
    if ( v40 == v41 )
    {
LABEL_67:
      v12 = v95;
      goto LABEL_68;
    }
    v42 = v95;
    v43 = 0;
    v44 = 4 * v96;
    while ( 1 )
    {
      v45 = *(_DWORD *)(*(_QWORD *)(a7 + 8LL * (_QWORD)v34) + v44);
      v46 = -1;
      if ( v42 )
      {
        v47 = (int *)v42[2];
        if ( (unsigned __int64)v47 >= v42[3] )
        {
          v71 = *v42;
          v94 = (_QWORD *)v44;
          v95 = v42;
          v46 = (*(__int64 (__fastcall **)(_QWORD *))(v71 + 72))(v42);
          v44 = (__int64)v94;
          v42 = v95;
        }
        else
        {
          v46 = *v47;
        }
        if ( v46 == -1 )
          v42 = 0;
      }
      if ( v45 == v46 )
        break;
      v27[v43] = v27[--v26];
      if ( v26 <= v43 )
        goto LABEL_43;
LABEL_35:
      v34 = (_QWORD *)v27[v43];
    }
    if ( v26 > ++v43 )
      goto LABEL_35;
LABEL_43:
    v95 = v42;
  }
  while ( v26 > 1 );
  LODWORD(v95) = -1;
  v12 = v42;
  v48 = v26;
  v24 = v27;
  v21 = v48;
LABEL_45:
  if ( v21 != 1 )
    goto LABEL_6;
  v49 = v12[2];
  if ( v49 >= v12[3] )
    (*(void (__fastcall **)(_QWORD *))(*v12 + 80LL))(v12);
  else
    v12[2] = v49 + 4;
  v50 = *(const wchar_t **)(a7 + 8LL * *v24);
  LODWORD(v94) = *v24;
  v51 = v96 + 1;
  v52 = wcslen(v50);
  v53 = v52;
  if ( v51 >= v52 )
  {
LABEL_107:
    if ( v53 != v51 )
      goto LABEL_68;
    LODWORD(v95) = -1;
    *v90 = (_DWORD)v94;
  }
  else
  {
    v54 = v50;
    v55 = v92;
    v56 = v93;
    v57 = (_QWORD *)v52;
    while ( 1 )
    {
      v59 = (int *)v12[2];
      if ( (unsigned __int64)v59 >= v12[3] )
      {
        v88 = *v12;
        v95 = v55;
        v96 = (unsigned __int64)v57;
        v60 = (*(__int64 (__fastcall **)(_QWORD *))(v88 + 72))(v12);
        v55 = v95;
        v57 = (_QWORD *)v96;
      }
      else
      {
        v60 = *v59;
      }
      v61 = v60 == -1;
      if ( v60 == -1 )
        v12 = 0;
      v62 = v60 == -1;
      v63 = v56 & (v55 != 0);
      if ( v63 )
      {
        v83 = (int *)v55[2];
        if ( (unsigned __int64)v83 >= v55[3] )
        {
          v86 = *v55;
          v93 = (__int64)v57;
          LOBYTE(v95) = v61;
          v96 = (unsigned __int64)v55;
          v84 = (*(__int64 (__fastcall **)(_QWORD *))(v86 + 72))(v55);
          v57 = (_QWORD *)v93;
          v62 = (char)v95;
          v55 = (_QWORD *)v96;
        }
        else
        {
          v84 = *v83;
        }
        if ( v84 == -1 )
          v55 = 0;
        else
          v63 = 0;
      }
      else
      {
        v63 = v56;
      }
      if ( v62 == v63 )
        break;
      v64 = v54[v51];
      v65 = -1;
      if ( v12 )
      {
        v66 = (int *)v12[2];
        if ( (unsigned __int64)v66 >= v12[3] )
        {
          v87 = *v12;
          v95 = v55;
          v96 = (unsigned __int64)v57;
          v65 = (*(__int64 (__fastcall **)(_QWORD *))(v87 + 72))(v12);
          v55 = v95;
          v57 = (_QWORD *)v96;
        }
        else
        {
          v65 = *v66;
        }
        if ( v65 == -1 )
          v12 = 0;
      }
      if ( v64 != v65 )
        break;
      v58 = v12[2];
      if ( v58 >= v12[3] )
      {
        v85 = *v12;
        v95 = v55;
        v96 = (unsigned __int64)v57;
        (*(void (__fastcall **)(_QWORD *))(v85 + 80))(v12);
        v55 = v95;
        v57 = (_QWORD *)v96;
      }
      else
      {
        v12[2] = v58 + 4;
      }
      if ( ++v51 >= (unsigned __int64)v57 )
      {
        v53 = (size_t)v57;
        goto LABEL_107;
      }
    }
LABEL_68:
    LODWORD(v95) = -1;
LABEL_6:
    *a10 |= 4u;
  }
  return v12;
}
