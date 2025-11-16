// Function: sub_221CDA0
// Address: 0x221cda0
//
_QWORD *__fastcall sub_221CDA0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        int a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  int v10; // r15d
  int v11; // r13d
  __int64 v14; // rax
  _QWORD *v15; // r9
  __int64 v16; // r10
  _BYTE *v17; // rdx
  void *v18; // rsp
  int *v19; // r14
  char v20; // cl
  char v21; // r13
  bool v22; // r8
  char v23; // bl
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // r12
  _QWORD *v26; // r13
  int v27; // eax
  _QWORD *v28; // r15
  int v29; // r9d
  bool v30; // cl
  char v31; // dl
  char v32; // si
  char v33; // r11
  unsigned __int64 v34; // rax
  __int64 v35; // rsi
  unsigned __int64 *v36; // rdx
  int *v37; // rcx
  unsigned __int64 v38; // rax
  _QWORD *v39; // r9
  int v40; // eax
  char *v41; // rax
  __int64 v43; // rax
  char v44; // cl
  __int64 v45; // rbx
  _QWORD *v46; // r15
  _BYTE *v47; // r13
  char v48; // r14
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // r13
  __int64 v51; // rbx
  void *v52; // rsp
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rax
  int v58; // eax
  bool v59; // zf
  _QWORD *v60; // rax
  __int64 v61; // rax
  int v62; // eax
  char *v63; // rax
  __int64 v64; // rax
  int v65; // eax
  bool v66; // zf
  _QWORD *v67; // rax
  __int64 v68; // rax
  int v69; // eax
  __int64 v70; // rax
  _BYTE v71[11]; // [rsp+0h] [rbp-70h] BYREF
  bool v72; // [rsp+Bh] [rbp-65h]
  int v73; // [rsp+Ch] [rbp-64h]
  _QWORD *v74; // [rsp+10h] [rbp-60h]
  _DWORD *v75; // [rsp+18h] [rbp-58h]
  _BYTE *v76; // [rsp+20h] [rbp-50h]
  __int64 v77; // [rsp+28h] [rbp-48h]
  _QWORD *v78; // [rsp+30h] [rbp-40h]
  _QWORD *v79; // [rsp+38h] [rbp-38h]
  __int64 v80; // [rsp+80h] [rbp+10h]
  __int64 v81; // [rsp+80h] [rbp+10h]
  __int64 v82; // [rsp+80h] [rbp+10h]
  __int64 v83; // [rsp+80h] [rbp+10h]
  __int64 v84; // [rsp+80h] [rbp+10h]
  __int64 v85; // [rsp+80h] [rbp+10h]
  __int64 v86; // [rsp+80h] [rbp+10h]

  v10 = a3;
  v11 = a3;
  v77 = a3;
  v75 = a6;
  v79 = a4;
  v78 = a2;
  v14 = sub_222F790(a9 + 208);
  v15 = a2;
  v16 = a7;
  v17 = (_BYTE *)v14;
  v18 = alloca(8 * a8 + 8);
  v19 = (int *)v71;
  v20 = v11 == -1;
  v21 = v20 & (a2 != 0);
  if ( v21 )
  {
    v20 = 0;
    if ( v78[2] >= v78[3] )
    {
      v53 = *v78;
      LOBYTE(v74) = 0;
      v76 = v17;
      v54 = (*(__int64 (__fastcall **)(_QWORD *))(v53 + 72))(v78);
      v15 = v78;
      v17 = v76;
      v20 = (char)v74;
      v16 = a7;
      if ( v54 == -1 )
      {
        v20 = v21;
        v15 = 0;
        v21 = 0;
      }
    }
  }
  v22 = a5 == -1;
  v23 = v22 && a4 != 0;
  if ( v23 )
  {
    if ( a4[2] >= a4[3] )
    {
      v64 = *a4;
      v84 = v16;
      LOBYTE(v73) = v22;
      LOBYTE(v74) = v20;
      v76 = v17;
      v78 = v15;
      v65 = (*(__int64 (__fastcall **)(_QWORD *))(v64 + 72))(a4);
      v15 = v78;
      v17 = v76;
      v66 = v65 == -1;
      v20 = (char)v74;
      v16 = v84;
      if ( v65 != -1 )
        v23 = 0;
      v67 = 0;
      v22 = v73;
      if ( !v66 )
        v67 = a4;
      v79 = v67;
    }
    else
    {
      v23 = 0;
    }
  }
  else
  {
    v23 = v22;
  }
  if ( v20 == v23 )
  {
    v24 = 0;
    v25 = 0;
    v26 = 0;
  }
  else
  {
    if ( v21 )
    {
      v63 = (char *)v15[2];
      if ( (unsigned __int64)v63 >= v15[3] )
      {
        v68 = *v15;
        v85 = v16;
        LOBYTE(v74) = v22;
        v76 = v17;
        v78 = v15;
        v69 = (*(__int64 (__fastcall **)(_QWORD *))(v68 + 72))(v15);
        v15 = v78;
        v16 = v85;
        v44 = v69;
        v22 = (char)v74;
        if ( v69 == -1 )
          v44 = -1;
        v17 = v76;
        if ( v69 == -1 )
          v15 = 0;
      }
      else
      {
        v44 = *v63;
      }
    }
    else
    {
      v44 = v10;
    }
    v24 = 2 * a8;
    if ( 2 * a8 )
    {
      v78 = (_QWORD *)(2 * a8);
      v25 = 0;
      v45 = v16;
      v73 = v10;
      v46 = 0;
      v47 = v17;
      v76 = v71;
      v48 = v44;
      v74 = v15;
      v72 = v22;
      do
      {
        while ( **(_BYTE **)(v45 + 8LL * (_QWORD)v46) != v48
             && (*(unsigned __int8 (__fastcall **)(_BYTE *))(*(_QWORD *)v47 + 16LL))(v47) != v48 )
        {
          v46 = (_QWORD *)((char *)v46 + 1);
          if ( v46 == v78 )
            goto LABEL_54;
        }
        *(_DWORD *)&v76[4 * v25++] = (_DWORD)v46;
        v46 = (_QWORD *)((char *)v46 + 1);
      }
      while ( v46 != v78 );
LABEL_54:
      v16 = v45;
      v15 = v74;
      v10 = v73;
      v24 = 0;
      v22 = v72;
      v19 = (int *)v76;
      v26 = 0;
      if ( v25 )
      {
        v49 = v74[2];
        if ( v49 >= v74[3] )
        {
          v70 = *v74;
          v86 = v16;
          LOBYTE(v76) = v72;
          v78 = v74;
          (*(void (__fastcall **)(_QWORD *))(v70 + 80))(v74);
          v16 = v86;
          v22 = (char)v76;
          v15 = v78;
        }
        else
        {
          v74[2] = v49 + 1;
        }
        v78 = v15;
        LOBYTE(v74) = v22;
        v50 = 0;
        v51 = v16;
        v52 = alloca(8 * v25 + 8);
        v76 = v71;
        do
        {
          *(_QWORD *)&v71[8 * v50] = strlen(*(const char **)(v51 + 8LL * v19[v50]));
          ++v50;
        }
        while ( v25 != v50 );
        v16 = v51;
        v15 = v78;
        v24 = v50;
        v22 = (char)v74;
        v26 = v76;
        v10 = -1;
        v25 = 1;
      }
    }
    else
    {
      v25 = 0;
      v26 = 0;
    }
  }
  v27 = v10;
  v28 = v15;
  v29 = v27;
LABEL_7:
  v30 = v29 == -1;
  if ( v30 && v28 != 0 )
  {
    if ( v28[2] >= v28[3] )
    {
      v61 = *v28;
      v83 = v16;
      LOBYTE(v73) = v22;
      LODWORD(v74) = v29;
      LOBYTE(v76) = v29 == -1;
      LOBYTE(v78) = v30 && v28 != 0;
      v62 = (*(__int64 (__fastcall **)(_QWORD *))(v61 + 72))(v28);
      v31 = (char)v78;
      v30 = (char)v76;
      v29 = (int)v74;
      v16 = v83;
      if ( v62 != -1 )
        v31 = 0;
      v22 = v73;
      if ( v62 == -1 )
        v28 = 0;
    }
    else
    {
      v31 = 0;
    }
  }
  else
  {
    v31 = v29 == -1;
  }
  if ( !v22 || v79 == 0 )
  {
    v32 = v22;
    goto LABEL_11;
  }
  if ( v79[2] >= v79[3] )
  {
    LOBYTE(v78) = v22 && v79 != 0;
    v57 = *v79;
    v82 = v16;
    v72 = v22;
    v73 = v29;
    LOBYTE(v74) = v30;
    LOBYTE(v76) = v31;
    v58 = (*(__int64 (__fastcall **)(_QWORD *))(v57 + 72))(v79);
    v32 = (char)v78;
    v31 = (char)v76;
    v59 = v58 == -1;
    v30 = (char)v74;
    v29 = v73;
    if ( v58 != -1 )
      v32 = 0;
    v60 = 0;
    if ( !v59 )
      v60 = v79;
    v16 = v82;
    v22 = v72;
    v79 = v60;
LABEL_11:
    if ( v32 == v31 )
      goto LABEL_29;
LABEL_12:
    if ( v28 && v30 )
    {
      v41 = (char *)v28[2];
      if ( (unsigned __int64)v41 < v28[3] )
      {
        v33 = *v41;
        if ( !v24 )
        {
LABEL_38:
          v39 = v28;
          goto LABEL_39;
        }
        goto LABEL_16;
      }
      v55 = *v28;
      v81 = v16;
      LOBYTE(v76) = v22;
      LODWORD(v78) = v29;
      v56 = (*(__int64 (__fastcall **)(_QWORD *))(v55 + 72))(v28);
      v16 = v81;
      v22 = (char)v76;
      v33 = v56;
      if ( v56 == -1 )
      {
        v33 = -1;
        v28 = 0;
      }
    }
    else
    {
      v33 = v29;
    }
    if ( !v24 )
      goto LABEL_38;
LABEL_16:
    v34 = 0;
    v35 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v36 = &v26[v34];
        if ( *v36 > v25 )
          break;
        ++v35;
        ++v34;
LABEL_18:
        if ( v24 <= v34 )
          goto LABEL_22;
      }
      v37 = &v19[v34];
      if ( *(_BYTE *)(*(_QWORD *)(v16 + 8LL * *v37) + v25) == v33 )
      {
        ++v34;
        goto LABEL_18;
      }
      *v37 = v19[--v24];
      *v36 = v26[v24];
      if ( v24 <= v34 )
      {
LABEL_22:
        if ( v35 == v24 )
          goto LABEL_29;
        v38 = v28[2];
        if ( v38 >= v28[3] )
        {
          v43 = *v28;
          v80 = v16;
          LOBYTE(v78) = v22;
          (*(void (__fastcall **)(_QWORD *))(v43 + 80))(v28);
          v16 = v80;
          v22 = (char)v78;
        }
        else
        {
          v28[2] = v38 + 1;
        }
        ++v25;
        v29 = -1;
        goto LABEL_7;
      }
    }
  }
  if ( v31 )
    goto LABEL_12;
LABEL_29:
  v39 = v28;
  if ( v24 == 1 )
  {
    if ( *v26 != v25 )
    {
LABEL_39:
      *a10 |= 4u;
      return v39;
    }
  }
  else if ( v24 != 2 || *v26 != v25 && v26[1] != v25 )
  {
    goto LABEL_39;
  }
  v40 = *v19;
  if ( *v19 >= (int)a8 )
    v40 = *v19 - a8;
  *v75 = v40;
  return v39;
}
