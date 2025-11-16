// Function: sub_C81390
// Address: 0xc81390
//
char __fastcall sub_C81390(_QWORD *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v9; // r12d
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  unsigned __int8 v13; // al
  unsigned __int64 v14; // rax
  __int64 v15; // rbx
  _BYTE *v16; // r8
  _BYTE *v17; // rbx
  char v18; // al
  char *v19; // rax
  _BYTE *v20; // r13
  size_t v21; // r14
  char *v22; // rdi
  char *v23; // r10
  char *v24; // rdx
  size_t v25; // rdi
  __int64 v26; // rax
  size_t v27; // rax
  unsigned __int64 v28; // r9
  char *v29; // r10
  size_t v30; // r9
  char *v31; // r10
  size_t v32; // r14
  size_t v33; // rdx
  size_t v34; // rbx
  const char *v35; // r13
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  size_t v38; // r14
  const char *v39; // rbx
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  const char **v42; // rax
  size_t v43; // rax
  const char *v44; // r14
  __int64 v45; // rdx
  const char **v46; // rdx
  size_t v48; // rax
  const char *v49; // r8
  __int64 v50; // rdx
  const char **v51; // rdx
  const char **v52; // rax
  const char **v53; // rax
  const char **v54; // rax
  char *s; // [rsp+8h] [rbp-1F8h]
  char v57; // [rsp+17h] [rbp-1E9h]
  _QWORD *v58; // [rsp+40h] [rbp-1C0h]
  size_t v59; // [rsp+40h] [rbp-1C0h]
  char *v60; // [rsp+48h] [rbp-1B8h]
  char *v61; // [rsp+48h] [rbp-1B8h]
  const char **v62; // [rsp+48h] [rbp-1B8h]
  size_t v63; // [rsp+48h] [rbp-1B8h]
  const char *v64; // [rsp+48h] [rbp-1B8h]
  _QWORD v65[4]; // [rsp+50h] [rbp-1B0h] BYREF
  __int16 v66; // [rsp+70h] [rbp-190h]
  const char *v67; // [rsp+80h] [rbp-180h] BYREF
  size_t v68; // [rsp+88h] [rbp-178h]
  __int64 v69; // [rsp+90h] [rbp-170h]
  _BYTE v70[40]; // [rsp+98h] [rbp-168h] BYREF
  const char *v71; // [rsp+C0h] [rbp-140h] BYREF
  size_t v72; // [rsp+C8h] [rbp-138h]
  __int64 v73; // [rsp+D0h] [rbp-130h]
  _BYTE v74[40]; // [rsp+D8h] [rbp-128h] BYREF
  const char *v75; // [rsp+100h] [rbp-100h] BYREF
  size_t v76; // [rsp+108h] [rbp-F8h]
  __int64 v77; // [rsp+110h] [rbp-F0h]
  _BYTE v78[40]; // [rsp+118h] [rbp-E8h] BYREF
  const char *v79; // [rsp+140h] [rbp-C0h] BYREF
  size_t v80; // [rsp+148h] [rbp-B8h]
  __int64 v81; // [rsp+150h] [rbp-B0h]
  _BYTE v82[40]; // [rsp+158h] [rbp-A8h] BYREF
  _BYTE *v83; // [rsp+180h] [rbp-80h] BYREF
  __int64 v84; // [rsp+188h] [rbp-78h]
  _BYTE v85[112]; // [rsp+190h] [rbp-70h] BYREF

  v9 = (unsigned int)a2;
  v67 = v70;
  v71 = v74;
  v75 = v78;
  v79 = v82;
  v83 = v85;
  v84 = 0x400000000LL;
  v11 = *(_BYTE *)(a3 + 32);
  v68 = 0;
  v69 = 32;
  v72 = 0;
  v73 = 32;
  v76 = 0;
  v77 = 32;
  v80 = 0;
  v81 = 32;
  if ( v11 > 1u )
  {
    if ( *(_BYTE *)(a3 + 33) != 1 || (unsigned __int8)(v11 - 3) > 3u )
    {
      a2 = (char *)&v67;
      sub_CA0EC0(a3, &v67);
      v50 = (unsigned int)v84;
      v49 = v67;
      v48 = v68;
      if ( HIDWORD(v84) < (unsigned __int64)(unsigned int)v84 + 1 )
      {
        a2 = v85;
        v59 = v68;
        v64 = v67;
        sub_C8D5F0(&v83, v85, (unsigned int)v84 + 1LL, 16);
        v50 = (unsigned int)v84;
        v48 = v59;
        v49 = v64;
      }
      goto LABEL_73;
    }
    if ( v11 == 4 )
    {
      v50 = 0;
      v49 = **(const char ***)a3;
      v48 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
      goto LABEL_73;
    }
    if ( v11 <= 4u )
    {
      if ( v11 != 3 )
        goto LABEL_102;
      v49 = *(const char **)a3;
      v48 = 0;
      if ( *(_QWORD *)a3 )
      {
        v62 = *(const char ***)a3;
        v48 = strlen(*(const char **)a3);
        v49 = (const char *)v62;
        v50 = 0;
        goto LABEL_73;
      }
    }
    else
    {
      v48 = *(_QWORD *)(a3 + 8);
      v49 = *(const char **)a3;
    }
    v50 = 0;
LABEL_73:
    v51 = (const char **)&v83[16 * v50];
    *v51 = v49;
    v51[1] = (const char *)v48;
    LODWORD(v84) = v84 + 1;
  }
  v12 = *(_BYTE *)(a4 + 32);
  if ( v12 > 1u )
  {
    if ( *(_BYTE *)(a4 + 33) == 1 && (unsigned __int8)(v12 - 3) <= 3u )
    {
      if ( v12 == 4 )
      {
        v54 = *(const char ***)a4;
        v44 = **(const char ***)a4;
        v43 = (size_t)v54[1];
      }
      else if ( v12 <= 4u )
      {
        if ( v12 != 3 )
          goto LABEL_102;
        v44 = *(const char **)a4;
        v43 = 0;
        if ( v44 )
          v43 = strlen(v44);
      }
      else
      {
        v43 = *(_QWORD *)(a4 + 8);
        v44 = *(const char **)a4;
      }
    }
    else
    {
      a2 = (char *)&v71;
      sub_CA0EC0(a4, &v71);
      v43 = v72;
      v44 = v71;
    }
    v45 = (unsigned int)v84;
    if ( (unsigned __int64)(unsigned int)v84 + 1 > HIDWORD(v84) )
    {
      a2 = v85;
      v63 = v43;
      sub_C8D5F0(&v83, v85, (unsigned int)v84 + 1LL, 16);
      v45 = (unsigned int)v84;
      v43 = v63;
    }
    v46 = (const char **)&v83[16 * v45];
    *v46 = v44;
    v46[1] = (const char *)v43;
    LODWORD(v84) = v84 + 1;
  }
  v13 = *(_BYTE *)(a5 + 32);
  if ( v13 > 1u )
  {
    if ( *(_BYTE *)(a5 + 33) == 1 && (unsigned __int8)(v13 - 3) <= 3u )
    {
      if ( v13 == 4 )
      {
        v52 = *(const char ***)a5;
        v39 = **(const char ***)a5;
        v38 = (size_t)v52[1];
      }
      else if ( v13 <= 4u )
      {
        if ( v13 != 3 )
          goto LABEL_102;
        v39 = *(const char **)a5;
        v38 = 0;
        if ( v39 )
          v38 = strlen(v39);
      }
      else
      {
        v38 = *(_QWORD *)(a5 + 8);
        v39 = *(const char **)a5;
      }
    }
    else
    {
      a2 = (char *)&v75;
      sub_CA0EC0(a5, &v75);
      v38 = v76;
      v39 = v75;
    }
    v40 = (unsigned int)v84;
    v41 = (unsigned int)v84 + 1LL;
    if ( v41 > HIDWORD(v84) )
    {
      a2 = v85;
      sub_C8D5F0(&v83, v85, v41, 16);
      v40 = (unsigned int)v84;
    }
    v42 = (const char **)&v83[16 * v40];
    *v42 = v39;
    v42[1] = (const char *)v38;
    LODWORD(v84) = v84 + 1;
  }
  LOBYTE(v14) = *(_BYTE *)(a6 + 32);
  if ( (unsigned __int8)v14 > 1u )
  {
    if ( *(_BYTE *)(a6 + 33) != 1 || (unsigned __int8)(v14 - 3) > 3u )
    {
      a2 = (char *)&v79;
      sub_CA0EC0(a6, &v79);
      v34 = v80;
      v35 = v79;
LABEL_52:
      v36 = (unsigned int)v84;
      v37 = (unsigned int)v84 + 1LL;
      if ( v37 > HIDWORD(v84) )
      {
        a2 = v85;
        sub_C8D5F0(&v83, v85, v37, 16);
        v36 = (unsigned int)v84;
      }
      v14 = (unsigned __int64)&v83[16 * v36];
      *(_QWORD *)(v14 + 8) = v34;
      *(_QWORD *)v14 = v35;
      LOBYTE(v14) = v84;
      v15 = (unsigned int)(v84 + 1);
      LODWORD(v84) = v84 + 1;
      goto LABEL_6;
    }
    if ( (_BYTE)v14 == 4 )
    {
      v53 = *(const char ***)a6;
      v35 = **(const char ***)a6;
      v34 = (size_t)v53[1];
      goto LABEL_52;
    }
    if ( (unsigned __int8)v14 > 4u )
    {
      v34 = *(_QWORD *)(a6 + 8);
      v35 = *(const char **)a6;
      goto LABEL_52;
    }
    if ( (_BYTE)v14 == 3 )
    {
      v35 = *(const char **)a6;
      v34 = 0;
      if ( v35 )
        v34 = strlen(v35);
      goto LABEL_52;
    }
LABEL_102:
    BUG();
  }
  v15 = (unsigned int)v84;
LABEL_6:
  v16 = v83;
  v17 = &v83[16 * v15];
  if ( v17 == v83 )
    goto LABEL_31;
  v18 = 47;
  if ( v9 == 3 )
    v18 = 92;
  v57 = v18;
  v19 = "/";
  if ( v9 > 1 )
    v19 = "\\/";
  v20 = v83;
  s = v19;
  v58 = a1 + 3;
  do
  {
    v14 = a1[1];
    if ( !v14 )
    {
      v21 = *((_QWORD *)v20 + 1);
      v22 = (char *)v21;
      if ( !v21 )
        goto LABEL_13;
LABEL_28:
      a2 = (char *)v9;
      LOBYTE(v14) = sub_C80220(**(_BYTE **)v20, v9);
      if ( (_BYTE)v14 )
      {
LABEL_29:
        v22 = (char *)a1[1];
        v21 = *((_QWORD *)v20 + 1);
LABEL_13:
        v23 = *(char **)v20;
        v24 = &v22[v21];
        if ( (unsigned __int64)&v22[v21] > a1[2] )
        {
LABEL_27:
          a2 = (char *)(a1 + 3);
          v60 = v23;
          LOBYTE(v14) = sub_C8D290(a1, v58, v24, 1);
          v22 = (char *)a1[1];
          v23 = v60;
        }
LABEL_14:
        if ( v21 )
        {
          a2 = v23;
          LOBYTE(v14) = (unsigned __int8)memcpy(&v22[*a1], v23, v21);
          v22 = (char *)a1[1];
        }
        v25 = (size_t)&v22[v21];
        goto LABEL_17;
      }
LABEL_21:
      v22 = (char *)a1[1];
      if ( !v22 )
        goto LABEL_26;
      a2 = (char *)v9;
      v66 = 261;
      v65[0] = *(_QWORD *)v20;
      v65[1] = *((_QWORD *)v20 + 1);
      LOBYTE(v14) = sub_C81280((__int64)v65, v9);
      if ( !(_BYTE)v14 )
      {
        v26 = a1[1];
        if ( (unsigned __int64)(v26 + 1) > a1[2] )
        {
          a2 = (char *)(a1 + 3);
          sub_C8D290(a1, v58, v26 + 1, 1);
          v26 = a1[1];
        }
        *(_BYTE *)(*a1 + v26) = v57;
        v14 = a1[1];
        v22 = (char *)(v14 + 1);
        a1[1] = v14 + 1;
LABEL_26:
        v21 = *((_QWORD *)v20 + 1);
        v23 = *(char **)v20;
        v24 = &v22[v21];
        if ( (unsigned __int64)&v22[v21] > a1[2] )
          goto LABEL_27;
        goto LABEL_14;
      }
      goto LABEL_29;
    }
    a2 = (char *)v9;
    LOBYTE(v14) = sub_C80220(*(_BYTE *)(*a1 + v14 - 1), v9);
    if ( !(_BYTE)v14 )
    {
      if ( !*((_QWORD *)v20 + 1) )
        goto LABEL_21;
      goto LABEL_28;
    }
    v27 = strlen(s);
    a2 = s;
    v14 = sub_C935B0(v20, s, v27, 0);
    v28 = *((_QWORD *)v20 + 1);
    v29 = *(char **)v20;
    if ( v14 > v28 )
    {
      v25 = a1[1];
      if ( v25 <= a1[2] )
        goto LABEL_17;
      v31 = &v29[v28];
      v33 = a1[1];
      v32 = 0;
    }
    else
    {
      v25 = a1[1];
      v30 = v28 - v14;
      v31 = &v29[v14];
      v32 = v30;
      v33 = v25 + v30;
      if ( v25 + v30 <= a1[2] )
      {
        if ( !v30 )
          goto LABEL_17;
LABEL_45:
        a2 = v31;
        LOBYTE(v14) = (unsigned __int8)memcpy((void *)(*a1 + v25), v31, v32);
        v25 = v32 + a1[1];
        goto LABEL_17;
      }
    }
    a2 = (char *)(a1 + 3);
    v61 = v31;
    LOBYTE(v14) = sub_C8D290(a1, v58, v33, 1);
    v25 = a1[1];
    v31 = v61;
    if ( v32 )
      goto LABEL_45;
LABEL_17:
    v20 += 16;
    a1[1] = v25;
  }
  while ( v17 != v20 );
  v16 = v83;
LABEL_31:
  if ( v16 != v85 )
    LOBYTE(v14) = _libc_free(v16, a2);
  if ( v79 != v82 )
    LOBYTE(v14) = _libc_free(v79, a2);
  if ( v75 != v78 )
    LOBYTE(v14) = _libc_free(v75, a2);
  if ( v71 != v74 )
    LOBYTE(v14) = _libc_free(v71, a2);
  if ( v67 != v70 )
    LOBYTE(v14) = _libc_free(v67, a2);
  return v14;
}
