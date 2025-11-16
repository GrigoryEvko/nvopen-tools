// Function: sub_2B40190
// Address: 0x2b40190
//
__int64 __fastcall sub_2B40190(
        unsigned __int8 *****a1,
        int *a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        unsigned int a7)
{
  unsigned __int8 ***v9; // r11
  unsigned __int8 **v10; // r10
  __int64 v11; // r9
  unsigned __int8 **v12; // rsi
  unsigned __int8 **v15; // rax
  unsigned __int8 *v16; // rdi
  unsigned __int8 *v17; // rcx
  unsigned int v18; // r9d
  __int64 v20; // rax
  __int64 v21; // r9
  unsigned __int8 **v22; // r9
  unsigned __int8 **v23; // rax
  int *v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rax
  int *v27; // rdx
  int *v28; // rcx
  int v29; // eax
  __int64 v30; // r8
  unsigned int v31; // ebx
  int *v32; // rcx
  int *i; // rdx
  __int64 *v34; // rdx
  int v35; // edi
  __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // r10
  void ***v39; // r9
  __int64 v40; // r10
  void ***v41; // r11
  __int64 v42; // rdx
  __int64 v43; // r10
  void ***v44; // r10
  void **v45; // rdx
  void **v46; // rdx
  void **v47; // rdx
  unsigned __int64 v48; // r10
  void *v49; // rax
  size_t v50; // r11
  unsigned int v51; // eax
  unsigned __int8 ****v52; // rax
  unsigned __int8 ****v53; // rsi
  bool v54; // al
  char v55; // al
  int v56; // r8d
  unsigned int v57; // ebx
  unsigned int v58; // r8d
  int *v59; // rcx
  unsigned __int64 v60; // rsi
  __int64 v61; // rax
  unsigned __int64 v62; // rcx
  __int64 v63; // rdx
  char v64; // al
  int *v65; // rdi
  void ***v66; // [rsp+8h] [rbp-B8h]
  size_t n; // [rsp+10h] [rbp-B0h]
  int na; // [rsp+10h] [rbp-B0h]
  void *src; // [rsp+18h] [rbp-A8h]
  unsigned int srca; // [rsp+18h] [rbp-A8h]
  int v71; // [rsp+20h] [rbp-A0h]
  int v72; // [rsp+20h] [rbp-A0h]
  int v73; // [rsp+20h] [rbp-A0h]
  void ***v74; // [rsp+20h] [rbp-A0h]
  int v75; // [rsp+30h] [rbp-90h]
  int v76; // [rsp+38h] [rbp-88h]
  int v78; // [rsp+3Ch] [rbp-84h]
  _QWORD v79[2]; // [rsp+40h] [rbp-80h] BYREF
  int *v80; // [rsp+50h] [rbp-70h] BYREF
  __int64 v81; // [rsp+58h] [rbp-68h]
  _BYTE v82[96]; // [rsp+60h] [rbp-60h] BYREF

  v9 = **a1;
  v10 = *v9;
  v11 = *((unsigned int *)v9 + 2);
  v12 = &(*v9)[v11];
  if ( v10 == &v10[v11] )
    return 0;
  v15 = ***a1;
  v16 = 0;
  do
  {
    while ( 1 )
    {
      v17 = *v15;
      if ( (unsigned int)**v15 - 12 > 1 )
        break;
LABEL_3:
      if ( v12 == ++v15 )
        goto LABEL_10;
    }
    if ( v16 )
    {
      if ( v16 != v17 )
        return 0;
      goto LABEL_3;
    }
    ++v15;
    v16 = v17;
  }
  while ( v12 != v15 );
LABEL_10:
  if ( !v16 )
    return 0;
  v20 = (v11 * 8) >> 3;
  v21 = (v11 * 8) >> 5;
  if ( !v21 )
  {
LABEL_19:
    if ( v20 != 2 )
    {
      if ( v20 != 3 )
      {
        if ( v20 == 1 )
        {
LABEL_22:
          if ( **v10 == 12 )
            goto LABEL_23;
        }
        return 0;
      }
      if ( **v10 == 12 )
        goto LABEL_23;
      ++v10;
    }
    if ( **v10 == 12 )
      goto LABEL_23;
    ++v10;
    goto LABEL_22;
  }
  v22 = &v10[4 * v21];
  while ( **v10 != 12 )
  {
    if ( *v10[1] == 12 )
    {
      if ( v12 != v10 + 1 )
        goto LABEL_24;
      return 0;
    }
    if ( *v10[2] == 12 )
    {
      if ( v12 != v10 + 2 )
        goto LABEL_24;
      return 0;
    }
    if ( *v10[3] == 12 )
    {
      v10 += 3;
      break;
    }
    v10 += 4;
    if ( v22 == v10 )
    {
      v20 = v12 - v10;
      goto LABEL_19;
    }
  }
LABEL_23:
  if ( v12 == v10 )
    return 0;
LABEL_24:
  v23 = v9[23];
  v18 = 0;
  if ( *((_DWORD *)v23 + 62) != 2 )
    return v18;
  if ( (_BYTE)a7 )
    goto LABEL_26;
  v34 = (__int64 *)a1[1];
  v35 = *((_DWORD *)v9 + 48);
  v36 = *v34;
  v37 = *((unsigned int *)v34 + 2);
  v38 = (unsigned int)(*((_DWORD *)v23 + 50) + 1);
  v39 = (void ***)(*v34 + 8 * v38);
  v40 = 8 * (v37 - v38);
  v41 = &v39[(unsigned __int64)v40 / 8];
  v42 = v40 >> 5;
  v43 = v40 >> 3;
  if ( v42 <= 0 )
  {
LABEL_52:
    if ( v43 != 2 )
    {
      if ( v43 != 3 )
      {
        if ( v43 != 1 )
          goto LABEL_55;
        goto LABEL_109;
      }
      if ( v23 == (*v39)[23] && v35 != *((_DWORD *)*v39 + 48) )
        goto LABEL_56;
      ++v39;
    }
    if ( v23 == (*v39)[23] && v35 != *((_DWORD *)*v39 + 48) )
      goto LABEL_56;
    ++v39;
LABEL_109:
    if ( v23 == (*v39)[23] )
    {
      if ( v35 == *((_DWORD *)*v39 + 48) )
        v39 = v41;
      goto LABEL_56;
    }
LABEL_55:
    v39 = v41;
    goto LABEL_56;
  }
  v44 = &v39[4 * v42];
  while ( v23 != (*v39)[23] || v35 == *((_DWORD *)*v39 + 48) )
  {
    v45 = v39[1];
    if ( v23 == v45[23] && v35 != *((_DWORD *)v45 + 48) )
    {
      ++v39;
      break;
    }
    v46 = v39[2];
    if ( v23 == v46[23] && v35 != *((_DWORD *)v46 + 48) )
    {
      v39 += 2;
      break;
    }
    v47 = v39[3];
    if ( v23 == v47[23] && v35 != *((_DWORD *)v47 + 48) )
    {
      v39 += 3;
      break;
    }
    v39 += 4;
    if ( v44 == v39 )
    {
      v43 = v41 - v39;
      goto LABEL_52;
    }
  }
LABEL_56:
  if ( v39 == (void ***)(v36 + 8 * v37) )
    return a7;
  v48 = *((unsigned int *)*v39 + 2);
  v49 = **v39;
  v80 = (int *)v82;
  v81 = 0x600000000LL;
  v50 = 8 * v48;
  if ( v48 > 6 )
  {
    v76 = a5;
    v66 = v39;
    n = 8 * v48;
    src = v49;
    v73 = v48;
    sub_C8D5F0((__int64)&v80, v82, v48, 8u, a5, (__int64)v39);
    LODWORD(v48) = v73;
    v49 = src;
    v50 = n;
    v39 = v66;
    v65 = &v80[2 * (unsigned int)v81];
    LODWORD(a5) = v76;
  }
  else
  {
    if ( !v50 )
      goto LABEL_59;
    v65 = (int *)v82;
  }
  na = v48;
  srca = a5;
  v74 = v39;
  memcpy(v65, v49, v50);
  LODWORD(v50) = v81;
  LODWORD(v48) = na;
  a5 = srca;
  v39 = v74;
LABEL_59:
  LODWORD(v81) = v50 + v48;
  v51 = *((_DWORD *)*v39 + 38);
  if ( v51 )
  {
    v72 = a5;
    sub_2B0FC00((__int64)(*v39)[18], v51, (__int64)a1[2], v36, a5, (__int64)a1[2]);
    sub_2B38DA0((unsigned int *)&v80, (__int64)*a1[2]);
    LODWORD(a5) = v72;
  }
  v52 = a1[3];
  v71 = a5;
  v53 = *a1;
  v79[0] = &v80;
  v79[1] = v52;
  v54 = sub_2B400F0((__int64)v79, v53);
  LODWORD(a5) = v71;
  if ( !v54 )
  {
    if ( v80 != (int *)v82 )
      _libc_free((unsigned __int64)v80);
    return a7;
  }
  if ( v80 != (int *)v82 )
  {
    _libc_free((unsigned __int64)v80);
    LODWORD(a5) = v71;
  }
LABEL_26:
  if ( a4 > a3 )
  {
    v75 = a5;
    v55 = sub_B4EFF0(a2, a3, a4, (int *)v79);
    LODWORD(a5) = v75;
    if ( v55 && !LODWORD(v79[0]) )
      goto LABEL_67;
LABEL_28:
    v24 = &a2[a3];
    v25 = (__int64)(4 * a3) >> 4;
    v26 = (__int64)(4 * a3) >> 2;
    if ( v25 > 0 )
    {
      v27 = a2;
      v28 = &a2[4 * v25];
      while ( 1 )
      {
        v29 = *v27;
        if ( *v27 != -1 )
          goto LABEL_35;
        v29 = v27[1];
        if ( v29 != -1 )
          goto LABEL_35;
        v29 = v27[2];
        if ( v29 != -1 )
          goto LABEL_35;
        v29 = v27[3];
        if ( v29 != -1 )
          goto LABEL_35;
        v27 += 4;
        if ( v28 == v27 )
        {
          v26 = v24 - v27;
          goto LABEL_85;
        }
      }
    }
    v27 = a2;
LABEL_85:
    if ( v26 != 2 )
    {
      if ( v26 != 3 )
      {
        if ( v26 != 1 )
        {
          v29 = *v24;
LABEL_35:
          v30 = a6 * (unsigned int)a5;
          v31 = a3 - v30;
          if ( v31 > a6 )
            v31 = a6;
          v32 = &a2[v31 + (unsigned int)v30];
          for ( i = &a2[v30]; i != v32; ++i )
            *i = v29;
          return 1;
        }
LABEL_93:
        v29 = *v27;
        if ( *v27 == -1 )
          v29 = *v24;
        goto LABEL_35;
      }
      v29 = *v27;
      if ( *v27 != -1 )
        goto LABEL_35;
      ++v27;
    }
    v29 = *v27;
    if ( *v27 != -1 )
      goto LABEL_35;
    ++v27;
    goto LABEL_93;
  }
  if ( a4 != a3 )
    goto LABEL_28;
  v78 = a5;
  v64 = sub_B4ED80(a2, a3, a3);
  LODWORD(a5) = v78;
  if ( !v64 )
    goto LABEL_28;
LABEL_67:
  v56 = a6 * a5;
  v80 = a2;
  v57 = a3 - v56;
  if ( v57 > a6 )
    v57 = a6;
  sub_2B097A0(&v80, v57 + v56);
  v80 = a2;
  sub_2B097A0(&v80, v58);
  v60 = (unsigned __int64)v80;
  if ( v80 != v59 )
  {
    v61 = 0;
    v62 = (unsigned __int64)((char *)(v59 - 1) - (char *)v80) >> 2;
    do
    {
      v63 = v61;
      *(_DWORD *)(v60 + 4 * v61) = v61;
      ++v61;
    }
    while ( v63 != v62 );
  }
  return 1;
}
