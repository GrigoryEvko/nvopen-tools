// Function: sub_97A150
// Address: 0x97a150
//
__int64 __fastcall sub_97A150(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 *a5, char a6)
{
  __int64 v10; // rsi
  __int64 v11; // rax
  _BYTE *v12; // r9
  _QWORD *v13; // r15
  size_t v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r15
  int v17; // eax
  __int64 result; // rax
  char *v19; // r13
  char v20; // al
  __int64 *v21; // rdx
  _BYTE *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r12
  unsigned __int64 v25; // r9
  _QWORD *v26; // rax
  _BYTE *v27; // rdx
  int v28; // r8d
  _QWORD *k; // rdx
  unsigned __int64 v30; // r9
  int v31; // r8d
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  _QWORD *m; // rdx
  size_t v35; // r10
  _QWORD *v36; // r15
  _QWORD *v37; // rdi
  __int64 *v38; // rsi
  __int64 v39; // r8
  _QWORD *v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // rdi
  __int64 *v43; // rax
  __int64 v44; // r12
  __int64 v45; // rbx
  unsigned __int64 v46; // r13
  _QWORD *v47; // rax
  _BYTE *v48; // rdx
  _QWORD *i; // rdx
  unsigned __int64 v50; // r13
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  _QWORD *j; // rdx
  __int64 v54; // rbx
  __int64 v55; // r15
  __int64 v56; // rdx
  _BYTE *v57; // r14
  _QWORD *v58; // rax
  __int64 v59; // r13
  char v60; // al
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rbx
  __int64 v65; // [rsp+10h] [rbp-110h]
  __int64 v66; // [rsp+30h] [rbp-F0h]
  int v67; // [rsp+30h] [rbp-F0h]
  size_t n; // [rsp+38h] [rbp-E8h]
  size_t nc; // [rsp+38h] [rbp-E8h]
  size_t nb; // [rsp+38h] [rbp-E8h]
  int na; // [rsp+38h] [rbp-E8h]
  __int64 *src; // [rsp+40h] [rbp-E0h]
  _BYTE *srca; // [rsp+40h] [rbp-E0h]
  unsigned int v74; // [rsp+48h] [rbp-D8h]
  __int64 v75; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v76; // [rsp+48h] [rbp-D8h]
  _QWORD *v78; // [rsp+50h] [rbp-D0h]
  __int64 v79; // [rsp+50h] [rbp-D0h]
  __int64 v80; // [rsp+50h] [rbp-D0h]
  __int64 v82; // [rsp+58h] [rbp-C8h]
  __int64 v83; // [rsp+58h] [rbp-C8h]
  __int64 v84; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v85; // [rsp+68h] [rbp-B8h]
  _BYTE *v86; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+78h] [rbp-A8h]
  _BYTE v88[48]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD *v89; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v90; // [rsp+B8h] [rbp-68h]
  _QWORD v91[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v92; // [rsp+D0h] [rbp-50h]
  __int64 v93; // [rsp+D8h] [rbp-48h]
  __int64 v94; // [rsp+E0h] [rbp-40h]

  if ( (unsigned __int8)sub_A73ED0(a1 + 72, 23) || (v10 = 23, (unsigned __int8)sub_B49560(a1, 23)) )
  {
    v10 = 4;
    if ( !(unsigned __int8)sub_A73ED0(a1 + 72, 4) )
    {
      v10 = 4;
      if ( !(unsigned __int8)sub_B49560(a1, 4) )
        return 0;
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0 )
    return 0;
  v74 = *(_DWORD *)(a2 + 36);
  if ( !v74 )
  {
    if ( !a5 )
      return 0;
    v11 = sub_B43CA0(a1);
    v89 = v91;
    v12 = *(_BYTE **)(v11 + 232);
    v13 = (_QWORD *)v11;
    v14 = *(_QWORD *)(v11 + 240);
    if ( &v12[v14] && !v12 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v86 = *(_BYTE **)(v11 + 240);
    if ( v14 > 0xF )
    {
      nc = v14;
      srca = v12;
      v41 = sub_22409D0(&v89, &v86, 0);
      v12 = srca;
      v14 = nc;
      v89 = (_QWORD *)v41;
      v42 = (_QWORD *)v41;
      v91[0] = v86;
    }
    else
    {
      if ( v14 == 1 )
      {
        LOBYTE(v91[0]) = *v12;
        v15 = v91;
        goto LABEL_11;
      }
      if ( !v14 )
      {
        v15 = v91;
LABEL_11:
        v90 = v14;
        *((_BYTE *)v15 + v14) = 0;
        v92 = v13[33];
        v93 = v13[34];
        v94 = v13[35];
        if ( (unsigned int)(v92 - 42) > 1 )
        {
          if ( v89 != v91 )
            j_j___libc_free_0(v89, v91[0] + 1LL);
          v10 = a2;
          if ( !(unsigned __int8)sub_981210(*a5, a2, &v89) )
            return 0;
        }
        else if ( v89 != v91 )
        {
          v10 = v91[0] + 1LL;
          j_j___libc_free_0(v89, v91[0] + 1LL);
        }
        goto LABEL_14;
      }
      v42 = v91;
    }
    v10 = (__int64)v12;
    memcpy(v42, v12, v14);
    v14 = (size_t)v86;
    v15 = v89;
    goto LABEL_11;
  }
LABEL_14:
  v16 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  if ( !a6 )
  {
    v17 = *(unsigned __int8 *)(v16 + 8);
    if ( (unsigned int)(v17 - 17) <= 1 )
      LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
    if ( (unsigned __int8)v17 <= 3u || (_BYTE)v17 == 5 || (v17 & 0xFD) == 4 )
      return 0;
  }
  v19 = (char *)sub_BD5D20(a2);
  v20 = *(_BYTE *)(v16 + 8);
  src = v21;
  if ( v20 == 17 )
  {
    v22 = (_BYTE *)sub_B2BEC0(a2);
    return sub_9798C0(v19, (unsigned __int64)src, v74, v16, a3, a4, v22, a5, a1);
  }
  if ( v20 == 18 )
  {
    sub_B2BEC0(a2);
    if ( v74 == 1248 && *a3 && (unsigned __int8)sub_AC30F0(*a3) )
      return sub_AD6450(v16, v10, v23);
    return 0;
  }
  if ( v20 != 15 )
    return sub_978F70(v19, (unsigned __int64)src, v74, (_QWORD *)v16, a3, a4, a5, a1);
  sub_B2BEC0(a2);
  if ( v74 == 179 )
  {
    v43 = *(__int64 **)(v16 + 16);
    v44 = v43[1];
    v45 = *v43;
    if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 <= 1 )
      v44 = **(_QWORD **)(v44 + 16);
    if ( *(_BYTE *)(v45 + 8) == 17 )
    {
      v46 = *(unsigned int *)(v45 + 32);
      v47 = v88;
      v48 = v88;
      v86 = v88;
      v87 = 0x400000000LL;
      if ( v46 )
      {
        if ( v46 > 4 )
        {
          sub_C8D5F0(&v86, v88, v46, 8);
          v48 = v86;
          v47 = &v86[8 * (unsigned int)v87];
        }
        for ( i = &v48[8 * v46]; i != v47; ++v47 )
        {
          if ( v47 )
            *v47 = 0;
        }
        LODWORD(v87) = v46;
      }
      v50 = *(unsigned int *)(v45 + 32);
      v89 = v91;
      v90 = 0x400000000LL;
      if ( v50 )
      {
        v51 = v91;
        v52 = v91;
        if ( v50 > 4 )
        {
          sub_C8D5F0(&v89, v91, v50, 8);
          v52 = v89;
          v51 = &v89[(unsigned int)v90];
        }
        for ( j = &v52[v50]; j != v51; ++v51 )
        {
          if ( v51 )
            *v51 = 0;
        }
        LODWORD(v90) = v50;
      }
      v54 = *(unsigned int *)(v45 + 32);
      if ( (_DWORD)v54 )
      {
        v80 = v16;
        v55 = 0;
        while ( 1 )
        {
          v57 = (_BYTE *)sub_AD69F0(*a3, (unsigned int)v55);
          if ( *v57 == 13 )
          {
            v56 = sub_ACADE0(v44);
          }
          else if ( *v57 == 18 )
          {
            v57 = (_BYTE *)sub_96A160((__int64)v57, v44);
          }
          else
          {
            v56 = 0;
            v57 = 0;
          }
          v58 = v89;
          v38 = (__int64 *)v86;
          *(_QWORD *)&v86[8 * v55] = v57;
          v58[v55] = v56;
          result = *(_QWORD *)&v86[8 * v55];
          if ( !result )
            break;
          if ( v54 == ++v55 )
          {
            v16 = v80;
            goto LABEL_110;
          }
        }
      }
      else
      {
LABEL_110:
        v63 = sub_AD3730(v89, (unsigned int)v90);
        v38 = &v84;
        v84 = sub_AD3730(v86, (unsigned int)v87);
        v85 = v63;
        result = sub_AD24A0(v16, &v84, 2);
      }
      v37 = v89;
      if ( v89 == v91 )
        goto LABEL_63;
      goto LABEL_62;
    }
    v59 = *a3;
    v60 = *(_BYTE *)*a3;
    if ( v60 == 13 )
    {
      v61 = sub_ACADE0(v44);
LABEL_105:
      v90 = v61;
      v89 = (_QWORD *)v59;
      return sub_AD24A0(v16, &v89, 2);
    }
    if ( v60 == 18 )
    {
      v59 = sub_96A160(*a3, v44);
      if ( v59 )
        goto LABEL_105;
    }
    return 0;
  }
  if ( v74 != 326 )
    return sub_978F70(v19, (unsigned __int64)src, v74, (_QWORD *)v16, a3, a4, a5, a1);
  v24 = **(_QWORD **)(v16 + 16);
  v78 = (_QWORD *)v24;
  if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
    v78 = **(_QWORD ***)(v24 + 16);
  if ( *(_BYTE *)(v24 + 8) != 17 )
  {
    v89 = (_QWORD *)*a3;
    v75 = sub_975D30(v19, (unsigned __int64)src, 325, v78, &v89, a5, a1);
    v62 = sub_975D30(v19, (unsigned __int64)src, 63, v78, &v89, a5, a1);
    if ( v62 && v75 )
    {
      v89 = (_QWORD *)v75;
      v90 = v62;
      return sub_AD24A0(v16, &v89, 2);
    }
    return 0;
  }
  v25 = *(unsigned int *)(v24 + 32);
  v26 = v88;
  v27 = v88;
  v86 = v88;
  v28 = v25;
  v87 = 0x600000000LL;
  if ( v25 )
  {
    if ( v25 > 6 )
    {
      na = v25;
      v76 = v25;
      sub_C8D5F0(&v86, v88, v25, 8);
      v27 = v86;
      v28 = na;
      v25 = v76;
      v26 = &v86[8 * (unsigned int)v87];
    }
    for ( k = &v27[8 * v25]; k != v26; ++v26 )
    {
      if ( v26 )
        *v26 = 0;
    }
    LODWORD(v87) = v28;
  }
  v30 = *(unsigned int *)(v24 + 32);
  v89 = v91;
  v90 = 0x600000000LL;
  v31 = v30;
  if ( v30 )
  {
    v32 = v91;
    v33 = v91;
    if ( v30 > 6 )
    {
      v67 = v30;
      nb = v30;
      sub_C8D5F0(&v89, v91, v30, 8);
      v33 = v89;
      v31 = v67;
      v30 = nb;
      v32 = &v89[(unsigned int)v90];
    }
    for ( m = &v33[v30]; m != v32; ++v32 )
    {
      if ( v32 )
        *v32 = 0;
    }
    LODWORD(v90) = v31;
  }
  if ( *(_DWORD *)(v24 + 32) )
  {
    v66 = *(unsigned int *)(v24 + 32);
    v65 = v16;
    v35 = 0;
    v36 = v78;
    do
    {
      n = v35;
      v84 = sub_AD69F0(*a3, (unsigned int)v35);
      v38 = src;
      v79 = sub_975D30(v19, (unsigned __int64)src, 325, v36, &v84, a5, a1);
      v39 = sub_975D30(v19, (unsigned __int64)src, 63, v36, &v84, a5, a1);
      v40 = v89;
      *(_QWORD *)&v86[8 * n] = v79;
      v40[n] = v39;
      result = *(_QWORD *)&v86[8 * n];
      if ( !result )
      {
        v37 = v89;
        goto LABEL_61;
      }
      v37 = v89;
      result = v89[n];
      if ( !result )
        goto LABEL_61;
      v35 = n + 1;
    }
    while ( n + 1 != v66 );
    v16 = v65;
  }
  else
  {
    v37 = v89;
  }
  v64 = sub_AD3730(v37, (unsigned int)v90);
  v38 = &v84;
  v84 = sub_AD3730(v86, (unsigned int)v87);
  v85 = v64;
  result = sub_AD24A0(v16, &v84, 2);
  v37 = v89;
LABEL_61:
  if ( v37 != v91 )
  {
LABEL_62:
    v82 = result;
    _libc_free(v37, v38);
    result = v82;
  }
LABEL_63:
  if ( v86 != v88 )
  {
    v83 = result;
    _libc_free(v86, v38);
    return v83;
  }
  return result;
}
