// Function: sub_1B6ACD0
// Address: 0x1b6acd0
//
__int64 __fastcall sub_1B6ACD0(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  const char *v19; // rax
  const char *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  unsigned int v23; // r12d
  unsigned __int64 v25; // rdx
  const char *v26; // r8
  size_t v27; // r15
  size_t v28; // rax
  char *v29; // rdx
  _BYTE *v30; // rdi
  char *v31; // rax
  __int64 v32; // rdi
  unsigned __int64 v33; // rdx
  const char *v34; // r8
  size_t v35; // r15
  size_t v36; // rax
  char *v37; // rdx
  _BYTE *v38; // rdi
  unsigned __int64 v39; // rdx
  const char *v40; // r10
  size_t v41; // r8
  size_t v42; // rax
  char *v43; // rdx
  _BYTE *v44; // rdi
  char *v45; // rax
  __int64 v46; // rdi
  size_t v47; // rdx
  size_t v48; // r13
  size_t v49; // r14
  _BYTE *v50; // r12
  __int64 v51; // rax
  _QWORD *v52; // rbx
  __int64 v53; // rax
  size_t v54; // rdx
  size_t v55; // rdx
  __int64 v56; // rax
  _QWORD *v57; // rdi
  _BYTE *v58; // r12
  __int64 v59; // rax
  const void *v60; // r8
  _BYTE *v61; // rdi
  size_t v62; // r9
  _BYTE *v63; // rdi
  __int64 v64; // rax
  _QWORD *v65; // rdi
  __int64 v66; // rax
  _QWORD *v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // rax
  size_t n; // [rsp+10h] [rbp-1A0h]
  void *src; // [rsp+18h] [rbp-198h]
  char srca; // [rsp+18h] [rbp-198h]
  const char *srcb; // [rsp+18h] [rbp-198h]
  const char *srcc; // [rsp+18h] [rbp-198h]
  const char *srcd; // [rsp+18h] [rbp-198h]
  __int64 v81; // [rsp+28h] [rbp-188h]
  __int64 *v82; // [rsp+38h] [rbp-178h]
  _BYTE *v83; // [rsp+38h] [rbp-178h]
  _BYTE *v84; // [rsp+58h] [rbp-158h]
  size_t v85; // [rsp+58h] [rbp-158h]
  size_t v86; // [rsp+58h] [rbp-158h]
  size_t v87; // [rsp+58h] [rbp-158h]
  _QWORD v88[2]; // [rsp+60h] [rbp-150h] BYREF
  __int16 v89; // [rsp+70h] [rbp-140h]
  void *v90; // [rsp+80h] [rbp-130h]
  size_t v91; // [rsp+88h] [rbp-128h]
  _QWORD v92[2]; // [rsp+90h] [rbp-120h] BYREF
  void *v93; // [rsp+A0h] [rbp-110h]
  size_t v94; // [rsp+A8h] [rbp-108h]
  _QWORD v95[2]; // [rsp+B0h] [rbp-100h] BYREF
  void *dest; // [rsp+C0h] [rbp-F0h]
  size_t v97; // [rsp+C8h] [rbp-E8h]
  _QWORD v98[2]; // [rsp+D0h] [rbp-E0h] BYREF
  _QWORD v99[2]; // [rsp+E0h] [rbp-D0h] BYREF
  _QWORD v100[2]; // [rsp+F0h] [rbp-C0h] BYREF
  char *v101; // [rsp+100h] [rbp-B0h] BYREF
  size_t v102; // [rsp+108h] [rbp-A8h]
  _QWORD v103[2]; // [rsp+110h] [rbp-A0h] BYREF
  unsigned __int64 v104[2]; // [rsp+120h] [rbp-90h] BYREF
  _BYTE v105[32]; // [rsp+130h] [rbp-80h] BYREF
  unsigned __int64 v106[2]; // [rsp+150h] [rbp-60h] BYREF
  _BYTE v107[80]; // [rsp+160h] [rbp-50h] BYREF

  *(_BYTE *)(a4 + 76) = 0;
  v90 = v92;
  v93 = v95;
  dest = v98;
  v82 = (__int64 *)a2;
  v91 = 0;
  LOBYTE(v92[0]) = 0;
  v94 = 0;
  LOBYTE(v95[0]) = 0;
  v97 = 0;
  LOBYTE(v98[0]) = 0;
  sub_16FD380(a4, (unsigned __int64)a2);
  v9 = *(_QWORD *)(a4 + 80);
  v81 = a4;
  if ( v9 )
    v9 = a4;
  while ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 80);
    v106[1] = 0x2000000000LL;
    v104[1] = 0x2000000000LL;
    v106[0] = (unsigned __int64)v107;
    v104[0] = (unsigned __int64)v105;
    v14 = sub_16FD110(v10, (unsigned __int64)a2, v6, v7, v8);
    if ( *(_DWORD *)(v14 + 32) != 1 )
    {
      BYTE1(v103[0]) = 1;
      v20 = "descriptor Key must be a scalar";
      goto LABEL_9;
    }
    v15 = sub_16FD200(v10, (unsigned __int64)a2, v11, v12, v13);
    if ( *((_DWORD *)v15 + 8) != 1 )
    {
      v101 = "descriptor value must be a scalar";
      LOWORD(v103[0]) = 259;
      v22 = sub_16FD200(v10, (unsigned __int64)a2, v16, v17, v18);
      sub_16F8270(v82, (__int64)v22, (__int64)&v101);
      goto LABEL_11;
    }
    a2 = v104;
    src = v15;
    v19 = sub_16F8F10(v14, v104);
    v13 = (__int64)src;
    if ( v11 != 6 )
    {
      if ( v11 != 9 )
        goto LABEL_8;
      v12 = 0x726F66736E617274LL;
      if ( *(_QWORD *)v19 != 0x726F66736E617274LL || v19[8] != 109 )
        goto LABEL_8;
      a2 = v106;
      v26 = sub_16F8F10((__int64)src, v106);
      v27 = v25;
      if ( v26 )
      {
        v99[0] = v25;
        v28 = v25;
        v101 = (char *)v103;
        if ( v25 > 0xF )
        {
          srcd = v26;
          v66 = sub_22409D0(&v101, v99, 0);
          v26 = srcd;
          v101 = (char *)v66;
          v67 = (_QWORD *)v66;
          v103[0] = v99[0];
        }
        else
        {
          if ( v25 == 1 )
          {
            LOBYTE(v103[0]) = *v26;
            v29 = (char *)v103;
            goto LABEL_30;
          }
          if ( !v25 )
          {
            v29 = (char *)v103;
            goto LABEL_30;
          }
          v67 = v103;
        }
        a2 = (unsigned __int64 *)v26;
        memcpy(v67, v26, v27);
        v28 = v99[0];
        v29 = v101;
LABEL_30:
        v102 = v28;
        v29[v28] = 0;
        v30 = dest;
        v31 = (char *)dest;
        if ( v101 != (char *)v103 )
        {
          a2 = (unsigned __int64 *)v102;
          if ( dest != v98 )
          {
            v32 = v98[0];
            dest = v101;
            v97 = v102;
            v98[0] = v103[0];
            if ( !v31 )
              goto LABEL_33;
LABEL_44:
            v101 = v31;
            v103[0] = v32;
LABEL_45:
            v102 = 0;
            *v31 = 0;
            if ( v101 != (char *)v103 )
            {
              a2 = (unsigned __int64 *)(v103[0] + 1LL);
              j_j___libc_free_0(v101, v103[0] + 1LL);
            }
            goto LABEL_47;
          }
          dest = v101;
          v97 = v102;
          v98[0] = v103[0];
LABEL_33:
          v101 = (char *)v103;
          v31 = (char *)v103;
          goto LABEL_45;
        }
        v55 = v102;
        if ( v102 )
        {
          if ( v102 == 1 )
          {
            *(_BYTE *)dest = v103[0];
          }
          else
          {
            a2 = v103;
            memcpy(dest, v103, v102);
          }
          v55 = v102;
          v30 = dest;
        }
      }
      else
      {
        LOBYTE(v103[0]) = 0;
        v30 = dest;
        v55 = 0;
        v101 = (char *)v103;
      }
      v97 = v55;
      v30[v55] = 0;
      v31 = v101;
      goto LABEL_45;
    }
    if ( *(_DWORD *)v19 != 1920298867 || *((_WORD *)v19 + 2) != 25955 )
    {
      if ( *(_DWORD *)v19 != 1735549300 || *((_WORD *)v19 + 2) != 29797 )
      {
LABEL_8:
        BYTE1(v103[0]) = 1;
        v20 = "unknown Key for Global Variable";
LABEL_9:
        v101 = (char *)v20;
        LOBYTE(v103[0]) = 3;
        v21 = sub_16FD110(v10, (unsigned __int64)a2, v11, v12, v13);
        sub_16F8270(v82, v21, (__int64)&v101);
        goto LABEL_11;
      }
      a2 = v106;
      v34 = sub_16F8F10((__int64)src, v106);
      v35 = v33;
      if ( v34 )
      {
        v99[0] = v33;
        v36 = v33;
        v101 = (char *)v103;
        if ( v33 > 0xF )
        {
          srcc = v34;
          v64 = sub_22409D0(&v101, v99, 0);
          v34 = srcc;
          v101 = (char *)v64;
          v65 = (_QWORD *)v64;
          v103[0] = v99[0];
        }
        else
        {
          if ( v33 == 1 )
          {
            LOBYTE(v103[0]) = *v34;
            v37 = (char *)v103;
            goto LABEL_41;
          }
          if ( !v33 )
          {
            v37 = (char *)v103;
            goto LABEL_41;
          }
          v65 = v103;
        }
        a2 = (unsigned __int64 *)v34;
        memcpy(v65, v34, v35);
        v36 = v99[0];
        v37 = v101;
LABEL_41:
        v102 = v36;
        v37[v36] = 0;
        v38 = v93;
        v31 = (char *)v93;
        if ( v101 != (char *)v103 )
        {
          a2 = (unsigned __int64 *)v102;
          if ( v93 == v95 )
          {
            v93 = v101;
            v94 = v102;
            v95[0] = v103[0];
          }
          else
          {
            v32 = v95[0];
            v93 = v101;
            v94 = v102;
            v95[0] = v103[0];
            if ( v31 )
              goto LABEL_44;
          }
          goto LABEL_33;
        }
        v54 = v102;
        if ( v102 )
        {
          if ( v102 == 1 )
          {
            *(_BYTE *)v93 = v103[0];
          }
          else
          {
            a2 = v103;
            memcpy(v93, v103, v102);
          }
          v54 = v102;
          v38 = v93;
        }
      }
      else
      {
        LOBYTE(v103[0]) = 0;
        v38 = v93;
        v54 = 0;
        v101 = (char *)v103;
      }
      v94 = v54;
      v38[v54] = 0;
      v31 = v101;
      goto LABEL_45;
    }
    v99[1] = 0;
    v99[0] = v100;
    LOBYTE(v100[0]) = 0;
    v40 = sub_16F8F10((__int64)src, v106);
    v41 = v39;
    if ( !v40 )
    {
      LOBYTE(v103[0]) = 0;
      v44 = v90;
      v47 = 0;
      v101 = (char *)v103;
LABEL_68:
      v91 = v47;
      v44[v47] = 0;
      v45 = v101;
      goto LABEL_62;
    }
    v88[0] = v39;
    v42 = v39;
    v101 = (char *)v103;
    if ( v39 > 0xF )
    {
      n = v39;
      srcb = v40;
      v56 = sub_22409D0(&v101, v88, 0);
      v40 = srcb;
      v41 = n;
      v101 = (char *)v56;
      v57 = (_QWORD *)v56;
      v103[0] = v88[0];
    }
    else
    {
      if ( v39 == 1 )
      {
        LOBYTE(v103[0]) = *v40;
        v43 = (char *)v103;
        goto LABEL_58;
      }
      if ( !v39 )
      {
        v43 = (char *)v103;
        goto LABEL_58;
      }
      v57 = v103;
    }
    memcpy(v57, v40, v41);
    v42 = v88[0];
    v43 = v101;
LABEL_58:
    v102 = v42;
    v43[v42] = 0;
    v44 = v90;
    v45 = (char *)v90;
    if ( v101 == (char *)v103 )
    {
      v47 = v102;
      if ( v102 )
      {
        if ( v102 == 1 )
          *(_BYTE *)v90 = v103[0];
        else
          memcpy(v90, v103, v102);
        v47 = v102;
        v44 = v90;
      }
      goto LABEL_68;
    }
    if ( v90 == v92 )
    {
      v90 = v101;
      v91 = v102;
      v92[0] = v103[0];
LABEL_82:
      v101 = (char *)v103;
      v45 = (char *)v103;
      goto LABEL_62;
    }
    v46 = v92[0];
    v90 = v101;
    v91 = v102;
    v92[0] = v103[0];
    if ( !v45 )
      goto LABEL_82;
    v101 = v45;
    v103[0] = v46;
LABEL_62:
    v102 = 0;
    *v45 = 0;
    if ( v101 != (char *)v103 )
      j_j___libc_free_0(v101, v103[0] + 1LL);
    sub_16C9340((__int64)&v101, (__int64)v90, v91, 0);
    a2 = v99;
    srca = sub_16C9430(&v101, v99);
    sub_16C93F0(&v101);
    if ( !srca )
    {
      sub_8FD6D0((__int64)&v101, "invalid regex: ", v99);
      v88[0] = &v101;
      v89 = 260;
      v73 = sub_16FD110(v10, (unsigned __int64)"invalid regex: ", v70, v71, v72);
      sub_16F8270(v82, v73, (__int64)v88);
      if ( v101 != (char *)v103 )
        j_j___libc_free_0(v101, v103[0] + 1LL);
      if ( (_QWORD *)v99[0] != v100 )
        j_j___libc_free_0(v99[0], v100[0] + 1LL);
LABEL_11:
      if ( (_BYTE *)v106[0] != v107 )
        _libc_free(v106[0]);
      if ( (_BYTE *)v104[0] != v105 )
        _libc_free(v104[0]);
LABEL_15:
      v23 = 0;
      goto LABEL_16;
    }
    if ( (_QWORD *)v99[0] != v100 )
    {
      a2 = (unsigned __int64 *)(v100[0] + 1LL);
      j_j___libc_free_0(v99[0], v100[0] + 1LL);
    }
LABEL_47:
    if ( (_BYTE *)v106[0] != v107 )
      _libc_free(v106[0]);
    if ( (_BYTE *)v104[0] != v105 )
      _libc_free(v104[0]);
    sub_16FD380(v9, (unsigned __int64)a2);
    if ( !*(_QWORD *)(v9 + 80) )
      v9 = 0;
  }
  v48 = v94;
  if ( (v94 == 0) == (v97 == 0) )
  {
    v107[1] = 1;
    v107[0] = 3;
    v106[0] = (unsigned __int64)"exactly one of transform or target must be specified";
    sub_16F8270(v82, v81, (__int64)v106);
    goto LABEL_15;
  }
  v49 = v91;
  if ( !v94 )
  {
    v83 = v90;
    v58 = dest;
    v85 = v97;
    v59 = sub_22077B0(80);
    v52 = (_QWORD *)v59;
    if ( !v59 )
      goto LABEL_76;
    v60 = v83;
    v61 = (_BYTE *)(v59 + 32);
    *(_DWORD *)(v59 + 8) = 2;
    v62 = v85;
    *(_QWORD *)(v59 + 16) = v59 + 32;
    *(_QWORD *)v59 = off_49853D0;
    if ( !v83 )
    {
      *(_QWORD *)(v59 + 24) = 0;
      *(_BYTE *)(v59 + 32) = 0;
LABEL_94:
      v63 = v52 + 8;
      v52[6] = v52 + 8;
      if ( !v58 )
        goto LABEL_99;
      v106[0] = v62;
      if ( v62 > 0xF )
      {
        v87 = v62;
        v69 = sub_22409D0(v52 + 6, v106, 0);
        v62 = v87;
        v52[6] = v69;
        v63 = (_BYTE *)v69;
        v52[8] = v106[0];
      }
      else
      {
        if ( v62 == 1 )
        {
          *((_BYTE *)v52 + 64) = *v58;
LABEL_98:
          v52[7] = v62;
          v63[v62] = 0;
          goto LABEL_76;
        }
        if ( !v62 )
          goto LABEL_98;
      }
      memcpy(v63, v58, v62);
      v62 = v106[0];
      v63 = (_BYTE *)v52[6];
      goto LABEL_98;
    }
    v106[0] = v49;
    if ( v49 > 0xF )
    {
      v68 = sub_22409D0(v59 + 16, v106, 0);
      v62 = v85;
      v60 = v83;
      v52[2] = v68;
      v61 = (_BYTE *)v68;
      v52[4] = v106[0];
    }
    else
    {
      if ( v49 == 1 )
      {
        *(_BYTE *)(v59 + 32) = *v83;
LABEL_93:
        v52[3] = v49;
        v61[v49] = 0;
        goto LABEL_94;
      }
      if ( !v49 )
        goto LABEL_93;
    }
    v86 = v62;
    memcpy(v61, v60, v49);
    v49 = v106[0];
    v61 = (_BYTE *)v52[2];
    v62 = v86;
    goto LABEL_93;
  }
  v84 = v90;
  v50 = v93;
  v51 = sub_22077B0(80);
  v52 = (_QWORD *)v51;
  if ( v51 )
  {
    *(_DWORD *)(v51 + 8) = 2;
    *(_QWORD *)v51 = off_49853A8;
    *(_QWORD *)(v51 + 16) = v51 + 32;
    if ( v84 )
    {
      sub_1B678F0((__int64 *)(v51 + 16), v84, (__int64)&v84[v49]);
    }
    else
    {
      *(_QWORD *)(v51 + 24) = 0;
      *(_BYTE *)(v51 + 32) = 0;
    }
    v52[6] = v52 + 8;
    if ( v50 )
    {
      sub_1B678F0(v52 + 6, v50, (__int64)&v50[v48]);
      goto LABEL_76;
    }
LABEL_99:
    v52[7] = 0;
    *((_BYTE *)v52 + 64) = 0;
  }
LABEL_76:
  v23 = 1;
  v53 = sub_22077B0(24);
  *(_QWORD *)(v53 + 16) = v52;
  sub_2208C80(v53, a5);
  ++*(_QWORD *)(a5 + 16);
LABEL_16:
  if ( dest != v98 )
    j_j___libc_free_0(dest, v98[0] + 1LL);
  if ( v93 != v95 )
    j_j___libc_free_0(v93, v95[0] + 1LL);
  if ( v90 != v92 )
    j_j___libc_free_0(v90, v92[0] + 1LL);
  return v23;
}
