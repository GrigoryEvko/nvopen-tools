// Function: sub_1A58B80
// Address: 0x1a58b80
//
__int64 __fastcall sub_1A58B80(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v9; // r13
  __int64 *v10; // r12
  __int64 result; // rax
  size_t v12; // r15
  char *v13; // r11
  int v14; // ecx
  __int64 v15; // r8
  __int64 v16; // rsi
  int v17; // ecx
  __int64 v18; // r10
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r15
  _QWORD **v22; // rax
  _QWORD *v23; // rdx
  unsigned int v24; // edi
  __int64 *v25; // rdx
  __int64 v26; // r15
  _QWORD **v27; // rdx
  _QWORD *v28; // rdx
  unsigned int k; // ecx
  char *v30; // rsi
  __int64 *v31; // rdi
  size_t v32; // rdx
  int v33; // r15d
  size_t v34; // r15
  char *v35; // r10
  int v36; // ecx
  char *v37; // r10
  __int64 *v38; // r12
  __int64 *v39; // r14
  __int64 v40; // rdi
  int v41; // ecx
  __int64 v42; // r11
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // r8
  _QWORD **v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // r8
  unsigned int v49; // esi
  __int64 *v50; // rdx
  __int64 v51; // r15
  _QWORD **v52; // rdx
  _QWORD *v53; // rdx
  unsigned int j; // ecx
  __int64 v55; // rcx
  __int64 *v56; // rax
  __int64 v57; // r11
  char *v58; // r9
  __int64 v59; // r10
  __int64 v60; // r8
  __int64 v61; // r10
  int v62; // ecx
  size_t v63; // rdx
  const void *v64; // rsi
  signed __int64 v65; // rcx
  char *v66; // rax
  char *v67; // rcx
  unsigned int v68; // eax
  int v69; // edx
  int v70; // eax
  __int64 v71; // rcx
  __int64 *v72; // rax
  int v73; // eax
  int i; // edx
  char *v75; // r10
  int v76; // edi
  int v77; // esi
  unsigned int v78; // eax
  __int64 v79; // [rsp+0h] [rbp-80h]
  __int64 v80; // [rsp+0h] [rbp-80h]
  int v81; // [rsp+8h] [rbp-78h]
  __int64 v82; // [rsp+8h] [rbp-78h]
  char *v83; // [rsp+8h] [rbp-78h]
  __int64 v84; // [rsp+8h] [rbp-78h]
  int v85; // [rsp+8h] [rbp-78h]
  __int64 v86; // [rsp+8h] [rbp-78h]
  __int64 v87; // [rsp+8h] [rbp-78h]
  char *v88; // [rsp+10h] [rbp-70h]
  int v89; // [rsp+10h] [rbp-70h]
  int v90; // [rsp+10h] [rbp-70h]
  char *v91; // [rsp+10h] [rbp-70h]
  int v92; // [rsp+10h] [rbp-70h]
  void *v93; // [rsp+10h] [rbp-70h]
  int v94; // [rsp+10h] [rbp-70h]
  int v95; // [rsp+10h] [rbp-70h]
  int na; // [rsp+18h] [rbp-68h]
  int nb; // [rsp+18h] [rbp-68h]
  int nc; // [rsp+18h] [rbp-68h]
  int nd; // [rsp+18h] [rbp-68h]
  size_t ne; // [rsp+18h] [rbp-68h]
  int nf; // [rsp+18h] [rbp-68h]
  int ng; // [rsp+18h] [rbp-68h]
  __int64 desta; // [rsp+20h] [rbp-60h]
  void *dest; // [rsp+20h] [rbp-60h]
  int destb; // [rsp+20h] [rbp-60h]
  int destc; // [rsp+20h] [rbp-60h]
  int destd; // [rsp+20h] [rbp-60h]
  int deste; // [rsp+20h] [rbp-60h]
  int destg; // [rsp+20h] [rbp-60h]
  int desth; // [rsp+20h] [rbp-60h]
  __int64 v113; // [rsp+28h] [rbp-58h]
  char *src; // [rsp+30h] [rbp-50h]
  char *v115; // [rsp+38h] [rbp-48h]
  int v116; // [rsp+38h] [rbp-48h]
  int v117; // [rsp+38h] [rbp-48h]
  int v118; // [rsp+38h] [rbp-48h]
  int v119; // [rsp+38h] [rbp-48h]
  __int64 *v120; // [rsp+40h] [rbp-40h] BYREF
  __int64 *v121; // [rsp+48h] [rbp-38h] BYREF

  v9 = a1;
  v10 = a2;
  result = a7;
  if ( a5 <= a7 )
    result = a5;
  if ( a4 > result )
  {
    v33 = a5;
    if ( a5 > a7 )
    {
      v120 = a1;
      v121 = a2;
      if ( a4 > a5 )
      {
        ne = (size_t)a6;
        v113 = a4 / 2;
        sub_1A4EE80(&v120, a4 / 2);
        src = (char *)v120;
        v72 = sub_1A504C0(a2, a3, v120, v71);
        v59 = a4;
        v58 = (char *)ne;
        v115 = (char *)v72;
        v57 = a7;
        v121 = v72;
        v60 = v72 - a2;
      }
      else
      {
        v88 = a6;
        desta = a5 / 2;
        sub_1A4EE80(&v121, a5 / 2);
        v115 = (char *)v121;
        v56 = sub_1A50320(a1, (__int64)a2, v121, v55);
        v57 = a7;
        v58 = v88;
        v59 = a4;
        v60 = desta;
        src = (char *)v56;
        v120 = v56;
        v113 = v56 - a1;
      }
      v61 = v59 - v113;
      if ( v61 <= v60 || v60 > v57 )
      {
        if ( v61 > v57 )
        {
          v87 = v57;
          v95 = (int)v58;
          ng = v60;
          desth = v61;
          v78 = (unsigned int)sub_1A50D10(src, (char *)a2, v115);
          LODWORD(v61) = desth;
          LODWORD(v60) = ng;
          LODWORD(v58) = v95;
          v57 = v87;
          LODWORD(v115) = v78;
          LODWORD(src) = (_DWORD)v120;
        }
        else if ( v61 )
        {
          v65 = (char *)a2 - src;
          if ( a2 != (__int64 *)src )
          {
            v82 = v57;
            v90 = v60;
            na = v61;
            v66 = (char *)memmove(v58, src, (char *)a2 - src);
            v57 = v82;
            LODWORD(v60) = v90;
            LODWORD(v61) = na;
            v65 = (char *)a2 - src;
            v58 = v66;
          }
          v67 = &v58[v65];
          if ( a2 != (__int64 *)v115 )
          {
            v80 = v57;
            v83 = v58;
            v91 = v67;
            nb = v60;
            destc = v61;
            memmove(src, a2, v115 - (char *)a2);
            v57 = v80;
            v58 = v83;
            v67 = v91;
            LODWORD(v60) = nb;
            LODWORD(v61) = destc;
          }
          v84 = v57;
          v92 = v60;
          nc = v61;
          destd = (int)v58;
          v68 = (unsigned int)sub_1A58B50(v58, v67, (__int64)v115);
          LODWORD(v58) = destd;
          LODWORD(v61) = nc;
          LODWORD(v60) = v92;
          v57 = v84;
          LODWORD(v115) = v68;
          LODWORD(src) = (_DWORD)v120;
        }
      }
      else if ( v60 )
      {
        if ( a2 == (__int64 *)v115 )
        {
          v86 = v57;
          v94 = (int)v58;
          nf = (_DWORD)v115 - (_DWORD)a2;
          destg = v60;
          v118 = v61;
          sub_1A58B50(src, a2, (__int64)a2);
          LODWORD(v61) = v118;
          LODWORD(v60) = destg;
          v62 = nf;
          LODWORD(v58) = v94;
          v57 = v86;
        }
        else
        {
          v79 = v57;
          v81 = v60;
          v89 = v61;
          dest = v58;
          memmove(v58, a2, v115 - (char *)a2);
          sub_1A58B50(src, a2, (__int64)v115);
          v62 = (_DWORD)v115 - (_DWORD)a2;
          LODWORD(v58) = (_DWORD)dest;
          LODWORD(v61) = v89;
          LODWORD(v60) = v81;
          v57 = v79;
          if ( v115 != (char *)a2 )
          {
            v63 = v115 - (char *)a2;
            v64 = dest;
            destb = v62;
            v116 = (int)v58;
            memmove(src, v64, v63);
            v57 = v79;
            LODWORD(v60) = v81;
            LODWORD(v61) = v89;
            v62 = destb;
            LODWORD(v58) = v116;
          }
        }
        LODWORD(v115) = (_DWORD)src + v62;
        LODWORD(src) = (_DWORD)v120;
      }
      else
      {
        LODWORD(v115) = (_DWORD)src;
      }
      v85 = v61;
      v93 = (void *)v57;
      nd = (int)v58;
      deste = v60;
      sub_1A58B80((_DWORD)a1, (_DWORD)src, (_DWORD)v115, v113, v60, (_DWORD)v58, v57, a8);
      return sub_1A58B80((_DWORD)v115, (_DWORD)v121, a3, v85, v33 - deste, nd, (__int64)v93, a8);
    }
    v34 = a3 - (_QWORD)a2;
    if ( a2 != (__int64 *)a3 )
    {
      result = (__int64)memmove(a6, a2, a3 - (_QWORD)a2);
      a6 = (char *)result;
    }
    v35 = &a6[v34];
    if ( a1 == a2 )
    {
      if ( a6 == v35 )
        return result;
      v32 = v34;
      v31 = (__int64 *)(a3 - v34);
      goto LABEL_78;
    }
    if ( a6 == v35 )
      return result;
    v36 = *(_DWORD *)(a8 + 24);
    v37 = v35 - 8;
    v38 = a2 - 1;
    v39 = (__int64 *)(a3 - 8);
    v40 = *(_QWORD *)v37;
    if ( !v36 )
      goto LABEL_42;
LABEL_30:
    v41 = v36 - 1;
    v42 = *(_QWORD *)(a8 + 8);
    v43 = v41 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v44 = (__int64 *)(v42 + 16LL * v43);
    v45 = *v44;
    if ( v40 == *v44 )
    {
LABEL_31:
      v46 = (_QWORD **)v44[1];
      if ( v46 )
      {
        v47 = *v46;
        for ( result = 1; v47; result = (unsigned int)(result + 1) )
          v47 = (_QWORD *)*v47;
        goto LABEL_34;
      }
    }
    else
    {
      v73 = 1;
      while ( v45 != -8 )
      {
        v77 = v73 + 1;
        v43 = v41 & (v73 + v43);
        v44 = (__int64 *)(v42 + 16LL * v43);
        v45 = *v44;
        if ( v40 == *v44 )
          goto LABEL_31;
        v73 = v77;
      }
    }
    result = 0;
LABEL_34:
    v48 = *v38;
    v49 = v41 & (((unsigned int)*v38 >> 9) ^ ((unsigned int)*v38 >> 4));
    v50 = (__int64 *)(v42 + 16LL * v49);
    v51 = *v50;
    if ( *v38 != *v50 )
    {
      for ( i = 1; ; i = v117 )
      {
        if ( v51 == -8 )
          goto LABEL_42;
        v49 = v41 & (i + v49);
        v117 = i + 1;
        v50 = (__int64 *)(v42 + 16LL * v49);
        v51 = *v50;
        if ( v48 == *v50 )
          break;
      }
    }
    v52 = (_QWORD **)v50[1];
    if ( !v52 )
      goto LABEL_42;
    v53 = *v52;
    for ( j = 1; v53; ++j )
      v53 = (_QWORD *)*v53;
    if ( j <= (unsigned int)result )
    {
LABEL_42:
      while ( 1 )
      {
        *v39 = v40;
        if ( a6 == v37 )
          return result;
        v37 -= 8;
LABEL_41:
        v36 = *(_DWORD *)(a8 + 24);
        v40 = *(_QWORD *)v37;
        --v39;
        if ( v36 )
          goto LABEL_30;
      }
    }
    *v39 = v48;
    if ( v9 != v38 )
    {
      --v38;
      goto LABEL_41;
    }
    v75 = v37 + 8;
    if ( a6 == v75 )
      return result;
    v32 = v75 - a6;
    v31 = (__int64 *)((char *)v39 - (v75 - a6));
LABEL_78:
    v30 = a6;
    return (__int64)memmove(v31, v30, v32);
  }
  v12 = (char *)a2 - (char *)a1;
  if ( a1 != a2 )
  {
    result = (__int64)memmove(a6, a1, v12);
    a6 = (char *)result;
  }
  v13 = &a6[v12];
  if ( a6 != &a6[v12] && a2 != (__int64 *)a3 )
  {
    while ( 1 )
    {
      v14 = *(_DWORD *)(a8 + 24);
      v15 = *(_QWORD *)a6;
      if ( !v14 )
        goto LABEL_64;
      v16 = *v10;
      v17 = v14 - 1;
      v18 = *(_QWORD *)(a8 + 8);
      v19 = v17 & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( *v10 != *v20 )
        break;
LABEL_10:
      v22 = (_QWORD **)v20[1];
      if ( !v22 )
        goto LABEL_67;
      v23 = *v22;
      for ( result = 1; v23; result = (unsigned int)(result + 1) )
        v23 = (_QWORD *)*v23;
LABEL_13:
      v24 = v17 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v25 = (__int64 *)(v18 + 16LL * v24);
      v26 = *v25;
      if ( v15 != *v25 )
      {
        v69 = 1;
        while ( v26 != -8 )
        {
          v24 = v17 & (v69 + v24);
          v119 = v69 + 1;
          v25 = (__int64 *)(v18 + 16LL * v24);
          v26 = *v25;
          if ( v15 == *v25 )
            goto LABEL_14;
          v69 = v119;
        }
LABEL_64:
        a6 += 8;
        v16 = v15;
        goto LABEL_19;
      }
LABEL_14:
      v27 = (_QWORD **)v25[1];
      if ( !v27 )
        goto LABEL_64;
      v28 = *v27;
      for ( k = 1; v28; ++k )
        v28 = (_QWORD *)*v28;
      if ( k <= (unsigned int)result )
        goto LABEL_64;
      ++v10;
LABEL_19:
      *v9++ = v16;
      if ( v13 == a6 )
        return result;
      if ( (__int64 *)a3 == v10 )
        goto LABEL_21;
    }
    v70 = 1;
    while ( v21 != -8 )
    {
      v76 = v70 + 1;
      v19 = v17 & (v70 + v19);
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v16 == *v20 )
        goto LABEL_10;
      v70 = v76;
    }
LABEL_67:
    result = 0;
    goto LABEL_13;
  }
LABEL_21:
  if ( v13 != a6 )
  {
    v30 = a6;
    v31 = v9;
    v32 = v13 - a6;
    return (__int64)memmove(v31, v30, v32);
  }
  return result;
}
