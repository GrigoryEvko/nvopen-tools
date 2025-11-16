// Function: sub_E1DAD0
// Address: 0xe1dad0
//
__int64 __fastcall sub_E1DAD0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  int v7; // r12d
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // r8
  char v15; // al
  __int64 **v16; // r12
  __int64 *v17; // r13
  __int64 v18; // rax
  __int64 v19; // r15
  char v20; // al
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rbx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r12
  __int64 v37; // rax
  char v38; // al
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rbx
  __int64 v46; // r8
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r12
  __int64 v52; // rax
  char v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // rax
  char *v61; // r13
  char *v62; // r9
  size_t v63; // rax
  _BYTE *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rsi
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  char v71; // bl
  __int64 v72; // rax
  __int64 v73; // rcx
  _BYTE *v74; // rax
  _OWORD *v75; // rdi
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // rbx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rax
  __int64 *v82; // r15
  __int64 v83; // rax
  char *v84; // rdi
  void *v85; // r13
  __int64 v86; // rdx
  __int64 v87; // rbx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rax
  char v92; // dl
  char *v93; // rax
  __int64 v94; // rbx
  __int64 v95; // rax
  char *v96; // rdx
  __int64 v97; // rdi
  char *v98; // rax
  size_t v99; // [rsp+8h] [rbp-E8h]
  void *src; // [rsp+10h] [rbp-E0h]
  char *srcb; // [rsp+10h] [rbp-E0h]
  char *srca; // [rsp+10h] [rbp-E0h]
  size_t n; // [rsp+18h] [rbp-D8h]
  __int64 na; // [rsp+18h] [rbp-D8h]
  __int64 v105; // [rsp+20h] [rbp-D0h]
  __int64 v106; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v107; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD *v108; // [rsp+40h] [rbp-B0h] BYREF
  __int64 ***v109; // [rsp+48h] [rbp-A8h]
  __int64 v110; // [rsp+50h] [rbp-A0h]
  __int64 v111; // [rsp+58h] [rbp-98h]
  _QWORD v112[3]; // [rsp+60h] [rbp-90h] BYREF
  _OWORD v113[4]; // [rsp+78h] [rbp-78h] BYREF
  _BYTE v114[56]; // [rsp+B8h] [rbp-38h] BYREF

  v106 = a2;
  v108 = (_QWORD *)a1;
  v109 = (__int64 ***)&v106;
  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Ty") )
  {
    v7 = *((_DWORD *)v108 + 198);
    v8 = v108 + 102;
    *((_DWORD *)v108 + 198) = v7 + 1;
    v9 = sub_E0E790((__int64)v8, 24, v3, v4, v5, v6);
    v13 = v9;
    if ( v9 )
    {
      v14 = 16417;
      *(_WORD *)(v9 + 8) = 16417;
      v15 = *(_BYTE *)(v9 + 10);
      *(_DWORD *)(v13 + 12) = 0;
      *(_DWORD *)(v13 + 16) = v7;
      *(_BYTE *)(v13 + 10) = v15 & 0xF0 | 5;
      *(_QWORD *)v13 = &unk_49DFA28;
      v16 = *v109;
      if ( !*v109 )
        goto LABEL_6;
      v17 = v16[1];
      if ( v17 != v16[2] )
      {
LABEL_5:
        v16[1] = v17 + 1;
        *v17 = v13;
LABEL_6:
        v18 = sub_E0E790(a1 + 816, 24, v10, v11, v14, v12);
        v19 = v18;
        if ( v18 )
        {
          v20 = *(_BYTE *)(v18 + 10);
          *(_QWORD *)(v19 + 16) = v13;
          *(_WORD *)(v19 + 8) = 35;
          *(_BYTE *)(v19 + 10) = v20 & 0xF0 | 5;
          *(_QWORD *)v19 = &unk_49DFAE8;
        }
        return v19;
      }
      v82 = *v16;
      na = (char *)v17 - (char *)*v16;
      if ( *v16 == (__int64 *)(v16 + 3) )
      {
        v84 = (char *)malloc(16 * (na >> 3), 24, na, 16 * (na >> 3), 16417, v12);
        if ( v84 )
        {
          v11 = 16 * (na >> 3);
          v10 = na;
          if ( v17 != v82 )
          {
            v93 = (char *)memmove(v84, v82, na);
            v11 = 16 * (na >> 3);
            v10 = na;
            v84 = v93;
          }
          *v16 = (__int64 *)v84;
          goto LABEL_43;
        }
      }
      else
      {
        v83 = realloc(v82);
        v11 = 16 * (na >> 3);
        v10 = na;
        *v16 = (__int64 *)v83;
        v84 = (char *)v83;
        if ( v83 )
        {
LABEL_43:
          v17 = (__int64 *)&v84[v10];
          v16[2] = (__int64 *)&v84[v11];
          goto LABEL_5;
        }
      }
LABEL_59:
      abort();
    }
    return 0;
  }
  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Tk") )
  {
    v26 = *(_BYTE *)(a1 + 778);
    *(_BYTE *)(a1 + 778) = 1;
    v29 = sub_E1D370(a1, 0, v22, v23, v24, v25);
    if ( v29 && (v36 = sub_E11180(&v108, 0, v27, v28, v30, v31)) != 0 )
    {
      v37 = sub_E0E790(a1 + 816, 32, v32, v33, v34, v35);
      v19 = v37;
      if ( v37 )
      {
        *(_WORD *)(v37 + 8) = 36;
        v38 = *(_BYTE *)(v37 + 10);
        *(_QWORD *)(v19 + 16) = v29;
        *(_QWORD *)(v19 + 24) = v36;
        *(_BYTE *)(v19 + 10) = v38 & 0xF0 | 5;
        *(_QWORD *)v19 = &unk_49DFB48;
      }
    }
    else
    {
      v19 = 0;
    }
    *(_BYTE *)(a1 + 778) = v26;
    return v19;
  }
  if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Tn") )
  {
    if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Tt") )
    {
      if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Tp") )
        return 0;
      v78 = sub_E1DAD0(a1, v106);
      if ( !v78 )
        return 0;
      v81 = sub_E0E790(a1 + 816, 24, v76, v77, v79, v80);
      v19 = v81;
      if ( v81 )
      {
        *(_QWORD *)(v81 + 16) = v78;
        *(_WORD *)(v81 + 8) = 39;
        *(_BYTE *)(v81 + 10) = *(_BYTE *)(v81 + 10) & 0xF0 | 5;
        *(_QWORD *)v81 = &unk_49DFC68;
      }
      return v19;
    }
    v19 = sub_E11180(&v108, 2, v54, v55, v56, v57);
    if ( !v19 )
      return v19;
    v60 = *(_QWORD *)(a1 + 24);
    v61 = *(char **)(a1 + 672);
    v62 = *(char **)(a1 + 664);
    v110 = a1;
    v105 = v60;
    v63 = *(_QWORD *)(a1 + 16);
    v112[0] = v113;
    n = v63;
    v112[1] = v113;
    v112[2] = v114;
    v111 = (v61 - v62) >> 3;
    memset(v113, 0, sizeof(v113));
    if ( v61 == *(char **)(a1 + 680) )
    {
      v94 = 16 * ((v61 - v62) >> 3);
      if ( v62 == (char *)(a1 + 688) )
      {
        v99 = v61 - v62;
        srca = v62;
        v97 = malloc(16 * ((v61 - v62) >> 3), v114, v61 - v62, v58, v59, v62);
        if ( !v97 )
          goto LABEL_59;
        v62 = srca;
        v96 = (char *)v99;
        if ( v61 != srca )
        {
          v98 = (char *)memmove((void *)v97, srca, v99);
          v96 = (char *)v99;
          v97 = (__int64)v98;
        }
        *(_QWORD *)(a1 + 664) = v97;
      }
      else
      {
        srcb = (char *)(v61 - v62);
        v95 = realloc(v62);
        v96 = srcb;
        *(_QWORD *)(a1 + 664) = v95;
        v97 = v95;
        if ( !v95 )
          goto LABEL_59;
      }
      v61 = &v96[v97];
      *(_QWORD *)(a1 + 680) = v97 + v94;
    }
    *(_QWORD *)(a1 + 672) = v61 + 8;
    *(_QWORD *)v61 = v112;
    v64 = *(_BYTE **)a1;
    v65 = *(_QWORD *)(a1 + 8);
LABEL_26:
    if ( v64 != (_BYTE *)v65 && *v64 == 69 )
    {
      v73 = 0;
      *(_QWORD *)a1 = v64 + 1;
LABEL_45:
      src = (void *)v73;
      v66 = 48;
      v85 = sub_E11E80((_QWORD *)a1, (__int64)(v105 - n) >> 3, v65, v73, v59, (__int64)v62);
      v87 = v86;
      v91 = sub_E0E790(a1 + 816, 48, v86, v88, v89, v90);
      if ( v91 )
      {
        *(_QWORD *)(v91 + 16) = v19;
        v19 = v91;
        *(_WORD *)(v91 + 8) = 38;
        v92 = *(_BYTE *)(v91 + 10);
        *(_QWORD *)(v91 + 24) = v85;
        *(_QWORD *)(v91 + 32) = v87;
        *(_QWORD *)(v91 + 40) = src;
        *(_BYTE *)(v91 + 10) = v92 & 0xF0 | 5;
        *(_QWORD *)v91 = &unk_49DFC08;
LABEL_35:
        v75 = (_OWORD *)v112[0];
        *(_QWORD *)(v110 + 672) = *(_QWORD *)(v110 + 664) + 8 * v111;
        if ( v75 != v113 )
          _libc_free(v75, v66);
        return v19;
      }
    }
    else
    {
      while ( 1 )
      {
        v66 = (__int64)v112;
        v107 = sub_E1DAD0(a1, v112);
        if ( !v107 )
          break;
        v66 = (__int64)&v107;
        sub_E18380(a1 + 16, &v107, v67, v68, v69, v70);
        v64 = *(_BYTE **)a1;
        v65 = *(_QWORD *)(a1 + 8);
        if ( *(_QWORD *)a1 != v65 )
        {
          if ( *v64 != 81 )
            goto LABEL_26;
          v71 = *(_BYTE *)(a1 + 778);
          *(_BYTE *)(a1 + 778) = 1;
          *(_QWORD *)a1 = v64 + 1;
          v72 = sub_E18BB0(a1);
          *(_BYTE *)(a1 + 778) = v71;
          v73 = v72;
          if ( v72 )
          {
            v74 = *(_BYTE **)a1;
            if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v74 == 69 )
            {
              *(_QWORD *)a1 = v74 + 1;
              goto LABEL_45;
            }
          }
          break;
        }
      }
    }
    v19 = 0;
    goto LABEL_35;
  }
  v45 = sub_E11180(&v108, 1, v39, v40, v41, v42);
  if ( !v45 )
    return 0;
  v51 = sub_E1AEA0(a1, 1, v43, v44, v46);
  if ( !v51 )
    return 0;
  v52 = sub_E0E790(a1 + 816, 32, v47, v48, v49, v50);
  v19 = v52;
  if ( v52 )
  {
    *(_WORD *)(v52 + 8) = 37;
    v53 = *(_BYTE *)(v52 + 10);
    *(_QWORD *)(v19 + 16) = v45;
    *(_QWORD *)(v19 + 24) = v51;
    *(_BYTE *)(v19 + 10) = v53 & 0xF0 | 5;
    *(_QWORD *)v19 = &unk_49DFBA8;
  }
  return v19;
}
