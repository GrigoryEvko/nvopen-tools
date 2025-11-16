// Function: sub_2A83180
// Address: 0x2a83180
//
__int64 __fastcall sub_2A83180(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdi
  char *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
  unsigned int v24; // r12d
  char *v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 *v28; // r9
  size_t v29; // r8
  char *v30; // rax
  char *v31; // rdi
  __int64 v32; // r8
  char *v33; // rax
  unsigned __int64 v34; // rdx
  char *v35; // r10
  size_t v36; // r8
  _QWORD *v37; // rax
  _QWORD *v38; // rdi
  __int64 v39; // r8
  size_t v40; // r14
  unsigned __int64 v41; // r13
  _BYTE *v42; // r12
  __int64 v43; // rax
  _QWORD *v44; // rbx
  _BYTE *v45; // rdi
  _BYTE *v46; // rdi
  _QWORD *v47; // rax
  _BYTE *v48; // r12
  __int64 v49; // rax
  _BYTE *v50; // rdi
  size_t v51; // r8
  _BYTE *v52; // rdi
  __int64 v53; // rax
  _QWORD *v54; // rdi
  size_t v55; // rdx
  __int64 v56; // rax
  _QWORD *v57; // rdi
  size_t v58; // rdx
  char *v59; // rax
  unsigned __int64 v60; // rdx
  unsigned __int64 *v61; // r9
  size_t v62; // r15
  _QWORD *v63; // rax
  _QWORD *v64; // rdi
  __int64 v65; // r8
  __int64 v66; // rax
  _QWORD *v67; // rdi
  size_t v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // rax
  char *src; // [rsp+10h] [rbp-1E0h]
  char *srca; // [rsp+10h] [rbp-1E0h]
  size_t n; // [rsp+28h] [rbp-1C8h]
  size_t na; // [rsp+28h] [rbp-1C8h]
  char nb; // [rsp+28h] [rbp-1C8h]
  size_t nc; // [rsp+28h] [rbp-1C8h]
  size_t nd; // [rsp+28h] [rbp-1C8h]
  size_t ne; // [rsp+28h] [rbp-1C8h]
  __int64 ***v86; // [rsp+30h] [rbp-1C0h]
  size_t v87; // [rsp+30h] [rbp-1C0h]
  size_t v88; // [rsp+30h] [rbp-1C0h]
  size_t v89; // [rsp+30h] [rbp-1C0h]
  _BYTE *v90; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 v91; // [rsp+58h] [rbp-198h]
  _QWORD v92[2]; // [rsp+60h] [rbp-190h] BYREF
  void *v93; // [rsp+70h] [rbp-180h]
  size_t v94; // [rsp+78h] [rbp-178h]
  _QWORD v95[2]; // [rsp+80h] [rbp-170h] BYREF
  void *dest; // [rsp+90h] [rbp-160h]
  size_t v97; // [rsp+98h] [rbp-158h]
  _QWORD v98[2]; // [rsp+A0h] [rbp-150h] BYREF
  unsigned __int64 v99[2]; // [rsp+B0h] [rbp-140h] BYREF
  _QWORD v100[2]; // [rsp+C0h] [rbp-130h] BYREF
  _QWORD *v101; // [rsp+D0h] [rbp-120h] BYREF
  size_t v102; // [rsp+D8h] [rbp-118h]
  _QWORD v103[2]; // [rsp+E0h] [rbp-110h] BYREF
  _QWORD *v104; // [rsp+F0h] [rbp-100h] BYREF
  size_t v105; // [rsp+F8h] [rbp-F8h]
  _QWORD v106[2]; // [rsp+100h] [rbp-F0h] BYREF
  char *v107; // [rsp+110h] [rbp-E0h] BYREF
  size_t v108; // [rsp+118h] [rbp-D8h]
  _QWORD v109[2]; // [rsp+120h] [rbp-D0h] BYREF
  __int16 v110; // [rsp+130h] [rbp-C0h]
  unsigned __int64 v111[3]; // [rsp+140h] [rbp-B0h] BYREF
  _BYTE v112[40]; // [rsp+158h] [rbp-98h] BYREF
  unsigned __int64 v113[3]; // [rsp+180h] [rbp-70h] BYREF
  _BYTE v114[88]; // [rsp+198h] [rbp-58h] BYREF

  *(_BYTE *)(a4 + 76) = 0;
  v90 = v92;
  v93 = v95;
  v86 = (__int64 ***)a2;
  v91 = 0;
  LOBYTE(v92[0]) = 0;
  v94 = 0;
  LOBYTE(v95[0]) = 0;
  dest = v98;
  v97 = 0;
  LOBYTE(v98[0]) = 0;
  sub_CAEB90(a4, (unsigned __int64)a2);
  v9 = *(_QWORD *)(a4 + 80);
  if ( v9 )
    v9 = a4;
  while ( 1 )
  {
    if ( !v9 )
    {
LABEL_59:
      v40 = v94;
      if ( (v94 == 0) == (v97 == 0) )
      {
        v114[9] = 1;
        v113[0] = (unsigned __int64)"exactly one of transform or target must be specified";
        v114[8] = 3;
        sub_CA89D0(v86, a4, (__int64)v113, 0);
        goto LABEL_16;
      }
      v41 = v91;
      if ( v94 )
      {
        v42 = v93;
        v43 = sub_22077B0(0x50u);
        v44 = (_QWORD *)v43;
        if ( !v43 )
        {
LABEL_72:
          v24 = 1;
          v47 = (_QWORD *)sub_22077B0(0x18u);
          v47[2] = v44;
          sub_2208C80(v47, a5);
          ++*(_QWORD *)(a5 + 16);
          goto LABEL_17;
        }
        *(_DWORD *)(v43 + 8) = 2;
        v45 = (_BYTE *)(v43 + 32);
        *(_QWORD *)v43 = off_49D3E08;
        *(_QWORD *)(v43 + 16) = v43 + 32;
        if ( &v90[v91] && !v90 )
          goto LABEL_151;
        v113[0] = v91;
        if ( v91 > 0xF )
        {
          v72 = sub_22409D0(v43 + 16, v113, 0);
          v44[2] = v72;
          v45 = (_BYTE *)v72;
          v44[4] = v113[0];
        }
        else
        {
          if ( v91 == 1 )
          {
            *(_BYTE *)(v43 + 32) = *v90;
            goto LABEL_67;
          }
          if ( !v91 )
          {
LABEL_67:
            v44[3] = v41;
            v45[v41] = 0;
            v46 = v44 + 8;
            v44[6] = v44 + 8;
            if ( v42 )
            {
              v113[0] = v40;
              if ( v40 > 0xF )
              {
                v71 = sub_22409D0((__int64)(v44 + 6), v113, 0);
                v44[6] = v71;
                v46 = (_BYTE *)v71;
                v44[8] = v113[0];
              }
              else if ( v40 == 1 )
              {
                *((_BYTE *)v44 + 64) = *v42;
LABEL_71:
                v44[7] = v40;
                v46[v40] = 0;
                goto LABEL_72;
              }
              memcpy(v46, v42, v40);
              v40 = v113[0];
              v46 = (_BYTE *)v44[6];
              goto LABEL_71;
            }
LABEL_151:
            sub_426248((__int64)"basic_string::_M_construct null not valid");
          }
        }
        memcpy(v45, v90, v91);
        v41 = v113[0];
        v45 = (_BYTE *)v44[2];
        goto LABEL_67;
      }
      v87 = v97;
      v48 = dest;
      v49 = sub_22077B0(0x50u);
      v44 = (_QWORD *)v49;
      if ( !v49 )
        goto LABEL_72;
      *(_DWORD *)(v49 + 8) = 2;
      v50 = (_BYTE *)(v49 + 32);
      v51 = v87;
      *(_QWORD *)v49 = off_49D3E30;
      *(_QWORD *)(v49 + 16) = v49 + 32;
      if ( &v90[v91] && !v90 )
        goto LABEL_151;
      v113[0] = v91;
      if ( v91 > 0xF )
      {
        v70 = sub_22409D0(v49 + 16, v113, 0);
        v51 = v87;
        v44[2] = v70;
        v50 = (_BYTE *)v70;
        v44[4] = v113[0];
      }
      else
      {
        if ( v91 == 1 )
        {
          *(_BYTE *)(v49 + 32) = *v90;
LABEL_79:
          v44[3] = v41;
          v50[v41] = 0;
          v52 = v44 + 8;
          v44[6] = v44 + 8;
          if ( &v48[v51] && !v48 )
            goto LABEL_151;
          v113[0] = v51;
          if ( v51 > 0xF )
          {
            v89 = v51;
            v69 = sub_22409D0((__int64)(v44 + 6), v113, 0);
            v51 = v89;
            v44[6] = v69;
            v52 = (_BYTE *)v69;
            v44[8] = v113[0];
          }
          else
          {
            if ( v51 == 1 )
            {
              *((_BYTE *)v44 + 64) = *v48;
LABEL_84:
              v44[7] = v51;
              v52[v51] = 0;
              goto LABEL_72;
            }
            if ( !v51 )
            {
LABEL_136:
              v52 = (_BYTE *)v44[6];
              goto LABEL_84;
            }
          }
          memcpy(v52, v48, v51);
          v51 = v113[0];
          goto LABEL_136;
        }
        if ( !v91 )
          goto LABEL_79;
      }
      v88 = v51;
      memcpy(v50, v90, v91);
      v41 = v113[0];
      v50 = (_BYTE *)v44[2];
      v51 = v88;
      goto LABEL_79;
    }
    v10 = *(_QWORD *)(v9 + 80);
    v111[1] = 0;
    v111[0] = (unsigned __int64)v112;
    v113[0] = (unsigned __int64)v114;
    v111[2] = 32;
    v113[1] = 0;
    v113[2] = 32;
    v11 = sub_CAE820(v10, (unsigned __int64)a2, v6, v7, v8);
    if ( *(_DWORD *)(v11 + 32) != 1 )
    {
      v107 = "descriptor Key must be a scalar";
      v110 = 259;
      v23 = sub_CAE820(v10, (unsigned __int64)a2, v12, v13, v14);
      goto LABEL_11;
    }
    n = v11;
    v15 = sub_CAE940(v10, (unsigned __int64)a2, v12, v13, v14);
    if ( *((_DWORD *)v15 + 8) != 1 )
    {
      v107 = "descriptor value must be a scalar";
      v110 = 259;
      v23 = (__int64)sub_CAE940(v10, (unsigned __int64)a2, v16, v17, n);
      goto LABEL_11;
    }
    v18 = n;
    na = (size_t)v15;
    v19 = sub_CA8C30(v18, v111);
    if ( v20 != 6 )
    {
      if ( v20 == 9 )
      {
        v21 = 0x726F66736E617274LL;
        if ( *(_QWORD *)v19 == 0x726F66736E617274LL && v19[8] == 109 )
        {
          a2 = v113;
          v26 = sub_CA8C30(na, v113);
          v107 = (char *)v109;
          v28 = (unsigned __int64 *)v26;
          v29 = v27;
          if ( &v26[v27] && !v26 )
            goto LABEL_151;
          v104 = (_QWORD *)v27;
          if ( v27 > 0xF )
          {
            src = v26;
            nc = v27;
            v53 = sub_22409D0((__int64)&v107, (unsigned __int64 *)&v104, 0);
            v29 = nc;
            v28 = (unsigned __int64 *)src;
            v107 = (char *)v53;
            v54 = (_QWORD *)v53;
            v109[0] = v104;
          }
          else
          {
            if ( v27 == 1 )
            {
              LOBYTE(v109[0]) = *v26;
              v30 = (char *)v109;
              goto LABEL_33;
            }
            if ( !v27 )
            {
              v30 = (char *)v109;
              goto LABEL_33;
            }
            v54 = v109;
          }
          a2 = v28;
          memcpy(v54, v28, v29);
          v29 = (size_t)v104;
          v30 = v107;
LABEL_33:
          v108 = v29;
          v30[v29] = 0;
          v31 = (char *)dest;
          if ( v107 == (char *)v109 )
          {
            v55 = v108;
            if ( v108 )
            {
              if ( v108 == 1 )
              {
                *(_BYTE *)dest = v109[0];
              }
              else
              {
                a2 = v109;
                memcpy(dest, v109, v108);
              }
              v55 = v108;
              v31 = (char *)dest;
            }
            v97 = v55;
            v31[v55] = 0;
            v31 = v107;
            goto LABEL_37;
          }
          a2 = (unsigned __int64 *)v108;
          if ( dest == v98 )
          {
            dest = v107;
            v97 = v108;
            v98[0] = v109[0];
          }
          else
          {
            v32 = v98[0];
            dest = v107;
            v97 = v108;
            v98[0] = v109[0];
            if ( v31 )
            {
              v107 = v31;
              v109[0] = v32;
              goto LABEL_37;
            }
          }
          v107 = (char *)v109;
          v31 = (char *)v109;
LABEL_37:
          v108 = 0;
          *v31 = 0;
          if ( v107 != (char *)v109 )
          {
            a2 = (unsigned __int64 *)(v109[0] + 1LL);
            j_j___libc_free_0((unsigned __int64)v107);
          }
          goto LABEL_54;
        }
      }
LABEL_9:
      v107 = "unknown Key for Global Variable";
      v110 = 259;
      v23 = sub_CAE820(v10, (unsigned __int64)v111, v20, v21, v22);
LABEL_11:
      sub_CA89D0(v86, v23, (__int64)&v107, 0);
      goto LABEL_12;
    }
    if ( *(_DWORD *)v19 != 1920298867 || *((_WORD *)v19 + 2) != 25955 )
    {
      if ( *(_DWORD *)v19 != 1735549300 || *((_WORD *)v19 + 2) != 29797 )
        goto LABEL_9;
      a2 = v113;
      v59 = sub_CA8C30(na, v113);
      v61 = (unsigned __int64 *)v59;
      v104 = v106;
      v62 = v60;
      if ( &v59[v60] && !v59 )
        goto LABEL_151;
      v107 = (char *)v60;
      if ( v60 > 0xF )
      {
        ne = (size_t)v59;
        v66 = sub_22409D0((__int64)&v104, (unsigned __int64 *)&v107, 0);
        v61 = (unsigned __int64 *)ne;
        v104 = (_QWORD *)v66;
        v67 = (_QWORD *)v66;
        v106[0] = v107;
      }
      else
      {
        if ( v60 == 1 )
        {
          LOBYTE(v106[0]) = *v59;
          v63 = v106;
          goto LABEL_111;
        }
        if ( !v60 )
        {
          v63 = v106;
          goto LABEL_111;
        }
        v67 = v106;
      }
      a2 = v61;
      memcpy(v67, v61, v62);
      v62 = (size_t)v107;
      v63 = v104;
LABEL_111:
      v105 = v62;
      *((_BYTE *)v63 + v62) = 0;
      v64 = v93;
      if ( v104 == v106 )
      {
        v68 = v105;
        if ( v105 )
        {
          if ( v105 == 1 )
          {
            *(_BYTE *)v93 = v106[0];
          }
          else
          {
            a2 = v106;
            memcpy(v93, v106, v105);
          }
          v68 = v105;
          v64 = v93;
        }
        v94 = v68;
        *((_BYTE *)v64 + v68) = 0;
        v64 = v104;
        goto LABEL_115;
      }
      a2 = (unsigned __int64 *)v105;
      if ( v93 == v95 )
      {
        v93 = v104;
        v94 = v105;
        v95[0] = v106[0];
      }
      else
      {
        v65 = v95[0];
        v93 = v104;
        v94 = v105;
        v95[0] = v106[0];
        if ( v64 )
        {
          v104 = v64;
          v106[0] = v65;
          goto LABEL_115;
        }
      }
      v104 = v106;
      v64 = v106;
LABEL_115:
      v105 = 0;
      *(_BYTE *)v64 = 0;
      if ( v104 != v106 )
      {
        a2 = (unsigned __int64 *)(v106[0] + 1LL);
        j_j___libc_free_0((unsigned __int64)v104);
      }
      goto LABEL_54;
    }
    v99[1] = 0;
    v99[0] = (unsigned __int64)v100;
    LOBYTE(v100[0]) = 0;
    v33 = sub_CA8C30(na, v113);
    v35 = v33;
    v101 = v103;
    v36 = v34;
    if ( &v33[v34] && !v33 )
      goto LABEL_151;
    v107 = (char *)v34;
    if ( v34 > 0xF )
    {
      srca = v33;
      nd = v34;
      v56 = sub_22409D0((__int64)&v101, (unsigned __int64 *)&v107, 0);
      v36 = nd;
      v35 = srca;
      v101 = (_QWORD *)v56;
      v57 = (_QWORD *)v56;
      v103[0] = v107;
    }
    else
    {
      if ( v34 == 1 )
      {
        LOBYTE(v103[0]) = *v33;
        v37 = v103;
        goto LABEL_45;
      }
      if ( !v34 )
      {
        v37 = v103;
        goto LABEL_45;
      }
      v57 = v103;
    }
    memcpy(v57, v35, v36);
    v36 = (size_t)v107;
    v37 = v101;
LABEL_45:
    v102 = v36;
    *((_BYTE *)v37 + v36) = 0;
    v38 = v90;
    if ( v101 == v103 )
    {
      v58 = v102;
      if ( v102 )
      {
        if ( v102 == 1 )
          *v90 = v103[0];
        else
          memcpy(v90, v103, v102);
        v58 = v102;
        v38 = v90;
      }
      v91 = v58;
      *((_BYTE *)v38 + v58) = 0;
      v38 = v101;
    }
    else
    {
      if ( v90 == (_BYTE *)v92 )
      {
        v90 = v101;
        v91 = v102;
        v92[0] = v103[0];
      }
      else
      {
        v39 = v92[0];
        v90 = v101;
        v91 = v102;
        v92[0] = v103[0];
        if ( v38 )
        {
          v101 = v38;
          v103[0] = v39;
          goto LABEL_49;
        }
      }
      v101 = v103;
      v38 = v103;
    }
LABEL_49:
    v102 = 0;
    *(_BYTE *)v38 = 0;
    if ( v101 != v103 )
      j_j___libc_free_0((unsigned __int64)v101);
    sub_C88F40((__int64)&v107, (__int64)v90, v91, 0);
    a2 = v99;
    nb = sub_C89030((__int64 *)&v107, v99);
    sub_C88FF0(&v107);
    if ( !nb )
      break;
    if ( (_QWORD *)v99[0] != v100 )
    {
      a2 = (unsigned __int64 *)(v100[0] + 1LL);
      j_j___libc_free_0(v99[0]);
    }
LABEL_54:
    if ( (_BYTE *)v113[0] != v114 )
      _libc_free(v113[0]);
    if ( (_BYTE *)v111[0] != v112 )
      _libc_free(v111[0]);
    sub_CAEB90(v9, (unsigned __int64)a2);
    if ( !*(_QWORD *)(v9 + 80) )
      goto LABEL_59;
  }
  sub_8FD6D0((__int64)&v104, "invalid regex: ", v99);
  v107 = (char *)&v104;
  v110 = 260;
  v76 = sub_CAE820(v10, (unsigned __int64)"invalid regex: ", v73, v74, v75);
  sub_CA89D0(v86, v76, (__int64)&v107, 0);
  if ( v104 != v106 )
    j_j___libc_free_0((unsigned __int64)v104);
  if ( (_QWORD *)v99[0] != v100 )
    j_j___libc_free_0(v99[0]);
LABEL_12:
  if ( (_BYTE *)v113[0] != v114 )
    _libc_free(v113[0]);
  if ( (_BYTE *)v111[0] != v112 )
    _libc_free(v111[0]);
LABEL_16:
  v24 = 0;
LABEL_17:
  if ( dest != v98 )
    j_j___libc_free_0((unsigned __int64)dest);
  if ( v93 != v95 )
    j_j___libc_free_0((unsigned __int64)v93);
  if ( v90 != (_BYTE *)v92 )
    j_j___libc_free_0((unsigned __int64)v90);
  return v24;
}
