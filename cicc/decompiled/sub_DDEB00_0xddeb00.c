// Function: sub_DDEB00
// Address: 0xddeb00
//
__int64 *__fastcall sub_DDEB00(__int64 *a1, __int64 **a2, char *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rbp
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r13
  _QWORD *v8; // r12
  __int64 *v10; // r15
  char *v11; // r9
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned __int64 v15; // r10
  char *v16; // rsi
  unsigned int v17; // r14d
  unsigned __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rax
  char *v22; // rax
  _QWORD *v23; // rbx
  char *v24; // r12
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r15
  __int64 v28; // rax
  char *v29; // rax
  __int64 *v30; // rdi
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // r15
  char *v35; // r12
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // r9
  char *v39; // rsi
  char *v40; // r14
  unsigned __int64 v41; // r8
  __int64 v42; // r12
  int v43; // r15d
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 *v48; // rcx
  __int64 v49; // r8
  int v50; // r12d
  char *v51; // r15
  char *v52; // r14
  int v53; // ebx
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  __int64 v57; // rax
  int v58; // ecx
  __int64 v59; // rdi
  int v60; // ecx
  unsigned int v61; // edx
  __int64 ***v62; // rax
  __int64 **v63; // r8
  __int64 v64; // r14
  __int64 v65; // r8
  unsigned __int64 v66; // r9
  unsigned __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // r8
  _BYTE **v70; // r14
  char *v71; // r15
  _BYTE **v72; // r12
  __int64 v73; // rax
  _BYTE *v74; // rbx
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rdx
  char *v78; // rsi
  __int64 v79; // r14
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r14
  __int64 v83; // r13
  char *v84; // r15
  __int64 *v85; // rbx
  __int64 *v86; // rax
  int v87; // eax
  int v88; // r9d
  __int64 **v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // [rsp-B8h] [rbp-B8h]
  unsigned __int64 v95; // [rsp-B8h] [rbp-B8h]
  __int64 v96; // [rsp-B0h] [rbp-B0h]
  _QWORD *v97; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 v98; // [rsp-B0h] [rbp-B0h]
  __int64 v99; // [rsp-B0h] [rbp-B0h]
  __int64 v100; // [rsp-A8h] [rbp-A8h]
  int v101; // [rsp-A8h] [rbp-A8h]
  char v102; // [rsp-A8h] [rbp-A8h]
  __int64 v103; // [rsp-A8h] [rbp-A8h]
  __int64 v104; // [rsp-A8h] [rbp-A8h]
  __int64 v105; // [rsp-A8h] [rbp-A8h]
  __int64 v106; // [rsp-A0h] [rbp-A0h]
  __int64 **v107; // [rsp-98h] [rbp-98h]
  __int64 v108; // [rsp-98h] [rbp-98h]
  int v109; // [rsp-90h] [rbp-90h]
  unsigned __int64 v110; // [rsp-90h] [rbp-90h]
  __int64 v111; // [rsp-90h] [rbp-90h]
  __int64 *v112; // [rsp-88h] [rbp-88h] BYREF
  __int64 v113; // [rsp-80h] [rbp-80h]
  _QWORD v114[15]; // [rsp-78h] [rbp-78h] BYREF

  v114[14] = v4;
  v114[11] = v6;
  v7 = (__int64)a1;
  v114[10] = v5;
  v8 = a2;
  v114[9] = v3;
  switch ( *((_WORD *)a2 + 12) )
  {
    case 0:
    case 1:
      return v8;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
      v32 = sub_D960E0((__int64)a2);
      v110 = v33;
      v101 = v33;
      if ( !(_DWORD)v33 )
        return v8;
      v107 = a2;
      v34 = 0;
      v35 = (char *)v32;
      v36 = (unsigned int)v33;
      while ( 1 )
      {
        v37 = sub_DDF4E0(a1, *(_QWORD *)&v35[8 * v34], a3);
        if ( *(_QWORD *)&v35[8 * v34] != v37 )
          break;
        if ( v36 == ++v34 )
          return (__int64 *)a2;
      }
      v39 = (char *)v114;
      v40 = v35;
      v41 = v34;
      v42 = (__int64)v107;
      v112 = v114;
      v43 = v34 + 1;
      v113 = 0x800000000LL;
      if ( v110 > 8 )
      {
        v95 = v41;
        v99 = v37;
        sub_C8D5F0((__int64)&v112, v114, v110, 8u, v41, v38);
        v41 = v95;
        v37 = v99;
        v39 = (char *)&v112[(unsigned int)v113];
      }
      v96 = v37;
      if ( v110 <= v41 )
        v41 = v110;
      sub_D932D0((__int64)&v112, v39, v40, &v40[8 * v41]);
      sub_D9B3A0((__int64)&v112, v96, v44, v45, v46, v47);
      if ( (_DWORD)v110 != v43 )
      {
        v50 = v43;
        v51 = v40;
        v52 = a3;
        v53 = v101;
        do
        {
          v54 = sub_DDF4E0(a1, *(_QWORD *)&v51[8 * v50], v52);
          v55 = (unsigned int)v113;
          if ( (unsigned __int64)(unsigned int)v113 + 1 > HIDWORD(v113) )
          {
            v104 = v54;
            sub_C8D5F0((__int64)&v112, v114, (unsigned int)v113 + 1LL, 8u, v49, (unsigned int)v113 + 1LL);
            v55 = (unsigned int)v113;
            v54 = v104;
          }
          v48 = v112;
          ++v50;
          v112[v55] = v54;
          LODWORD(v113) = v113 + 1;
        }
        while ( v53 != v50 );
        v42 = (__int64)v107;
      }
      a2 = (__int64 **)v42;
      v56 = sub_DD3B30(a1, v42, &v112, (__int64)v48, v49);
      v30 = v112;
      v8 = (_QWORD *)v56;
      if ( v112 == v114 )
        return v8;
      goto LABEL_20;
    case 8:
      v10 = a2[5];
      v109 = (int)v10;
      if ( !(_DWORD)v10 )
        goto LABEL_63;
      v11 = (char *)a2[4];
      v12 = 0;
      while ( 1 )
      {
        v13 = sub_DDF4E0(a1, *(_QWORD *)&v11[8 * v12], a3);
        v11 = (char *)a2[4];
        if ( *(_QWORD *)&v11[8 * v12] != v13 )
          break;
        if ( ++v12 == (unsigned int)v10 )
          goto LABEL_63;
      }
      v15 = v12;
      v16 = (char *)v114;
      v17 = v12 + 1;
      v18 = v8[5];
      v112 = v114;
      v113 = 0x800000000LL;
      if ( v18 > 8 )
      {
        v98 = v15;
        v103 = v13;
        sub_C8D5F0((__int64)&v112, v114, v18, 8u, v14, (__int64)v11);
        v11 = (char *)v8[4];
        v18 = v8[5];
        v15 = v98;
        v13 = v103;
        v16 = (char *)&v112[(unsigned int)v113];
      }
      v100 = v13;
      if ( v15 <= v18 )
        v18 = v15;
      sub_D932D0((__int64)&v112, v16, v11, &v11[8 * v18]);
      v20 = (unsigned int)v113;
      v21 = v100;
      if ( (unsigned __int64)(unsigned int)v113 + 1 > HIDWORD(v113) )
      {
        sub_C8D5F0((__int64)&v112, v114, (unsigned int)v113 + 1LL, 8u, v19, (unsigned int)v113 + 1LL);
        v20 = (unsigned int)v113;
        v21 = v100;
      }
      v112[v20] = v21;
      LODWORD(v113) = v113 + 1;
      if ( (_DWORD)v10 != v17 )
      {
        v22 = a3;
        v23 = v8;
        v24 = v22;
        do
        {
          v27 = sub_DDF4E0(a1, *(_QWORD *)(v23[4] + 8LL * v17), v24);
          v28 = (unsigned int)v113;
          if ( (unsigned __int64)(unsigned int)v113 + 1 > HIDWORD(v113) )
          {
            sub_C8D5F0((__int64)&v112, v114, (unsigned int)v113 + 1LL, 8u, v25, v26);
            v28 = (unsigned int)v113;
          }
          ++v17;
          v112[v28] = v27;
          LODWORD(v113) = v113 + 1;
        }
        while ( v109 != v17 );
        v29 = v24;
        v8 = v23;
        a3 = v29;
      }
      a2 = &v112;
      v8 = sub_DBFF60((__int64)a1, (unsigned int *)&v112, v8[6], *((_WORD *)v8 + 14) & 1);
      if ( *((_WORD *)v8 + 12) != 8 )
        goto LABEL_19;
      if ( v112 != v114 )
        _libc_free(v112, &v112);
LABEL_63:
      v78 = (char *)v8[6];
      if ( v78 != a3 )
      {
        while ( a3 )
        {
          a3 = *(char **)a3;
          if ( v78 == a3 )
            return v8;
        }
        v79 = sub_DCF3A0(a1, v78, 0);
        if ( v79 != sub_D970F0((__int64)a1) )
          return sub_DD0540((__int64)v8, v79, a1);
      }
      return v8;
    case 0xF:
      v111 = (__int64)*(a2 - 1);
      if ( *(_BYTE *)v111 <= 0x1Cu )
        return v8;
      if ( *(_BYTE *)v111 != 84 )
        goto LABEL_44;
      v57 = a1[6];
      v58 = *(_DWORD *)(v57 + 24);
      a2 = *(__int64 ***)(v111 + 40);
      v59 = *(_QWORD *)(v57 + 8);
      if ( !v58 )
        goto LABEL_44;
      v60 = v58 - 1;
      v61 = v60 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v62 = (__int64 ***)(v59 + 16LL * v61);
      v63 = *v62;
      if ( a2 == *v62 )
        goto LABEL_42;
      v87 = 1;
      while ( 2 )
      {
        if ( v63 == (__int64 **)-4096LL )
          goto LABEL_44;
        v88 = v87 + 1;
        v61 = v60 & (v87 + v61);
        v62 = (__int64 ***)(v59 + 16LL * v61);
        v63 = *v62;
        if ( a2 != *v62 )
        {
          v87 = v88;
          continue;
        }
        break;
      }
LABEL_42:
      v64 = (__int64)v62[1];
      if ( !v64 || a3 != *(char **)v64 || a2 != **(__int64 ****)(v64 + 32) )
        goto LABEL_44;
      a2 = v62[1];
      v108 = sub_DCF3A0((__int64 *)v7, (char *)v64, 0);
      if ( !sub_D968A0(v108) || (*(_DWORD *)(v111 + 4) & 0x7FFFFFF) == 0 )
        goto LABEL_92;
      v81 = 0;
      v106 = v64;
      v105 = v7;
      v82 = v64 + 56;
      v83 = 0;
      v84 = a3;
      v85 = 0;
      break;
    default:
      BUG();
  }
  do
  {
    if ( !(unsigned __int8)sub_B19060(
                             v82,
                             *(_QWORD *)(*(_QWORD *)(v111 - 8) + 32LL * *(unsigned int *)(v111 + 72) + 8 * v83),
                             v80,
                             v81) )
    {
      a2 = *(__int64 ***)(v111 - 8);
      if ( v85 )
      {
        v86 = a2[4 * v83];
        if ( !v86 || v86 != v85 )
        {
          v64 = v106;
          v7 = v105;
          a3 = v84;
          goto LABEL_92;
        }
      }
      else
      {
        v85 = a2[4 * v83];
      }
    }
    ++v83;
  }
  while ( (*(_DWORD *)(v111 + 4) & 0x7FFFFFFu) > (unsigned int)v83 );
  v89 = (__int64 **)v85;
  v64 = v106;
  a3 = v84;
  v7 = v105;
  a2 = v89;
  if ( v89 )
    return sub_DD8400(v7, (__int64)a2);
LABEL_92:
  if ( !sub_D96A50(v108) )
  {
    a2 = (__int64 **)v108;
    if ( (unsigned __int8)sub_DBE090(v7, v108) )
    {
      if ( (*(_DWORD *)(v111 + 4) & 0x7FFFFFF) == 2 )
      {
        a2 = *(__int64 ***)(*(_QWORD *)(v111 - 8)
                          + 32LL
                          * ((unsigned __int8)sub_B19060(
                                                v64 + 56,
                                                *(_QWORD *)(*(_QWORD *)(v111 - 8) + 32LL * *(unsigned int *)(v111 + 72)),
                                                v90,
                                                v91)
                           ^ 1u));
        if ( (unsigned __int8)sub_D48480(v64, (__int64)a2, v92, v93) )
          return sub_DD8400(v7, (__int64)a2);
      }
    }
  }
  if ( !*(_WORD *)(v108 + 24) )
  {
    a2 = (__int64 **)sub_DA8A30(v7, v111, *(_QWORD *)(v108 + 32) + 24LL, v64);
    if ( a2 )
      return sub_DD8400(v7, (__int64)a2);
  }
LABEL_44:
  if ( sub_D90BC0((unsigned __int8 *)v111) )
  {
    v112 = v114;
    v113 = 0x400000000LL;
    v67 = *(_DWORD *)(v111 + 4) & 0x7FFFFFF;
    if ( v67 > 4 )
    {
      a2 = (__int64 **)v114;
      sub_C8D5F0((__int64)&v112, v114, v67, 8u, v65, v66);
      v67 = *(_DWORD *)(v111 + 4) & 0x7FFFFFF;
    }
    v68 = 32 * v67;
    if ( (*(_BYTE *)(v111 + 7) & 0x40) != 0 )
    {
      v69 = *(_QWORD *)(v111 - 8);
      v70 = (_BYTE **)(v69 + v68);
    }
    else
    {
      v70 = (_BYTE **)v111;
      v69 = v111 - v68;
    }
    if ( (_BYTE **)v69 != v70 )
    {
      v97 = v8;
      v71 = a3;
      v72 = (_BYTE **)v69;
      v102 = 0;
      do
      {
        v74 = *v72;
        if ( **v72 <= 0x15u )
        {
          v73 = (unsigned int)v113;
          if ( (unsigned __int64)(unsigned int)v113 + 1 > HIDWORD(v113) )
          {
            a2 = (__int64 **)v114;
            sub_C8D5F0((__int64)&v112, v114, (unsigned int)v113 + 1LL, 8u, v69, v66);
            v73 = (unsigned int)v113;
          }
          v112[v73] = (__int64)v74;
          LODWORD(v113) = v113 + 1;
        }
        else
        {
          a2 = (__int64 **)*((_QWORD *)v74 + 1);
          if ( !sub_D97040(v7, (__int64)a2)
            || (a2 = (__int64 **)sub_DD8400(v7, (__int64)v74),
                v75 = sub_DDF4E0(v7, a2, v71),
                v102 |= a2 != (__int64 **)v75,
                (v76 = sub_D938E0(v75)) == 0) )
          {
            v8 = v97;
            goto LABEL_19;
          }
          v77 = (unsigned int)v113;
          v66 = (unsigned int)v113 + 1LL;
          if ( v66 > HIDWORD(v113) )
          {
            a2 = (__int64 **)v114;
            v94 = v76;
            sub_C8D5F0((__int64)&v112, v114, (unsigned int)v113 + 1LL, 8u, v69, v66);
            v77 = (unsigned int)v113;
            v76 = v94;
          }
          v112[v77] = v76;
          LODWORD(v113) = v113 + 1;
        }
        v72 += 4;
      }
      while ( v70 != v72 );
      v8 = v97;
      if ( v102 )
      {
        a2 = (__int64 **)sub_97D230(
                           (unsigned __int8 *)v111,
                           v112,
                           (unsigned int)v113,
                           *(_BYTE **)(v7 + 8),
                           *(__int64 **)(v7 + 24),
                           0);
        if ( a2 )
          v8 = sub_DD8400(v7, (__int64)a2);
      }
    }
LABEL_19:
    v30 = v112;
    if ( v112 != v114 )
LABEL_20:
      _libc_free(v30, a2);
  }
  return v8;
}
