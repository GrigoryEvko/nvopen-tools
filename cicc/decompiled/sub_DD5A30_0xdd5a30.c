// Function: sub_DD5A30
// Address: 0xdd5a30
//
_QWORD *__fastcall sub_DD5A30(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rsi
  int v6; // eax
  int v7; // edi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r8
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // rcx
  __int64 v21; // rbx
  _QWORD *v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r10
  __int64 v27; // rdi
  unsigned int v28; // r14d
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r15
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  bool v40; // si
  __int64 *v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  _QWORD *v44; // rsi
  __int64 v45; // r14
  __int64 v46; // rcx
  __int64 *v47; // rax
  int v49; // eax
  int v50; // r9d
  __int64 *v51; // rbx
  __int64 *v52; // rax
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // r14
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // r8
  int v69; // r9d
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  _QWORD *v73; // rsi
  unsigned int v74; // r9d
  __int16 v75; // r13
  int v76; // eax
  unsigned __int8 v77; // al
  __int64 v78; // rax
  char v79; // al
  char v80; // al
  __int64 v81; // [rsp+0h] [rbp-140h]
  __int64 v82; // [rsp+18h] [rbp-128h]
  unsigned int v84; // [rsp+28h] [rbp-118h]
  unsigned int v85; // [rsp+28h] [rbp-118h]
  __int64 v86; // [rsp+30h] [rbp-110h]
  __int64 v87; // [rsp+30h] [rbp-110h]
  __int64 v88; // [rsp+30h] [rbp-110h]
  __int64 v89; // [rsp+38h] [rbp-108h]
  __int64 v90; // [rsp+38h] [rbp-108h]
  __int64 v91; // [rsp+38h] [rbp-108h]
  __int64 v92; // [rsp+38h] [rbp-108h]
  __int64 v93; // [rsp+38h] [rbp-108h]
  __int64 v94; // [rsp+38h] [rbp-108h]
  __int64 *v95; // [rsp+48h] [rbp-F8h] BYREF
  _BYTE *v96; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v97; // [rsp+58h] [rbp-E8h]
  _BYTE v98[64]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v99; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v100; // [rsp+A8h] [rbp-98h]
  __int64 v101; // [rsp+B0h] [rbp-90h]
  __int64 v102; // [rsp+B8h] [rbp-88h] BYREF
  unsigned int v103; // [rsp+C0h] [rbp-80h]
  char v104; // [rsp+C8h] [rbp-78h]
  __int64 v105; // [rsp+F8h] [rbp-48h] BYREF
  __int64 v106; // [rsp+100h] [rbp-40h]
  bool v107; // [rsp+108h] [rbp-38h]

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 48);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v3 + 8);
  v6 = *(_DWORD *)(v3 + 24);
  if ( !v6 )
    return 0;
  v7 = v6 - 1;
  v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( v4 != *v9 )
  {
    v49 = 1;
    while ( v10 != -4096 )
    {
      v50 = v49 + 1;
      v8 = v7 & (v49 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v4 == *v9 )
        goto LABEL_3;
      v49 = v50;
    }
    return 0;
  }
LABEL_3:
  v11 = v9[1];
  if ( !v11 || **(_QWORD **)(v11 + 32) != v4 || (*(_DWORD *)(v2 + 4) & 0x7FFFFFF) == 0 )
    return 0;
  v12 = 0;
  v13 = 8LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v14 = 0;
  v15 = 0;
  do
  {
    v16 = *(_QWORD *)(v2 - 8);
    v17 = *(_QWORD *)(v16 + 4 * v12);
    v18 = *(_QWORD *)(32LL * *(unsigned int *)(v2 + 72) + v16 + v12);
    if ( *(_BYTE *)(v11 + 84) )
    {
      v19 = *(_QWORD **)(v11 + 64);
      v20 = &v19[*(unsigned int *)(v11 + 76)];
      if ( v19 != v20 )
      {
        while ( v18 != *v19 )
        {
          if ( v20 == ++v19 )
            goto LABEL_49;
        }
LABEL_12:
        if ( v15 )
        {
          if ( v17 != v15 )
            return 0;
        }
        else
        {
          v15 = v17;
        }
        goto LABEL_14;
      }
    }
    else
    {
      v86 = v13;
      v91 = v14;
      v47 = sub_C8CA60(v11 + 56, v18);
      v14 = v91;
      v13 = v86;
      if ( v47 )
        goto LABEL_12;
    }
LABEL_49:
    if ( v14 )
    {
      if ( v14 != v17 )
        return 0;
    }
    else
    {
      v14 = v17;
    }
LABEL_14:
    v12 += 8;
  }
  while ( v13 != v12 );
  v21 = v15;
  if ( v14 == 0 || v15 == 0 )
    return 0;
  v89 = v14;
  v22 = (_QWORD *)sub_DD8B50(a1, v2, v15, v14);
  if ( v22 )
    return v22;
  v95 = sub_DA3860((_QWORD *)a1, v2);
  sub_DB77A0(a1, v2, (__int64)v95);
  v23 = sub_DD8400(a1, v21);
  v25 = v89;
  v26 = v23;
  if ( *(_WORD *)(v23 + 24) != 5 )
  {
    v100 = 0;
    v51 = &v102;
    v101 = 1;
    v99 = a1;
    v52 = &v102;
    do
    {
      *v52 = -4096;
      v52 += 2;
    }
    while ( v52 != &v105 );
    v105 = v11;
    LOBYTE(v106) = 1;
    v53 = sub_DD3C70((__int64)&v99, v26, v24, (__int64)&v105, v89);
    v54 = v89;
    v55 = v53;
    if ( !(_BYTE)v106 )
    {
      v67 = sub_D970F0(a1);
      v54 = v89;
      v55 = v67;
    }
    if ( (v101 & 1) == 0 )
    {
      v88 = v54;
      sub_C7D6A0(v102, 16LL * v103, 8);
      v54 = v88;
    }
    v100 = 0;
    v101 = 1;
    v99 = a1;
    do
    {
      *v51 = -4096;
      v51 += 2;
    }
    while ( v51 != &v105 );
    v105 = v11;
    v87 = v54;
    LOWORD(v106) = 0;
    v56 = sub_DD45D0((__int64)&v99, v55);
    v57 = v87;
    v58 = v56;
    if ( (_WORD)v106 )
    {
      v59 = sub_D970F0(a1);
      v57 = v87;
      v58 = v59;
    }
    if ( (v101 & 1) == 0 )
    {
      v92 = v57;
      sub_C7D6A0(v102, 16LL * v103, 8);
      v57 = v92;
    }
    v93 = v57;
    if ( v55 != sub_D970F0(a1)
      && v58 != sub_D970F0(a1)
      && (unsigned __int8)sub_DBEBD0(a1, v55)
      && (unsigned __int8)sub_D9B790(v55, v58, v60, v61, v62, v63)
      && v58 == sub_DD8400(a1, v93) )
    {
      v22 = (_QWORD *)v55;
      sub_DAB940(a1, (__int64)&v95, 1, v64, v65, v66);
      sub_DB77A0(a1, v2, v55);
      return v22;
    }
    goto LABEL_47;
  }
  v27 = *(_QWORD *)(v23 + 40);
  v28 = v27;
  if ( (_DWORD)v27 )
  {
    v29 = *(_QWORD *)(v23 + 32);
    v30 = 0;
    while ( 1 )
    {
      v28 = v30;
      if ( v95 == *(__int64 **)(v29 + 8 * v30) )
        break;
      v28 = ++v30;
      if ( (unsigned int)v27 == v30 )
      {
        v30 = v28;
        break;
      }
    }
  }
  else
  {
    v30 = 0;
  }
  if ( v27 == v30 )
  {
LABEL_47:
    sub_D98440(a1, v2);
    return v22;
  }
  v96 = v98;
  v97 = 0x800000000LL;
  if ( (_DWORD)v27 )
  {
    v81 = v2;
    v82 = v21;
    v31 = v26;
    v32 = 0;
    while ( v28 == (_DWORD)v32 )
    {
LABEL_40:
      if ( (unsigned int)v27 == ++v32 )
      {
        v21 = v82;
        v25 = v89;
        v22 = 0;
        v2 = v81;
        goto LABEL_42;
      }
    }
    v33 = *(_QWORD *)(*(_QWORD *)(v31 + 32) + 8 * v32);
    v34 = sub_D47930(v11);
    v37 = v34;
    if ( v34 )
    {
      v38 = v34 + 48;
      v39 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v39 == v38 )
        goto LABEL_106;
      if ( !v39 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v39 - 24) - 30 > 0xA )
LABEL_106:
        BUG();
      if ( *(_BYTE *)(v39 - 24) == 31 && (*(_DWORD *)(v39 - 20) & 0x7FFFFFF) == 3 )
      {
        v37 = *(_QWORD *)(v39 - 120);
        v40 = **(_QWORD **)(v11 + 32) == *(_QWORD *)(v39 - 56);
        goto LABEL_33;
      }
    }
    else
    {
      v40 = 0;
LABEL_33:
      v100 = 0;
      v101 = 1;
      v99 = a1;
      v41 = &v102;
      do
      {
        *v41 = -4096;
        v41 += 2;
      }
      while ( v41 != &v105 );
      v107 = v40;
      v105 = v11;
      v106 = v37;
      v33 = sub_DD4F00(&v99, v33);
      if ( (v101 & 1) == 0 )
        sub_C7D6A0(v102, 16LL * v103, 8);
    }
    v42 = (unsigned int)v97;
    v43 = (unsigned int)v97 + 1LL;
    if ( v43 > HIDWORD(v97) )
    {
      sub_C8D5F0((__int64)&v96, v98, v43, 8u, v35, v36);
      v42 = (unsigned int)v97;
    }
    *(_QWORD *)&v96[8 * v42] = v33;
    LODWORD(v97) = v97 + 1;
    goto LABEL_40;
  }
LABEL_42:
  v90 = v25;
  v44 = sub_DC7EB0((__int64 *)a1, (__int64)&v96, 0, 0);
  v45 = (__int64)v44;
  if ( !sub_DADE90(a1, (__int64)v44, v11) && (*((_WORD *)v44 + 12) != 8 || v44[6] != v11) )
  {
    if ( v96 != v98 )
      _libc_free(v96, v44);
    goto LABEL_47;
  }
  sub_D94080((__int64)&v99, (unsigned __int8 *)v21, *(__int64 **)(a1 + 40), v46, v90);
  v68 = v90;
  if ( v104 )
  {
    v69 = 0;
    if ( (_DWORD)v99 == 13 && v100 == v2 )
    {
      v69 = 2 * (BYTE1(v102) != 0);
      if ( (_BYTE)v102 )
        v69 |= 4u;
    }
    goto LABEL_80;
  }
  v77 = *(_BYTE *)v21;
  if ( *(_BYTE *)v21 <= 0x1Cu )
  {
    v69 = 0;
    if ( v77 != 5 || *(_WORD *)(v21 + 2) != 34 )
      goto LABEL_80;
  }
  else if ( v77 != 63 )
  {
LABEL_89:
    v69 = 0;
    goto LABEL_80;
  }
  v78 = *(_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
  if ( !v78 || v2 != v78 )
    goto LABEL_89;
  v69 = 0;
  v79 = *(_BYTE *)(v21 + 1) >> 1;
  if ( v79 )
  {
    if ( (v79 & 4) != 0 || (v79 & 2) != 0 && (v80 = sub_DBED40(a1, (__int64)v44), v68 = v90, v80) )
      v69 = 3;
    else
      v69 = 1;
  }
LABEL_80:
  v84 = v69;
  v94 = sub_DD8400(a1, v68);
  v22 = sub_DC1960(a1, v94, (__int64)v44, v11, v84);
  sub_DAB940(a1, (__int64)&v95, 1, v70, v71, v72);
  v73 = (_QWORD *)v2;
  sub_DB77A0(a1, v2, (__int64)v22);
  v74 = v84;
  if ( *((_WORD *)v22 + 12) == 8 )
  {
    v75 = *((_WORD *)v22 + 14);
    v76 = sub_DCF420((__int64 *)a1, (__int64)v22);
    v73 = v22;
    sub_D97270(a1, (__int64)v22, v75 & 7 | v76);
    v74 = v84;
  }
  v85 = v74;
  if ( *(_BYTE *)v21 > 0x1Cu )
  {
    v73 = (_QWORD *)v45;
    if ( sub_DADE90(a1, v45, v11) )
    {
      v73 = (_QWORD *)v21;
      if ( (unsigned __int8)sub_DD8750(a1, v21, v11) )
      {
        v73 = sub_DC7ED0((__int64 *)a1, v94, v45, 0, 0);
        sub_DC1960(a1, (__int64)v73, v45, v11, v85);
      }
    }
  }
  if ( v96 != v98 )
    _libc_free(v96, v73);
  return v22;
}
