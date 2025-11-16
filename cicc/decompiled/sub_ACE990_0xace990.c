// Function: sub_ACE990
// Address: 0xace990
//
__int64 __fastcall sub_ACE990(__int64 *a1, __int64 a2)
{
  int v3; // r13d
  __int64 v5; // rax
  unsigned __int8 *v6; // r12
  __int64 v7; // r15
  bool v8; // di
  int v9; // ecx
  __int64 v10; // rsi
  __int64 v11; // rdx
  char v12; // r9
  unsigned int v14; // eax
  char **v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r8
  char *v19; // rdx
  char *v20; // r12
  __int64 *v21; // r15
  __int64 *v22; // r13
  _BYTE *v23; // rax
  _QWORD *v24; // r14
  __int64 v25; // rax
  size_t v26; // r13
  char *v27; // r14
  __int64 v28; // rax
  int v29; // eax
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  int v33; // eax
  __int64 *v34; // rax
  unsigned __int8 v35; // al
  __int64 *v36; // r15
  __int64 *v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  char *v40; // rdx
  __int64 *v41; // r15
  __int64 *v42; // r14
  _BYTE *v43; // rdx
  _QWORD *v44; // rax
  __int16 v45; // r13
  __int64 v46; // rax
  __int64 v47; // r13
  char *v48; // r14
  __int64 v49; // rax
  char *v50; // rdx
  __int64 *v51; // r15
  __int64 *v52; // r14
  _BYTE *v53; // rdx
  _QWORD *v54; // rax
  int v55; // r13d
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // r13
  char *v59; // r14
  __int64 v60; // rax
  __int64 *v61; // r15
  __int64 *v62; // r13
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  int v66; // eax
  char *v67; // rdx
  __int64 *v68; // r15
  __int64 *v69; // r14
  _BYTE *v70; // rax
  _QWORD *v71; // r13
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // r13
  char *v75; // r14
  __int64 v76; // rax
  __int64 *v77; // r15
  __int64 *v78; // r13
  __int64 v79; // rax
  char *v80; // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rdx
  unsigned int v83; // edx
  int v84; // eax
  _BYTE *v85; // [rsp+8h] [rbp-D8h]
  unsigned int v86; // [rsp+8h] [rbp-D8h]
  _BYTE *v87; // [rsp+8h] [rbp-D8h]
  unsigned int v88; // [rsp+8h] [rbp-D8h]
  _BYTE *v89; // [rsp+8h] [rbp-D8h]
  unsigned int v90; // [rsp+8h] [rbp-D8h]
  unsigned int v91; // [rsp+8h] [rbp-D8h]
  unsigned int v92; // [rsp+8h] [rbp-D8h]
  __int64 v93; // [rsp+8h] [rbp-D8h]
  char **v94; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v95; // [rsp+18h] [rbp-C8h]
  char *v96; // [rsp+20h] [rbp-C0h] BYREF
  size_t i; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v98; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE v99[168]; // [rsp+38h] [rbp-A8h] BYREF

  v3 = a2;
  v5 = sub_BCDA70(*(_QWORD *)(*a1 + 8), a2);
  v6 = (unsigned __int8 *)*a1;
  v7 = v5;
  v8 = sub_AC30F0(*a1);
  v9 = *v6;
  v10 = (unsigned int)(v9 - 12);
  LOBYTE(v11) = (unsigned int)v10 <= 1;
  v12 = qword_4F81228 & ((_BYTE)v9 == 18);
  if ( ((unsigned __int8)qword_4F81308 & ((_BYTE)v9 == 17)) == 0 )
  {
    LOBYTE(v11) = v8 | v11;
    if ( !(_BYTE)v11 )
    {
      if ( !v12 )
      {
        if ( (_BYTE)v9 == 13 )
          return sub_ACADE0((__int64 **)v7);
LABEL_10:
        v15 = (char **)*((_QWORD *)v6 + 1);
        v16 = 0;
        if ( sub_AC5240((__int64)v15) )
        {
          if ( *v6 == 17 )
          {
            v10 = 8;
            if ( (unsigned __int8)sub_BCAC40(*((_QWORD *)v6 + 1), 8) )
            {
              v20 = v99;
              v21 = &a1[a2];
              i = 0;
              v22 = a1;
              v96 = v99;
              v98 = 16;
              if ( a1 == v21 )
              {
LABEL_20:
                v26 = i;
                v27 = v96;
                v28 = sub_BD5C60(*a1, v10, v19);
                v10 = (__int64)v27;
                v16 = sub_AC99A0(v28, v27, v26);
                goto LABEL_21;
              }
              while ( 1 )
              {
                v23 = (_BYTE *)*v22;
                if ( *(_BYTE *)*v22 != 17 )
                  goto LABEL_50;
                v24 = (_QWORD *)*((_QWORD *)v23 + 3);
                if ( *((_DWORD *)v23 + 8) > 0x40u )
                  v24 = (_QWORD *)*v24;
                v25 = i;
                if ( i + 1 > v98 )
                {
                  v10 = (__int64)v99;
                  sub_C8D290(&v96, v99, i + 1, 1);
                  v25 = i;
                }
                v19 = v96;
                ++v22;
                v96[v25] = (char)v24;
                ++i;
                if ( v21 == v22 )
                  goto LABEL_20;
              }
            }
            v10 = 16;
            if ( (unsigned __int8)sub_BCAC40(*((_QWORD *)v6 + 1), 16) )
            {
              v20 = v99;
              v41 = &a1[a2];
              i = 0;
              v42 = a1;
              v96 = v99;
              v98 = 16;
              if ( a1 != v41 )
              {
                while ( 1 )
                {
                  v43 = (_BYTE *)*v42;
                  if ( *(_BYTE *)*v42 != 17 )
                    break;
                  v44 = (_QWORD *)*((_QWORD *)v43 + 3);
                  if ( *((_DWORD *)v43 + 8) > 0x40u )
                    v44 = (_QWORD *)*v44;
                  v45 = (__int16)v44;
                  v46 = i;
                  if ( i + 1 > v98 )
                  {
                    v10 = (__int64)v99;
                    sub_C8D290(&v96, v99, i + 1, 2);
                    v46 = i;
                  }
                  v40 = v96;
                  ++v42;
                  *(_WORD *)&v96[2 * v46] = v45;
                  ++i;
                  if ( v41 == v42 )
                    goto LABEL_59;
                }
LABEL_50:
                v16 = 0;
                goto LABEL_21;
              }
LABEL_59:
              v47 = i;
              v48 = v96;
              v49 = sub_BD5C60(*a1, v10, v40);
              v10 = (__int64)v48;
              v16 = sub_AC99E0(v49, v48, v47);
              goto LABEL_21;
            }
            v10 = 32;
            if ( (unsigned __int8)sub_BCAC40(*((_QWORD *)v6 + 1), 32) )
            {
              v20 = (char *)&v98;
              v51 = &a1[a2];
              v52 = a1;
              v96 = (char *)&v98;
              for ( i = 0x1000000000LL; v51 != v52; LODWORD(i) = i + 1 )
              {
                v53 = (_BYTE *)*v52;
                if ( *(_BYTE *)*v52 != 17 )
                  goto LABEL_50;
                v54 = (_QWORD *)*((_QWORD *)v53 + 3);
                if ( *((_DWORD *)v53 + 8) > 0x40u )
                  v54 = (_QWORD *)*v54;
                v55 = (int)v54;
                v56 = (unsigned int)i;
                v57 = (unsigned int)i + 1LL;
                if ( v57 > HIDWORD(i) )
                {
                  v10 = (__int64)&v98;
                  sub_C8D5F0(&v96, &v98, v57, 4);
                  v56 = (unsigned int)i;
                }
                v50 = v96;
                ++v52;
                *(_DWORD *)&v96[4 * v56] = v55;
              }
              v58 = (unsigned int)i;
              v59 = v96;
              v60 = sub_BD5C60(*a1, v10, v50);
              v10 = (__int64)v59;
              v16 = sub_AC9A10(v60, v59, v58);
LABEL_21:
              if ( v96 != v20 )
                _libc_free(v96, v10);
              return v16;
            }
            v10 = 64;
            if ( (unsigned __int8)sub_BCAC40(*((_QWORD *)v6 + 1), 64) )
            {
              v20 = (char *)&v98;
              v68 = &a1[a2];
              v69 = a1;
              v96 = (char *)&v98;
              for ( i = 0x1000000000LL; v68 != v69; LODWORD(i) = i + 1 )
              {
                v70 = (_BYTE *)*v69;
                if ( *(_BYTE *)*v69 != 17 )
                  goto LABEL_50;
                v71 = (_QWORD *)*((_QWORD *)v70 + 3);
                if ( *((_DWORD *)v70 + 8) > 0x40u )
                  v71 = (_QWORD *)*v71;
                v72 = (unsigned int)i;
                v73 = (unsigned int)i + 1LL;
                if ( v73 > HIDWORD(i) )
                {
                  v10 = (__int64)&v98;
                  sub_C8D5F0(&v96, &v98, v73, 8);
                  v72 = (unsigned int)i;
                }
                v67 = v96;
                ++v69;
                *(_QWORD *)&v96[8 * v72] = v71;
              }
              v74 = (unsigned int)i;
              v75 = v96;
              v76 = sub_BD5C60(*a1, v10, v67);
              v10 = (__int64)v75;
              v16 = sub_AC9A50(v76, v75, v74);
              goto LABEL_21;
            }
          }
          else if ( *v6 == 18 )
          {
            v35 = *(_BYTE *)(*((_QWORD *)v6 + 1) + 8LL);
            if ( v35 <= 1u )
            {
              v20 = v99;
              v36 = &a1[a2];
              i = 0;
              v37 = a1;
              v96 = v99;
              v98 = 16;
              if ( a1 == v36 )
              {
LABEL_49:
                v10 = (__int64)v96;
                v16 = sub_AC9A90(*(_QWORD *)(*a1 + 8), v96, i);
                goto LABEL_21;
              }
              while ( 1 )
              {
                v85 = (_BYTE *)*v37;
                if ( *(_BYTE *)*v37 != 18 )
                  goto LABEL_50;
                v38 = sub_C33340(v15, v10, *v37, v17, v18);
                v15 = (char **)&v94;
                v10 = (__int64)(v85 + 24);
                if ( *((_QWORD *)v85 + 3) == v38 )
                  sub_C3E660(&v94, v10);
                else
                  sub_C3A850(&v94, v10);
                v86 = v95;
                if ( v95 > 0x40 )
                {
                  v15 = (char **)&v94;
                  v66 = sub_C444A0(&v94);
                  v18 = 0xFFFFFFFFLL;
                  if ( v86 - v66 <= 0x40 )
                    v18 = *(unsigned __int16 *)v94;
                }
                else
                {
                  v18 = (unsigned __int16)v94;
                }
                v39 = i;
                if ( i + 1 > v98 )
                {
                  v15 = &v96;
                  v10 = (__int64)v99;
                  v91 = v18;
                  sub_C8D290(&v96, v99, i + 1, 2);
                  v39 = i;
                  v18 = v91;
                }
                *(_WORD *)&v96[2 * v39] = v18;
                ++i;
                if ( v95 > 0x40 )
                {
                  v15 = v94;
                  if ( v94 )
                    j_j___libc_free_0_0(v94);
                }
                if ( v36 == ++v37 )
                  goto LABEL_49;
              }
            }
            if ( v35 == 2 )
            {
              v20 = (char *)&v98;
              v77 = &a1[a2];
              v78 = a1;
              v96 = (char *)&v98;
              for ( i = 0x1000000000LL; v77 != v78; ++v78 )
              {
                v89 = (_BYTE *)*v78;
                if ( *(_BYTE *)*v78 != 18 )
                  goto LABEL_50;
                v79 = sub_C33340(v15, v10, *v78, v17, v18);
                v15 = (char **)&v94;
                v10 = (__int64)(v89 + 24);
                if ( *((_QWORD *)v89 + 3) == v79 )
                  sub_C3E660(&v94, v10);
                else
                  sub_C3A850(&v94, v10);
                v90 = v95;
                if ( v95 > 0x40 )
                {
                  v15 = (char **)&v94;
                  v83 = v90 - sub_C444A0(&v94);
                  LODWORD(v80) = -1;
                  if ( v83 <= 0x40 )
                    v80 = *v94;
                }
                else
                {
                  LODWORD(v80) = (_DWORD)v94;
                }
                v18 = (unsigned int)v80;
                v81 = (unsigned int)i;
                v17 = HIDWORD(i);
                v82 = (unsigned int)i + 1LL;
                if ( v82 > HIDWORD(i) )
                {
                  v15 = &v96;
                  v10 = (__int64)&v98;
                  v92 = v18;
                  sub_C8D5F0(&v96, &v98, v82, 4);
                  v81 = (unsigned int)i;
                  v18 = v92;
                }
                *(_DWORD *)&v96[4 * v81] = v18;
                LODWORD(i) = i + 1;
                if ( v95 > 0x40 )
                {
                  v15 = v94;
                  if ( v94 )
                    j_j___libc_free_0_0(v94);
                }
              }
              v10 = (__int64)v96;
              v16 = sub_AC9AC0(*(_QWORD *)(*a1 + 8), v96, (unsigned int)i);
              goto LABEL_21;
            }
            if ( v35 == 3 )
            {
              v20 = (char *)&v98;
              v61 = &a1[a2];
              v62 = a1;
              v96 = (char *)&v98;
              for ( i = 0x1000000000LL; v61 != v62; ++v62 )
              {
                v87 = (_BYTE *)*v62;
                if ( *(_BYTE *)*v62 != 18 )
                  goto LABEL_50;
                v63 = sub_C33340(v15, v10, *v62, v17, v18);
                v15 = (char **)&v94;
                v10 = (__int64)(v87 + 24);
                if ( *((_QWORD *)v87 + 3) == v63 )
                  sub_C3E660(&v94, v10);
                else
                  sub_C3A850(&v94, v10);
                v88 = v95;
                if ( v95 > 0x40 )
                {
                  v15 = (char **)&v94;
                  v84 = sub_C444A0(&v94);
                  v18 = -1;
                  if ( v88 - v84 <= 0x40 )
                    v18 = (__int64)*v94;
                }
                else
                {
                  v18 = (__int64)v94;
                }
                v64 = (unsigned int)i;
                v17 = HIDWORD(i);
                v65 = (unsigned int)i + 1LL;
                if ( v65 > HIDWORD(i) )
                {
                  v15 = &v96;
                  v10 = (__int64)&v98;
                  v93 = v18;
                  sub_C8D5F0(&v96, &v98, v65, 8);
                  v64 = (unsigned int)i;
                  v18 = v93;
                }
                *(_QWORD *)&v96[8 * v64] = v18;
                LODWORD(i) = i + 1;
                if ( v95 > 0x40 )
                {
                  v15 = v94;
                  if ( v94 )
                    j_j___libc_free_0_0(v94);
                }
              }
              v10 = (__int64)v96;
              v16 = sub_AC9AF0(*(_QWORD *)(*a1 + 8), v96, (unsigned int)i);
              goto LABEL_21;
            }
          }
          return 0;
        }
        return v16;
      }
      if ( (_DWORD)a2 == 1 )
        goto LABEL_29;
LABEL_7:
      v14 = 1;
      while ( 1 )
      {
        v11 = v14;
        if ( v6 != (unsigned __int8 *)a1[v14] )
          goto LABEL_10;
        if ( v3 == ++v14 )
          goto LABEL_24;
      }
    }
  }
  if ( (_DWORD)a2 != 1 )
    goto LABEL_7;
LABEL_24:
  if ( v8 )
    return sub_AC9350((__int64 **)v7);
  if ( (_BYTE)v9 == 13 )
    return sub_ACADE0((__int64 **)v7);
  if ( (unsigned int)v10 <= 1 )
    return sub_ACA8A0((__int64 **)v7);
  if ( v12 )
  {
LABEL_29:
    v29 = *(_DWORD *)(v7 + 32);
    BYTE4(v94) = *(_BYTE *)(v7 + 8) == 18;
    LODWORD(v94) = v29;
    v30 = (__int64 *)sub_BD5C60(v6, v10, v11);
    return sub_ACE4D0(v30, (__int64)v94, (_QWORD *)v6 + 3, v31, v32);
  }
  if ( ((unsigned __int8)qword_4F81308 & ((_BYTE)v9 == 17)) == 0 )
    goto LABEL_10;
  v33 = *(_DWORD *)(v7 + 32);
  BYTE4(v96) = *(_BYTE *)(v7 + 8) == 18;
  LODWORD(v96) = v33;
  v34 = (__int64 *)sub_BD5C60(v6, v10, v11);
  return sub_ACD980(v34, (__int64)v96, (__int64)(v6 + 24));
}
