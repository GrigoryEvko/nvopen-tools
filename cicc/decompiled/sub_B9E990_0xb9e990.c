// Function: sub_B9E990
// Address: 0xb9e990
//
__int64 __fastcall sub_B9E990(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 v3; // al
  bool v4; // dl
  unsigned int v5; // ecx
  unsigned int v6; // ecx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  int v12; // ecx
  __int64 **v13; // rdx
  __int64 **v14; // rsi
  __int64 **v15; // rax
  int v16; // edx
  int v17; // edx
  __int64 v18; // rdx
  unsigned __int8 v19; // al
  bool v20; // dl
  unsigned int v21; // ecx
  unsigned int v22; // ecx
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // rsi
  int v28; // ecx
  __int64 **v29; // rdx
  __int64 **v30; // rsi
  __int64 **v31; // rax
  int v32; // edx
  int v33; // edx
  __int64 v34; // rdx
  __int64 **v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // r14
  __int64 **v38; // r13
  __int64 **v39; // r12
  __int64 **v40; // rbx
  unsigned int v41; // eax
  unsigned __int64 v42; // r8
  __int64 **v43; // r14
  unsigned int v44; // eax
  unsigned int v45; // eax
  _BYTE *v46; // rsi
  unsigned int v47; // eax
  __int64 v48; // rbx
  unsigned __int64 v49; // r15
  _BYTE *v50; // r12
  __int64 **v51; // r14
  unsigned int v52; // eax
  unsigned __int64 v53; // r8
  _BYTE *v54; // rbx
  unsigned int v55; // eax
  unsigned int v56; // eax
  __int64 *v57; // rsi
  __int64 *v58; // rbx
  __int64 v59; // r13
  __int64 *v60; // rdi
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 *v66; // rdi
  __int64 v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // rdx
  __int64 *v70; // rdx
  __int64 *v71; // rdi
  __int64 v72; // r13
  __int64 *v73; // r12
  __int64 *v74; // rbx
  __int64 v75; // rdi
  __int64 v76; // rbx
  _BYTE *v77; // r12
  __int64 v78; // rdi
  __int64 **v79; // rbx
  __int64 **v80; // r12
  __int64 *v81; // rdi
  __int64 **v82; // rbx
  __int64 **v83; // r12
  __int64 *v84; // rdi
  __int64 **v85; // rbx
  __int64 **v86; // r12
  __int64 *v87; // rdi
  __int64 v89; // r12
  unsigned __int64 v90; // r12
  _QWORD *v92; // [rsp+10h] [rbp-1F0h]
  _QWORD *v93; // [rsp+10h] [rbp-1F0h]
  __int64 v95; // [rsp+30h] [rbp-1D0h]
  __int64 v96; // [rsp+30h] [rbp-1D0h]
  __int64 **v97; // [rsp+30h] [rbp-1D0h]
  __int64 **v98; // [rsp+30h] [rbp-1D0h]
  __int64 v99; // [rsp+38h] [rbp-1C8h]
  __int64 v100; // [rsp+38h] [rbp-1C8h]
  __int64 **v101; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v102; // [rsp+48h] [rbp-1B8h]
  _BYTE v103[32]; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 **v104; // [rsp+70h] [rbp-190h] BYREF
  __int64 v105; // [rsp+78h] [rbp-188h]
  _BYTE v106[32]; // [rsp+80h] [rbp-180h] BYREF
  __int64 *v107; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v108; // [rsp+A8h] [rbp-158h]
  _BYTE v109[48]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 **v110; // [rsp+E0h] [rbp-120h] BYREF
  __int64 v111; // [rsp+E8h] [rbp-118h]
  _BYTE v112[64]; // [rsp+F0h] [rbp-110h] BYREF
  _BYTE *v113; // [rsp+130h] [rbp-D0h] BYREF
  __int64 v114; // [rsp+138h] [rbp-C8h]
  _BYTE v115[64]; // [rsp+140h] [rbp-C0h] BYREF
  __int64 *v116; // [rsp+180h] [rbp-80h] BYREF
  unsigned int v117; // [rsp+188h] [rbp-78h]
  __int64 v118; // [rsp+190h] [rbp-70h] BYREF
  unsigned int v119; // [rsp+198h] [rbp-68h]

  if ( !a1 )
    return 0;
  v2 = a2;
  if ( !a2 )
    return 0;
  if ( a1 == a2 )
    return a1;
  v101 = (__int64 **)v103;
  v102 = 0x100000000LL;
  v105 = 0x100000000LL;
  v3 = *(_BYTE *)(a1 - 16);
  v104 = (__int64 **)v106;
  v4 = (v3 & 2) != 0;
  if ( (v3 & 2) != 0 )
    v5 = *(_DWORD *)(a1 - 24);
  else
    v5 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v6 = v5 >> 1;
  if ( v6 )
  {
    v7 = 0;
    v99 = 16LL * (v6 - 1);
    while ( 1 )
    {
      if ( v4 )
        v8 = *(_QWORD *)(a1 - 32);
      else
        v8 = a1 + -16 - 8LL * ((v3 >> 2) & 0xF);
      v18 = *(_QWORD *)(*(_QWORD *)(v8 + v7) + 136LL);
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + v7 + 8) + 136LL);
      LODWORD(v114) = *(_DWORD *)(v9 + 32);
      if ( (unsigned int)v114 > 0x40 )
      {
        v95 = v18;
        sub_C43780(&v113, v9 + 24);
        v18 = v95;
        LODWORD(v111) = *(_DWORD *)(v95 + 32);
        if ( (unsigned int)v111 <= 0x40 )
        {
LABEL_11:
          v110 = *(__int64 ***)(v18 + 24);
          goto LABEL_12;
        }
      }
      else
      {
        v113 = *(_BYTE **)(v9 + 24);
        LODWORD(v111) = *(_DWORD *)(v18 + 32);
        if ( (unsigned int)v111 <= 0x40 )
          goto LABEL_11;
      }
      sub_C43780(&v110, v18 + 24);
LABEL_12:
      sub_AADC30((__int64)&v116, (__int64)&v110, (__int64 *)&v113);
      v10 = (unsigned int)v102;
      v11 = (unsigned int)v102 + 1LL;
      v12 = v102;
      if ( v11 > HIDWORD(v102) )
      {
        if ( v101 > &v116 || &v116 >= &v101[4 * (unsigned int)v102] )
        {
          sub_9D5330((__int64)&v101, v11);
          v10 = (unsigned int)v102;
          v13 = v101;
          v14 = &v116;
          v12 = v102;
        }
        else
        {
          v98 = v101;
          sub_9D5330((__int64)&v101, v11);
          v13 = v101;
          v10 = (unsigned int)v102;
          v14 = (__int64 **)((char *)v101 + (char *)&v116 - (char *)v98);
          v12 = v102;
        }
      }
      else
      {
        v13 = v101;
        v14 = &v116;
      }
      v15 = &v13[4 * v10];
      if ( v15 )
      {
        v16 = *((_DWORD *)v14 + 2);
        *((_DWORD *)v14 + 2) = 0;
        *((_DWORD *)v15 + 2) = v16;
        *v15 = *v14;
        v17 = *((_DWORD *)v14 + 6);
        *((_DWORD *)v14 + 6) = 0;
        v12 = v102;
        *((_DWORD *)v15 + 6) = v17;
        v15[2] = v14[2];
      }
      LODWORD(v102) = v12 + 1;
      if ( v119 > 0x40 && v118 )
        j_j___libc_free_0_0(v118);
      if ( v117 > 0x40 && v116 )
        j_j___libc_free_0_0(v116);
      if ( (unsigned int)v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      if ( (unsigned int)v114 > 0x40 && v113 )
        j_j___libc_free_0_0(v113);
      if ( v7 == v99 )
      {
        v2 = a2;
        break;
      }
      v3 = *(_BYTE *)(a1 - 16);
      v7 += 16;
      v4 = (v3 & 2) != 0;
    }
  }
  v19 = *(_BYTE *)(v2 - 16);
  v20 = (v19 & 2) != 0;
  if ( (v19 & 2) != 0 )
    v21 = *(_DWORD *)(v2 - 24);
  else
    v21 = (*(_WORD *)(v2 - 16) >> 6) & 0xF;
  v22 = v21 >> 1;
  if ( v22 )
  {
    v23 = 0;
    v100 = 16LL * (v22 - 1);
    while ( 1 )
    {
      if ( v20 )
        v24 = *(_QWORD *)(v2 - 32);
      else
        v24 = v2 + -16 - 8LL * ((v19 >> 2) & 0xF);
      v34 = *(_QWORD *)(*(_QWORD *)(v24 + v23) + 136LL);
      v25 = *(_QWORD *)(*(_QWORD *)(v24 + v23 + 8) + 136LL);
      LODWORD(v114) = *(_DWORD *)(v25 + 32);
      if ( (unsigned int)v114 > 0x40 )
      {
        v96 = v34;
        sub_C43780(&v113, v25 + 24);
        v34 = v96;
        LODWORD(v111) = *(_DWORD *)(v96 + 32);
        if ( (unsigned int)v111 <= 0x40 )
        {
LABEL_43:
          v110 = *(__int64 ***)(v34 + 24);
          goto LABEL_44;
        }
      }
      else
      {
        v113 = *(_BYTE **)(v25 + 24);
        LODWORD(v111) = *(_DWORD *)(v34 + 32);
        if ( (unsigned int)v111 <= 0x40 )
          goto LABEL_43;
      }
      sub_C43780(&v110, v34 + 24);
LABEL_44:
      sub_AADC30((__int64)&v116, (__int64)&v110, (__int64 *)&v113);
      v26 = (unsigned int)v105;
      v27 = (unsigned int)v105 + 1LL;
      v28 = v105;
      if ( v27 > HIDWORD(v105) )
      {
        if ( v104 > &v116 || (v97 = v104, &v116 >= &v104[4 * (unsigned int)v105]) )
        {
          sub_9D5330((__int64)&v104, v27);
          v26 = (unsigned int)v105;
          v29 = v104;
          v30 = &v116;
          v28 = v105;
        }
        else
        {
          sub_9D5330((__int64)&v104, v27);
          v29 = v104;
          v26 = (unsigned int)v105;
          v30 = (__int64 **)((char *)v104 + (char *)&v116 - (char *)v97);
          v28 = v105;
        }
      }
      else
      {
        v29 = v104;
        v30 = &v116;
      }
      v31 = &v29[4 * v26];
      if ( v31 )
      {
        v32 = *((_DWORD *)v30 + 2);
        *((_DWORD *)v30 + 2) = 0;
        *((_DWORD *)v31 + 2) = v32;
        *v31 = *v30;
        v33 = *((_DWORD *)v30 + 6);
        *((_DWORD *)v30 + 6) = 0;
        v28 = v105;
        *((_DWORD *)v31 + 6) = v33;
        v31[2] = v30[2];
      }
      LODWORD(v105) = v28 + 1;
      if ( v119 > 0x40 && v118 )
        j_j___libc_free_0_0(v118);
      if ( v117 > 0x40 && v116 )
        j_j___libc_free_0_0(v116);
      if ( (unsigned int)v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      if ( (unsigned int)v114 > 0x40 && v113 )
        j_j___libc_free_0_0(v113);
      if ( v100 == v23 )
        break;
      v19 = *(_BYTE *)(v2 - 16);
      v23 += 16;
      v20 = (v19 & 2) != 0;
    }
  }
  v35 = (__int64 **)v112;
  v110 = (__int64 **)v112;
  v111 = 0x200000000LL;
  if ( v101 != &v101[4 * (unsigned int)v102] )
  {
    v36 = 0;
    v37 = 0;
    v38 = v101 + 4;
    v39 = v101;
    v40 = &v101[4 * (unsigned int)v102];
    while ( 1 )
    {
      v43 = &v35[4 * v37];
      if ( v43 )
      {
        v44 = *((_DWORD *)v39 + 2);
        *((_DWORD *)v43 + 2) = v44;
        if ( v44 <= 0x40 )
        {
          *(_QWORD *)v43 = *v39;
          v41 = *((_DWORD *)v39 + 6);
          *((_DWORD *)v43 + 6) = v41;
          if ( v41 <= 0x40 )
            goto LABEL_69;
        }
        else
        {
          sub_C43780((__int64 **)v43, v39);
          v45 = *((_DWORD *)v39 + 6);
          *((_DWORD *)v43 + 6) = v45;
          if ( v45 <= 0x40 )
          {
LABEL_69:
            *((_QWORD *)v43 + 2) = v39[2];
            v36 = v111;
            goto LABEL_70;
          }
        }
        sub_C43780(v43 + 16, v39 + 2);
        v36 = v111;
      }
LABEL_70:
      ++v36;
      v39 = v38;
      LODWORD(v111) = v36;
      if ( v40 == v38 )
        break;
      v37 = v36;
      v35 = v110;
      v42 = v36 + 1LL;
      if ( v42 > HIDWORD(v111) )
      {
        if ( v110 > v38 || &v110[4 * v36] <= v38 )
        {
          sub_9D5330((__int64)&v110, v42);
          v37 = (unsigned int)v111;
          v35 = v110;
          v36 = v111;
        }
        else
        {
          v89 = (char *)v38 - (char *)v110;
          sub_9D5330((__int64)&v110, v42);
          v35 = v110;
          v37 = (unsigned int)v111;
          v39 = (__int64 **)((char *)v110 + v89);
          v36 = v111;
        }
      }
      v38 += 4;
    }
  }
  v46 = v115;
  v113 = v115;
  v114 = 0x200000000LL;
  if ( v104 != &v104[4 * (unsigned int)v105] )
  {
    v47 = 0;
    v48 = 0;
    v49 = (unsigned __int64)(v104 + 4);
    v50 = v104;
    v51 = &v104[4 * (unsigned int)v105];
    while ( 1 )
    {
      v54 = &v46[32 * v48];
      if ( v54 )
      {
        v55 = *((_DWORD *)v50 + 2);
        *((_DWORD *)v54 + 2) = v55;
        if ( v55 <= 0x40 )
        {
          *(_QWORD *)v54 = *(_QWORD *)v50;
          v52 = *((_DWORD *)v50 + 6);
          *((_DWORD *)v54 + 6) = v52;
          if ( v52 <= 0x40 )
            goto LABEL_80;
        }
        else
        {
          sub_C43780(v54, v50);
          v56 = *((_DWORD *)v50 + 6);
          *((_DWORD *)v54 + 6) = v56;
          if ( v56 <= 0x40 )
          {
LABEL_80:
            *((_QWORD *)v54 + 2) = *((_QWORD *)v50 + 2);
            v47 = v114;
            goto LABEL_81;
          }
        }
        sub_C43780(v54 + 16, v50 + 16);
        v47 = v114;
      }
LABEL_81:
      ++v47;
      v50 = (_BYTE *)v49;
      LODWORD(v114) = v47;
      if ( v51 == (__int64 **)v49 )
        break;
      v48 = v47;
      v46 = v113;
      v53 = v47 + 1LL;
      if ( v53 > HIDWORD(v114) )
      {
        if ( (unsigned __int64)v113 > v49 || (unsigned __int64)&v113[32 * v47] <= v49 )
        {
          sub_9D5330((__int64)&v113, v53);
          v48 = (unsigned int)v114;
          v46 = v113;
          v47 = v114;
        }
        else
        {
          v90 = v49 - (_QWORD)v113;
          sub_9D5330((__int64)&v113, v53);
          v46 = v113;
          v48 = (unsigned int)v114;
          v50 = &v113[v90];
          v47 = v114;
        }
      }
      v49 += 32LL;
    }
  }
  v57 = (__int64 *)&v110;
  sub_ABFB50(&v116, (__int64)&v110, (__int64)&v113);
  if ( v117 )
  {
    v107 = (__int64 *)v109;
    v108 = 0x600000000LL;
    v58 = &v116[4 * v117];
    v59 = (__int64)v116;
    do
    {
      v60 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
        v60 = (__int64 *)*v60;
      v61 = sub_ACCFD0(v60, v59);
      v62 = sub_B98A20(v61, v59);
      v63 = (unsigned int)v108;
      if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
      {
        v93 = v62;
        sub_C8D5F0(&v107, v109, (unsigned int)v108 + 1LL, 8);
        v63 = (unsigned int)v108;
        v62 = v93;
      }
      v64 = v59 + 16;
      v107[v63] = (__int64)v62;
      v65 = *(_QWORD *)(a1 + 8);
      LODWORD(v108) = v108 + 1;
      v66 = (__int64 *)(v65 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v65 & 4) != 0 )
        v66 = (__int64 *)*v66;
      v67 = sub_ACCFD0(v66, v64);
      v68 = sub_B98A20(v67, v64);
      v69 = (unsigned int)v108;
      if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
      {
        v92 = v68;
        sub_C8D5F0(&v107, v109, (unsigned int)v108 + 1LL, 8);
        v69 = (unsigned int)v108;
        v68 = v92;
      }
      v59 += 32;
      v107[v69] = (__int64)v68;
      v70 = (__int64 *)(unsigned int)(v108 + 1);
      LODWORD(v108) = v108 + 1;
    }
    while ( v58 != (__int64 *)v59 );
    v57 = v107;
    v71 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
      v71 = (__int64 *)*v71;
    v72 = sub_B9C770(v71, v107, v70, 0, 1);
    if ( v107 != (__int64 *)v109 )
      _libc_free(v107, v57);
    v73 = v116;
    v74 = &v116[4 * v117];
    if ( v116 == v74 )
      goto LABEL_112;
    do
    {
      v74 -= 4;
      if ( *((_DWORD *)v74 + 6) > 0x40u )
      {
        v75 = v74[2];
        if ( v75 )
          j_j___libc_free_0_0(v75);
      }
      if ( *((_DWORD *)v74 + 2) > 0x40u && *v74 )
        j_j___libc_free_0_0(*v74);
    }
    while ( v73 != v74 );
  }
  else
  {
    v72 = 0;
  }
  v73 = v116;
LABEL_112:
  if ( v73 != &v118 )
    _libc_free(v73, v57);
  v76 = (__int64)v113;
  v77 = &v113[32 * (unsigned int)v114];
  if ( v113 != v77 )
  {
    do
    {
      v77 -= 32;
      if ( *((_DWORD *)v77 + 6) > 0x40u )
      {
        v78 = *((_QWORD *)v77 + 2);
        if ( v78 )
          j_j___libc_free_0_0(v78);
      }
      if ( *((_DWORD *)v77 + 2) > 0x40u && *(_QWORD *)v77 )
        j_j___libc_free_0_0(*(_QWORD *)v77);
    }
    while ( (_BYTE *)v76 != v77 );
    v77 = v113;
  }
  if ( v77 != v115 )
    _libc_free(v77, v57);
  v79 = v110;
  v80 = &v110[4 * (unsigned int)v111];
  if ( v110 != v80 )
  {
    do
    {
      v80 -= 4;
      if ( *((_DWORD *)v80 + 6) > 0x40u )
      {
        v81 = (__int64)v80[2];
        if ( v81 )
          j_j___libc_free_0_0((__int64 *)v81);
      }
      if ( *((_DWORD *)v80 + 2) > 0x40u && *v80 )
        j_j___libc_free_0_0(*v80);
    }
    while ( v79 != v80 );
    v80 = v110;
  }
  if ( v80 != (__int64 **)v112 )
    _libc_free(v80, v57);
  v82 = v104;
  v83 = &v104[4 * (unsigned int)v105];
  if ( v104 != v83 )
  {
    do
    {
      v83 -= 4;
      if ( *((_DWORD *)v83 + 6) > 0x40u )
      {
        v84 = v83[2];
        if ( v84 )
          j_j___libc_free_0_0(v84);
      }
      if ( *((_DWORD *)v83 + 2) > 0x40u && *v83 )
        j_j___libc_free_0_0(*v83);
    }
    while ( v82 != v83 );
    v83 = v104;
  }
  if ( v83 != (__int64 **)v106 )
    _libc_free(v83, v57);
  v85 = v101;
  v86 = &v101[4 * (unsigned int)v102];
  if ( v101 != v86 )
  {
    do
    {
      v86 -= 4;
      if ( *((_DWORD *)v86 + 6) > 0x40u )
      {
        v87 = v86[2];
        if ( v87 )
          j_j___libc_free_0_0(v87);
      }
      if ( *((_DWORD *)v86 + 2) > 0x40u && *v86 )
        j_j___libc_free_0_0(*v86);
    }
    while ( v85 != v86 );
    v86 = v101;
  }
  if ( v86 != (__int64 **)v103 )
    _libc_free(v86, v57);
  return v72;
}
