// Function: sub_2D352C0
// Address: 0x2d352c0
//
__int64 __fastcall sub_2D352C0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned int v14; // esi
  unsigned int v15; // r13d
  _DWORD *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  unsigned int v21; // esi
  _QWORD *v22; // r14
  unsigned int v23; // eax
  _DWORD *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // eax
  unsigned int v29; // esi
  _DWORD *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned int v34; // eax
  unsigned int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned int v39; // esi
  __int64 v40; // rdx
  __int64 v41; // r8
  __int64 v42; // r9
  _DWORD *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned int v49; // r14d
  __int64 v50; // rax
  int v51; // edx
  unsigned int v52; // esi
  _BYTE *v53; // r14
  unsigned __int64 v54; // rax
  unsigned int *v55; // r8
  __int64 v56; // rax
  int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rax
  int v60; // edx
  unsigned int v61; // esi
  unsigned int *v63; // rax
  __int64 v64; // r8
  __int64 v65; // r9
  unsigned int *v66; // r14
  unsigned int *v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  unsigned int *v70; // r13
  unsigned int *v71; // rax
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rax
  unsigned int v75; // r13d
  unsigned __int64 *v76; // rax
  __int64 v77; // rsi
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // r14
  unsigned int *v80; // r13
  unsigned int *v81; // rax
  __int64 v82; // r8
  __int64 v83; // r9
  _DWORD *v85; // [rsp+48h] [rbp-298h]
  unsigned int v86; // [rsp+50h] [rbp-290h]
  __int64 v87; // [rsp+50h] [rbp-290h]
  unsigned int v88; // [rsp+58h] [rbp-288h]
  unsigned int v89; // [rsp+58h] [rbp-288h]
  __int64 v90; // [rsp+58h] [rbp-288h]
  unsigned int v91; // [rsp+58h] [rbp-288h]
  unsigned int v92; // [rsp+58h] [rbp-288h]
  bool v93; // [rsp+60h] [rbp-280h]
  unsigned int v94; // [rsp+60h] [rbp-280h]
  unsigned int v95; // [rsp+60h] [rbp-280h]
  unsigned int *v96; // [rsp+60h] [rbp-280h]
  unsigned int v97; // [rsp+60h] [rbp-280h]
  __int64 v98; // [rsp+68h] [rbp-278h]
  __int64 v99; // [rsp+70h] [rbp-270h] BYREF
  _BYTE *v100; // [rsp+78h] [rbp-268h] BYREF
  __int64 v101; // [rsp+80h] [rbp-260h]
  _BYTE v102[72]; // [rsp+88h] [rbp-258h] BYREF
  __int64 v103; // [rsp+D0h] [rbp-210h] BYREF
  _BYTE *v104; // [rsp+D8h] [rbp-208h]
  __int64 v105; // [rsp+E0h] [rbp-200h]
  _BYTE v106[72]; // [rsp+E8h] [rbp-1F8h] BYREF
  _DWORD *v107; // [rsp+130h] [rbp-1B0h] BYREF
  _BYTE *v108; // [rsp+138h] [rbp-1A8h] BYREF
  __int64 v109; // [rsp+140h] [rbp-1A0h]
  _BYTE v110[72]; // [rsp+148h] [rbp-198h] BYREF
  _DWORD *v111; // [rsp+190h] [rbp-150h] BYREF
  _BYTE *v112; // [rsp+198h] [rbp-148h]
  __int64 v113; // [rsp+1A0h] [rbp-140h]
  _BYTE v114[72]; // [rsp+1A8h] [rbp-138h] BYREF
  _DWORD *v115; // [rsp+1F0h] [rbp-F0h] BYREF
  _BYTE *v116; // [rsp+1F8h] [rbp-E8h] BYREF
  __int64 v117; // [rsp+200h] [rbp-E0h]
  _BYTE v118[72]; // [rsp+208h] [rbp-D8h] BYREF
  _DWORD *v119; // [rsp+250h] [rbp-90h] BYREF
  _QWORD *v120; // [rsp+258h] [rbp-88h]
  __int64 v121; // [rsp+260h] [rbp-80h]
  _DWORD *v122; // [rsp+268h] [rbp-78h] BYREF
  unsigned __int64 v123; // [rsp+270h] [rbp-70h]

  *(_QWORD *)(a1 + 200) = a2 + 32;
  v9 = a1 + 8;
  *(_QWORD *)(v9 - 8) = 0;
  *(_QWORD *)(v9 + 176) = 0;
  memset(
    (void *)(v9 & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8 * ((unsigned __int64)((unsigned int)a1 - (v9 & 0xFFFFFFF8) + 192) >> 3));
  *(_QWORD *)(a1 + 192) = 0;
  v100 = v102;
  v101 = 0x400000000LL;
  v99 = a3;
  sub_2D29C80((__int64)&v99, 0, a3, 0, a5, a6);
  v13 = *(unsigned int *)(v99 + 192);
  if ( (_DWORD)v13 )
  {
    v10 = (unsigned int)v101;
    v74 = (unsigned int)(v101 - 1);
    if ( (unsigned int)v13 > (unsigned int)v74 )
    {
      v75 = *(_DWORD *)(v99 + 192);
      do
      {
        v77 = (__int64)v100;
        v78 = v10 + 1;
        v11 = *(_QWORD *)(*(_QWORD *)&v100[16 * v74] + 8LL * *(unsigned int *)&v100[16 * v74 + 12]) & 0x3FLL;
        v79 = *(_QWORD *)(*(_QWORD *)&v100[16 * v74] + 8LL * *(unsigned int *)&v100[16 * v74 + 12])
            & 0xFFFFFFFFFFFFFFC0LL;
        v13 = v11 + 1;
        if ( v78 > HIDWORD(v101) )
        {
          v98 = v11 + 1;
          sub_C8D5F0((__int64)&v100, v102, v78, 0x10u, v13, v12);
          v77 = (__int64)v100;
          v13 = v98;
        }
        v76 = (unsigned __int64 *)(v77 + 16LL * (unsigned int)v101);
        *v76 = v79;
        v76[1] = v13;
        v74 = (unsigned int)v101;
        v10 = (unsigned int)(v101 + 1);
        LODWORD(v101) = v101 + 1;
      }
      while ( v75 > (unsigned int)v74 );
    }
  }
  v14 = *(_DWORD *)(a3 + 196);
  v103 = a3;
  v104 = v106;
  v105 = 0x400000000LL;
  sub_2D29C80((__int64)&v103, v14, v10, v11, v13, v12);
  while ( 1 )
  {
    v93 = sub_2D28840((__int64)&v99, (__int64)&v103);
    if ( v93 )
      break;
    v15 = *(_DWORD *)(*(_QWORD *)&v100[16 * (unsigned int)v101 - 16]
                    + 8LL * *(unsigned int *)&v100[16 * (unsigned int)v101 - 4]
                    + 4);
    v18 = *(unsigned int *)sub_2D289F0((__int64)&v99);
    v119 = a4;
    v120 = &v122;
    v121 = 0x400000000LL;
    v20 = a4[48];
    if ( v20 )
    {
      sub_2D2BC70((__int64)&v119, v18, (__int64)v16, v17, v18, v19);
    }
    else
    {
      v21 = a4[49];
      if ( v21 )
      {
        v16 = a4 + 1;
        while ( (unsigned int)v18 >= *v16 )
        {
          ++v20;
          v16 += 2;
          if ( v21 == v20 )
            goto LABEL_10;
        }
        v21 = v20;
      }
LABEL_10:
      sub_2D29C80((__int64)&v119, v21, (__int64)v16, v17, v18, v19);
    }
    v22 = v120;
    if ( !(_DWORD)v121 || *((_DWORD *)v120 + 3) >= *((_DWORD *)v120 + 2) )
    {
      if ( v120 != &v122 )
        _libc_free((unsigned __int64)v120);
      goto LABEL_43;
    }
    v23 = *(_DWORD *)sub_2D289F0((__int64)&v119);
    if ( v22 != &v122 )
    {
      v88 = v23;
      _libc_free((unsigned __int64)v22);
      v23 = v88;
    }
    if ( v23 >= v15 )
      goto LABEL_43;
    v26 = *(unsigned int *)sub_2D289F0((__int64)&v99);
    v107 = a4;
    v108 = v110;
    v109 = 0x400000000LL;
    v28 = a4[48];
    if ( v28 )
    {
      sub_2D2BC70((__int64)&v107, v26, (__int64)v24, v25, v26, v27);
    }
    else
    {
      v29 = a4[49];
      if ( v29 )
      {
        v24 = a4 + 1;
        while ( (unsigned int)v26 >= *v24 )
        {
          ++v28;
          v24 += 2;
          if ( v29 == v28 )
            goto LABEL_22;
        }
        v29 = v28;
      }
LABEL_22:
      sub_2D29C80((__int64)&v107, v29, (__int64)v24, v25, v26, v27);
    }
    v89 = *(_DWORD *)sub_2D289F0((__int64)&v107);
    v86 = *(_DWORD *)sub_2D289F0((__int64)&v99);
    v32 = *(unsigned int *)sub_2D28A10((__int64)&v99);
    v111 = a4;
    v112 = v114;
    v113 = 0x400000000LL;
    v34 = a4[48];
    if ( v34 )
    {
      sub_2D2BC70((__int64)&v111, v32, (__int64)v30, v31, v32, v33);
    }
    else
    {
      v35 = a4[49];
      if ( v35 )
      {
        v30 = a4 + 1;
        while ( (unsigned int)v32 >= *v30 )
        {
          ++v34;
          v30 += 2;
          if ( v35 == v34 )
            goto LABEL_29;
        }
        v35 = v34;
      }
LABEL_29:
      sub_2D29C80((__int64)&v111, v35, (__int64)v30, v31, v32, v33);
    }
    v39 = a4[49];
    v119 = a4;
    v120 = &v122;
    v121 = 0x400000000LL;
    sub_2D29C80((__int64)&v119, v39, v36, 0x400000000LL, v37, v38);
    if ( sub_2D28840((__int64)&v111, (__int64)&v119)
      || (v85 = (_DWORD *)sub_2D289F0((__int64)&v111),
          v43 = (_DWORD *)sub_2D28A10((__int64)&v99),
          v40 = (__int64)v85,
          *v85 >= *v43) )
    {
      if ( v120 != &v122 )
        _libc_free((unsigned __int64)v120);
      v44 = (unsigned int)v109;
      v115 = v107;
      v116 = v118;
      v117 = 0x400000000LL;
      if ( (_DWORD)v109 )
        goto LABEL_87;
      if ( v89 >= v86 )
        goto LABEL_52;
    }
    else
    {
      if ( v120 != &v122 )
        _libc_free((unsigned __int64)v120);
      v44 = v86;
      if ( v89 >= v86 )
      {
        v115 = v107;
        v116 = v118;
        v117 = 0x400000000LL;
        if ( !(_DWORD)v109 )
        {
LABEL_76:
          v95 = *(_DWORD *)sub_2D28A30((__int64)&v99);
          if ( v95 && v95 == *(_DWORD *)sub_2D28A30((__int64)&v111) )
          {
            v92 = v95;
            v96 = (unsigned int *)sub_2D28A10((__int64)&v99);
            v63 = (unsigned int *)sub_2D289F0((__int64)&v111);
            sub_2D35160(a1, *v63, *v96, v92, v64, v65);
          }
          goto LABEL_52;
        }
        v93 = 1;
LABEL_87:
        sub_2D23820((__int64)&v116, (__int64)&v108, v40, v44, v41, v42);
        if ( v89 >= v86 )
          goto LABEL_75;
        goto LABEL_73;
      }
      if ( sub_2D28840((__int64)&v107, (__int64)&v111) )
      {
        v49 = *(_DWORD *)sub_2D28A30((__int64)&v99);
        if ( v49 && v49 == *(_DWORD *)sub_2D28A30((__int64)&v107) )
        {
          v80 = (unsigned int *)sub_2D28A10((__int64)&v99);
          v81 = (unsigned int *)sub_2D289F0((__int64)&v99);
          sub_2D35160(a1, *v81, *v80, v49, v82, v83);
        }
        goto LABEL_39;
      }
      v93 = 1;
      v115 = v107;
      v116 = v118;
      v117 = 0x400000000LL;
      if ( (_DWORD)v109 )
        sub_2D23820((__int64)&v116, (__int64)&v108, v45, v46, v47, v48);
    }
LABEL_73:
    v91 = *(_DWORD *)sub_2D28A30((__int64)&v99);
    if ( v91 && v91 == *(_DWORD *)sub_2D28A30((__int64)&v107) )
    {
      v70 = (unsigned int *)sub_2D28A10((__int64)&v107);
      v71 = (unsigned int *)sub_2D289F0((__int64)&v99);
      sub_2D35160(a1, *v71, *v70, v91, v72, v73);
    }
    sub_2D23A60((__int64)&v115);
LABEL_75:
    if ( v93 )
      goto LABEL_76;
LABEL_52:
    v53 = v116;
    while ( 1 )
    {
      v119 = a4;
      v120 = &v122;
      v121 = 0x400000000LL;
      v54 = ((unsigned __int64)(unsigned int)a4[49] << 32) | (unsigned int)a4[49];
      if ( a4[48] )
      {
        v123 = ((unsigned __int64)(unsigned int)a4[49] << 32) | (unsigned int)a4[49];
        LODWORD(v121) = 1;
        v122 = a4 + 2;
      }
      else
      {
        v122 = a4;
        v123 = v54;
        LODWORD(v121) = 1;
      }
      if ( sub_2D28840((__int64)&v115, (__int64)&v119) )
        break;
      v55 = (unsigned int *)sub_2D289F0((__int64)&v115);
      v56 = (__int64)&v100[16 * (unsigned int)v101 - 16];
      v87 = *(_QWORD *)v56;
      v90 = *(unsigned int *)(v56 + 12);
      v94 = *(_DWORD *)(*(_QWORD *)v56 + 8 * v90 + 4);
      if ( *v55 >= v94 || *(_DWORD *)sub_2D28A10((__int64)&v115) > v94 )
        break;
      v57 = *(_DWORD *)(v87 + 4 * v90 + 128);
      if ( v57 )
      {
        v58 = (unsigned int)v117;
        if ( *(_DWORD *)(*(_QWORD *)&v53[16 * (unsigned int)v117 - 16]
                       + 4LL * *(unsigned int *)&v53[16 * (unsigned int)v117 - 4]
                       + 128) != v57 )
          goto LABEL_60;
        v97 = *(_DWORD *)(*(_QWORD *)&v53[16 * (unsigned int)v117 - 16]
                        + 4LL * *(unsigned int *)&v53[16 * (unsigned int)v117 - 4]
                        + 128);
        v66 = (unsigned int *)sub_2D28A10((__int64)&v115);
        v67 = (unsigned int *)sub_2D289F0((__int64)&v115);
        sub_2D35160(a1, *v67, *v66, v97, v68, v69);
        v53 = v116;
      }
      v58 = (unsigned int)v117;
LABEL_60:
      v59 = (__int64)&v53[16 * v58 - 16];
      v60 = *(_DWORD *)(v59 + 12) + 1;
      *(_DWORD *)(v59 + 12) = v60;
      v53 = v116;
      if ( v60 == *(_DWORD *)&v116[16 * (unsigned int)v117 - 8] )
      {
        v61 = v115[48];
        if ( v61 )
        {
          sub_F03D40((__int64 *)&v116, v61);
          v53 = v116;
        }
      }
    }
    if ( v53 != v118 )
      _libc_free((unsigned __int64)v53);
LABEL_39:
    if ( v112 != v114 )
      _libc_free((unsigned __int64)v112);
    if ( v108 != v110 )
      _libc_free((unsigned __int64)v108);
LABEL_43:
    v50 = (__int64)&v100[16 * (unsigned int)v101 - 16];
    v51 = *(_DWORD *)(v50 + 12) + 1;
    *(_DWORD *)(v50 + 12) = v51;
    if ( v51 == *(_DWORD *)&v100[16 * (unsigned int)v101 - 8] )
    {
      v52 = *(_DWORD *)(v99 + 192);
      if ( v52 )
        sub_F03D40((__int64 *)&v100, v52);
    }
  }
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  return a1;
}
