// Function: sub_20CB200
// Address: 0x20cb200
//
__int64 **__fastcall sub_20CB200(
        __int64 **a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 ***a5,
        int a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v13; // r14
  _QWORD *v14; // r14
  __int64 v15; // rax
  __int64 **v16; // rax
  __int64 **v17; // r15
  __int64 **v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 **v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r12
  bool v25; // zf
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // r12
  bool v30; // cc
  __int64 *v31; // r12
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 *v35; // r15
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 *v48; // r15
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rsi
  unsigned __int8 *v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rdi
  unsigned __int64 *v56; // r14
  __int64 v57; // rax
  unsigned __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 *v64; // r12
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rsi
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 *v72; // r15
  __int64 v73; // rax
  __int64 v74; // rcx
  __int64 v75; // rsi
  __int64 v76; // rsi
  unsigned __int8 *v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rdi
  __int64 *v80; // r15
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rsi
  __int64 v84; // rsi
  unsigned __int8 *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 v88; // r9
  __int64 *v89; // r15
  __int64 v90; // rcx
  __int64 v91; // rax
  __int64 v92; // rsi
  __int64 v93; // rsi
  __int64 v94; // r15
  unsigned __int8 *v95; // rsi
  __int64 v96; // rax
  __int64 v97; // rdi
  unsigned __int64 v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // rsi
  __int64 v102; // rdx
  unsigned __int8 *v103; // rsi
  __int64 v104; // [rsp+8h] [rbp-D8h]
  __int64 v105; // [rsp+8h] [rbp-D8h]
  __int64 v106; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v107; // [rsp+18h] [rbp-C8h]
  _BYTE *v108; // [rsp+20h] [rbp-C0h]
  unsigned __int64 *v109; // [rsp+28h] [rbp-B8h]
  __int64 v111; // [rsp+30h] [rbp-B0h]
  __int64 v112; // [rsp+38h] [rbp-A8h]
  __int64 *v113; // [rsp+38h] [rbp-A8h]
  __int64 v114; // [rsp+48h] [rbp-98h] BYREF
  __int64 v115[2]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v116; // [rsp+60h] [rbp-80h]
  __int64 v117[2]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v118; // [rsp+80h] [rbp-60h]
  _QWORD v119[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v120; // [rsp+A0h] [rbp-40h]

  v13 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 56LL);
  v112 = sub_15F2050(a3);
  v14 = (_QWORD *)sub_15E0530(v13);
  v108 = (_BYTE *)sub_1632FA0(v112);
  v15 = sub_127FA20((__int64)v108, a4);
  a1[1] = (__int64 *)a4;
  v107 = (unsigned __int64)(v15 + 7) >> 3;
  v113 = (__int64 *)sub_1644C60(v14, 8 * a6);
  *a1 = v113;
  v16 = *a5;
  if ( *((_BYTE *)*a5 + 8) == 16 )
    v16 = (__int64 **)*v16[2];
  v17 = (__int64 **)sub_1647190(v113, *((_DWORD *)v16 + 2) >> 8);
  v118 = 257;
  v18 = (__int64 **)sub_15A9620((__int64)v108, (__int64)v14, 0);
  if ( v18 != *a5 )
  {
    if ( *((_BYTE *)a5 + 16) > 0x10u )
    {
      v120 = 257;
      v96 = sub_15FDBD0(45, (__int64)a5, (__int64)v18, (__int64)v119, 0);
      v97 = a2[1];
      a5 = (__int64 ***)v96;
      if ( v97 )
      {
        v109 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v97 + 40, v96);
        v98 = *v109;
        v99 = (unsigned __int64)a5[3] & 7;
        a5[4] = (__int64 **)v109;
        v98 &= 0xFFFFFFFFFFFFFFF8LL;
        a5[3] = (__int64 **)(v98 | v99);
        *(_QWORD *)(v98 + 8) = a5 + 3;
        *v109 = *v109 & 7 | (unsigned __int64)(a5 + 3);
      }
      sub_164B780((__int64)a5, v117);
      v100 = *a2;
      if ( *a2 )
      {
        v115[0] = *a2;
        sub_1623A60((__int64)v115, v100, 2);
        v101 = (__int64)a5[6];
        v102 = (__int64)(a5 + 6);
        if ( v101 )
        {
          sub_161E7C0((__int64)(a5 + 6), v101);
          v102 = (__int64)(a5 + 6);
        }
        v103 = (unsigned __int8 *)v115[0];
        a5[6] = (__int64 **)v115[0];
        if ( v103 )
          sub_1623210((__int64)v115, v103, v102);
      }
    }
    else
    {
      a5 = (__int64 ***)sub_15A46C0(45, a5, v18, 0);
    }
  }
  v117[0] = (__int64)"AlignedAddr";
  v116 = 257;
  v118 = 259;
  v19 = sub_15A0680((__int64)*a5, ~(unsigned __int64)(unsigned int)(a6 - 1), 0);
  v20 = sub_1281C00(a2, (__int64)a5, v19, (__int64)v115);
  v21 = v20;
  if ( v17 != *(__int64 ***)v20 )
  {
    if ( *(_BYTE *)(v20 + 16) > 0x10u )
    {
      v120 = 257;
      v86 = sub_15FDBD0(46, v20, (__int64)v17, (__int64)v119, 0);
      v87 = a2[1];
      v88 = v86;
      if ( v87 )
      {
        v89 = (__int64 *)a2[2];
        v104 = v86;
        sub_157E9D0(v87 + 40, v86);
        v88 = v104;
        v90 = *v89;
        v91 = *(_QWORD *)(v104 + 24);
        *(_QWORD *)(v104 + 32) = v89;
        v90 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v104 + 24) = v90 | v91 & 7;
        *(_QWORD *)(v90 + 8) = v104 + 24;
        *v89 = *v89 & 7 | (v104 + 24);
      }
      v105 = v88;
      sub_164B780(v88, v117);
      v92 = *a2;
      v21 = v105;
      if ( *a2 )
      {
        v114 = *a2;
        sub_1623A60((__int64)&v114, v92, 2);
        v21 = v105;
        v93 = *(_QWORD *)(v105 + 48);
        v94 = v105 + 48;
        if ( v93 )
        {
          sub_161E7C0(v105 + 48, v93);
          v21 = v105;
        }
        v95 = (unsigned __int8 *)v114;
        *(_QWORD *)(v21 + 48) = v114;
        if ( v95 )
        {
          v106 = v21;
          sub_1623210((__int64)&v114, v95, v94);
          v21 = v106;
        }
      }
    }
    else
    {
      v21 = sub_15A46C0(46, (__int64 ***)v20, v17, 0);
    }
  }
  v119[0] = "PtrLSB";
  v120 = 259;
  v22 = *a5;
  a1[2] = (__int64 *)v21;
  v23 = sub_15A0680((__int64)v22, (unsigned int)(a6 - 1), 0);
  v24 = sub_1281C00(a2, (__int64)a5, v23, (__int64)v119);
  v25 = *v108 == 0;
  v118 = 257;
  if ( v25 )
  {
    v26 = sub_15A0680(*(_QWORD *)v24, 3, 0);
    if ( *(_BYTE *)(v24 + 16) > 0x10u || *(_BYTE *)(v26 + 16) > 0x10u )
    {
      v120 = 257;
      v33 = sub_15FB440(23, (__int64 *)v24, v26, (__int64)v119, 0);
      v34 = a2[1];
      v27 = v33;
      if ( v34 )
      {
        v35 = (__int64 *)a2[2];
        sub_157E9D0(v34 + 40, v33);
        v36 = *(_QWORD *)(v27 + 24);
        v37 = *v35;
        *(_QWORD *)(v27 + 32) = v35;
        v37 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v27 + 24) = v37 | v36 & 7;
        *(_QWORD *)(v37 + 8) = v27 + 24;
        *v35 = *v35 & 7 | (v27 + 24);
      }
      sub_164B780(v27, v117);
      v38 = *a2;
      if ( *a2 )
      {
        v115[0] = *a2;
        sub_1623A60((__int64)v115, v38, 2);
        v39 = *(_QWORD *)(v27 + 48);
        if ( v39 )
          sub_161E7C0(v27 + 48, v39);
        v40 = (unsigned __int8 *)v115[0];
        *(_QWORD *)(v27 + 48) = v115[0];
        if ( v40 )
          sub_1623210((__int64)v115, v40, v27 + 48);
      }
    }
    else
    {
      v27 = sub_15A2D50((__int64 *)v24, v26, 0, 0, a7, a8, a9);
    }
  }
  else
  {
    v116 = 257;
    v41 = sub_15A0680(*(_QWORD *)v24, (unsigned int)(a6 - v107), 0);
    v42 = v41;
    if ( *(_BYTE *)(v24 + 16) > 0x10u
      || *(_BYTE *)(v41 + 16) > 0x10u
      || (v111 = v41,
          v43 = sub_15A2A30((__int64 *)0x1C, (__int64 *)v24, v41, 0, 0, a7, a8, a9),
          v42 = v111,
          (v44 = v43) == 0) )
    {
      v120 = 257;
      v62 = sub_15FB440(28, (__int64 *)v24, v42, (__int64)v119, 0);
      v63 = a2[1];
      v44 = v62;
      if ( v63 )
      {
        v64 = (__int64 *)a2[2];
        sub_157E9D0(v63 + 40, v62);
        v65 = *(_QWORD *)(v44 + 24);
        v66 = *v64;
        *(_QWORD *)(v44 + 32) = v64;
        v66 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v44 + 24) = v66 | v65 & 7;
        *(_QWORD *)(v66 + 8) = v44 + 24;
        *v64 = *v64 & 7 | (v44 + 24);
      }
      sub_164B780(v44, v115);
      v67 = *a2;
      if ( *a2 )
      {
        v114 = *a2;
        sub_1623A60((__int64)&v114, v67, 2);
        v68 = *(_QWORD *)(v44 + 48);
        if ( v68 )
          sub_161E7C0(v44 + 48, v68);
        v69 = (unsigned __int8 *)v114;
        *(_QWORD *)(v44 + 48) = v114;
        if ( v69 )
          sub_1623210((__int64)&v114, v69, v44 + 48);
      }
    }
    v45 = sub_15A0680(*(_QWORD *)v44, 3, 0);
    if ( *(_BYTE *)(v44 + 16) > 0x10u || *(_BYTE *)(v45 + 16) > 0x10u )
    {
      v120 = 257;
      v70 = sub_15FB440(23, (__int64 *)v44, v45, (__int64)v119, 0);
      v71 = a2[1];
      v27 = v70;
      if ( v71 )
      {
        v72 = (__int64 *)a2[2];
        sub_157E9D0(v71 + 40, v70);
        v73 = *(_QWORD *)(v27 + 24);
        v74 = *v72;
        *(_QWORD *)(v27 + 32) = v72;
        v74 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v27 + 24) = v74 | v73 & 7;
        *(_QWORD *)(v74 + 8) = v27 + 24;
        *v72 = *v72 & 7 | (v27 + 24);
      }
      sub_164B780(v27, v117);
      v75 = *a2;
      if ( *a2 )
      {
        v114 = *a2;
        sub_1623A60((__int64)&v114, v75, 2);
        v76 = *(_QWORD *)(v27 + 48);
        if ( v76 )
          sub_161E7C0(v27 + 48, v76);
        v77 = (unsigned __int8 *)v114;
        *(_QWORD *)(v27 + 48) = v114;
        if ( v77 )
          sub_1623210((__int64)&v114, v77, v27 + 48);
      }
    }
    else
    {
      v27 = sub_15A2D50((__int64 *)v44, v45, 0, 0, a7, a8, a9);
    }
  }
  a1[3] = (__int64 *)v27;
  v117[0] = (__int64)"ShiftAmt";
  v118 = 259;
  if ( v113 != *(__int64 **)v27 )
  {
    if ( *(_BYTE *)(v27 + 16) > 0x10u )
    {
      v120 = 257;
      v78 = sub_15FDBD0(36, v27, (__int64)v113, (__int64)v119, 0);
      v79 = a2[1];
      v27 = v78;
      if ( v79 )
      {
        v80 = (__int64 *)a2[2];
        sub_157E9D0(v79 + 40, v78);
        v81 = *(_QWORD *)(v27 + 24);
        v82 = *v80;
        *(_QWORD *)(v27 + 32) = v80;
        v82 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v27 + 24) = v82 | v81 & 7;
        *(_QWORD *)(v82 + 8) = v27 + 24;
        *v80 = *v80 & 7 | (v27 + 24);
      }
      sub_164B780(v27, v117);
      v83 = *a2;
      if ( *a2 )
      {
        v115[0] = *a2;
        sub_1623A60((__int64)v115, v83, 2);
        v84 = *(_QWORD *)(v27 + 48);
        if ( v84 )
          sub_161E7C0(v27 + 48, v84);
        v85 = (unsigned __int8 *)v115[0];
        *(_QWORD *)(v27 + 48) = v115[0];
        if ( v85 )
          sub_1623210((__int64)v115, v85, v27 + 48);
      }
    }
    else
    {
      v27 = sub_15A46C0(36, (__int64 ***)v27, (__int64 **)v113, 0);
    }
  }
  a1[3] = (__int64 *)v27;
  v117[0] = (__int64)"Mask";
  v118 = 259;
  v28 = sub_15A0680((__int64)v113, (1 << (8 * v107)) - 1, 0);
  if ( *(_BYTE *)(v28 + 16) > 0x10u || *(_BYTE *)(v27 + 16) > 0x10u )
  {
    v120 = 257;
    v46 = sub_15FB440(23, (__int64 *)v28, v27, (__int64)v119, 0);
    v47 = a2[1];
    v29 = v46;
    if ( v47 )
    {
      v48 = (__int64 *)a2[2];
      sub_157E9D0(v47 + 40, v46);
      v49 = *(_QWORD *)(v29 + 24);
      v50 = *v48;
      *(_QWORD *)(v29 + 32) = v48;
      v50 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v29 + 24) = v50 | v49 & 7;
      *(_QWORD *)(v50 + 8) = v29 + 24;
      *v48 = *v48 & 7 | (v29 + 24);
    }
    sub_164B780(v29, v117);
    v51 = *a2;
    if ( *a2 )
    {
      v115[0] = *a2;
      sub_1623A60((__int64)v115, v51, 2);
      v52 = *(_QWORD *)(v29 + 48);
      if ( v52 )
        sub_161E7C0(v29 + 48, v52);
      v53 = (unsigned __int8 *)v115[0];
      *(_QWORD *)(v29 + 48) = v115[0];
      if ( v53 )
        sub_1623210((__int64)v115, v53, v29 + 48);
    }
  }
  else
  {
    v29 = sub_15A2D50((__int64 *)v28, v27, 0, 0, a7, a8, a9);
  }
  v117[0] = (__int64)"Inv_Mask";
  v118 = 259;
  v30 = *(_BYTE *)(v29 + 16) <= 0x10u;
  a1[4] = (__int64 *)v29;
  if ( v30 )
  {
    v31 = (__int64 *)sub_15A2B00((__int64 *)v29, a7, a8, a9);
  }
  else
  {
    v120 = 257;
    v54 = sub_15FB630((__int64 *)v29, (__int64)v119, 0);
    v55 = a2[1];
    v31 = (__int64 *)v54;
    if ( v55 )
    {
      v56 = (unsigned __int64 *)a2[2];
      sub_157E9D0(v55 + 40, v54);
      v57 = v31[3];
      v58 = *v56;
      v31[4] = (__int64)v56;
      v58 &= 0xFFFFFFFFFFFFFFF8LL;
      v31[3] = v58 | v57 & 7;
      *(_QWORD *)(v58 + 8) = v31 + 3;
      *v56 = *v56 & 7 | (unsigned __int64)(v31 + 3);
    }
    sub_164B780((__int64)v31, v117);
    v59 = *a2;
    if ( *a2 )
    {
      v115[0] = *a2;
      sub_1623A60((__int64)v115, v59, 2);
      v60 = v31[6];
      if ( v60 )
        sub_161E7C0((__int64)(v31 + 6), v60);
      v61 = (unsigned __int8 *)v115[0];
      v31[6] = v115[0];
      if ( v61 )
        sub_1623210((__int64)v115, v61, (__int64)(v31 + 6));
    }
  }
  a1[5] = v31;
  return a1;
}
