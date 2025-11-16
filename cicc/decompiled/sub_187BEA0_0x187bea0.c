// Function: sub_187BEA0
// Address: 0x187bea0
//
__int64 __fastcall sub_187BEA0(__int64 *a1, __int64 *a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  __int64 v8; // r13
  _BYTE *v10; // r15
  __int64 v12; // r11
  _QWORD *v13; // r13
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int8 *v22; // rsi
  __int64 **v23; // rsi
  unsigned __int64 v24; // rdi
  __int64 v25; // r13
  unsigned __int8 v26; // al
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // r13
  __int64 v32; // rdi
  __int64 v33; // r9
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // r11
  __int64 *v39; // r10
  __int64 v40; // rax
  __int64 v41; // rdi
  unsigned __int64 v42; // rsi
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // r14
  _QWORD **v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdi
  __int64 *v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 *v55; // r13
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 *v58; // rax
  __int64 v59; // r14
  __int64 **v60; // r15
  unsigned int v61; // ebx
  unsigned int v62; // eax
  __int64 v63; // rdx
  unsigned __int8 v64; // al
  __int64 v65; // rax
  __int64 v66; // r13
  unsigned __int8 v67; // al
  __int64 v68; // rax
  __int64 v69; // rbx
  unsigned int v70; // r14d
  __int64 v72; // rax
  _QWORD *v73; // rax
  __int64 v74; // r15
  __int64 **v75; // rax
  __int64 *v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rdi
  __int64 *v79; // rbx
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 *v84; // rbx
  __int64 v85; // rax
  __int64 v86; // rcx
  __int64 v87; // rax
  __int64 v88; // rdi
  __int64 *v89; // rbx
  __int64 v90; // rax
  __int64 v91; // rcx
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 *v94; // rbx
  __int64 v95; // rax
  __int64 v96; // rcx
  __int64 v97; // rax
  __int64 *v98; // rbx
  __int64 v99; // rax
  __int64 v100; // rsi
  unsigned int v101; // ebx
  bool v102; // al
  unsigned int v103; // ebx
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rsi
  __int64 v107; // rax
  int v108; // [rsp+4h] [rbp-ACh]
  __int64 v109; // [rsp+8h] [rbp-A8h]
  __int64 v110; // [rsp+10h] [rbp-A0h]
  __int64 v111; // [rsp+18h] [rbp-98h]
  __int64 v112; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v113; // [rsp+20h] [rbp-90h]
  _QWORD *v114; // [rsp+20h] [rbp-90h]
  __int64 v115; // [rsp+20h] [rbp-90h]
  __int64 *v116; // [rsp+20h] [rbp-90h]
  int v117; // [rsp+28h] [rbp-88h]
  __int64 *v118; // [rsp+28h] [rbp-88h]
  __int64 v119; // [rsp+28h] [rbp-88h]
  __int64 *v120; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v121; // [rsp+38h] [rbp-78h] BYREF
  __int64 v122[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v123; // [rsp+50h] [rbp-60h]
  __int64 v124[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v125; // [rsp+70h] [rbp-40h]

  v8 = (__int64)a4;
  if ( *(_DWORD *)a3 != 2 )
  {
    v10 = *(_BYTE **)(a3 + 32);
    if ( byte_4FAC5A0 && !a1[2] )
    {
      v32 = a1[6];
      v33 = *a1;
      v124[0] = (__int64)"bits_use";
      v125 = 259;
      v10 = (_BYTE *)sub_15E57E0(v32, 0, 8, (__int64)v124, (__int64)v10, v33);
    }
    v120 = (__int64 *)v8;
    v12 = a1[6];
    v123 = 257;
    if ( v10[16] > 0x10u || *(_BYTE *)(v8 + 16) > 0x10u )
    {
      v125 = 257;
      if ( !v12 )
      {
        v72 = *(_QWORD *)v10;
        if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
          v72 = **(_QWORD **)(v72 + 16);
        v12 = *(_QWORD *)(v72 + 24);
      }
      v112 = v12;
      v34 = sub_1648A60(72, 2u);
      v13 = v34;
      if ( v34 )
      {
        v111 = (__int64)v34;
        v110 = (__int64)(v34 - 6);
        v35 = *(_QWORD *)v10;
        if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
          v35 = **(_QWORD **)(v35 + 16);
        v108 = *(_DWORD *)(v35 + 8) >> 8;
        v36 = (__int64 *)sub_15F9F50(v112, (__int64)&v120, 1);
        v37 = sub_1646BA0(v36, v108);
        v38 = v112;
        v39 = (__int64 *)v37;
        v40 = *(_QWORD *)v10;
        if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 || (v40 = *v120, *(_BYTE *)(*v120 + 8) == 16) )
        {
          v58 = sub_16463B0(v39, *(_QWORD *)(v40 + 32));
          v38 = v112;
          v39 = v58;
        }
        v109 = v38;
        sub_15F1EA0((__int64)v13, (__int64)v39, 32, v110, 2, 0);
        v13[7] = v109;
        v13[8] = sub_15F9F50(v109, (__int64)&v120, 1);
        sub_15F9CE0((__int64)v13, (__int64)v10, (__int64 *)&v120, 1, (__int64)v124);
      }
      else
      {
        v111 = 0;
      }
      v41 = a2[1];
      if ( v41 )
      {
        v113 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v41 + 40, (__int64)v13);
        v42 = *v113;
        v43 = v13[3] & 7LL;
        v13[4] = v113;
        v42 &= 0xFFFFFFFFFFFFFFF8LL;
        v13[3] = v42 | v43;
        *(_QWORD *)(v42 + 8) = v13 + 3;
        *v113 = *v113 & 7 | (unsigned __int64)(v13 + 3);
      }
      sub_164B780(v111, v122);
      sub_12A86E0(a2, (__int64)v13);
    }
    else
    {
      v121 = (__int64 *)v8;
      BYTE4(v124[0]) = 0;
      v13 = (_QWORD *)sub_15A2E80(v12, (__int64)v10, &v121, 1u, 0, (__int64)v124, 0);
    }
    v125 = 257;
    v14 = (__int64)sub_1648A60(64, 1u);
    if ( v14 )
      sub_15F9210(v14, *(_QWORD *)(*v13 + 24LL), (__int64)v13, 0, 0, 0);
    v15 = a2[1];
    if ( v15 )
    {
      v16 = (__int64 *)a2[2];
      sub_157E9D0(v15 + 40, v14);
      v17 = *(_QWORD *)(v14 + 24);
      v18 = *v16;
      *(_QWORD *)(v14 + 32) = v16;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v14 + 24) = v18 | v17 & 7;
      *(_QWORD *)(v18 + 8) = v14 + 24;
      *v16 = *v16 & 7 | (v14 + 24);
    }
    sub_164B780(v14, v124);
    v19 = *a2;
    if ( *a2 )
    {
      v122[0] = *a2;
      sub_1623A60((__int64)v122, v19, 2);
      v20 = *(_QWORD *)(v14 + 48);
      v21 = v14 + 48;
      if ( v20 )
      {
        sub_161E7C0(v14 + 48, v20);
        v21 = v14 + 48;
      }
      v22 = (unsigned __int8 *)v122[0];
      *(_QWORD *)(v14 + 48) = v122[0];
      if ( v22 )
        sub_1623210((__int64)v122, v22, v21);
    }
    v23 = (__int64 **)a1[6];
    v24 = *(_QWORD *)(a3 + 40);
    v123 = 257;
    v25 = sub_15A4180(v24, v23, 0);
    v26 = *(_BYTE *)(v25 + 16);
    if ( v26 <= 0x10u )
    {
      if ( v26 == 13 )
      {
        v70 = *(_DWORD *)(v25 + 32);
        if ( v70 <= 0x40
           ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v70) == *(_QWORD *)(v25 + 24)
           : v70 == (unsigned int)sub_16A58F0(v25 + 24) )
        {
LABEL_19:
          v27 = a1[6];
          v123 = 257;
          v28 = sub_159C470(v27, 0, 0);
          v29 = v28;
          if ( *(_BYTE *)(v14 + 16) <= 0x10u && *(_BYTE *)(v28 + 16) <= 0x10u )
            return sub_15A37B0(0x21u, (_QWORD *)v14, (_QWORD *)v28, 0);
          v125 = 257;
          v44 = sub_1648A60(56, 2u);
          v30 = (__int64)v44;
          if ( v44 )
          {
            v45 = (__int64)v44;
            v46 = *(_QWORD ***)v14;
            if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
            {
              v114 = v46[4];
              v47 = (__int64 *)sub_1643320(*v46);
              v48 = (__int64)sub_16463B0(v47, (unsigned int)v114);
            }
            else
            {
              v48 = sub_1643320(*v46);
            }
            sub_15FEC10(v30, v48, 51, 33, v14, v29, (__int64)v124, 0);
          }
          else
          {
            v45 = 0;
          }
          v49 = a2[1];
          if ( v49 )
          {
            v50 = (__int64 *)a2[2];
            sub_157E9D0(v49 + 40, v30);
            v51 = *(_QWORD *)(v30 + 24);
            v52 = *v50;
            *(_QWORD *)(v30 + 32) = v50;
            v52 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v30 + 24) = v52 | v51 & 7;
            *(_QWORD *)(v52 + 8) = v30 + 24;
            *v50 = *v50 & 7 | (v30 + 24);
          }
          sub_164B780(v45, v122);
          sub_12A86E0(a2, v30);
          return v30;
        }
      }
      if ( *(_BYTE *)(v14 + 16) <= 0x10u )
      {
        v14 = sub_15A2CF0((__int64 *)v14, v25, a5, a6, a7);
        goto LABEL_19;
      }
    }
    v125 = 257;
    v53 = sub_15FB440(26, (__int64 *)v14, v25, (__int64)v124, 0);
    v54 = a2[1];
    v14 = v53;
    if ( v54 )
    {
      v55 = (__int64 *)a2[2];
      sub_157E9D0(v54 + 40, v53);
      v56 = *(_QWORD *)(v14 + 24);
      v57 = *v55;
      *(_QWORD *)(v14 + 32) = v55;
      v57 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v14 + 24) = v57 | v56 & 7;
      *(_QWORD *)(v57 + 8) = v14 + 24;
      *v55 = *v55 & 7 | (v14 + 24);
    }
    sub_164B780(v14, v122);
    sub_12A86E0(a2, v14);
    goto LABEL_19;
  }
  v59 = *(_QWORD *)(a3 + 48);
  v60 = *(__int64 ***)v59;
  v115 = *a4;
  v117 = *(_DWORD *)(*(_QWORD *)v59 + 8LL) >> 8;
  v123 = 257;
  v61 = sub_16431D0(v115);
  v62 = sub_16431D0((__int64)v60);
  if ( v61 >= v62 )
  {
    if ( v60 == (__int64 **)v115 || v61 == v62 )
      goto LABEL_49;
    if ( *(_BYTE *)(v8 + 16) <= 0x10u )
    {
      v8 = sub_15A46C0(36, (__int64 ***)v8, v60, 0);
      goto LABEL_49;
    }
    v125 = 257;
    v8 = sub_15FDBD0(36, v8, (__int64)v60, (__int64)v124, 0);
    v105 = a2[1];
    if ( v105 )
    {
      v116 = (__int64 *)a2[2];
      sub_157E9D0(v105 + 40, v8);
      v106 = *v116;
      v107 = *(_QWORD *)(v8 + 24) & 7LL;
      *(_QWORD *)(v8 + 32) = v116;
      v106 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v8 + 24) = v106 | v107;
      *(_QWORD *)(v106 + 8) = v8 + 24;
      *v116 = *v116 & 7 | (v8 + 24);
    }
LABEL_95:
    sub_164B780(v8, v122);
    sub_12A86E0(a2, v8);
    goto LABEL_49;
  }
  if ( v60 != (__int64 **)v115 )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
    {
      v125 = 257;
      v8 = sub_15FDBD0(37, v8, (__int64)v60, (__int64)v124, 0);
      v97 = a2[1];
      if ( v97 )
      {
        v98 = (__int64 *)a2[2];
        sub_157E9D0(v97 + 40, v8);
        v99 = *(_QWORD *)(v8 + 24);
        v100 = *v98;
        *(_QWORD *)(v8 + 32) = v98;
        v100 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v100 | v99 & 7;
        *(_QWORD *)(v100 + 8) = v8 + 24;
        *v98 = *v98 & 7 | (v8 + 24);
      }
      goto LABEL_95;
    }
    v8 = sub_15A46C0(37, (__int64 ***)v8, v60, 0);
  }
LABEL_49:
  v123 = 257;
  v63 = sub_159C470((__int64)v60, (unsigned int)(v117 - 1), 0);
  v64 = *(_BYTE *)(v63 + 16);
  if ( v64 > 0x10u )
    goto LABEL_84;
  if ( v64 == 13 )
  {
    v103 = *(_DWORD *)(v63 + 32);
    if ( v103 <= 0x40 )
    {
      if ( *(_QWORD *)(v63 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v103) )
        goto LABEL_53;
    }
    else
    {
      v119 = v63;
      v104 = sub_16A58F0(v63 + 24);
      v63 = v119;
      if ( v103 == v104 )
        goto LABEL_53;
    }
  }
  if ( *(_BYTE *)(v8 + 16) > 0x10u )
  {
LABEL_84:
    v125 = 257;
    v87 = sub_15FB440(26, (__int64 *)v8, v63, (__int64)v124, 0);
    v88 = a2[1];
    v8 = v87;
    if ( v88 )
    {
      v89 = (__int64 *)a2[2];
      sub_157E9D0(v88 + 40, v87);
      v90 = *(_QWORD *)(v8 + 24);
      v91 = *v89;
      *(_QWORD *)(v8 + 32) = v89;
      v91 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v8 + 24) = v91 | v90 & 7;
      *(_QWORD *)(v91 + 8) = v8 + 24;
      *v89 = *v89 & 7 | (v8 + 24);
    }
    sub_164B780(v8, v122);
    sub_12A86E0(a2, v8);
  }
  else
  {
    v8 = sub_15A2CF0((__int64 *)v8, v63, a5, a6, a7);
  }
LABEL_53:
  v123 = 257;
  v65 = sub_159C470((__int64)v60, 1, 0);
  if ( *(_BYTE *)(v65 + 16) > 0x10u || *(_BYTE *)(v8 + 16) > 0x10u )
  {
    v125 = 257;
    v82 = sub_15FB440(23, (__int64 *)v65, v8, (__int64)v124, 0);
    v83 = a2[1];
    v66 = v82;
    if ( v83 )
    {
      v84 = (__int64 *)a2[2];
      sub_157E9D0(v83 + 40, v82);
      v85 = *(_QWORD *)(v66 + 24);
      v86 = *v84;
      *(_QWORD *)(v66 + 32) = v84;
      v86 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v66 + 24) = v86 | v85 & 7;
      *(_QWORD *)(v86 + 8) = v66 + 24;
      *v84 = *v84 & 7 | (v66 + 24);
    }
    sub_164B780(v66, v122);
    sub_12A86E0(a2, v66);
  }
  else
  {
    v66 = sub_15A2D50((__int64 *)v65, v8, 0, 0, a5, a6, a7);
  }
  v123 = 257;
  v67 = *(_BYTE *)(v66 + 16);
  if ( v67 > 0x10u )
    goto LABEL_87;
  if ( v67 != 13
    || ((v101 = *(_DWORD *)(v66 + 32), v101 <= 0x40)
      ? (v102 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v101) == *(_QWORD *)(v66 + 24))
      : (v102 = v101 == (unsigned int)sub_16A58F0(v66 + 24)),
        !v102) )
  {
    if ( *(_BYTE *)(v59 + 16) <= 0x10u )
    {
      v59 = sub_15A2CF0((__int64 *)v59, v66, a5, a6, a7);
      goto LABEL_60;
    }
LABEL_87:
    v125 = 257;
    v92 = sub_15FB440(26, (__int64 *)v59, v66, (__int64)v124, 0);
    v93 = a2[1];
    v59 = v92;
    if ( v93 )
    {
      v94 = (__int64 *)a2[2];
      sub_157E9D0(v93 + 40, v92);
      v95 = *(_QWORD *)(v59 + 24);
      v96 = *v94;
      *(_QWORD *)(v59 + 32) = v94;
      v96 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v59 + 24) = v96 | v95 & 7;
      *(_QWORD *)(v96 + 8) = v59 + 24;
      *v94 = *v94 & 7 | (v59 + 24);
    }
    sub_164B780(v59, v122);
    sub_12A86E0(a2, v59);
  }
LABEL_60:
  v123 = 257;
  v68 = sub_159C470((__int64)v60, 0, 0);
  v69 = v68;
  if ( *(_BYTE *)(v59 + 16) <= 0x10u && *(_BYTE *)(v68 + 16) <= 0x10u )
    return sub_15A37B0(0x21u, (_QWORD *)v59, (_QWORD *)v68, 0);
  v125 = 257;
  v73 = sub_1648A60(56, 2u);
  v30 = (__int64)v73;
  if ( v73 )
  {
    v74 = (__int64)v73;
    v75 = *(__int64 ***)v59;
    if ( *(_BYTE *)(*(_QWORD *)v59 + 8LL) == 16 )
    {
      v118 = v75[4];
      v76 = (__int64 *)sub_1643320(*v75);
      v77 = (__int64)sub_16463B0(v76, (unsigned int)v118);
    }
    else
    {
      v77 = sub_1643320(*v75);
    }
    sub_15FEC10(v30, v77, 51, 33, v59, v69, (__int64)v124, 0);
  }
  else
  {
    v74 = 0;
  }
  v78 = a2[1];
  if ( v78 )
  {
    v79 = (__int64 *)a2[2];
    sub_157E9D0(v78 + 40, v30);
    v80 = *(_QWORD *)(v30 + 24);
    v81 = *v79;
    *(_QWORD *)(v30 + 32) = v79;
    v81 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v30 + 24) = v81 | v80 & 7;
    *(_QWORD *)(v81 + 8) = v30 + 24;
    *v79 = *v79 & 7 | (v30 + 24);
  }
  sub_164B780(v74, v122);
  sub_12A86E0(a2, v30);
  return v30;
}
