// Function: sub_172F9F0
// Address: 0x172f9f0
//
unsigned __int8 *__fastcall sub_172F9F0(_QWORD *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  unsigned int v8; // r13d
  unsigned int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r10
  int v14; // ebx
  int v15; // ebx
  bool v16; // al
  __int64 v17; // rdx
  char v18; // di
  __int64 v19; // rcx
  __int64 v20; // r9
  _BYTE *v21; // r11
  char v22; // al
  bool v23; // al
  _BYTE *v24; // r11
  __int64 v25; // rsi
  int v26; // eax
  __int64 *v27; // r11
  __int64 *v28; // rax
  unsigned int v29; // et0
  __int64 v30; // rax
  __int64 v31; // rdi
  bool v32; // al
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // eax
  bool v36; // al
  __int64 v37; // rdi
  unsigned __int8 *v38; // rax
  __int64 v39; // rdi
  __int64 *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rdi
  int v45; // eax
  char v46; // r8
  __int64 v47; // rcx
  bool v48; // al
  int v49; // eax
  int v50; // eax
  bool v51; // al
  int v52; // eax
  char v53; // al
  char v54; // al
  __int64 *v55; // r14
  int v56; // eax
  __int64 v57; // r9
  _BYTE *v58; // r11
  __int64 v59; // r10
  int v60; // eax
  char v61; // al
  char v62; // al
  __int64 v63; // rdi
  int v64; // eax
  __int64 v65; // r14
  __int64 *v66; // r13
  int v67; // eax
  unsigned __int8 *v68; // r13
  __int64 v69; // rax
  unsigned __int8 *v70; // r13
  __int64 v71; // rax
  __int64 v72; // rdi
  _BYTE *v73; // [rsp+8h] [rbp-108h]
  __int64 v74; // [rsp+10h] [rbp-100h]
  __int64 v75; // [rsp+18h] [rbp-F8h]
  _BYTE *v76; // [rsp+20h] [rbp-F0h]
  bool v77; // [rsp+20h] [rbp-F0h]
  __int64 v78; // [rsp+28h] [rbp-E8h]
  __int64 *v79; // [rsp+28h] [rbp-E8h]
  unsigned int v80; // [rsp+30h] [rbp-E0h]
  __int64 v81; // [rsp+38h] [rbp-D8h]
  __int64 *v82; // [rsp+38h] [rbp-D8h]
  __int64 v83; // [rsp+38h] [rbp-D8h]
  __int64 v84; // [rsp+40h] [rbp-D0h]
  const void **v85; // [rsp+40h] [rbp-D0h]
  _BYTE *v86; // [rsp+40h] [rbp-D0h]
  __int64 v87; // [rsp+40h] [rbp-D0h]
  _BYTE *v88; // [rsp+48h] [rbp-C8h]
  _BYTE *v89; // [rsp+48h] [rbp-C8h]
  const void **v90; // [rsp+48h] [rbp-C8h]
  __int64 v91; // [rsp+48h] [rbp-C8h]
  __int64 *v92; // [rsp+48h] [rbp-C8h]
  _BYTE *v93; // [rsp+50h] [rbp-C0h]
  __int64 v94; // [rsp+50h] [rbp-C0h]
  __int64 v95; // [rsp+50h] [rbp-C0h]
  __int64 v96; // [rsp+50h] [rbp-C0h]
  __int64 v97; // [rsp+50h] [rbp-C0h]
  __int64 v98; // [rsp+50h] [rbp-C0h]
  __int64 v99; // [rsp+50h] [rbp-C0h]
  __int64 v100; // [rsp+50h] [rbp-C0h]
  __int64 v101; // [rsp+58h] [rbp-B8h]
  __int64 v102; // [rsp+60h] [rbp-B0h]
  __int64 v103; // [rsp+60h] [rbp-B0h]
  int v104; // [rsp+68h] [rbp-A8h]
  __int64 v105; // [rsp+68h] [rbp-A8h]
  __int64 v106; // [rsp+68h] [rbp-A8h]
  __int64 *v107; // [rsp+68h] [rbp-A8h]
  __int64 v108; // [rsp+68h] [rbp-A8h]
  __int64 *v109; // [rsp+68h] [rbp-A8h]
  __int64 v110; // [rsp+68h] [rbp-A8h]
  __int64 v111; // [rsp+68h] [rbp-A8h]
  unsigned __int8 *v112; // [rsp+68h] [rbp-A8h]
  _BYTE *v113; // [rsp+68h] [rbp-A8h]
  __int64 *v114; // [rsp+68h] [rbp-A8h]
  unsigned __int8 *v115; // [rsp+68h] [rbp-A8h]
  __int64 *v116; // [rsp+70h] [rbp-A0h] BYREF
  __int64 ***v117; // [rsp+78h] [rbp-98h] BYREF
  __int64 v118[2]; // [rsp+80h] [rbp-90h] BYREF
  __int64 **v119; // [rsp+90h] [rbp-80h] BYREF
  int v120; // [rsp+98h] [rbp-78h]
  __int64 **v121; // [rsp+A0h] [rbp-70h] BYREF
  int v122; // [rsp+A8h] [rbp-68h]
  __int64 **v123; // [rsp+B0h] [rbp-60h] BYREF
  int v124; // [rsp+B8h] [rbp-58h]
  __int64 **v125; // [rsp+C0h] [rbp-50h] BYREF
  __int64 ****v126; // [rsp+C8h] [rbp-48h]
  __int16 v127; // [rsp+D0h] [rbp-40h]

  v9 = *(unsigned __int16 *)(a3 + 18);
  v8 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v104 = (unsigned __int16)v9;
  BYTE1(v9) &= ~0x80u;
  if ( !sub_14CF780(*(_WORD *)(a2 + 18) & 0x7FFF, v9) )
    goto LABEL_4;
  v10 = *(_QWORD *)(a2 - 48);
  v11 = *(_QWORD *)(a3 - 48);
  if ( v10 == *(_QWORD *)(a3 - 24) && *(_QWORD *)(a2 - 24) == v11 )
  {
    *(_WORD *)(a2 + 18) = sub_15FF5D0(*(_WORD *)(a2 + 18) & 0x7FFF) | *(_WORD *)(a2 + 18) & 0x8000;
    sub_16484A0((__int64 *)(a2 - 48), (__int64 *)(a2 - 24));
    v10 = *(_QWORD *)(a2 - 48);
    v11 = *(_QWORD *)(a3 - 48);
  }
  if ( v11 != v10 || *(_QWORD *)(a2 - 24) != *(_QWORD *)(a3 - 24) )
  {
LABEL_4:
    v12 = (__int64)sub_172D480(a2, (__int64 *)a3, 1u, a1[1], a4, a5, a6);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v12 = sub_1728E00(a1, a2, a3, 0);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v12 = sub_1728E00(a1, a3, a2, 0);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v12 = (__int64)sub_172B0D0(a2, a3, 1, a1[1], a4, a5, a6);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v20 = *(_QWORD *)(a3 - 48);
    v21 = *(_BYTE **)(a3 - 24);
    v101 = *(_QWORD *)(a2 - 48);
    v103 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v103 + 16) == 13 )
    {
      if ( v21 )
      {
        if ( v21[16] != 13 )
          return (unsigned __int8 *)v12;
        if ( v8 != v9 || (_BYTE *)v103 != v21 )
        {
LABEL_20:
          if ( v9 != 32 || v8 != 32 )
          {
LABEL_22:
            if ( v101 != v20 || ((v8 - 35) & 0xFFFFFFFD) == 0 )
              return (unsigned __int8 *)v12;
LABEL_24:
            if ( ((v9 - 37) & 0xFFFFFFFB) == 0 || (v104 & 0xFFFF7FFB) == 0x23 || ((v8 - 39) & 0xFFFFFFFD) == 0 )
              return (unsigned __int8 *)v12;
LABEL_27:
            v106 = v12;
            v93 = v21;
            v22 = sub_14CF780(v8, v9);
            v12 = v106;
            if ( !v22 )
              return (unsigned __int8 *)v12;
            v23 = sub_15FF7F0(v8);
            v24 = v93;
            if ( v23 )
            {
              v25 = (__int64)(v93 + 24);
              v94 = v106;
              v107 = (__int64 *)v24;
              v26 = sub_16AEA10(v103 + 24, v25);
              v27 = v107;
              v12 = v94;
              if ( v26 <= 0 )
              {
LABEL_30:
                v28 = v27;
                v27 = (__int64 *)v103;
                v29 = v8;
                v8 = v9;
                v9 = v29;
                v103 = (__int64)v28;
              }
            }
            else
            {
              v95 = v106;
              if ( v8 - 32 <= 1 )
              {
                v113 = v24;
                if ( sub_15FF7F0(v9) )
                {
                  v50 = sub_16AEA10(v103 + 24, (__int64)(v113 + 24));
                  v27 = (__int64 *)v113;
                  v12 = v95;
                  if ( v50 > 0 )
                    goto LABEL_31;
                }
                else
                {
                  v60 = sub_16A9900(v103 + 24, (unsigned __int64 *)v113 + 3);
                  v27 = (__int64 *)v113;
                  v12 = v95;
                  if ( v60 > 0 )
                    goto LABEL_31;
                }
LABEL_53:
                if ( v9 != 36 )
                {
                  if ( v9 == 40 )
                  {
                    v110 = v12;
                    v41 = sub_1727110(v27, a4, a5, a6);
                    v12 = v110;
                    if ( v103 == v41 )
                    {
                      v42 = a1[1];
                      v127 = 257;
                      return sub_17203D0(v42, 40, v101, v103, (__int64 *)&v125);
                    }
                  }
                  return (unsigned __int8 *)v12;
                }
                v98 = v12;
                v114 = v27;
                if ( v103 == sub_1727110(v27, a4, a5, a6) )
                {
                  v63 = a1[1];
                  v127 = 257;
                  return sub_17203D0(v63, 36, v101, v103, (__int64 *)&v125);
                }
                v51 = sub_13D01C0(v103 + 24);
                v12 = v98;
                if ( v51 )
                {
                  sub_13A38D0((__int64)&v123, v103 + 24);
                  sub_16A7490((__int64)&v123, 1);
                  v52 = v124;
                  v124 = 0;
                  LODWORD(v126) = v52;
                  v47 = (__int64)(v114 + 3);
                  v125 = v123;
LABEL_66:
                  v46 = 0;
LABEL_61:
                  v112 = sub_17288D0((__int64)a1, v101, (__int64)&v125, v47, v46, 1, a4, a5, a6);
                  sub_135E100((__int64 *)&v125);
                  sub_135E100((__int64 *)&v123);
                  return v112;
                }
                return (unsigned __int8 *)v12;
              }
              v109 = (__int64 *)v24;
              v35 = sub_16A9900(v103 + 24, (unsigned __int64 *)v24 + 3);
              v27 = v109;
              v12 = v95;
              if ( v35 <= 0 )
                goto LABEL_30;
            }
LABEL_31:
            if ( v9 == 34 )
            {
              v111 = v12;
              if ( v8 == 33 )
              {
                v43 = sub_1727140(v27, a4, a5, a6);
                v12 = v111;
                if ( v103 == v43 )
                {
                  v44 = a1[1];
                  v127 = 257;
                  return sub_17203D0(v44, 34, v101, v103, (__int64 *)&v125);
                }
                return (unsigned __int8 *)v12;
              }
              sub_13A38D0((__int64)&v123, (__int64)(v27 + 3));
              sub_16A7490((__int64)&v123, 1);
              v49 = v124;
              v124 = 0;
              LODWORD(v126) = v49;
              v47 = v103 + 24;
              v125 = v123;
              goto LABEL_66;
            }
            if ( v9 == 38 )
            {
              v108 = v12;
              if ( v8 == 33 )
              {
                v30 = sub_1727140(v27, a4, a5, a6);
                v12 = v108;
                if ( v103 == v30 )
                {
                  v31 = a1[1];
                  v127 = 257;
                  return sub_17203D0(v31, 38, v101, v103, (__int64 *)&v125);
                }
                return (unsigned __int8 *)v12;
              }
              sub_13A38D0((__int64)&v123, (__int64)(v27 + 3));
              sub_16A7490((__int64)&v123, 1);
              v45 = v124;
              v124 = 0;
              v46 = 1;
              LODWORD(v126) = v45;
              v47 = v103 + 24;
              v125 = v123;
              goto LABEL_61;
            }
            v40 = v27;
            v27 = (__int64 *)v103;
            v9 = v8;
            v103 = (__int64)v40;
            goto LABEL_53;
          }
          v33 = *(_QWORD *)(a2 + 8);
          if ( !v33 || *(_QWORD *)(v33 + 8) || (v34 = *(_QWORD *)(a3 + 8)) == 0 || *(_QWORD *)(v34 + 8) )
          {
LABEL_42:
            if ( v101 != v20 || (v104 & 0xFFFF7FFB) == 0x23 )
              return (unsigned __int8 *)v12;
            goto LABEL_27;
          }
          v84 = v12;
          v90 = (const void **)v21;
          v99 = v20;
          v123 = &v116;
          v53 = sub_172F8F0(&v123, v20);
          v20 = v99;
          v21 = v90;
          v12 = v84;
          if ( !v53 )
            goto LABEL_81;
          v81 = v84;
          v85 = v90;
          v125 = (__int64 **)v116;
          v126 = &v117;
          v54 = sub_172F970((__int64)&v125, v101);
          v20 = v99;
          v21 = v90;
          v12 = v81;
          if ( v54 )
          {
            v55 = (__int64 *)v103;
          }
          else
          {
LABEL_81:
            v87 = v12;
            v92 = (__int64 *)v21;
            v100 = v20;
            v123 = &v116;
            v61 = sub_172F8F0(&v123, v101);
            v20 = v100;
            v21 = v92;
            v12 = v87;
            if ( !v61 )
              goto LABEL_42;
            v125 = (__int64 **)v116;
            v126 = &v117;
            v62 = sub_172F970((__int64)&v125, v100);
            v20 = v100;
            v21 = v92;
            v12 = v87;
            if ( !v62 )
              goto LABEL_42;
            v55 = v92;
            v85 = (const void **)v103;
          }
          v75 = v12;
          v76 = v21;
          v78 = v20;
          v80 = *(_DWORD *)(*v55 + 8) >> 8;
          sub_13D0120((__int64)v118, v80, *((_DWORD *)*v85 + 2) >> 8);
          v82 = (__int64 *)(v117 + 3);
          sub_13A38D0((__int64)&v123, (__int64)v118);
          sub_1727280((__int64 *)&v123, v82);
          v56 = v124;
          v124 = 0;
          LODWORD(v126) = v56;
          v125 = v123;
          if ( sub_13D01C0((__int64)&v125) )
          {
            v73 = v76;
            v74 = v78;
            v79 = v55 + 3;
            sub_13A38D0((__int64)&v119, (__int64)v118);
            sub_1727280((__int64 *)&v119, v55 + 3);
            v64 = v120;
            v120 = 0;
            v122 = v64;
            v121 = v119;
            v77 = sub_13D01C0((__int64)&v121);
            sub_135E100((__int64 *)&v121);
            sub_135E100((__int64 *)&v119);
            sub_135E100((__int64 *)&v125);
            sub_135E100((__int64 *)&v123);
            v57 = v74;
            v58 = v73;
            v59 = v75;
            if ( v77 )
            {
              v65 = a1[1];
              v127 = 257;
              v66 = (__int64 *)(v117 + 3);
              sub_13A38D0((__int64)&v121, (__int64)v118);
              sub_1727260((__int64 *)&v121, v66);
              v67 = v122;
              v68 = (unsigned __int8 *)v116;
              v122 = 0;
              v124 = v67;
              v123 = v121;
              v69 = sub_15A1070(*v116, (__int64)&v123);
              v70 = sub_1729500(v65, v68, v69, (__int64 *)&v125, a4, a5, a6);
              sub_135E100((__int64 *)&v123);
              sub_135E100((__int64 *)&v121);
              sub_16A5C50((__int64)&v125, v85 + 3, v80);
              sub_1727260((__int64 *)&v125, v79);
              v124 = (int)v126;
              v123 = v125;
              v71 = sub_159C0E0(**v117, (__int64)&v123);
              v72 = a1[1];
              v127 = 257;
              v115 = sub_17203D0(v72, 32, (__int64)v70, v71, (__int64 *)&v125);
              sub_135E100((__int64 *)&v123);
              sub_135E100(v118);
              return v115;
            }
          }
          else
          {
            sub_135E100((__int64 *)&v125);
            sub_135E100((__int64 *)&v123);
            v57 = v78;
            v58 = v76;
            v59 = v75;
          }
          v83 = v59;
          v86 = v58;
          v91 = v57;
          sub_135E100(v118);
          v20 = v91;
          v21 = v86;
          v12 = v83;
          goto LABEL_42;
        }
        if ( v8 == 36 )
        {
          v97 = *(_QWORD *)(a3 - 48);
          v89 = *(_BYTE **)(a3 - 24);
          v48 = sub_14A9C60(v103 + 24);
          v20 = v97;
          if ( !v48 )
          {
            v21 = v89;
            v12 = 0;
            if ( v101 != v97 )
              return (unsigned __int8 *)v12;
            goto LABEL_24;
          }
        }
        else
        {
          if ( v8 != 32 )
            goto LABEL_22;
          v88 = *(_BYTE **)(a3 - 24);
          v96 = *(_QWORD *)(a3 - 48);
          v36 = sub_13D01C0(v103 + 24);
          v20 = v96;
          v21 = v88;
          v12 = 0;
          if ( !v36 )
            goto LABEL_20;
        }
        v37 = a1[1];
        v127 = 257;
        v38 = sub_172AC10(v37, v101, v20, (__int64 *)&v125, a4, a5, a6);
        v39 = a1[1];
        v127 = 257;
        return sub_17203D0(v39, v8, (__int64)v38, v103, (__int64 *)&v125);
      }
    }
    else if ( v21 )
    {
      return (unsigned __int8 *)v12;
    }
    BUG();
  }
  v102 = *(_QWORD *)(a2 - 24);
  v105 = v10;
  v14 = sub_14CF5F0(a2, 0);
  v15 = sub_14CF5F0(a3, 0) & v14;
  v16 = sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF);
  v17 = v105;
  v18 = 1;
  v19 = v102;
  if ( !v16 )
  {
    v32 = sub_15FF7F0(*(_WORD *)(a3 + 18) & 0x7FFF);
    v19 = v102;
    v17 = v105;
    v18 = v32;
  }
  return sub_1727CB0(v18, v15, v17, v19, a1[1]);
}
