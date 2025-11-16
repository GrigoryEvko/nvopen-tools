// Function: sub_172E860
// Address: 0x172e860
//
unsigned __int8 *__fastcall sub_172E860(_QWORD *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v7; // r13
  unsigned int v8; // ebx
  unsigned int v9; // r15d
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r11
  int v14; // ebx
  int v15; // ebx
  bool v16; // al
  __int64 v17; // rdx
  char v18; // di
  __int64 v19; // rcx
  __int64 v20; // rax
  bool v21; // al
  __int64 v22; // r11
  __int64 *v23; // r10
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int8 *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // al
  bool v30; // al
  __int64 v31; // r11
  int v32; // eax
  bool v33; // al
  int v34; // eax
  __int64 v35; // rdx
  char v36; // r8
  bool v37; // al
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  char v41; // dl
  __int64 v42; // rcx
  __int64 v43; // rax
  char v44; // dl
  __int64 v45; // rcx
  __int64 v46; // rax
  bool v47; // al
  bool v48; // al
  int v49; // eax
  char v50; // al
  int v51; // eax
  bool v52; // al
  __int64 v53; // r12
  int v54; // eax
  __int64 v55; // rdx
  unsigned __int8 *v56; // rax
  __int64 v57; // rdi
  int v58; // eax
  __int64 v59; // rdx
  int v60; // eax
  int v61; // eax
  __int64 v62; // r15
  unsigned __int8 *v63; // rax
  __int64 v64; // rdi
  unsigned __int8 *v65; // rax
  __int64 v66; // rdi
  __int16 v67; // si
  unsigned __int8 *v68; // [rsp+28h] [rbp-148h]
  unsigned __int8 *v69; // [rsp+30h] [rbp-140h]
  __int64 v70; // [rsp+38h] [rbp-138h]
  __int64 v71; // [rsp+40h] [rbp-130h]
  __int64 v72; // [rsp+40h] [rbp-130h]
  __int64 *v73; // [rsp+48h] [rbp-128h]
  int v74; // [rsp+50h] [rbp-120h]
  __int64 v75; // [rsp+50h] [rbp-120h]
  int v76; // [rsp+50h] [rbp-120h]
  __int64 v77; // [rsp+50h] [rbp-120h]
  char v78; // [rsp+50h] [rbp-120h]
  __int64 v79; // [rsp+58h] [rbp-118h]
  __int64 *v80; // [rsp+58h] [rbp-118h]
  __int64 v81; // [rsp+60h] [rbp-110h]
  unsigned int v82; // [rsp+68h] [rbp-108h]
  __int64 v83; // [rsp+68h] [rbp-108h]
  __int64 v84; // [rsp+68h] [rbp-108h]
  __int64 v85; // [rsp+68h] [rbp-108h]
  __int64 v86; // [rsp+68h] [rbp-108h]
  __int64 v88; // [rsp+78h] [rbp-F8h]
  __int64 v89; // [rsp+78h] [rbp-F8h]
  __int64 v90; // [rsp+78h] [rbp-F8h]
  __int64 v91; // [rsp+78h] [rbp-F8h]
  unsigned __int8 *v92; // [rsp+78h] [rbp-F8h]
  unsigned __int8 *v93; // [rsp+78h] [rbp-F8h]
  __int64 v94; // [rsp+80h] [rbp-F0h] BYREF
  int v95; // [rsp+88h] [rbp-E8h]
  __int64 v96[2]; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v97; // [rsp+A0h] [rbp-D0h] BYREF
  int v98; // [rsp+A8h] [rbp-C8h]
  __int64 v99[2]; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v100; // [rsp+C0h] [rbp-B0h] BYREF
  int v101; // [rsp+C8h] [rbp-A8h]
  __int64 v102; // [rsp+D0h] [rbp-A0h] BYREF
  int v103; // [rsp+D8h] [rbp-98h]
  __int64 v104; // [rsp+E0h] [rbp-90h] BYREF
  int v105; // [rsp+E8h] [rbp-88h]
  __int64 v106; // [rsp+F0h] [rbp-80h] BYREF
  int v107; // [rsp+F8h] [rbp-78h]
  __int64 v108; // [rsp+100h] [rbp-70h] BYREF
  int v109; // [rsp+108h] [rbp-68h]
  __int16 v110; // [rsp+110h] [rbp-60h]
  __int64 v111; // [rsp+120h] [rbp-50h] BYREF
  int v112; // [rsp+128h] [rbp-48h]
  __int16 v113; // [rsp+130h] [rbp-40h]

  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(unsigned __int16 *)(a3 + 18);
  v82 = v8;
  BYTE1(v8) &= ~0x80u;
  if ( *(_BYTE *)(v7 + 16) != 13 )
    v7 = 0;
  v9 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v88 = *(_QWORD *)(a3 - 24);
  if ( *(_BYTE *)(v88 + 16) == 13 )
  {
    if ( (*(_WORD *)(a2 + 18) & 0x7FFF) != v8 || (*(_WORD *)(a2 + 18) & 0x7FFFu) - 36 > 1 )
      goto LABEL_6;
    if ( !v7 )
      goto LABEL_6;
    v38 = *(_QWORD *)(a2 + 8);
    if ( !v38 )
      goto LABEL_6;
    if ( *(_QWORD *)(v38 + 8) )
      goto LABEL_6;
    v39 = *(_QWORD *)(a3 + 8);
    if ( !v39 || *(_QWORD *)(v39 + 8) || *(_QWORD *)v7 != *(_QWORD *)v88 )
      goto LABEL_6;
    v80 = (__int64 *)(v7 + 24);
    if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v88 + 24) )
        goto LABEL_6;
    }
    else if ( !sub_16A5220((__int64)v80, (const void **)(v88 + 24)) )
    {
      goto LABEL_6;
    }
    v40 = *(_QWORD *)(a2 - 48);
    v41 = *(_BYTE *)(v40 + 16);
    if ( v41 == 35 )
    {
      v69 = *(unsigned __int8 **)(v40 - 48);
      if ( !v69 )
        goto LABEL_6;
      v70 = *(_QWORD *)(v40 - 24);
      if ( *(_BYTE *)(v70 + 16) != 13 )
        goto LABEL_6;
    }
    else
    {
      if ( v41 != 5 )
        goto LABEL_6;
      if ( *(_WORD *)(v40 + 18) != 11 )
        goto LABEL_6;
      v42 = *(_DWORD *)(v40 + 20) & 0xFFFFFFF;
      v69 = *(unsigned __int8 **)(v40 - 24 * v42);
      if ( !v69 )
        goto LABEL_6;
      v70 = *(_QWORD *)(v40 + 24 * (1 - v42));
      if ( *(_BYTE *)(v70 + 16) != 13 )
        goto LABEL_6;
    }
    v43 = *(_QWORD *)(a3 - 48);
    v44 = *(_BYTE *)(v43 + 16);
    if ( v44 == 35 )
    {
      v68 = *(unsigned __int8 **)(v43 - 48);
      if ( !v68 )
        goto LABEL_6;
      v71 = *(_QWORD *)(v43 - 24);
      if ( *(_BYTE *)(v71 + 16) != 13 )
        goto LABEL_6;
    }
    else
    {
      if ( v44 != 5 )
        goto LABEL_6;
      if ( *(_WORD *)(v43 + 18) != 11 )
        goto LABEL_6;
      v45 = *(_DWORD *)(v43 + 20) & 0xFFFFFFF;
      v68 = *(unsigned __int8 **)(v43 - 24 * v45);
      if ( !v68 )
        goto LABEL_6;
      v71 = *(_QWORD *)(v43 + 24 * (1 - v45));
      if ( *(_BYTE *)(v71 + 16) != 13 )
        goto LABEL_6;
    }
    v77 = v70 + 24;
    if ( (int)sub_16A9900(v70 + 24, (unsigned __int64 *)v80) > 0 )
    {
      v73 = (__int64 *)(v71 + 24);
      if ( (int)sub_16A9900(v71 + 24, (unsigned __int64 *)v80) > 0 )
      {
        sub_13A38D0((__int64)&v111, v77);
        sub_1727240(&v111, v73);
        v95 = v112;
        v94 = v111;
        if ( v68 == v69 && sub_14A9C60((__int64)&v94) )
        {
          v58 = sub_16A9900(v77, (unsigned __int64 *)v73);
          v59 = v71;
          if ( v58 >= 0 )
            v59 = v70;
          v72 = v59;
          sub_13A38D0((__int64)&v111, (__int64)v73);
          sub_1455DC0((__int64)v96, (__int64)&v111);
          sub_135E100(&v111);
          sub_13A38D0((__int64)&v111, (__int64)v96);
          sub_16A7200((__int64)&v111, v80);
          v98 = v112;
          v97 = v111;
          sub_13A38D0((__int64)&v111, v77);
          sub_1455DC0((__int64)v99, (__int64)&v111);
          sub_135E100(&v111);
          sub_13A38D0((__int64)&v111, (__int64)v99);
          sub_16A7200((__int64)&v111, v80);
          v101 = v112;
          v100 = v111;
          sub_13A38D0((__int64)&v111, (__int64)v96);
          sub_1727240(&v111, v99);
          v103 = v112;
          v102 = v111;
          sub_13A38D0((__int64)&v111, (__int64)&v97);
          sub_1727240(&v111, &v100);
          v105 = v112;
          v104 = v111;
          if ( (int)sub_16AEA10((__int64)v99, (__int64)v96) <= 0 )
          {
            sub_13A38D0((__int64)&v111, (__int64)v96);
            sub_16A7590((__int64)&v111, v99);
            v107 = v112;
            v106 = v111;
          }
          else
          {
            sub_13A38D0((__int64)&v111, (__int64)v99);
            sub_16A7590((__int64)&v111, v96);
            v60 = v112;
            v112 = 0;
            v107 = v60;
            v106 = v111;
            sub_135E100(&v111);
          }
          if ( sub_14A9C60((__int64)&v102)
            && sub_1455820((__int64)&v102, &v104)
            && (int)sub_16A9900((__int64)&v106, (unsigned __int64 *)v80) > 0 )
          {
            sub_13A38D0((__int64)&v108, (__int64)&v94);
            sub_13D0570((__int64)&v108);
            v61 = v109;
            v109 = 0;
            v112 = v61;
            v111 = v108;
            v62 = sub_15A1070(*(_QWORD *)v70, (__int64)&v111);
            sub_135E100(&v111);
            sub_135E100(&v108);
            v113 = 257;
            v63 = sub_1729500(a1[1], v69, v62, &v111, a4, a5, a6);
            v64 = a1[1];
            v113 = 257;
            v65 = sub_17094A0(v64, (__int64)v63, v72, &v111, 0, 0, a4, a5, a6);
            v66 = a1[1];
            v67 = *(_WORD *)(a2 + 18);
            v113 = 257;
            v93 = sub_17203D0(v66, v67 & 0x7FFF, (__int64)v65, v7, &v111);
            sub_135E100(&v106);
            sub_135E100(&v104);
            sub_135E100(&v102);
            sub_135E100(&v100);
            sub_135E100(v99);
            sub_135E100(&v97);
            sub_135E100(v96);
            sub_135E100(&v94);
            return v93;
          }
          sub_135E100(&v106);
          sub_135E100(&v104);
          sub_135E100(&v102);
          sub_135E100(&v100);
          sub_135E100(v99);
          sub_135E100(&v97);
          sub_135E100(v96);
        }
        sub_135E100(&v94);
      }
    }
  }
  else
  {
    v88 = 0;
  }
LABEL_6:
  if ( !sub_14CF780(v9, v8) )
    goto LABEL_9;
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
LABEL_9:
    v12 = (__int64)sub_172D480(a2, (__int64 *)a3, 0, a1[1], a4, a5, a6);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v81 = *(_QWORD *)(a2 - 48);
    v79 = *(_QWORD *)(a3 - 48);
    v20 = *(_QWORD *)(a2 + 8);
    if ( v20 && !*(_QWORD *)(v20 + 8) || (v28 = *(_QWORD *)(a3 + 8)) != 0 && !*(_QWORD *)(v28 + 8) )
    {
      if ( v9 == 32
        && v7
        && (*(_DWORD *)(v7 + 32) <= 0x40u
          ? (v21 = *(_QWORD *)(v7 + 24) == 0)
          : (v74 = *(_DWORD *)(v7 + 32), v21 = v74 == (unsigned int)sub_16A57B0(v7 + 24)),
            v21) )
      {
        if ( v8 == 36 )
        {
          v23 = *(__int64 **)(a3 - 24);
          if ( (__int64 *)v81 != v23 )
            goto LABEL_39;
          v22 = v79;
        }
        else
        {
          if ( v81 != v79 )
            goto LABEL_39;
          if ( v8 != 34 )
            goto LABEL_39;
          v22 = *(_QWORD *)(a3 - 24);
          if ( !v22 )
            goto LABEL_39;
          v23 = (__int64 *)v81;
        }
      }
      else
      {
        if ( v8 != 32 || !v88 )
          goto LABEL_39;
        if ( *(_DWORD *)(v88 + 32) <= 0x40u )
        {
          v37 = *(_QWORD *)(v88 + 24) == 0;
        }
        else
        {
          v76 = *(_DWORD *)(v88 + 32);
          v37 = v76 == (unsigned int)sub_16A57B0(v88 + 24);
        }
        if ( !v37 )
          goto LABEL_39;
        if ( v9 == 36 )
        {
          v23 = *(__int64 **)(a2 - 24);
          if ( (__int64 *)v79 != v23 )
            goto LABEL_39;
          v22 = v81;
        }
        else
        {
          if ( v81 != v79 )
            goto LABEL_39;
          if ( v9 != 34 )
            goto LABEL_39;
          v22 = *(_QWORD *)(a2 - 24);
          if ( !v22 )
            goto LABEL_39;
          v23 = (__int64 *)v79;
        }
      }
      v75 = v22;
      if ( v22 && v23 )
      {
        v90 = (__int64)v23;
        v110 = 257;
        v24 = a1[1];
        v113 = 257;
        v25 = sub_15A0930(*v23, -1);
        v26 = sub_17094A0(v24, v90, v25, &v108, 0, 0, a4, a5, a6);
        if ( v26[16] > 0x10u || *(_BYTE *)(v75 + 16) > 0x10u )
          return sub_1727440(v24, 35, (__int64)v26, v75, &v111);
        v91 = sub_15A37B0(0x23u, v26, (_QWORD *)v75, 0);
        v27 = sub_14DBA30(v91, *(_QWORD *)(v24 + 96), 0);
        v12 = v91;
        if ( v27 )
          return (unsigned __int8 *)v27;
        return (unsigned __int8 *)v12;
      }
    }
LABEL_39:
    v12 = sub_1728E00(a1, a2, a3, 1);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v12 = sub_1728E00(a1, a3, a2, 1);
    if ( v12 )
      return (unsigned __int8 *)v12;
    v12 = (__int64)sub_172B0D0(a2, a3, 0, a1[1], a4, a5, a6);
    if ( v12 || !v88 || !v7 )
      return (unsigned __int8 *)v12;
    if ( v9 == 33 && v9 == v8 && v7 == v88 )
    {
      v52 = sub_13D01C0(v7 + 24);
      v12 = 0;
      if ( v52 )
      {
        v113 = 257;
        v56 = sub_172AC10(a1[1], v81, v79, &v111, a4, a5, a6);
        v57 = a1[1];
        v113 = 257;
        return sub_17203D0(v57, 33, (__int64)v56, v7, &v111);
      }
      if ( v81 != v79 )
        return (unsigned __int8 *)v12;
      goto LABEL_50;
    }
    if ( v8 != 32 || v9 != 36 )
    {
      if ( v81 != v79 || ((v9 - 35) & 0xFFFFFFFD) == 0 )
        return (unsigned __int8 *)v12;
LABEL_50:
      if ( ((v8 - 37) & 0xFFFFFFFB) == 0 || (v82 & 0xFFFF7FFB) == 0x23 || ((v9 - 39) & 0xFFFFFFFD) == 0 )
        return (unsigned __int8 *)v12;
LABEL_53:
      v84 = v12;
      v29 = sub_14CF780(v9, v8);
      v12 = v84;
      if ( v29 )
      {
        v30 = sub_15FF7F0(v9);
        v31 = v84;
        if ( v30 || v9 - 32 <= 1 && (v48 = sub_15FF7F0(v8), v31 = v84, v48) )
        {
          v86 = v31;
          v49 = sub_16AEA10(v7 + 24, v88 + 24);
          v12 = v86;
          v33 = v49 > 0;
        }
        else
        {
          v85 = v31;
          v32 = sub_16A9900(v7 + 24, (unsigned __int64 *)(v88 + 24));
          v12 = v85;
          v33 = v32 > 0;
        }
        if ( v33 )
        {
          v46 = v7;
          v7 = v88;
          v88 = v46;
          LODWORD(v46) = v9;
          v9 = v8;
          v8 = v46;
        }
        if ( v9 == 36 )
        {
          if ( v8 == 32 )
            return (unsigned __int8 *)v12;
          sub_13A38D0((__int64)&v108, v88 + 24);
          sub_16A7490((__int64)&v108, 1);
          v51 = v109;
          v35 = v7 + 24;
          v109 = 0;
          v36 = 0;
          v112 = v51;
          v111 = v108;
        }
        else
        {
          if ( v9 != 40 || v8 == 32 )
            return (unsigned __int8 *)v12;
          sub_13A38D0((__int64)&v108, v88 + 24);
          sub_16A7490((__int64)&v108, 1);
          v34 = v109;
          v35 = v7 + 24;
          v109 = 0;
          v36 = 1;
          v112 = v34;
          v111 = v108;
        }
        v92 = sub_17288D0((__int64)a1, v81, v35, (__int64)&v111, v36, 0, a4, a5, a6);
        sub_135E100(&v111);
        sub_135E100(&v108);
        return v92;
      }
      return (unsigned __int8 *)v12;
    }
    v50 = *(_BYTE *)(v81 + 16);
    if ( v50 == 35 )
    {
      if ( v79 != *(_QWORD *)(v81 - 48) || (v53 = *(_QWORD *)(v81 - 24), *(_BYTE *)(v53 + 16) != 13) )
      {
LABEL_109:
        if ( v81 != v79 || (v82 & 0xFFFF7FFB) == 0x23 )
          return (unsigned __int8 *)v12;
        goto LABEL_53;
      }
    }
    else
    {
      if ( v50 != 5 )
        goto LABEL_109;
      if ( *(_WORD *)(v81 + 18) != 11 )
        goto LABEL_109;
      v55 = *(_DWORD *)(v81 + 20) & 0xFFFFFFF;
      if ( v79 != *(_QWORD *)(v81 - 24 * v55) )
        goto LABEL_109;
      v53 = *(_QWORD *)(v81 + 24 * (1 - v55));
      if ( *(_BYTE *)(v53 + 16) != 13 )
        goto LABEL_109;
    }
    sub_13A38D0((__int64)&v108, v88 + 24);
    sub_16A7200((__int64)&v108, (__int64 *)(v53 + 24));
    v54 = v109;
    v109 = 0;
    v112 = v54;
    v111 = v108;
    v78 = sub_1455820((__int64)&v111, (_QWORD *)(v7 + 24));
    sub_135E100(&v111);
    sub_135E100(&v108);
    v12 = 0;
    if ( v78 )
    {
      v113 = 257;
      return sub_17203D0(a1[1], 37, v81, v7, &v111);
    }
    goto LABEL_109;
  }
  v83 = *(_QWORD *)(a2 - 24);
  v89 = v10;
  v14 = sub_14CF5F0(a2, 0);
  v15 = sub_14CF5F0(a3, 0) | v14;
  v16 = sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF);
  v17 = v89;
  v18 = 1;
  v19 = v83;
  if ( !v16 )
  {
    v47 = sub_15FF7F0(*(_WORD *)(a3 + 18) & 0x7FFF);
    v19 = v83;
    v17 = v89;
    v18 = v47;
  }
  return sub_1727CB0(v18, v15, v17, v19, a1[1]);
}
