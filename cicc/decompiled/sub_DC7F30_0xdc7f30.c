// Function: sub_DC7F30
// Address: 0xdc7f30
//
__int64 __fastcall sub_DC7F30(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4, unsigned int a5)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdi
  unsigned int v13; // r15d
  __int64 *v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v18; // rsi
  __int16 v19; // ax
  __int64 v20; // r15
  int v21; // r8d
  __int64 v22; // r9
  bool v23; // al
  __int64 v24; // r15
  __int64 v25; // rsi
  char v26; // al
  unsigned int v27; // r8d
  __int64 *v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  char v31; // r15
  int v32; // eax
  __int64 *v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rax
  bool v36; // r15
  __int64 v37; // rax
  int v38; // eax
  char v39; // dl
  __int64 v40; // rdi
  char v41; // al
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rax
  __int64 v47; // r15
  __int64 v48; // rax
  _QWORD *v49; // rax
  char v50; // al
  __int64 *v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rax
  __int64 v55; // r15
  __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rcx
  unsigned int v61; // r8d
  __int64 v62; // r15
  __int64 v63; // rax
  _QWORD *v64; // rax
  unsigned int v65; // eax
  __int64 v66; // rsi
  __int64 v67; // rdx
  __int64 v68; // rcx
  unsigned int v69; // r8d
  __int64 v70; // r15
  __int64 v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 v74; // r15
  __int64 v75; // rax
  _QWORD *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r15
  __int64 v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 v82; // r15
  __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 *v85; // rax
  __int64 v86; // rax
  _QWORD *v87; // rax
  _QWORD *v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rdx
  bool v91; // al
  __int64 v92; // rdx
  __int64 v93; // rcx
  char v94; // [rsp+0h] [rbp-90h]
  char v95; // [rsp+0h] [rbp-90h]
  int v96; // [rsp+8h] [rbp-88h]
  unsigned int v97; // [rsp+8h] [rbp-88h]
  __int64 v98; // [rsp+8h] [rbp-88h]
  unsigned int v99; // [rsp+8h] [rbp-88h]
  unsigned int v100; // [rsp+8h] [rbp-88h]
  bool v101; // [rsp+8h] [rbp-88h]
  unsigned int v102; // [rsp+8h] [rbp-88h]
  bool v103; // [rsp+8h] [rbp-88h]
  __int64 v104; // [rsp+8h] [rbp-88h]
  char v105; // [rsp+8h] [rbp-88h]
  char v106; // [rsp+8h] [rbp-88h]
  unsigned int v107; // [rsp+8h] [rbp-88h]
  bool v108; // [rsp+8h] [rbp-88h]
  unsigned int v109; // [rsp+8h] [rbp-88h]
  unsigned int v110; // [rsp+8h] [rbp-88h]
  bool v111; // [rsp+8h] [rbp-88h]
  __int64 v112; // [rsp+8h] [rbp-88h]
  char v113; // [rsp+8h] [rbp-88h]
  __int64 v114; // [rsp+8h] [rbp-88h]
  __int64 v115; // [rsp+8h] [rbp-88h]
  char v116; // [rsp+10h] [rbp-80h]
  char v117; // [rsp+10h] [rbp-80h]
  unsigned int v118; // [rsp+1Ch] [rbp-74h]
  int v119; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v120; // [rsp+30h] [rbp-60h] BYREF
  int v121; // [rsp+38h] [rbp-58h]
  __int64 v122; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v123; // [rsp+48h] [rbp-48h]
  __int64 v124[8]; // [rsp+50h] [rbp-40h] BYREF

  v118 = a5;
  if ( a5 > 2 )
    return 0;
  while ( 1 )
  {
    v9 = *a3;
    if ( *(_WORD *)(*a3 + 24) )
    {
      v116 = 0;
    }
    else
    {
      v10 = *a4;
      if ( !*(_WORD *)(*a4 + 24) )
      {
        LOBYTE(v11) = sub_B532C0(*(_QWORD *)(v9 + 32) + 24LL, (_QWORD *)(*(_QWORD *)(v10 + 32) + 24LL), *(_DWORD *)a2);
        v12 = *a1;
        v13 = v11;
        if ( !(_BYTE)v11 )
        {
          v28 = (__int64 *)sub_B2BE50(v12);
          v13 = 1;
          v29 = sub_ACD720(v28);
          v30 = sub_DA2570((__int64)a1, v29);
          *a4 = (__int64)v30;
          *a3 = (__int64)v30;
          *(_DWORD *)a2 = 33;
          *(_BYTE *)(a2 + 4) = 0;
          return v13;
        }
LABEL_5:
        v14 = (__int64 *)sub_B2BE50(v12);
        v15 = sub_ACD720(v14);
        v16 = sub_DA2570((__int64)a1, v15);
        *a4 = (__int64)v16;
        *a3 = (__int64)v16;
        *(_DWORD *)a2 = 32;
        *(_BYTE *)(a2 + 4) = 0;
        return v13;
      }
      *a3 = v10;
      *a4 = v9;
      v31 = *(_BYTE *)(a2 + 4);
      v32 = sub_B52F50(*(_DWORD *)a2);
      *(_BYTE *)(a2 + 4) = v31;
      *(_DWORD *)a2 = v32;
      v116 = 1;
    }
    v18 = *a4;
    v19 = *(_WORD *)(*a4 + 24);
    if ( v19 != 8 )
      goto LABEL_15;
    v20 = *(_QWORD *)(v18 + 48);
    if ( !sub_DADE90((__int64)a1, *a3, v20) || !(v36 = sub_DAEB50((__int64)a1, *a3, **(_QWORD **)(v20 + 32))) )
    {
      v18 = *a4;
      v19 = *(_WORD *)(*a4 + 24);
LABEL_15:
      v21 = *(_DWORD *)a2;
      goto LABEL_16;
    }
    v37 = *a3;
    *a3 = *a4;
    *a4 = v37;
    v117 = *(_BYTE *)(a2 + 4);
    v38 = sub_B52F50(*(_DWORD *)a2);
    v39 = v117;
    v116 = v36;
    *(_DWORD *)a2 = v38;
    v21 = v38;
    *(_BYTE *)(a2 + 4) = v39;
    v18 = *a4;
    v19 = *(_WORD *)(*a4 + 24);
LABEL_16:
    if ( v19 )
    {
      v40 = *a3;
      goto LABEL_32;
    }
    v22 = *(_QWORD *)(v18 + 32) + 24LL;
    if ( (unsigned int)(v21 - 32) > 1 )
      break;
LABEL_18:
    v96 = v21;
    v23 = sub_9867B0(v22);
    v24 = *a3;
    v21 = v96;
    if ( v23 && *(_WORD *)(v24 + 24) == 5 && *(_QWORD *)(v24 + 40) == 2 )
    {
      v88 = *(_QWORD **)(v24 + 32);
      v89 = *v88;
      if ( *(_WORD *)(*v88 + 24LL) == 6 && *(_QWORD *)(v89 + 40) == 2 )
      {
        v115 = *v88;
        v91 = sub_D96960(**(_QWORD **)(v89 + 32));
        v92 = v115;
        if ( !v91 )
        {
          v88 = *(_QWORD **)(v24 + 32);
          goto LABEL_96;
        }
        v93 = *(_QWORD *)(*(_QWORD *)(v24 + 32) + 8LL);
LABEL_102:
        *a3 = v93;
        v18 = *(_QWORD *)(*(_QWORD *)(v92 + 32) + 8LL);
        v116 = v91;
        *a4 = v18;
        v40 = *a3;
        v21 = *(_DWORD *)a2;
      }
      else
      {
LABEL_96:
        v90 = v88[1];
        if ( *(_WORD *)(v90 + 24) == 6 && *(_QWORD *)(v90 + 40) == 2 )
        {
          v114 = v88[1];
          v91 = sub_D96960(**(_QWORD **)(v90 + 32));
          v92 = v114;
          if ( v91 )
          {
            v93 = **(_QWORD **)(v24 + 32);
            goto LABEL_102;
          }
        }
        v18 = *a4;
        v40 = *a3;
        v21 = *(_DWORD *)a2;
      }
LABEL_32:
      v99 = v21;
      v41 = sub_D90F00(v40, v18);
      v27 = v99;
      if ( !v41 )
        goto LABEL_33;
      goto LABEL_21;
    }
    v25 = *a4;
LABEL_20:
    v97 = v21;
    v26 = sub_D90F00(v24, v25);
    v27 = v97;
    if ( !v26 )
      goto LABEL_34;
LABEL_21:
    v13 = sub_B535D0(v27);
    if ( (_BYTE)v13 )
    {
      v12 = *a1;
      goto LABEL_5;
    }
    v13 = sub_B53600(*(_DWORD *)a2);
    if ( (_BYTE)v13 )
    {
      v85 = (__int64 *)sub_B2BE50(*a1);
      v86 = sub_ACD720(v85);
      v87 = sub_DA2570((__int64)a1, v86);
      *a4 = (__int64)v87;
      *a3 = (__int64)v87;
      *(_DWORD *)a2 = 33;
      *(_BYTE *)(a2 + 4) = 0;
      return v13;
    }
    v27 = *(_DWORD *)a2;
LABEL_33:
    if ( v27 == 39 )
    {
      v66 = sub_DBB9F0((__int64)a1, *a4, 1u, 0);
      sub_AB14C0((__int64)&v122, v66);
      v106 = sub_986B30(&v122, v66, v67, v68, v69);
      sub_969240(&v122);
      if ( v106 )
      {
        v77 = sub_DBB9F0((__int64)a1, *a3, 1u, 0);
        sub_AB13A0((__int64)&v122, v77);
        if ( v123 <= 0x40 )
        {
          v95 = v123;
          v112 = v122;
          sub_969240(&v122);
          if ( v112 == (1LL << (v95 - 1)) - 1 )
            goto LABEL_8;
        }
        else
        {
          if ( (*(_QWORD *)(v122 + 8LL * ((v123 - 1) >> 6)) & (1LL << ((unsigned __int8)v123 - 1))) == 0 )
          {
            v109 = v123 - 1;
            if ( v109 == (unsigned int)sub_C445E0((__int64)&v122) )
            {
LABEL_7:
              sub_969240(&v122);
LABEL_8:
              if ( !v116 )
                return 0;
              goto LABEL_9;
            }
          }
          sub_969240(&v122);
        }
        v78 = *a3;
        v79 = sub_D95540(*a4);
        v80 = sub_DA2C50((__int64)a1, v79, 1, 1u);
        *a3 = (__int64)sub_DC7ED0(a1, (__int64)v80, v78, 4u, 0);
        *(_DWORD *)a2 = 38;
        *(_BYTE *)(a2 + 4) = 0;
      }
      else
      {
        v70 = *a4;
        v71 = sub_D95540(*a4);
        v72 = sub_DA2C50((__int64)a1, v71, -1, 1u);
        *a4 = (__int64)sub_DC7ED0(a1, (__int64)v72, v70, 4u, 0);
        *(_DWORD *)a2 = 38;
        *(_BYTE *)(a2 + 4) = 0;
      }
      goto LABEL_9;
    }
LABEL_34:
    if ( v27 > 0x27 )
    {
      if ( v27 != 41 )
        goto LABEL_8;
      v46 = sub_DBB9F0((__int64)a1, *a4, 1u, 0);
      sub_AB13A0((__int64)&v122, v46);
      if ( v123 <= 0x40 )
      {
        v94 = v123;
        v104 = v122;
        sub_969240(&v122);
        if ( v104 != (1LL << (v94 - 1)) - 1 )
          goto LABEL_47;
      }
      else
      {
        if ( (*(_QWORD *)(v122 + 8LL * ((v123 - 1) >> 6)) & (1LL << ((unsigned __int8)v123 - 1))) != 0
          || (v102 = v123 - 1, v102 != (unsigned int)sub_C445E0((__int64)&v122)) )
        {
          sub_969240(&v122);
LABEL_47:
          v47 = *a4;
          v48 = sub_D95540(*a4);
          v49 = sub_DA2C50((__int64)a1, v48, 1, 1u);
          *a4 = (__int64)sub_DC7ED0(a1, (__int64)v49, v47, 4u, 0);
          *(_DWORD *)a2 = 40;
          *(_BYTE *)(a2 + 4) = 0;
          goto LABEL_9;
        }
        sub_969240(&v122);
      }
      v58 = sub_DBB9F0((__int64)a1, *a3, 1u, 0);
      sub_AB14C0((__int64)&v122, v58);
      v105 = sub_986B30(&v122, v58, v59, v60, v61);
      sub_969240(&v122);
      if ( v105 )
        goto LABEL_8;
      v62 = *a3;
      v63 = sub_D95540(*a4);
      v64 = sub_DA2C50((__int64)a1, v63, -1, 1u);
      *a3 = (__int64)sub_DC7ED0(a1, (__int64)v64, v62, 4u, 0);
      *(_DWORD *)a2 = 40;
      *(_BYTE *)(a2 + 4) = 0;
    }
    else if ( v27 == 35 )
    {
      v73 = sub_DBB9F0((__int64)a1, *a4, 0, 0);
      sub_AB0A00((__int64)&v122, v73);
      if ( v123 <= 0x40 )
      {
        v108 = v122 == 0;
      }
      else
      {
        v107 = v123;
        v108 = v107 == (unsigned int)sub_C444A0((__int64)&v122);
      }
      sub_969240(&v122);
      if ( v108 )
      {
        v81 = sub_DBB9F0((__int64)a1, *a3, 0, 0);
        sub_AB0910((__int64)&v122, v81);
        if ( !v123 )
          goto LABEL_7;
        if ( v123 <= 0x40 )
        {
          v111 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v123) == v122;
        }
        else
        {
          v110 = v123;
          v111 = v110 == (unsigned int)sub_C445E0((__int64)&v122);
        }
        sub_969240(&v122);
        if ( v111 )
          goto LABEL_8;
        v82 = *a3;
        v83 = sub_D95540(*a4);
        v84 = sub_DA2C50((__int64)a1, v83, 1, 1u);
        *a3 = (__int64)sub_DC7ED0(a1, (__int64)v84, v82, 2u, 0);
        *(_DWORD *)a2 = 34;
        *(_BYTE *)(a2 + 4) = 0;
      }
      else
      {
        v74 = *a4;
        v75 = sub_D95540(*a4);
        v76 = sub_DA2C50((__int64)a1, v75, -1, 1u);
        *a4 = (__int64)sub_DC7ED0(a1, (__int64)v76, v74, 0, 0);
        *(_DWORD *)a2 = 34;
        *(_BYTE *)(a2 + 4) = 0;
      }
    }
    else
    {
      if ( v27 != 37 )
        goto LABEL_8;
      v42 = sub_DBB9F0((__int64)a1, *a4, 0, 0);
      sub_AB0910((__int64)&v122, v42);
      if ( v123 )
      {
        if ( v123 <= 0x40 )
        {
          v101 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v123) == v122;
        }
        else
        {
          v100 = v123;
          v101 = v100 == (unsigned int)sub_C445E0((__int64)&v122);
        }
        sub_969240(&v122);
        if ( !v101 )
        {
          v43 = *a4;
          v44 = sub_D95540(*a4);
          v45 = sub_DA2C50((__int64)a1, v44, 1, 1u);
          *a4 = (__int64)sub_DC7ED0(a1, (__int64)v45, v43, 2u, 0);
          *(_DWORD *)a2 = 36;
          *(_BYTE *)(a2 + 4) = 0;
          goto LABEL_9;
        }
      }
      else
      {
        sub_969240(&v122);
      }
      v54 = sub_DBB9F0((__int64)a1, *a3, 0, 0);
      sub_AB0A00((__int64)&v122, v54);
      v103 = sub_9867B0((__int64)&v122);
      sub_969240(&v122);
      if ( v103 )
        goto LABEL_8;
      v55 = *a3;
      v56 = sub_D95540(*a4);
      v57 = sub_DA2C50((__int64)a1, v56, -1, 1u);
      *a3 = (__int64)sub_DC7ED0(a1, (__int64)v57, v55, 0, 0);
      *(_DWORD *)a2 = 36;
      *(_BYTE *)(a2 + 4) = 0;
    }
LABEL_9:
    if ( ++v118 == 3 )
      return 0;
  }
  v98 = *(_QWORD *)(v18 + 32) + 24LL;
  sub_AB1A50((__int64)&v122, v21, v98);
  if ( sub_AAF760((__int64)&v122) )
  {
    v33 = (__int64 *)sub_B2BE50(*a1);
    v34 = sub_ACD720(v33);
    v35 = sub_DA2570((__int64)a1, v34);
    *a4 = (__int64)v35;
    *a3 = (__int64)v35;
    *(_DWORD *)a2 = 32;
    *(_BYTE *)(a2 + 4) = 0;
    goto LABEL_28;
  }
  if ( !sub_AAF7D0((__int64)&v122) )
  {
    v121 = 1;
    v120 = 0;
    v50 = sub_AAFB30((__int64)&v122, &v119, (__int64)&v120);
    if ( v50 && (unsigned int)(v119 - 32) <= 1 )
    {
      *(_DWORD *)a2 = v119;
      *(_BYTE *)(a2 + 4) = 0;
      v113 = v50;
      *a4 = (__int64)sub_DA26C0(a1, (__int64)&v120);
      sub_969240(&v120);
      sub_969240(v124);
      sub_969240(&v122);
      v18 = *a4;
      v40 = *a3;
      v21 = *(_DWORD *)a2;
      v116 = v113;
    }
    else
    {
      sub_969240(&v120);
      sub_969240(v124);
      sub_969240(&v122);
      v21 = *(_DWORD *)a2;
      v22 = v98;
      switch ( *(_DWORD *)a2 )
      {
        case ' ':
        case '!':
          goto LABEL_18;
        case '#':
          *(_DWORD *)a2 = 34;
          goto LABEL_64;
        case '%':
          *(_DWORD *)a2 = 36;
          goto LABEL_61;
        case '\'':
          *(_DWORD *)a2 = 38;
LABEL_64:
          *(_BYTE *)(a2 + 4) = 0;
          sub_9865C0((__int64)&v120, v98);
          sub_C46F20((__int64)&v120, 1u);
          break;
        case ')':
          *(_DWORD *)a2 = 40;
LABEL_61:
          *(_BYTE *)(a2 + 4) = 0;
          sub_9865C0((__int64)&v120, v98);
          sub_C46A40((__int64)&v120, 1);
          break;
        default:
          v25 = *a4;
          v24 = *a3;
          goto LABEL_20;
      }
      v65 = v121;
      v121 = 0;
      v123 = v65;
      v122 = v120;
      *a4 = (__int64)sub_DA26C0(a1, (__int64)&v122);
      sub_969240(&v122);
      sub_969240(&v120);
      v18 = *a4;
      v21 = *(_DWORD *)a2;
      v116 = 1;
      v40 = *a3;
    }
    goto LABEL_32;
  }
  v51 = (__int64 *)sub_B2BE50(*a1);
  v52 = sub_ACD720(v51);
  v53 = sub_DA2570((__int64)a1, v52);
  *a4 = (__int64)v53;
  *a3 = (__int64)v53;
  *(_DWORD *)a2 = 33;
  *(_BYTE *)(a2 + 4) = 0;
LABEL_28:
  sub_969240(v124);
  v13 = 1;
  sub_969240(&v122);
  return v13;
}
