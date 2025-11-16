// Function: sub_14C18C0
// Address: 0x14c18c0
//
__int64 __fastcall sub_14C18C0(__int64 *a1, unsigned int a2, __int64 *a3)
{
  int v3; // r15d
  unsigned int *v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // rdi
  int *v12; // rax
  int v13; // eax
  unsigned __int8 v14; // dl
  unsigned int v15; // ecx
  unsigned int v16; // ebx
  __int64 v17; // rax
  bool v18; // al
  unsigned int v19; // r8d
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // ecx
  int v24; // eax
  bool v25; // al
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r8
  _QWORD *v28; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int v31; // eax
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // eax
  _QWORD *v36; // rax
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rax
  unsigned int v42; // ebx
  unsigned __int64 *v43; // rsi
  __int64 v44; // rax
  bool v45; // al
  _QWORD *v46; // rax
  int v47; // eax
  __int64 v48; // r12
  _QWORD *v49; // rax
  __int64 v50; // rax
  char v51; // al
  unsigned int v52; // ebx
  unsigned __int64 v53; // rax
  unsigned int v54; // eax
  __int64 v55; // rax
  unsigned int v56; // esi
  unsigned __int64 *v57; // r8
  __int64 v58; // rax
  bool v59; // al
  _QWORD *v60; // rax
  unsigned int v61; // r12d
  int v62; // r14d
  int v63; // r13d
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned int v66; // eax
  __int64 **v67; // rax
  unsigned int v68; // ebx
  unsigned int v69; // r13d
  __int64 *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rbx
  __int64 *v73; // rax
  unsigned int v74; // eax
  __int64 v75; // rax
  _QWORD *v76; // rax
  unsigned int v77; // eax
  unsigned int v78; // ecx
  unsigned int v79; // ebx
  unsigned __int64 v80; // rax
  int v81; // eax
  unsigned __int64 v82; // rax
  __int64 **v83; // rax
  __int64 *v84; // r8
  bool v85; // al
  __int64 v86; // rax
  __int64 *v87; // r8
  bool v88; // al
  __int64 v89; // [rsp+8h] [rbp-A8h]
  unsigned int v90; // [rsp+10h] [rbp-A0h]
  __int64 *v91; // [rsp+10h] [rbp-A0h]
  __int64 *v92; // [rsp+10h] [rbp-A0h]
  __int64 v93; // [rsp+18h] [rbp-98h]
  int v94; // [rsp+20h] [rbp-90h]
  unsigned int v95; // [rsp+20h] [rbp-90h]
  unsigned int v96; // [rsp+20h] [rbp-90h]
  bool v97; // [rsp+20h] [rbp-90h]
  bool v98; // [rsp+20h] [rbp-90h]
  unsigned int v99; // [rsp+28h] [rbp-88h]
  unsigned int v100; // [rsp+28h] [rbp-88h]
  unsigned int v101; // [rsp+28h] [rbp-88h]
  unsigned int v102; // [rsp+30h] [rbp-80h]
  unsigned int v103; // [rsp+30h] [rbp-80h]
  __int64 v104; // [rsp+30h] [rbp-80h]
  int v105; // [rsp+30h] [rbp-80h]
  unsigned __int64 **v106; // [rsp+30h] [rbp-80h]
  unsigned __int64 **v107; // [rsp+30h] [rbp-80h]
  unsigned int v108; // [rsp+38h] [rbp-78h]
  unsigned int v109; // [rsp+38h] [rbp-78h]
  unsigned int v110; // [rsp+38h] [rbp-78h]
  __int64 v111; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v112; // [rsp+48h] [rbp-68h]
  __int64 v113; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v114; // [rsp+58h] [rbp-58h]
  __int64 *v115; // [rsp+60h] [rbp-50h] BYREF
  int v116; // [rsp+68h] [rbp-48h]
  __int64 v117; // [rsp+70h] [rbp-40h] BYREF
  int v118; // [rsp+78h] [rbp-38h]

  v3 = 0;
  while ( 2 )
  {
    v7 = (unsigned int *)sub_16D40F0(qword_4FBB370);
    if ( v7 )
      v8 = *v7;
    else
      v8 = qword_4FBB370[2];
    if ( a2 >= v8 )
      goto LABEL_30;
    v9 = *a1;
    v10 = *(_BYTE *)(*a1 + 8);
    if ( v10 != 16 )
    {
      v11 = *a3;
      if ( v10 != 15 )
        goto LABEL_7;
LABEL_32:
      v108 = sub_15A95F0(v11, v9);
      v12 = (int *)sub_16D40F0(qword_4FBB370);
      if ( v12 )
        goto LABEL_8;
      goto LABEL_33;
    }
    v11 = *a3;
    v9 = **(_QWORD **)(v9 + 16);
    if ( *(_BYTE *)(v9 + 8) == 15 )
      goto LABEL_32;
LABEL_7:
    v108 = sub_127FA20(v11, v9);
    v12 = (int *)sub_16D40F0(qword_4FBB370);
    if ( v12 )
    {
LABEL_8:
      v13 = *v12;
      goto LABEL_9;
    }
LABEL_33:
    v13 = qword_4FBB370[2];
LABEL_9:
    if ( a2 == v13 )
    {
LABEL_30:
      v15 = 1;
      return v15 + v3;
    }
    v14 = *((_BYTE *)a1 + 16);
    if ( v14 > 0x17u )
    {
      v24 = v14 - 24;
LABEL_37:
      switch ( v24 )
      {
        case 11:
          v100 = a2 + 1;
          v32 = (_QWORD *)sub_13CF970((__int64)a1);
          v102 = sub_14C18C0(*v32, a2 + 1, a3);
          if ( v102 == 1 )
            goto LABEL_113;
          v33 = sub_13CF970((__int64)a1);
          v34 = *(_QWORD *)(v33 + 24);
          if ( *(_BYTE *)(v34 + 16) > 0x10u )
            goto LABEL_54;
          if ( !(unsigned __int8)sub_1596070(v34) )
            goto LABEL_53;
          sub_14AA4E0((__int64)&v115, v108);
          v83 = (__int64 **)sub_13CF970((__int64)a1);
          sub_14B86A0(*v83, (__int64)&v115, v100, a3);
          sub_13A38D0((__int64)&v111, (__int64)&v115);
          v84 = &v111;
          if ( v112 <= 0x40 )
          {
            v111 |= 1uLL;
            sub_14A9260((unsigned __int64 *)&v111);
          }
          else
          {
            *(_QWORD *)v111 |= 1uLL;
          }
          v91 = v84;
          v114 = v112;
          v112 = 0;
          v113 = v111;
          v97 = sub_1454FB0((__int64)&v113);
          sub_135E100(&v113);
          sub_135E100(v91);
          if ( v97 )
            goto LABEL_130;
          v85 = sub_13D0200((__int64 *)&v115, v116 - 1);
          v22 = v102;
          if ( v85 )
            goto LABEL_29;
          sub_135E100(&v117);
          sub_135E100((__int64 *)&v115);
LABEL_53:
          v33 = sub_13CF970((__int64)a1);
LABEL_54:
          v35 = sub_14C18C0(*(_QWORD *)(v33 + 24), v100, a3);
          if ( v35 != 1 )
            goto LABEL_55;
          goto LABEL_67;
        case 13:
          v101 = a2 + 1;
          v39 = sub_13CF970((__int64)a1);
          v102 = sub_14C18C0(*(_QWORD *)(v39 + 24), a2 + 1, a3);
          if ( v102 == 1 )
            goto LABEL_113;
          v40 = (_QWORD *)sub_13CF970((__int64)a1);
          if ( *(_BYTE *)(*v40 + 16LL) > 0x10u )
            goto LABEL_66;
          if ( !(unsigned __int8)sub_1593BB0(*v40) )
            goto LABEL_65;
          sub_14AA4E0((__int64)&v115, v108);
          v86 = sub_13CF970((__int64)a1);
          sub_14B86A0(*(__int64 **)(v86 + 24), (__int64)&v115, v101, a3);
          sub_13A38D0((__int64)&v111, (__int64)&v115);
          v87 = &v111;
          if ( v112 <= 0x40 )
          {
            v111 |= 1uLL;
            sub_14A9260((unsigned __int64 *)&v111);
          }
          else
          {
            *(_QWORD *)v111 |= 1uLL;
          }
          v92 = v87;
          v114 = v112;
          v112 = 0;
          v113 = v111;
          v98 = sub_1454FB0((__int64)&v113);
          sub_135E100(&v113);
          sub_135E100(v92);
          if ( v98 )
          {
LABEL_130:
            v22 = v108;
            goto LABEL_29;
          }
          v88 = sub_13D0200((__int64 *)&v115, v116 - 1);
          v22 = v102;
          if ( v88 )
            goto LABEL_29;
          sub_135E100(&v117);
          sub_135E100((__int64 *)&v115);
LABEL_65:
          v40 = (_QWORD *)sub_13CF970((__int64)a1);
LABEL_66:
          v35 = sub_14C18C0(*v40, v101, a3);
          if ( v35 == 1 )
            goto LABEL_67;
LABEL_55:
          if ( v102 <= v35 )
            v35 = v102;
          v15 = v35 - 1;
          return v15 + v3;
        case 15:
          v36 = (_QWORD *)sub_13CF970((__int64)a1);
          v102 = sub_14C18C0(*v36, a2 + 1, a3);
          if ( v102 == 1 )
            goto LABEL_113;
          v37 = sub_13CF970((__int64)a1);
          v38 = sub_14C18C0(*(_QWORD *)(v37 + 24), a2 + 1, a3);
          if ( v38 == 1 )
            goto LABEL_67;
          if ( v108 < 2 * (v108 + 1) - v102 - v38 )
            goto LABEL_30;
          v15 = v102 + v108 + 1 - 2 * (v108 + 1) + v38;
          return v15 + v3;
        case 18:
          v115 = &v113;
          v41 = sub_13CF970((__int64)a1);
          if ( !(unsigned __int8)sub_13D2630(&v115, *(_BYTE **)(v41 + 24)) )
            goto LABEL_67;
          v42 = *(_DWORD *)(v113 + 8);
          v43 = *(unsigned __int64 **)v113;
          v44 = 1LL << ((unsigned __int8)v42 - 1);
          if ( v42 > 0x40 )
          {
            if ( (v43[(v42 - 1) >> 6] & v44) != 0 )
              goto LABEL_67;
            v45 = v42 == (unsigned int)sub_16A57B0(v113);
          }
          else
          {
            if ( (v44 & (unsigned __int64)v43) != 0 )
              goto LABEL_67;
            v45 = v43 == 0;
          }
          if ( v45 )
            goto LABEL_67;
          v46 = (_QWORD *)sub_13CF970((__int64)a1);
          v47 = sub_14C18C0(*v46, a2 + 1, a3);
          v48 = v113;
          v15 = v47 + *(_DWORD *)(v48 + 8) - 1 - sub_1455840(v113);
          if ( v15 > v108 )
            v15 = v108;
          return v15 + v3;
        case 21:
          v115 = &v113;
          v55 = sub_13CF970((__int64)a1);
          if ( !(unsigned __int8)sub_13D2630(&v115, *(_BYTE **)(v55 + 24)) )
            goto LABEL_67;
          v56 = *(_DWORD *)(v113 + 8);
          v57 = *(unsigned __int64 **)v113;
          v58 = 1LL << ((unsigned __int8)v56 - 1);
          if ( v56 > 0x40 )
          {
            if ( (v57[(v56 - 1) >> 6] & v58) != 0 )
              goto LABEL_67;
            v105 = *(_DWORD *)(v113 + 8);
            v59 = v105 == (unsigned int)sub_16A57B0(v113);
          }
          else
          {
            if ( (v58 & (unsigned __int64)v57) != 0 )
              goto LABEL_67;
            v59 = v57 == 0;
          }
          if ( v59 )
            goto LABEL_67;
          v60 = (_QWORD *)sub_13CF970((__int64)a1);
          v61 = sub_14C18C0(*v60, a2 + 1, a3);
          sub_13A38D0((__int64)&v115, v113);
          sub_16A7770(&v115);
          v62 = v116;
          v63 = sub_1455840((__int64)&v115);
          sub_135E100((__int64 *)&v115);
          v15 = v108 - v62 + v63;
          if ( v15 < v61 )
            v15 = v61;
          return v15 + v3;
        case 23:
          v115 = &v113;
          v75 = sub_13CF970((__int64)a1);
          if ( !(unsigned __int8)sub_13D2630(&v115, *(_BYTE **)(v75 + 24)) )
            goto LABEL_67;
          v76 = (_QWORD *)sub_13CF970((__int64)a1);
          v77 = sub_14C18C0(*v76, a2 + 1, a3);
          v78 = v77;
          v79 = *(_DWORD *)(v113 + 8);
          if ( v79 > 0x40 )
          {
            v107 = (unsigned __int64 **)v113;
            v96 = v77;
            if ( v79 - (unsigned int)sub_16A57B0(v113) <= 0x40 )
            {
              v82 = **v107;
              if ( v108 > v82 && v82 < v96 )
              {
                v15 = v96 - v82;
                return v15 + v3;
              }
            }
          }
          else
          {
            v80 = *(_QWORD *)v113;
            if ( (unsigned __int64)v108 > *(_QWORD *)v113 && v78 > v80 )
            {
              v15 = v78 - v80;
              return v15 + v3;
            }
          }
          goto LABEL_67;
        case 25:
          v49 = (_QWORD *)sub_13CF970((__int64)a1);
          v103 = sub_14C18C0(*v49, a2 + 1, a3);
          v115 = &v113;
          v50 = sub_13CF970((__int64)a1);
          v51 = sub_13D2630(&v115, *(_BYTE **)(v50 + 24));
          v15 = v103;
          if ( !v51 )
            return v15 + v3;
          v52 = *(_DWORD *)(v113 + 8);
          if ( v52 > 0x40 )
          {
            v95 = v103;
            v106 = (unsigned __int64 **)v113;
            v81 = sub_16A57B0(v113);
            v15 = v95;
            if ( v52 - v81 <= 0x40 )
            {
              v53 = **v106;
              if ( v108 > v53 )
              {
LABEL_79:
                v54 = v15 + v53;
                v15 = v108;
                if ( v54 <= v108 )
                  v15 = v54;
                return v15 + v3;
              }
            }
          }
          else
          {
            v53 = *(_QWORD *)v113;
            if ( (unsigned __int64)v108 > *(_QWORD *)v113 )
              goto LABEL_79;
          }
LABEL_67:
          v102 = 1;
          v14 = *((_BYTE *)a1 + 16);
          goto LABEL_13;
        case 26:
        case 27:
        case 28:
          v28 = (_QWORD *)sub_13CF970((__int64)a1);
          v102 = sub_14C18C0(*v28, a2 + 1, a3);
          if ( v102 == 1 )
            goto LABEL_113;
          v29 = sub_13CF970((__int64)a1);
          v30 = sub_14C18C0(*(_QWORD *)(v29 + 24), a2 + 1, a3);
          v14 = *((_BYTE *)a1 + 16);
          if ( v102 <= v30 )
            v30 = v102;
          v102 = v30;
          goto LABEL_13;
        case 38:
          ++a2;
          v67 = (__int64 **)sub_13CF970((__int64)a1);
          a1 = *v67;
          v3 = v108 + v3 - sub_16431D0(**v67);
          continue;
        case 53:
          v102 = 1;
          v68 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
          if ( v68 - 1 > 3 )
            goto LABEL_13;
          v69 = a2 + 1;
          if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
            v70 = (__int64 *)*(a1 - 1);
          else
            v70 = &a1[-3 * v68];
          v15 = sub_14C18C0(*v70, v69, a3);
          if ( v68 == 1 )
            return v15 + v3;
          v71 = v68 - 2;
          v72 = 24;
          v104 = 24 * v71 + 48;
          while ( v15 != 1 )
          {
            if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
              v73 = (__int64 *)*(a1 - 1);
            else
              v73 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
            v110 = v15;
            v74 = sub_14C18C0(v73[(unsigned __int64)v72 / 8], v69, a3);
            v15 = v110;
            if ( v110 > v74 )
              v15 = v74;
            v72 += 24;
            if ( v104 == v72 )
              return v15 + v3;
          }
          goto LABEL_30;
        case 55:
          v64 = sub_13CF970((__int64)a1);
          v102 = sub_14C18C0(*(_QWORD *)(v64 + 24), a2 + 1, a3);
          if ( v102 == 1 )
          {
LABEL_113:
            v14 = *((_BYTE *)a1 + 16);
            goto LABEL_13;
          }
          v65 = sub_13CF970((__int64)a1);
          v66 = sub_14C18C0(*(_QWORD *)(v65 + 48), a2 + 1, a3);
          v15 = v102;
          if ( v102 > v66 )
            v15 = v66;
          return v15 + v3;
        case 59:
          ++a2;
          a1 = *(__int64 **)sub_13CF970((__int64)a1);
          continue;
        default:
          goto LABEL_12;
      }
    }
    break;
  }
  if ( v14 == 5 )
  {
    v24 = *((unsigned __int16 *)a1 + 9);
    goto LABEL_37;
  }
LABEL_12:
  v102 = 1;
LABEL_13:
  if ( v14 > 0x10u || *(_BYTE *)(*a1 + 8) != 16 )
    goto LABEL_25;
  v15 = v108;
  v94 = *(_QWORD *)(*a1 + 32);
  if ( v94 )
  {
    v16 = 0;
    while ( 1 )
    {
      v99 = v15;
      v17 = sub_15A0A60(a1, v16);
      if ( !v17 || *(_BYTE *)(v17 + 16) != 13 )
        break;
      v89 = v17;
      v93 = v17 + 24;
      v90 = *(_DWORD *)(v17 + 32);
      v18 = sub_13D0200((__int64 *)(v17 + 24), v90 - 1);
      v19 = v90;
      v15 = v99;
      if ( v18 )
      {
        v20 = sub_13D05A0(v93);
        v15 = v99;
        v19 = v20;
      }
      else if ( v90 > 0x40 )
      {
        v31 = sub_16A57B0(v93);
        v15 = v99;
        v19 = v31;
      }
      else
      {
        v26 = *(_QWORD *)(v89 + 24);
        if ( v26 )
        {
          _BitScanReverse64(&v27, v26);
          v19 = v90 - 64 + (v27 ^ 0x3F);
        }
      }
      if ( v15 > v19 )
        v15 = v19;
      if ( v94 == ++v16 )
        goto LABEL_24;
    }
LABEL_25:
    sub_14AA4E0((__int64)&v115, v108);
    sub_14B86A0(a1, (__int64)&v115, a2, a3);
    if ( sub_13D0200((__int64 *)&v115, v116 - 1) )
    {
      v21 = sub_13D05A0((__int64)&v115);
    }
    else
    {
      v25 = sub_13D0200(&v117, v118 - 1);
      v22 = v102;
      if ( !v25 )
      {
LABEL_29:
        v109 = v22;
        sub_135E100(&v117);
        sub_135E100((__int64 *)&v115);
        v15 = v109;
        return v15 + v3;
      }
      v21 = sub_13D05A0((__int64)&v117);
    }
    v22 = v102;
    if ( v102 < v21 )
      v22 = v21;
    goto LABEL_29;
  }
LABEL_24:
  if ( !v15 )
    goto LABEL_25;
  return v15 + v3;
}
