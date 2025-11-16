// Function: sub_AABE40
// Address: 0xaabe40
//
__int64 __fastcall sub_AABE40(unsigned int a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v6; // rbx
  unsigned __int8 *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int8 v11; // al
  unsigned __int64 *v12; // rdi
  __int64 v13; // rdx
  int v14; // r11d
  int v15; // eax
  unsigned __int8 *v16; // rsi
  unsigned __int8 *v17; // r8
  unsigned __int8 v18; // al
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned int v25; // eax
  unsigned int v26; // r8d
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r8
  unsigned __int64 v29; // rax
  char v30; // al
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // rdx
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r13
  __int64 v39; // rsi
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 result; // rax
  unsigned __int8 *v43; // rdx
  unsigned __int8 *v44; // rsi
  __int64 v45; // rdx
  unsigned __int8 *v46; // rax
  int v47; // eax
  bool v48; // al
  int v49; // eax
  bool v50; // al
  __int64 v51; // rax
  unsigned __int8 *v52; // r12
  _QWORD *v53; // rax
  __int64 v54; // rax
  unsigned int v55; // eax
  unsigned __int64 v56; // rdx
  _QWORD *v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // rax
  _QWORD *v60; // rdi
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // r15
  unsigned int v64; // ebx
  unsigned int v65; // r13d
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // r15
  unsigned int v68; // ebx
  unsigned int v69; // r13d
  unsigned __int64 v70; // rax
  unsigned int v71; // ebx
  int v72; // eax
  bool v73; // al
  unsigned int v74; // ebx
  int v75; // eax
  __int64 v76; // rcx
  bool v77; // al
  unsigned __int64 v78; // r15
  unsigned int v79; // ebx
  unsigned int v80; // r13d
  unsigned __int64 v81; // rax
  unsigned __int8 *v82; // rsi
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  _QWORD **v87; // rax
  _BYTE *v88; // rax
  unsigned int v89; // eax
  __int64 v90; // rdi
  unsigned int v91; // ebx
  int v92; // eax
  unsigned int v93; // eax
  __int64 v94; // rdi
  unsigned int v95; // ebx
  int v96; // eax
  int v97; // eax
  int v98; // eax
  int v99; // eax
  _BYTE *v100; // rax
  __int64 v101; // rax
  int v102; // eax
  unsigned int v103; // eax
  __int64 v104; // [rsp+18h] [rbp-F8h]
  __int64 v105; // [rsp+18h] [rbp-F8h]
  __int64 v106; // [rsp+20h] [rbp-F0h]
  __int64 v107; // [rsp+20h] [rbp-F0h]
  __int64 v108; // [rsp+20h] [rbp-F0h]
  __int64 v109; // [rsp+20h] [rbp-F0h]
  unsigned int v110; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v111; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v112; // [rsp+20h] [rbp-F0h]
  __int64 v113; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v114; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v115; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v116; // [rsp+20h] [rbp-F0h]
  unsigned int v117; // [rsp+28h] [rbp-E8h]
  _QWORD *v118; // [rsp+28h] [rbp-E8h]
  __int64 v119; // [rsp+28h] [rbp-E8h]
  __int64 v120; // [rsp+28h] [rbp-E8h]
  __int64 v121; // [rsp+28h] [rbp-E8h]
  __int64 v122; // [rsp+28h] [rbp-E8h]
  __int64 v123; // [rsp+28h] [rbp-E8h]
  __int64 v124; // [rsp+28h] [rbp-E8h]
  __int64 v125; // [rsp+28h] [rbp-E8h]
  unsigned int v126; // [rsp+28h] [rbp-E8h]
  __int64 v127; // [rsp+28h] [rbp-E8h]
  __int64 v128; // [rsp+28h] [rbp-E8h]
  __int64 v129; // [rsp+28h] [rbp-E8h]
  __int64 v130; // [rsp+28h] [rbp-E8h]
  unsigned __int64 *v131; // [rsp+28h] [rbp-E8h]
  char v132; // [rsp+28h] [rbp-E8h]
  unsigned int v133; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v134; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v135; // [rsp+38h] [rbp-D8h]
  _QWORD *v136; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v137; // [rsp+48h] [rbp-C8h]
  _QWORD *v138; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v139; // [rsp+58h] [rbp-B8h]
  _BYTE v140[176]; // [rsp+60h] [rbp-B0h] BYREF

  v6 = (1LL << a1) & 0x70066000;
  while ( 1 )
  {
    v7 = (unsigned __int8 *)*((_QWORD *)a2 + 1);
    v8 = sub_AD93D0(a1, v7, 0, 0);
    if ( v8 )
    {
      if ( (unsigned __int8 *)v8 == a2 )
        return (__int64)a3;
    }
    else
    {
      v7 = (unsigned __int8 *)*((_QWORD *)a2 + 1);
      v8 = sub_AD93D0(a1, v7, 1, 0);
      if ( !v8 )
        goto LABEL_5;
    }
    if ( (unsigned __int8 *)v8 == a3 )
      return (__int64)a2;
LABEL_5:
    v11 = *a2;
    v12 = (unsigned __int64 *)*((_QWORD *)a2 + 1);
    if ( *a2 == 13 )
      return sub_ACADE0(v12);
    v13 = *a3;
    if ( (_BYTE)v13 == 13 )
      return sub_ACADE0(v12);
    v14 = *((unsigned __int8 *)v12 + 8);
    if ( v14 != 17 )
    {
      v9 = (unsigned int)v11 - 12;
      if ( (unsigned int)v9 <= 1
        || (v7 = (unsigned __int8 *)((unsigned int)(unsigned __int8)v13 - 12), (unsigned int)v7 <= 1) )
      {
        v7 = (unsigned __int8 *)(a1 - 13);
        switch ( a1 )
        {
          case 0xDu:
          case 0xFu:
            return sub_ACA8A0(v12);
          case 0xEu:
          case 0x12u:
          case 0x15u:
          case 0x18u:
            goto LABEL_78;
          case 0x10u:
            v138 = 0;
            if ( (unsigned __int8)sub_AABC70(&v138, (__int64)a2, v13, v9, v10) )
            {
              v10 = (__int64)a3;
              if ( (unsigned int)*a3 - 12 <= 1 )
                return v10;
            }
            v9 = (unsigned int)*a2 - 12;
LABEL_78:
            if ( (unsigned int)v9 <= 1 )
            {
              v10 = (__int64)a2;
              if ( (unsigned int)*a3 - 12 <= 1 )
                return v10;
            }
            return sub_AD8F60(*((_QWORD *)a2 + 1), 0, 0, v9, v10);
          case 0x11u:
            if ( (unsigned int)v9 <= 1 )
            {
              v10 = (__int64)a2;
              if ( (unsigned int)(unsigned __int8)v13 - 12 <= 1 )
                return v10;
            }
            if ( v11 == 17 )
            {
              v52 = a2 + 24;
              goto LABEL_129;
            }
            if ( (unsigned int)(v14 - 17) > 1 )
              goto LABEL_225;
            v88 = (_BYTE *)sub_AD7630(a2, 0);
            if ( v88 && *v88 == 17 )
            {
              v12 = (unsigned __int64 *)*((_QWORD *)a2 + 1);
              v52 = v88 + 24;
            }
            else
            {
              LOBYTE(v13) = *a3;
LABEL_225:
              if ( (_BYTE)v13 == 17 )
              {
                v12 = (unsigned __int64 *)*((_QWORD *)a2 + 1);
                v52 = a3 + 24;
              }
              else
              {
                if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a3 + 1) + 8LL) - 17 > 1
                  || (v100 = (_BYTE *)sub_AD7630(a3, 0)) == 0
                  || *v100 != 17 )
                {
LABEL_73:
                  v12 = (unsigned __int64 *)*((_QWORD *)a2 + 1);
                  return sub_AD6530(v12);
                }
                v12 = (unsigned __int64 *)*((_QWORD *)a2 + 1);
                v52 = v100 + 24;
              }
            }
LABEL_129:
            v53 = *(_QWORD **)v52;
            if ( *((_DWORD *)v52 + 2) > 0x40u )
              v53 = (_QWORD *)*v53;
            if ( ((unsigned __int8)v53 & 1) != 0 )
              return sub_ACA8A0(v12);
            return sub_AD6530(v12);
          case 0x13u:
          case 0x14u:
          case 0x16u:
          case 0x17u:
            if ( !(unsigned __int8)sub_AA9140((__int64)a3) )
              goto LABEL_73;
            goto LABEL_82;
          case 0x19u:
          case 0x1Au:
          case 0x1Bu:
            if ( (unsigned int)(v13 - 12) <= 1 )
              goto LABEL_82;
            return sub_AD6530(v12);
          case 0x1Cu:
            if ( (unsigned int)v9 > 1 )
              return sub_AD6530(v12);
            v10 = (__int64)a2;
            if ( (unsigned int)(v13 - 12) > 1 )
              return sub_AD6530(v12);
            return v10;
          case 0x1Du:
            if ( (unsigned int)v9 <= 1 )
            {
              v10 = (__int64)a2;
              if ( (unsigned int)(v13 - 12) <= 1 )
                return v10;
            }
            return sub_AD62B0(v12);
          case 0x1Eu:
            if ( (unsigned int)v9 > 1 || (unsigned int)(v13 - 12) > 1 )
              return sub_ACA8A0(v12);
            return sub_AD6530(v12);
          case 0x1Fu:
            BUG();
          default:
            break;
        }
      }
    }
    if ( (_BYTE)v13 == 17 )
      break;
    if ( v11 != 17 )
      goto LABEL_23;
    if ( a1 > 0x1E || !v6 )
      goto LABEL_20;
    if ( (unsigned __int8)sub_AC47B0(a1) )
    {
      v43 = a2;
      v44 = a3;
      return sub_AD5570(a1, v44, v43, 0, 0);
    }
LABEL_106:
    v46 = a2;
    a2 = a3;
    a3 = v46;
  }
  v7 = (unsigned __int8 *)*((_QWORD *)a3 + 1);
  v12 = (unsigned __int64 *)a1;
  if ( a3 == (unsigned __int8 *)sub_AD6840(a1, v7, 0) )
    return (__int64)a3;
  if ( a1 <= 0x17 )
  {
    if ( a1 > 0x15 )
    {
      v9 = *((unsigned int *)a3 + 8);
      v12 = (unsigned __int64 *)(a3 + 24);
      if ( (unsigned int)v9 <= 0x40 )
      {
        v48 = *((_QWORD *)a3 + 3) == 1;
      }
      else
      {
        v110 = *((_DWORD *)a3 + 8);
        v47 = sub_C444A0(v12);
        v9 = v110;
        v12 = (unsigned __int64 *)(a3 + 24);
        v13 = v110 - 1;
        v48 = (_DWORD)v13 == v47;
      }
      if ( v48 )
        return sub_AD6530(*((_QWORD *)a3 + 1));
      if ( (unsigned int)v9 <= 0x40 )
      {
        v50 = *((_QWORD *)a3 + 3) == 0;
      }
      else
      {
        v126 = v9;
        v49 = sub_C444A0(v12);
        v9 = v126;
        v50 = v126 == v49;
      }
      if ( v50 )
      {
LABEL_82:
        v12 = (unsigned __int64 *)*((_QWORD *)a3 + 1);
        return sub_ACADE0(v12);
      }
    }
    else if ( a1 - 19 <= 1 )
    {
      v13 = *((unsigned int *)a3 + 8);
      if ( (unsigned int)v13 <= 0x40 )
      {
        if ( !*((_QWORD *)a3 + 3) )
          goto LABEL_82;
      }
      else
      {
        v12 = (unsigned __int64 *)(a3 + 24);
        v117 = *((_DWORD *)a3 + 8);
        v15 = sub_C444A0(a3 + 24);
        v13 = v117;
        if ( v117 == v15 )
          goto LABEL_82;
      }
    }
    goto LABEL_18;
  }
  if ( a1 != 28 )
    goto LABEL_18;
  v11 = *a2;
  if ( *a2 == 5 )
  {
    if ( *((_WORD *)a2 + 1) != 47 )
      goto LABEL_53;
    v23 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)v23 > 3u )
      goto LABEL_53;
    v24 = *(_QWORD *)(v23 + 40);
    if ( !v24 )
    {
      if ( *(_BYTE *)v23 != 3 )
        goto LABEL_53;
      v9 = *(unsigned __int16 *)(v23 + 34);
      LOWORD(v9) = ((unsigned __int16)v9 >> 1) & 0x3F;
      if ( !(_WORD)v9 )
        goto LABEL_53;
      LOBYTE(v9) = v9 - 1;
      if ( !(_BYTE)v9 )
        goto LABEL_53;
LABEL_36:
      v13 = (unsigned __int8)v9;
      v135 = *((_DWORD *)a3 + 8);
      if ( (unsigned __int8)v9 > v135 )
        v13 = v135;
      v26 = v13;
      if ( v135 > 0x40 )
      {
        v133 = v13;
        v12 = &v134;
        v7 = 0;
        sub_C43690(&v134, 0, 0);
        v26 = v133;
      }
      else
      {
        v134 = 0;
      }
      if ( v26 )
      {
        if ( v26 > 0x40 )
        {
          v12 = &v134;
          v7 = 0;
          sub_C43C90(&v134, 0, v26);
        }
        else
        {
          v13 = v134;
          v9 = 64 - v26;
          v27 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26);
          if ( v135 > 0x40 )
            *(_QWORD *)v134 |= v27;
          else
            v134 |= v27;
        }
      }
      v137 = *((_DWORD *)a3 + 8);
      if ( v137 > 0x40 )
      {
        v7 = a3 + 24;
        sub_C43780(&v136, a3 + 24);
        v12 = (unsigned __int64 *)&v136;
        if ( v137 > 0x40 )
        {
          v7 = (unsigned __int8 *)&v134;
          sub_C43B90(&v136, &v134);
          v103 = v137;
          v10 = (__int64)v136;
          v137 = 0;
          LODWORD(v139) = v103;
          v138 = v136;
          if ( v103 > 0x40 )
          {
            v7 = a3 + 24;
            v131 = v136;
            v12 = (unsigned __int64 *)&v138;
            v30 = sub_C43C50(&v138, a3 + 24);
            v10 = (__int64)v131;
            if ( v131 )
            {
              v12 = v131;
              v132 = v30;
              j_j___libc_free_0_0(v12);
              v30 = v132;
              if ( v137 > 0x40 )
              {
                v12 = v136;
                if ( v136 )
                {
                  j_j___libc_free_0_0(v136);
                  v30 = v132;
                }
              }
            }
            goto LABEL_48;
          }
          v29 = *((_QWORD *)a3 + 3);
LABEL_47:
          v30 = v10 == v29;
LABEL_48:
          if ( v30 )
          {
            v10 = sub_AD6530(*((_QWORD *)a3 + 1));
            if ( v135 > 0x40 )
            {
              v60 = (_QWORD *)v134;
              if ( v134 )
              {
LABEL_149:
                v128 = v10;
                j_j___libc_free_0_0(v60);
                return v128;
              }
            }
            return v10;
          }
          if ( v135 > 0x40 )
          {
            v12 = (unsigned __int64 *)v134;
            if ( v134 )
              j_j___libc_free_0_0(v134);
          }
          goto LABEL_18;
        }
        v28 = (unsigned __int64)v136;
        v29 = *((_QWORD *)a3 + 3);
      }
      else
      {
        v28 = *((_QWORD *)a3 + 3);
        v29 = v28;
      }
      v10 = v134 & v28;
      goto LABEL_47;
    }
    v12 = *(unsigned __int64 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v7 = (unsigned __int8 *)(v24 + 312);
    v106 = *(_QWORD *)(v23 + 40);
    v25 = sub_BD5420(v12, v24 + 312);
    if ( !*(_BYTE *)v12 )
    {
      v13 = v106;
      v9 = 2;
      if ( !*(_BYTE *)(v106 + 331) )
        goto LABEL_36;
    }
    v9 = v25;
    if ( (unsigned __int64)(1LL << v25) > 1 )
      goto LABEL_36;
LABEL_18:
    v11 = *a2;
  }
  if ( v11 == 17 )
  {
LABEL_20:
    if ( *a3 == 17 )
    {
      v16 = a2 + 24;
      v17 = a3 + 24;
      switch ( a1 )
      {
        case 0xDu:
          v137 = *((_DWORD *)a2 + 8);
          if ( v137 > 0x40 )
          {
            sub_C43780(&v136, v16);
            v17 = a3 + 24;
          }
          else
          {
            v136 = (_QWORD *)*((_QWORD *)a2 + 3);
          }
          sub_C45EE0(&v136, v17);
          LODWORD(v139) = v137;
          v138 = v136;
          goto LABEL_144;
        case 0xFu:
          v137 = *((_DWORD *)a2 + 8);
          if ( v137 > 0x40 )
          {
            sub_C43780(&v136, v16);
            v17 = a3 + 24;
          }
          else
          {
            v136 = (_QWORD *)*((_QWORD *)a2 + 3);
          }
          sub_C46B40(&v136, v17);
          LODWORD(v139) = v137;
          v138 = v136;
          goto LABEL_144;
        case 0x11u:
          sub_C472A0(&v138, v16, v17);
          goto LABEL_179;
        case 0x13u:
          sub_C4A1D0(&v138, v16, v17);
          goto LABEL_179;
        case 0x14u:
          v74 = *((_DWORD *)a3 + 8);
          if ( v74 )
          {
            if ( v74 <= 0x40 )
            {
              v76 = 64 - v74;
              v77 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v74) == *((_QWORD *)a3 + 3);
            }
            else
            {
              v75 = sub_C445E0(a3 + 24);
              v17 = a3 + 24;
              v16 = a2 + 24;
              v77 = v74 == v75;
            }
            if ( !v77 )
              goto LABEL_186;
          }
          v93 = *((_DWORD *)a2 + 8);
          v94 = *((_QWORD *)a2 + 3);
          v95 = v93 - 1;
          if ( v93 <= 0x40 )
          {
            v76 = v95;
            if ( v94 == 1LL << v95 )
              return sub_ACADE0(*((_QWORD *)a2 + 1));
          }
          else
          {
            v76 = v95;
            if ( (*(_QWORD *)(v94 + 8LL * (v95 >> 6)) & (1LL << v95)) != 0 )
            {
              v112 = v17;
              v96 = sub_C44590(v16);
              v17 = v112;
              if ( v96 == v95 )
                return sub_ACADE0(*((_QWORD *)a2 + 1));
            }
          }
LABEL_186:
          sub_C4A3E0(&v138, v16, v17, v76);
          goto LABEL_179;
        case 0x16u:
          sub_C4B490(&v138, v16, v17);
          goto LABEL_179;
        case 0x17u:
          v71 = *((_DWORD *)a3 + 8);
          if ( v71 )
          {
            if ( v71 <= 0x40 )
            {
              v73 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v71) == *((_QWORD *)a3 + 3);
            }
            else
            {
              v72 = sub_C445E0(a3 + 24);
              v17 = a3 + 24;
              v16 = a2 + 24;
              v73 = v71 == v72;
            }
            if ( !v73 )
              goto LABEL_178;
          }
          v89 = *((_DWORD *)a2 + 8);
          v90 = *((_QWORD *)a2 + 3);
          v91 = v89 - 1;
          if ( v89 <= 0x40 )
          {
            if ( 1LL << v91 == v90 )
              return sub_ACADE0(*((_QWORD *)a2 + 1));
          }
          else if ( (*(_QWORD *)(v90 + 8LL * (v91 >> 6)) & (1LL << v91)) != 0 )
          {
            v111 = v17;
            v92 = sub_C44590(v16);
            v17 = v111;
            if ( v91 == v92 )
              return sub_ACADE0(*((_QWORD *)a2 + 1));
          }
LABEL_178:
          sub_C4B8A0(&v138, v16, v17);
LABEL_179:
          v10 = sub_AD8D80(*((_QWORD *)a2 + 1), &v138);
          if ( (unsigned int)v139 > 0x40 )
            goto LABEL_166;
          return v10;
        case 0x19u:
          v78 = *((unsigned int *)a2 + 8);
          v79 = *((_DWORD *)a3 + 8);
          v80 = *((_DWORD *)a2 + 8);
          if ( v79 > 0x40 )
          {
            v99 = sub_C444A0(a3 + 24);
            v17 = a3 + 24;
            v16 = a2 + 24;
            if ( v79 - v99 > 0x40 )
              return sub_ACADE0(*((_QWORD *)a2 + 1));
            v81 = **((_QWORD **)a3 + 3);
          }
          else
          {
            v81 = *((_QWORD *)a3 + 3);
          }
          if ( v78 <= v81 )
            return sub_ACADE0(*((_QWORD *)a2 + 1));
          LODWORD(v139) = v80;
          if ( v80 > 0x40 )
          {
            v116 = v17;
            sub_C43780(&v138, v16);
            v17 = v116;
          }
          else
          {
            v138 = (_QWORD *)*((_QWORD *)a2 + 3);
          }
          sub_C47AC0(&v138, v17);
          goto LABEL_165;
        case 0x1Au:
          v67 = *((unsigned int *)a2 + 8);
          v68 = *((_DWORD *)a3 + 8);
          v69 = *((_DWORD *)a2 + 8);
          if ( v68 > 0x40 )
          {
            v97 = sub_C444A0(a3 + 24);
            v17 = a3 + 24;
            v16 = a2 + 24;
            if ( v68 - v97 > 0x40 )
              return sub_ACADE0(*((_QWORD *)a2 + 1));
            v70 = **((_QWORD **)a3 + 3);
          }
          else
          {
            v70 = *((_QWORD *)a3 + 3);
          }
          if ( v67 <= v70 )
            return sub_ACADE0(*((_QWORD *)a2 + 1));
          LODWORD(v139) = v69;
          if ( v69 > 0x40 )
          {
            v115 = v17;
            sub_C43780(&v138, v16);
            v17 = v115;
          }
          else
          {
            v138 = (_QWORD *)*((_QWORD *)a2 + 3);
          }
          sub_C48380(&v138, v17);
          goto LABEL_165;
        case 0x1Bu:
          v63 = *((unsigned int *)a2 + 8);
          v64 = *((_DWORD *)a3 + 8);
          v65 = *((_DWORD *)a2 + 8);
          if ( v64 > 0x40 )
          {
            v98 = sub_C444A0(a3 + 24);
            v17 = a3 + 24;
            v16 = a2 + 24;
            if ( v64 - v98 > 0x40 )
              return sub_ACADE0(*((_QWORD *)a2 + 1));
            v66 = **((_QWORD **)a3 + 3);
          }
          else
          {
            v66 = *((_QWORD *)a3 + 3);
          }
          if ( v63 <= v66 )
            return sub_ACADE0(*((_QWORD *)a2 + 1));
          LODWORD(v139) = v65;
          if ( v65 > 0x40 )
          {
            v114 = v17;
            sub_C43780(&v138, v16);
            v17 = v114;
          }
          else
          {
            v138 = (_QWORD *)*((_QWORD *)a2 + 3);
          }
          sub_C44D10(&v138, v17);
LABEL_165:
          v10 = sub_AD8D80(*((_QWORD *)a2 + 1), &v138);
          if ( (unsigned int)v139 > 0x40 )
          {
LABEL_166:
            v60 = v138;
            if ( v138 )
              goto LABEL_149;
          }
          return v10;
        case 0x1Cu:
          v55 = *((_DWORD *)a2 + 8);
          v137 = v55;
          if ( v55 <= 0x40 )
          {
            v62 = *((_QWORD *)a2 + 3);
LABEL_155:
            v57 = (_QWORD *)(*((_QWORD *)a3 + 3) & v62);
            v136 = v57;
            goto LABEL_143;
          }
          sub_C43780(&v136, v16);
          v55 = v137;
          if ( v137 <= 0x40 )
          {
            v62 = (unsigned __int64)v136;
            goto LABEL_155;
          }
          sub_C43B90(&v136, a3 + 24);
          v55 = v137;
          v57 = v136;
          goto LABEL_143;
        case 0x1Du:
          v55 = *((_DWORD *)a2 + 8);
          v137 = v55;
          if ( v55 <= 0x40 )
          {
            v61 = *((_QWORD *)a2 + 3);
LABEL_152:
            v57 = (_QWORD *)(*((_QWORD *)a3 + 3) | v61);
            v136 = v57;
            goto LABEL_143;
          }
          sub_C43780(&v136, v16);
          v55 = v137;
          if ( v137 <= 0x40 )
          {
            v61 = (unsigned __int64)v136;
            goto LABEL_152;
          }
          sub_C43BD0(&v136, a3 + 24);
          v55 = v137;
          v57 = v136;
          goto LABEL_143;
        case 0x1Eu:
          v55 = *((_DWORD *)a2 + 8);
          v137 = v55;
          if ( v55 <= 0x40 )
          {
            v56 = *((_QWORD *)a2 + 3);
LABEL_142:
            v57 = (_QWORD *)(*((_QWORD *)a3 + 3) ^ v56);
            v136 = v57;
            goto LABEL_143;
          }
          sub_C43780(&v136, v16);
          v55 = v137;
          if ( v137 <= 0x40 )
          {
            v56 = (unsigned __int64)v136;
            goto LABEL_142;
          }
          sub_C43C10(&v136, a3 + 24);
          v55 = v137;
          v57 = v136;
LABEL_143:
          LODWORD(v139) = v55;
          v138 = v57;
LABEL_144:
          v58 = *((_QWORD *)a2 + 1);
          v137 = 0;
          v59 = sub_AD8D80(v58, &v138);
          v10 = v59;
          if ( (unsigned int)v139 > 0x40 && v138 )
          {
            v127 = v59;
            j_j___libc_free_0_0(v138);
            v10 = v127;
          }
          if ( v137 > 0x40 )
          {
            v60 = v136;
            if ( v136 )
              goto LABEL_149;
          }
          return v10;
        default:
          break;
      }
    }
    if ( a2 == (unsigned __int8 *)sub_AD6840(a1, *((_QWORD *)a2 + 1), 1) )
      return (__int64)a2;
LABEL_53:
    v31 = *((_QWORD *)a2 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
      goto LABEL_92;
    goto LABEL_54;
  }
LABEL_23:
  if ( v11 != 18 )
    goto LABEL_53;
  v18 = *a3;
  if ( *a3 == 18 )
  {
    v19 = sub_C33340(v12, v7, v13, v9, v10);
    v20 = a2 + 24;
    v118 = (_QWORD *)v19;
    if ( *((_QWORD *)a2 + 3) == v19 )
      sub_C3C790(&v138, v20);
    else
      sub_C33EB0(&v138, v20);
    switch ( a1 )
    {
      case 0xEu:
        v82 = a3 + 24;
        if ( v118 == v138 )
          sub_C3D800(&v138, v82, 1);
        else
          sub_C3ADF0(&v138, v82, 1);
        goto LABEL_200;
      case 0x10u:
        v82 = a3 + 24;
        if ( v118 == v138 )
          sub_C3D820(&v138, v82, 1, v22);
        else
          sub_C3B1F0(&v138, v82, 1, v22);
        goto LABEL_200;
      case 0x12u:
        v82 = a3 + 24;
        if ( v118 == v138 )
          sub_C3F5C0(&v138, v82, 1);
        else
          sub_C3B950(&v138, v82, 1);
        goto LABEL_200;
      case 0x15u:
        v82 = a3 + 24;
        if ( v118 == v138 )
          sub_C3EF50(&v138, v82, 1);
        else
          sub_C3B6C0(&v138, v82, 1);
        goto LABEL_200;
      case 0x18u:
        v82 = a3 + 24;
        if ( v118 == v138 )
          sub_C3EC80(&v138, v82, v21, v22);
        else
          sub_C3BE30(&v138, v82, v21, v22);
LABEL_200:
        v87 = &v138;
        if ( v118 == v138 )
          v87 = (_QWORD **)v139;
        if ( (*((_BYTE *)v87 + 20) & 7) == 1 && (unsigned __int8)sub_C33750(v138, v82, v83, v84, v85, v86) )
          goto LABEL_137;
        v129 = sub_AD8F10(*((_QWORD *)a2 + 1), &v138);
        sub_91D830(&v138);
        return v129;
      default:
LABEL_137:
        sub_91D830(&v138);
        goto LABEL_53;
    }
  }
  v31 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
  {
LABEL_103:
    if ( v18 != 5 || a1 > 0x1E || !v6 )
      goto LABEL_96;
    goto LABEL_106;
  }
LABEL_54:
  v119 = v31;
  v32 = sub_AD7630(a3, 0);
  v33 = v119;
  v34 = v32;
  if ( v32 )
  {
    if ( a1 - 19 <= 1 || a1 - 22 <= 1 )
    {
      v107 = v119;
      v120 = v32;
      v35 = sub_AC30F0(v32);
      v34 = v120;
      v33 = v107;
      if ( v35 )
        return sub_ACADE0(v107);
    }
    v108 = v33;
    v121 = v34;
    v36 = sub_AD7630(a2, 0);
    v33 = v108;
    if ( v36 )
    {
      v105 = v108;
      v113 = v36;
      if ( (unsigned __int8)sub_AC47B0(a1) )
        v101 = sub_AD5570(a1, v113, v121, 0, 0);
      else
        v101 = sub_AABE40(a1, v113, v121);
      if ( v101 )
      {
        v102 = *(_DWORD *)(v105 + 32);
        BYTE4(v138) = *(_BYTE *)(v105 + 8) == 18;
        LODWORD(v138) = v102;
        return sub_AD5E10((size_t)v138);
      }
      return 0;
    }
  }
  if ( *(_BYTE *)(v33 + 8) == 17 )
  {
    v122 = v33;
    v138 = v140;
    v139 = 0x1000000000LL;
    v109 = sub_BCCE00(*(_QWORD *)v33, 32);
    if ( *(_DWORD *)(v122 + 32) )
    {
      v104 = *(unsigned int *)(v122 + 32);
      v37 = 0;
      while ( 1 )
      {
        v123 = sub_AD64C0(v109, v37, 0);
        v38 = sub_AD5840(a2, v123, 0);
        v124 = sub_AD5840(a3, v123, 0);
        v39 = v38;
        v10 = (unsigned __int8)sub_AC47B0(a1) ? sub_AD5570(a1, v38, v124, 0, 0) : sub_AABE40(a1, v38, v124);
        if ( !v10 )
          break;
        v40 = (unsigned int)v139;
        v41 = (unsigned int)v139 + 1LL;
        if ( v41 > HIDWORD(v139) )
        {
          v130 = v10;
          sub_C8D5F0(&v138, v140, v41, 8);
          v40 = (unsigned int)v139;
          v10 = v130;
        }
        ++v37;
        v138[v40] = v10;
        v39 = (unsigned int)(v139 + 1);
        LODWORD(v139) = v139 + 1;
        if ( v104 == v37 )
          goto LABEL_68;
      }
    }
    else
    {
      v39 = (unsigned int)v139;
LABEL_68:
      v10 = sub_AD3730(v138, v39);
    }
    if ( v138 != (_QWORD *)v140 )
    {
      v125 = v10;
      _libc_free(v138, v39);
      return v125;
    }
    return v10;
  }
LABEL_92:
  if ( *a2 != 5 )
  {
    v18 = *a3;
    goto LABEL_103;
  }
  if ( (((a1 - 13) & 0xFFFFFFFB) == 0 || a1 - 28 <= 2) && a1 == *((unsigned __int16 *)a2 + 1) )
  {
    v51 = sub_AD5570(a1, *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))], a3, 0, 0);
    v43 = (unsigned __int8 *)v51;
    if ( *(_BYTE *)v51 != 5 || a1 != *(unsigned __int16 *)(v51 + 2) )
    {
      v44 = *(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      return sub_AD5570(a1, v44, v43, 0, 0);
    }
  }
LABEL_96:
  if ( !(unsigned __int8)sub_BCAC40(*((_QWORD *)a2 + 1), 1) )
    return 0;
  switch ( a1 )
  {
    case 0xDu:
    case 0xFu:
      result = sub_AD5820(a2, a3);
      break;
    case 0x13u:
    case 0x14u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
      return (__int64)a2;
    case 0x16u:
    case 0x17u:
      v54 = sub_BD5C60(a2, 1, v45);
      result = sub_ACD720(v54);
      break;
    default:
      return 0;
  }
  return result;
}
