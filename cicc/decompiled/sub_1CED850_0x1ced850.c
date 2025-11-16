// Function: sub_1CED850
// Address: 0x1ced850
//
void __fastcall sub_1CED850(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r15
  unsigned __int8 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  unsigned __int8 v8; // cl
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  unsigned __int8 *v14; // rsi
  __int64 v15; // r12
  __int64 **v16; // rdx
  unsigned __int8 *v17; // rbx
  __int64 v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  unsigned __int64 *v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  _BYTE *v26; // rsi
  _QWORD *v27; // rdx
  unsigned __int8 *v28; // r12
  bool v29; // bl
  __int64 v30; // rdx
  char v31; // al
  __int64 *v32; // rax
  unsigned __int8 *v33; // rsi
  __int64 v34; // r14
  _QWORD *v35; // rax
  unsigned __int8 *v36; // rsi
  __int64 **v37; // rax
  _QWORD *v38; // rbx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 **v42; // rsi
  __int64 v43; // rdx
  _QWORD *v44; // rax
  _QWORD *v45; // r12
  unsigned __int64 *v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rsi
  unsigned __int8 *v50; // rsi
  _QWORD *v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // rdx
  _QWORD *v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rbx
  __int64 v61; // r12
  unsigned __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rdx
  unsigned __int8 *v66; // rsi
  __int64 *v67; // rbx
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  unsigned int v72; // r13d
  _QWORD *v73; // rax
  __int64 *v74; // rax
  unsigned __int64 *v75; // r12
  __int64 v76; // rax
  unsigned __int64 v77; // rcx
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  __int64 *v80; // r12
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rsi
  unsigned __int8 *v84; // rsi
  char v85; // al
  __int64 v86; // [rsp+8h] [rbp-148h]
  __int64 v87; // [rsp+20h] [rbp-130h]
  __int64 v88; // [rsp+20h] [rbp-130h]
  __int64 v89; // [rsp+28h] [rbp-128h]
  __int64 *v90; // [rsp+28h] [rbp-128h]
  __int64 v91; // [rsp+28h] [rbp-128h]
  __int64 v92; // [rsp+28h] [rbp-128h]
  unsigned __int64 *v93; // [rsp+28h] [rbp-128h]
  __int64 *v94; // [rsp+30h] [rbp-120h] BYREF
  unsigned __int8 *v95; // [rsp+38h] [rbp-118h] BYREF
  _QWORD *v96; // [rsp+40h] [rbp-110h] BYREF
  unsigned __int8 *v97; // [rsp+48h] [rbp-108h] BYREF
  __int64 v98; // [rsp+50h] [rbp-100h] BYREF
  _BYTE *v99; // [rsp+58h] [rbp-F8h]
  _BYTE *v100; // [rsp+60h] [rbp-F0h]
  __int64 v101; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v102; // [rsp+78h] [rbp-D8h]
  _BYTE *v103; // [rsp+80h] [rbp-D0h]
  __int64 v104[2]; // [rsp+90h] [rbp-C0h] BYREF
  char v105; // [rsp+A0h] [rbp-B0h]
  char v106; // [rsp+A1h] [rbp-AFh]
  unsigned __int8 *v107[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v108; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v109; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v110; // [rsp+D8h] [rbp-78h]
  unsigned __int64 *v111; // [rsp+E0h] [rbp-70h]
  _QWORD *v112; // [rsp+E8h] [rbp-68h]
  __int64 v113; // [rsp+F0h] [rbp-60h]
  int v114; // [rsp+F8h] [rbp-58h]
  __int64 v115; // [rsp+100h] [rbp-50h]
  __int64 v116; // [rsp+108h] [rbp-48h]

  v1 = *(_QWORD *)(a1 + 80);
  v98 = 0;
  v99 = 0;
  v100 = 0;
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)(v1 + 24);
  v3 = v1 + 16;
  if ( v2 != v3 )
  {
    do
    {
      if ( !v2 )
        BUG();
      if ( *(_BYTE *)(v2 - 8) == 53 )
      {
        v4 = *(_QWORD *)(v2 - 16);
        if ( v4 )
        {
          while ( 1 )
          {
            v5 = (unsigned __int8 *)sub_1648700(v4);
            if ( v5[16] != 55 )
              goto LABEL_7;
            v109 = v5;
            v6 = *((_QWORD *)v5 - 3);
            if ( !v6 )
              goto LABEL_7;
            if ( v6 != v2 - 24 )
              goto LABEL_7;
            v7 = *((_QWORD *)v5 - 6);
            if ( !v7 )
              goto LABEL_7;
            v8 = *(_BYTE *)(v7 + 16);
            if ( v8 > 0x10u )
              break;
            v9 = *(_QWORD *)(v2 - 16);
            if ( v9 && !*(_QWORD *)(v9 + 8) )
              goto LABEL_7;
            v10 = v99;
            if ( v99 == v100 )
            {
LABEL_133:
              sub_190D490((__int64)&v98, v10, &v109);
LABEL_7:
              v4 = *(_QWORD *)(v4 + 8);
              if ( !v4 )
                goto LABEL_19;
            }
            else
            {
              if ( v99 )
                goto LABEL_17;
LABEL_18:
              v99 = v10 + 8;
              v4 = *(_QWORD *)(v4 + 8);
              if ( !v4 )
                goto LABEL_19;
            }
          }
          if ( v8 > 0x17u )
          {
            v52 = *((_QWORD *)v5 - 6);
            while ( 1 )
            {
              if ( v8 == 53 )
                goto LABEL_104;
              if ( v8 == 71 )
              {
                v52 = *(_QWORD *)(v52 - 24);
              }
              else
              {
                if ( v8 != 56 || (v88 = v7, v85 = sub_15FA1F0(v52), v7 = v88, !v85) )
                {
LABEL_98:
                  v8 = *(_BYTE *)(v7 + 16);
                  break;
                }
                v52 = *(_QWORD *)(v52 - 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF));
              }
              v8 = *(_BYTE *)(v52 + 16);
              if ( v8 <= 0x17u )
                goto LABEL_98;
            }
          }
          if ( v8 != 54 )
          {
            v53 = *(_QWORD *)(v7 + 8);
            if ( !v53 )
              goto LABEL_7;
            if ( *(_QWORD *)(v53 + 8) )
              goto LABEL_7;
            v54 = *(_QWORD *)(v2 - 16);
            if ( !v54 || *(_QWORD *)(v54 + 8) )
              goto LABEL_7;
          }
LABEL_104:
          v10 = v99;
          if ( v99 != v100 )
          {
            if ( !v99 )
              goto LABEL_18;
            v5 = v109;
LABEL_17:
            *(_QWORD *)v10 = v5;
            v10 = v99;
            goto LABEL_18;
          }
          goto LABEL_133;
        }
      }
LABEL_19:
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v3 != v2 );
    v11 = v98;
    v101 = 0;
    v102 = 0;
    v103 = 0;
    v12 = (__int64)&v99[-v98] >> 3;
    if ( !(_DWORD)v12 )
      goto LABEL_116;
    v13 = 0;
    v86 = 8LL * (unsigned int)(v12 - 1);
    while ( 1 )
    {
      v94 = *(__int64 **)(*(_QWORD *)(v11 + v13) - 48LL);
      v28 = (unsigned __int8 *)*v94;
      if ( (unsigned __int8)(*(_BYTE *)(*v94 + 8) - 2) > 1u
        && !sub_1642F90(*v94, 8)
        && !sub_1642F90((__int64)v28, 16)
        && !sub_1642F90((__int64)v28, 32)
        && !sub_1642F90((__int64)v28, 64)
        && v28[8] != 15 )
      {
        goto LABEL_51;
      }
      v109 = v28;
      v29 = sub_1642F90((__int64)v28, 8);
      if ( v29 )
      {
        v29 = 0;
        v51 = (_QWORD *)sub_15E0530(a1);
        v90 = 0;
        v109 = (unsigned __int8 *)sub_1643340(v51);
      }
      else
      {
        v90 = 0;
        if ( v28[8] == 15 )
        {
          v30 = *((_QWORD *)v28 + 3);
          v31 = *(_BYTE *)(v30 + 8);
          if ( v31 == 16 )
            v31 = *(_BYTE *)(**(_QWORD **)(v30 + 16) + 8LL);
          if ( v31 != 11 )
          {
            v90 = 0;
            if ( (unsigned __int8)(v31 - 1) > 5u )
            {
              v72 = *((_DWORD *)v28 + 2);
              v29 = 1;
              v73 = (_QWORD *)sub_15E0530(a1);
              v74 = (__int64 *)sub_1643350(v73);
              v90 = (__int64 *)sub_1646BA0(v74, v72 >> 8);
              v109 = (unsigned __int8 *)v90;
            }
          }
        }
      }
      v87 = sub_15E26F0(*(__int64 **)(a1 + 40), 4179, (__int64 *)&v109, 1);
      v32 = (__int64 *)(v13 + v98);
      v33 = *(unsigned __int8 **)(*(_QWORD *)(v13 + v98) + 48LL);
      v95 = v33;
      if ( v33 )
      {
        sub_1623A60((__int64)&v95, (__int64)v33, 2);
        v32 = (__int64 *)(v13 + v98);
      }
      v34 = *v32;
      v35 = (_QWORD *)sub_16498A0(*v32);
      v109 = 0;
      v112 = v35;
      v113 = 0;
      v114 = 0;
      v115 = 0;
      v116 = 0;
      v110 = *(_QWORD *)(v34 + 40);
      v111 = (unsigned __int64 *)(v34 + 24);
      v36 = *(unsigned __int8 **)(v34 + 48);
      v107[0] = v36;
      if ( v36 )
      {
        sub_1623A60((__int64)v107, (__int64)v36, 2);
        if ( v109 )
          sub_161E7C0((__int64)&v109, (__int64)v109);
        v109 = v107[0];
        if ( v107[0] )
          sub_1623210((__int64)v107, v107[0], (__int64)&v109);
      }
      v107[0] = v95;
      if ( v95 )
        break;
      v14 = v109;
      if ( v109 )
        goto LABEL_23;
LABEL_26:
      if ( sub_1642F90((__int64)v28, 8) )
      {
        v106 = 1;
        v104[0] = (__int64)"cast";
        v105 = 3;
        v37 = (__int64 **)sub_1643340(v112);
        v38 = v94;
        if ( v37 != (__int64 **)*v94 )
        {
          if ( *((_BYTE *)v94 + 16) > 0x10u )
          {
            v108 = 257;
            v38 = (_QWORD *)sub_15FDED0(v94, (__int64)v37, (__int64)v107, 0);
            if ( v110 )
            {
              v75 = v111;
              sub_157E9D0(v110 + 40, (__int64)v38);
              v76 = v38[3];
              v77 = *v75;
              v38[4] = v75;
              v77 &= 0xFFFFFFFFFFFFFFF8LL;
              v38[3] = v77 | v76 & 7;
              *(_QWORD *)(v77 + 8) = v38 + 3;
              *v75 = *v75 & 7 | (unsigned __int64)(v38 + 3);
            }
            sub_164B780((__int64)v38, v104);
            if ( v109 )
            {
              v97 = v109;
              sub_1623A60((__int64)&v97, (__int64)v109, 2);
              v78 = v38[6];
              if ( v78 )
                sub_161E7C0((__int64)(v38 + 6), v78);
              v79 = v97;
              v38[6] = v97;
              if ( v79 )
                sub_1623210((__int64)&v97, v79, (__int64)(v38 + 6));
            }
          }
          else
          {
            v38 = (_QWORD *)sub_15A4620((__int64 ***)v94, v37);
          }
        }
        v107[0] = (unsigned __int8 *)"move";
        v108 = 259;
        v39 = *(_QWORD *)(v87 + 24);
        v96 = v38;
        v40 = sub_1285290((__int64 *)&v109, v39, v87, (int)&v96, 1, (__int64)v107, 0);
        v106 = 1;
        v41 = v40;
        v105 = 3;
        v104[0] = (__int64)"cast";
        v42 = (__int64 **)sub_1643330(v112);
        if ( v42 != *(__int64 ***)v41 )
        {
          if ( *(_BYTE *)(v41 + 16) > 0x10u )
          {
            v108 = 257;
            v41 = sub_15FDF30((_QWORD *)v41, (__int64)v42, (__int64)v107, 0);
            if ( v110 )
            {
              v80 = (__int64 *)v111;
              sub_157E9D0(v110 + 40, v41);
              v81 = *(_QWORD *)(v41 + 24);
              v82 = *v80;
              *(_QWORD *)(v41 + 32) = v80;
              v82 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v41 + 24) = v82 | v81 & 7;
              *(_QWORD *)(v82 + 8) = v41 + 24;
              *v80 = *v80 & 7 | (v41 + 24);
            }
            sub_164B780(v41, v104);
            if ( v109 )
            {
              v97 = v109;
              sub_1623A60((__int64)&v97, (__int64)v109, 2);
              v83 = *(_QWORD *)(v41 + 48);
              if ( v83 )
                sub_161E7C0(v41 + 48, v83);
              v84 = v97;
              *(_QWORD *)(v41 + 48) = v97;
              if ( v84 )
                sub_1623210((__int64)&v97, v84, v41 + 48);
            }
          }
          else
          {
            v41 = sub_15A4670((__int64 ***)v41, v42);
          }
        }
        v43 = *(_QWORD *)(*(_QWORD *)(v98 + v13) - 24LL);
        v108 = 257;
        v91 = v43;
        v44 = sub_1648A60(64, 2u);
        v45 = v44;
        if ( v44 )
          sub_15F9650((__int64)v44, v41, v91, 0, 0);
        if ( v110 )
        {
          v46 = v111;
          sub_157E9D0(v110 + 40, (__int64)v45);
          v47 = v45[3];
          v48 = *v46;
          v45[4] = v46;
          v48 &= 0xFFFFFFFFFFFFFFF8LL;
          v45[3] = v48 | v47 & 7;
          *(_QWORD *)(v48 + 8) = v45 + 3;
          *v46 = *v46 & 7 | (unsigned __int64)(v45 + 3);
        }
        sub_164B780((__int64)v45, (__int64 *)v107);
        if ( v109 )
        {
          v104[0] = (__int64)v109;
          sub_1623A60((__int64)v104, (__int64)v109, 2);
          v49 = v45[6];
          if ( v49 )
            sub_161E7C0((__int64)(v45 + 6), v49);
          v50 = (unsigned __int8 *)v104[0];
          v45[6] = v104[0];
          if ( v50 )
            sub_1623210((__int64)v104, v50, (__int64)(v45 + 6));
        }
      }
      else
      {
        if ( v29 )
        {
          v15 = *(_QWORD *)(*(_QWORD *)(v98 + v13) - 24LL);
          v106 = 1;
          v104[0] = (__int64)"bit_cast_addr";
          v105 = 3;
          v16 = (__int64 **)sub_1646BA0(v90, 0);
          if ( v16 != *(__int64 ***)v15 )
          {
            if ( *(_BYTE *)(v15 + 16) > 0x10u )
            {
              v108 = 257;
              v15 = sub_15FDBD0(47, v15, (__int64)v16, (__int64)v107, 0);
              if ( v110 )
              {
                v67 = (__int64 *)v111;
                sub_157E9D0(v110 + 40, v15);
                v68 = *(_QWORD *)(v15 + 24);
                v69 = *v67;
                *(_QWORD *)(v15 + 32) = v67;
                v69 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v15 + 24) = v69 | v68 & 7;
                *(_QWORD *)(v69 + 8) = v15 + 24;
                *v67 = *v67 & 7 | (v15 + 24);
              }
              sub_164B780(v15, v104);
              if ( v109 )
              {
                v97 = v109;
                sub_1623A60((__int64)&v97, (__int64)v109, 2);
                v70 = *(_QWORD *)(v15 + 48);
                if ( v70 )
                  sub_161E7C0(v15 + 48, v70);
                v71 = v97;
                *(_QWORD *)(v15 + 48) = v97;
                if ( v71 )
                  sub_1623210((__int64)&v97, v71, v15 + 48);
              }
            }
            else
            {
              v15 = sub_15A46C0(47, (__int64 ***)v15, v16, 0);
            }
          }
          v17 = (unsigned __int8 *)v94;
          v106 = 1;
          v104[0] = (__int64)"bit_cast_pointer";
          v105 = 3;
          if ( v90 != (__int64 *)*v94 )
          {
            if ( *((_BYTE *)v94 + 16) > 0x10u )
            {
              v108 = 257;
              v17 = (unsigned __int8 *)sub_15FDBD0(47, (__int64)v94, (__int64)v90, (__int64)v107, 0);
              if ( v110 )
              {
                v93 = v111;
                sub_157E9D0(v110 + 40, (__int64)v17);
                v62 = *v93;
                v63 = *((_QWORD *)v17 + 3) & 7LL;
                *((_QWORD *)v17 + 4) = v93;
                v62 &= 0xFFFFFFFFFFFFFFF8LL;
                *((_QWORD *)v17 + 3) = v62 | v63;
                *(_QWORD *)(v62 + 8) = v17 + 24;
                *v93 = *v93 & 7 | (unsigned __int64)(v17 + 24);
              }
              sub_164B780((__int64)v17, v104);
              if ( v109 )
              {
                v97 = v109;
                sub_1623A60((__int64)&v97, (__int64)v109, 2);
                v64 = *((_QWORD *)v17 + 6);
                v65 = (__int64)(v17 + 48);
                if ( v64 )
                {
                  sub_161E7C0((__int64)(v17 + 48), v64);
                  v65 = (__int64)(v17 + 48);
                }
                v66 = v97;
                *((_QWORD *)v17 + 6) = v97;
                if ( v66 )
                  sub_1623210((__int64)&v97, v66, v65);
              }
            }
            else
            {
              v17 = (unsigned __int8 *)sub_15A46C0(47, (__int64 ***)v94, (__int64 **)v90, 0);
            }
          }
          v97 = v17;
          v107[0] = (unsigned __int8 *)"move";
          v108 = 259;
          v18 = sub_1285290((__int64 *)&v109, *(_QWORD *)(v87 + 24), v87, (int)&v97, 1, (__int64)v107, 0);
          v108 = 257;
          v89 = v18;
          v19 = sub_1648A60(64, 2u);
          v20 = v19;
          if ( v19 )
            sub_15F9650((__int64)v19, v89, v15, 0, 0);
        }
        else
        {
          v107[0] = (unsigned __int8 *)"move";
          v108 = 259;
          v55 = sub_1285290((__int64 *)&v109, *(_QWORD *)(v87 + 24), v87, (int)&v94, 1, (__int64)v107, 0);
          v56 = *(_QWORD *)(*(_QWORD *)(v98 + v13) - 24LL);
          v108 = 257;
          v92 = v56;
          v57 = sub_1648A60(64, 2u);
          v20 = v57;
          if ( v57 )
            sub_15F9650((__int64)v57, v55, v92, 0, 0);
        }
        if ( v110 )
        {
          v21 = v111;
          sub_157E9D0(v110 + 40, (__int64)v20);
          v22 = v20[3];
          v23 = *v21;
          v20[4] = v21;
          v23 &= 0xFFFFFFFFFFFFFFF8LL;
          v20[3] = v23 | v22 & 7;
          *(_QWORD *)(v23 + 8) = v20 + 3;
          *v21 = *v21 & 7 | (unsigned __int64)(v20 + 3);
        }
        sub_164B780((__int64)v20, (__int64 *)v107);
        if ( v109 )
        {
          v104[0] = (__int64)v109;
          sub_1623A60((__int64)v104, (__int64)v109, 2);
          v24 = v20[6];
          if ( v24 )
            sub_161E7C0((__int64)(v20 + 6), v24);
          v25 = (unsigned __int8 *)v104[0];
          v20[6] = v104[0];
          if ( v25 )
            sub_1623210((__int64)v104, v25, (__int64)(v20 + 6));
        }
      }
      v26 = v102;
      v27 = (_QWORD *)(v13 + v98);
      if ( v102 == v103 )
      {
        sub_190D490((__int64)&v101, v102, v27);
      }
      else
      {
        if ( v102 )
        {
          *(_QWORD *)v102 = *v27;
          v26 = v102;
        }
        v102 = v26 + 8;
      }
      if ( v109 )
        sub_161E7C0((__int64)&v109, (__int64)v109);
      if ( v95 )
        sub_161E7C0((__int64)&v95, (__int64)v95);
LABEL_51:
      if ( v86 == v13 )
      {
        v58 = v101;
        v59 = (__int64)&v102[-v101] >> 3;
        if ( (_DWORD)v59 )
        {
          v60 = 0;
          v61 = 8LL * (unsigned int)(v59 - 1);
          while ( 1 )
          {
            sub_15F20C0(*(_QWORD **)(v58 + v60));
            v58 = v101;
            if ( v60 == v61 )
              break;
            v60 += 8;
          }
        }
        if ( v58 )
          j_j___libc_free_0(v58, &v103[-v58]);
        goto LABEL_116;
      }
      v11 = v98;
      v13 += 8;
    }
    sub_1623A60((__int64)v107, (__int64)v95, 2);
    v14 = v109;
    if ( v109 )
LABEL_23:
      sub_161E7C0((__int64)&v109, (__int64)v14);
    v109 = v107[0];
    if ( v107[0] )
      sub_1623210((__int64)v107, v107[0], (__int64)&v109);
    goto LABEL_26;
  }
LABEL_116:
  if ( v98 )
    j_j___libc_free_0(v98, &v100[-v98]);
}
