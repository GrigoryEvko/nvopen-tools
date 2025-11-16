// Function: sub_2C9B570
// Address: 0x2c9b570
//
__int64 __fastcall sub_2C9B570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  __int64 v9; // r12
  unsigned __int8 **v10; // r12
  __int64 v11; // rax
  int v12; // edx
  signed __int64 v13; // rbx
  __int64 v14; // rdx
  _BYTE *v15; // rsi
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _BYTE *v24; // rsi
  _BYTE *v25; // rsi
  __int64 v26; // rbx
  unsigned __int8 v27; // al
  unsigned int *v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // rax
  _BYTE *v31; // rsi
  __int64 v32; // r15
  __int64 *v33; // rdi
  __int64 *v34; // rax
  __int64 v35; // r9
  _BYTE *v36; // rsi
  unsigned int *v37; // rax
  unsigned __int64 v38; // r13
  __int64 *v39; // rax
  __int64 *v40; // rax
  _QWORD *v41; // rax
  __int64 *v42; // rdx
  _QWORD *v43; // r15
  unsigned __int8 *v44; // r8
  unsigned __int8 v45; // al
  __int64 v46; // rbx
  unsigned __int64 v48; // rax
  __int64 v49; // rdx
  bool v50; // zf
  __int64 v51; // r14
  __int64 v52; // rbx
  unsigned int *v53; // rax
  _BYTE *v54; // rsi
  __int64 v55; // rsi
  __int64 *v56; // r15
  signed __int64 v57; // r13
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // r13
  _BYTE *v61; // rsi
  __int64 v62; // r12
  bool v63; // r8
  unsigned int *v64; // rax
  __int64 *v65; // rax
  unsigned int *v66; // r11
  __int64 v67; // rdi
  __int64 v68; // r10
  __int64 v69; // r9
  unsigned int *v70; // rax
  __int64 *v71; // rax
  __int64 *v72; // rdx
  __int64 v73; // r8
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rsi
  __int64 v78; // rax
  int v79; // eax
  __int64 v80; // rcx
  unsigned int v81; // edx
  __int64 v82; // rax
  __int64 v86; // [rsp+28h] [rbp-1C8h]
  bool v87; // [rsp+33h] [rbp-1BDh]
  unsigned int v88; // [rsp+34h] [rbp-1BCh]
  __int64 *v90; // [rsp+40h] [rbp-1B0h]
  __int64 v91; // [rsp+40h] [rbp-1B0h]
  __int64 *v92; // [rsp+40h] [rbp-1B0h]
  __int64 v94; // [rsp+50h] [rbp-1A0h]
  unsigned __int8 **v95; // [rsp+58h] [rbp-198h]
  _QWORD *v96; // [rsp+58h] [rbp-198h]
  _QWORD *v97; // [rsp+60h] [rbp-190h]
  __int64 *v98; // [rsp+68h] [rbp-188h]
  __int64 v99; // [rsp+68h] [rbp-188h]
  _QWORD *v100; // [rsp+70h] [rbp-180h] BYREF
  unsigned int *v101; // [rsp+78h] [rbp-178h] BYREF
  unsigned __int64 v102; // [rsp+80h] [rbp-170h] BYREF
  _BYTE *v103; // [rsp+88h] [rbp-168h]
  _BYTE *v104; // [rsp+90h] [rbp-160h]
  unsigned __int64 v105; // [rsp+A0h] [rbp-150h] BYREF
  _BYTE *v106; // [rsp+A8h] [rbp-148h]
  __int64 v107; // [rsp+B0h] [rbp-140h]
  unsigned __int64 v108; // [rsp+C0h] [rbp-130h] BYREF
  _BYTE *v109; // [rsp+C8h] [rbp-128h]
  _BYTE *v110; // [rsp+D0h] [rbp-120h]
  __int64 *v111; // [rsp+E0h] [rbp-110h] BYREF
  _BYTE *v112; // [rsp+E8h] [rbp-108h]
  _BYTE *v113; // [rsp+F0h] [rbp-100h]
  const char *v114; // [rsp+100h] [rbp-F0h] BYREF
  unsigned int v115; // [rsp+108h] [rbp-E8h]
  _QWORD *v116; // [rsp+110h] [rbp-E0h] BYREF
  unsigned int v117; // [rsp+118h] [rbp-D8h]
  char v118; // [rsp+120h] [rbp-D0h]
  char v119; // [rsp+121h] [rbp-CFh]
  unsigned int *v120; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v121; // [rsp+138h] [rbp-B8h]
  _QWORD *v122; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v123; // [rsp+148h] [rbp-A8h]
  char v124; // [rsp+150h] [rbp-A0h]
  void *v125; // [rsp+1B0h] [rbp-40h]

  if ( !a2 )
    return 0;
  v87 = (*(_BYTE *)(a5 + 1) & 2) != 0;
  v94 = sub_D97090(*(_QWORD *)(a1 + 184), *(_QWORD *)(a5 + 8));
  v86 = *(_QWORD *)(a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF));
  v97 = sub_DA2C50(*(_QWORD *)(a1 + 184), v94, 0, 0);
  if ( (*(_BYTE *)(a5 + 7) & 0x40) != 0 )
    v9 = *(_QWORD *)(a5 - 8);
  else
    v9 = a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF);
  v10 = (unsigned __int8 **)(v9 + 32);
  v11 = sub_BB5290(a5);
  v12 = *(_DWORD *)(a5 + 4);
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v13 = v11 & 0xFFFFFFFFFFFFFFF9LL | 4;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v98 = (__int64 *)(a5 + 32 * (1LL - (v12 & 0x7FFFFFF)));
  if ( (__int64 *)a5 != v98 )
  {
    v95 = v10;
    v14 = 0;
    v15 = 0;
    v88 = v87 ? 4 : 0;
    while ( 1 )
    {
      v16 = *v98;
      v100 = (_QWORD *)*v98;
      if ( v15 == (_BYTE *)v14 )
      {
        sub_9281F0((__int64)&v105, v15, &v100);
      }
      else
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = v16;
          v15 = v106;
        }
        v106 = v15 + 8;
      }
      v17 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v18 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)a6 == v97 )
      {
LABEL_70:
        v120 = 0;
        sub_2C965A0((__int64)&v102, &v120);
        goto LABEL_28;
      }
      if ( !v13 )
        goto LABEL_41;
      v19 = (v13 >> 1) & 3;
      if ( ((v13 >> 1) & 3) == 0 )
        break;
      if ( v19 == 2 )
      {
        v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v17 )
          goto LABEL_42;
        goto LABEL_41;
      }
      if ( v19 != 1 || !v17 )
        goto LABEL_41;
      v14 = *(_QWORD *)(v17 + 24);
LABEL_42:
      v28 = (unsigned int *)sub_DCAD70(*(__int64 **)(a1 + 184), v94, v14);
      v29 = *(_QWORD *)(a1 + 184);
      v101 = v28;
      v30 = sub_DD8400(v29, (__int64)v100);
      v31 = v109;
      v32 = (__int64)v30;
      if ( v109 == v110 )
      {
        sub_2C95B10((__int64)&v108, v109, &v101);
      }
      else
      {
        if ( v109 )
        {
          *(_QWORD *)v109 = v101;
          v31 = v109;
        }
        v109 = v31 + 8;
      }
      v122 = sub_DC5140(*(_QWORD *)(a1 + 184), v32, v94, 0);
      v33 = *(__int64 **)(a1 + 184);
      v120 = (unsigned int *)&v122;
      v123 = (__int64)v101;
      v121 = 0x200000002LL;
      v34 = sub_DC8BD0(v33, (__int64)&v120, v88, 0);
      v35 = (__int64)v34;
      if ( v120 != (unsigned int *)&v122 )
      {
        v90 = v34;
        _libc_free((unsigned __int64)v120);
        v35 = (__int64)v90;
      }
      if ( *(_WORD *)(v35 + 24) )
      {
        sub_2C95190(&v114, v35, *(__int64 **)(a1 + 184));
        if ( sub_D968A0((__int64)v114) )
          goto LABEL_70;
        v92 = *(__int64 **)(a1 + 184);
        v39 = sub_DD8400((__int64)v92, (__int64)v100);
        sub_2C95190(&v120, (__int64)v39, v92);
        v40 = (__int64 *)v121;
        if ( !v121 )
          v40 = sub_DA2C50(*(_QWORD *)(a1 + 184), v94, 0, 0);
        v111 = v40;
        sub_2C965A0((__int64)&v102, &v111);
        v41 = sub_DC5140(*(_QWORD *)(a1 + 184), (__int64)v120, v94, 0);
        v42 = sub_DCA690(*(__int64 **)(a1 + 184), (__int64)v41, (__int64)v101, v88, 0);
        if ( v114 != (const char *)v42 )
          goto LABEL_83;
        v97 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v97, (__int64)v42, 0, 0);
      }
      else
      {
        v91 = v35;
        v97 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v97, v35, 0, 0);
        if ( sub_D968A0(v91) )
        {
          v120 = 0;
          v36 = v103;
          if ( v103 != v104 )
          {
            if ( v103 )
            {
              *(_QWORD *)v103 = 0;
              v36 = v103;
            }
LABEL_53:
            v103 = v36 + 8;
            goto LABEL_28;
          }
        }
        else
        {
          v37 = (unsigned int *)sub_DA2C50(*(_QWORD *)(a1 + 184), v94, 0, 0);
          v36 = v103;
          v120 = v37;
          if ( v103 != v104 )
          {
            if ( v103 )
            {
              *(_QWORD *)v103 = v37;
              v36 = v103;
            }
            goto LABEL_53;
          }
        }
        sub_2C96410((__int64)&v102, v36, &v120);
      }
LABEL_28:
      v98 += 4;
      if ( !v13 )
        goto LABEL_35;
      v26 = (v13 >> 1) & 3;
      if ( v26 == 2 )
      {
        if ( v17 )
          goto LABEL_31;
LABEL_35:
        v18 = sub_BCBAE0(v17, *v95, v23);
        goto LABEL_31;
      }
      if ( v26 != 1 || !v17 )
        goto LABEL_35;
      v18 = *(_QWORD *)(v17 + 24);
LABEL_31:
      v27 = *(_BYTE *)(v18 + 8);
      if ( v27 == 16 )
      {
        v13 = *(_QWORD *)(v18 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
      }
      else if ( (unsigned int)v27 - 17 > 1 )
      {
        v38 = v18 & 0xFFFFFFFFFFFFFFF9LL;
        v13 = 0;
        if ( v27 == 15 )
          v13 = v38;
      }
      else
      {
        v13 = v18 & 0xFFFFFFFFFFFFFFF9LL | 2;
      }
      v95 += 4;
      if ( (__int64 *)a5 == v98 )
        goto LABEL_75;
      v15 = v106;
      v14 = v107;
    }
    if ( v17 )
    {
      v20 = (_QWORD *)v100[3];
      if ( *((_DWORD *)v100 + 8) > 0x40u )
        v20 = (_QWORD *)*v20;
      if ( (_DWORD)v20 )
        goto LABEL_83;
      v21 = sub_DA2D20(*(_QWORD *)(a1 + 184), v94, v13 & 0xFFFFFFFFFFFFFFF8LL, 0);
      v22 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v97, (__int64)v21, 0, 0);
      v24 = v103;
      v120 = 0;
      v97 = v22;
      if ( v103 == v104 )
      {
        sub_2C96410((__int64)&v102, v103, &v120);
      }
      else
      {
        if ( v103 )
        {
          *(_QWORD *)v103 = 0;
          v24 = v103;
        }
        v103 = v24 + 8;
      }
      v120 = 0;
      v25 = v109;
      if ( v109 == v110 )
      {
        sub_2C96410((__int64)&v108, v109, &v120);
      }
      else
      {
        if ( v109 )
        {
          *(_QWORD *)v109 = 0;
          v25 = v109;
        }
        v109 = v25 + 8;
      }
      goto LABEL_28;
    }
LABEL_41:
    v14 = sub_BCBAE0(v13 & 0xFFFFFFFFFFFFFFF8LL, *v95, v14);
    goto LABEL_42;
  }
LABEL_75:
  v100 = 0;
  if ( (_QWORD *)a6 == v97 )
  {
    v43 = 0;
    goto LABEL_82;
  }
  v43 = sub_DCC810(*(__int64 **)(a1 + 184), a6, (__int64)v97, 0, 0);
  v44 = sub_BD3990((unsigned __int8 *)v86, a6);
  v45 = *v44;
  if ( *v44 <= 0x1Cu )
  {
    if ( v45 == 5 && *((_WORD *)v44 + 1) == 34 )
    {
LABEL_78:
      v46 = sub_2C9B570(a1, a2, a3, *(_QWORD *)(v86 + 8), (_DWORD)v44, (_DWORD)v43, (__int64)&v100);
      if ( v46 )
      {
        if ( v43 == v100 && sub_D968A0((__int64)v97) )
          goto LABEL_83;
        v86 = v46;
        goto LABEL_92;
      }
    }
  }
  else if ( v45 == 63 )
  {
    goto LABEL_78;
  }
  v100 = v43;
LABEL_82:
  if ( !sub_D968A0((__int64)v97) )
  {
    if ( !v86 )
      goto LABEL_84;
LABEL_92:
    v48 = v102;
    v49 = (__int64)&v103[-v102] >> 3;
    v50 = *(_BYTE *)a5 == 63;
    v111 = 0;
    v112 = 0;
    v113 = 0;
    if ( !v50 )
    {
      if ( !(_DWORD)v49 )
      {
        v57 = 0;
        v56 = 0;
LABEL_104:
        v118 = 0;
        v58 = sub_BB5290(a5);
        v124 = 0;
        v59 = v58;
        if ( v118 )
        {
          LODWORD(v121) = v115;
          if ( v115 > 0x40 )
            sub_C43780((__int64)&v120, (const void **)&v114);
          else
            v120 = (unsigned int *)v114;
          LODWORD(v123) = v117;
          if ( v117 > 0x40 )
            sub_C43780((__int64)&v122, (const void **)&v116);
          else
            v122 = v116;
          v124 = 1;
        }
        v86 = sub_AD9FD0(v59, (unsigned __int8 *)v86, v56, v57, v87 ? 3 : 0, (__int64)&v120, 0);
        if ( v124 )
        {
          v124 = 0;
          if ( (unsigned int)v123 > 0x40 && v122 )
            j_j___libc_free_0_0((unsigned __int64)v122);
          if ( (unsigned int)v121 > 0x40 && v120 )
            j_j___libc_free_0_0((unsigned __int64)v120);
        }
        if ( v118 )
        {
          v118 = 0;
          if ( v117 > 0x40 && v116 )
            j_j___libc_free_0_0((unsigned __int64)v116);
          if ( v115 > 0x40 && v114 )
            j_j___libc_free_0_0((unsigned __int64)v114);
        }
        *a7 = v100;
        if ( a4 != *(_QWORD *)(v86 + 8) )
          v86 = sub_AD4C90(v86, (__int64 **)a4, 0);
        goto LABEL_109;
      }
      v51 = 0;
      v52 = 8LL * (unsigned int)(v49 - 1);
      while ( 1 )
      {
        v55 = *(_QWORD *)(v48 + v51);
        if ( v55 )
        {
          v53 = (unsigned int *)sub_F8DB90(a3, v55, 0, a2 + 24, 0);
          v54 = v112;
          v120 = v53;
          if ( v112 == v113 )
            goto LABEL_102;
        }
        else
        {
          v54 = v112;
          v53 = *(unsigned int **)(v105 + v51);
          v120 = v53;
          if ( v112 == v113 )
          {
LABEL_102:
            sub_262AD50((__int64)&v111, v54, &v120);
            if ( v52 == v51 )
              goto LABEL_103;
            goto LABEL_99;
          }
        }
        if ( v54 )
        {
          *(_QWORD *)v54 = v53;
          v54 = v112;
        }
        v112 = v54 + 8;
        if ( v52 == v51 )
        {
LABEL_103:
          v56 = v111;
          v57 = (v112 - (_BYTE *)v111) >> 3;
          goto LABEL_104;
        }
LABEL_99:
        v48 = v102;
        v51 += 8;
      }
    }
    if ( !(_DWORD)v49 )
    {
      v96 = 0;
LABEL_174:
      if ( v96 == v97 && v100 == v43 )
        goto LABEL_170;
      sub_23D0AB0((__int64)&v120, a5, 0, 0, 0);
      v77 = *(_QWORD *)(a5 + 72);
      v119 = 1;
      v114 = "newGep";
      v118 = 3;
      v86 = sub_921130(&v120, v77, v86, (_BYTE **)v111, (v112 - (_BYTE *)v111) >> 3, (__int64)&v114, 0);
      if ( v96 )
      {
        if ( v100 )
          *a7 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v100, (__int64)v96, 0, 0);
        else
          *a7 = v96;
      }
      else
      {
        *a7 = v100;
      }
      v78 = *(_QWORD *)(v86 + 8);
      if ( a4 != v78 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(v78 + 8) - 17 <= 1 )
          v78 = **(_QWORD **)(v78 + 16);
        v79 = *(_DWORD *)(v78 + 8) >> 8;
        v80 = a4;
        if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
          v80 = **(_QWORD **)(a4 + 16);
        v81 = *(_DWORD *)(v80 + 8);
        v119 = 1;
        v118 = 3;
        v114 = "newBit";
        if ( v81 >> 8 == v79 )
          v82 = sub_2C91010((__int64 *)&v120, 49, v86, a4, (__int64)&v114, 0, (int)v101, 0);
        else
          v82 = sub_2C91010((__int64 *)&v120, 50, v86, a4, (__int64)&v114, 0, (int)v101, 0);
        v86 = v82;
      }
      nullsub_61();
      v125 = &unk_49DA100;
      nullsub_63();
      if ( v120 != (unsigned int *)&v122 )
        _libc_free((unsigned __int64)v120);
LABEL_109:
      if ( v111 )
        j_j___libc_free_0((unsigned __int64)v111);
      goto LABEL_84;
    }
    v60 = 0;
    v96 = 0;
    v99 = 8LL * (unsigned int)(v49 - 1);
    while ( 1 )
    {
      v62 = *(_QWORD *)(v48 + v60);
      if ( !v62 )
        break;
      v101 = 0;
      v63 = sub_D968A0(v62);
      v64 = v101;
      if ( !v63 )
      {
        v65 = sub_DD8400(*(_QWORD *)(a1 + 184), *(_QWORD *)(v105 + v60));
        v66 = (unsigned int *)v65;
        if ( *((_WORD *)v65 + 12) )
        {
          sub_2C95190(&v120, (__int64)v65, *(__int64 **)(a1 + 184));
          v66 = v120;
        }
        v67 = *(_QWORD *)(v60 + v105);
        if ( *(_BYTE *)v67 > 0x1Cu )
        {
          v68 = *(_QWORD *)(v67 + 40);
          if ( *(_QWORD *)(a2 + 40) != v68 )
          {
            v69 = *(_QWORD *)(v67 + 16);
            if ( v69 )
            {
              if ( *(_QWORD *)(v69 + 8) )
              {
                do
                {
                  v73 = *(_QWORD *)(v69 + 24);
                  if ( *(_BYTE *)v73 <= 0x1Cu )
                    BUG();
                  if ( *(_BYTE *)v73 == 84 )
                  {
                    if ( (*(_DWORD *)(v73 + 4) & 0x7FFFFFF) != 0 )
                    {
                      v74 = *(_QWORD *)(v73 - 8);
                      v75 = 0;
                      while ( 1 )
                      {
                        v76 = *(_QWORD *)(v74 + 4 * v75);
                        if ( v76 )
                        {
                          if ( v67 == v76 && v68 != *(_QWORD *)(v74 + 32LL * *(unsigned int *)(v73 + 72) + v75) )
                            break;
                        }
                        v75 += 8;
                        if ( v75 == 8LL * (*(_DWORD *)(v73 + 4) & 0x7FFFFFF) )
                          goto LABEL_172;
                      }
LABEL_170:
                      v86 = 0;
                      goto LABEL_109;
                    }
                  }
                  else if ( v68 != *(_QWORD *)(v73 + 40) )
                  {
                    goto LABEL_170;
                  }
LABEL_172:
                  v69 = *(_QWORD *)(v69 + 8);
                }
                while ( v69 );
              }
            }
          }
        }
        v114 = 0;
        v70 = *(unsigned int **)(v60 + v105);
        v121 = (__int64)v66;
        v120 = v70;
        v64 = (unsigned int *)sub_2C9AA80(a1, (__int64 *)&v120, (__int64 *)&v114);
        v101 = v64;
        if ( !v64 )
          goto LABEL_170;
        if ( v114 )
        {
          v71 = sub_DCA690(*(__int64 **)(a1 + 184), *(_QWORD *)(v108 + v60), (__int64)v114, 0, 0);
          if ( v96 )
            v96 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v96, (__int64)v71, 0, 0);
          else
            v96 = v71;
          v64 = v101;
        }
      }
      if ( v64 )
      {
        v61 = v112;
        if ( v112 != v113 )
          goto LABEL_136;
        v72 = (__int64 *)&v101;
LABEL_159:
        sub_9281F0((__int64)&v111, v61, v72);
        goto LABEL_139;
      }
      v64 = (unsigned int *)sub_F8DB90(a3, v62, 0, a5 + 24, 0);
      v61 = v112;
      v120 = v64;
      if ( v112 != v113 )
      {
LABEL_136:
        if ( v61 )
        {
          *(_QWORD *)v61 = v64;
          v61 = v112;
        }
LABEL_138:
        v112 = v61 + 8;
        goto LABEL_139;
      }
      sub_928380((__int64)&v111, v112, &v120);
LABEL_139:
      if ( v99 == v60 )
        goto LABEL_174;
      v48 = v102;
      v60 += 8;
    }
    v61 = v112;
    v72 = (__int64 *)(v60 + v105);
    if ( v112 != v113 )
    {
      if ( v112 )
      {
        *(_QWORD *)v112 = *v72;
        v61 = v112;
      }
      goto LABEL_138;
    }
    goto LABEL_159;
  }
LABEL_83:
  v86 = 0;
LABEL_84:
  if ( v108 )
    j_j___libc_free_0(v108);
  if ( v105 )
    j_j___libc_free_0(v105);
  if ( v102 )
    j_j___libc_free_0(v102);
  return v86;
}
