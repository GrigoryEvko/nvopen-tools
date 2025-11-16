// Function: sub_2D62430
// Address: 0x2d62430
//
__int64 __fastcall sub_2D62430(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned int v2; // r12d
  int v4; // esi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // r15
  __int64 (*v12)(); // r14
  _QWORD *v14; // rax
  unsigned __int8 *v15; // r14
  unsigned __int8 v16; // al
  int v17; // r13d
  __int64 v18; // rax
  unsigned __int8 **v19; // rcx
  unsigned __int8 **v20; // r13
  unsigned __int8 *v21; // rax
  int v22; // eax
  unsigned __int8 *v23; // rdx
  int v24; // eax
  unsigned int v25; // eax
  bool v26; // al
  unsigned int v27; // eax
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned __int16 v30; // ax
  unsigned __int8 *v31; // r12
  unsigned __int8 *v32; // rax
  __int64 v33; // r13
  unsigned __int8 *v34; // rbx
  unsigned __int8 *v35; // r15
  unsigned __int8 *v36; // rax
  int v37; // esi
  bool v38; // r14
  unsigned int v39; // eax
  unsigned int v40; // r10d
  char v41; // dl
  unsigned __int8 *v42; // rax
  __int64 v43; // rax
  char v44; // dl
  int v45; // r13d
  __int64 v46; // rax
  __int64 v47; // r9
  unsigned int v48; // r10d
  int v49; // r14d
  __int64 v50; // r12
  __int64 v51; // rsi
  __int64 v52; // rbx
  unsigned int v53; // r15d
  unsigned __int64 v54; // r8
  __int64 v55; // r15
  int v56; // eax
  unsigned __int8 *v57; // rdx
  unsigned __int8 *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdx
  unsigned __int8 *v61; // rdi
  unsigned __int8 *v62; // rsi
  __int64 v63; // rcx
  unsigned int *v64; // rsi
  __int64 *v65; // r8
  __int64 v66; // rdx
  __int64 v67; // rax
  bool v68; // zf
  unsigned __int8 *v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // rax
  unsigned int v72; // r15d
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  signed __int64 v76; // r15
  signed __int64 v77; // r14
  int v78; // edx
  int v79; // r12d
  unsigned __int8 **v80; // r13
  int v81; // ebx
  unsigned __int8 *v82; // rcx
  int v83; // edx
  __int64 v84; // r9
  __int64 v85; // r8
  __int64 v86; // rax
  int v87; // edx
  bool v88; // of
  __int64 v89; // rax
  int v90; // edx
  unsigned __int8 *v91; // rax
  int v92; // r10d
  bool v93; // al
  __int64 *v94; // [rsp+8h] [rbp-108h]
  __int64 v95; // [rsp+10h] [rbp-100h]
  __int64 *v96; // [rsp+18h] [rbp-F8h]
  __int64 v97; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v98; // [rsp+20h] [rbp-F0h]
  unsigned int v99; // [rsp+20h] [rbp-F0h]
  unsigned int v100; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v101; // [rsp+28h] [rbp-E8h]
  __int64 (*v102)(); // [rsp+28h] [rbp-E8h]
  unsigned __int8 v103; // [rsp+28h] [rbp-E8h]
  unsigned __int8 **v104; // [rsp+30h] [rbp-E0h]
  unsigned int v105; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v106; // [rsp+30h] [rbp-E0h]
  __int64 v107; // [rsp+30h] [rbp-E0h]
  unsigned __int8 **v108; // [rsp+30h] [rbp-E0h]
  __int64 v109; // [rsp+38h] [rbp-D8h]
  int v110; // [rsp+44h] [rbp-CCh] BYREF
  __int64 v111; // [rsp+48h] [rbp-C8h]
  __int64 *v112; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v113; // [rsp+58h] [rbp-B8h]
  _BYTE v114[32]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v115; // [rsp+80h] [rbp-90h]
  __int64 v116; // [rsp+88h] [rbp-88h]
  __int64 v117; // [rsp+90h] [rbp-80h]
  unsigned __int8 *v118; // [rsp+98h] [rbp-78h]
  __int64 *v119; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v120; // [rsp+A8h] [rbp-68h]
  _BYTE v121[32]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v122; // [rsp+D0h] [rbp-40h]
  unsigned __int8 *v123; // [rsp+D8h] [rbp-38h]

  v2 = (unsigned __int8)byte_5017F08;
  v110 = -1;
  if ( byte_5017F08 )
    return 0;
  if ( (_BYTE)qword_5017E28 )
  {
    v4 = -1;
    goto LABEL_4;
  }
  v11 = a1[2];
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 488LL);
  if ( v12 == sub_2D565F0 )
    return 0;
  v14 = (_QWORD *)sub_986520((__int64)a2);
  if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, int *))v12)(
          v11,
          *(_QWORD *)(*v14 + 8LL),
          v14[4],
          &v110) )
    return 0;
  v4 = v110;
LABEL_4:
  v5 = a1[4];
  v6 = a1[2];
  v118 = a2;
  v7 = a1[102];
  v8 = *((_QWORD *)a2 + 5);
  v122 = v4;
  v117 = v5;
  v120 = 0x400000000LL;
  v9 = *((_QWORD *)a2 + 2);
  v115 = v7;
  v116 = v6;
  v119 = (__int64 *)v121;
  v123 = 0;
  if ( !v9 )
    return v2;
LABEL_5:
  if ( *(_QWORD *)(v9 + 8) )
    goto LABEL_6;
  v15 = *(unsigned __int8 **)(v9 + 24);
  if ( v8 != *((_QWORD *)v15 + 5) )
    goto LABEL_6;
  v16 = *v15;
  if ( *v15 != 62 )
  {
    v17 = v16;
    if ( (unsigned int)v16 - 42 > 0x11 )
      goto LABEL_6;
    v18 = sub_986520((__int64)v15);
    v19 = (unsigned __int8 **)(v18 + 32LL * (*((_DWORD *)v15 + 1) & 0x7FFFFFF));
    if ( (unsigned __int8 **)v18 == v19 )
      goto LABEL_37;
    v20 = (unsigned __int8 **)v18;
    while ( 1 )
    {
      v23 = *v20;
      if ( (_DWORD)v120 )
        v21 = (unsigned __int8 *)v119[(unsigned int)v120 - 1];
      else
        v21 = v118;
      if ( v23 != v21 )
      {
        v22 = *v23;
        if ( (unsigned __int8)(v22 - 17) > 1u && (unsigned int)(v22 - 12) > 1 )
          goto LABEL_6;
        goto LABEL_23;
      }
      v104 = v19;
      v24 = sub_BD2910((__int64)v20);
      v19 = v104;
      if ( v24 == 1 )
      {
        v25 = *v15 - 29;
        if ( v25 <= 0x17 )
        {
          if ( v25 > 0x15 )
            goto LABEL_6;
          if ( *v15 != 50 )
          {
            if ( v25 > 0x12 )
              goto LABEL_6;
            goto LABEL_23;
          }
LABEL_30:
          v26 = sub_B451C0((__int64)v15);
          v19 = v104;
          if ( !v26 )
            goto LABEL_6;
          goto LABEL_23;
        }
        if ( *v15 == 53 )
          goto LABEL_30;
      }
LABEL_23:
      v20 += 4;
      if ( v19 == v20 )
      {
        v17 = *v15;
LABEL_37:
        v27 = sub_2FEBEF0(v116, (unsigned int)(v17 - 29));
        if ( !v27 )
          goto LABEL_6;
        if ( !(_BYTE)qword_5017E28 )
        {
          v105 = v27;
          v28 = v116;
          v29 = sub_986520((__int64)v118);
          v30 = sub_2D5BAE0(v28, v115, *(__int64 **)(*(_QWORD *)v29 + 8LL), 1);
          if ( v30 != 1 && (!v30 || !*(_QWORD *)(v28 + 8LL * v30 + 112)) )
            goto LABEL_6;
          if ( v105 <= 0x1F3 && (*(_BYTE *)(v105 + 500LL * v30 + v28 + 6414) & 0xFB) != 0 )
            goto LABEL_6;
        }
        sub_9C95B0((__int64)&v119, (__int64)v15);
        v9 = *((_QWORD *)v15 + 2);
        if ( !v9 )
          goto LABEL_6;
        goto LABEL_5;
      }
    }
  }
  v123 = v15;
  if ( !(_DWORD)v120 )
    goto LABEL_6;
  if ( (_BYTE)qword_5017E28 )
    goto LABEL_49;
  if ( (v118[7] & 0x40) != 0 )
    v69 = (unsigned __int8 *)*((_QWORD *)v118 - 1);
  else
    v69 = &v118[-32 * (*((_DWORD *)v118 + 1) & 0x7FFFFFF)];
  v109 = *(_QWORD *)(*(_QWORD *)v69 + 8LL);
  v70 = *(_QWORD *)(*((_QWORD *)v15 - 4) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v70 + 8) - 17 <= 1 )
    v70 = **(_QWORD **)(v70 + 16);
  v107 = v116;
  v99 = *(_DWORD *)(v70 + 8) >> 8;
  _BitScanReverse64(&v71, 1LL << (*((_WORD *)v15 + 1) >> 1));
  v102 = *(__int64 (**)())(*(_QWORD *)v116 + 808LL);
  v72 = 63 - (v71 ^ 0x3F);
  v73 = sub_2D5BAE0(v116, v115, *(__int64 **)(*((_QWORD *)v15 - 8) + 8LL), 0);
  if ( v102 == sub_2D56600
    || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, __int64))v102)(
          v107,
          v73,
          v74,
          v99,
          v72,
          0,
          0,
          v73) )
  {
    v10 = v119;
    v2 = 0;
    goto LABEL_7;
  }
  v75 = sub_DFD3F0(v117);
  v76 = v122;
  v77 = v75;
  v103 = v2;
  v79 = v78;
  v80 = (unsigned __int8 **)v119;
  v81 = 0;
  v108 = (unsigned __int8 **)&v119[(unsigned int)v120];
  while ( v108 != v80 )
  {
    v91 = *v80;
    if ( ((*v80)[7] & 0x40) != 0 )
      v82 = (unsigned __int8 *)*((_QWORD *)v91 - 1);
    else
      v82 = &v91[-32 * (*((_DWORD *)v91 + 1) & 0x7FFFFFF)];
    v83 = **(unsigned __int8 **)v82;
    if ( (unsigned __int8)(v83 - 17) > 1u && (unsigned int)(v83 - 12) > 1 )
    {
      v85 = 0;
      v84 = 2;
    }
    else
    {
      v84 = 0;
      v85 = 2;
    }
    v95 = v84;
    v97 = v85;
    v86 = sub_DFD800(v117, (unsigned int)*v91 - 29, *((_QWORD *)v91 + 1), 0, v85, v84, 0, 0, 0, 0);
    if ( v87 == 1 )
      v79 = 1;
    v88 = __OFADD__(v86, v77);
    v77 += v86;
    if ( v88 )
    {
      v77 = 0x8000000000000000LL;
      if ( v86 > 0 )
        v77 = 0x7FFFFFFFFFFFFFFFLL;
    }
    v89 = sub_DFD800(v117, (unsigned int)**v80 - 29, v109, 0, v97, v95, 0, 0, 0, 0);
    if ( v90 == 1 )
      v81 = 1;
    v88 = __OFADD__(v89, v76);
    v76 += v89;
    if ( v88 )
    {
      v76 = 0x8000000000000000LL;
      if ( v89 > 0 )
        v76 = 0x7FFFFFFFFFFFFFFFLL;
    }
    ++v80;
  }
  v92 = v79;
  v2 = v103;
  if ( v92 == v81 )
    v93 = v77 > v76;
  else
    v93 = v81 < v92;
  if ( v93 )
  {
LABEL_49:
    v10 = v119;
    v94 = &v119[(unsigned int)v120];
    if ( v119 != v94 )
    {
      v96 = v119;
      do
      {
        v31 = (unsigned __int8 *)*v96;
        sub_BD84D0(*v96, (__int64)v118);
        if ( (v118[7] & 0x40) != 0 )
          v32 = (unsigned __int8 *)*((_QWORD *)v118 - 1);
        else
          v32 = &v118[-32 * (*((_DWORD *)v118 + 1) & 0x7FFFFFF)];
        v33 = 32LL * (*((_DWORD *)v31 + 1) & 0x7FFFFFF);
        *((_QWORD *)v31 + 1) = *(_QWORD *)(*(_QWORD *)v32 + 8LL);
        if ( (v31[7] & 0x40) != 0 )
        {
          v34 = (unsigned __int8 *)*((_QWORD *)v31 - 1);
          v106 = &v34[v33];
        }
        else
        {
          v106 = v31;
          v34 = &v31[-v33];
        }
        while ( v106 != v34 )
        {
          v35 = *(unsigned __int8 **)v34;
          v36 = v118;
          if ( *(unsigned __int8 **)v34 == v118 )
          {
            if ( (v35[7] & 0x40) != 0 )
              v65 = (__int64 *)*((_QWORD *)v35 - 1);
            else
              v65 = (__int64 *)&v35[-32 * (*((_DWORD *)v35 + 1) & 0x7FFFFFF)];
            v55 = *v65;
            goto LABEL_84;
          }
          v37 = *v35;
          if ( (_BYTE)v37 == 12 )
          {
            v40 = -1;
            v38 = 1;
            v41 = v118[7] & 0x40;
            goto LABEL_67;
          }
          v38 = v37 == 13 || (unsigned __int8)(*v35 - 17) <= 1u;
          if ( !v38 )
            goto LABEL_158;
          if ( v37 == 13 )
            goto LABEL_66;
          if ( (unsigned int)sub_BD2910((__int64)v34) == 1 )
          {
            v39 = *v31 - 29;
            if ( v39 > 0x17 )
            {
              if ( *v31 != 53 )
                goto LABEL_101;
LABEL_64:
              if ( !sub_B451C0((__int64)v31) )
                goto LABEL_65;
              goto LABEL_101;
            }
            if ( v39 > 0x15 )
              goto LABEL_65;
            if ( *v31 == 50 )
              goto LABEL_64;
            if ( v39 > 0x12 )
            {
LABEL_65:
              v36 = v118;
LABEL_66:
              v40 = -1;
              v41 = v36[7] & 0x40;
LABEL_67:
              if ( v41 )
                v42 = (unsigned __int8 *)*((_QWORD *)v36 - 1);
              else
                v42 = &v36[-32 * (*((_DWORD *)v36 + 1) & 0x7FFFFFF)];
              v43 = *(_QWORD *)(*(_QWORD *)v42 + 8LL);
              v44 = *(_BYTE *)(v43 + 8);
              v45 = *(_DWORD *)(v43 + 32);
              LODWORD(v111) = v45;
              BYTE4(v111) = v44 == 18;
              if ( !v38 )
              {
                v100 = v40;
                if ( v44 == 18 )
LABEL_158:
                  BUG();
                v112 = (__int64 *)v114;
                v113 = 0x400000000LL;
                v46 = sub_ACADE0(*((__int64 ***)v35 + 1));
                if ( v45 )
                {
                  v48 = v100;
                  v98 = v31;
                  v49 = 0;
                  v50 = (__int64)v35;
                  v101 = v34;
                  LODWORD(v51) = v113;
                  v52 = v46;
                  v53 = v48;
                  do
                  {
                    v66 = (unsigned int)v51;
                    v54 = (unsigned int)v51 + 1LL;
                    if ( v53 == v49 )
                    {
                      if ( v54 > HIDWORD(v113) )
                      {
                        sub_C8D5F0((__int64)&v112, v114, v54, 8u, v54, v47);
                        v66 = (unsigned int)v113;
                      }
                      v112[v66] = v50;
                      v51 = (unsigned int)(v113 + 1);
                      LODWORD(v113) = v113 + 1;
                    }
                    else
                    {
                      if ( v54 > HIDWORD(v113) )
                      {
                        sub_C8D5F0((__int64)&v112, v114, v54, 8u, v54, v47);
                        v66 = (unsigned int)v113;
                      }
                      v112[v66] = v52;
                      v51 = (unsigned int)(v113 + 1);
                      LODWORD(v113) = v113 + 1;
                    }
                    ++v49;
                  }
                  while ( v45 != v49 );
                  v31 = v98;
                  v34 = v101;
                }
                else
                {
                  v51 = (unsigned int)v113;
                }
                v55 = sub_AD3730(v112, v51);
                if ( v112 != (__int64 *)v114 )
                  _libc_free((unsigned __int64)v112);
                goto LABEL_84;
              }
              goto LABEL_114;
            }
          }
LABEL_101:
          v36 = v118;
          v41 = v118[7] & 0x40;
          if ( v41 )
          {
            v62 = (unsigned __int8 *)*((_QWORD *)v118 - 1);
            v63 = *((_QWORD *)v62 + 4);
            if ( *(_BYTE *)v63 == 17 )
              goto LABEL_103;
          }
          else
          {
            v62 = &v118[-32 * (*((_DWORD *)v118 + 1) & 0x7FFFFFF)];
            v63 = *((_QWORD *)v62 + 4);
            if ( *(_BYTE *)v63 == 17 )
            {
LABEL_103:
              v40 = *(_DWORD *)(v63 + 32);
              v64 = *(unsigned int **)(v63 + 24);
              if ( v40 > 0x40 )
              {
                v40 = *v64;
              }
              else if ( v40 )
              {
                v40 = (__int64)((_QWORD)v64 << (64 - (unsigned __int8)v40)) >> (64 - (unsigned __int8)v40);
              }
              v38 = 0;
              goto LABEL_67;
            }
          }
          v67 = *(_QWORD *)(*(_QWORD *)v62 + 8LL);
          v68 = *(_BYTE *)(v67 + 8) == 18;
          LODWORD(v67) = *(_DWORD *)(v67 + 32);
          BYTE4(v111) = v68;
          LODWORD(v111) = v67;
LABEL_114:
          v55 = sub_AD5E10(v111, v35);
LABEL_84:
          v56 = sub_BD2910((__int64)v34);
          if ( (v31[7] & 0x40) != 0 )
            v57 = (unsigned __int8 *)*((_QWORD *)v31 - 1);
          else
            v57 = &v31[-32 * (*((_DWORD *)v31 + 1) & 0x7FFFFFF)];
          v58 = &v57[32 * v56];
          if ( *(_QWORD *)v58 )
          {
            v59 = *((_QWORD *)v58 + 1);
            **((_QWORD **)v58 + 2) = v59;
            if ( v59 )
              *(_QWORD *)(v59 + 16) = *((_QWORD *)v58 + 2);
          }
          *(_QWORD *)v58 = v55;
          if ( v55 )
          {
            v60 = *(_QWORD *)(v55 + 16);
            *((_QWORD *)v58 + 1) = v60;
            if ( v60 )
              *(_QWORD *)(v60 + 16) = v58 + 8;
            *((_QWORD *)v58 + 2) = v55 + 16;
            *(_QWORD *)(v55 + 16) = v58;
          }
          v34 += 32;
        }
        sub_B44530(v118, (__int64)v31);
        if ( (v118[7] & 0x40) != 0 )
          v61 = (unsigned __int8 *)*((_QWORD *)v118 - 1);
        else
          v61 = &v118[-32 * (*((_DWORD *)v118 + 1) & 0x7FFFFFF)];
        sub_AC2B30((__int64)v61, (__int64)v31);
        ++v96;
      }
      while ( v94 != v96 );
      v10 = v119;
    }
    v2 = 1;
    goto LABEL_7;
  }
LABEL_6:
  v10 = v119;
LABEL_7:
  if ( v10 != (__int64 *)v121 )
    _libc_free((unsigned __int64)v10);
  return v2;
}
