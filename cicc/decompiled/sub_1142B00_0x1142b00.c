// Function: sub_1142B00
// Address: 0x1142b00
//
unsigned __int8 *__fastcall sub_1142B00(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v7; // r13
  __int64 v9; // rbx
  int v10; // eax
  unsigned __int8 *v11; // r10
  unsigned __int8 v12; // dl
  unsigned __int8 *result; // rax
  unsigned int v14; // ecx
  unsigned __int8 v15; // dl
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rdi
  _BYTE **v19; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // r15
  _BYTE *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r10
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  unsigned int **v29; // rdi
  __int64 v30; // rdx
  unsigned int **v31; // r9
  unsigned int v32; // eax
  __int64 v33; // rax
  bool v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // r14
  __int64 v38; // r12
  char v39; // bl
  __int16 v40; // r13
  _BYTE *v41; // r14
  _BYTE *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int v45; // r10d
  int v46; // eax
  __int64 v47; // rcx
  __int64 v48; // r13
  unsigned __int8 *v49; // r15
  __int64 v50; // r12
  unsigned int v51; // r14d
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r14
  int v59; // edx
  __int64 v60; // rsi
  __int64 v61; // r12
  __int64 v62; // rcx
  __int64 v63; // rdx
  unsigned int v64; // ecx
  __int64 v65; // rdx
  int v66; // eax
  _QWORD *v67; // rdi
  __int64 *v68; // rax
  __int64 v69; // rax
  int v70; // r9d
  __int64 v71; // r13
  __int64 v72; // r12
  __int64 v73; // r8
  __int64 v74; // r11
  unsigned __int8 v75; // al
  __int64 v76; // rax
  int v77; // eax
  unsigned int **v78; // rdi
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // [rsp+8h] [rbp-B8h]
  int v82; // [rsp+10h] [rbp-B0h]
  int v83; // [rsp+14h] [rbp-ACh]
  __int64 v84; // [rsp+18h] [rbp-A8h]
  __int64 v85; // [rsp+20h] [rbp-A0h]
  int v86; // [rsp+28h] [rbp-98h]
  int v87; // [rsp+28h] [rbp-98h]
  __int64 v88; // [rsp+30h] [rbp-90h]
  _BYTE *v89; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v90; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v91; // [rsp+30h] [rbp-90h]
  int v92; // [rsp+30h] [rbp-90h]
  __int64 v93; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v94; // [rsp+38h] [rbp-88h]
  __int64 v95; // [rsp+38h] [rbp-88h]
  unsigned int **v96; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v97; // [rsp+38h] [rbp-88h]
  int v98; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v99; // [rsp+38h] [rbp-88h]
  char v100; // [rsp+38h] [rbp-88h]
  __int64 v101; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v102; // [rsp+38h] [rbp-88h]
  __int64 v103; // [rsp+38h] [rbp-88h]
  int v104; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v105; // [rsp+38h] [rbp-88h]
  __int64 v106; // [rsp+40h] [rbp-80h] BYREF
  __int64 v107; // [rsp+48h] [rbp-78h]
  unsigned __int64 v108; // [rsp+50h] [rbp-70h]
  __int64 v109; // [rsp+58h] [rbp-68h]
  unsigned __int64 v110; // [rsp+60h] [rbp-60h] BYREF
  __int64 v111; // [rsp+68h] [rbp-58h]
  __int16 v112; // [rsp+80h] [rbp-40h]

  v5 = a2;
  v7 = a1;
  v9 = a4;
  if ( sub_B532B0(a4) )
    return 0;
  if ( *a3 != 63 )
    a3 = sub_BD3990(a3, a2);
  v10 = *(_DWORD *)(a2 + 4);
  v106 = v9;
  v11 = *(unsigned __int8 **)(a2 - 32LL * (v10 & 0x7FFFFFF));
  if ( v11 == a3 )
  {
    if ( (unsigned int)(v9 - 32) <= 1 || *(_BYTE *)(a2 + 1) >> 1 )
    {
      v37 = sub_F20BF0(a1, a2, 0);
      v38 = sub_AD6530(*((_QWORD *)v37 + 1), a2);
      v39 = *(_BYTE *)(a2 + 1) >> 1;
      if ( (v39 & 4) != 0 )
      {
        v112 = 257;
        result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( result )
        {
          v102 = result;
          sub_1113300((__int64)result, v106, (__int64)v37, v38, (__int64)&v110);
          result = v102;
        }
        result[1] = result[1] & 1 | (2 * ((result[1] >> 1) & 0xFE | ((v39 & 2) != 0)));
      }
      else
      {
        v40 = sub_B52E90(v106);
        v112 = 257;
        result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( result )
        {
          v99 = result;
          sub_1113300((__int64)result, v40, (__int64)v37, v38, (__int64)&v110);
          return v99;
        }
      }
      return result;
    }
  }
  else
  {
    v12 = *a3;
    if ( (*(_BYTE *)(a2 + 1) & 2) == 0 || (unsigned int)(v9 - 32) > 1 || v12 > 0x15u )
      goto LABEL_7;
    v97 = *(unsigned __int8 **)(a2 - 32LL * (v10 & 0x7FFFFFF));
    v34 = sub_AC30F0((__int64)a3);
    v11 = v97;
    if ( v34 )
    {
      v35 = *((_QWORD *)a3 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 <= 1 )
        v35 = **(_QWORD **)(v35 + 16);
      v90 = v97;
      v98 = *(_DWORD *)(v35 + 8) >> 8;
      v36 = sub_B43CB0(a5);
      if ( !sub_B2F070(v36, v98) )
      {
        v57 = *(_QWORD *)(a2 + 8);
        v58 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        v59 = *(unsigned __int8 *)(v57 + 8);
        v60 = *(_QWORD *)(v58 + 8);
        if ( (unsigned int)(v59 - 17) <= 1 && *(_BYTE *)(v60 + 8) == 14 )
        {
          v77 = *(_DWORD *)(v57 + 32);
          v78 = *(unsigned int ***)(a1 + 32);
          BYTE4(v108) = (_BYTE)v59 == 18;
          LODWORD(v108) = v77;
          v112 = 257;
          v79 = sub_B37620(v78, v108, v58, (__int64 *)&v110);
          v60 = *(_QWORD *)(v79 + 8);
          v58 = v79;
        }
        v61 = sub_ADB060((unsigned __int64)a3, v60);
        v112 = 257;
        result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( !result )
          return result;
        v62 = v61;
        v63 = v58;
        goto LABEL_70;
      }
      v12 = *a3;
      v11 = v90;
LABEL_7:
      if ( v12 > 0x1Cu )
        goto LABEL_8;
LABEL_11:
      if ( v12 != 5 || *((_WORD *)a3 + 1) != 34 )
        return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
      goto LABEL_13;
    }
  }
  v12 = *a3;
  if ( *a3 <= 0x1Cu )
    goto LABEL_11;
LABEL_8:
  if ( v12 != 63 )
    return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
LABEL_13:
  v14 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
  v15 = *(_BYTE *)(a2 + 1) >> 1;
  v16 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  LODWORD(v17) = v15 & (a3[1] >> 1);
  v18 = *(_QWORD *)&a3[-32 * v14];
  if ( (unsigned __int8 *)v18 == v11 )
  {
    if ( v14 == (_DWORD)v16
      && (v92 = v15 & (a3[1] >> 1), v103 = sub_BB5290(v5),
                                    v44 = sub_BB5290((__int64)a3),
                                    LODWORD(v17) = v92,
                                    v103 == v44) )
    {
      v45 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
      if ( v45 == 1 )
        goto LABEL_94;
      v46 = *(_DWORD *)(v5 + 4);
      v83 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
      v47 = 0;
      v82 = 0;
      v87 = v9;
      v104 = v92;
      v17 = v45;
      v93 = v7;
      v48 = v5;
      v49 = a3;
      v50 = v46 & 0x7FFFFFF;
      v81 = a5;
      v51 = 1;
      do
      {
        v52 = *(_QWORD *)&v49[32 * (v51 - v17)];
        v53 = *(_QWORD *)(v48 + 32 * (v51 - v50));
        if ( v52 != v53 )
        {
          v54 = *(_QWORD *)(v53 + 8);
          v84 = v17;
          v85 = *(_QWORD *)(v52 + 8);
          v110 = sub_BCAE30(v85);
          v111 = v55;
          v108 = sub_BCAE30(v54);
          v17 = v84;
          v109 = v56;
          if ( v108 != v110
            || (_BYTE)v109 != (_BYTE)v111
            || (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v48 + 8) + 8LL) - 17 <= 1
            && ((unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 > 1
             || (unsigned int)*(unsigned __int8 *)(v85 + 8) - 17 > 1)
            || v82 )
          {
            a3 = v49;
            LODWORD(v17) = v104;
            v5 = v48;
            LODWORD(v9) = v87;
            v7 = v93;
            goto LABEL_50;
          }
          v82 = 1;
          v47 = v51;
        }
        ++v51;
      }
      while ( v51 != v83 );
      v73 = v50;
      v74 = v17;
      a3 = v49;
      LOBYTE(v17) = v104;
      v5 = v48;
      LODWORD(v9) = v87;
      v7 = v93;
      a5 = v81;
      if ( !v82 )
      {
LABEL_94:
        v75 = sub_B535D0(v9);
        v76 = sub_AD64C0(*(_QWORD *)(a5 + 8), v75, 0);
        return sub_F162A0(v7, a5, v76);
      }
      if ( v104 )
        return (unsigned __int8 *)sub_1111430(
                                    (unsigned int *)&v106,
                                    v104,
                                    *(_QWORD *)(v5 + 32 * (v47 - v73)),
                                    *(_QWORD *)&a3[32 * (v47 - v74)]);
      if ( (unsigned int)(v87 - 32) > 1 )
        return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
    }
    else
    {
LABEL_50:
      if ( (unsigned int)(v9 - 32) > 1 && !(_DWORD)v17 )
        return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
    }
    v100 = v17;
    v41 = sub_F20BF0(v7, v5, 1);
    v42 = sub_F20BF0(v7, (__int64)a3, 1);
    return (unsigned __int8 *)sub_1111430((unsigned int *)&v106, v100, (__int64)v41, (__int64)v42);
  }
  if ( v14 != (_DWORD)v16
    || *(_QWORD *)(v18 + 8) != *(_QWORD *)(*(_QWORD *)(v5 - 32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)) + 8LL) )
  {
    goto LABEL_15;
  }
  v86 = v15 & (a3[1] >> 1);
  v91 = v11;
  v101 = sub_BB5290(v5);
  v43 = sub_BB5290((__int64)a3);
  v11 = v91;
  if ( v101 == v43 )
  {
    v16 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
    if ( (_DWORD)v16 != 1 )
    {
      v64 = 1;
      while ( *(_QWORD *)&a3[32 * (v64 - (unsigned __int64)(*((_DWORD *)a3 + 1) & 0x7FFFFFF))] == *(_QWORD *)(v5 + 32 * (v64 - (unsigned __int64)(unsigned int)v16)) )
      {
        if ( ++v64 == (_DWORD)v16 )
        {
          v65 = *(_QWORD *)(*(_QWORD *)(v5 - 32LL * (unsigned int)v16) + 8LL);
          goto LABEL_76;
        }
      }
      goto LABEL_55;
    }
    v65 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 8LL);
LABEL_76:
    if ( (unsigned int)*(unsigned __int8 *)(v65 + 8) - 17 > 1 )
    {
      v69 = sub_BCB2A0(*(_QWORD **)v65);
      v70 = v86;
      v11 = v91;
    }
    else
    {
      v66 = *(_DWORD *)(v65 + 32);
      v67 = *(_QWORD **)v65;
      BYTE4(v107) = *(_BYTE *)(v65 + 8) == 18;
      LODWORD(v107) = v66;
      v68 = (__int64 *)sub_BCB2A0(v67);
      v16 = v107;
      v69 = sub_BCE1B0(v68, v107);
      v11 = v91;
      v70 = v86;
    }
    if ( *(_QWORD *)(a5 + 8) != v69 || (unsigned int)(v9 - 32) > 1 && !v70 )
      goto LABEL_55;
    v71 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
    v72 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v112 = 257;
    result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
    if ( !result )
      return result;
    v62 = v72;
    v63 = v71;
LABEL_70:
    v105 = result;
    sub_1113300((__int64)result, v9, v63, v62, (__int64)&v110);
    return v105;
  }
LABEL_55:
  v15 = *(_BYTE *)(v5 + 1) >> 1;
LABEL_15:
  if ( (v15 & 1) == 0 || (a3[1] & 2) == 0 )
    return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
  v19 = (_BYTE **)(v5 + 32 * (1LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)));
  if ( (_BYTE **)v5 != v19 )
  {
    while ( **v19 == 17 )
    {
      v19 += 4;
      if ( (_BYTE **)v5 == v19 )
        goto LABEL_23;
    }
    v20 = *(_QWORD *)(v5 + 16);
    if ( !v20 || *(_QWORD *)(v20 + 8) )
      return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
  }
LABEL_23:
  v21 = &a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
  if ( a3 != v21 )
  {
    while ( **(_BYTE **)v21 == 17 )
    {
      v21 += 32;
      if ( a3 == v21 )
        goto LABEL_29;
    }
    v22 = *((_QWORD *)a3 + 2);
    if ( !v22 || *(_QWORD *)(v22 + 8) )
      return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
  }
LABEL_29:
  v94 = sub_BD3990(v11, v16);
  if ( v94 != sub_BD3990(*(unsigned __int8 **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)], v16)
    || (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17 <= 1 )
  {
    return (unsigned __int8 *)sub_1140460(v5, (__int64)a3, v9, *(_QWORD *)(v7 + 88), v7);
  }
  v23 = sub_F20BF0(v7, v5, 0);
  v24 = sub_F20BF0(v7, (__int64)a3, 0);
  v25 = *((_QWORD *)v23 + 1);
  v26 = (__int64)v24;
  if ( *((_QWORD *)v24 + 1) != v25 )
  {
    v95 = *((_QWORD *)v24 + 1);
    v88 = (__int64)v24;
    v108 = sub_BCAE30(*((_QWORD *)v23 + 1));
    v109 = v27;
    v28 = sub_BCAE30(v95);
    v29 = *(unsigned int ***)(v7 + 32);
    v110 = v28;
    v111 = v30;
    v112 = 257;
    if ( v28 <= v108 )
    {
      v80 = sub_A82DA0(v29, (__int64)v23, v95, (__int64)&v110, 0, 0);
      v26 = v88;
      v23 = (_BYTE *)v80;
    }
    else
    {
      v26 = sub_A82DA0(v29, v88, v25, (__int64)&v110, 0, 0);
    }
  }
  v31 = *(unsigned int ***)(v7 + 32);
  v89 = (_BYTE *)v26;
  v112 = 257;
  v96 = v31;
  v32 = sub_B52E90(v9);
  v33 = sub_92B530(v96, v32, (__int64)v23, v89, (__int64)&v110);
  return sub_F162A0(v7, a5, v33);
}
