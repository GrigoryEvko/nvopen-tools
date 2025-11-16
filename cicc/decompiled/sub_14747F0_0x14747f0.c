// Function: sub_14747F0
// Address: 0x14747f0
//
__int64 __fastcall sub_14747F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r15
  __int16 v8; // ax
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // r12
  __int16 v12; // ax
  __int16 v13; // ax
  unsigned int v14; // r13d
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // eax
  _QWORD *v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // r14
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  unsigned int v31; // r13d
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rbx
  __int64 v35; // rsi
  __int64 v36; // rax
  unsigned int v37; // eax
  __int16 v38; // r8
  unsigned int v39; // ebx
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // rax
  char v43; // r8
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  unsigned int v49; // r8d
  unsigned int v50; // ebx
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rcx
  int v56; // r14d
  __int64 v57; // r13
  unsigned int v58; // r14d
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 v63; // rax
  __int16 v64; // dx
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rcx
  unsigned int v70; // ebx
  __int64 v71; // r13
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int16 v75; // r8
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int16 v83; // bx
  __int64 v84; // rax
  __int64 v85; // rax
  unsigned int v86; // [rsp+Ch] [rbp-1C4h]
  __int64 v87; // [rsp+10h] [rbp-1C0h]
  __int64 v88; // [rsp+18h] [rbp-1B8h]
  __int64 v89; // [rsp+20h] [rbp-1B0h]
  __int64 v90; // [rsp+20h] [rbp-1B0h]
  __int64 v91; // [rsp+28h] [rbp-1A8h]
  unsigned __int64 v92; // [rsp+28h] [rbp-1A8h]
  __int64 v93; // [rsp+30h] [rbp-1A0h]
  __int64 v94; // [rsp+30h] [rbp-1A0h]
  __int64 v95; // [rsp+30h] [rbp-1A0h]
  unsigned int v96; // [rsp+38h] [rbp-198h]
  __int64 v97; // [rsp+38h] [rbp-198h]
  __int64 v98; // [rsp+38h] [rbp-198h]
  __int64 v99; // [rsp+40h] [rbp-190h]
  __int64 v100; // [rsp+40h] [rbp-190h]
  unsigned int v101; // [rsp+48h] [rbp-188h]
  _QWORD *v102; // [rsp+48h] [rbp-188h]
  __int64 v103; // [rsp+48h] [rbp-188h]
  __int64 v104; // [rsp+50h] [rbp-180h]
  __int64 v105; // [rsp+50h] [rbp-180h]
  __int64 v106; // [rsp+58h] [rbp-178h]
  __int64 v107; // [rsp+58h] [rbp-178h]
  __int64 v108; // [rsp+60h] [rbp-170h]
  __int64 v109; // [rsp+60h] [rbp-170h]
  __int64 *v110; // [rsp+60h] [rbp-170h]
  __int64 v111; // [rsp+60h] [rbp-170h]
  __int64 *v112; // [rsp+60h] [rbp-170h]
  char v113; // [rsp+60h] [rbp-170h]
  __int64 v114; // [rsp+60h] [rbp-170h]
  __int64 v115; // [rsp+60h] [rbp-170h]
  unsigned int v116; // [rsp+60h] [rbp-170h]
  __int64 v117; // [rsp+60h] [rbp-170h]
  __int64 v118; // [rsp+60h] [rbp-170h]
  unsigned int v119; // [rsp+60h] [rbp-170h]
  __int64 v120; // [rsp+68h] [rbp-168h]
  __int64 v121; // [rsp+78h] [rbp-158h] BYREF
  __int64 v122[2]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v123[2]; // [rsp+90h] [rbp-140h] BYREF
  __int64 v124[2]; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v125; // [rsp+B0h] [rbp-120h] BYREF
  __int64 *v126; // [rsp+C0h] [rbp-110h] BYREF
  int v127; // [rsp+C8h] [rbp-108h]
  __int64 v128; // [rsp+D0h] [rbp-100h] BYREF
  __int64 *v129; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v130; // [rsp+E8h] [rbp-E8h]
  __int64 v131[4]; // [rsp+F0h] [rbp-E0h] BYREF
  unsigned __int64 v132[2]; // [rsp+110h] [rbp-C0h] BYREF
  _BYTE v133[176]; // [rsp+120h] [rbp-B0h] BYREF

  v4 = a1;
  while ( 1 )
  {
    a3 = sub_1456E10(a1, a3);
    v8 = *(_WORD *)(a2 + 24);
    if ( !v8 )
    {
      v9 = sub_15A3CB0(*(_QWORD *)(a2 + 32), a3, 0);
      return sub_145CE20(a1, v9);
    }
    if ( v8 != 2 )
      break;
    a2 = *(_QWORD *)(a2 + 32);
    ++a4;
  }
  v11 = a3;
  v132[0] = (unsigned __int64)v133;
  v132[1] = 0x2000000000LL;
  sub_16BD3E0(v132, 2);
  sub_16BD4C0(v132, a2);
  sub_16BD4C0(v132, a3);
  v108 = a1 + 816;
  v121 = 0;
  result = sub_16BDDE0(a1 + 816, v132, &v121);
  if ( !result )
  {
    if ( a4 > dword_4F9AC40 )
      goto LABEL_22;
    v12 = *(_WORD *)(a2 + 24);
    if ( v12 == 1 )
    {
      v91 = *(_QWORD *)(a2 + 32);
      sub_14779E0(v122, a1, v91);
      v99 = sub_1456C90(a1, *(_QWORD *)(a2 + 40));
      v101 = sub_1456C90(a1, a3);
      sub_158D430(v124, v122, v99);
      sub_158CEA0(&v126, v124, v101);
      sub_158DFD0(&v129, v122, v101);
      LOBYTE(v101) = sub_158BB40(&v126, &v129);
      sub_135E100(v131);
      sub_135E100((__int64 *)&v129);
      sub_135E100(&v128);
      sub_135E100((__int64 *)&v126);
      sub_135E100(&v125);
      sub_135E100(v124);
      if ( (_BYTE)v101 )
      {
        v109 = sub_1483B20(a1, v91, a3);
        sub_135E100(v123);
        sub_135E100(v122);
        result = v109;
        goto LABEL_7;
      }
      sub_135E100(v123);
      sub_135E100(v122);
      v12 = *(_WORD *)(a2 + 24);
    }
    if ( v12 != 7 || *(_QWORD *)(a2 + 40) != 2 )
    {
LABEL_17:
      if ( (unsigned __int8)sub_1484A80(a1, a2, &v126) )
      {
        v23 = sub_14747F0(a1, v129, a3, a4 + 1);
        v24 = sub_14747F0(a1, v126, a3, a4 + 1);
        result = sub_1484870(a1, v24, v23);
        goto LABEL_7;
      }
      v13 = *(_WORD *)(a2 + 24);
      if ( v13 == 6 )
      {
        v14 = a4 + 1;
        v15 = sub_14747F0(a1, *(_QWORD *)(a2 + 40), a3, a4 + 1);
        v16 = sub_14747F0(a1, *(_QWORD *)(a2 + 32), a3, v14);
        result = sub_1483CF0(a1, v16, v15);
        goto LABEL_7;
      }
      if ( v13 == 4 )
      {
        if ( (*(_BYTE *)(a2 + 26) & 2) != 0 )
        {
          v25 = a4 + 1;
          v129 = v131;
          v130 = 0x400000000LL;
          v26 = *(_QWORD *)(a2 + 32);
          v27 = *(_QWORD *)(a2 + 40);
          if ( v26 != v26 + 8 * v27 )
          {
            v110 = (__int64 *)(v26 + 8 * v27);
            v28 = *(__int64 **)(a2 + 32);
            do
            {
              v29 = *v28++;
              v126 = (__int64 *)sub_14747F0(a1, v29, v11, v25);
              sub_1458920((__int64)&v129, &v126);
            }
            while ( v110 != v28 );
          }
          result = sub_147DD40(a1, &v129, 2, v25);
          v30 = (unsigned __int64)v129;
          if ( v129 == v131 )
            goto LABEL_7;
          goto LABEL_32;
        }
        v45 = **(_QWORD **)(a2 + 32);
        if ( *(_WORD *)(v45 + 24) )
          goto LABEL_21;
        sub_1468E70((__int64)v124, a1, *(_QWORD *)(v45 + 32), a2);
        if ( !sub_13A38F0((__int64)v124, 0) )
        {
          v46 = sub_145CF40(a1, (__int64)v124);
          v47 = sub_14747F0(a1, v46, a3, a4);
          sub_13A38D0((__int64)&v126, (__int64)v124);
          sub_1455DC0((__int64)&v129, (__int64)&v126);
          v48 = sub_145CF40(a1, (__int64)&v129);
          v49 = a4;
          v50 = a4 + 1;
          v51 = sub_13A5B00(a1, v48, a2, 0, v49);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v126);
          v52 = sub_14747F0(a1, v51, a3, v50);
          v114 = sub_13A5B00(a1, v47, v52, 6u, v50);
          sub_135E100(v124);
          result = v114;
          goto LABEL_7;
        }
        sub_135E100(v124);
        v13 = *(_WORD *)(a2 + 24);
      }
      if ( v13 == 5 )
      {
        if ( (*(_BYTE *)(a2 + 26) & 2) != 0 )
        {
          v31 = a4 + 1;
          v129 = v131;
          v130 = 0x400000000LL;
          v32 = *(__int64 **)(a2 + 32);
          v33 = *(_QWORD *)(a2 + 40);
          v34 = v32;
          if ( v32 != &v32[v33] )
          {
            v112 = &v32[v33];
            do
            {
              v35 = *v34++;
              v126 = (__int64 *)sub_14747F0(a1, v35, v11, v31);
              sub_1458920((__int64)&v129, &v126);
            }
            while ( v112 != v34 );
            v4 = a1;
          }
          result = sub_147EE30(v4, &v129, 2, v31);
          v30 = (unsigned __int64)v129;
          if ( v129 == v131 )
            goto LABEL_7;
LABEL_32:
          v111 = result;
          _libc_free(v30);
          result = v111;
          goto LABEL_7;
        }
        if ( *(_QWORD *)(a2 + 40) == 2 )
        {
          v18 = *(_QWORD **)(a2 + 32);
          v105 = *v18;
          if ( !*(_WORD *)(*v18 + 24LL) )
          {
            v53 = *(_QWORD *)(*v18 + 32LL);
            if ( *(_DWORD *)(v53 + 32) > 0x40u )
            {
              v102 = *(_QWORD **)(a2 + 32);
              v17 = sub_16A5940(v53 + 24);
              v18 = v102;
              if ( v17 == 1 )
                goto LABEL_54;
            }
            else
            {
              v54 = *(_QWORD *)(v53 + 24);
              if ( v54 && (v54 & (v54 - 1)) == 0 )
              {
LABEL_54:
                v55 = v18[1];
                if ( *(_WORD *)(v55 + 24) == 1 )
                {
                  v115 = v55;
                  v56 = sub_1456C90(a1, *(_QWORD *)(v55 + 40));
                  v57 = *(_QWORD *)(v105 + 32);
                  v58 = sub_1455840(v57 + 24) - *(_DWORD *)(v57 + 32) + v56 + 1;
                  v59 = sub_15E0530(*(_QWORD *)(a1 + 24));
                  v60 = sub_1644900(v59, v58);
                  v61 = sub_14835F0(a1, *(_QWORD *)(v115 + 32), v60, 0);
                  v62 = sub_14747F0(a1, v61, a3, 0);
                  v63 = sub_14747F0(a1, v105, a3, 0);
                  result = sub_13A5B60(a1, v63, v62, 2u, a4 + 1);
                  goto LABEL_7;
                }
              }
            }
          }
        }
      }
LABEL_21:
      result = sub_16BDDE0(v108, v132, &v121);
      if ( result )
        goto LABEL_7;
LABEL_22:
      v19 = sub_16BD760(v132, a1 + 864);
      v21 = v20;
      v22 = sub_145CDC0(0x30u, (__int64 *)(a1 + 864));
      if ( v22 )
      {
        v106 = v22;
        sub_1456320(v22, v19, v21, a2, a3);
        v22 = v106;
      }
      v107 = v22;
      sub_16BDA20(v108, v22, v121);
      sub_146DBF0(a1, v107);
      result = v107;
      goto LABEL_7;
    }
    v100 = **(_QWORD **)(a2 + 32);
    v104 = sub_13A5BC0((_QWORD *)a2, a1);
    v36 = sub_1456040(**(_QWORD **)(a2 + 32));
    v37 = sub_1456C90(a1, v36);
    v38 = *(_WORD *)(a2 + 26);
    v96 = v37;
    v103 = *(_QWORD *)(a2 + 48);
    if ( (v38 & 2) != 0 )
      goto LABEL_43;
    v64 = sub_1479410(a1, a2);
    if ( (v64 & 6) != 0 )
      v64 |= 1u;
    v38 = v64 | *(_WORD *)(a2 + 26);
    *(_WORD *)(a2 + 26) = v38;
    if ( (v38 & 2) != 0 )
      goto LABEL_43;
    v92 = sub_1474260(a1, v103);
    if ( sub_14562D0(v92)
      || (v65 = sub_1456040(v100),
          v89 = sub_1483B20(a1, v92, v65),
          v66 = sub_1456040(v92),
          v92 != sub_1483B20(a1, v89, v66)) )
    {
LABEL_61:
      if ( !sub_14562D0(v92) || *(_BYTE *)(a1 + 32) )
        goto LABEL_87;
      v67 = *(_QWORD *)(a1 + 48);
      if ( !*(_BYTE *)(v67 + 184) )
      {
        v95 = *(_QWORD *)(a1 + 48);
        sub_14CDF70(v95);
        v67 = v95;
      }
      if ( *(_DWORD *)(v67 + 16) )
      {
LABEL_87:
        if ( (unsigned __int8)sub_1477C30(a1, v104) )
        {
          sub_1477A60(&v126, a1, v104);
          sub_135E0D0((__int64)v124, v96, 0, 0);
          sub_1455E30((__int64)&v129, (__int64)v124, (__int64)&v126);
          v97 = sub_145CF40(a1, (__int64)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100(v124);
          sub_135E100((__int64 *)&v126);
          if ( (unsigned __int8)sub_1474350(a1, v103, 0x24u, a2, v97)
            || (unsigned __int8)sub_1474770(a1, 0x24u, a2, v97) )
          {
LABEL_66:
            v38 = *(_WORD *)(a2 + 26) | 3;
            *(_WORD *)(a2 + 26) = v38;
LABEL_43:
            v39 = a4 + 1;
            v113 = v38;
            v40 = sub_14747F0(a1, v104, a3, v39);
LABEL_44:
            v41 = v40;
            v42 = sub_1491390(a2, a3, a1, v39);
            v43 = v113;
            v44 = v42;
LABEL_45:
            result = sub_14799E0(a1, v44, v41, v103, v43 & 7);
            goto LABEL_7;
          }
        }
        else if ( (unsigned __int8)sub_1477B50(a1, v104) )
        {
          v74 = sub_1477920(a1, v104, 1);
          sub_158ACE0(&v126, v74);
          sub_135E0D0((__int64)v124, v96, -1, 1u);
          sub_1455E30((__int64)&v129, (__int64)v124, (__int64)&v126);
          v98 = sub_145CF40(a1, (__int64)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100(v124);
          sub_135E100((__int64 *)&v126);
          if ( (unsigned __int8)sub_1474350(a1, v103, 0x22u, a2, v98)
            || (unsigned __int8)sub_1474770(a1, 0x22u, a2, v98) )
          {
            v39 = a4 + 1;
            v75 = *(_WORD *)(a2 + 26) | 1;
            *(_WORD *)(a2 + 26) = v75;
            v113 = v75;
            v40 = sub_147B0D0(a1, v104, a3, v39);
            goto LABEL_44;
          }
        }
      }
      if ( !*(_WORD *)(v100 + 24) )
      {
        v93 = *(_QWORD *)(v100 + 32) + 24LL;
        sub_14689D0((__int64)v124, a1, v93, v104);
        if ( !sub_13A38F0((__int64)v124, 0) )
        {
          v68 = sub_145CF40(a1, (__int64)v124);
          v69 = a4;
          v70 = a4 + 1;
          v71 = sub_14747F0(a1, v68, a3, v69);
          v116 = *(_WORD *)(a2 + 26) & 7;
          sub_13A38D0((__int64)&v126, v93);
          sub_16A7590(&v126, v124);
          LODWORD(v130) = v127;
          v127 = 0;
          v129 = v126;
          v72 = sub_145CF40(a1, (__int64)&v129);
          v117 = sub_14799E0(a1, v72, v104, v103, v116);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v126);
          v73 = sub_14747F0(a1, v117, a3, v70);
          v118 = sub_13A5B00(a1, v71, v73, 6u, v70);
          sub_135E100(v124);
          result = v118;
          goto LABEL_7;
        }
        sub_135E100(v124);
      }
      if ( !(unsigned __int8)sub_147AA20(a1, v100, v104, v103) )
        goto LABEL_17;
      goto LABEL_66;
    }
    v76 = sub_15E0530(*(_QWORD *)(a1 + 24));
    v94 = sub_1644900(v76, 2 * v96);
    v77 = sub_13A5B60(a1, v89, v104, 0, a4 + 1);
    v78 = sub_13A5B00(a1, v100, v77, 0, a4 + 1);
    v88 = sub_14747F0(a1, v78, v94, a4 + 1);
    v86 = a4 + 1;
    v87 = sub_14747F0(a1, v100, v94, a4 + 1);
    v90 = sub_14747F0(a1, v89, v94, a4 + 1);
    v79 = sub_14747F0(a1, v104, v94, a4 + 1);
    v80 = sub_13A5B60(a1, v90, v79, 0, a4 + 1);
    if ( v88 == sub_13A5B00(a1, v87, v80, 0, a4 + 1) )
    {
      v119 = a4 + 1;
      v83 = *(_WORD *)(a2 + 26) | 3;
      *(_WORD *)(a2 + 26) = v83;
      v84 = sub_14747F0(a1, v104, a3, v86);
    }
    else
    {
      v81 = sub_147B0D0(a1, v104, v94, v86);
      v82 = sub_13A5B60(a1, v90, v81, 0, v86);
      if ( v88 != sub_13A5B00(a1, v87, v82, 0, v86) )
        goto LABEL_61;
      v119 = a4 + 1;
      v83 = *(_WORD *)(a2 + 26) | 1;
      *(_WORD *)(a2 + 26) = v83;
      v84 = sub_147B0D0(a1, v104, a3, v86);
    }
    v41 = v84;
    v85 = sub_1491390(a2, a3, a1, v119);
    v43 = v83;
    v44 = v85;
    goto LABEL_45;
  }
LABEL_7:
  if ( (_BYTE *)v132[0] != v133 )
  {
    v120 = result;
    _libc_free(v132[0]);
    return v120;
  }
  return result;
}
