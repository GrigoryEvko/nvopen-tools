// Function: sub_147B0D0
// Address: 0x147b0d0
//
__int64 __fastcall sub_147B0D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r15
  __int16 v8; // ax
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // r14
  __int16 v12; // ax
  __int64 *v13; // rdx
  __int64 *v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r13d
  __int64 *v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // ebx
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // r8d
  unsigned int v31; // ebx
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // r11
  unsigned int v35; // r8d
  __int64 v36; // rdx
  __int16 v37; // ax
  __int16 v38; // ax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int16 v42; // r12
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rcx
  unsigned int v48; // ebx
  __int16 v49; // r13
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int16 v60; // bx
  __int64 v61; // rax
  __int64 v62; // r12
  __int64 v63; // rax
  unsigned int v64; // [rsp+4h] [rbp-1BCh]
  __int64 v65; // [rsp+8h] [rbp-1B8h]
  __int64 v66; // [rsp+10h] [rbp-1B0h]
  __int64 v67; // [rsp+10h] [rbp-1B0h]
  __int64 v68; // [rsp+18h] [rbp-1A8h]
  __int64 v69; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v70; // [rsp+20h] [rbp-1A0h]
  __int64 v71; // [rsp+20h] [rbp-1A0h]
  __int64 v72; // [rsp+28h] [rbp-198h]
  unsigned int v73; // [rsp+30h] [rbp-190h]
  int v74; // [rsp+30h] [rbp-190h]
  __int64 v75; // [rsp+30h] [rbp-190h]
  __int64 v76; // [rsp+30h] [rbp-190h]
  __int64 v77; // [rsp+30h] [rbp-190h]
  __int64 v78; // [rsp+30h] [rbp-190h]
  __int64 v79; // [rsp+38h] [rbp-188h]
  __int64 v80; // [rsp+40h] [rbp-180h]
  __int64 v81; // [rsp+40h] [rbp-180h]
  __int64 v82; // [rsp+40h] [rbp-180h]
  __int64 *v83; // [rsp+40h] [rbp-180h]
  __int64 v84; // [rsp+48h] [rbp-178h]
  __int64 v85; // [rsp+48h] [rbp-178h]
  __int64 v86; // [rsp+50h] [rbp-170h]
  __int64 v87; // [rsp+50h] [rbp-170h]
  __int64 *v88; // [rsp+50h] [rbp-170h]
  __int64 v89; // [rsp+50h] [rbp-170h]
  __int64 v90; // [rsp+50h] [rbp-170h]
  unsigned int v91; // [rsp+50h] [rbp-170h]
  unsigned int v92; // [rsp+50h] [rbp-170h]
  __int64 v93; // [rsp+58h] [rbp-168h]
  __int64 v94; // [rsp+68h] [rbp-158h] BYREF
  __int64 v95; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v96; // [rsp+78h] [rbp-148h]
  __int64 v97; // [rsp+80h] [rbp-140h] BYREF
  unsigned int v98; // [rsp+88h] [rbp-138h]
  __int64 v99[2]; // [rsp+90h] [rbp-130h] BYREF
  __int64 v100; // [rsp+A0h] [rbp-120h] BYREF
  __int64 *v101; // [rsp+B0h] [rbp-110h] BYREF
  int v102; // [rsp+B8h] [rbp-108h]
  __int64 v103; // [rsp+C0h] [rbp-100h] BYREF
  __int64 *v104; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v105; // [rsp+D8h] [rbp-E8h]
  __int64 v106[4]; // [rsp+E0h] [rbp-E0h] BYREF
  unsigned __int64 v107[2]; // [rsp+100h] [rbp-C0h] BYREF
  _BYTE v108[176]; // [rsp+110h] [rbp-B0h] BYREF

  v4 = a1;
  while ( 1 )
  {
    a3 = sub_1456E10(a1, a3);
    v8 = *(_WORD *)(a2 + 24);
    if ( !v8 )
    {
      v9 = sub_15A4460(*(_QWORD *)(a2 + 32), a3, 0);
      return sub_145CE20(a1, v9);
    }
    if ( v8 != 3 )
      break;
    a2 = *(_QWORD *)(a2 + 32);
    ++a4;
  }
  v11 = a3;
  if ( v8 == 2 )
    return sub_14747F0(a1, *(_QWORD *)(a2 + 32), a3, a4 + 1);
  v107[0] = (unsigned __int64)v108;
  v107[1] = 0x2000000000LL;
  sub_16BD3E0(v107, 3);
  sub_16BD4C0(v107, a2);
  sub_16BD4C0(v107, a3);
  v86 = a1 + 816;
  v94 = 0;
  result = sub_16BDDE0(a1 + 816, v107, &v94);
  if ( result )
    goto LABEL_9;
  if ( a4 > dword_4F9AC40 )
  {
LABEL_35:
    v19 = sub_16BD760(v107, a1 + 864);
    v21 = v20;
    v22 = sub_145CDC0(0x30u, (__int64 *)(a1 + 864));
    if ( v22 )
    {
      v84 = v22;
      sub_1456330(v22, v19, v21, a2, a3);
      v22 = v84;
    }
    v85 = v22;
    sub_16BDA20(v86, v22, v94);
    sub_146DBF0(a1, v85);
    result = v85;
    goto LABEL_9;
  }
  if ( byte_4F9A9A0 )
  {
    v12 = *(_WORD *)(a2 + 24);
    if ( v12 == 1 )
    {
      v66 = *(_QWORD *)(a2 + 32);
      v13 = sub_1477920(a1, v66, 1u);
      v96 = *((_DWORD *)v13 + 2);
      if ( v96 > 0x40 )
      {
        v83 = v13;
        sub_16A4FD0(&v95, v13);
        v13 = v83;
      }
      else
      {
        v95 = *v13;
      }
      v98 = *((_DWORD *)v13 + 6);
      if ( v98 > 0x40 )
        sub_16A4FD0(&v97, v13 + 2);
      else
        v97 = v13[2];
      v80 = sub_1456C90(a1, *(_QWORD *)(a2 + 40));
      v73 = sub_1456C90(a1, a3);
      sub_158D430(v99, &v95, v80);
      sub_158D100(&v101, v99, v73);
      sub_158E080(&v104, &v95, v73);
      LOBYTE(v73) = sub_158BB40(&v101, &v104);
      sub_135E100(v106);
      sub_135E100((__int64 *)&v104);
      sub_135E100(&v103);
      sub_135E100((__int64 *)&v101);
      sub_135E100(&v100);
      sub_135E100(v99);
      if ( (_BYTE)v73 )
      {
        v87 = sub_1483BD0(a1, v66, a3);
        sub_135E100(&v97);
        sub_135E100(&v95);
        result = v87;
        goto LABEL_9;
      }
      sub_135E100(&v97);
      sub_135E100(&v95);
      v12 = *(_WORD *)(a2 + 24);
    }
    if ( v12 == 4 )
    {
      if ( (*(_BYTE *)(a2 + 26) & 4) != 0 )
      {
        v104 = v106;
        v105 = 0x400000000LL;
        v14 = *(__int64 **)(a2 + 32);
        v15 = *(_QWORD *)(a2 + 40);
        v16 = a4 + 1;
        if ( v14 != &v14[v15] )
        {
          v88 = &v14[v15];
          v17 = v14;
          do
          {
            v18 = *v17++;
            v101 = (__int64 *)sub_147B0D0(a1, v18, v11, v16);
            sub_1458920((__int64)&v104, &v101);
          }
          while ( v88 != v17 );
          v4 = a1;
        }
        result = sub_147DD40(v4, &v104, 4, v16);
        if ( v104 != v106 )
        {
          v89 = result;
          _libc_free((unsigned __int64)v104);
          result = v89;
        }
        goto LABEL_9;
      }
      v27 = **(_QWORD **)(a2 + 32);
      if ( *(_WORD *)(v27 + 24) )
        goto LABEL_23;
      sub_1468E70((__int64)v99, a1, *(_QWORD *)(v27 + 32), a2);
      if ( !sub_13A38F0((__int64)v99, 0) )
      {
        v28 = sub_145CF40(a1, (__int64)v99);
        v82 = sub_147B0D0(a1, v28, a3, a4);
        sub_13A38D0((__int64)&v101, (__int64)v99);
        sub_1455DC0((__int64)&v104, (__int64)&v101);
        v29 = sub_145CF40(a1, (__int64)&v104);
        v30 = a4;
        v31 = a4 + 1;
        v32 = sub_13A5B00(a1, v29, a2, 0, v30);
        sub_135E100((__int64 *)&v104);
        sub_135E100((__int64 *)&v101);
        v33 = sub_147B0D0(a1, v32, a3, v31);
        v34 = v82;
        v35 = v31;
        v36 = v33;
LABEL_46:
        v90 = sub_13A5B00(a1, v34, v36, 6u, v35);
        sub_135E100(v99);
        result = v90;
        goto LABEL_9;
      }
      sub_135E100(v99);
      v12 = *(_WORD *)(a2 + 24);
    }
    if ( v12 != 7 || *(_QWORD *)(a2 + 40) != 2 )
      goto LABEL_23;
    v72 = **(_QWORD **)(a2 + 32);
    v81 = sub_13A5BC0((_QWORD *)a2, a1);
    v23 = sub_1456040(**(_QWORD **)(a2 + 32));
    v74 = sub_1456C90(a1, v23);
    v79 = *(_QWORD *)(a2 + 48);
    if ( (*(_BYTE *)(a2 + 26) & 4) != 0 )
      goto LABEL_42;
    v37 = sub_1479410(a1, a2);
    if ( (v37 & 6) != 0 )
      v37 |= 1u;
    v38 = *(_WORD *)(a2 + 26) | v37;
    *(_WORD *)(a2 + 26) = v38;
    if ( (v38 & 4) != 0 )
    {
LABEL_42:
      v24 = a4 + 1;
      v25 = sub_147B0D0(a1, v81, a3, v24);
      v26 = sub_148E580(a2, a3, a1, v24);
      result = sub_14799E0(a1, v26, v25, v79, 4u);
      goto LABEL_9;
    }
    v70 = sub_1474260(a1, v79);
    if ( sub_14562D0(v70) )
      goto LABEL_55;
    v39 = sub_1456040(v72);
    v68 = sub_1483B20(a1, v70, v39);
    v40 = sub_1456040(v70);
    if ( v70 != sub_1483B20(a1, v68, v40) )
      goto LABEL_55;
    v53 = sub_15E0530(*(_QWORD *)(a1 + 24));
    v77 = sub_1644900(v53, (unsigned int)(2 * v74));
    v54 = sub_13A5B60(a1, v68, v81, 0, a4 + 1);
    v55 = sub_13A5B00(a1, v72, v54, 0, a4 + 1);
    v67 = sub_147B0D0(a1, v55, v77, a4 + 1);
    v64 = a4 + 1;
    v65 = sub_147B0D0(a1, v72, v77, a4 + 1);
    v69 = sub_14747F0(a1, v68, v77, a4 + 1);
    v56 = sub_147B0D0(a1, v81, v77, a4 + 1);
    v57 = sub_13A5B60(a1, v69, v56, 0, a4 + 1);
    if ( v67 == sub_13A5B00(a1, v65, v57, 0, a4 + 1) )
    {
      v92 = a4 + 1;
      v60 = *(_WORD *)(a2 + 26) | 5;
      *(_WORD *)(a2 + 26) = v60;
      v61 = sub_147B0D0(a1, v81, a3, v64);
    }
    else
    {
      v58 = sub_14747F0(a1, v81, v77, v64);
      v59 = sub_13A5B60(a1, v69, v58, 0, v64);
      if ( v67 != sub_13A5B00(a1, v65, v59, 0, v64) )
      {
LABEL_55:
        if ( !sub_14562D0(v70) || *(_BYTE *)(a1 + 32) )
          goto LABEL_74;
        v45 = *(_QWORD *)(a1 + 48);
        if ( !*(_BYTE *)(v45 + 184) )
        {
          v78 = *(_QWORD *)(a1 + 48);
          sub_14CDF70(v78);
          v45 = v78;
        }
        if ( *(_DWORD *)(v45 + 16) )
        {
LABEL_74:
          v41 = sub_1477E50(v81, &v104, a1);
          if ( v41 )
          {
            v75 = v41;
            if ( (unsigned __int8)sub_1474350(a1, v79, (unsigned int)v104, a2, v41)
              || (unsigned __int8)sub_1474770(a1, (unsigned int)v104, a2, v75) )
            {
              goto LABEL_60;
            }
          }
        }
        if ( !*(_WORD *)(v72 + 24) )
        {
          v71 = *(_QWORD *)(v72 + 32) + 24LL;
          sub_14689D0((__int64)v99, a1, v71, v81);
          if ( !sub_13A38F0((__int64)v99, 0) )
          {
            v46 = sub_145CF40(a1, (__int64)v99);
            v47 = a4;
            v48 = a4 + 1;
            v76 = sub_147B0D0(a1, v46, a3, v47);
            v49 = *(_WORD *)(a2 + 26);
            sub_13A38D0((__int64)&v101, v71);
            sub_16A7590(&v101, v99);
            LODWORD(v105) = v102;
            v102 = 0;
            v104 = v101;
            v50 = sub_145CF40(a1, (__int64)&v104);
            v51 = sub_14799E0(a1, v50, v81, v79, v49 & 7);
            sub_135E100((__int64 *)&v104);
            sub_135E100((__int64 *)&v101);
            v52 = sub_147B0D0(a1, v51, a3, v48);
            v34 = v76;
            v35 = v48;
            v36 = v52;
            goto LABEL_46;
          }
          sub_135E100(v99);
        }
        if ( (unsigned __int8)sub_147ADD0(a1, v72, v81, v79) )
        {
LABEL_60:
          v91 = a4 + 1;
          v42 = *(_WORD *)(a2 + 26) | 5;
          *(_WORD *)(a2 + 26) = v42;
          v43 = sub_147B0D0(a1, v81, a3, a4 + 1);
          v44 = sub_148E580(a2, a3, a1, v91);
          result = sub_14799E0(a1, v44, v43, v79, v42 & 7);
          goto LABEL_9;
        }
        goto LABEL_23;
      }
      v92 = a4 + 1;
      v60 = *(_WORD *)(a2 + 26) | 1;
      *(_WORD *)(a2 + 26) = v60;
      v61 = sub_14747F0(a1, v81, a3, v64);
    }
    v62 = v61;
    v63 = sub_148E580(a2, a3, a1, v92);
    result = sub_14799E0(a1, v63, v62, v79, v60 & 7);
    goto LABEL_9;
  }
LABEL_23:
  if ( (unsigned __int8)sub_1477BC0(a1, a2) )
  {
    result = sub_14747F0(a1, a2, a3, a4 + 1);
    goto LABEL_9;
  }
  result = sub_16BDDE0(v86, v107, &v94);
  if ( !result )
    goto LABEL_35;
LABEL_9:
  if ( (_BYTE *)v107[0] != v108 )
  {
    v93 = result;
    _libc_free(v107[0]);
    return v93;
  }
  return result;
}
