// Function: sub_DC42C0
// Address: 0xdc42c0
//
_QWORD *__fastcall sub_DC42C0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r15
  __int16 v6; // dx
  __int64 v7; // r12
  unsigned int v8; // eax
  _QWORD *result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rbx
  __int16 v23; // ax
  unsigned int v24; // ebx
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 *v27; // r13
  __int64 *v28; // r12
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdi
  unsigned int v36; // r12d
  __int64 *v37; // r13
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rax
  unsigned int v45; // r12d
  __int64 v46; // r13
  __int64 v47; // rax
  void *v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rbx
  __int64 *v51; // rax
  __int64 *v52; // rdx
  __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  _QWORD *v58; // rax
  _QWORD *v59; // rax
  _QWORD *v60; // rbx
  unsigned int v61; // r12d
  __int64 v62; // rax
  void *v63; // r13
  __int64 v64; // rdx
  __int64 v65; // rbx
  __int64 *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  int v69; // eax
  _QWORD *v70; // rax
  __int16 v71; // bx
  _QWORD *v72; // rax
  unsigned int v73; // r12d
  __int64 v74; // r13
  __int64 v75; // rax
  _QWORD *v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // rax
  __int64 v83; // r12
  _QWORD *v84; // rax
  __int64 v85; // [rsp+0h] [rbp-1C0h]
  __int64 v86; // [rsp+8h] [rbp-1B8h]
  __int64 v87; // [rsp+10h] [rbp-1B0h]
  _QWORD *v88; // [rsp+10h] [rbp-1B0h]
  __int64 v89; // [rsp+18h] [rbp-1A8h]
  __int64 v90; // [rsp+18h] [rbp-1A8h]
  int v91; // [rsp+20h] [rbp-1A0h]
  __int64 v92; // [rsp+20h] [rbp-1A0h]
  __int64 v93; // [rsp+20h] [rbp-1A0h]
  __int64 v94; // [rsp+28h] [rbp-198h]
  unsigned int v95; // [rsp+30h] [rbp-190h]
  __int64 v96; // [rsp+30h] [rbp-190h]
  __int64 v97; // [rsp+38h] [rbp-188h]
  __int64 v98; // [rsp+38h] [rbp-188h]
  unsigned int v99; // [rsp+38h] [rbp-188h]
  __int64 *v100; // [rsp+40h] [rbp-180h]
  __int64 v101; // [rsp+40h] [rbp-180h]
  __int64 *v102; // [rsp+40h] [rbp-180h]
  __int64 *v103; // [rsp+48h] [rbp-178h]
  __int64 v104; // [rsp+48h] [rbp-178h]
  _QWORD *v105; // [rsp+48h] [rbp-178h]
  __int64 *v106; // [rsp+48h] [rbp-178h]
  __int64 v107; // [rsp+48h] [rbp-178h]
  _QWORD *v108; // [rsp+50h] [rbp-170h]
  _QWORD *v109; // [rsp+50h] [rbp-170h]
  __int64 v110; // [rsp+58h] [rbp-168h] BYREF
  __int64 *v111; // [rsp+68h] [rbp-158h] BYREF
  __int64 v112[2]; // [rsp+70h] [rbp-150h] BYREF
  __int64 v113[2]; // [rsp+80h] [rbp-140h] BYREF
  __int64 v114[2]; // [rsp+90h] [rbp-130h] BYREF
  __int64 v115; // [rsp+A0h] [rbp-120h] BYREF
  __int64 *v116; // [rsp+B0h] [rbp-110h] BYREF
  int v117; // [rsp+B8h] [rbp-108h]
  __int64 v118; // [rsp+C0h] [rbp-100h] BYREF
  __int64 *v119; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v120; // [rsp+D8h] [rbp-E8h]
  __int64 v121[4]; // [rsp+E0h] [rbp-E0h] BYREF
  _BYTE *v122; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v123; // [rsp+108h] [rbp-B8h]
  _BYTE v124[176]; // [rsp+110h] [rbp-B0h] BYREF

  v110 = a2;
  v5 = sub_D97090(a1, a3);
  v6 = *(_WORD *)(a2 + 24);
  switch ( v6 )
  {
    case 0:
      v7 = *(_QWORD *)(a2 + 32);
      v8 = sub_D97050(a1, v5);
      sub_C44830((__int64)&v122, (_DWORD *)(v7 + 24), v8);
      result = sub_DA26C0((__int64 *)a1, (__int64)&v122);
      if ( (unsigned int)v123 > 0x40 )
      {
        if ( v122 )
        {
          v108 = result;
          j_j___libc_free_0_0(v122);
          return v108;
        }
      }
      return result;
    case 4:
      return (_QWORD *)sub_DC5000(a1, *(_QWORD *)(v110 + 32), v5, a4 + 1);
    case 3:
      return sub_DC2B70(a1, *(_QWORD *)(v110 + 32), v5, a4 + 1);
  }
  v122 = v124;
  v123 = 0x2000000000LL;
  sub_9C8C60((__int64)&v122, 4);
  sub_D953B0((__int64)&v122, v110, v10, v11, v12, v13);
  sub_D953B0((__int64)&v122, v5, v14, v15, v16, v17);
  v18 = (__int64 *)&v122;
  v111 = 0;
  v103 = (__int64 *)(a1 + 1032);
  result = sub_C65B40(a1 + 1032, (__int64)&v122, (__int64 *)&v111, (__int64)off_49DEA80);
  if ( !result )
  {
    if ( a4 > dword_4F89148 )
    {
      v48 = sub_C65D30((__int64)&v122, (unsigned __int64 *)(a1 + 1064));
      v50 = v49;
      v51 = (__int64 *)sub_A777F0(0x30u, (__int64 *)(a1 + 1064));
      if ( v51 )
      {
        v100 = v51;
        sub_D96B90((__int64)v51, (__int64)v48, v50, v110, v5);
        v51 = v100;
      }
      v101 = (__int64)v51;
      sub_C657C0(v103, v51, v111, (__int64)off_49DEA80);
      v52 = &v110;
      goto LABEL_46;
    }
    v22 = v110;
    if ( !(_BYTE)qword_4F88888 )
      goto LABEL_21;
    v23 = *(_WORD *)(v110 + 24);
    if ( v23 == 2 )
    {
      v89 = *(_QWORD *)(v110 + 32);
      sub_DBE000((__int64)v112, a1, v89);
      v24 = sub_D97050(a1, *(_QWORD *)(v110 + 40));
      v95 = sub_D97050(a1, v5);
      sub_AB4490((__int64)v114, (__int64)v112, v24);
      sub_AB41D0((__int64)&v116, (__int64)v114, v95);
      sub_AB4E00((__int64)&v119, (__int64)v112, v95);
      LOBYTE(v95) = sub_AB1BB0((__int64)&v116, (__int64)&v119);
      sub_969240(v121);
      sub_969240((__int64 *)&v119);
      sub_969240(&v118);
      sub_969240((__int64 *)&v116);
      sub_969240(&v115);
      sub_969240(v114);
      if ( (_BYTE)v95 )
      {
        v18 = (__int64 *)v89;
        v104 = sub_DC5140(a1, v89, v5, a4);
        sub_969240(v113);
        sub_969240(v112);
        result = (_QWORD *)v104;
        goto LABEL_11;
      }
      sub_969240(v113);
      sub_969240(v112);
      v22 = v110;
      v23 = *(_WORD *)(v110 + 24);
    }
    if ( v23 == 5 )
    {
      if ( (*(_BYTE *)(v22 + 28) & 4) != 0 )
      {
        v36 = a4 + 1;
        v119 = v121;
        v120 = 0x400000000LL;
        v37 = *(__int64 **)(v22 + 32);
        v106 = &v37[*(_QWORD *)(v22 + 40)];
        while ( v106 != v37 )
        {
          v38 = *v37++;
          v39 = sub_DC5000(a1, v38, v5, v36);
          sub_D9B3A0((__int64)&v119, v39, v40, v41, v42, v43);
        }
        v18 = (__int64 *)&v119;
        result = (_QWORD *)sub_DC7EB0(a1, &v119, 4, v36);
        v35 = (__int64)v119;
        if ( v119 == v121 )
          goto LABEL_11;
        goto LABEL_30;
      }
      v53 = **(_QWORD **)(v22 + 32);
      if ( *(_WORD *)(v53 + 24) )
        goto LABEL_21;
      sub_DB5770((__int64)v114, a1, *(_QWORD *)(v53 + 32), v22);
      if ( !sub_D94970((__int64)v114, 0) )
      {
        v54 = sub_DA26C0((__int64 *)a1, (__int64)v114);
        v97 = sub_DC5000(a1, v54, v5, a4);
        sub_9865C0((__int64)&v116, (__int64)v114);
        sub_AADAA0((__int64)&v119, (__int64)&v116, v55, v56, v57);
        v58 = sub_DA26C0((__int64 *)a1, (__int64)&v119);
        v59 = (_QWORD *)sub_DC7ED0(a1, v58, v22, 0, a4);
LABEL_51:
        v60 = v59;
        v61 = a4 + 1;
        sub_969240((__int64 *)&v119);
        sub_969240((__int64 *)&v116);
        v62 = sub_DC5000(a1, v60, v5, v61);
        v18 = (__int64 *)v97;
        v107 = sub_DC7ED0(a1, v97, v62, 6, v61);
        sub_969240(v114);
        result = (_QWORD *)v107;
        goto LABEL_11;
      }
      sub_969240(v114);
      v22 = v110;
      v23 = *(_WORD *)(v110 + 24);
    }
    if ( v23 != 8 || *(_QWORD *)(v22 + 40) != 2 )
    {
LABEL_21:
      if ( (unsigned __int8)sub_DBED40(a1, v22) )
      {
        v18 = (__int64 *)v110;
        result = sub_DC2B70(a1, v110, v5, a4 + 1);
        goto LABEL_11;
      }
      v25 = v110;
      if ( ((*(_WORD *)(v110 + 24) - 10) & 0xFFFD) == 0 )
      {
        v119 = v121;
        v120 = 0x400000000LL;
        v26 = *(_QWORD *)(v110 + 32);
        if ( v26 + 8LL * *(_QWORD *)(v110 + 40) != v26 )
        {
          v27 = (__int64 *)(v26 + 8LL * *(_QWORD *)(v110 + 40));
          v28 = *(__int64 **)(v110 + 32);
          do
          {
            v29 = *v28++;
            v30 = sub_DC5000(a1, v29, v5, 0);
            sub_D9B3A0((__int64)&v119, v30, v31, v32, v33, v34);
          }
          while ( v27 != v28 );
          v25 = v110;
        }
        v18 = (__int64 *)&v119;
        if ( *(_WORD *)(v25 + 24) == 12 )
          result = (_QWORD *)sub_DCE150(a1, &v119);
        else
          result = (_QWORD *)sub_DCDF90(a1, &v119);
        v35 = (__int64)v119;
        if ( v119 == v121 )
          goto LABEL_11;
LABEL_30:
        v105 = result;
        _libc_free(v35, &v119);
        result = v105;
        goto LABEL_11;
      }
      v18 = (__int64 *)&v122;
      result = sub_C65B40((__int64)v103, (__int64)&v122, (__int64 *)&v111, (__int64)off_49DEA80);
      if ( result )
        goto LABEL_11;
      v63 = sub_C65D30((__int64)&v122, (unsigned __int64 *)(a1 + 1064));
      v65 = v64;
      v66 = (__int64 *)sub_A777F0(0x30u, (__int64 *)(a1 + 1064));
      if ( v66 )
      {
        v102 = v66;
        sub_D96B90((__int64)v66, (__int64)v63, v65, v110, v5);
        v66 = v102;
      }
      v101 = (__int64)v66;
      sub_C657C0(v103, v66, v111, (__int64)off_49DEA80);
      v119 = (__int64 *)v110;
      v52 = (__int64 *)&v119;
LABEL_46:
      v18 = (__int64 *)v101;
      sub_DAEE00(a1, v101, v52, 1);
      result = (_QWORD *)v101;
      goto LABEL_11;
    }
    v90 = **(_QWORD **)(v22 + 32);
    v96 = sub_D33D80((_QWORD *)v22, a1, v19, v20, v21);
    v44 = sub_D95540(**(_QWORD **)(v22 + 32));
    v91 = sub_D97050(a1, v44);
    v94 = *(_QWORD *)(v22 + 48);
    if ( (*(_BYTE *)(v22 + 28) & 4) != 0 )
    {
      v45 = a4 + 1;
      v46 = sub_DE0960(v22, v5, a1, v45);
      v47 = sub_DC5000(a1, v96, v5, v45);
      v18 = (__int64 *)v46;
      result = sub_DC1960(a1, v46, v47, v94, 4u);
      goto LABEL_11;
    }
    v98 = sub_DCF3A0(a1, v94, 1);
    if ( sub_D96A50(v98)
      || (v67 = sub_D95540(v90),
          v87 = sub_DC5760(a1, v98, v67, a4),
          v68 = sub_D95540(v98),
          v98 != sub_DC5760(a1, v87, v68, a4)) )
    {
LABEL_59:
      v69 = sub_DDE1E0(a1, v22);
      sub_D97270(a1, v22, v69);
      if ( (*(_BYTE *)(v22 + 28) & 4) == 0 )
      {
        if ( !*(_WORD *)(v90 + 24) )
        {
          v92 = *(_QWORD *)(v90 + 32) + 24LL;
          sub_DB56A0((__int64)v114, a1, v92, v96);
          if ( !sub_D94970((__int64)v114, 0) )
          {
            v70 = sub_DA26C0((__int64 *)a1, (__int64)v114);
            v97 = sub_DC5000(a1, v70, v5, a4);
            v71 = *(_WORD *)(v22 + 28);
            sub_9865C0((__int64)&v116, v92);
            sub_C46B40((__int64)&v116, v114);
            LODWORD(v120) = v117;
            v117 = 0;
            v119 = v116;
            v72 = sub_DA26C0((__int64 *)a1, (__int64)&v119);
            v59 = sub_DC1960(a1, (__int64)v72, v96, v94, v71 & 7);
            goto LABEL_51;
          }
          sub_969240(v114);
        }
        if ( !(unsigned __int8)sub_DC3F80((__int64 *)a1, v90, v96, v94) )
        {
          v22 = v110;
          goto LABEL_21;
        }
        sub_D97270(a1, v22, 4);
      }
      v73 = a4 + 1;
      v74 = sub_DE0960(v22, v5, a1, v73);
      v75 = sub_DC5000(a1, v96, v5, v73);
      v18 = (__int64 *)v74;
      result = sub_DC1960(a1, v74, v75, v94, *(_WORD *)(v22 + 28) & 7);
      goto LABEL_11;
    }
    v76 = (_QWORD *)sub_B2BE50(*(_QWORD *)a1);
    v99 = a4 + 1;
    v93 = sub_BCCE00(v76, 2 * v91);
    v77 = sub_DCA690(a1, v87, v96, 0, a4 + 1);
    v78 = sub_DC7ED0(a1, v90, v77, 0, a4 + 1);
    v86 = sub_DC5000(a1, v78, v93, a4 + 1);
    v85 = sub_DC5000(a1, v90, v93, a4 + 1);
    v88 = sub_DC2B70(a1, v87, v93, a4 + 1);
    v79 = sub_DC5000(a1, v96, v93, a4 + 1);
    v80 = sub_DCA690(a1, v88, v79, 0, a4 + 1);
    if ( v86 == sub_DC7ED0(a1, v85, v80, 0, a4 + 1) )
    {
      sub_D97270(a1, v22, 4);
      v83 = sub_DE0960(v22, v5, a1, v99);
      v84 = (_QWORD *)sub_DC5000(a1, v96, v5, v99);
    }
    else
    {
      v81 = sub_DC2B70(a1, v96, v93, v99);
      v82 = sub_DCA690(a1, v88, v81, 0, v99);
      if ( v86 != sub_DC7ED0(a1, v85, v82, 0, v99) )
        goto LABEL_59;
      sub_D97270(a1, v22, 1);
      v83 = sub_DE0960(v22, v5, a1, v99);
      v84 = sub_DC2B70(a1, v96, v5, v99);
    }
    v18 = (__int64 *)v83;
    result = sub_DC1960(a1, v83, (__int64)v84, v94, *(_WORD *)(v22 + 28) & 7);
  }
LABEL_11:
  if ( v122 != v124 )
  {
    v109 = result;
    _libc_free(v122, v18);
    return v109;
  }
  return result;
}
