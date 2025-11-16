// Function: sub_387C030
// Address: 0x387c030
//
__int64 __fastcall sub_387C030(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r14
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned __int64 v16; // rbx
  __int64 *v17; // rax
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // r15
  __int64 v22; // rax
  int v23; // r12d
  int v24; // r15d
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // r14
  unsigned __int64 v28; // rax
  __int64 v29; // r13
  __int64 *v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r15
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  char v39; // dl
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int64 v46; // rbx
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // r15
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 v52; // rdx
  _QWORD *v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rax
  _BYTE *v57; // rsi
  __int64 v58; // rax
  __int64 *v59; // r15
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 *v62; // r12
  __int64 v63; // rax
  __int64 v64; // r15
  __int64 ***v65; // rax
  double v66; // xmm4_8
  double v67; // xmm5_8
  __int64 v68; // r15
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // r15
  __int64 v72; // r13
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 *v76; // rdi
  __int64 v77; // rax
  char v78; // di
  unsigned int v79; // esi
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rsi
  unsigned __int8 *v84; // rsi
  _QWORD *v85; // r15
  __int64 *v86; // rbx
  __int64 *v87; // rax
  __int64 v88; // r12
  unsigned __int64 v89; // r15
  unsigned __int64 *v90; // rdi
  __int64 v91; // rax
  unsigned __int64 v92; // r15
  __int64 *v93; // rbx
  __int64 v94; // rsi
  __int64 *v95; // r12
  __int64 v96; // rax
  __int64 v97; // r13
  __int64 v98; // rax
  _QWORD *v99; // r15
  __int64 v100; // r12
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  double v104; // xmm4_8
  double v105; // xmm5_8
  __int64 ***v106; // rax
  __int64 v107; // [rsp+8h] [rbp-E8h]
  __int64 v109; // [rsp+18h] [rbp-D8h]
  __int64 v111; // [rsp+28h] [rbp-C8h]
  __int64 v112; // [rsp+30h] [rbp-C0h]
  __int64 v113; // [rsp+30h] [rbp-C0h]
  __int64 **v114; // [rsp+38h] [rbp-B8h]
  __int64 v115; // [rsp+48h] [rbp-A8h] BYREF
  __int64 *v116[2]; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE v117[16]; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 *v118; // [rsp+70h] [rbp-80h] BYREF
  __int64 v119; // [rsp+78h] [rbp-78h]
  unsigned __int64 s[2]; // [rsp+80h] [rbp-70h] BYREF
  int v121; // [rsp+90h] [rbp-60h]
  _BYTE v122[88]; // [rsp+98h] [rbp-58h] BYREF

  v10 = (__int64 *)a1;
  v11 = a2;
  v12 = *(_QWORD *)a1;
  v13 = sub_1456040(**(_QWORD **)(a2 + 32));
  v14 = sub_1456E10(v12, v13);
  v114 = (__int64 **)v14;
  v112 = *(_QWORD *)(a2 + 48);
  if ( *(_BYTE *)(a1 + 256) )
    v15 = sub_1BF9AF0(*(_QWORD *)(a2 + 48), 0, v14);
  else
    v15 = sub_13FCD20(v112);
  if ( !v15 || (v16 = sub_1456C90(*(_QWORD *)a1, *(_QWORD *)v15), v16 < sub_1456C90(*(_QWORD *)a1, (__int64)v114)) )
  {
    if ( sub_14560B0(**(_QWORD **)(a2 + 32)) )
    {
      v17 = *(__int64 **)(v112 + 32);
      v18 = *v17;
      v19 = *(_QWORD *)(*v17 + 8);
      if ( v19 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v19) + 16) - 25) > 9u )
        {
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
            goto LABEL_41;
        }
        v20 = *(_QWORD *)(v18 + 48);
        v21 = v19;
        LOWORD(s[0]) = 259;
        v22 = v20 - 24;
        if ( !v20 )
          v22 = 0;
        v23 = 0;
        v111 = v22;
        v118 = (unsigned __int64 *)"indvar";
        while ( 1 )
        {
          v21 = *(_QWORD *)(v21 + 8);
          if ( !v21 )
            break;
          while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v21) + 16) - 25) <= 9u )
          {
            v21 = *(_QWORD *)(v21 + 8);
            ++v23;
            if ( !v21 )
              goto LABEL_14;
          }
        }
LABEL_14:
        v24 = v23 + 1;
      }
      else
      {
LABEL_41:
        v111 = *(_QWORD *)(v18 + 48);
        if ( v111 )
        {
          v24 = 0;
          v19 = 0;
          v111 -= 24;
        }
        else
        {
          v19 = 0;
          v24 = 0;
        }
        v118 = (unsigned __int64 *)"indvar";
        LOWORD(s[0]) = 259;
      }
      v25 = sub_1648B60(64);
      v15 = v25;
      if ( v25 )
      {
        sub_15F1EA0(v25, (__int64)v114, 53, 0, 0, v111);
        *(_DWORD *)(v15 + 56) = v24;
        sub_164B780(v15, (__int64 *)&v118);
        sub_1648880(v15, *(_DWORD *)(v15 + 56), 1);
      }
      sub_38740E0(a1, v15);
      v118 = 0;
      v119 = (__int64)v122;
      s[0] = (unsigned __int64)v122;
      s[1] = 4;
      v121 = 0;
      v109 = sub_15A0680((__int64)v114, 1, 0);
      if ( !v19 )
        goto LABEL_30;
      v26 = sub_1648700(v19);
      v27 = v19;
LABEL_26:
      v35 = v26[5];
      sub_1412190((__int64)&v118, v35);
      if ( !v39 )
      {
        v77 = 0x17FFFFFFE8LL;
        v78 = *(_BYTE *)(v15 + 23) & 0x40;
        v79 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
        if ( v79 )
        {
          v37 = v15 - 24LL * v79;
          v80 = 24LL * *(unsigned int *)(v15 + 56) + 8;
          v81 = 0;
          do
          {
            v36 = v15 - 24LL * v79;
            if ( v78 )
              v36 = *(_QWORD *)(v15 - 8);
            if ( v35 == *(_QWORD *)(v36 + v80) )
            {
              v77 = 24 * v81;
              goto LABEL_58;
            }
            ++v81;
            v80 += 8;
          }
          while ( v79 != (_DWORD)v81 );
          v77 = 0x17FFFFFFE8LL;
        }
LABEL_58:
        if ( v78 )
        {
          v82 = *(_QWORD *)(v15 - 8);
        }
        else
        {
          v36 = 24LL * v79;
          v82 = v15 - v36;
        }
        sub_1704F80(v15, *(_QWORD *)(v82 + v77), v35, v36, v37, v38);
LABEL_24:
        while ( 1 )
        {
          v27 = *(_QWORD *)(v27 + 8);
          if ( !v27 )
            goto LABEL_29;
LABEL_25:
          v26 = sub_1648700(v27);
          if ( (unsigned __int8)(*((_BYTE *)v26 + 16) - 25) <= 9u )
            goto LABEL_26;
        }
      }
      if ( !sub_1377F70(v112 + 56, v35) )
      {
        v42 = sub_15A06D0(v114, v35, v40, v41);
        sub_1704F80(v15, v42, v35, v43, v44, v45);
        v27 = *(_QWORD *)(v27 + 8);
        if ( v27 )
          goto LABEL_25;
LABEL_29:
        v10 = (__int64 *)a1;
        v11 = a2;
LABEL_30:
        if ( s[0] != v119 )
          _libc_free(s[0]);
        goto LABEL_35;
      }
      v28 = sub_157EBA0(v35);
      v117[1] = 1;
      v116[0] = (__int64 *)"indvar.next";
      v117[0] = 3;
      v29 = sub_15FB440(11, (__int64 *)v15, v109, (__int64)v116, v28);
      v30 = *(__int64 **)(sub_157EBA0(v35) + 48);
      v116[0] = v30;
      if ( v30 )
      {
        sub_1623A60((__int64)v116, (__int64)v30, 2);
        v31 = v29 + 48;
        if ( (__int64 **)(v29 + 48) == v116 )
        {
          if ( v116[0] )
            sub_161E7C0((__int64)v116, (__int64)v116[0]);
          goto LABEL_23;
        }
        v83 = *(_QWORD *)(v29 + 48);
        if ( !v83 )
        {
LABEL_65:
          v84 = (unsigned __int8 *)v116[0];
          *(__int64 **)(v29 + 48) = v116[0];
          if ( v84 )
            sub_1623210((__int64)v116, v84, v31);
          goto LABEL_23;
        }
      }
      else
      {
        v31 = v29 + 48;
        if ( (__int64 **)(v29 + 48) == v116 || (v83 = *(_QWORD *)(v29 + 48)) == 0 )
        {
LABEL_23:
          sub_38740E0(a1, v29);
          sub_1704F80(v15, v29, v35, v32, v33, v34);
          goto LABEL_24;
        }
      }
      v107 = v31;
      sub_161E7C0(v31, v83);
      v31 = v107;
      goto LABEL_65;
    }
    goto LABEL_44;
  }
  v46 = sub_1456C90(*(_QWORD *)a1, *(_QWORD *)v15);
  if ( v46 > sub_1456C90(*(_QWORD *)a1, (__int64)v114) )
  {
    v89 = *(_QWORD *)(a2 + 40);
    v90 = s;
    v118 = s;
    v119 = 0x400000000LL;
    if ( v89 > 4 )
    {
      sub_16CD150((__int64)&v118, s, v89, 8, v47, v48);
      v90 = v118;
    }
    LODWORD(v119) = v89;
    if ( 8LL * (unsigned int)v89 )
      memset(v90, 0, 8LL * (unsigned int)v89);
    v91 = *(_QWORD *)(a2 + 40);
    if ( (_DWORD)v91 )
    {
      v92 = 0;
      v93 = (__int64 *)v15;
      v113 = 8LL * (unsigned int)v91;
      do
      {
        v94 = *(_QWORD *)(*(_QWORD *)(v11 + 32) + v92);
        v95 = (__int64 *)&v118[v92 / 8];
        v92 += 8LL;
        *v95 = sub_147BE70(*v10, v94, *v93);
      }
      while ( v92 != v113 );
    }
    v96 = sub_14785F0(*v10, (__int64 **)&v118, *(_QWORD *)(v11 + 48), *(_WORD *)(v11 + 26) & 1);
    v97 = sub_3875200(v10, v96, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
    v98 = sub_386F050(v97, v10[34]);
    v99 = (_QWORD *)*v10;
    v100 = v98;
    v101 = v98 - 24;
    if ( v100 )
      v100 = v101;
    v102 = sub_145DC80(*v10, v97);
    v103 = sub_14835F0(v99, v102, (__int64)v114, 0, a3, a4);
    v106 = sub_38767A0(v10, v103, 0, v100, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v104, v105, a9, a10);
    v76 = (__int64 *)v118;
    v15 = (__int64)v106;
    if ( v118 == s )
      return v15;
LABEL_49:
    _libc_free((unsigned __int64)v76);
    return v15;
  }
  if ( !sub_14560B0(**(_QWORD **)(a2 + 32)) )
  {
LABEL_44:
    v57 = *(_BYTE **)(a2 + 32);
    v58 = *(_QWORD *)(v11 + 40);
    v118 = s;
    v119 = 0x400000000LL;
    sub_145C5B0((__int64)&v118, v57, &v57[8 * v58]);
    v59 = (__int64 *)v118;
    *v59 = sub_145CF80(*(_QWORD *)a1, (__int64)v114, 0, 0);
    v60 = sub_14785F0(*(_QWORD *)a1, (__int64 **)&v118, v112, *(_WORD *)(v11 + 26) & 1);
    v61 = *(_QWORD *)a1;
    v62 = (__int64 *)v60;
    v63 = **(_QWORD **)(v11 + 32);
    v116[0] = v62;
    v115 = v63;
    sub_3871090(&v115, v116, v61, a3, a4);
    v64 = sub_1456040(v115);
    if ( *(_BYTE *)(v64 + 8) == 15 && (unsigned __int16)(*(_WORD *)(v115 + 24) - 5) > 1u )
    {
      v65 = (__int64 ***)sub_3875200((__int64 *)a1, v115, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
      v15 = sub_3878B90((__int64 *)a1, (__int64)v116[0], v64, v114, v65, a3, a4, a5, a6, v66, v67, a9, a10);
    }
    else
    {
      v68 = *(_QWORD *)a1;
      v69 = sub_3875200((__int64 *)a1, **(_QWORD **)(v11 + 32), *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
      v70 = sub_145DC80(v68, v69);
      v71 = *(_QWORD *)a1;
      v72 = v70;
      v73 = sub_3875200((__int64 *)a1, (__int64)v62, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
      v74 = sub_145DC80(v71, v73);
      v75 = sub_13A5B00(*(_QWORD *)a1, v72, v74, 0, 0);
      v15 = sub_3875200((__int64 *)a1, v75, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
    }
    v76 = (__int64 *)v118;
    if ( v118 == s )
      return v15;
    goto LABEL_49;
  }
LABEL_35:
  if ( *(_QWORD *)(v11 + 40) != 2 )
    goto LABEL_36;
  if ( !sub_1456110(*(_QWORD *)(*(_QWORD *)(v11 + 32) + 8LL)) )
  {
    v85 = (_QWORD *)*v10;
    if ( *(_QWORD *)(v11 + 40) == 2 )
    {
      v86 = (__int64 *)sub_1489E40(*v10, *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8LL), *(_QWORD *)v15);
      v87 = (__int64 *)sub_145DC80(*v10, v15);
      v116[1] = v86;
      v116[0] = v87;
      v119 = 0x200000000LL;
      v118 = s;
      sub_145C5B0((__int64)&v118, v116, v117);
      v88 = sub_147EE30(v85, (__int64 **)&v118, 0, 0, a3, a4);
      if ( v118 != s )
        _libc_free((unsigned __int64)v118);
      v52 = (__int64)v114;
      v54 = v88;
      v53 = v85;
      goto LABEL_39;
    }
LABEL_36:
    v49 = sub_145DC80(*v10, v15);
    v50 = sub_1489E40(*v10, v11, *(_QWORD *)v15);
    if ( *(_WORD *)(v50 + 24) == 7 )
      v11 = v50;
    v51 = sub_1487810(v11, v49, (_QWORD *)*v10, a3, a4);
    v52 = (__int64)v114;
    v53 = (_QWORD *)*v10;
    v54 = (__int64)v51;
LABEL_39:
    v55 = sub_1483C80(v53, v54, v52, a3, a4);
    return sub_3875200(v10, v55, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  }
  return v15;
}
