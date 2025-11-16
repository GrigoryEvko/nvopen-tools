// Function: sub_3879DA0
// Address: 0x3879da0
//
__int64 __fastcall sub_3879DA0(
        __int64 *a1,
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
  __int64 v10; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r13
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // r15
  __int64 v18; // r10
  __int64 v19; // rax
  _DWORD *v20; // rdi
  _DWORD *v21; // rsi
  __int64 v22; // r13
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // rbx
  __int64 **v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 ***v29; // rax
  __int64 ***v30; // r13
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 ***v33; // rax
  __int64 v34; // rax
  __int64 ***v36; // rax
  __int64 ***v37; // r13
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 ***v40; // rax
  __int64 v41; // rax
  __int64 v42; // r8
  char v43; // di
  unsigned int v44; // esi
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdx
  char v52; // al
  int v53; // r9d
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 v56; // rcx
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // r15
  __int64 v60; // rax
  __int64 v61; // r15
  __int64 v62; // rax
  __int64 v63; // r15
  __int64 v64; // rax
  __int64 v65; // r15
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 ***v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 *v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 ***v74; // rbx
  __int64 v75; // rax
  double v76; // xmm4_8
  double v77; // xmm5_8
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // [rsp+10h] [rbp-D0h]
  unsigned int v81; // [rsp+10h] [rbp-D0h]
  __int64 v82; // [rsp+18h] [rbp-C8h]
  __int64 ***v83; // [rsp+18h] [rbp-C8h]
  __int64 v84; // [rsp+20h] [rbp-C0h]
  __int64 v85; // [rsp+20h] [rbp-C0h]
  __int64 v86; // [rsp+28h] [rbp-B8h]
  __int64 v87; // [rsp+28h] [rbp-B8h]
  char v88; // [rsp+28h] [rbp-B8h]
  __int64 v89; // [rsp+30h] [rbp-B0h]
  __int64 v90; // [rsp+30h] [rbp-B0h]
  __int64 v91; // [rsp+30h] [rbp-B0h]
  __int64 **v92; // [rsp+38h] [rbp-A8h]
  char v93; // [rsp+47h] [rbp-99h] BYREF
  __int64 **v94; // [rsp+48h] [rbp-98h] BYREF
  __int64 v95; // [rsp+50h] [rbp-90h] BYREF
  __int16 v96; // [rsp+60h] [rbp-80h]
  __int64 *v97; // [rsp+70h] [rbp-70h] BYREF
  _BYTE *v98; // [rsp+78h] [rbp-68h]
  _BYTE *v99; // [rsp+80h] [rbp-60h]
  __int64 v100; // [rsp+88h] [rbp-58h]
  int v101; // [rsp+90h] [rbp-50h]
  _BYTE v102[72]; // [rsp+98h] [rbp-48h] BYREF

  v10 = a2;
  v86 = sub_1456040(**(_QWORD **)(a2 + 32));
  v12 = sub_1456E10(*a1, v86);
  v13 = *(_QWORD *)(a2 + 48);
  v92 = (__int64 **)v12;
  v82 = (__int64)(a1 + 19);
  if ( sub_1498DE0((__int64)(a1 + 19), v13) )
  {
    v97 = 0;
    v98 = v102;
    v99 = v102;
    v100 = 2;
    v101 = 0;
    sub_1412190((__int64)&v97, v13);
    v10 = sub_1499950(a2, (__int64)&v97, (_QWORD *)*a1, a3, a4);
    if ( v98 != v99 )
      _libc_free((unsigned __int64)v99);
  }
  v14 = **(_QWORD **)(v10 + 32);
  v89 = 0;
  if ( !sub_146D930(*a1, v14, **(_QWORD **)(v13 + 32)) )
  {
    v63 = *a1;
    v64 = sub_1456040(**(_QWORD **)(v10 + 32));
    v65 = sub_145CF80(v63, v64, 0, 0);
    v85 = *(_QWORD *)(v10 + 48);
    v90 = *a1;
    v81 = *(_WORD *)(v10 + 26) & 1;
    v66 = sub_13A5BC0((_QWORD *)v10, *a1);
    v67 = sub_14799E0(v90, v65, v66, v85, v81);
    v89 = v14;
    v14 = v65;
    v10 = v67;
  }
  v80 = sub_13A5BC0((_QWORD *)v10, *a1);
  v17 = v86;
  v84 = 0;
  if ( !sub_146D920(*a1, v80, **(_QWORD **)(v13 + 32)) )
  {
    v59 = *a1;
    v60 = sub_1456040(**(_QWORD **)(v10 + 32));
    v61 = sub_145CF80(v59, v60, 1, 0);
    if ( !sub_14560B0(v14) )
    {
      v91 = *a1;
      v78 = sub_1456040(**(_QWORD **)(v10 + 32));
      v79 = sub_145CF80(v91, v78, 0, 0);
      v89 = v14;
      v14 = v79;
    }
    v10 = sub_14799E0(*a1, v14, v61, *(_QWORD *)(v10 + 48), *(_WORD *)(v10 + 26) & 1);
    v62 = v80;
    if ( v80 )
    {
      v80 = v61;
      v17 = (__int64)v92;
      v84 = v62;
    }
    else
    {
      v80 = v61;
      v17 = v86;
      v84 = 0;
    }
  }
  v18 = v17;
  if ( *(_BYTE *)(v86 + 8) == 15 )
  {
    v19 = a1[1];
    v20 = *(_DWORD **)(v19 + 408);
    v21 = &v20[*(unsigned int *)(v19 + 416)];
    LODWORD(v97) = *(_DWORD *)(v86 + 8) >> 8;
    if ( v21 != sub_386EF90(v20, (__int64)v21, (int *)&v97) )
      v18 = sub_1456040(**(_QWORD **)(v10 + 32));
  }
  v94 = 0;
  v93 = 0;
  v22 = sub_3878E40(a1, v10, v13, v18, v92, (__int64 *)&v94, a3, a4, a5, a6, v15, v16, a9, a10, &v93);
  if ( sub_1498DE0(v82, v13) )
  {
    v42 = sub_13FCB50(v13);
    v43 = *(_BYTE *)(v22 + 23) & 0x40;
    v44 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
    if ( v44 )
    {
      v45 = 24LL * *(unsigned int *)(v22 + 56) + 8;
      v46 = 0;
      while ( 1 )
      {
        v47 = v22 - 24LL * v44;
        if ( v43 )
          v47 = *(_QWORD *)(v22 - 8);
        if ( v42 == *(_QWORD *)(v47 + v45) )
          break;
        ++v46;
        v45 += 8;
        if ( v44 == (_DWORD)v46 )
          goto LABEL_59;
      }
      v48 = 24 * v46;
      if ( v43 )
        goto LABEL_42;
    }
    else
    {
LABEL_59:
      v48 = 0x17FFFFFFE8LL;
      if ( v43 )
      {
LABEL_42:
        v49 = *(_QWORD *)(v22 - 8);
LABEL_43:
        v50 = *(_QWORD *)(v49 + v48);
        if ( *(_BYTE *)(v50 + 16) <= 0x17u )
          goto LABEL_58;
        v51 = a1[35];
        v87 = v50;
        if ( v51 )
          v51 -= 24;
        v52 = sub_15CCEE0(*(_QWORD *)(*a1 + 56), v50, v51);
        v50 = v87;
        if ( v52 )
        {
LABEL_58:
          v22 = v50;
        }
        else
        {
          v88 = 0;
          if ( *(_BYTE *)(v17 + 8) != 15 && sub_1456260(v80) )
          {
            v88 = 1;
            v80 = sub_1480620(*a1, v80, 0);
          }
          sub_38701C0(&v97, a1 + 33, (__int64)a1, v50, (int)&v97, v53);
          v56 = *(_QWORD *)(**(_QWORD **)(v13 + 32) + 48LL);
          if ( v56 )
            v56 -= 24;
          v83 = sub_38767A0(a1, v80, v92, v56, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v54, v55, a9, a10);
          sub_3870260((__int64)&v97);
          v22 = sub_3878BC0(a1, v22, (__int64)v83, a3, a4, a5, a6, v57, v58, a9, a10, v13, v17, v92, v88);
        }
        goto LABEL_9;
      }
    }
    v49 = v22 - 24LL * v44;
    goto LABEL_43;
  }
LABEL_9:
  if ( v94 )
  {
    v25 = *(_QWORD *)v22;
    if ( v25 != sub_1456E10(*a1, *(_QWORD *)v22) )
    {
      v26 = (__int64 **)sub_1456E10(*a1, v25);
      v22 = (__int64)sub_38744E0(a1, v22, v26, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v27, v28, a9, a10);
    }
    if ( v94 != *(__int64 ***)v22 )
    {
      LOWORD(v99) = 257;
      v22 = (__int64)sub_38723F0(a1 + 33, 36, v22, v94, (__int64 *)&v97);
      sub_38740E0((__int64)a1, v22);
    }
    if ( v93 )
    {
      LOWORD(v99) = 257;
      v68 = sub_38761C0(
              a1,
              **(_QWORD **)(v10 + 32),
              v94,
              (__m128)a3,
              *(double *)a4.m128i_i64,
              a5,
              a6,
              v23,
              v24,
              a9,
              a10);
      v22 = (__int64)sub_38718D0(
                       a1 + 33,
                       (__int64)v68,
                       v22,
                       (__int64 *)&v97,
                       0,
                       0,
                       *(double *)a3.m128i_i64,
                       *(double *)a4.m128i_i64,
                       a5);
      sub_38740E0((__int64)a1, v22);
    }
  }
  if ( v84 )
  {
    v29 = sub_38744E0(a1, v22, v92, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v23, v24, a9, a10);
    LOWORD(v99) = 257;
    v30 = v29;
    v33 = sub_38761C0(a1, v84, v92, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v31, v32, a9, a10);
    if ( *((_BYTE *)v30 + 16) > 0x10u || *((_BYTE *)v33 + 16) > 0x10u )
    {
      v22 = (__int64)sub_3872540(a1 + 33, 15, (__int64 *)v30, (__int64)v33, (__int64 *)&v97, 0, 0);
    }
    else
    {
      v22 = sub_15A2C20((__int64 *)v30, (__int64)v33, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
      v34 = sub_14DBA30(v22, a1[41], 0);
      if ( v34 )
        v22 = v34;
    }
    sub_38740E0((__int64)a1, v22);
  }
  if ( v89 )
  {
    if ( *(_BYTE *)(v17 + 8) == 15 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) == 11 )
      {
        v74 = sub_38761C0(a1, v89, (__int64 **)v17, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v23, v24, a9, a10);
        v75 = sub_145DC80(*a1, v22);
        return sub_3878B90(a1, v75, v17, v92, v74, a3, a4, a5, a6, v76, v77, a9, a10);
      }
      else
      {
        return sub_3878B90(a1, v89, v17, v92, (__int64 ***)v22, a3, a4, a5, a6, v23, v24, a9, a10);
      }
    }
    else
    {
      v36 = sub_38744E0(a1, v22, v92, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v23, v24, a9, a10);
      v96 = 257;
      v37 = v36;
      v40 = sub_38761C0(a1, v89, v92, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v38, v39, a9, a10);
      if ( *((_BYTE *)v37 + 16) > 0x10u || *((_BYTE *)v40 + 16) > 0x10u )
      {
        LOWORD(v99) = 257;
        v69 = sub_15FB440(11, (__int64 *)v37, (__int64)v40, (__int64)&v97, 0);
        v70 = a1[34];
        v22 = v69;
        if ( v70 )
        {
          v71 = (__int64 *)a1[35];
          sub_157E9D0(v70 + 40, v69);
          v72 = *(_QWORD *)(v22 + 24);
          v73 = *v71;
          *(_QWORD *)(v22 + 32) = v71;
          v73 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v22 + 24) = v73 | v72 & 7;
          *(_QWORD *)(v73 + 8) = v22 + 24;
          *v71 = *v71 & 7 | (v22 + 24);
        }
        sub_164B780(v22, &v95);
        sub_12A86E0(a1 + 33, v22);
      }
      else
      {
        v22 = sub_15A2B30((__int64 *)v37, (__int64)v40, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        v41 = sub_14DBA30(v22, a1[41], 0);
        if ( v41 )
          v22 = v41;
      }
      sub_38740E0((__int64)a1, v22);
    }
  }
  return v22;
}
