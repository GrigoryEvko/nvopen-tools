// Function: sub_3876850
// Address: 0x3876850
//
_QWORD *__fastcall sub_3876850(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // r15
  __int64 **v14; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 ***v17; // r14
  __int64 **v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 ***v21; // r13
  __int64 v22; // rsi
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  bool v25; // cc
  _QWORD *v26; // r12
  __int64 v27; // rax
  _QWORD *v29; // rax
  __int64 v30; // r15
  __int64 **v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdi
  unsigned __int64 *v35; // r13
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 *v41; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v42; // [rsp+18h] [rbp-78h] BYREF
  const char *v43; // [rsp+20h] [rbp-70h] BYREF
  char v44; // [rsp+30h] [rbp-60h]
  char v45; // [rsp+31h] [rbp-5Fh]
  _QWORD v46[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v47; // [rsp+50h] [rbp-40h]

  v13 = (__int64)(a1 + 33);
  v14 = (__int64 **)sub_1456040(*(_QWORD *)(a2 + 40));
  v17 = sub_38767A0(a1, *(_QWORD *)(a2 + 40), v14, a3, a4, a5, a6, a7, v15, v16, a10, a11);
  v18 = (__int64 **)sub_1456040(*(_QWORD *)(a2 + 48));
  v21 = sub_38767A0(a1, *(_QWORD *)(a2 + 48), v18, a3, a4, a5, a6, a7, v19, v20, a10, a11);
  a1[34] = *(_QWORD *)(a3 + 40);
  a1[35] = a3 + 24;
  v22 = *(_QWORD *)(a3 + 48);
  v46[0] = v22;
  if ( v22 )
  {
    sub_1623A60((__int64)v46, v22, 2);
    v23 = a1[33];
    if ( !v23 )
      goto LABEL_4;
  }
  else
  {
    v23 = a1[33];
    if ( !v23 )
      goto LABEL_6;
  }
  sub_161E7C0(v13, v23);
LABEL_4:
  v24 = (unsigned __int8 *)v46[0];
  a1[33] = v46[0];
  if ( v24 )
    sub_1623210((__int64)v46, v24, v13);
LABEL_6:
  v45 = 1;
  v44 = 3;
  v25 = *((_BYTE *)v17 + 16) <= 0x10u;
  v43 = "ident.check";
  if ( v25 && *((_BYTE *)v21 + 16) <= 0x10u )
  {
    v26 = (_QWORD *)sub_15A37B0(0x21u, v17, v21, 0);
    v27 = sub_14DBA30((__int64)v26, a1[41], 0);
    if ( v27 )
      return (_QWORD *)v27;
  }
  else
  {
    v47 = 257;
    v29 = sub_1648A60(56, 2u);
    v26 = v29;
    if ( v29 )
    {
      v30 = (__int64)v29;
      v31 = *v17;
      if ( *((_BYTE *)*v17 + 8) == 16 )
      {
        v41 = v31[4];
        v32 = (__int64 *)sub_1643320(*v31);
        v33 = (__int64)sub_16463B0(v32, (unsigned int)v41);
      }
      else
      {
        v33 = sub_1643320(*v31);
      }
      sub_15FEC10((__int64)v26, v33, 51, 33, (__int64)v17, (__int64)v21, (__int64)v46, 0);
    }
    else
    {
      v30 = 0;
    }
    v34 = a1[34];
    if ( v34 )
    {
      v35 = (unsigned __int64 *)a1[35];
      sub_157E9D0(v34 + 40, (__int64)v26);
      v36 = v26[3];
      v37 = *v35;
      v26[4] = v35;
      v37 &= 0xFFFFFFFFFFFFFFF8LL;
      v26[3] = v37 | v36 & 7;
      *(_QWORD *)(v37 + 8) = v26 + 3;
      *v35 = *v35 & 7 | (unsigned __int64)(v26 + 3);
    }
    sub_164B780(v30, (__int64 *)&v43);
    v38 = a1[33];
    if ( v38 )
    {
      v42 = (unsigned __int8 *)a1[33];
      sub_1623A60((__int64)&v42, v38, 2);
      v39 = v26[6];
      if ( v39 )
        sub_161E7C0((__int64)(v26 + 6), v39);
      v40 = v42;
      v26[6] = v42;
      if ( v40 )
        sub_1623210((__int64)&v42, v40, (__int64)(v26 + 6));
    }
  }
  return v26;
}
