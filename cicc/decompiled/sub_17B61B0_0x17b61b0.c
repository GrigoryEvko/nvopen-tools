// Function: sub_17B61B0
// Address: 0x17b61b0
//
__int64 __fastcall sub_17B61B0(__int64 **a1, __int64 *a2)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r15
  _QWORD *v24; // rax
  _QWORD *v25; // r13
  __int64 v26; // rdi
  unsigned __int64 *v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 *v33; // r13
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // rsi
  __int64 v43; // rax
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  __int64 v46; // [rsp+8h] [rbp-88h]
  _QWORD *v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int8 *v49; // [rsp+18h] [rbp-78h] BYREF
  __int64 v50[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v51; // [rsp+30h] [rbp-60h]
  __int64 *v52; // [rsp+40h] [rbp-50h]
  __int64 v53; // [rsp+48h] [rbp-48h]
  __int64 v54; // [rsp+50h] [rbp-40h]
  __int64 v55[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = **a1;
  if ( v4 && byte_4FA2980 )
    return v4;
  v5 = a2[1];
  v6 = *a2;
  v7 = *(_QWORD *)(v5 + 56);
  v48 = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)&v48, v6, 2);
    v8 = a2[1];
    v9 = *a2;
    v52 = a2;
    v53 = v8;
    v10 = a2[2];
    v55[0] = v9;
    v54 = v10;
    if ( v9 )
      sub_1623A60((__int64)v55, v9, 2);
  }
  else
  {
    v53 = v5;
    v43 = a2[2];
    v52 = a2;
    v54 = v43;
    v55[0] = 0;
  }
  v50[0] = (__int64)"trap";
  v51 = 259;
  v11 = sub_15E0530(v7);
  v12 = *a1;
  v46 = v11;
  v13 = (_QWORD *)sub_22077B0(64);
  v14 = v46;
  if ( v13 )
  {
    v47 = v13;
    sub_157FB60(v13, v14, (__int64)v50, v7, 0);
    v13 = v47;
  }
  *v12 = (__int64)v13;
  v15 = **a1;
  a2[1] = v15;
  a2[2] = v15 + 40;
  v16 = sub_15E26F0(*(__int64 **)(v7 + 40), 205, 0, 0);
  v51 = 257;
  v17 = sub_17B5EF0((__int64)a2, *(_QWORD *)(v16 + 24), v16, 0, 0, v50, 0);
  v50[0] = *(_QWORD *)(v17 + 56);
  v18 = (__int64 *)sub_16498A0(v17);
  v19 = sub_1563AB0(v50, v18, -1, 29);
  *(_QWORD *)(v17 + 56) = v19;
  v50[0] = v19;
  v20 = (__int64 *)sub_16498A0(v17);
  v21 = sub_1563AB0(v50, v20, -1, 30);
  v22 = v48;
  *(_QWORD *)(v17 + 56) = v21;
  v50[0] = v22;
  if ( !v22 )
  {
    v23 = v17 + 48;
    if ( (__int64 *)(v17 + 48) == v50 )
      goto LABEL_12;
    v44 = *(_QWORD *)(v17 + 48);
    if ( !v44 )
      goto LABEL_12;
LABEL_48:
    sub_161E7C0(v23, v44);
    goto LABEL_49;
  }
  v23 = v17 + 48;
  sub_1623A60((__int64)v50, v22, 2);
  if ( (__int64 *)(v17 + 48) == v50 )
  {
    if ( v50[0] )
      sub_161E7C0((__int64)v50, v50[0]);
    goto LABEL_12;
  }
  v44 = *(_QWORD *)(v17 + 48);
  if ( v44 )
    goto LABEL_48;
LABEL_49:
  v45 = (unsigned __int8 *)v50[0];
  *(_QWORD *)(v17 + 48) = v50[0];
  if ( v45 )
    sub_1623210((__int64)v50, v45, v23);
LABEL_12:
  v51 = 257;
  v24 = sub_1648A60(56, 0);
  v25 = v24;
  if ( v24 )
    sub_15F82A0((__int64)v24, a2[3], 0);
  v26 = a2[1];
  if ( v26 )
  {
    v27 = (unsigned __int64 *)a2[2];
    sub_157E9D0(v26 + 40, (__int64)v25);
    v28 = v25[3];
    v29 = *v27;
    v25[4] = v27;
    v29 &= 0xFFFFFFFFFFFFFFF8LL;
    v25[3] = v29 | v28 & 7;
    *(_QWORD *)(v29 + 8) = v25 + 3;
    *v27 = *v27 & 7 | (unsigned __int64)(v25 + 3);
  }
  sub_164B780((__int64)v25, v50);
  v30 = *a2;
  if ( *a2 )
  {
    v49 = (unsigned __int8 *)*a2;
    sub_1623A60((__int64)&v49, v30, 2);
    v31 = v25[6];
    if ( v31 )
      sub_161E7C0((__int64)(v25 + 6), v31);
    v32 = v49;
    v25[6] = v49;
    if ( v32 )
      sub_1623210((__int64)&v49, v32, (__int64)(v25 + 6));
  }
  v33 = v52;
  v34 = v54;
  v4 = **a1;
  v35 = v53;
  if ( !v53 )
  {
    v52[1] = 0;
    v33[2] = 0;
    goto LABEL_29;
  }
  v52[1] = v53;
  v33[2] = v34;
  if ( v34 == v35 + 40 )
    goto LABEL_29;
  if ( !v34 )
    BUG();
  v36 = *(_QWORD *)(v34 + 24);
  v50[0] = v36;
  if ( v36 )
  {
    sub_1623A60((__int64)v50, v36, 2);
    v37 = *v33;
    if ( !*v33 )
      goto LABEL_27;
  }
  else
  {
    v37 = *v33;
    if ( !*v33 )
      goto LABEL_29;
  }
  sub_161E7C0((__int64)v33, v37);
LABEL_27:
  v38 = (unsigned __int8 *)v50[0];
  *v33 = v50[0];
  if ( v38 )
  {
    sub_1623210((__int64)v50, v38, (__int64)v33);
    v33 = v52;
  }
  else
  {
    if ( v50[0] )
      sub_161E7C0((__int64)v50, v50[0]);
    v33 = v52;
  }
LABEL_29:
  v50[0] = v55[0];
  if ( v55[0] )
  {
    sub_1623A60((__int64)v50, v55[0], 2);
    if ( v33 == v50 )
      goto LABEL_43;
    v39 = *v33;
    if ( !*v33 )
    {
LABEL_33:
      v40 = (unsigned __int8 *)v50[0];
      *v33 = v50[0];
      if ( v40 )
      {
        sub_1623210((__int64)v50, v40, (__int64)v33);
        v41 = v55[0];
LABEL_35:
        if ( v41 )
          sub_161E7C0((__int64)v55, v41);
        goto LABEL_37;
      }
LABEL_43:
      if ( v50[0] )
        sub_161E7C0((__int64)v50, v50[0]);
      v41 = v55[0];
      goto LABEL_35;
    }
LABEL_32:
    sub_161E7C0((__int64)v33, v39);
    goto LABEL_33;
  }
  if ( v33 != v50 )
  {
    v39 = *v33;
    if ( !*v33 )
      goto LABEL_43;
    goto LABEL_32;
  }
LABEL_37:
  if ( v48 )
    sub_161E7C0((__int64)&v48, v48);
  return v4;
}
